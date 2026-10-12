"""Пакетные признаки B; фактические данные требуют допуска Э1-10."""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.config import validate_feature_schema
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.contracts import InputValidationError, require_int64
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CandidateResult, FeatureBatch
from coveo_recsys.contracts.retrieval import (
    NO_METADATA_CONFIG_ID,
    RetrievalConfig,
    validate_retrieval_profile,
)
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.metadata import open_metadata_source
from coveo_recsys.data.periods import session_key
from coveo_recsys.data.prepared import TableWriter, read_report, table_info, verify_tables
from coveo_recsys.features import FeatureEngine
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.retrieval import retrieve, retrieve_fallback
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

REPORT = "reports/features.json"
IDENTITY = "features/identity.json"
TABLES = {
    "features/candidates.parquet": CandidateResult,
    "features/fallback.parquet": CandidateResult,
    "features/values.parquet": FeatureBatch,
}
LIMITS = {
    "max_requests",
    "batch_size",
    "max_observed_skus",
    "max_seconds",
    "max_memory_bytes",
    "max_result_bytes",
}


def run_features(
    project_root: Path,
    settings_path: Path,
    run_dir: Path,
    run_id: str,
    *,
    period: str,
    required_config_id: str | None = None,
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from coveo_recsys.cli.preparation import _peak_memory

    if period != "B":
        raise InputValidationError("Для features build разрешён только период B")
    settings_bytes = settings_path.read_bytes()
    settings = strict_json_loads(settings_bytes)
    required = {"preparation", "historical", "feature_schema", "retrieval", "limits"}
    if (
        not isinstance(settings, dict)
        or not required <= set(settings)
        or set(settings) - required - {"metadata_path", "admission", "pilot_requests"}
        or not run_id
    ):
        raise InputValidationError("Неверные настройки признаков")
    limits = settings["limits"]
    if set(limits) != LIMITS or any(
        require_int64(value, nonnegative=True) == 0 for value in limits.values()
    ):
        raise InputValidationError("Нужны положительные ресурсные пределы Э3")
    prep = settings["preparation"]
    history = settings["historical"]
    if set(prep) != {"path", "report_sha256"} or set(history) != {"path", "package_sha256"}:
        raise InputValidationError("Нужны закреплённые входы Э3")
    source = (project_root / prep["path"]).resolve()
    prepared = read_report(source, report_sha256=prep["report_sha256"], verify_tables=False)
    historical_root = (project_root / history["path"]).resolve()
    destination = run_dir.resolve()
    if any(
        destination == path or destination in path.parents or path in destination.parents
        for path in (source, historical_root)
    ):
        raise InputValidationError("Результат пересекается с входным запуском")
    assert_source_binding(project_root)
    code = source_sha256(project_root)
    started = time.monotonic()
    historical_store = LocalArtifactStore(historical_root)
    head = historical_store.describe(PACKAGE_PATH)
    if head.sha256 != history["package_sha256"]:
        raise InputValidationError("Неверный хеш исторического пакета")
    schema_path = project_root / settings["feature_schema"]
    schema_bytes = schema_path.read_bytes()
    schema = strict_json_loads(schema_bytes)
    if validate_feature_schema(schema):
        raise InputValidationError("Неверная схема признаков")
    config_id = schema["config_id"]
    recipe = get_ranker_recipe(config_id)
    if required_config_id is not None and config_id != required_config_id:
        raise InputValidationError("Команда требует схему профиля без метаданных")
    if config_id == NO_METADATA_CONFIG_ID and "metadata_path" in settings:
        raise InputValidationError("Профиль без метаданных не использует metadata_path")
    configuration = dict(settings["retrieval"])
    if "quotas" in configuration:
        configuration["quotas"] = tuple(configuration["quotas"])
    config = RetrievalConfig(**configuration)
    validate_retrieval_profile(config_id, config)
    from coveo_recsys.cli.admission import require_training_admission

    def admission() -> str | None:
        return require_training_admission(
            project_root,
            settings.get("admission"),
            prepared=prepared,
            historical=head,
            feature_schema_bytes=schema_bytes,
            recipe=recipe,
            retrieval=asdict(config),
        )

    manifest_hash = admission()
    pilot_requests = settings.get("pilot_requests")
    selected = None
    if pilot_requests is not None:
        total = prepared["tables"]["datasets/B/prefixes.parquet"]["rows"]
        if not 0 < require_int64(pilot_requests, nonnegative=True) <= total:
            raise InputValidationError("Размер пилота вне подготовленного B")
        plan = from_json(prepared["plan_json"].encode(), PreparationPlan)
        selected = set(
            sorted(prepared["selection"]["B"], key=lambda sid: session_key(sid, plan.seed))[
                :pilot_requests
            ]
        )
        if len(selected) != pilot_requests:
            raise InputValidationError("Недостаточно выбранных ID для пилота")
    completion_status = "completed" if selected is None else "pilot_completed"
    package = load_historical(historical_store)
    catalog_hash = historical_store.describe("historical/catalog.parquet").sha256
    store = LocalArtifactStore(destination)
    identity = {
        "format_version": "features_v1",
        "config_id": config_id,
        "run_id": run_id,
        "scope": prepared["scope"],
        "period": period,
        "source_sha256": code,
        "settings_sha256": sha256_bytes(settings_bytes),
        "preparation_sha256": prep["report_sha256"],
        "preparation_code": prepared["source_sha256"],
        "historical_sha256": head.sha256,
        "historical_descriptor_sha256": head.descriptor_sha256,
        "historical_code": head.code_sha256,
        "catalog_sha256": catalog_hash,
        "feature_schema_file_sha256": sha256_bytes(schema_bytes),
        "configuration": asdict(config),
        "manifest_sha256": manifest_hash,
        **({"pilot_requests": pilot_requests} if selected is not None else {}),
    }
    identity_bytes = canonical_json_bytes(identity)
    identity = strict_json_loads(identity_bytes)
    metadata_path = (
        project_root.joinpath(settings["metadata_path"]) if "metadata_path" in settings else None
    )

    # JSON использует списки вместо кортежей; сравниваем канонические байты.
    def inputs_unchanged() -> None:
        if (
            source_sha256(project_root) != code
            or settings_path.read_bytes() != settings_bytes
            or schema_path.read_bytes() != schema_bytes
        ):
            raise InputValidationError("Код, схема или настройки изменились")
        read_report(source, report_sha256=prep["report_sha256"], verify_tables=False)
        verify_tables(source, prepared, paths=["datasets/B/prefixes.parquet"])
        if historical_store.describe(PACKAGE_PATH).descriptor_sha256 != head.descriptor_sha256:
            raise InputValidationError("Исторический пакет изменился")
        for path, digest in package.paths.items():
            if historical_store.describe(path).descriptor_sha256 != digest:
                raise InputValidationError("Часть исторического пакета изменена")
        if (
            metadata_path is not None
            and sha256_file(metadata_path) != package.input_hashes["metadata_sha256"]
        ):
            raise InputValidationError("Поставка метаданных изменилась")
        if admission() != manifest_hash:
            raise InputValidationError("Допуск изменился")

    def output_tables_unchanged(tables: dict[str, Any]) -> None:
        for path, kind in TABLES.items():
            if table_info(store._path(path), kind, period="B") != tables[path]:
                raise InputValidationError("Изменена таблица признаков или её описание")

    if (destination / REPORT).exists():
        result = strict_json_loads(store._path(REPORT).read_bytes())
        if (
            result.get("status") != completion_status
            or canonical_json_bytes(result.get("identity")) != identity_bytes
            or set(result.get("tables", {})) != set(TABLES)
            or result.get("feature_schema") != schema
            or result.get("limits") != limits
            or any(
                info.get("rows") != result.get("requests")
                for info in result.get("tables", {}).values()
            )
        ):
            raise InputValidationError("Готовые признаки несовместимы с запуском")
        output_tables_unchanged(result["tables"])
        inputs_unchanged()
        return result
    if (destination / IDENTITY).exists():
        if store._path(IDENTITY).read_bytes() != identity_bytes:
            raise InputValidationError("Незавершённый запуск несовместим")
    else:
        store._atomic_write(IDENTITY, identity_bytes)
    audit: list[dict[str, Any]] = []
    access = DatasetAccess(
        source,
        code_sha256=prepared["source_sha256"],
        run_id=prepared["run_id"],
        scope=prepared["scope"],
        audit=audit.append,
    )
    stage = "prefixes"
    processed = 0
    total_requests = prepared["tables"]["datasets/B/prefixes.parquet"]["rows"]
    last_progress = 0.0
    last_stage: str | None = None

    def check_budget(extra_bytes: int = 0) -> None:
        nonlocal last_progress, last_stage
        elapsed = time.monotonic() - started
        peak = _peak_memory()["peak_process_memory_bytes"]
        size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        if (
            elapsed > limits["max_seconds"]
            or peak is None
            or peak > limits["max_memory_bytes"]
            or size + extra_bytes > limits["max_result_bytes"]
        ):
            raise InputValidationError("Превышен предел ресурсов Э3")
        if stage != last_stage or elapsed - last_progress >= 30:
            print(
                json.dumps(
                    {
                        "step": "features",
                        "stage": stage,
                        "processed_requests": processed,
                        "total_requests": total_requests,
                        "elapsed_seconds": elapsed,
                        "peak_process_memory_bytes": peak,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            last_progress = elapsed
            last_stage = stage

    observed: set[str] = set()
    requests = 0
    check_budget()
    for _, items in access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix"):
        requests += len(items)
        if requests > limits["max_requests"]:
            raise InputValidationError("Превышено число запросов; усечение запрещено")
        for item in items:
            observed.update(
                x.sku for x in (*item.prefix.detail_history, *item.prefix.full_last_detail)
            )
        if len(observed) > limits["max_observed_skus"]:
            raise InputValidationError("Превышен предел наблюдаемых SKU")
        processed = requests
        check_budget()
    stage = "metadata"
    check_budget()
    unknown = observed - set(package.catalog.skus)
    observed_records = {}
    if unknown and config_id != NO_METADATA_CONFIG_ID:
        if "metadata_path" not in settings:
            raise InputValidationError(
                "Для SKU вне каталога нужна закреплённая поставка метаданных"
            )
        profile = strict_json_loads(prepared["profile_json"])
        if (
            sha256_bytes(canonical_json_bytes(profile))
            != package.input_hashes["metadata_profile_sha256"]
        ):
            raise InputValidationError("Профиль метаданных несовместим с историческим пакетом")
        assert metadata_path is not None
        metadata = open_metadata_source(
            metadata_path,
            profile,
            expected_sha256=package.input_hashes["metadata_sha256"],
        )
        observed_records = {
            item.sku: item
            for item in metadata.read(sorted(unknown, key=lambda s: s.encode("utf-8")))
        }
    index = RetrievalIndex(package, catalog_hash, observed_records, config_id=config_id)
    engine = FeatureEngine(index, schema, config)
    stage = "features"
    processed = 0
    check_budget()
    total_rows = 0
    calculation_seconds = 0.0
    with (
        TableWriter(
            destination / "features/candidates.parquet",
            CandidateResult,
            batch_rows=limits["batch_size"],
        ) as main_writer,
        TableWriter(
            destination / "features/fallback.parquet",
            CandidateResult,
            batch_rows=limits["batch_size"],
        ) as reserve_writer,
        TableWriter(
            destination / "features/values.parquet", FeatureBatch, batch_rows=limits["batch_size"]
        ) as feature_writer,
    ):
        for sid, items in access.read(
            "datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix"
        ):
            if selected is not None and sid not in selected:
                check_budget()
                continue
            calculation_started = time.perf_counter()
            main, reserve, features = [], [], []
            for item in items:
                context = bind_context(item, index)
                scores = score_request(context, index, config)
                candidates = retrieve(context, index, config, scores=scores)
                main.append(candidates)
                reserve.append(retrieve_fallback(context, index, config))
                features.append(engine.build(context, candidates, scores=scores))
                total_rows += len(candidates.skus)
            main_writer.append(sid, main)
            reserve_writer.append(sid, reserve)
            feature_writer.append(sid, features)
            processed += len(items)
            check_budget()
            calculation_seconds += time.perf_counter() - calculation_started
        calculation_started = time.perf_counter()
    calculation_seconds += time.perf_counter() - calculation_started
    if processed != (pilot_requests if selected is not None else total_requests):
        raise InputValidationError("Число рассчитанных запросов не соответствует плану")
    stage = "validation"
    check_budget()
    tables = {}
    for path, kind in TABLES.items():
        tables[path] = table_info(destination / path, kind, period="B")
        check_budget()
    result = {
        "status": completion_status,
        "identity": identity,
        "feature_schema": schema,
        "requests": processed,
        "candidate_rows": total_rows,
        "tables": tables,
        "reads": audit,
        "limits": limits,
        "elapsed_seconds": time.monotonic() - started,
        "calculation_seconds_including_write": calculation_seconds,
        **_peak_memory(),
    }
    if before_publish is not None:
        before_publish()
    inputs_unchanged()
    output_tables_unchanged(tables)
    stage = "publication"
    result.update(elapsed_seconds=time.monotonic() - started, **_peak_memory())
    encoded = canonical_json_bytes(result)
    check_budget(len(encoded))
    store._atomic_write(REPORT, encoded)
    store._path(IDENTITY).unlink()
    return result
