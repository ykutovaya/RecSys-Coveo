"""Общий порционный путь C_smoke и D на готовых моделях."""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from itertools import zip_longest
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
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.evaluation import (
    FINAL_POLICIES,
    POLICIES,
    EvaluationRecord,
    ReplayRecord,
)
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock, FeatureBatch
from coveo_recsys.contracts.retrieval import (
    NO_METADATA_CONFIG_ID,
    RetrievalConfig,
    validate_retrieval_profile,
)
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.metadata import open_metadata_source
from coveo_recsys.data.periods import session_key
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info, verify_tables
from coveo_recsys.data.prepared import read_report as read_preparation_report
from coveo_recsys.evaluation.replay import recommend
from coveo_recsys.evaluation.sensitivity import compare_multiplicities
from coveo_recsys.features import FeatureEngine
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.ranking.load import RankerModel, load_ranker
from coveo_recsys.ranking.matrix import validate_no_metadata_schema
from coveo_recsys.ranking.train import validate_recipe
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

REPORT = "reports/replay.json"
MARKDOWN = "reports/replay.md"
IDENTITY = "replay/identity.json"
ANSWERS = "replay/answers.parquet"
SESSIONS = "replay/sessions.parquet"
TABLES = {ANSWERS: ReplayRecord, SESSIONS: EvaluationRecord}


def load_sensitivity_model(
    root: Path,
    settings: dict[str, Any],
    *,
    training_hash: str,
    historical_hash: str,
    catalog_hash: str,
    scope: str,
) -> tuple[dict[str, Any], RetrievalConfig, RankerModel]:
    """Проверить собственные B, схему, рецепт и модель варианта без метаданных."""
    if not isinstance(settings, dict) or set(settings) != {
        "model",
        "training_features",
        "feature_schema",
        "recipe",
        "retrieval",
    }:
        raise InputValidationError("Неполные входы чувствительности без метаданных")
    schema_bytes = (root / settings["feature_schema"]).read_bytes()
    schema = strict_json_loads(schema_bytes)
    validate_no_metadata_schema(
        tuple(column["name"] for column in schema["columns"]),
        sha256_bytes(canonical_json_bytes(schema)),
    )
    if validate_feature_schema(schema) or schema["event_semantics"] != "set_v1":
        raise InputValidationError("Неверная схема без метаданных")
    recipe_bytes = (root / settings["recipe"]).read_bytes()
    recipe = strict_json_loads(recipe_bytes)
    validate_recipe(recipe, config_id=NO_METADATA_CONFIG_ID)
    configuration = dict(settings["retrieval"])
    if "quotas" in configuration:
        configuration["quotas"] = tuple(configuration["quotas"])
    config = RetrievalConfig(**configuration)
    validate_retrieval_profile(NO_METADATA_CONFIG_ID, config)
    feature_bytes = (
        root / settings["training_features"]["path"] / "reports/features.json"
    ).read_bytes()
    model_bytes = (root / settings["model"]["path"] / "reports/ranker.json").read_bytes()
    if (
        sha256_bytes(feature_bytes) != settings["training_features"]["report_sha256"]
        or sha256_bytes(model_bytes) != settings["model"]["report_sha256"]
    ):
        raise InputValidationError("Изменены отчёты варианта без метаданных")
    features, summary = strict_json_loads(feature_bytes), strict_json_loads(model_bytes)
    expected = {
        "period": "B",
        "scope": scope,
        "preparation_sha256": training_hash,
        "historical_sha256": historical_hash,
        "catalog_sha256": catalog_hash,
        "feature_schema_file_sha256": sha256_bytes(schema_bytes),
    }
    if (
        features.get("status") != "completed"
        or features.get("feature_schema") != schema
        or any(
            features["identity"].get(k) != v or summary["identity"].get(k) != v
            for k, v in expected.items()
        )
        or canonical_json_bytes(features["identity"].get("configuration"))
        != canonical_json_bytes(asdict(config))
        or summary["identity"].get("features_sha256") != sha256_bytes(feature_bytes)
        or summary["identity"].get("recipe_sha256") != sha256_bytes(recipe_bytes)
        or summary.get("recipe") != recipe
    ):
        raise InputValidationError("Несовместимые собственные входы варианта без метаданных")
    model = load_ranker(root / settings["model"]["path"], expected_identity=summary["identity"])
    if model.summary != summary:
        raise InputValidationError("Модель без метаданных изменилась при загрузке")
    return schema, config, model


def read_report(run: Path) -> dict[str, Any]:
    """Проверить готовые таблицы и воспроизвести Markdown без доступа к сырью."""
    from coveo_recsys.evaluation.reporting import render_report

    if not run.is_dir():
        raise InputValidationError("Отсутствует завершённый запуск воспроизведения")
    store = LocalArtifactStore(run)
    summary = strict_json_loads(store._path(REPORT).read_bytes())
    period = summary.get("identity", {}).get("period") if isinstance(summary, dict) else None
    policies = FINAL_POLICIES if period == "D" else POLICIES
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "completed"
        or summary.get("format_version") != "replay_v1"
        or summary.get("identity", {}).get("format_version") != "replay_v1"
        or summary.get("identity", {}).get("mode") != "ideal_deterministic"
        or period not in {"C_smoke", "D"}
        or summary.get("identity", {}).get("policies") != list(policies)
        or type(summary.get("requests")) is not int
        or summary["requests"] < 0
        or set(summary.get("tables", {})) != set(TABLES)
    ):
        raise InputValidationError("Нет совместимого завершённого отчёта")
    for relative, kind in TABLES.items():
        actual = table_info(store._path(relative), kind, period=period)
        if actual != summary["tables"][relative] or actual["rows"] != summary["requests"] * len(
            policies
        ):
            raise InputValidationError("Повреждена таблица завершённого воспроизведения")
    markdown = render_report(summary).encode("utf-8")
    if store._path(MARKDOWN).exists():
        if store._path(MARKDOWN).read_bytes() != markdown:
            raise InputValidationError("Markdown отличается от сохранённого отчёта")
    else:
        store._atomic_write(MARKDOWN, markdown)
    return summary


def run_replay(
    project_root: Path,
    settings_path: Path,
    run_dir: Path,
    run_id: str,
    *,
    period: str = "C_smoke",
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if period not in {"C_smoke", "D"}:
        raise InputValidationError("Разрешены только replay C_smoke и evaluate D")
    from coveo_recsys.cli.admission import require_final_admission, require_replay_admission
    from coveo_recsys.cli.features import LIMITS
    from coveo_recsys.cli.preparation import _peak_memory
    from coveo_recsys.evaluation.reporting import (
        render_report,
        summarize_replay,
        validate_evaluation_plan,
    )

    started = time.monotonic()
    settings_bytes = settings_path.read_bytes()
    settings = strict_json_loads(settings_bytes)
    final_gate = (
        (lambda: require_final_admission(project_root, settings)) if period == "D" else None
    )
    final_digest = final_gate() if final_gate is not None else None
    policies = FINAL_POLICIES if period == "D" else POLICIES
    required = {
        "preparation",
        "training_preparation",
        "training_features",
        "preparation_plan",
        "historical",
        "model",
        "feature_schema",
        "recipe",
        "retrieval",
        "evaluation_plan",
        "limits",
    }
    if period == "D":
        required.add("metadata_sensitivity")
    if (
        not isinstance(settings, dict)
        or not required <= settings.keys()
        or settings.keys() - required - {"metadata_path", "admission", "pilot_sessions"}
        or not run_id
    ):
        raise InputValidationError("Неверные настройки воспроизведения")
    pilot_sessions: int | None = None
    if "pilot_sessions" in settings:
        pilot_sessions = require_int64(settings["pilot_sessions"], nonnegative=True)
        if pilot_sessions == 0:
            raise InputValidationError("Размер пилота должен быть положительным целым")
    limits = settings["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != LIMITS
        or any(require_int64(value, nonnegative=True) == 0 for value in limits.values())
    ):
        raise InputValidationError("Нужны положительные ресурсные пределы воспроизведения")
    for name in ("preparation", "training_preparation", "training_features", "historical", "model"):
        digest_name = "package_sha256" if name == "historical" else "report_sha256"
        if not isinstance(settings[name], dict) or set(settings[name]) != {"path", digest_name}:
            raise InputValidationError("Нужны закреплённые входы воспроизведения")
    roots = {
        name: (project_root / settings[name]["path"]).resolve()
        for name in (
            "preparation",
            "training_preparation",
            "training_features",
            "historical",
            "model",
        )
    }
    destination = run_dir.resolve()
    if any(
        destination == path or destination in path.parents or path in destination.parents
        for path in roots.values()
    ):
        raise InputValidationError("Результат пересекается с входным запуском")
    if any(not path.is_dir() for name, path in roots.items() if name != "preparation"):
        raise InputValidationError("Отсутствует входной запуск")
    assert_source_binding(project_root)
    code = source_sha256(project_root)
    paths = {
        name: project_root / settings[name]
        for name in ("preparation_plan", "feature_schema", "recipe")
    }
    contents = {name: path.read_bytes() for name, path in paths.items()}
    plan = from_json(contents["preparation_plan"], PreparationPlan)
    schema = strict_json_loads(contents["feature_schema"])
    recipe = strict_json_loads(contents["recipe"])
    if validate_feature_schema(schema) or schema["event_semantics"] != "set_v1":
        raise InputValidationError("Неверная базовая схема признаков")
    validate_recipe(recipe)
    evaluation_plan = settings["evaluation_plan"]
    validate_evaluation_plan(evaluation_plan, plan)
    if evaluation_plan["period"] != period or (period == "D" and pilot_sessions is not None):
        raise InputValidationError("План не соответствует полному оценочному периоду")
    configuration = dict(settings["retrieval"])
    if "quotas" in configuration:
        configuration["quotas"] = tuple(configuration["quotas"])
    config = RetrievalConfig(**configuration)
    source = roots["preparation"]
    training = read_preparation_report(
        roots["training_preparation"],
        settings["training_preparation"]["report_sha256"],
        verify_tables=False,
    )
    historical_store = LocalArtifactStore(roots["historical"])
    head = historical_store.describe(PACKAGE_PATH)
    if head.sha256 != settings["historical"]["package_sha256"]:
        raise InputValidationError("Изменён исторический пакет")
    feature_bytes = (roots["training_features"] / "reports/features.json").read_bytes()
    model_bytes = (roots["model"] / "reports/ranker.json").read_bytes()
    if (
        sha256_bytes(feature_bytes) != settings["training_features"]["report_sha256"]
        or sha256_bytes(model_bytes) != settings["model"]["report_sha256"]
    ):
        raise InputValidationError("Изменён отчёт признаков или готовой модели")
    features, model_summary = strict_json_loads(feature_bytes), strict_json_loads(model_bytes)
    fid, mid = features["identity"], model_summary["identity"]
    catalog_hash = historical_store.describe("historical/catalog.parquet").sha256
    schema_hash = sha256_bytes(canonical_json_bytes(schema))
    if (
        features.get("status") != "completed"
        or fid.get("format_version") != "features_v1"
        or fid.get("period") != "B"
        or fid.get("scope") != plan.scope
        or fid.get("preparation_sha256") != settings["training_preparation"]["report_sha256"]
        or fid.get("historical_sha256") != head.sha256
        or fid.get("historical_descriptor_sha256") != head.descriptor_sha256
        or fid.get("catalog_sha256") != catalog_hash
        or fid.get("feature_schema_file_sha256") != sha256_bytes(contents["feature_schema"])
        or features.get("feature_schema") != schema
        or canonical_json_bytes(fid.get("configuration")) != canonical_json_bytes(asdict(config))
        or mid.get("features_sha256") != sha256_bytes(feature_bytes)
        or mid.get("catalog_sha256") != catalog_hash
        or mid.get("feature_schema_sha256") != schema_hash
        or mid.get("recipe_sha256") != sha256_bytes(contents["recipe"])
    ):
        raise InputValidationError("Признаки, схема, отбор или рецепт не соответствуют модели")
    model = load_ranker(roots["model"], expected_identity=mid)
    sensitivity = None
    if period == "D":
        sensitivity = load_sensitivity_model(
            project_root,
            settings["metadata_sensitivity"],
            training_hash=settings["training_preparation"]["report_sha256"],
            historical_hash=head.sha256,
            catalog_hash=catalog_hash,
            scope=plan.scope,
        )
        for name in ("model", "training_features"):
            path = (project_root / settings["metadata_sensitivity"][name]["path"]).resolve()
            if destination == path or destination in path.parents or path in destination.parents:
                raise InputValidationError("Результат пересекается с входами чувствительности")
    package = load_historical(historical_store)
    metadata_path = (
        project_root / settings["metadata_path"] if "metadata_path" in settings else None
    )
    if (
        metadata_path is not None
        and sha256_file(metadata_path) != package.input_hashes["metadata_sha256"]
    ):
        raise InputValidationError("Исходная поставка метаданных изменилась")
    # Все модели, пакет и допуск проверены. Только теперь читается сводка D.
    prepared = read_preparation_report(
        source, settings["preparation"]["report_sha256"], verify_tables=False
    )
    if prepared["inputs"]["plan"] != sha256_bytes(contents["preparation_plan"]):
        raise InputValidationError("План не соответствует подготовке оценки")

    def admission() -> str | None:
        return require_replay_admission(
            project_root,
            settings.get("admission"),
            training_prepared=training,
            historical=head,
            feature_schema_bytes=contents["feature_schema"],
            recipe=recipe,
            retrieval=asdict(config),
            model_summary=model_summary,
            plan=plan,
            prepared=prepared,
            evaluate=True,
            period=period,
            final_gate=final_gate,
        )

    manifest_hash = admission()  # До чтения любой таблицы C_smoke, включая её хеширование.
    input_tables = [
        f"datasets/{period}/prefixes.parquet",
        f"canonical/{period}.parquet",
        f"datasets/{period}/targets.parquet",
    ]
    if not set(input_tables) <= prepared["tables"].keys():
        raise InputValidationError("Нет полных таблиц C_smoke")
    prepared_requests = prepared["tables"][input_tables[0]]["rows"]
    prepared_selected = set(prepared["selection"][period])
    if prepared_requests != len(prepared_selected):
        raise InputValidationError("Число первых запросов не соответствует выбранным сессиям")
    if pilot_sessions is not None and pilot_sessions > prepared_requests:
        raise InputValidationError("Размер пилота превышает закреплённую выборку C_smoke")
    selected_ids = tuple(sorted(prepared_selected, key=lambda sid: session_key(sid, plan.seed)))
    if pilot_sessions is not None:
        selected_ids = selected_ids[:pilot_sessions]
    selected = set(selected_ids)
    total_requests = len(selected)
    if total_requests > limits["max_requests"]:
        raise InputValidationError("Превышено число запросов; усечение запрещено")
    identity = {
        "format_version": "replay_v1",
        "run_id": run_id,
        "scope": plan.scope,
        "period": period,
        "mode": "ideal_deterministic",
        "evaluation_protocol": plan.task.evaluation_protocol,
        "source_sha256": code,
        "settings_sha256": sha256_bytes(settings_bytes),
        "preparation_sha256": settings["preparation"]["report_sha256"],
        "preparation_code": prepared["source_sha256"],
        "preparation_plan_sha256": sha256_bytes(contents["preparation_plan"]),
        "training_preparation_sha256": settings["training_preparation"]["report_sha256"],
        "training_features_sha256": sha256_bytes(feature_bytes),
        "historical_sha256": head.sha256,
        "historical_descriptor_sha256": head.descriptor_sha256,
        "historical_code": head.code_sha256,
        "catalog_sha256": catalog_hash,
        "model_report_sha256": sha256_bytes(model_bytes),
        "model_sha256": model_summary["model_sha256"],
        "model_identity": mid,
        "feature_schema_sha256": schema_hash,
        "feature_schema_file_sha256": sha256_bytes(contents["feature_schema"]),
        "recipe_sha256": sha256_bytes(contents["recipe"]),
        "configuration": asdict(config),
        "policies": policies,
        "pilot_sessions": pilot_sessions,
        "selected_ids": selected_ids,
        "selected_count": total_requests,
        "selected_ids_sha256": sha256_bytes(canonical_json_bytes(selected_ids)),
        "evaluation_plan": evaluation_plan,
        "manifest_sha256": manifest_hash,
        **(
            {
                "metadata_sensitivity": {
                    "settings": settings["metadata_sensitivity"],
                    "model_identity": sensitivity[2].summary["identity"],
                }
            }
            if sensitivity is not None
            else {}
        ),
    }
    identity_bytes = canonical_json_bytes(identity)
    identity = strict_json_loads(identity_bytes)
    store = LocalArtifactStore(destination)

    def inputs_unchanged(*, targets: bool) -> None:
        if admission() != manifest_hash:
            raise InputValidationError("Допуск изменился")
        if sensitivity is not None:
            checked = load_sensitivity_model(
                project_root,
                settings["metadata_sensitivity"],
                training_hash=settings["training_preparation"]["report_sha256"],
                historical_hash=head.sha256,
                catalog_hash=catalog_hash,
                scope=plan.scope,
            )
            if (
                checked[0] != sensitivity[0]
                or checked[1] != sensitivity[1]
                or checked[2].summary != sensitivity[2].summary
            ):
                raise InputValidationError("Входы чувствительности изменились")
        if (
            source_sha256(project_root) != code
            or settings_path.read_bytes() != settings_bytes
            or any(path.read_bytes() != contents[name] for name, path in paths.items())
            or (roots["training_features"] / "reports/features.json").read_bytes() != feature_bytes
            or (roots["model"] / "reports/ranker.json").read_bytes() != model_bytes
            or sha256_file(roots["model"] / "models/ranker.txt") != model_summary["model_sha256"]
        ):
            raise InputValidationError("Код, настройки или готовая модель изменились")
        read_preparation_report(
            source, settings["preparation"]["report_sha256"], verify_tables=False
        )
        read_preparation_report(
            roots["training_preparation"],
            settings["training_preparation"]["report_sha256"],
            verify_tables=False,
        )
        verify_tables(source, prepared, paths=input_tables if targets else input_tables[:2])
        if historical_store.describe(PACKAGE_PATH).descriptor_sha256 != head.descriptor_sha256:
            raise InputValidationError("Исторический пакет изменился")
        for path, digest in package.paths.items():
            if historical_store.describe(path).descriptor_sha256 != digest:
                raise InputValidationError("Часть исторического пакета изменена")
        if (
            metadata_path is not None
            and sha256_file(metadata_path) != package.input_hashes["metadata_sha256"]
        ):
            raise InputValidationError("Исходная поставка метаданных изменилась")

    def outputs_unchanged(result: dict[str, Any]) -> None:
        if set(result.get("tables", {})) != set(TABLES):
            raise InputValidationError("Неполные таблицы воспроизведения")
        for relative, kind in TABLES.items():
            if table_info(store._path(relative), kind, period=period) != result["tables"][relative]:
                raise InputValidationError("Изменена таблица воспроизведения")

    if store._path(REPORT).exists():
        ready = strict_json_loads(store._path(REPORT).read_bytes())
        if (
            ready.get("status") != "completed"
            or canonical_json_bytes(ready.get("identity")) != identity_bytes
            or ready.get("requests") != total_requests
            or ready.get("limits") != limits
        ):
            raise InputValidationError("Готовое воспроизведение несовместимо с запуском")
        outputs_unchanged(ready)
        if store._path(MARKDOWN).read_bytes() != render_report(ready).encode("utf-8"):
            raise InputValidationError("Markdown не соответствует сохранённому результату")
        inputs_unchanged(targets=True)
        return ready
    if store._path(IDENTITY).exists():
        if store._path(IDENTITY).read_bytes() != identity_bytes:
            raise InputValidationError("Незавершённое воспроизведение несовместимо")
    else:
        if any(destination.rglob("*")):
            raise InputValidationError("Чужой непустой каталог результата")
        store._atomic_write(IDENTITY, identity_bytes)

    def clear_unpublished() -> None:
        if store._path(REPORT).exists() or store._path(IDENTITY).read_bytes() != identity_bytes:
            raise InputValidationError("Нельзя очистить опубликованный или чужой результат")
        for relative in (*TABLES, MARKDOWN):
            store._path(relative).unlink(missing_ok=True)

    clear_unpublished()

    def d_access_gate() -> str:
        # Полный pre_D проверен выше; между чтениями проверяем закреплённый паспорт.
        # Полные входы повторно проверяются перед публикацией.
        if final_digest is None:
            raise InputValidationError("Нет допуска pre_D")
        entry = settings.get("admission")
        if entry is not None and sha256_file(project_root / entry["manifest"]) != final_digest:
            raise InputValidationError("Допуск pre_D изменился")
        if settings_path.read_bytes() != settings_bytes:
            raise InputValidationError("Настройки D изменились")
        return final_digest

    audit: list[dict[str, Any]] = []
    access = DatasetAccess(
        source,
        code_sha256=prepared["source_sha256"],
        run_id=prepared["run_id"],
        scope=prepared["scope"],
        audit=audit.append,
        d_admission=d_access_gate if period == "D" else None,
    )
    stage, processed, last_stage, last_progress = "prefixes", 0, "", 0.0

    def check_budget(extra_bytes: int = 0) -> None:
        nonlocal last_stage, last_progress
        elapsed = time.monotonic() - started
        peak = _peak_memory()["peak_process_memory_bytes"]
        size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        if (
            elapsed > limits["max_seconds"]
            or peak is None
            or peak > limits["max_memory_bytes"]
            or size + extra_bytes > limits["max_result_bytes"]
        ):
            raise InputValidationError("Превышен предел ресурсов воспроизведения")
        if stage != last_stage or elapsed - last_progress >= 30:
            print(
                json.dumps(
                    {
                        "step": "replay",
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
            last_stage, last_progress = stage, elapsed

    try:
        verify_tables(source, prepared, paths=input_tables[:2])
        check_budget()
        observed: set[str] = set()
        seen: set[str] = set()
        request_ids: set[str] = set()
        interval = next(part for part in plan.periods if part.name == period)
        for sid, items in access.read(input_tables[0], PreparedRequest, purpose="prefix"):
            if sid not in prepared_selected or sid in seen or len(items) != 1:
                raise InputValidationError("Чужая сессия или неединственный первый запрос")
            item = items[0]
            if (
                item.request_id in request_ids
                or item.source_namespace != plan.source_namespace
                or not interval.start_ms
                <= item.t_q
                < interval.end_ms - plan.task.horizon_ms - plan.task.gap_ms
                or item.a_q != item.t_q
            ):
                raise InputValidationError("Неверный ID, время или пространство первого запроса")
            seen.add(sid)
            request_ids.add(item.request_id)
            if sid in selected:
                observed.update(
                    x.sku for x in (*item.prefix.detail_history, *item.prefix.full_last_detail)
                )
                if len(observed) > limits["max_observed_skus"]:
                    raise InputValidationError("Превышен предел наблюдаемых SKU")
                processed += 1
            check_budget()
        if seen != prepared_selected:
            raise InputValidationError("Не все выбранные сессии имеют запрос")
        stage = "metadata"
        check_budget()
        unknown = observed - set(package.catalog.skus)
        observed_records = {}
        if unknown:
            if metadata_path is None:
                raise InputValidationError("Для SKU вне A_G нужна исходная поставка метаданных")
            profile = strict_json_loads(prepared["profile_json"])
            if (
                sha256_bytes(canonical_json_bytes(profile))
                != package.input_hashes["metadata_profile_sha256"]
            ):
                raise InputValidationError("Профиль метаданных не соответствует A_G")
            metadata = open_metadata_source(
                metadata_path, profile, expected_sha256=package.input_hashes["metadata_sha256"]
            )
            observed_records = {
                item.sku: item for item in metadata.read(sorted(unknown, key=id_bytes))
            }
        index = RetrievalIndex(package, catalog_hash, observed_records)
        engine = FeatureEngine(index, schema, config)
        multiset_engine = FeatureEngine(index, {**schema, "event_semantics": "multiset_v1"}, config)
        sensitivity_engine = None
        if sensitivity is not None:
            sensitivity_engine = FeatureEngine(
                RetrievalIndex(package, catalog_hash, config_id=NO_METADATA_CONFIG_ID),
                sensitivity[0],
                sensitivity[1],
            )
        stage, processed = "recommendations", 0
        check_budget()
        with TableWriter(
            store._path(ANSWERS), ReplayRecord, batch_rows=limits["batch_size"]
        ) as writer:
            streams = (
                (
                    row
                    for row in access.read(input_tables[0], PreparedRequest, purpose="prefix")
                    if row[0] in selected
                ),
                (
                    row
                    for row in iter_sessions(
                        source / input_tables[1],
                        CanonicalBlock,
                        batch_size=limits["batch_size"],
                        max_session_records=plan.max_session_blocks,
                        max_session_bytes=plan.max_session_bytes,
                    )
                    if row[0] in selected
                ),
            )
            for first, second in zip_longest(*streams):
                if first is None or second is None or first[0] != second[0] or len(first[1]) != 1:
                    raise InputValidationError(
                        "Канонические блоки не соответствуют первым запросам"
                    )
                sid, items = first
                item, blocks = items[0], second[1]
                rows = []
                context = bind_context(item, index)
                for policy in policies:
                    active_engine, active_model, active_schema = engine, model, schema
                    active_context = context
                    if policy == NO_METADATA_CONFIG_ID:
                        if sensitivity is None or sensitivity_engine is None:
                            raise InputValidationError("Нет собственной модели чувствительности")
                        active_engine, active_model, active_schema = (
                            sensitivity_engine,
                            sensitivity[2],
                            sensitivity[0],
                        )
                        active_context = bind_context(item, active_engine.index)
                    diagnostics: dict[str, Any] = {}
                    tick = time.monotonic()
                    answer = recommend(
                        active_context,
                        active_engine,
                        active_model,
                        policy="GCS+Rk" if policy == NO_METADATA_CONFIG_ID else policy,
                        package_id=active_model.summary["model_sha256"],
                        trace_ref=f"{run_id}/{ANSWERS}#{item.request_id}/{policy}",
                        diagnostics=diagnostics,
                        diagnostic_depth=evaluation_plan["diagnostic_depth"],
                    )
                    elapsed_ms = max(0, round((time.monotonic() - tick) * 1000))
                    answer = replace(answer, elapsed_ms=elapsed_ms)
                    batch = diagnostics.pop("features")
                    saved = {
                        "source_lists": diagnostics["source_lists"],
                        "feature_mask_counts": None,
                        "feature_rows": None,
                        "mask_reason": "not_applicable",
                    }
                    if policy in {"GCS+Rk", NO_METADATA_CONFIG_ID}:
                        if not isinstance(batch, FeatureBatch):
                            raise InputValidationError(
                                "Готовая модель не сформировала базовые признаки"
                            )
                        saved.update(
                            feature_rows=len(batch.skus),
                            mask_reason=None,
                            feature_mask_counts={
                                column["name"]: sum(not row[i] for row in batch.masks)
                                for i, column in enumerate(active_schema["columns"])
                            },
                        )
                    if policy == "GCS+Rk":
                        saved["sensitivity"] = compare_multiplicities(
                            item,
                            blocks,
                            plan.task,
                            engine,
                            multiset_engine,
                            base_candidates=answer.candidates,
                            base_features=batch,
                        )
                    rows.append(
                        ReplayRecord(sid, policy, answer, canonical_json_bytes(saved).decode())
                    )
                writer.append(sid, rows)
                processed += 1
                check_budget()
        if processed != total_requests:
            raise InputValidationError("Число ответов не соответствует входу")
        stage = "evaluation"
        check_budget()
        summary = summarize_replay(
            source,
            prepared,
            store._path(ANSWERS),
            package,
            plan,
            evaluation_plan,
            destination,
            check_budget,
            selected_ids=selected_ids if pilot_sessions is not None else None,
            d_admission=d_access_gate if period == "D" else None,
        )
        tables = {
            path: table_info(store._path(path), kind, period=period)
            for path, kind in TABLES.items()
        }
        if any(info["rows"] != total_requests * len(policies) for info in tables.values()):
            raise InputValidationError("Не все политики сохранены для каждой сессии")
        result = {
            **summary,
            "format_version": "replay_v1",
            "status": "completed",
            "run_id": run_id,
            "identity": identity,
            "requests": total_requests,
            "tables": tables,
            "limits": limits,
            "reads": audit,
            "elapsed_seconds": time.monotonic() - started,
            **_peak_memory(),
        }
        if before_publish is not None:
            before_publish()
        stage = "validation"
        check_budget()
        inputs_unchanged(targets=True)
        outputs_unchanged(result)
        result.update(elapsed_seconds=time.monotonic() - started, **_peak_memory())
        result = strict_json_loads(canonical_json_bytes(result))
        markdown = render_report(result).encode("utf-8")
        encoded = canonical_json_bytes(result)
        stage = "publication"
        check_budget(len(markdown) + len(encoded))
        store._atomic_write(MARKDOWN, markdown)
        store._atomic_write(REPORT, encoded)
        store._path(IDENTITY).unlink()
        return result
    except BaseException:
        if not store._path(REPORT).exists():
            clear_unpublished()
        raise
