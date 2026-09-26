"""Обучение по сохранённым таблицам B; без повторного разбора сырых событий."""

import json
import sys
import time
from collections.abc import Callable, Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.config import validate_feature_schema
from coveo_recsys.contracts import InputValidationError, require_int64
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.records import CandidateResult, FeatureBatch
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.prepared import iter_sessions, read_report, table_info, verify_tables
from coveo_recsys.environment import environment_report
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.ranking.load import load_ranker
from coveo_recsys.ranking.table import build_training_table
from coveo_recsys.ranking.train import fit_ranker, validate_recipe
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

REPORT = "reports/ranker.json"
MODEL = "models/ranker.txt"
IDENTITY = "models/identity.json"


def run_training(
    project_root: Path,
    settings_path: Path,
    run_dir: Path,
    run_id: str,
    *,
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from coveo_recsys.cli.admission import require_training_admission
    from coveo_recsys.cli.features import TABLES
    from coveo_recsys.cli.preparation import _peak_memory
    from coveo_recsys.ranking import objective

    settings_bytes = settings_path.read_bytes()
    settings = strict_json_loads(settings_bytes)
    required = {"features", "preparation", "historical", "feature_schema", "recipe", "limits"}
    if (
        not isinstance(settings, dict)
        or not required <= settings.keys()
        or settings.keys() - required - {"admission"}
        or not run_id
    ):
        raise InputValidationError("Неверные настройки обучения")
    limits = settings["limits"]
    if set(limits) != {
        "max_requests",
        "max_rows",
        "max_seconds",
        "max_memory_bytes",
        "max_result_bytes",
    } or any(require_int64(v, nonnegative=True) == 0 for v in limits.values()):
        raise InputValidationError("Нужны положительные пределы обучения")
    for name, digest in [
        ("features", "report_sha256"),
        ("preparation", "report_sha256"),
        ("historical", "package_sha256"),
    ]:
        if set(settings[name]) != {"path", digest}:
            raise InputValidationError("Вход обучения не закреплён")
    source = (project_root / settings["preparation"]["path"]).resolve()
    feature_root = (project_root / settings["features"]["path"]).resolve()
    historical_root = (project_root / settings["historical"]["path"]).resolve()
    destination = run_dir.resolve()
    if any(
        destination == p or destination in p.parents or p in destination.parents
        for p in (source, feature_root, historical_root)
    ):
        raise InputValidationError("Результат пересекается с входным запуском")
    assert_source_binding(project_root)
    code = source_sha256(project_root)
    started = time.monotonic()
    schema_path, recipe_path = (project_root / settings[k] for k in ("feature_schema", "recipe"))
    schema_bytes, recipe_bytes = schema_path.read_bytes(), recipe_path.read_bytes()
    schema, recipe = strict_json_loads(schema_bytes), strict_json_loads(recipe_bytes)
    if validate_feature_schema(schema):
        raise InputValidationError("Неверная схема признаков")
    validate_recipe(recipe)
    prepared = read_report(source, settings["preparation"]["report_sha256"], verify_tables=False)
    historical_store = LocalArtifactStore(historical_root)
    head = historical_store.describe(PACKAGE_PATH)
    if head.sha256 != settings["historical"]["package_sha256"]:
        raise InputValidationError("Хеш исторического пакета не совпал")
    feature_store = LocalArtifactStore(feature_root)
    feature_bytes = feature_store._path("reports/features.json").read_bytes()
    if sha256_bytes(feature_bytes) != settings["features"]["report_sha256"]:
        raise InputValidationError("Изменён отчёт признаков")
    features = strict_json_loads(feature_bytes)
    fid = features["identity"]
    if (
        features.get("status") != "completed"
        or fid.get("format_version") != "features_v1"
        or fid.get("period") != "B"
        or fid.get("scope") != prepared["scope"]
        or fid.get("preparation_sha256") != settings["preparation"]["report_sha256"]
        or fid.get("historical_sha256") != head.sha256
        or fid.get("historical_descriptor_sha256") != head.descriptor_sha256
        or features.get("feature_schema") != schema
        or fid.get("feature_schema_file_sha256") != sha256_bytes(schema_bytes)
        or set(features.get("tables", {})) != set(TABLES)
    ):
        raise InputValidationError("Признаки несовместимы с входами обучения")

    def admission() -> str | None:
        return require_training_admission(
            project_root,
            settings.get("admission"),
            prepared=prepared,
            historical=head,
            feature_schema_bytes=schema_bytes,
            recipe=recipe,
            retrieval=fid["configuration"],
        )

    manifest_hash = admission()  # До любого чтения таблиц B.
    package = load_historical(historical_store)
    catalog_hash = historical_store.describe("historical/catalog.parquet").sha256
    if fid.get("catalog_sha256") != catalog_hash:
        raise InputValidationError("Признаки другого каталога")
    if features["requests"] > limits["max_requests"]:
        raise InputValidationError("Превышено число запросов; усечение запрещено")
    columns = schema["columns"]
    numeric_columns = tuple(c["name"] for c in columns if c["dtype"] != "category")
    categorical_columns = tuple(c["name"] for c in columns if c["dtype"] == "category")
    schema_hash = sha256_bytes(canonical_json_bytes(schema))
    objective_hash = sha256_file(Path(objective.__file__))
    identity = {
        "run_id": run_id,
        "scope": prepared["scope"],
        "period": "B",
        "evaluation_protocol": "observed_purchases_v1",
        "source_sha256": code,
        "settings_sha256": sha256_bytes(settings_bytes),
        "features_sha256": sha256_bytes(feature_bytes),
        "preparation_sha256": settings["preparation"]["report_sha256"],
        "historical_sha256": head.sha256,
        "catalog_sha256": catalog_hash,
        "feature_schema_file_sha256": sha256_bytes(schema_bytes),
        "feature_schema_sha256": schema_hash,
        "recipe_sha256": sha256_bytes(recipe_bytes),
        "objective_sha256": objective_hash,
        "manifest_sha256": manifest_hash,
    }
    identity_bytes = canonical_json_bytes(identity)
    store = LocalArtifactStore(destination)
    audit: list[dict[str, Any]] = []
    last_progress = 0.0

    def check_budget(extra_bytes: int = 0) -> None:
        nonlocal last_progress
        elapsed = time.monotonic() - started
        peak = _peak_memory()["peak_process_memory_bytes"]
        size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        if (
            elapsed > limits["max_seconds"]
            or peak is None
            or peak > limits["max_memory_bytes"]
            or size + extra_bytes > limits["max_result_bytes"]
        ):
            raise InputValidationError("Превышен предел ресурсов обучения")
        if elapsed - last_progress >= 10:
            print(
                json.dumps(
                    {
                        "step": "training",
                        "elapsed_seconds": elapsed,
                        "peak_process_memory_bytes": peak,
                    }
                ),
                file=sys.stderr,
            )
            last_progress = elapsed

    def inputs_unchanged() -> None:
        if (
            source_sha256(project_root) != code
            or settings_path.read_bytes() != settings_bytes
            or schema_path.read_bytes() != schema_bytes
            or recipe_path.read_bytes() != recipe_bytes
            or feature_store._path("reports/features.json").read_bytes() != feature_bytes
        ):
            raise InputValidationError("Код, настройки или признаки изменились")
        read_report(source, settings["preparation"]["report_sha256"], verify_tables=False)
        verify_tables(
            source, prepared, paths=["datasets/B/prefixes.parquet", "datasets/B/targets.parquet"]
        )
        for path, kind in TABLES.items():
            if table_info(feature_store._path(path), kind, period="B") != features["tables"][path]:
                raise InputValidationError("Таблица признаков изменена")
        if historical_store.describe(PACKAGE_PATH).descriptor_sha256 != head.descriptor_sha256:
            raise InputValidationError("Исторический пакет изменился")
        for path, digest in package.paths.items():
            if historical_store.describe(path).descriptor_sha256 != digest:
                raise InputValidationError("Часть исторического пакета изменена")
        if admission() != manifest_hash:
            raise InputValidationError("Допуск изменился")

    inputs_unchanged()
    check_budget()
    if store._path(REPORT).exists():
        ready = strict_json_loads(store._path(REPORT).read_bytes())
        if ready.get("identity") != identity:
            raise InputValidationError("Готовый результат несовместим с запуском")
        if ready.get("status") == "completed":
            load_ranker(destination, expected_identity=identity)
        elif ready.get("status") != "insufficient" or store._path(MODEL).exists():
            raise InputValidationError("Неверный результат обучения")
        return ready
    if store._path(IDENTITY).exists():
        if store._path(IDENTITY).read_bytes() != identity_bytes:
            raise InputValidationError("Незавершённое обучение несовместимо")
    else:
        store._atomic_write(IDENTITY, identity_bytes)
    access = DatasetAccess(
        source,
        code_sha256=prepared["source_sha256"],
        run_id=prepared["run_id"],
        scope=prepared["scope"],
        audit=audit.append,
    )

    def examples() -> Iterator[tuple[PreparedRequest, CandidateResult, FeatureBatch, TargetRecord]]:
        streams = (
            access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix"),
            iter_sessions(feature_store._path("features/candidates.parquet"), CandidateResult),
            iter_sessions(feature_store._path("features/values.parquet"), FeatureBatch),
            access.read("datasets/B/targets.parquet", TargetRecord, purpose="labels"),
        )
        count = 0
        catalog = set(package.catalog.skus)
        for rows in zip_longest(*streams):
            if any(row is None for row in rows):
                raise InputValidationError(
                    "Отсутствует запись запроса, кандидатов, признаков или цели"
                )
            valid = [row for row in rows if row is not None]
            if len({row[0] for row in valid}) != 1 or any(len(row[1]) != 1 for row in valid):
                raise InputValidationError("Несогласованные ключи или число первых запросов")
            request, candidates, batch, target = (row[1][0] for row in valid)
            if not (
                isinstance(request, PreparedRequest)
                and isinstance(candidates, CandidateResult)
                and isinstance(batch, FeatureBatch)
                and isinstance(target, TargetRecord)
            ):
                raise InputValidationError("Неверные типы записей обучающей группы")
            if not set(candidates.skus) <= catalog or set(candidates.skus) & set(
                request.prefix.purchased_skus
            ):
                raise InputValidationError("Недопустимые кандидаты обучения")
            count += 1
            if count > limits["max_requests"]:
                raise InputValidationError("Превышен предел запросов")
            check_budget()
            yield request, candidates, batch, target
        if count != features["requests"]:
            raise InputValidationError("Число запросов не соответствует сводке")

    table = build_training_table(
        examples(),
        feature_schema_sha256=schema_hash,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        max_candidates=fid["configuration"]["max_candidates"],
        max_rows=limits["max_rows"],
        max_memory_bytes=limits["max_memory_bytes"],
        max_disk_bytes=limits["max_result_bytes"],
    )
    check_budget()
    result: dict[str, Any] = {
        "format_version": "ranker_v1",
        "status": "insufficient",
        "reason": "no_training_pairs",
        "identity": identity,
        "feature_schema_sha256": schema_hash,
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "groups": table.statistics,
        "group_order_sha256": sha256_bytes(canonical_json_bytes(table.group_keys)),
        "limits": limits,
        "reads": audit,
        "environment": environment_report(project_root),
    }
    if table.statistics["pair_count"]:
        booster, training = fit_ranker(table, recipe, check_budget=check_budget)
        model_bytes = booster.model_to_string(num_iteration=booster.num_trees()).encode("utf-8")
        check_budget(len(model_bytes))
        if store._path(MODEL).exists():
            if store._path(MODEL).read_bytes() != model_bytes:
                raise InputValidationError("Повторное обучение изменило незавершённую модель")
        else:
            store._atomic_write(MODEL, model_bytes)
        result.update(
            training, status="completed", reason=None, model_sha256=sha256_bytes(model_bytes)
        )
    if before_publish is not None:
        before_publish()
    inputs_unchanged()
    if result["status"] == "completed":
        payload = store._path(MODEL).read_bytes()
        if sha256_bytes(payload) != result["model_sha256"]:
            raise InputValidationError("Модель изменилась до публикации")
        restored = lgb.Booster(model_str=payload.decode("utf-8"))
        if (
            restored.num_trees() != booster.num_trees()
            or restored.feature_name() != booster.feature_name()
        ):
            raise InputValidationError("Сохранённая модель несовместима")
        boundaries = np.cumsum((0, *table.group_sizes))
        first = 0
        while first < len(table.group_sizes):
            last = first + 1
            while (
                last < len(table.group_sizes) and boundaries[last + 1] - boundaries[first] <= 4096
            ):
                last += 1
            start, stop = int(boundaries[first]), int(boundaries[last])
            rows = table.data[start:stop]
            before = np.asarray(
                booster.predict(
                    rows, raw_score=True, num_iteration=booster.num_trees(), num_threads=1
                )
            )
            after = np.asarray(
                restored.predict(
                    rows, raw_score=True, num_iteration=restored.num_trees(), num_threads=1
                )
            )
            if not np.isfinite(after).all() or not np.allclose(
                before, after, atol=1e-10, rtol=1e-8
            ):
                raise InputValidationError("Сохранение изменило баллы модели")
            for group in range(first, last):
                low, high = boundaries[group : group + 2] - start
                if not np.array_equal(
                    np.argsort(-before[low:high], kind="stable"),
                    np.argsort(-after[low:high], kind="stable"),
                ):
                    raise InputValidationError("Сохранение изменило порядок баллов группы")
            first = last
            check_budget()
    result.update(elapsed_seconds=time.monotonic() - started, **_peak_memory())
    payload_size = store._path(MODEL).stat().st_size if result["status"] == "completed" else 0
    result["result_bytes"] = payload_size
    while True:
        encoded = canonical_json_bytes(result)
        total = payload_size + len(encoded)
        if result["result_bytes"] == total:
            break
        result["result_bytes"] = total
    check_budget(len(encoded))
    store._atomic_write(REPORT, encoded)
    store._path(IDENTITY).unlink()
    return strict_json_loads(encoded)
