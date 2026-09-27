"""Порционный локальный C_smoke на неизменной готовой модели."""

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
from coveo_recsys.contracts.evaluation import POLICIES, EvaluationRecord, ReplayRecord
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock, FeatureBatch
from coveo_recsys.contracts.retrieval import RetrievalConfig
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
from coveo_recsys.ranking.load import load_ranker
from coveo_recsys.ranking.train import validate_recipe
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

REPORT = "reports/replay.json"
MARKDOWN = "reports/replay.md"
IDENTITY = "replay/identity.json"
ANSWERS = "replay/answers.parquet"
SESSIONS = "replay/sessions.parquet"
TABLES = {ANSWERS: ReplayRecord, SESSIONS: EvaluationRecord}


def read_report(run: Path) -> dict[str, Any]:
    """Проверить готовые таблицы и воспроизвести Markdown без доступа к сырью."""
    from coveo_recsys.evaluation.reporting import render_report

    if not run.is_dir():
        raise InputValidationError("Отсутствует завершённый запуск воспроизведения")
    store = LocalArtifactStore(run)
    summary = strict_json_loads(store._path(REPORT).read_bytes())
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "completed"
        or summary.get("format_version") != "replay_v1"
        or summary.get("identity", {}).get("format_version") != "replay_v1"
        or summary.get("identity", {}).get("mode") != "ideal_deterministic"
        or summary.get("identity", {}).get("period") != "C_smoke"
        or summary.get("identity", {}).get("policies") != list(POLICIES)
        or type(summary.get("requests")) is not int
        or summary["requests"] < 0
        or set(summary.get("tables", {})) != set(TABLES)
    ):
        raise InputValidationError("Нет совместимого завершённого отчёта")
    for relative, kind in TABLES.items():
        actual = table_info(store._path(relative), kind, period="C_smoke")
        if actual != summary["tables"][relative] or actual["rows"] != summary["requests"] * len(
            POLICIES
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
    if period != "C_smoke":
        raise InputValidationError("Для replay разрешён только период C_smoke")
    from coveo_recsys.cli.admission import require_replay_admission
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
    if any(not root.is_dir() for root in roots.values()):
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
    configuration = dict(settings["retrieval"])
    if "quotas" in configuration:
        configuration["quotas"] = tuple(configuration["quotas"])
    config = RetrievalConfig(**configuration)
    source = roots["preparation"]
    prepared = read_preparation_report(
        source, settings["preparation"]["report_sha256"], verify_tables=False
    )
    training = read_preparation_report(
        roots["training_preparation"],
        settings["training_preparation"]["report_sha256"],
        verify_tables=False,
    )
    if prepared["inputs"]["plan"] != sha256_bytes(contents["preparation_plan"]):
        raise InputValidationError("План не соответствует подготовленному C_smoke")
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
        )

    manifest_hash = admission()  # До чтения любой таблицы C_smoke, включая её хеширование.
    package = load_historical(historical_store)
    input_tables = [
        "datasets/C_smoke/prefixes.parquet",
        "canonical/C_smoke.parquet",
        "datasets/C_smoke/targets.parquet",
    ]
    if not set(input_tables) <= prepared["tables"].keys():
        raise InputValidationError("Нет полных таблиц C_smoke")
    prepared_requests = prepared["tables"][input_tables[0]]["rows"]
    prepared_selected = set(prepared["selection"]["C_smoke"])
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
    metadata_path = (
        project_root / settings["metadata_path"] if "metadata_path" in settings else None
    )
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
        "policies": POLICIES,
        "pilot_sessions": pilot_sessions,
        "selected_ids": selected_ids,
        "selected_count": total_requests,
        "selected_ids_sha256": sha256_bytes(canonical_json_bytes(selected_ids)),
        "evaluation_plan": evaluation_plan,
        "manifest_sha256": manifest_hash,
    }
    identity_bytes = canonical_json_bytes(identity)
    identity = strict_json_loads(identity_bytes)
    store = LocalArtifactStore(destination)

    def inputs_unchanged(*, targets: bool) -> None:
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
        if admission() != manifest_hash:
            raise InputValidationError("Допуск изменился")

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
    audit: list[dict[str, Any]] = []
    access = DatasetAccess(
        source,
        code_sha256=prepared["source_sha256"],
        run_id=prepared["run_id"],
        scope=prepared["scope"],
        audit=audit.append,
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
                for policy in POLICIES:
                    diagnostics: dict[str, Any] = {}
                    tick = time.monotonic()
                    answer = recommend(
                        context,
                        engine,
                        model,
                        policy=policy,
                        package_id=model_summary["model_sha256"],
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
                    if policy == "GCS+Rk":
                        if not isinstance(batch, FeatureBatch):
                            raise InputValidationError(
                                "Готовая модель не сформировала базовые признаки"
                            )
                        saved.update(
                            feature_rows=len(batch.skus),
                            mask_reason=None,
                            feature_mask_counts={
                                column["name"]: sum(not row[i] for row in batch.masks)
                                for i, column in enumerate(schema["columns"])
                            },
                            sensitivity=compare_multiplicities(
                                item,
                                blocks,
                                plan.task,
                                engine,
                                multiset_engine,
                                base_candidates=answer.candidates,
                                base_features=batch,
                            ),
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
        )
        tables = {
            path: table_info(store._path(path), kind, period=period)
            for path, kind in TABLES.items()
        }
        if any(info["rows"] != total_requests * len(POLICIES) for info in tables.values()):
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
