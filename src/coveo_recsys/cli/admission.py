"""Привязка будущего обучения B к фактическому preflight до чтения таблиц."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    ArtifactDescriptor,
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    strict_json_loads,
)
from coveo_recsys.config import validate_manifest
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.data.io import FILES, parse_profile
from coveo_recsys.ranking import objective
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256


def _same(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise InputValidationError(f"Допуск B: не совпадает {name}")


def _milliseconds(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not value.endswith("Z") or timestamp.microsecond % 1000:
        raise InputValidationError("Допуск B требует UTC с точностью до миллисекунд")
    delta = timestamp - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def require_training_admission(
    project_root: Path,
    admission: dict[str, Any] | None,
    *,
    prepared: dict[str, Any],
    historical: ArtifactDescriptor,
    feature_schema_bytes: bytes,
    recipe: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
) -> str | None:
    """Искусственный путь не даёт Э1-10; фактический требует связанного паспорта.

    Вызывающая команда уже проверила отчёт подготовки и исторический пакет.
    Свидетельство data закрепляет точные план и отчёт; контрольные суммы исходных
    CSV повторно проверяет существующий валидатор, без разбора будущих событий.
    """
    if prepared.get("scope") == historical.scope == "synthetic":
        return None
    if prepared.get("scope") != "actual" or historical.scope != "actual" or admission is None:
        raise InputValidationError(
            "Реальный B требует отдельного допуска Э1-10 без смешения областей"
        )
    try:
        if set(admission) != {"manifest", "manifest_sha256", "run_dir", "input_paths"}:
            raise InputValidationError("Нужны паспорт, его SHA-256, каталог и три исходных пути")
        manifest_path = project_root / admission["manifest"]
        manifest_bytes = manifest_path.read_bytes()
        digest = sha256_bytes(manifest_bytes)
        _same(digest, admission["manifest_sha256"], "SHA-256 паспорта")
        manifest = strict_json_loads(manifest_bytes)
        if manifest["scope"] not in {"actual", "offline_first_run"}:
            raise InputValidationError("Искусственный паспорт не допускает реальные периоды")
        run_dir = project_root / admission["run_dir"]
        if not run_dir.is_dir():
            raise InputValidationError("Нет каталога свидетельств допуска")
        store = LocalArtifactStore(run_dir)
        input_paths = admission["input_paths"]
        _same(set(input_paths), set(FILES.values()), "состав исходных файлов")
        paths = {name: project_root / value for name, value in input_paths.items()}
        assert_source_binding(project_root)
        repro = manifest["reproducibility"]
        for field, current in {
            "code_commit_or_archive_sha256": source_sha256(project_root),
            "dependency_lockfile_sha256": sha256_bytes((project_root / "uv.lock").read_bytes()),
            "feature_schema_sha256": sha256_bytes(feature_schema_bytes),
            "ranker_callback_sha256": sha256_bytes(Path(objective.__file__).read_bytes()),
        }.items():
            _same(repro[field], current, field)
        _same(manifest["ranker"]["resolved_parameters"], recipe, "рецепт ранжировщика")
        _same(prepared["status"], "completed", "готовность подготовки")
        _same(prepared["telemetry_complete"], None, "неизвестная полнота телеметрии")
        inputs = prepared["inputs"]
        file_hashes = {item["name"]: item["sha256"] for item in manifest["dataset"]["files"]}
        for key, filename in FILES.items():
            _same(inputs[key], file_hashes[filename], f"подготовка/{filename}")
            _same(historical.inputs[key], inputs[key], f"исторический пакет/{filename}")
        plan_bytes = prepared["plan_json"].encode("utf-8")
        profile_bytes = prepared["profile_json"].encode("utf-8")
        _same(sha256_bytes(plan_bytes), inputs["plan"], "план подготовки")
        _same(sha256_bytes(profile_bytes), inputs["profile"], "профиль подготовки")
        _same(historical.inputs["profile"], inputs["profile"], "профиль исторического пакета")
        _same(manifest["dataset"]["schema_sha256"], inputs["profile"], "схема CSV")
        plan = from_json(plan_bytes, PreparationPlan)
        profile = parse_profile(strict_json_loads(profile_bytes), sha256=inputs["profile"])
        _same((plan.scope, profile.scope), ("actual", "actual"), "область настроек")
        _same(plan.source_namespace, profile.namespace, "пространство источника")
        _same(manifest["dataset"]["vector_dimensions"], profile.dimensions, "размерности векторов")
        _same(plan.coverage.confirmed_at_ms, None, "неизвестное покрытие плана")
        for name in ("start", "end"):
            _same(
                getattr(plan.coverage, f"{name}_ms"),
                _milliseconds(manifest["dataset"][f"coverage_{name}_utc"]),
                f"coverage.{name}",
            )
        _same(
            plan.coverage.gaps,
            tuple(
                (_milliseconds(gap["start_utc"]), _milliseconds(gap["end_utc"]))
                for gap in manifest["dataset"]["telemetry_gaps"]
            ),
            "разрывы покрытия",
        )
        for key, filename in FILES.items():
            _same(profile.expected_files[key]["expected_sha256"], inputs[key], filename)
        _same(plan.diagnostic_sessions, 0, "отсутствие технического отбора")
        _same(plan.seed, manifest["dataset"]["session_sampling"]["seed"], "seed отбора")
        _same(plan.seed, manifest["reproducibility"]["seed"], "seed запуска")
        task, base = manifest["task"], manifest["base_model"]
        for name, expected in {
            "horizon_ms": task["horizon_seconds"] * 1000,
            "gap_ms": task["inactivity_seconds"] * 1000,
            "max_requests": task["max_requests_per_session"],
            "event_semantics": manifest["dataset"]["event_semantics"],
            "evaluation_protocol": manifest["evaluation_protocol"],
            "query_dimension": profile.dimensions["query"],
            "detail_history_limit": base["history_max_details"],
            "block_history_limit": base["stored_history_max_blocks"],
        }.items():
            _same(getattr(plan.task, name), expected, f"task.{name}")
        _same(prepared["evaluation_protocol"], manifest["evaluation_protocol"], "протокол")
        for period in plan.periods:
            interval = manifest["periods"][f"period_{period.name}"]
            _same(period.start_ms, _milliseconds(interval["start_utc"]), f"начало {period.name}")
            _same(period.end_ms, _milliseconds(interval["end_utc"]), f"конец {period.name}")
        sampled = prepared["sampled"]
        _same(sampled["rule"], "lowest_seeded_hash_after_calendar_admission", "правило отбора")
        _same(sampled["normative_sample"], True, "календарный допуск")
        selected = prepared["selection"]["B"]
        budget = next(period.sample_size for period in plan.periods if period.name == "B")
        if not 0 < len(selected) <= budget or len(set(selected)) != len(selected):
            raise InputValidationError("Неверный состав или размер выборки B")
        count = manifest["dataset"]["session_sampling"]["session_count_by_period"]["period_B"]
        _same(count, len(selected), "число выбранных сессий B")
        _same(sampled["period_counts"]["B"]["selected"], count, "сводка отбора B")
        data = strict_json_loads(store.read_by_sha256(manifest["evidence"]["data"]))
        _same(data.get("preparation_plan_sha256"), inputs["plan"], "план в свидетельстве data")
        _same(
            data.get("preparation_report_sha256"),
            sha256_bytes(canonical_json_bytes(prepared)),
            "отчёт подготовки в свидетельстве data",
        )
        if retrieval is None:
            raise InputValidationError("Не заданы фактические параметры отбора кандидатов")
        quotas = base["quotas"]
        _same(
            list(retrieval["quotas"]),
            [
                quotas[name]
                for name in (
                    "graph_detail_detail",
                    "graph_detail_add",
                    "graph_detail_purchase",
                    "description",
                    "image",
                    "search",
                    "recent",
                    "popularity",
                )
            ],
            "квоты отбора",
        )
        for name, expected in {
            "max_candidates": task["candidate_limit"],
            "history_limit": base["history_max_details"],
            "tau_history_ms": base["history_decay_seconds"] * 1000,
            "search_max_age_ms": base["search_expiry_seconds"] * 1000,
            "vector_epsilon": base["vector_epsilon"],
        }.items():
            _same(retrieval[name], expected, f"retrieval.{name}")
        for name, expected in {
            "snapshot_time_ms": plan.periods[0].end_ms,
            "graph_max_group_distance": base["graph_max_group_distance"],
            "graph_window_ms": base["graph_window_seconds"] * 1000,
            "graph_tau_ms": base["graph_decay_seconds"] * 1000,
            "graph_min_support": base["graph_min_sessions"],
            "graph_max_neighbors": base["graph_max_degree"],
            "graph_regularization": base["graph_row_regularizer"],
            "popularity_smoothing": base["popularity_smoothing"],
            "vector_epsilon": base["vector_epsilon"],
        }.items():
            _same(historical.configuration[name], expected, f"historical.{name}")
        result = validate_manifest(manifest, "preflight", store, input_paths=paths)
        if not result.gate_passed:
            detail = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
            raise InputValidationError(f"Э1-10 preflight не пройден: {detail}")
        return digest
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise InputValidationError(str(exc)) from exc
