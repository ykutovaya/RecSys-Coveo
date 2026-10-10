"""Привязка будущего обучения B к фактическому preflight до чтения таблиц."""

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    ArtifactDescriptor,
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.config import validate_manifest
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json
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
    stage: str = "preflight",
    operation: str | None = None,
) -> str | None:
    """Искусственный путь не даёт Э1-10; фактический требует связанного паспорта.

    Вызывающая команда уже проверила отчёт подготовки и исторический пакет.
    Свидетельство data закрепляет точные план и отчёт; контрольные суммы исходных
    CSV повторно проверяет существующий валидатор, без разбора будущих событий.
    """
    if stage not in {"preflight", "post_training", "pre_D"}:
        raise InputValidationError("Неверная стадия допуска B/C_smoke")
    if prepared.get("scope") == historical.scope == "synthetic":
        return None
    if prepared.get("scope") != "actual" or historical.scope != "actual" or admission is None:
        raise InputValidationError(
            "Реальный B требует отдельного допуска Э1-10 без смешения областей"
        )
    try:
        expected = {"manifest", "manifest_sha256", "run_dir", "input_paths"}
        if stage == "pre_D":
            expected.add("plan_references")
        if set(admission) != expected:
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
        result = validate_manifest(
            manifest,
            stage,
            store,
            input_paths=paths,
            operation=operation,
            **({"plan_references": admission["plan_references"]} if stage == "pre_D" else {}),
        )
        if not result.gate_passed:
            detail = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
            raise InputValidationError(f"Допуск {stage} не пройден: {detail}")
        return digest
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise InputValidationError(str(exc)) from exc


def require_replay_admission(
    project_root: Path,
    admission: dict[str, Any] | None,
    *,
    training_prepared: dict[str, Any],
    historical: ArtifactDescriptor,
    feature_schema_bytes: bytes,
    recipe: dict[str, Any],
    retrieval: dict[str, Any],
    model_summary: dict[str, Any],
    plan: PreparationPlan,
    prepared: dict[str, Any] | None = None,
    evaluate: bool = False,
    period: str = "C_smoke",
    final_gate: Callable[[], str] | None = None,
) -> str | None:
    """Допуск C_smoke до подготовки; готовый набор проверяется перед выдачей.

    Код и preflight принадлежат потребителю. Идентичность модели сохраняет
    производителя, но связывается с принятым B, схемой и историческим пакетом.
    """
    if period not in {"C_smoke", "D"} or [p.name for p in plan.periods if p.sample_size] != [
        period
    ]:
        raise InputValidationError("Оценочная подготовка разрешена только для C_smoke")
    final_digest = None
    if period == "D":
        if final_gate is None or not (final_digest := final_gate()):
            raise InputValidationError("Нет допуска pre_D")
    try:
        original_plan = from_json(training_prepared["plan_json"].encode(), PreparationPlan)
        for name in ("scope", "source_namespace", "task", "coverage", "seed"):
            _same(getattr(plan, name), getattr(original_plan, name), f"C_smoke/{name}")
        _same(
            [(p.name, p.start_ms, p.end_ms) for p in plan.periods],
            [(p.name, p.start_ms, p.end_ms) for p in original_plan.periods],
            "границы C_smoke",
        )
        _same(plan.diagnostic_sessions, 0, "нормативный отбор C_smoke")
        _same(model_summary["status"], "completed", "готовность модели")
        identity = model_summary["identity"]
        for key, value in {
            "scope": plan.scope,
            "period": "B",
            "historical_sha256": historical.sha256,
            "preparation_sha256": sha256_bytes(canonical_json_bytes(training_prepared)),
            "feature_schema_file_sha256": sha256_bytes(feature_schema_bytes),
        }.items():
            _same(identity[key], value, f"модель/{key}")
        _same(model_summary["recipe"], recipe, "рецепт готовой модели")
        if prepared is not None:
            _same(prepared["status"], "completed", "готовность C_smoke")
            _same(prepared["scope"], plan.scope, "область C_smoke")
            prepared_plan_bytes = prepared["plan_json"].encode("utf-8")
            _same(from_json(prepared_plan_bytes, PreparationPlan), plan, "план C_smoke")
            _same(
                sha256_bytes(prepared_plan_bytes), prepared["inputs"]["plan"], "байты плана C_smoke"
            )
            _same(prepared["telemetry_complete"], None, "телеметрия C_smoke")
            for name, digest in training_prepared["inputs"].items():
                if name != "plan":
                    _same(prepared["inputs"][name], digest, f"C_smoke/{name}")
            selected = prepared["selection"][period]
            limit = next(p.sample_size for p in plan.periods if p.name == period)
            if (
                len(selected) > limit
                or len(selected) != len(set(selected))
                or any(ids for name, ids in prepared["selection"].items() if name != period)
            ):
                raise InputValidationError("Состав C_smoke не соответствует плану")
        if period == "D":
            return final_digest
        if plan.scope == historical.scope == "synthetic":
            return None
        digest = require_training_admission(
            project_root,
            admission,
            prepared=training_prepared,
            historical=historical,
            feature_schema_bytes=feature_schema_bytes,
            recipe=recipe,
            retrieval=retrieval,
            stage="post_training",
            operation="evaluate" if evaluate else None,
        )
        if admission is None or digest is None:
            raise InputValidationError("Нет фактического допуска C_smoke")
        manifest_bytes = (project_root / admission["manifest"]).read_bytes()
        _same(sha256_bytes(manifest_bytes), digest, "неизменность допуска")
        manifest = strict_json_loads(manifest_bytes)
        store = LocalArtifactStore(project_root / admission["run_dir"])
        data = strict_json_loads(store.read_by_sha256(manifest["evidence"]["data"]))
        _same(
            data.get("evaluation_preparation_plan_sha256"),
            sha256_bytes(to_json(plan)),
            "план C_smoke в допуске",
        )
        models = strict_json_loads(
            store.read_by_sha256(manifest["artifacts"]["trained_models_sha256"])
        )
        entries = [
            row
            for row in models["configurations"]
            if row["config_id"] == manifest["base_model"]["config_id"]
        ]
        if len(entries) != 1:
            raise InputValidationError("Нет единственной закреплённой базовой модели")
        for name, value in {
            "model_sha256": model_summary["model_sha256"],
            "feature_schema_sha256": sha256_bytes(feature_schema_bytes),
            "actual_tree_count": model_summary["actual_tree_count"],
            "training_status": "passed",
        }.items():
            _same(entries[0][name], value, f"индекс модели/{name}")
        return digest
    except (OSError, TypeError, ValueError, KeyError, StopIteration) as exc:
        raise InputValidationError(str(exc)) from exc


def final_execution_binding(project_root: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Часть before_D, известная до подготовки D; не читает её результат."""
    execution = {
        key: value for key, value in settings.items() if key not in {"admission", "preparation"}
    }
    files = {"preparation_plan": sha256_file(project_root / settings["preparation_plan"])}
    for name, profile in (("primary", settings), ("metadata", settings["metadata_sensitivity"])):
        for key in ("feature_schema", "recipe"):
            files[f"{name}/{key}"] = sha256_file(project_root / profile[key])
    return {"settings": execution, "files": files}


def require_final_admission(project_root: Path, settings: dict[str, Any]) -> str:
    """Действующий pre_D и точные входы двух моделей, до любого обращения к D."""
    from coveo_recsys.config.manifest import fixed_policy_without_selection
    from coveo_recsys.contracts.retrieval import BASE_CONFIG_ID, NO_METADATA_CONFIG_ID
    from coveo_recsys.data.prepared import read_report
    from coveo_recsys.evaluation.reporting import validate_evaluation_plan
    from coveo_recsys.historical.load import PACKAGE_PATH

    try:
        if not isinstance(settings, dict):
            raise InputValidationError("Для допуска pre_D нужен объект настроек")
        admission = settings.get("admission")
        if not isinstance(admission, dict) or set(admission) != {
            "manifest",
            "manifest_sha256",
            "run_dir",
            "input_paths",
            "plan_references",
        }:
            raise InputValidationError("Для D нужен допуск pre_D с внешней ссылкой before_D")
        manifest_bytes = (project_root / admission["manifest"]).read_bytes()
        digest = sha256_bytes(manifest_bytes)
        _same(digest, admission["manifest_sha256"], "SHA-256 pre_D")
        manifest = strict_json_loads(manifest_bytes)
        _same(manifest["stage"], "pre_D", "стадия D")
        store_path = project_root / admission["run_dir"]
        if not store_path.is_dir():
            raise InputValidationError("Нет свидетельств pre_D")
        store = LocalArtifactStore(store_path)
        result = validate_manifest(
            manifest,
            "pre_D",
            store,
            plan_references=admission["plan_references"],
            input_paths={
                name: project_root / value for name, value in admission["input_paths"].items()
            },
            operation="evaluate",
        )
        if not result.gate_passed:
            raise InputValidationError(
                "Допуск pre_D не пройден: "
                + "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
            )
        if not fixed_policy_without_selection(manifest):
            raise InputValidationError("evaluate поддерживает одну закреплённую базу без подбора")
        if "pilot_sessions" in settings:
            raise InputValidationError("D оценивается целиком по закреплённому плану, не как пилот")
        plan = from_json(
            (project_root / settings["preparation_plan"]).read_bytes(), PreparationPlan
        )
        evaluation = settings["evaluation_plan"]
        validate_evaluation_plan(evaluation, plan)
        _same(evaluation["period"], "D", "период итоговой оценки")
        _same([p.name for p in plan.periods if p.sample_size], ["D"], "отбор только D")
        if plan.scope == "actual" and manifest["scope"] not in {"actual", "offline_first_run"}:
            raise InputValidationError("Искусственный допуск не открывает реальный D")
        frozen = strict_json_loads(store.read_by_sha256(admission["plan_references"]["before_D"]))
        _same(
            frozen.get("execution"),
            final_execution_binding(project_root, settings),
            "параметры evaluate в before_D",
        )
        intervals, inference = evaluation["intervals"], manifest["inference"]
        for actual, expected in (
            (intervals["grid_step_ms"], inference["calendar_cell_seconds"] * 1000),
            (intervals["block_cells"], inference["block_cells"]),
            (intervals["sensitivity_block_cells"], inference["sensitivity_block_cells"]),
            (
                intervals["half_width_fraction"]["GCS+Rk"],
                inference["single_policy_half_width_fraction"],
            ),
        ):
            _same(actual, expected, "основная оценка в before_D")
        _same(
            intervals["dependence"]["GCS+Rk"],
            inference["dependence_assumptions"]["single_policy_hit"],
            "предпосылки основной оценки",
        )
        training = read_report(
            project_root / settings["training_preparation"]["path"],
            settings["training_preparation"]["report_sha256"],
            verify_tables=False,
        )
        original = from_json(training["plan_json"].encode(), PreparationPlan)
        for key in ("scope", "source_namespace", "task", "coverage", "seed"):
            _same(getattr(plan, key), getattr(original, key), f"план D/{key}")
        _same(
            [(p.name, p.start_ms, p.end_ms) for p in plan.periods],
            [(p.name, p.start_ms, p.end_ms) for p in original.periods],
            "границы плана D",
        )
        _same(plan.diagnostic_sessions, 0, "нормативная подготовка D")
        historical = LocalArtifactStore(project_root / settings["historical"]["path"]).describe(
            PACKAGE_PATH
        )
        _same(historical.sha256, settings["historical"]["package_sha256"], "пакет D")
        # Проверки B и исходных файлов остаются прежними; D здесь не читается.
        require_training_admission(
            project_root,
            admission,
            prepared=training,
            historical=historical,
            feature_schema_bytes=(project_root / settings["feature_schema"]).read_bytes(),
            recipe=strict_json_loads((project_root / settings["recipe"]).read_bytes()),
            retrieval=asdict(
                RetrievalConfig(
                    **{
                        **settings["retrieval"],
                        **(
                            {"quotas": tuple(settings["retrieval"]["quotas"])}
                            if "quotas" in settings["retrieval"]
                            else {}
                        ),
                    }
                )
            ),
            stage="pre_D",
            operation="evaluate",
        )
        models = strict_json_loads(
            store.read_by_sha256(manifest["artifacts"]["trained_models_sha256"])
        )
        for config_id, profile in (
            (BASE_CONFIG_ID, settings),
            (NO_METADATA_CONFIG_ID, settings["metadata_sensitivity"]),
        ):
            content = (project_root / profile["model"]["path"] / "reports/ranker.json").read_bytes()
            _same(sha256_bytes(content), profile["model"]["report_sha256"], "отчёт готовой модели")
            model = strict_json_loads(content)
            entries = [row for row in models["configurations"] if row["config_id"] == config_id]
            if len(entries) != 1:
                raise InputValidationError("В pre_D нужны обе закреплённые модели")
            for key, value in {
                "model_sha256": model["model_sha256"],
                "feature_schema_sha256": sha256_file(project_root / profile["feature_schema"]),
                "actual_tree_count": model["actual_tree_count"],
                "training_status": "passed",
            }.items():
                _same(entries[0][key], value, f"модель {config_id}/{key}")
            _same(
                sha256_file(project_root / profile["model"]["path"] / "models/ranker.txt"),
                model["model_sha256"],
                "байты модели",
            )
        return digest
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        raise InputValidationError(str(exc)) from exc


def require_evaluation_preparation(
    project_root: Path, settings_path: Path, plan: PreparationPlan, *, profile_sha256: str
) -> str | None:
    """Проверяет готовые входы оценки, не требуя ещё не созданный C_smoke."""
    from coveo_recsys.config import validate_feature_schema
    from coveo_recsys.data.prepared import read_report
    from coveo_recsys.evaluation.reporting import validate_evaluation_plan
    from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
    from coveo_recsys.ranking.load import load_ranker
    from coveo_recsys.ranking.train import validate_recipe

    try:
        settings = strict_json_loads(settings_path.read_bytes())
        period = settings["evaluation_plan"]["period"]
        final_gate = (
            (lambda: require_final_admission(project_root, settings)) if period == "D" else None
        )
        if final_gate is not None:
            final_gate()
        validate_evaluation_plan(settings["evaluation_plan"], plan)
        retrieval = dict(settings["retrieval"])
        if "quotas" in retrieval:
            retrieval["quotas"] = tuple(retrieval["quotas"])
        retrieval = strict_json_loads(canonical_json_bytes(asdict(RetrievalConfig(**retrieval))))
        planned = from_json(
            (project_root / settings["preparation_plan"]).read_bytes(), PreparationPlan
        )
        _same(planned, plan, "план подготовки C_smoke")
        schema_bytes = (project_root / settings["feature_schema"]).read_bytes()
        if validate_feature_schema(strict_json_loads(schema_bytes)):
            raise InputValidationError("Неверная схема признаков")
        recipe = strict_json_loads((project_root / settings["recipe"]).read_bytes())
        validate_recipe(recipe)
        training = settings["training_preparation"]
        prepared = read_report(
            project_root / training["path"], training["report_sha256"], verify_tables=False
        )
        _same(profile_sha256, prepared["inputs"]["profile"], "профиль входов C_smoke")
        historical_store = LocalArtifactStore(project_root / settings["historical"]["path"])
        historical = historical_store.describe(PACKAGE_PATH)
        _same(historical.sha256, settings["historical"]["package_sha256"], "исторический пакет")
        if period == "D":
            package = load_historical(historical_store)
            if "metadata_path" in settings:
                _same(
                    sha256_file(project_root / settings["metadata_path"]),
                    package.input_hashes["metadata_sha256"],
                    "поставка метаданных",
                )
        model_root = project_root / settings["model"]["path"]
        model_bytes = (model_root / "reports/ranker.json").read_bytes()
        _same(sha256_bytes(model_bytes), settings["model"]["report_sha256"], "отчёт модели")
        model_summary = strict_json_loads(model_bytes)
        model = load_ranker(model_root, expected_identity=model_summary["identity"])
        _same(model.summary, model_summary, "неизменность модели")
        feature_bytes = (
            project_root / settings["training_features"]["path"] / "reports/features.json"
        ).read_bytes()
        feature_hash = sha256_bytes(feature_bytes)
        _same(feature_hash, settings["training_features"]["report_sha256"], "отчёт признаков B")
        _same(feature_hash, model_summary["identity"]["features_sha256"], "признаки готовой модели")
        features = strict_json_loads(feature_bytes)
        _same(features["status"], "completed", "готовность признаков B")
        for name, expected in {
            "period": "B",
            "preparation_sha256": training["report_sha256"],
            "historical_sha256": historical.sha256,
            "feature_schema_file_sha256": sha256_bytes(schema_bytes),
            "configuration": retrieval,
        }.items():
            _same(features["identity"][name], expected, f"признаки B/{name}")
        if period == "D":
            from coveo_recsys.cli.replay import load_sensitivity_model

            load_sensitivity_model(
                project_root,
                settings["metadata_sensitivity"],
                training_hash=training["report_sha256"],
                historical_hash=historical.sha256,
                catalog_hash=model_summary["identity"]["catalog_sha256"],
                scope=plan.scope,
            )
        return require_replay_admission(
            project_root,
            settings.get("admission"),
            training_prepared=prepared,
            historical=historical,
            feature_schema_bytes=schema_bytes,
            recipe=recipe,
            retrieval=retrieval,
            model_summary=model_summary,
            plan=plan,
            period=period,
            final_gate=final_gate,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InputValidationError(str(exc)) from exc
