"""Сборка допуска нового B/C_smoke из проверенных файлов текущего запуска."""

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.config import init_manifest, validate_manifest, verification_configuration
from coveo_recsys.config.resources import validator_fingerprint_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.io import FILES, parse_profile
from coveo_recsys.data.prepared import REPORT, read_report, verify_tables
from coveo_recsys.environment import environment_report
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.ranking import objective
from coveo_recsys.ranking.load import load_ranker
from coveo_recsys.ranking.train import validate_recipe
from coveo_recsys.verification.preparation import DATA_RULES
from coveo_recsys.verification.provenance import assert_source_binding, source_inventory
from coveo_recsys.verification.test_evidence import read_test_evidence, stage_test_rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputValidationError(message)


def checked_preparation(
    path: Path, report: dict[str, Any], code: str, period: str
) -> dict[str, Any]:
    checks: dict[str, Any] = strict_json_loads((path / "reports/data-check.json").read_bytes())
    digest = sha256_file(path / REPORT)
    _require(
        checks.get("check_report_format_version") == "preparation_check_v1"
        and checks.get("status") == "passed"
        and report["status"] == "completed"
        and checks.get("scope") == report["scope"] == "actual"
        and checks.get("period") == period
        and checks.get("current_code_sha256") == report["source_sha256"] == code
        and all(checks.get(key) is True for key in DATA_RULES),
        "Нет полной независимой проверки текущей подготовки",
    )
    expected = {
        "run_id": report["run_id"],
        "tables": report["tables"],
        "inputs": report["inputs"],
        "preparation_code_sha256": code,
        "report_sha256": digest,
        "plan_sha256": report["inputs"]["plan"],
    }
    count = len(report["selection"][period])
    _require(
        all(checks.get("bindings", {}).get(key) == value for key, value in expected.items())
        and checks.get("preparation_report_sha256")
        == digest
        == sha256_bytes(canonical_json_bytes(report))
        and checks.get("preparation_plan_sha256") == report["inputs"]["plan"]
        and checks.get("session_count") == checks.get("request_count") == count > 0
        and report["request_count"] == count
        and all(not ids for name, ids in report["selection"].items() if name != period)
        and checks.get("overlapping_sessions") == checks.get("ag_intersection_count") == 0
        and checks.get("telemetry_completeness") == "unknown"
        and report["telemetry_complete"] is None,
        "Проверка данных не связана с полными таблицами и планом подготовки",
    )
    return checks


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    content = canonical_json_bytes(manifest)
    if path.exists():
        _require(path.read_bytes() == content, "Уже сохранён другой паспорт")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _utc(milliseconds: int) -> str:
    return (
        (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=milliseconds))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def assemble_admission(
    root: Path,
    run_dir: Path,
    *,
    run_id: str,
    input_dir: Path,
    prepared_dir: Path,
    ag_dir: Path,
    historical_dir: Path,
    test_run: Path,
    stage: str = "preflight",
    features_dir: Path | None = None,
    model_dir: Path | None = None,
    evaluation_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Проверить новый допуск; evaluation_plan_path — PreparationPlan C_smoke."""
    _require(stage in {"preflight", "post_training"}, "Неверная стадия допуска")
    if stage == "post_training":
        _require(
            all(path is not None for path in (features_dir, model_dir, evaluation_plan_path)),
            "Для post_training нужны признаки, модель и план C_smoke",
        )
    root, run_dir = root.resolve(), run_dir.resolve()
    raw_root = input_dir.resolve()
    _require(
        run_dir != raw_root and run_dir not in raw_root.parents and raw_root not in run_dir.parents,
        "Каталог допуска пересекается с исходными CSV",
    )
    paths = [run_dir, prepared_dir, ag_dir, historical_dir, test_run]
    paths.extend(path for path in (features_dir, model_dir, evaluation_plan_path) if path)
    _require(all(path.resolve().is_relative_to(root) for path in paths), "Выходы вне проекта")
    _require(
        all(
            run_dir != path.resolve()
            and run_dir not in path.resolve().parents
            and path.resolve() not in run_dir.parents
            for path in paths[1:]
        ),
        "Каталог допуска пересекается с входами",
    )
    assert_source_binding(root)
    inventory = source_inventory(root)
    source_bytes = canonical_json_bytes(inventory)
    code = sha256_bytes(source_bytes)
    evidence = read_test_evidence(
        root,
        test_run / "pytest.xml",
        strict_json_loads((test_run / "source-inventory.json").read_bytes()),
    )
    test_result = strict_json_loads((test_run / "result.json").read_bytes())
    _require(
        test_result.get("status") == "passed"
        and "error" not in test_result
        and type(test_result.get("exit_code")) is int
        and test_result["exit_code"] == 0
        and test_result.get("source_sha256") == evidence.summary["source_sha256"] == code
        and test_result.get("junit_sha256") == evidence.summary["junit_sha256"],
        "Нет завершённого успешного pytest текущих исходников",
    )
    prepared, ag = read_report(prepared_dir), read_report(ag_dir)
    checks = checked_preparation(prepared_dir, prepared, code, "B")
    ag_checks = checked_preparation(ag_dir, ag, code, "A_G")
    report_hash = sha256_bytes(canonical_json_bytes(prepared))
    ag_hash = sha256_bytes(canonical_json_bytes(ag))
    plan = from_json(prepared["plan_json"].encode(), PreparationPlan)
    ag_plan = from_json(ag["plan_json"].encode(), PreparationPlan)
    _require(
        not set(prepared["selection"]["B"]) & set(ag["selection"]["A_G"])
        and plan.scope == ag_plan.scope == "actual"
        and plan.task == ag_plan.task
        and plan.coverage == ag_plan.coverage
        and plan.seed == ag_plan.seed
        and plan.diagnostic_sessions == ag_plan.diagnostic_sessions == 0
        and [(p.name, p.start_ms, p.end_ms) for p in plan.periods]
        == [(p.name, p.start_ms, p.end_ms) for p in ag_plan.periods],
        "Несовместимые периоды или пересечение A_G/B",
    )
    historical_store = LocalArtifactStore(historical_dir)
    historical = load_historical(historical_store)
    head = historical_store.describe(PACKAGE_PATH)
    _require(
        head.scope == "actual"
        and historical.input_hashes["preparation_report"] == ag_hash
        and checks["bindings"].get("ag_report_sha256") == ag_hash
        and checks["bindings"].get("historical_package_sha256") == head.sha256
        and all(
            prepared["inputs"][kind] == ag["inputs"][kind] == head.inputs[kind]
            for kind in (*FILES, "profile")
        ),
        "Проверка B или исторический пакет относятся к другому A_G",
    )
    profile_bytes = prepared["profile_json"].encode()
    profile_data = strict_json_loads(profile_bytes)
    profile = parse_profile(profile_data, sha256=sha256_bytes(profile_bytes))
    manifest = init_manifest(run_id, scope="actual")
    manifest["stage"] = stage
    recipe = strict_json_loads((root / "configs/ranker_recipe.json").read_bytes())
    validate_recipe(recipe)
    manifest["ranker"]["resolved_parameters"] = recipe
    dataset = manifest["dataset"]
    dataset.update(
        version=profile_data["release"],
        schema_sha256=sha256_bytes(profile_bytes),
        vector_dimensions=dict(profile.dimensions),
        coverage_start_utc=_utc(plan.coverage.start_ms),
        coverage_end_utc=_utc(plan.coverage.end_ms),
        telemetry_gaps=[{"start_utc": _utc(a), "end_utc": _utc(b)} for a, b in plan.coverage.gaps],
        files=[{"name": name, "sha256": prepared["inputs"][kind]} for kind, name in FILES.items()],
    )
    dataset["session_sampling"]["session_count_by_period"] = {
        f"period_{p.name}": len(ag["selection"]["A_G"])
        if p.name == "A_G"
        else len(prepared["selection"]["B"])
        if p.name == "B"
        else 0
        for p in plan.periods
    }
    for period in plan.periods:
        manifest["periods"][f"period_{period.name}"] = {
            "start_utc": _utc(period.start_ms),
            "end_utc": _utc(period.end_ms),
        }
    store = LocalArtifactStore(run_dir)
    config_id = manifest["base_model"]["config_id"]

    def publish(name: str, content: bytes) -> str:
        return store.publish_bytes(
            "admission/" + name,
            content,
            code_sha256=code,
            run_id=run_id,
            scope="actual",
            config_id=config_id,
            reuse=True,
        ).sha256

    envelope = dict(run_id=run_id, scope="actual", config_id=config_id, evidence_scope="run")
    environment = strict_json_loads(canonical_json_bytes(environment_report(root)))
    schema_bytes = (root / "configs/feature_schema.json").read_bytes()
    repro = manifest["reproducibility"]
    defaults = {
        "library": "lightgbm",
        "version": environment["libraries"]["lightgbm"],
        "resolved_parameters": recipe,
    }
    for key, name, content in (
        ("code_commit_or_archive_sha256", "source-inventory.json", source_bytes),
        ("dependency_lockfile_sha256", "uv.lock", (root / "uv.lock").read_bytes()),
        ("feature_schema_sha256", "feature-schema.json", schema_bytes),
        (
            "manifest_schema_sha256",
            "manifest-schema.json",
            (root / "schemas/manifest.schema.json").read_bytes(),
        ),
        ("manifest_validator_sha256", "validator.json", validator_fingerprint_bytes()),
        ("ranker_callback_sha256", "objective.py", Path(objective.__file__).read_bytes()),
        ("resolved_library_defaults_sha256", "defaults.json", canonical_json_bytes(defaults)),
    ):
        repro[key] = publish(name, content)
    repro["platform_and_compiler"] = environment
    repro["random_generators_and_versions"] = {
        "seed": plan.seed,
        "numpy": environment["libraries"]["numpy"],
        "numpy_default_bit_generator": type(
            np.random.default_rng(plan.seed).bit_generator
        ).__name__,
        "hypothesis": environment["libraries"]["hypothesis"],
        "hypothesis_max_examples": 1000,
        "lightgbm": environment["libraries"]["lightgbm"],
        "lightgbm_configured_seeds": {k: v for k, v in recipe["model"].items() if "seed" in k},
    }
    publish("supply-profile.json", profile_bytes)
    publish("preparation-plan.json", prepared["plan_json"].encode())
    _require(
        publish("pytest.xml", (test_run / "pytest.xml").read_bytes())
        == evidence.summary["junit_sha256"],
        "JUnit изменился после проверки",
    )
    publish("pytest-result.json", canonical_json_bytes(test_result))
    publish("test-summary.json", canonical_json_bytes(evidence.summary))
    check_hashes = {
        "B": publish("data-check.json", canonical_json_bytes(checks)),
        "A_G": publish("ag-data-check.json", canonical_json_bytes(ag_checks)),
    }
    data = dict(
        **envelope,
        kind="data",
        **{key: checks[key] for key in DATA_RULES},
        overlapping_sessions=0,
        telemetry_completeness="unknown",
        input_size_bytes=checks["input_size_bytes"],
        input_sha256=[item["sha256"] for item in dataset["files"]],
        preparation_plan_sha256=prepared["inputs"]["plan"],
        preparation_report_sha256=report_hash,
        historical_package_sha256=head.sha256,
        independent_data_check_sha256=check_hashes["B"],
        independent_ag_check_sha256=check_hashes["A_G"],
        current_code_sha256=code,
    )
    if stage == "post_training":
        assert (
            model_dir is not None and features_dir is not None and evaluation_plan_path is not None
        )
        evaluation_bytes = evaluation_plan_path.read_bytes()
        evaluation_plan = from_json(evaluation_bytes, PreparationPlan)
        _require(
            [p.name for p in evaluation_plan.periods if p.sample_size] == ["C_smoke"]
            and all(
                getattr(evaluation_plan, name) == getattr(plan, name)
                for name in ("scope", "source_namespace", "task", "coverage", "seed")
            )
            and evaluation_plan.diagnostic_sessions == 0
            and [(p.name, p.start_ms, p.end_ms) for p in evaluation_plan.periods]
            == [(p.name, p.start_ms, p.end_ms) for p in plan.periods],
            "План C_smoke не соответствует закреплённым периодам",
        )
        feature_bytes = (features_dir / "reports/features.json").read_bytes()
        features = strict_json_loads(feature_bytes)
        model = load_ranker(model_dir)
        summary, identity = model.summary, model.summary["identity"]
        shared = dict(
            scope="actual",
            period="B",
            preparation_sha256=report_hash,
            historical_sha256=head.sha256,
            feature_schema_file_sha256=sha256_bytes(schema_bytes),
            source_sha256=code,
        )
        _require(
            features.get("status") == "completed"
            and all(
                features["identity"].get(k) == v and identity.get(k) == v for k, v in shared.items()
            )
            and identity["features_sha256"] == sha256_bytes(feature_bytes)
            and identity["recipe_sha256"] == sha256_file(root / "configs/ranker_recipe.json")
            and summary["recipe"] == recipe
            and features["feature_schema"] == strict_json_loads(schema_bytes)
            and features["requests"] == len(prepared["selection"]["B"])
            and canonical_json_bytes(features["identity"]["configuration"])
            == canonical_json_bytes(asdict(RetrievalConfig())),
            "Готовая модель не соответствует B, признакам и рецепту",
        )
        data["evaluation_preparation_plan_sha256"] = sha256_bytes(to_json(evaluation_plan))
        publish("evaluation-preparation-plan.json", evaluation_bytes)
        configuration_hash = publish(
            "model-configuration.json",
            canonical_json_bytes({"config_id": config_id, "ranker": manifest["ranker"]}),
        )
        model_hash = publish("ranker.txt", (model_dir / "models/ranker.txt").read_bytes())
        _require(model_hash == summary["model_sha256"], "Модель изменилась при загрузке")
        publish("ranker-report.json", canonical_json_bytes(summary))
        manifest["artifacts"]["trained_models_sha256"] = publish(
            "trained-models.json",
            canonical_json_bytes(
                dict(
                    **envelope,
                    kind="trained_models",
                    configurations=[
                        dict(
                            config_id=config_id,
                            configuration_sha256=configuration_hash,
                            model_sha256=model_hash,
                            feature_schema_sha256=sha256_bytes(schema_bytes),
                            actual_tree_count=summary["actual_tree_count"],
                            training_status="passed",
                        )
                    ],
                )
            ),
        )
        manifest["verification"]["math_status"] = "passed"
    manifest["evidence"]["data"] = publish("data-evidence.json", canonical_json_bytes(data))
    references = f"Независимые проверки данных A_G/B: {check_hashes}; модель проверена при загрузке, если применима"
    for current in ("preflight",) if stage == "preflight" else ("preflight", "post_training"):
        operation = "evaluate" if current == "post_training" else None
        result = dict(
            **envelope,
            kind="verification_results",
            stage=current,
            operation=operation,
            manifest_reference=run_id,
            code_version=code,
            configuration_or_reference=sha256_bytes(
                canonical_json_bytes(verification_configuration(manifest))
            ),
            random_seed=plan.seed,
            applicable_plan_reference=None,
            tests=stage_test_rows(manifest, current, evidence, references, operation=operation),
        )
        manifest["verification"]["results_by_stage"][current] = publish(
            current + ".json", canonical_json_bytes(result)
        )
    manifest["verification"]["data_status"] = "passed"
    raw_paths = {name: (input_dir / name).resolve() for name in FILES.values()}
    validation = validate_manifest(
        manifest,
        stage,
        store,
        input_paths=raw_paths,
        operation="evaluate" if stage == "post_training" else None,
    )
    _require(validation.gate_passed, f"Допуск {stage} не пройден: {validation.issues}")
    _require(source_inventory(root) == inventory, "Код изменился во время сборки допуска")
    for path, report, checked in ((prepared_dir, prepared, checks), (ag_dir, ag, ag_checks)):
        verify_tables(path, report)
        _require(
            sha256_file(path / REPORT) == checked["bindings"]["report_sha256"],
            "Изменён отчёт подготовки",
        )
        _require(
            sha256_file(path / "reports/data-check.json")
            == sha256_bytes(canonical_json_bytes(checked)),
            "Изменена проверка данных",
        )
    manifest_path = run_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return dict(
        manifest=manifest_path.relative_to(root).as_posix(),
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        run_dir=run_dir.relative_to(root).as_posix(),
        input_paths={name: str(path) for name, path in raw_paths.items()},
    )
