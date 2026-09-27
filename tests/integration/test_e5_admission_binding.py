"""Настоящий валидатор связывает новый потребитель со старой моделью и планом C.

Все свидетельства искусственные и помечены как тестовые. Сырые события C_smoke
не создаются и не читаются; проверяется состав допуска, а не качество модели.
"""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.cli.admission import require_evaluation_preparation, require_replay_admission
from coveo_recsys.config import verification_configuration
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.serialization import from_json, to_json
from tests.integration.test_e5_reporting import evaluation_plan
from tests.reference.manifest_cases import make_manifest_case
from tests.unit.test_e4_admission import admitted  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bound_smoke(request, tmp_path):
    admission, original, manifest, save, publish, data = request.getfixturevalue("admitted")
    store = LocalArtifactStore(Path(admission["run_dir"]))
    original_plan = from_json(original["prepared"]["plan_json"].encode(), PreparationPlan)
    plan = replace(
        original_plan,
        periods=tuple(
            replace(p, sample_size=3 if p.name == "C_smoke" else 0) for p in original_plan.periods
        ),
    )
    data["evaluation_preparation_plan_sha256"] = sha256_bytes(to_json(plan))
    manifest["evidence"]["data"] = publish(canonical_json_bytes(data))
    model = {
        "status": "completed",
        "actual_tree_count": 1,
        "model_sha256": publish(b"TEST_ONLY_NOT_A_RANKER"),
        "recipe": deepcopy(original["recipe"]),
        "identity": {
            "scope": "actual",
            "period": "B",
            "source_sha256": "3" * 64,
            "historical_sha256": original["historical"].sha256,
            "preparation_sha256": sha256_bytes(canonical_json_bytes(original["prepared"])),
            "feature_schema_file_sha256": sha256_bytes(original["feature_schema_bytes"]),
        },
    }
    envelope = {
        "run_id": manifest["run_id"],
        "scope": "actual",
        "config_id": manifest["base_model"]["config_id"],
        "evidence_scope": "run",
    }
    configuration = {"config_id": envelope["config_id"], "ranker": manifest["ranker"]}
    manifest["artifacts"]["trained_models_sha256"] = publish(
        canonical_json_bytes(
            {
                **envelope,
                "kind": "trained_models",
                "configurations": [
                    {
                        "config_id": envelope["config_id"],
                        "configuration_sha256": publish(canonical_json_bytes(configuration)),
                        "model_sha256": model["model_sha256"],
                        "feature_schema_sha256": model["identity"]["feature_schema_file_sha256"],
                        "actual_tree_count": 1,
                        "training_status": "passed",
                    }
                ],
            }
        )
    )
    template_store = LocalArtifactStore(tmp_path / "stage-template")
    template, _ = make_manifest_case(template_store, "post_training", operation="evaluate")
    stage_report = json.loads(
        template_store.read_by_sha256(template["verification"]["results_by_stage"]["post_training"])
    )
    stage_report.update(
        **envelope,
        code_version=manifest["reproducibility"]["code_commit_or_archive_sha256"],
        configuration_or_reference=sha256_bytes(
            canonical_json_bytes(verification_configuration(manifest))
        ),
    )
    manifest["stage"] = "post_training"
    manifest["verification"]["results_by_stage"]["post_training"] = publish(
        canonical_json_bytes(stage_report)
    )
    save()
    kwargs = {
        "training_prepared": original["prepared"],
        "historical": original["historical"],
        "feature_schema_bytes": original["feature_schema_bytes"],
        "recipe": original["recipe"],
        "retrieval": original["retrieval"],
        "model_summary": model,
        "plan": plan,
    }
    prepared = {
        "status": "completed",
        "scope": "actual",
        "telemetry_complete": None,
        "plan_json": to_json(plan).decode(),
        "inputs": {**original["prepared"]["inputs"], "plan": sha256_bytes(to_json(plan))},
        "selection": {
            p.name: ["smoke-session"] if p.name == "C_smoke" else [] for p in plan.periods
        },
    }
    return admission, kwargs, prepared, manifest, stage_report, save, publish, store


def test_actual_gate_composes_current_checks_and_old_producer_without_c_tables(bound_smoke):
    admission, kwargs, _, manifest, *_ = bound_smoke
    assert (
        kwargs["model_summary"]["identity"]["source_sha256"]
        != manifest["reproducibility"]["code_commit_or_archive_sha256"]
    )
    assert (
        require_replay_admission(ROOT, admission, **kwargs, evaluate=True)
        == admission["manifest_sha256"]
    )


@pytest.mark.parametrize("damage", ["old_check_code", "failed_B07", "missing_B12", "model", "B"])
def test_real_validator_rejects_broken_post_training_or_model_binding(bound_smoke, damage):
    admission, kwargs, _, manifest, report, save, publish, _ = bound_smoke
    if damage == "old_check_code":
        report["code_version"] = "3" * 64
    elif damage == "failed_B07":
        item = next(row for row in report["tests"] if row["test_id"] == "B07")
        item["status"] = "failed"
        item["counts"].update(passed=0, failed=1)
    elif damage == "missing_B12":
        report["tests"] = [row for row in report["tests"] if row["test_id"] != "B12"]
    elif damage == "model":
        kwargs["model_summary"]["model_sha256"] = "4" * 64
    else:
        kwargs["model_summary"]["identity"]["preparation_sha256"] = "4" * 64
    if damage in {"old_check_code", "failed_B07", "missing_B12"}:
        manifest["verification"]["results_by_stage"]["post_training"] = publish(
            canonical_json_bytes(report)
        )
    save()
    with pytest.raises(InputValidationError):
        require_replay_admission(ROOT, admission, **kwargs, evaluate=True)


def test_completed_smoke_accepts_exact_pretty_plan_bytes_with_matching_digest(bound_smoke):
    admission, kwargs, prepared, *_ = bound_smoke
    content = (
        json.dumps(json.loads(to_json(kwargs["plan"])), ensure_ascii=False, indent=2) + "\n"
    ).encode()
    prepared["plan_json"] = content.decode()
    prepared["inputs"]["plan"] = sha256_bytes(content)
    assert (
        require_replay_admission(ROOT, admission, **kwargs, prepared=prepared)
        == admission["manifest_sha256"]
    )


def test_completed_smoke_rejects_plan_bytes_with_another_digest(bound_smoke):
    admission, kwargs, prepared, *_ = bound_smoke
    prepared["inputs"]["plan"] = "7" * 64
    with pytest.raises(InputValidationError):
        require_replay_admission(ROOT, admission, **kwargs, prepared=prepared)


@pytest.fixture
def preparation_settings(bound_smoke, tmp_path, monkeypatch):
    """Заменены лишь загрузчики фиктивных B/модели, оба допуска и планы настоящие."""
    from coveo_recsys.data import prepared as prepared_module
    from coveo_recsys.historical.load import PACKAGE_PATH
    from coveo_recsys.ranking import load as load_module

    admission, kwargs, _, _, _, _, _, store = bound_smoke
    historical = kwargs["historical"]
    store.publish_bytes(
        PACKAGE_PATH,
        store.read_bytes(historical.path),
        inputs=historical.inputs,
        configuration=historical.configuration,
        code_sha256=historical.code_sha256,
        run_id=historical.run_id,
        scope=historical.scope,
        config_id=historical.config_id,
    )
    feature = {
        "status": "completed",
        "identity": {
            "period": "B",
            "preparation_sha256": kwargs["model_summary"]["identity"]["preparation_sha256"],
            "historical_sha256": historical.sha256,
            "feature_schema_file_sha256": sha256_bytes(kwargs["feature_schema_bytes"]),
            "configuration": kwargs["retrieval"],
        },
    }
    feature_path = tmp_path / "features/reports/features.json"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(canonical_json_bytes(feature))
    kwargs["model_summary"]["identity"]["features_sha256"] = sha256_bytes(feature_path.read_bytes())
    model_path = tmp_path / "model/reports/ranker.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(canonical_json_bytes(kwargs["model_summary"]))
    training_path = tmp_path / "training/reports/preparation.json"
    training_path.parent.mkdir(parents=True)
    training_path.write_bytes(canonical_json_bytes(kwargs["training_prepared"]))
    plan_path, schema_path, recipe_path = (
        tmp_path / name for name in ("smoke.json", "schema.json", "recipe.json")
    )
    plan_path.write_bytes(to_json(kwargs["plan"]))
    schema_path.write_bytes(kwargs["feature_schema_bytes"])
    recipe_path.write_bytes(canonical_json_bytes(kwargs["recipe"]))
    evaluation = evaluation_plan()
    period = next(p for p in kwargs["plan"].periods if p.name == "C_smoke")
    width = (
        period.end_ms
        - period.start_ms
        - kwargs["plan"].task.horizon_ms
        - kwargs["plan"].task.gap_ms
    )
    assert width % 10 == 0
    evaluation["intervals"]["grid_step_ms"] = width // 10
    settings = {
        "preparation_plan": str(plan_path),
        "preparation": None,
        "training_preparation": {
            "path": str(training_path.parents[1]),
            "report_sha256": sha256_bytes(training_path.read_bytes()),
        },
        "training_features": {
            "path": str(feature_path.parents[1]),
            "report_sha256": sha256_bytes(feature_path.read_bytes()),
        },
        "historical": {"path": str(store.root), "package_sha256": historical.sha256},
        "model": {
            "path": str(model_path.parents[1]),
            "report_sha256": sha256_bytes(model_path.read_bytes()),
        },
        "feature_schema": str(schema_path),
        "recipe": str(recipe_path),
        "retrieval": kwargs["retrieval"],
        "evaluation_plan": evaluation,
        "admission": admission,
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_bytes(canonical_json_bytes(settings))
    loaded = []

    def saved_report(directory, report_sha256, *, verify_tables):
        assert verify_tables is False
        content = (directory / "reports/preparation.json").read_bytes()
        assert sha256_bytes(content) == report_sha256
        return json.loads(content)

    def saved_model(directory, *, expected_identity):
        content = json.loads((directory / "reports/ranker.json").read_bytes())
        assert content["identity"] == expected_identity
        loaded.append(directory)
        return SimpleNamespace(summary=content)

    monkeypatch.setattr(prepared_module, "read_report", saved_report)
    monkeypatch.setattr(load_module, "load_ranker", saved_model)
    return (
        settings_path,
        kwargs["plan"],
        kwargs["training_prepared"]["inputs"]["profile"],
        admission,
        loaded,
    )


@pytest.mark.parametrize("changed_profile", [False, True])
def test_preparation_binds_actual_input_profile_before_loading_model(
    preparation_settings, changed_profile
):
    path, plan, profile_digest, admission, loaded = preparation_settings
    if changed_profile:
        with pytest.raises(InputValidationError):
            require_evaluation_preparation(ROOT, path, plan, profile_sha256="9" * 64)
        assert loaded == []
    else:
        assert (
            require_evaluation_preparation(ROOT, path, plan, profile_sha256=profile_digest)
            == admission["manifest_sha256"]
        )
        assert len(loaded) == 1
