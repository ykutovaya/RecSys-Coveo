"""C_smoke открывается после обучения, но до появления его результатов."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def replay_admission(tmp_path, monkeypatch):
    from coveo_recsys.cli import admission as module

    base = from_json((ROOT / "configs/coveo_preparation.json").read_bytes(), PreparationPlan)
    plan = replace(
        base,
        periods=tuple(
            replace(p, sample_size=10 if p.name == "C_smoke" else 0) for p in base.periods
        ),
    )
    schema_bytes = (ROOT / "configs/feature_schema.json").read_bytes()
    recipe = json.loads((ROOT / "configs/ranker_recipe.json").read_bytes())
    model = {
        "status": "completed",
        "actual_tree_count": 200,
        "model_sha256": "a" * 64,
        "recipe": recipe,
        "identity": {
            "scope": "actual",
            "period": "B",
            "source_sha256": "old-producer",
            "feature_schema_file_sha256": sha256_bytes(schema_bytes),
            "historical_sha256": "b" * 64,
            "preparation_sha256": "c" * 64,
        },
    }
    store = LocalArtifactStore(tmp_path / "admission")

    def publish(value):
        data = canonical_json_bytes(value)
        return store.publish_bytes(
            f"evidence/{sha256_bytes(data)}",
            data,
            code_sha256="1" * 64,
            run_id="test",
            scope="actual",
            config_id="gcs_rk_v2_7",
        ).sha256

    index = {
        "configurations": [
            {
                "config_id": "gcs_rk_v2_7",
                "model_sha256": model["model_sha256"],
                "feature_schema_sha256": sha256_bytes(schema_bytes),
                "actual_tree_count": 200,
                "training_status": "passed",
            }
        ]
    }
    manifest = {
        "base_model": {"config_id": "gcs_rk_v2_7"},
        "artifacts": {"trained_models_sha256": publish(index)},
        "evidence": {
            "data": publish({"evaluation_preparation_plan_sha256": sha256_bytes(to_json(plan))})
        },
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    admission = {
        "manifest": str(path),
        "manifest_sha256": sha256_bytes(path.read_bytes()),
        "run_dir": str(store.root),
        "input_paths": {},
    }
    prepared = {
        "scope": "actual",
        "plan_json": to_json(base).decode(),
        "inputs": {"profile": "p", "browsing": "x"},
    }
    model["identity"]["preparation_sha256"] = sha256_bytes(canonical_json_bytes(prepared))
    calls = []

    def checked_training(*args, **kwargs):
        result = module.validate_manifest(
            manifest, kwargs["stage"], store, operation=kwargs["operation"]
        )
        if not result.gate_passed:
            raise InputValidationError("Проверки не пройдены")
        return admission["manifest_sha256"]

    monkeypatch.setattr(module, "require_training_admission", checked_training)

    def validate(*args, **kwargs):
        calls.append((args[1], kwargs.get("operation")))
        return SimpleNamespace(gate_passed=True, issues=())

    monkeypatch.setattr(module, "validate_manifest", validate)
    kwargs = dict(
        training_prepared=prepared,
        historical=SimpleNamespace(scope="actual", sha256="b" * 64),
        feature_schema_bytes=schema_bytes,
        recipe=recipe,
        retrieval=asdict(RetrievalConfig()),
        model_summary=model,
        plan=plan,
    )
    return module, admission, kwargs, calls


def test_smoke_gate_precedes_preparation_and_uses_frozen_model(replay_admission):
    module, admission, kwargs, calls = replay_admission
    assert (
        module.require_replay_admission(ROOT, admission, **kwargs, evaluate=True)
        == admission["manifest_sha256"]
    )
    assert calls == [("post_training", "evaluate")]
    assert kwargs["model_summary"]["identity"]["source_sha256"] == "old-producer"


@pytest.mark.parametrize("period", ["A_G", "B", "A_N", "C_select", "E", "D"])
def test_smoke_gate_rejects_other_active_periods_before_validation(replay_admission, period):
    module, admission, kwargs, calls = replay_admission
    kwargs["plan"] = replace(
        kwargs["plan"],
        periods=tuple(
            replace(p, sample_size=10 if p.name == period else 0) for p in kwargs["plan"].periods
        ),
    )
    with pytest.raises(InputValidationError):
        module.require_replay_admission(ROOT, admission, **kwargs)
    assert calls == []


@pytest.mark.parametrize("damage", ["plan", "model", "schema", "stage"])
def test_smoke_gate_rejects_changed_binding(replay_admission, damage, monkeypatch):
    module, admission, kwargs, _ = replay_admission
    if damage == "plan":
        kwargs["plan"] = replace(kwargs["plan"], max_seconds=kwargs["plan"].max_seconds + 1)
    elif damage == "model":
        kwargs["model_summary"]["model_sha256"] = "d" * 64
    elif damage == "schema":
        kwargs["model_summary"]["identity"]["feature_schema_file_sha256"] = "e" * 64
    else:
        monkeypatch.setattr(
            module,
            "validate_manifest",
            lambda *a, **k: SimpleNamespace(gate_passed=False, issues=()),
        )
    with pytest.raises(InputValidationError):
        module.require_replay_admission(ROOT, admission, **kwargs)


@pytest.mark.parametrize("damage", [None, "defaults", "model_hash", "retrieval", "plan"])
def test_preparation_admission_checks_ready_model_without_requiring_c_tables(
    replay_admission, tmp_path, monkeypatch, damage
):
    from coveo_recsys.data import prepared as prepared_module
    from coveo_recsys.ranking import load as load_module

    module, _, kwargs, _ = replay_admission
    historical = LocalArtifactStore(tmp_path / "historical")
    head = historical.publish_bytes(
        "historical/package.json",
        b"{}",
        code_sha256="1" * 64,
        run_id="h",
        scope="actual",
        config_id="base",
    )
    model = kwargs["model_summary"]
    model["identity"]["historical_sha256"] = head.sha256
    feature = {
        "status": "completed",
        "identity": {
            "period": "B",
            "preparation_sha256": model["identity"]["preparation_sha256"],
            "historical_sha256": head.sha256,
            "feature_schema_file_sha256": sha256_bytes(kwargs["feature_schema_bytes"]),
            "configuration": kwargs["retrieval"],
        },
    }
    feature_path = tmp_path / "features/reports/features.json"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(canonical_json_bytes(feature))
    model["identity"]["features_sha256"] = sha256_bytes(feature_path.read_bytes())
    model_path = tmp_path / "model/reports/ranker.json"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(canonical_json_bytes(model))
    plan_path = tmp_path / "c_plan.json"
    plan_path.write_bytes(to_json(kwargs["plan"]))
    schema_path = tmp_path / "schema.json"
    schema_path.write_bytes(kwargs["feature_schema_bytes"])
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(canonical_json_bytes(kwargs["recipe"]))
    settings = {
        "preparation_plan": str(plan_path),
        "preparation": None,
        "training_preparation": {
            "path": str(tmp_path / "B"),
            "report_sha256": model["identity"]["preparation_sha256"],
        },
        "training_features": {
            "path": str(feature_path.parents[1]),
            "report_sha256": model["identity"]["features_sha256"],
        },
        "historical": {"path": str(historical.root), "package_sha256": head.sha256},
        "model": {
            "path": str(model_path.parents[1]),
            "report_sha256": sha256_bytes(model_path.read_bytes()),
        },
        "feature_schema": str(schema_path),
        "recipe": str(recipe_path),
        "retrieval": dict(kwargs["retrieval"]),
        "evaluation_plan": {"period": "C_smoke"},
    }
    if damage == "defaults":
        settings["retrieval"] = {}
    elif damage == "model_hash":
        settings["model"]["report_sha256"] = "f" * 64
    elif damage == "retrieval":
        settings["retrieval"]["max_candidates"] += 1
    elif damage == "plan":
        plan_path.write_bytes(to_json(replace(kwargs["plan"], max_seconds=1)))
    path = tmp_path / "settings.json"
    path.write_bytes(canonical_json_bytes(settings))
    checked = []
    monkeypatch.setattr(prepared_module, "read_report", lambda *a, **k: kwargs["training_prepared"])
    monkeypatch.setattr(
        load_module,
        "load_ranker",
        lambda *a, **k: checked.append(k) or SimpleNamespace(summary=model),
    )
    monkeypatch.setattr(module, "require_replay_admission", lambda *a, **k: "a" * 64)
    # Разбор плана оценок проверяется отдельно; здесь проверяем привязки допуска.
    import coveo_recsys.evaluation.reporting as reporting

    monkeypatch.setattr(reporting, "validate_evaluation_plan", lambda *_: None)
    if damage not in (None, "defaults"):
        with pytest.raises(InputValidationError):
            module.require_evaluation_preparation(ROOT, path, kwargs["plan"], profile_sha256="p")
    else:
        assert (
            module.require_evaluation_preparation(ROOT, path, kwargs["plan"], profile_sha256="p")
            == "a" * 64
        )
        assert checked == [{"expected_identity": model["identity"]}]
