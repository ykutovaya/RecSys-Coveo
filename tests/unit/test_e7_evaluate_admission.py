"""D закрыт до проверки плана; отсутствие подбора не отменяет pre_D."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, strict_json_loads
from coveo_recsys.config import validate_manifest
from coveo_recsys.config.manifest import plan_inputs, verification_configuration
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.access import DatasetAccess
from tests.reference.manifest_cases import make_manifest_case


def fixed_manifest(store):
    manifest, plans = make_manifest_case(store, "pre_D", operation="evaluate")
    evaluation = strict_json_loads(store.read_by_sha256(plans["before_D"]))
    manifest["freeze"]["plan_references"]["before_C_select"] = None
    manifest["evidence"]["selection_plan"] = None
    for name in ("selected_policies_sha256", "selection_report_sha256"):
        manifest["artifacts"][name] = None
    evaluation["inputs"] = plan_inputs(manifest, "before_D")
    digest = store.publish_bytes(
        "fixed-plan.json",
        canonical_json_bytes(evaluation),
        code_sha256=manifest["reproducibility"]["code_commit_or_archive_sha256"],
        run_id=manifest["run_id"],
        scope=manifest["scope"],
        config_id=manifest["base_model"]["config_id"],
    ).sha256
    manifest["freeze"]["plan_references"]["before_D"] = digest
    manifest["evidence"]["evaluation_plan"] = digest
    for stage, reference in (("pre_selection", None), ("pre_D", digest)):
        report = strict_json_loads(
            store.read_by_sha256(manifest["verification"]["results_by_stage"][stage])
        )
        report["applicable_plan_reference"] = reference
        manifest["verification"]["results_by_stage"][stage] = store.publish_bytes(
            f"fixed-{stage}.json",
            canonical_json_bytes(report),
            code_sha256=manifest["reproducibility"]["code_commit_or_archive_sha256"],
            run_id=manifest["run_id"],
            scope=manifest["scope"],
            config_id=manifest["base_model"]["config_id"],
        ).sha256
    return manifest, {"before_D": digest}


def test_fixed_single_policy_needs_no_selection_but_still_needs_frozen_plan(tmp_path):
    store = LocalArtifactStore(tmp_path / "evidence")
    manifest, plans = fixed_manifest(store)
    result = validate_manifest(
        manifest, "pre_D", store, plan_references=plans, operation="evaluate"
    )
    assert result.gate_passed, result.issues
    assert not validate_manifest(manifest, "pre_D", store, operation="evaluate").gate_passed
    changed = deepcopy(manifest)
    changed["inference"]["block_cells"] += 1
    assert not validate_manifest(
        changed, "pre_D", store, plan_references=plans, operation="evaluate"
    ).gate_passed


@pytest.mark.parametrize("observed", [0, 1, 60000])
def test_actual_d_count_is_a_result_not_a_change_of_frozen_inputs(tmp_path, observed):
    manifest, _ = fixed_manifest(LocalArtifactStore(tmp_path / "evidence"))
    original = deepcopy(manifest)
    changed = deepcopy(manifest)
    changed["dataset"]["session_sampling"]["session_count_by_period"]["period_D"] = observed
    assert plan_inputs(changed, "before_D") == plan_inputs(manifest, "before_D")
    assert verification_configuration(changed) == verification_configuration(manifest)
    assert manifest == original
    changed["dataset"]["session_sampling"]["session_count_by_period"]["period_B"] += 1
    assert plan_inputs(changed, "before_D") != plan_inputs(manifest, "before_D")


@pytest.mark.parametrize(
    "purpose,filename,kind",
    [("prefix", "prefixes.parquet", PreparedRequest), ("labels", "targets.parquet", TargetRecord)],
)
def test_d_access_requires_admission_before_report_or_table_io(
    tmp_path, monkeypatch, purpose, filename, kind
):
    import coveo_recsys.data.access as module

    def forbidden(*args, **kwargs):
        pytest.fail("D прочитан до допуска")

    monkeypatch.setattr(module, "read_report", forbidden)
    calls = []

    def denied():
        calls.append("pre_D")
        raise InputValidationError("D закрыт")

    reader = DatasetAccess(tmp_path, code_sha256="0" * 64, run_id="test", d_admission=denied)
    with pytest.raises(InputValidationError, match="D закрыт"):
        reader.read(f"datasets/D/{filename}", kind, purpose=purpose)
    assert calls == ["pre_D"]


@pytest.mark.parametrize("configuration", [{"preparation": {"path": "D-never-open"}}, [], None])
def test_evaluate_refuses_missing_admission_before_touching_preparation(
    tmp_path, monkeypatch, configuration
):
    import coveo_recsys.cli.replay as module
    from coveo_recsys.cli.replay import run_replay

    settings = tmp_path / "evaluate.json"
    settings.write_bytes(canonical_json_bytes(configuration))

    def forbidden(*args, **kwargs):
        pytest.fail("Подготовка D прочитана без допуска")

    monkeypatch.setattr(module, "read_preparation_report", forbidden)
    with pytest.raises(InputValidationError, match="допуск|pre_D"):
        run_replay(tmp_path, settings, tmp_path / "result", "test", period="D")
    assert not (tmp_path / "result").exists()


def test_d_preparation_requires_pre_d_even_for_small_test_inputs(tmp_path):
    from coveo_recsys.cli.preparation import Preparation

    root = Path(__file__).resolve().parents[2]
    original = from_json(
        (root / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan
    )
    plan = replace(
        original,
        periods=tuple(replace(p, sample_size=int(p.name == "D")) for p in original.periods),
    )
    path = tmp_path / "plan.json"
    path.write_bytes(to_json(plan))
    settings = tmp_path / "evaluation.json"
    settings.write_bytes(canonical_json_bytes({"evaluation_plan": {"period": "D"}}))
    runner = Preparation(LocalArtifactStore(tmp_path / "result"), root, "denied")
    # Исходников намеренно нет: отказ допуска обязан предшествовать их проверке.
    with pytest.raises(InputValidationError, match="pre_D"):
        runner.prepare(
            tmp_path / "absent-input",
            root / "configs/synthetic_csv.json",
            path,
            batch_size=2,
            max_batch_chars=256,
            evaluation_config=settings,
        )
