from copy import deepcopy
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.config import init_manifest, validate_manifest
from tests.reference.manifest_cases import STAGES, make_manifest_case


@pytest.mark.parametrize("stage", STAGES)
def test_all_stages_have_independent_positive_and_missing_result_cases(
    tmp_path: Path, stage: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert result.gate_passed, result.issues
    changed = deepcopy(manifest)
    changed["verification"]["results_by_stage"]["preflight"] = None
    failure = validate_manifest(changed, stage, store, plan_references=plans)
    assert not failure.gate_passed and not failure.results_complete


def test_template_and_changed_stage_policy_cannot_pass(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    draft = init_manifest("empty")
    assert not validate_manifest(draft, "preflight", store).gate_passed
    manifest, plans = make_manifest_case(store, "preflight")
    manifest["stage_validation"]["primary_estimate_required_fields"]["single_policy"] = []
    assert not validate_manifest(manifest, "preflight", store, plan_references=plans).semantic_valid


def test_legacy_test_registry_is_not_accepted(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    manifest["verification"]["test_registry"] = []
    assert not validate_manifest(manifest, "preflight", store, plan_references=plans).gate_passed


def test_future_load_unneeded_and_unfit_calibration_does_not_block_ranking(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_D")
    assert manifest["artifacts"]["calibrator_sha256"] is None
    assert validate_manifest(manifest, "pre_D", store, plan_references=plans).gate_passed
    manifest["verification"]["results_by_stage"]["pre_D"] = None
    assert not validate_manifest(manifest, "pre_D", store, plan_references=plans).results_complete


def test_external_plan_reference_and_immutable_evaluation_inputs(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_D")
    assert not validate_manifest(manifest, "post_D", store).gate_passed
    manifest["inference"]["single_policy_half_width_per_1000"] = 999
    result = validate_manifest(manifest, "post_D", store, plan_references=plans)
    assert any(issue.code == "evaluation_plan_changed" for issue in result.issues)


def test_checksum_profile_does_not_admit_pilot(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_AB")
    manifest["operations"]["artifact_authentication"] = "local_checksum_only"
    result = validate_manifest(manifest, "pre_AB", store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.code == "signature_required" for issue in result.issues)
