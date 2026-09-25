"""Применимость проверок и независимость исторической оценки от сервиса."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.config import init_manifest, required_checks, validate_manifest
from tests.reference.manifest_cases import make_manifest_case
from tests.unit.test_manifest_adversarial import publish


@pytest.mark.parametrize(
    "stage", ["preflight", "post_training", "pre_selection", "pre_D", "post_D"]
)
def test_local_stages_do_not_require_service_checks(stage: str) -> None:
    identifiers = {row["test_id"] for row in required_checks(init_manifest("local"), stage)}
    assert not identifiers & {"B03.service", "B14", "B15", "B16", "B17", "B18"}


def test_first_training_does_not_require_future_calibration_metadata_or_intervals() -> None:
    manifest = init_manifest("first")
    identifiers = {row["test_id"] for row in required_checks(manifest, "post_training")}
    assert {"B10.data", "B10.ranker"} <= identifiers
    assert not identifiers & {"B10.calibration", "B10.metadata", "B11", "B12", "B18"}
    calibration = {
        row["test_id"] for row in required_checks(manifest, "pre_selection", operation="calibrate")
    }
    assert {"B10.calibration", "B11"} <= calibration
    evaluation = {
        row["test_id"] for row in required_checks(manifest, "post_training", operation="evaluate")
    }
    assert "B12" in evaluation


@pytest.mark.parametrize("stage", ["pre_selection", "pre_D", "post_D"])
def test_local_evaluation_completes_without_load_or_snapshot_chain(
    tmp_path: Path, stage: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    assert manifest["verification"]["serve_status"] == "not_run"
    assert manifest["verification"]["results_by_stage"]["post_load"] is None
    assert "load" not in manifest["evidence"]
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert result.gate_passed, result.issues


def test_pinned_plan_cannot_be_replaced_with_edited_manifest(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_D")
    changed = deepcopy(manifest)
    changed["freeze"]["plan_references"]["before_D"] = "0" * 64
    result = validate_manifest(changed, "post_D", store, plan_references=plans)
    assert any(issue.code == "plan_reference" for issue in result.issues)


def test_first_training_allows_unresolved_calibrator(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_training")
    manifest["calibration"]["resolved_parameters"] = None
    result = validate_manifest(manifest, "post_training", store, plan_references=plans)
    assert result.gate_passed, result.issues


def test_calibration_checks_and_selected_plan_are_required_before_E(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_selection", operation="calibrate")
    result = validate_manifest(
        manifest, "pre_selection", store, plan_references=plans, operation="calibrate"
    )
    assert result.gate_passed, result.issues
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["pre_selection"])
    )
    report["tests"] = [row for row in report["tests"] if row["test_id"] != "B11"]
    manifest["verification"]["results_by_stage"]["pre_selection"] = publish(store, manifest, report)
    result = validate_manifest(
        manifest, "pre_selection", store, plan_references=plans, operation="calibrate"
    )
    assert not result.gate_passed
    assert any(issue.path == "B11" and issue.code == "not_run" for issue in result.issues)


@pytest.mark.parametrize(
    "counts",
    [
        dict(collected=0, passed=0, failed=0, skipped=0, errors=0),
        dict(collected=1, passed=0, failed=0, skipped=1, errors=0),
        dict(collected=1, passed=0, failed=1, skipped=0, errors=0),
        dict(collected=2, passed=1, failed=0, skipped=0, errors=0),
    ],
)
def test_zero_skipped_failed_or_inconsistent_execution_cannot_pass(
    tmp_path: Path, counts: dict
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["preflight"])
    )
    report["tests"][0]["counts"] = counts
    manifest["verification"]["results_by_stage"]["preflight"] = publish(store, manifest, report)
    result = validate_manifest(manifest, "preflight", store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.code == "test_counts" for issue in result.issues)


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_version", "0" * 64),
        ("configuration_or_reference", "0" * 64),
        ("random_seed", 18),
        ("manifest_reference", "other-run"),
    ],
)
def test_report_must_match_executed_code_configuration_and_seed(
    tmp_path: Path, field: str, value
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["preflight"])
    )
    report[field] = value
    manifest["verification"]["results_by_stage"]["preflight"] = publish(store, manifest, report)
    result = validate_manifest(manifest, "preflight", store, plan_references=plans)
    assert any(issue.code == "report_configuration" for issue in result.issues)


def test_calibration_report_must_describe_calibration_operation(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_selection", operation="calibrate")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["pre_selection"])
    )
    report["operation"] = None
    manifest["verification"]["results_by_stage"]["pre_selection"] = publish(store, manifest, report)
    result = validate_manifest(
        manifest, "pre_selection", store, plan_references=plans, operation="calibrate"
    )
    assert any(issue.code == "report_operation" for issue in result.issues)
