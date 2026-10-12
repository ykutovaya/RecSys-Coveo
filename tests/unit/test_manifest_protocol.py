"""Применимость проверок и независимость исторической оценки от сервиса."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.config import init_manifest, required_checks, validate_manifest
from coveo_recsys.config.schema import structure_issues, validate_stage_report
from tests.reference.manifest_cases import make_manifest_case
from tests.unit.test_manifest_adversarial import publish


def test_template_describes_first_request_registered_purchase_protocol() -> None:
    manifest = init_manifest("observed")
    assert manifest["manifest_version"] == "2.10"
    assert manifest["evaluation_protocol"] == "observed_purchases_v1"
    assert manifest["task"]["max_requests_per_session"] == 1
    assert (
        manifest["task"]["estimand"] == "first_request_hit_at_20_among_recorded_purchase_sessions"
    )
    assert manifest["task"]["target_rule"] == (
        "distinct_registered_purchase_skus_in_same_session_after_first_request"
        "_until_min_horizon_last_recorded_event"
    )
    assert manifest["task"]["evaluation_cohort"] == "sessions_with_nonempty_registered_target"
    assert manifest["task"]["empty_target_outcome"] == "unknown"
    assert manifest["task"]["reserved_periods"] == ["E"]
    assert manifest["dataset"]["telemetry_completeness"] == "unknown"
    assert manifest["calibration"] == {
        "enabled": False,
        "reason": "not_used_in_observed_purchases_v1",
    }
    assert manifest["verification"]["calibration_status"] == "not_used"
    assert manifest["base_model"]["historical_purchase_feature"] == {
        "enabled": False,
        "reason": "telemetry_completeness_unknown",
    }
    assert manifest["inference"]["precision_unit"] == "fraction"
    assert manifest["inference"]["half_width_fraction"] is None
    assert manifest["inference"]["single_policy_half_width_fraction"] is None
    assert set(manifest["inference"]["dependence_assumptions"]) == {
        "paired_hit",
        "single_policy_hit",
    }
    assert not structure_issues(manifest)


@pytest.mark.parametrize(
    "path,value",
    [
        ("manifest_version", "2.8"),
        ("evaluation_protocol", "complete_observation"),
        ("task.max_requests_per_session", 10),
        ("task.estimand", "buying_sessions_per_1000_eligible_sessions"),
        ("task.target_rule", "last_registered_purchase"),
        ("task.evaluation_cohort", "all_sessions"),
        ("task.empty_target_outcome", "negative"),
        ("task.reserved_periods", []),
        ("dataset.telemetry_completeness", "complete"),
        ("calibration.enabled", True),
        ("verification.calibration_status", "passed"),
        ("base_model.historical_purchase_feature.enabled", True),
        ("inference.precision_unit", "per_1000"),
    ],
)
def test_schema_rejects_a_changed_observed_protocol(path: str, value: object) -> None:
    manifest = init_manifest("observed")
    node = manifest
    *parents, field = path.split(".")
    for part in parents:
        node = node[part]
    node[field] = value
    assert structure_issues(manifest)


@pytest.mark.parametrize(
    "path",
    [
        "freeze.plan_references.before_E",
        "base_model.purchase_frequency_smoothing",
        "calibration.resolved_parameters",
        "inference.half_width_per_1000",
        "inference.single_policy_half_width_per_1000",
        "inference.dependence_assumptions.paired_por",
        "inference.dependence_assumptions.single_policy_por",
        "artifacts.calibrator_sha256",
        "evidence.calibration_plan",
        "evidence.calibration",
    ],
)
def test_schema_rejects_obsolete_protocol_fields(path: str) -> None:
    manifest = init_manifest("observed")
    node = manifest
    *parents, field = path.split(".")
    for part in parents:
        node = node[part]
    assert field not in node
    node[field] = None
    assert structure_issues(manifest)


@pytest.mark.parametrize("field", ["half_width_fraction", "single_policy_half_width_fraction"])
@pytest.mark.parametrize(
    "value,valid", [(None, True), (0, False), (0.01, True), (1, True), (1.01, False)]
)
def test_precision_threshold_is_a_positive_fraction(
    field: str, value: float | None, valid: bool
) -> None:
    manifest = init_manifest("observed")
    manifest["inference"][field] = value
    assert (not structure_issues(manifest)) is valid


@pytest.mark.parametrize("scope", ["actual", "synthetic"])
def test_preflight_allows_zero_session_counts_in_closed_periods(tmp_path: Path, scope: str) -> None:
    manifest, _ = make_manifest_case(LocalArtifactStore(tmp_path), "preflight")
    manifest["scope"] = scope
    counts = manifest["dataset"]["session_sampling"]["session_count_by_period"]
    counts.update({name: 1 if name in {"period_A_G", "period_B"} else 0 for name in counts})
    assert not structure_issues(manifest)


@pytest.mark.parametrize(
    "period,count",
    [(name, -1) for name in ("A_G", "A_N", "B", "C_smoke", "C_select", "E", "D")]
    + [("A_G", 0), ("B", 0)],
)
def test_session_counts_reject_negative_sizes_and_empty_training_inputs(
    tmp_path: Path, period: str, count: int
) -> None:
    manifest, _ = make_manifest_case(LocalArtifactStore(tmp_path), "preflight")
    manifest["dataset"]["session_sampling"]["session_count_by_period"][f"period_{period}"] = count
    issues = structure_issues(manifest)
    assert any(issue.path == "dataset.session_sampling.session_count_by_period" for issue in issues)


@pytest.mark.parametrize("config_id", ["gcs_rk_v2_7", "research_variant"])
@pytest.mark.parametrize("penalty", [0, 1])
def test_probability_dependent_diversity_is_disabled_for_every_recipe(
    config_id: str, penalty: int
) -> None:
    manifest = init_manifest("observed", scope="research")
    manifest["base_model"].update(config_id=config_id, diversity_penalty=penalty)
    assert (not structure_issues(manifest)) is (penalty == 0)


@pytest.mark.parametrize(
    "stage", ["preflight", "post_training", "pre_selection", "pre_D", "post_D"]
)
def test_local_stages_do_not_require_service_checks(stage: str) -> None:
    identifiers = {row["test_id"] for row in required_checks(init_manifest("local"), stage)}
    assert not identifiers & {"B03.service", "B14", "B15", "B16", "B17", "B18"}


def test_first_training_does_not_require_future_metadata_or_intervals() -> None:
    manifest = init_manifest("first")
    identifiers = {row["test_id"] for row in required_checks(manifest, "post_training")}
    assert {"B10.data", "B10.ranker", "B11"} <= identifiers
    assert not identifiers & {"B10.calibration", "B10.metadata", "B12", "B18"}
    evaluation = {
        row["test_id"] for row in required_checks(manifest, "post_training", operation="evaluate")
    }
    assert "B12" in evaluation


@pytest.mark.parametrize("stage", ["preflight", "post_training", "pre_selection"])
@pytest.mark.parametrize("estimate", ["single_policy", "paired"])
def test_precision_planning_can_wait_until_pre_d(tmp_path: Path, stage: str, estimate: str) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    manifest["inference"].update(
        primary_estimate_type=estimate,
        single_policy_half_width_fraction=None,
        half_width_fraction=None,
        planned_outcome_variance=None,
        dependence_multiplier=None,
        pilot_provenance=None,
        dependence_assumptions={"single_policy_hit": None, "paired_hit": None},
    )
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert result.gate_passed, result.issues


@pytest.mark.parametrize("estimate", ["single_policy", "paired"])
@pytest.mark.parametrize(
    "field",
    [
        "half_width_fraction",
        "planned_outcome_variance",
        "dependence_assumptions",
        "dependence_multiplier",
        "pilot_provenance",
    ],
)
def test_precision_planning_is_required_by_pre_d(tmp_path: Path, estimate: str, field: str) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_D")
    manifest["inference"]["primary_estimate_type"] = estimate
    path = f"inference.{field}"
    if field == "dependence_assumptions":
        path += f".{estimate}_hit"
    elif field == "half_width_fraction" and estimate == "single_policy":
        path = "inference.single_policy_half_width_fraction"
    node = manifest
    *parents, name = path.split(".")
    for parent in parents:
        node = node[parent]
    node[name] = None
    result = validate_manifest(manifest, "pre_D", store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.path == path and issue.code == "required" for issue in result.issues)


@pytest.mark.parametrize("stage", ["preflight", "post_training"])
@pytest.mark.parametrize("field", ["primary_estimate_type", "primary_policy_config_id"])
def test_precision_delay_does_not_delay_estimate_identity(
    tmp_path: Path, stage: str, field: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    manifest["inference"][field] = None
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.path == f"inference.{field}" for issue in result.issues)


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


def test_final_evaluation_does_not_require_calibration(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_D")
    assert manifest["verification"]["calibration_status"] == "not_used"
    assert "before_E" not in manifest["freeze"]["plan_references"]
    assert "calibration" not in manifest["evidence"]
    checks = {row["test_id"]: row for row in required_checks(manifest, "pre_D")}
    assert "B10.calibration" not in checks
    assert checks["B11"]["required_by_stage"] == "post_training"
    assert "pre_D" not in checks["B11"]["repeat_at"]
    result = validate_manifest(manifest, "pre_D", store, plan_references=plans)
    assert result.gate_passed, result.issues


def test_calibration_is_not_an_available_operation() -> None:
    with pytest.raises(ValueError):
        required_checks(init_manifest("observed"), "pre_selection", operation="calibrate")


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


def test_stage_report_rejects_obsolete_calibration_operation(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, _ = make_manifest_case(store, "pre_selection")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["pre_selection"])
    )
    report["operation"] = "calibrate"
    assert any(
        issue.path == "verification.report.operation" for issue in validate_stage_report(report)
    )
