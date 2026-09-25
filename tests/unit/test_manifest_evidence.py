"""Искусственные отчёты проверяют договор, а не качество модели или эффект A/B."""

from copy import deepcopy

import pytest

from coveo_recsys.config.evidence import (
    GUARDRAILS,
    validate_final_evaluation,
    validate_online_plan,
    validate_online_results,
)


def example() -> tuple[dict, dict, dict]:
    manifest = {
        "evidence": {"evaluation_plan": "a" * 64},
        "task": {"horizon_seconds": 86400, "inactivity_seconds": 1800},
        "inference": {
            "primary_estimate_type": "single_policy",
            "planned_cells": 1200,
            "block_cells": 60,
            "bootstrap_repetitions": 2000,
            "primary_interval": "calendar_moving_block_bootstrap",
            "alpha": 0.05,
            "single_policy_half_width_per_1000": 10,
        },
        "verification": {"offline_status": "passed", "business_status": "passed"},
        "online_experiment": {
            "protocol_sha256": "b" * 64,
            "assignment_unit": "session",
            "estimand": "session_conversion",
            "business_threshold": 0,
            "sample_size_and_horizon": {
                "n_per_arm": 1000,
                "start_utc": "2020-04-01T00:00:00Z",
                "end_utc": "2020-04-10T00:00:00Z",
                "analysis_not_before_utc": "2020-04-12T00:00:00Z",
                "alpha": 0.05,
                "power": 0.8,
                "delta_plan": 50,
                "baseline_probability": 0.1,
                "interval_method": "clopper_pearson_bonferroni",
                "assumptions_justification": "synthetic",
                "sample_size_justification": "synthetic",
                "missing_outcome_policy": "require_complete",
            },
            "srm_plan": {
                "assignment_mechanism": "independent_bernoulli",
                "cohort_membership_rule": "eligible before assignment",
                "conditional_null_justification": "synthetic fixed membership",
                "test_method": "exact_two_sided_binomial",
                "familywise_error_control_justification": "Bonferroni L(G+1)",
                "schedule_utc": ["2020-04-05T00:00:00Z"],
                "number_of_checks": 1,
                "alpha": 0.001,
                "expected_treatment_probability": 0.5,
                "strata": [],
            },
            "guardrail_margins": {
                name: {
                    "limit": 10 if name == "latency_ms" else 0.01,
                    "comparison": "le",
                    "unit": "ms" if name == "latency_ms" else "fraction",
                    "window_seconds": 60,
                    "denominator": "assigned_units",
                    "minimum_count": 100,
                    "responsible_action": "pause",
                    "schedule_utc": ["2020-04-05T00:00:00Z"],
                }
                for name in sorted(GUARDRAILS)
            },
        },
    }
    offline = {
        "evaluation_plan_sha256": "a" * 64,
        "metrics": {"N": 1000, "POR": 200.0},
        "intervals": {"POR": [190.0, 210.0]},
        "maturity": {"complete": True},
        "metadata_sensitivity": {"evaluated": True},
        "statuses": {"offline_status": "passed"},
        "contract_violations": [],
        "interval_details": {
            "valid": True,
            "reason": None,
            "method": "calendar_moving_block_bootstrap",
            "confidence_level": 0.95,
        },
        "information": {
            "full_blocks": 20,
            "nonempty_blocks": 20,
            "category_counts": {"0": 800, "1": 200},
            "category_blocks": {"0": 20, "1": 20},
            "undefined_repetitions": 0,
            "dependence_assumptions_satisfied": True,
        },
    }
    online = {
        "protocol_sha256": "b" * 64,
        "assignment_unit": "session",
        "estimand": "session_conversion",
        "assigned_counts": {"control": 1000, "treatment": 1000},
        "outcomes": {"control": 100, "treatment": 140},
        "unknown_outcomes": {"control": 0, "treatment": 0},
        "mature": True,
        "integrity_passed": True,
        "itt_complete": True,
        "assumptions_satisfied": True,
        "srm_passed": True,
        "stopping_status": "completed",
        "analysis_mode": "confirmatory",
        "interval": {
            "lower": 10.0,
            "upper": 70.0,
            "valid": True,
            "reason": None,
            "method": "clopper_pearson_bonferroni",
            "confidence_level": 0.95,
        },
        "business_status": "passed",
        "reason": None,
        "guardrail_results": {
            name: {
                "value": 0.0,
                "count": 1000,
                "status": "passed",
                "reason": None,
                **{key: rule[key] for key in ("unit", "limit", "comparison")},
            }
            for name, rule in manifest["online_experiment"]["guardrail_margins"].items()
        },
    }
    return manifest, offline, online


def assert_gate_only(issues: tuple) -> None:
    assert issues
    assert {issue.layer for issue in issues} == {"gate"}, issues


def test_positive_contracts() -> None:
    manifest, offline, online = example()
    assert not validate_final_evaluation(offline, manifest)
    assert not validate_online_plan(manifest)
    assert not validate_online_results(online, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["metrics"].update(N=0),
        lambda r: r["intervals"].update(POR=[200.0, 200.0]),
        lambda r: r.update(statuses=["passed"]),
        lambda r: r["metrics"].update(POR=900.0),
        lambda r: r.update(evaluation_plan_sha256="c" * 64),
        lambda r: r["information"].update(nonempty_blocks=10),
        lambda r: r["information"].update(undefined_repetitions=1),
        lambda r: r["information"].update(dependence_assumptions_satisfied=False),
        lambda r: r["interval_details"].update(confidence_level=0.1),
        lambda r: r["metrics"].update(N=True),
        lambda r: r["metrics"].update(POR=10**400),
    ],
)
def test_false_offline_passed_rejected(mutation) -> None:
    manifest, offline, _ = example()
    mutation(offline)
    assert validate_final_evaluation(offline, manifest)


def test_empty_offline_result_is_retained_insufficient() -> None:
    manifest, offline, _ = example()
    manifest["verification"]["offline_status"] = "insufficient"
    offline["metrics"] = {"N": 0, "POR": None}
    offline["intervals"]["POR"] = None
    offline["interval_details"].update(valid=False, reason="no mature sessions")
    offline["information"].update(
        nonempty_blocks=0, category_counts={"0": 0, "1": 0}, category_blocks={"0": 0, "1": 0}
    )
    offline["statuses"]["offline_status"] = "insufficient"
    offline["reason"] = "no mature sessions"
    assert_gate_only(validate_final_evaluation(offline, manifest))


@pytest.mark.parametrize("condition", ["degenerate", "width", "rare", "violation"])
def test_negative_offline_results_are_not_structural_errors(condition: str) -> None:
    manifest, offline, _ = example()
    status = "failed" if condition == "violation" else "insufficient"
    manifest["verification"]["offline_status"] = status
    offline["statuses"]["offline_status"] = status
    offline["reason"] = condition
    if condition == "degenerate":
        offline["intervals"]["POR"] = [200.0, 200.0]
        offline["interval_details"].update(valid=False, reason=condition)
    elif condition == "width":
        offline["intervals"]["POR"] = [180.0, 220.0]
    elif condition == "rare":
        offline["metrics"]["POR"] = 1.0
        offline["information"].update(
            category_counts={"0": 999, "1": 1}, category_blocks={"0": 20, "1": 1}
        )
        offline["intervals"]["POR"] = [0.0, 4.0]
        offline["interval_details"].update(valid=False, reason=condition)
    else:
        offline["contract_violations"] = ["changed source"]
    assert_gate_only(validate_final_evaluation(offline, manifest))


def test_paired_categories_and_signed_estimate() -> None:
    manifest, offline, _ = example()
    manifest["inference"].update(primary_estimate_type="paired", half_width_per_1000=10)
    offline["metrics"] = {"N": 1000, "DeltaPOR": -100.0}
    offline["intervals"] = {"DeltaPOR": [-110.0, -90.0]}
    offline["information"].update(
        category_counts={"-1": 200, "0": 700, "1": 100},
        category_blocks={"-1": 20, "0": 20, "1": 20},
    )
    assert not validate_final_evaluation(offline, manifest)


def test_offline_denominator_cannot_exceed_selected_sessions() -> None:
    manifest, offline, _ = example()
    manifest["dataset"] = {"session_sampling": {"session_count_by_period": {"period_D": 10}}}
    assert any(x.code == "sample_count" for x in validate_final_evaluation(offline, manifest))


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "business_threshold", -1),
        ("srm_plan", "number_of_checks", 0),
        ("srm_plan", "number_of_checks", 2),
        ("srm_plan", "alpha", 0.5),
        ("srm_plan", "expected_treatment_probability", 0),
        ("srm_plan", "schedule_utc", []),
        ("srm_plan", "strata", [{"stratum_id": "future_clicks"}]),
        ("srm_plan", "membership_uses_assignment_or_exposure_history", True),
        ("sample_size_and_horizon", "alpha", 0),
        ("sample_size_and_horizon", "delta_plan", 0),
        ("sample_size_and_horizon", "n_per_arm", True),
        ("sample_size_and_horizon", "missing_outcome_policy", []),
        ("sample_size_and_horizon", "analysis_not_before_utc", "2020-04-10T00:00:00Z"),
    ],
)
def test_invalid_online_plans(section, key, value) -> None:
    manifest, _, _ = example()
    target = (
        manifest["online_experiment"] if section is None else manifest["online_experiment"][section]
    )
    target[key] = value
    assert validate_online_plan(manifest)


def test_null_srm_values_and_valid_stratum() -> None:
    manifest, _, _ = example()
    broken = deepcopy(manifest)
    broken["online_experiment"]["srm_plan"] = dict.fromkeys(broken["online_experiment"]["srm_plan"])
    assert validate_online_plan(broken)
    manifest["online_experiment"]["srm_plan"]["strata"] = [
        {
            "stratum_id": "region",
            "pre_assignment_fields": ["region"],
            "membership_rule_version": "1",
            "membership_rule_sha256": "d" * 64,
            "conditional_null_justification": "fixed assignment-time region",
            "missing_value_policy": "separate stratum",
            "expected_treatment_probability": 0.5,
        }
    ]
    assert not validate_online_plan(manifest)


def test_unhashable_guard_comparison_and_status_are_rejected() -> None:
    manifest, offline, online = example()
    manifest["online_experiment"]["guardrail_margins"]["latency_ms"]["comparison"] = []
    assert validate_online_plan(manifest)
    offline["statuses"]["offline_status"] = {}
    assert validate_final_evaluation(offline, manifest)
    online["business_status"] = {}
    assert validate_online_results(online, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(protocol_sha256="c" * 64),
        lambda r: r["outcomes"].update(control=1001),
        lambda r: r["assigned_counts"].update(control=True),
        lambda r: r["interval"].update(lower=80.0),
        lambda r: r["interval"].update(lower=float("nan")),
        lambda r: r["interval"].update(method="chosen_after_outcomes"),
        lambda r: r.update(stopping_status="safety_stop"),
        lambda r: r.update(guardrail_results={}),
        lambda r: r["guardrail_results"]["latency_ms"].update(limit=999),
        lambda r: r["guardrail_results"]["latency_ms"].update(value=999),
        lambda r: r.update(analysis_mode=[]),
    ],
)
def test_false_online_passed_rejected(mutation) -> None:
    manifest, _, online = example()
    mutation(online)
    assert validate_online_results(online, manifest)


@pytest.mark.parametrize(
    "bounds,status",
    [([-20.0, 0.0], "failed"), ([0.0, 20.0], "insufficient"), ([-10.0, 10.0], "insufficient")],
)
def test_business_decision_boundary_and_retained_negative(bounds, status) -> None:
    manifest, _, online = example()
    online["interval"].update(lower=bounds[0], upper=bounds[1])
    online.update(business_status=status, reason="threshold not established")
    manifest["verification"]["business_status"] = status
    assert_gate_only(validate_online_results(online, manifest))


@pytest.mark.parametrize("condition", ["stop", "unknown", "guard", "empty"])
def test_incomplete_or_failed_online_outcomes_remain_writable(condition: str) -> None:
    manifest, _, online = example()
    status = "failed" if condition == "guard" else "insufficient"
    online.update(business_status=status, reason=condition)
    manifest["verification"]["business_status"] = status
    if condition == "stop":
        online.update(stopping_status="safety_stop", analysis_mode="descriptive")
    elif condition == "unknown":
        online["unknown_outcomes"]["control"] = 1
    elif condition == "guard":
        online["guardrail_results"]["latency_ms"].update(value=20, status="failed")
    else:
        online["assigned_counts"]["control"] = online["outcomes"]["control"] = 0
        online["interval"].update(valid=False, lower=None, upper=None, reason=condition)
    assert_gate_only(validate_online_results(online, manifest))


def test_missing_required_section_is_not_negative_measurement() -> None:
    manifest, offline, online = example()
    del offline["information"]
    assert any(x.layer == "results" for x in validate_final_evaluation(offline, manifest))
    del online["interval"]
    assert any(x.layer == "results" for x in validate_online_results(online, manifest))


def test_explicit_unknown_online_interval_is_a_retained_result() -> None:
    manifest, _, online = example()
    manifest["verification"]["business_status"] = "insufficient"
    online.update(interval=None, business_status="insufficient", reason="not enough information")
    assert_gate_only(validate_online_results(online, manifest))


def test_registered_missing_bounds_use_full_assignment_denominators() -> None:
    manifest, _, online = example()
    size = manifest["online_experiment"]["sample_size_and_horizon"]
    size.update(
        missing_outcome_policy="registered_bounds",
        missing_outcome_justification="synthetic bound analysis",
    )
    online["unknown_outcomes"].update(control=10, treatment=20)
    online["missing_outcome_bounds"] = {"lower": 30.0, "upper": 60.0}
    assert not validate_online_plan(manifest)
    assert not validate_online_results(online, manifest)
    online["missing_outcome_bounds"]["lower"] = 100.0
    assert any(x.code == "missing_bounds" for x in validate_online_results(online, manifest))
