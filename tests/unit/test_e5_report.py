"""План задаётся до целей; Markdown отображает сохранённые численные результаты."""

from copy import deepcopy

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.evaluation.reporting import (
    _aggregate_sensitivity,
    render_report,
    validate_evaluation_plan,
)
from tests.integration.test_e5_reporting import evaluation_plan, preparation_plan


@pytest.mark.parametrize(
    "path,value",
    [
        (("period",), "D"),
        (("k",), True),
        (("k",), 10),
        (("diagnostic_depth",), False),
        (("extra",), 1),
        (("policies",), ["GCS+Rk", "P"]),
        (("intervals", "grid_step_ms"), 3),
        (("intervals", "block_cells"), True),
        (("intervals", "sensitivity_block_cells"), [1, 2]),
        (("intervals", "mc_error_method"), "invented"),
        (("intervals", "dependence", "P", "supported"), 1),
        (("intervals", "dependence", "P", "assumptions"), [""]),
        (("intervals", "half_width_fraction", "P"), True),
        (("intervals", "half_width_fraction", "P"), float("nan")),
    ],
)
def test_invalid_evaluation_plan_is_rejected_before_any_outcomes(path, value):
    plan = evaluation_plan()
    target = plan
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(InputValidationError):
        validate_evaluation_plan(plan, preparation_plan())


def test_pilot_plan_accepts_unknown_precision_and_dependence_without_filling_them():
    plan = evaluation_plan()
    before = deepcopy(plan)
    validate_evaluation_plan(plan, preparation_plan())
    assert plan == before


def test_render_preserves_saved_values_reasons_and_all_blocks_without_recalculation():
    summary = {
        "period": "C_smoke",
        "evaluation_protocol": "observed_purchases_v1",
        "replay_mode": "ideal_deterministic",
        "run_id": "consumer",
        "inputs": {"preparation_plan": "a" * 64, "model": "b" * 64},
        "policies": {
            "GCS+Rk": {
                "metrics": {"Hit@20": {"value": 0.123456789, "denominator": 7}},
                "sources": {"g": {"delta_fixed": {"value": 0.25}}},
                "slices": {"cart=present": {"N": 4}},
                "cost": {"elapsed_ms_total": 123},
            }
        },
        "intervals": {
            "GCS+Rk": {
                "metric": {
                    "value": 0.123456789,
                    "status": "insufficient",
                    "reason": "insufficient_nonempty_blocks",
                    "interval": None,
                }
            }
        },
        "multiplicity_sensitivity": {"measured_sessions": 5},
        "S_offline": "not_run",
        "business_effect": "not_observed",
    }
    before = deepcopy(summary)
    rendered = render_report(summary)
    for text in (
        "C_smoke",
        "observed_purchases_v1",
        "ideal_deterministic",
        "consumer",
        "0.123456789",
        "insufficient_nonempty_blocks",
        "delta_fixed",
        "cart=present",
        "123",
        "measured_sessions",
        "not_observed",
        "a" * 64,
        "b" * 64,
    ):
        assert text in rendered
    assert summary == before


@pytest.mark.parametrize("missing_target", [False, True])
def test_sensitivity_consumes_generator_once_and_preserves_exact_counts(missing_target):
    measured = dict(
        status="measured",
        target_known=True,
        set_schema_sha256="a" * 64,
        multiset_schema_sha256="b" * 64,
        affected=True,
        prefix_changed=True,
        candidate_changed=False,
        common_candidates=3,
        set_candidate_hit=1,
        multiset_candidate_hit=0,
        changed_features=[dict(name="x", changed_rows=2, mask_changed_rows=1, max_abs_delta=0.5)],
    )
    consumed = []

    def rows():
        consumed.append("measured")
        yield measured
        consumed.append("unknown")
        yield dict(
            measured, target_known=False, set_candidate_hit=None, multiset_candidate_hit=None
        )
        consumed.append("missing")
        yield dict(status="not_measured", reason="fallback", target_known=missing_target)

    result = _aggregate_sensitivity(rows())
    assert consumed == ["measured", "unknown", "missing"]
    reason = "incomplete_sensitivity_measurement" if missing_target else None
    assert result == dict(
        scope="prefix_sensitivity_fixed_historical",
        historical_catalog_changed=False,
        model_retrained=False,
        multiset_model_applied=False,
        sessions=3,
        measured_sessions=2,
        target_sessions=2 if missing_target else 1,
        measured_target_sessions=1,
        missing_reasons={"fallback": 1},
        affected_sessions=2,
        prefix_changed_sessions=2,
        candidate_changed_sessions=0,
        common_candidate_rows=6,
        schema_pairs=[["a" * 64, "b" * 64]],
        changed_features={"x": dict(changed_rows=4, mask_changed_rows=2, max_abs_delta=0.5)},
        set_candidate_coverage=dict(
            value=None if missing_target else 1.0,
            denominator=2 if missing_target else 1,
            reason=reason,
        ),
        multiset_candidate_coverage=dict(
            value=None if missing_target else 0.0,
            denominator=2 if missing_target else 1,
            reason=reason,
        ),
    )
