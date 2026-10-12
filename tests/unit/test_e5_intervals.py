"""Э5-04: один расчёт календарных интервалов для охвата и парной разности."""

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.evaluation.intervals import (
    _bootstrap_repetitions,
    _monte_carlo_error,
    calendar_intervals,
    nearest_rank,
)
from tests.reference.e5_intervals_oracle import mc_ranks, repetitions


def plan(cells=40, **changes):
    return dict(
        start_ms=100,
        end_ms=100 + cells * 10 + 3,
        horizon_ms=2,
        gap_ms=1,
        grid_step_ms=10,
        block_cells=2,
        sensitivity_block_cells=(1, 3),
        dependence_assumptions=("synthetic stationary process",),
        dependence_supported=True,
        half_width_fraction=1.0,
        **changes,
    )


def informative(hits=100, hit_blocks=20, full_blocks=20):
    times, values = [], []
    for index in range(hits):
        block = (
            index if index < hit_blocks else (index % hit_blocks if index < 4 * hit_blocks else 0)
        )
        times.append(100 + block * 20)
        values.append(1)
    for block in range(full_blocks):
        times.extend([110 + block * 20] * 6)
        values.extend([0] * 6)
    return times, values


def test_dense_grid_keeps_empty_cells_and_excludes_calendar_tail():
    result, details = calendar_intervals([100, 109, 120, 149], [1, 0, 1, 0], **plan(5))
    assert details["grid"]["cells"] == 5
    assert details["grid"]["end_ms"] == 150
    assert [cell["n"] for cell in details["cells"]] == [2, 0, 1, 0, 1]
    assert [cell["value"] for cell in details["cells"]] == [0.5, None, 1, None, 0]
    assert result.value == 0.5 and result.denominator == 4


@pytest.mark.parametrize("time,valid", [(99, False), (100, True), (149, True), (150, False)])
def test_admissible_start_window_boundaries(time, valid):
    if valid:
        assert calendar_intervals([time], [1], **plan(5))[0].value == 1
    else:
        with pytest.raises(InputValidationError):
            calendar_intervals([time], [1], **plan(5))


@pytest.mark.parametrize(
    "changes",
    [
        {"end_ms": 154},
        {"grid_step_ms": 0},
        {"block_cells": 0},
        {"block_cells": 6},
        {"sensitivity_block_cells": (1, 1)},
        {"sensitivity_block_cells": (1, 2)},
        {"half_width_fraction": 0},
        {"half_width_fraction": float("nan")},
    ],
)
def test_invalid_plan_is_an_error_not_statistical_insufficiency(changes):
    options = plan(5)
    options.update(changes)
    with pytest.raises(InputValidationError):
        calendar_intervals([100], [1], **options)


@pytest.mark.parametrize(
    "times,hits,other",
    [
        ([100], [], None),
        ([100], [2], None),
        ([100], [None], None),
        ([100.0], [1], None),
        ([100], [1], []),
        ([100], [1], [-1]),
    ],
)
def test_invalid_session_outcomes_are_errors(times, hits, other):
    with pytest.raises(InputValidationError):
        calendar_intervals(times, hits, other_hit=other, **plan(5))


def test_bootstrap_matches_literal_indices_with_unequal_counts_and_truncated_last_block():
    counts, sums = [2, 1, 1, 1, 1], [0, 0, -1, 1, 1]
    expected = repetitions(sums, counts, 2)
    actual = _bootstrap_repetitions(sums, counts, 2)
    np.testing.assert_array_equal(actual, expected)
    result, details = calendar_intervals(
        [100, 100, 110, 120, 130, 140],
        [1, 0, 1, 0, 1, 1],
        other_hit=[0, 1, 1, 1, 0, 0],
        **plan(5),
    )
    assert result.value == 1 / 6
    assert result.interval == (nearest_rank(expected, 0.025), nearest_rank(expected, 0.975))
    assert details["estimate_type"] == "paired"
    assert [item["block_cells"] for item in details["sensitivity"]] == [1, 3]
    assert details["plan"]["bootstrap_repetitions"] == 2000
    assert details["plan"]["bootstrap_seed"] == 17


@pytest.mark.parametrize(
    "probability,expected", [(0.01, 1), (0.25, 1), (0.5, 2), (0.75, 3), (1, 4)]
)
def test_nearest_rank_without_interpolation(probability, expected):
    assert nearest_rank([4, 1, 3, 2], probability) == expected


@pytest.mark.parametrize(
    "hits,blocks,qualified", [(99, 19, False), (99, 20, False), (100, 19, False), (100, 20, True)]
)
def test_category_information_thresholds(hits, blocks, qualified):
    result, details = calendar_intervals(*informative(hits, blocks), **plan())
    information = details["primary"]["information"]
    assert information["category_counts"] == {"0": 120, "1": hits}
    assert information["category_blocks"] == {"0": 20, "1": blocks}
    assert ("insufficient_outcome_categories" not in details["primary"]["reasons"]) == qualified
    assert result.status == ("passed" if qualified else "insufficient")


@pytest.mark.parametrize("full_blocks", [19, 20])
def test_full_nonoverlapping_blocks_threshold(full_blocks):
    result, details = calendar_intervals(
        *informative(100, full_blocks, full_blocks), **plan(2 * full_blocks)
    )
    assert details["primary"]["information"]["full_blocks"] == full_blocks
    assert ("insufficient_full_blocks" in details["primary"]["reasons"]) == (full_blocks < 20)
    assert result.status == ("passed" if full_blocks == 20 else "insufficient")


def test_empty_and_partial_blocks_do_not_create_information():
    times, hits = informative(99, 19, 19)
    times += [500] * 20
    hits += [1] * 20
    _, details = calendar_intervals(times, hits, **plan(41))
    info = details["primary"]["information"]
    assert info["full_blocks"] == 20 and info["nonempty_blocks"] == 19
    assert info["category_counts"]["1"] == 119
    assert info["category_blocks"]["1"] == 19


@pytest.mark.parametrize("categories", [(-1, 0), (-1, 1), (0, 1)])
def test_pair_requires_any_two_supported_categories(categories):
    times, pattern = informative()
    differences = [categories[value] for value in pattern]
    first, second = (
        [int(value == 1) for value in differences],
        [int(value == -1) for value in differences],
    )
    result, details = calendar_intervals(times, first, other_hit=second, **plan())
    assert result.status == "passed"
    assert details["primary"]["information"]["supported_categories"] == 2


def test_undefined_repetition_invalidates_whole_interval_without_dropping_it():
    result, details = calendar_intervals([100], [1], **plan(40))
    assert result.value == 1 and result.interval is None and not result.interval_valid
    assert details["primary"]["information"]["undefined_repetitions"] > 0
    assert "undefined_bootstrap_repetitions" in details["primary"]["reasons"]
    assert details["primary"]["mc_error"] is None


def test_no_target_sessions_preserves_dense_grid_and_unknown_point():
    result, details = calendar_intervals([], [], **plan())
    assert result.value is None and result.denominator == 0 and result.status == "insufficient"
    assert len(details["cells"]) == 40
    assert details["primary"]["information"]["undefined_repetitions"] == 2000


@pytest.mark.parametrize("value", [0, 1])
def test_zero_variation_is_not_empirical_precision(value):
    result, details = calendar_intervals(list(range(100, 500)) * 2, [value] * 800, **plan())
    assert result.value == value and result.interval == (value, value)
    assert result.status == "insufficient" and not result.interval_valid
    assert {"zero_empirical_variance", "degenerate_interval"} <= set(details["primary"]["reasons"])


def test_equal_interval_bounds_are_insufficient_even_with_both_outcomes():
    times = [100 + 10 * cell for cell in range(40) for _ in range(5)]
    result, details = calendar_intervals(
        times, [cell % 2 for cell in range(40) for _ in range(5)], **plan()
    )
    assert details["primary"]["information"]["supported_categories"] == 2
    assert result.interval == (0.5, 0.5) and not result.interval_valid
    assert "degenerate_interval" in details["primary"]["reasons"]


@pytest.mark.parametrize("supported", [None, False, True])
def test_dependence_is_confirmed_for_each_evaluated_functional(supported):
    options = plan()
    options["dependence_supported"] = supported
    result, details = calendar_intervals(*informative(), **options)
    assert result.interval is not None
    assert result.interval_valid is (supported is True)
    assert details["primary"]["information"]["dependence_assumptions_satisfied"] is supported


def test_pilot_without_planned_precision_retains_interval_but_cannot_pass():
    options = plan()
    options["half_width_fraction"] = None
    result, details = calendar_intervals(*informative(), **options)
    assert result.value is not None and result.interval_valid
    assert result.status == "insufficient" and result.reason == "precision_not_planned"
    assert details["primary"]["precision_status"] == "not_planned"


@pytest.mark.parametrize("offset,passed", [(-1e-6, False), (0, True), (1e-6, True)])
def test_planned_halfwidth_boundary(offset, passed):
    _, baseline = calendar_intervals(*informative(), **plan())
    options = plan()
    options["half_width_fraction"] = baseline["primary"]["half_width"] + offset
    result, _ = calendar_intervals(*informative(), **options)
    assert result.interval_valid
    assert result.status == ("passed" if passed else "insufficient")


def test_mc_error_uses_independent_binomial_order_ranks():
    values = np.linspace(-0.5, 0.7, 2000)
    result = _monte_carlo_error(values)
    assert result["method"] == "binomial_order_statistic_95pct"
    for key, p in (("lower", 0.025), ("upper", 0.975)):
        lower, upper = mc_ranks(2000, p)
        assert result[key]["ranks"] == [lower, upper]
        assert result[key]["interval"] == [values[lower - 1], values[upper - 1]]
        bound = nearest_rank(values, p)
        assert result[key]["max_absolute_error"] == max(
            bound - values[lower - 1], values[upper - 1] - bound
        )


def test_random_stream_is_repeatable_and_does_not_depend_on_global_rng():
    first = calendar_intervals(*informative(), **plan())
    np.random.seed(9876)
    np.random.random(30)
    second = calendar_intervals(*informative(), **plan())
    assert first == second
