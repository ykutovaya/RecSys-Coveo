"""Граничные случаи B05: ранги, насыщение и численные производные."""

import math

import numpy as np
import pytest

from tests.reference.e4_objective_oracle import direct_derivatives, frozen_pairs, pair_loss


@pytest.mark.parametrize("difference", [-5.0, -0.1, 0.0, 0.1, 5.0])
def test_derivatives_match_central_differences_with_frozen_pair_weights(difference):
    from coveo_recsys.ranking.objective import session_lambdas

    scores = np.array([difference, 0.0])
    labels, groups, skus = (1, 0), (2,), ("я", "a")
    pairs = frozen_pairs(scores, labels, groups, skus)
    actual_g, actual_h = session_lambdas(scores, labels, groups, skus, dtype=np.float64)
    step = 1e-4
    loss = pair_loss(scores, pairs)
    for index in range(len(scores)):
        before, after = scores.copy(), scores.copy()
        before[index] -= step
        after[index] += step
        lower, upper = pair_loss(before, pairs), pair_loss(after, pairs)
        expected_g = (upper - lower) / (2 * step)
        expected_h = (upper - 2 * loss + lower) / step**2
        assert abs(actual_g[index] - expected_g) <= 1e-7
        assert abs(actual_h[index] - expected_h) <= 1e-6


@pytest.mark.parametrize("difference", [-1e4, -100.0, 0.0, 100.0, 1e4])
def test_saturation_preserves_small_curvature_and_does_not_floor_zero(difference):
    from coveo_recsys.ranking.objective import session_lambdas

    scores = (difference, 0.0)
    labels, groups, skus = (1, 0), (2,), ("a", "b")
    actual_g, actual_h = session_lambdas(scores, labels, groups, skus, dtype=np.float64)
    expected_g, expected_h = direct_derivatives(scores, labels, groups, skus)
    np.testing.assert_allclose(actual_g, expected_g, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(actual_h, expected_h, atol=0, rtol=1e-10)
    assert math.isfinite(pair_loss(scores, frozen_pairs(scores, labels, groups, skus)))
    assert np.all(np.isfinite(actual_g)) and np.all(np.isfinite(actual_h))
    if abs(difference) == 100:
        assert np.all(actual_h > 0)
    elif abs(difference) == 1e4:
        assert np.array_equal(actual_h, [0.0, 0.0])


@pytest.mark.parametrize("positive_rank", [19, 20, 21, 22])
def test_pairs_crossing_rank_twenty_use_full_group_not_truncated_group(positive_rank):
    from coveo_recsys.ranking.objective import session_lambdas

    scores = np.linspace(1.0, -1.0, 22)
    labels = [int(index == positive_rank - 1) for index in range(22)]
    skus = tuple(f"sku-{index:02d}" for index in range(22))
    actual = session_lambdas(scores, labels, (22,), skus, dtype=np.float64)
    expected = direct_derivatives(scores, labels, (22,), skus)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
    assert actual[0][positive_rank - 1] < 0
    if positive_rank > 20:
        other_below_depth = 21 if positive_rank == 21 else 20
        assert actual[0][other_below_depth] == 0
        assert actual[1][other_below_depth] == 0


def test_each_call_recomputes_ranks_instead_of_reusing_first_iteration_weights():
    from coveo_recsys.ranking.objective import session_lambdas

    labels, groups, skus = (1, 0, 1, 0), (4,), ("a", "b", "c", "я")
    results = []
    for scores in ((4.0, 3.0, 2.0, 1.0), (1.0, 2.0, 3.0, 4.0)):
        actual = session_lambdas(scores, labels, groups, skus, dtype=np.float64)
        expected = direct_derivatives(scores, labels, groups, skus)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        results.append(actual)
    assert not np.array_equal(results[0], results[1])
