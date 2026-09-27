"""1000 наборов seed 17: отношение сумм и совместное перемещение пары."""

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.evaluation.intervals import _bootstrap_repetitions, _calendar_sums
from tests.reference.e5_intervals_oracle import grid, repetitions

NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)


@st.composite
def cells(draw):
    size = draw(st.integers(3, 15))
    counts = draw(st.lists(st.integers(0, 8), min_size=size, max_size=size))
    first = [draw(st.integers(0, count)) for count in counts]
    second = [draw(st.integers(0, count)) for count in counts]
    return counts, first, second, draw(st.integers(1, size))


@NORMATIVE
@seed(17)
@given(cells())
def test_joint_pair_repetitions_match_literal_blocks_and_sign_change(case):
    counts, first, second, length = case
    differences = [a - b for a, b in zip(first, second, strict=True)]
    actual = _bootstrap_repetitions(differences, counts, length, repetitions=13)
    np.testing.assert_array_equal(actual, repetitions(differences, counts, length, 13))
    reverse = _bootstrap_repetitions(
        [-value for value in differences], counts, length, repetitions=13
    )
    np.testing.assert_array_equal(reverse, -actual)
    assert all(-1 <= value <= 1 for value in actual if np.isfinite(value))


@NORMATIVE
@seed(17)
@given(st.lists(st.tuples(st.integers(0, 149), st.integers(-1, 1)), max_size=50))
def test_calendar_aggregation_matches_literal_cells_and_is_order_invariant(rows):
    times, values = [100 + row[0] for row in rows], [row[1] for row in rows]
    counts, sums, categories = _calendar_sums(times, values, 100, 10, 15, (-1, 0, 1))
    expected_counts, expected_sums = grid(times, values, 100, 10, 15)
    assert counts.tolist() == expected_counts and sums.tolist() == expected_sums
    reversed_result = _calendar_sums(times[::-1], values[::-1], 100, 10, 15, (-1, 0, 1))
    for first, second in zip((counts, sums, categories), reversed_result, strict=True):
        np.testing.assert_array_equal(first, second)
    assert int(counts.sum()) == len(rows)
    assert int(sums.sum()) == sum(values)
