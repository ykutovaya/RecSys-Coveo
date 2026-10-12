"""Каждое свойство: 1000 случаев, seed=17, Hypothesis со сжатием контрпримера."""

import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts import INT64_MAX, INT64_MIN, NumericError
from coveo_recsys.numerics import (
    checked_add,
    checked_multiply,
    checked_subtract,
    compensated_sum,
    last_group_max,
    last_group_mean,
    masked_softmax,
    normalize_external_vector,
    sigmoid,
    softplus,
    sort_ids,
    sort_scored_ids,
)
from tests.reference import numerics_oracle as oracle

ATOL, RTOL = 1e-10, 1e-8
NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
FINITE = st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False)


@NORMATIVE
@seed(17)
@given(FINITE)
def test_elementwise_against_decimal(value):
    np.testing.assert_allclose(softplus(value), oracle.softplus(value), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(sigmoid(value), oracle.sigmoid(value), atol=ATOL, rtol=RTOL)


@st.composite
def masked_values(draw):
    values = draw(st.lists(FINITE, max_size=12))
    mask = draw(st.lists(st.booleans(), min_size=len(values), max_size=len(values)))
    return values, np.array(mask, dtype=bool)


@NORMATIVE
@seed(17)
@given(masked_values())
def test_masked_distribution_against_decimal(case):
    values, mask = case
    expected_lse, probabilities, logs = oracle.log_distribution(values, mask)
    actual = masked_softmax(values, mask)
    assert actual.available == bool(np.any(mask))
    if expected_lse is None:
        assert actual.logsumexp is None
    else:
        np.testing.assert_allclose(actual.logsumexp, expected_lse, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(actual.probabilities.sum(), 1.0, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.probabilities, probabilities, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.log_probabilities, logs, atol=ATOL, rtol=RTOL)
    assert np.all(actual.probabilities[~mask] == 0.0)
    assert np.all(np.isfinite(actual.log_probabilities[mask]))


@NORMATIVE
@seed(17)
@given(
    st.lists(
        st.floats(-1e200, 1e200, allow_nan=False, allow_infinity=False), min_size=1, max_size=20
    )
)
def test_vector_direction_against_decimal(values):
    norm, direction = oracle.normalize(values)
    actual = normalize_external_vector(values, len(values))
    assert actual.mask == (norm > 1e-12)
    if actual.mask:
        np.testing.assert_allclose(actual.vector, direction, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(np.dot(actual.vector, actual.vector), 1.0, atol=ATOL, rtol=RTOL)
    else:
        assert actual.reason == "vector_zero"
        assert np.all(actual.vector == 0.0)


@NORMATIVE
@seed(17)
@given(st.lists(st.floats(-1e200, 1e200, allow_nan=False, allow_infinity=False), max_size=20))
def test_compensated_sum_against_decimal(values):
    np.testing.assert_allclose(
        compensated_sum(values), oracle.exact_sum(values), atol=ATOL, rtol=RTOL
    )


@NORMATIVE
@seed(17)
@given(st.integers(INT64_MIN, INT64_MAX), st.integers(INT64_MIN, INT64_MAX))
def test_checked_integer_arithmetic(left, right):
    for operation, expected in (
        (checked_add, left + right),
        (checked_subtract, left - right),
        (checked_multiply, left * right),
    ):
        if INT64_MIN <= expected <= INT64_MAX:
            assert operation(left, right) == expected
        else:
            with pytest.raises(NumericError):
                operation(left, right)


@st.composite
def groups(draw):
    values, mask = draw(masked_values())
    counts = draw(st.lists(st.integers(1, 1000), min_size=len(values), max_size=len(values)))
    permutation = draw(st.permutations(tuple(range(len(values)))))
    return values, mask, counts, permutation


@NORMATIVE
@seed(17)
@given(groups())
def test_last_group_both_semantics_and_permutations(case):
    values, mask, counts, permutation = case
    for semantics in ("set_v1", "multiset_v1"):
        expected_mean, expected_max, expected_mask = oracle.group(values, mask, counts, semantics)
        mean = last_group_mean(values, mask, counts, semantics=semantics)
        maximum = last_group_max(values, mask)
        assert mean.mask == maximum.mask == expected_mask
        np.testing.assert_allclose(mean.value, expected_mean, atol=ATOL, rtol=RTOL)
        assert maximum.value == expected_max
        reordered = last_group_mean(
            [values[i] for i in permutation],
            mask[list(permutation)],
            [counts[i] for i in permutation],
            semantics=semantics,
        )
        assert mean.mask == reordered.mask
        np.testing.assert_allclose(mean.value, reordered.value, atol=ATOL, rtol=RTOL)


@NORMATIVE
@seed(17)
@given(st.lists(st.integers(1, 10**12), min_size=1, max_size=100))
def test_maximum_finite_mean_does_not_overflow(weights):
    largest = np.finfo(float).max
    near_largest = float(np.nextafter(largest, 0.0))
    for values in (
        [largest] * len(weights),
        [-largest] * len(weights),
        [largest if i % 2 else near_largest for i in range(len(weights))],
    ):
        result = last_group_mean(values, [True] * len(weights), weights, semantics="multiset_v1")
        expected, _, _ = oracle.group(values, [True] * len(weights), weights, "multiset_v1")
        assert result.mask and np.isfinite(result.value)
        # Для FP64 нормативен допуск, а не побитовое равенство суммы весов единице.
        np.testing.assert_allclose(result.value, expected, atol=ATOL, rtol=RTOL)


@NORMATIVE
@seed(17)
@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=8),
            FINITE,
        ),
        max_size=20,
    )
)
def test_exact_sorting_and_ties(items):
    expected_ids = tuple(sorted((sku for sku, _ in items), key=lambda sku: sku.encode("utf-8")))
    expected_scores = tuple(sorted(items, key=lambda pair: (-pair[1], pair[0].encode("utf-8"))))
    assert sort_ids(sku for sku, _ in items) == expected_ids
    assert sort_scored_ids(items) == expected_scores
