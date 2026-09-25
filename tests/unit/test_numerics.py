"""Малые примеры ранних частей B01/B05; не полный успех нормативных ID."""

import math

import numpy as np
import pytest

from coveo_recsys.contracts import INT64_MAX, INT64_MIN, InputValidationError, NumericError
from coveo_recsys.numerics import (
    checked_add,
    checked_multiply,
    checked_subtract,
    compensated_sum,
    counter_add,
    last_group_max,
    last_group_mean,
    log_softmax,
    logsumexp,
    masked_softmax,
    normalize_external_vector,
    scaled_norm,
    seconds_to_milliseconds,
    sigmoid,
    softplus,
    sort_ids,
    sort_scored_ids,
)
from tests.reference import numerics_oracle as oracle

ATOL, RTOL = 1e-10, 1e-8


@pytest.mark.parametrize("value", [-1e4, -100.0, 0.0, 100.0, 1e4])
def test_b05_saturated_elementwise(value):
    np.testing.assert_allclose(softplus(value), oracle.softplus(value), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(sigmoid(value), oracle.sigmoid(value), atol=ATOL, rtol=RTOL)
    assert np.isfinite(softplus(value))
    assert np.isfinite(sigmoid(value))


def test_b05_lse_centered_log_softmax_and_mask_before_maximum():
    logits = np.array([1e4, -100.0, 0.0, 100.0, -1e4])
    mask = np.array([False, True, True, True, True])
    expected_lse, expected_probabilities, expected_logs = oracle.log_distribution(logits, mask)
    actual = masked_softmax(logits, mask)
    assert actual.available
    np.testing.assert_allclose(actual.logsumexp, expected_lse, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.probabilities, expected_probabilities, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.log_probabilities, expected_logs, atol=ATOL, rtol=RTOL)
    assert actual.probabilities[-1] == 0.0
    assert math.isfinite(actual.log_probabilities[-1])
    np.testing.assert_allclose(log_softmax([1e200, 1e200]), [-math.log(2)] * 2)
    assert logsumexp([1e200, 1e200]) == 1e200
    assert logsumexp([-1e308, 1e308]) == 1e308


@pytest.mark.parametrize(
    "logits,mask", [([], None), ([], []), ([1.0, -2.0], [False, False]), ([-np.inf], [False])]
)
def test_empty_support_is_not_an_invalid_subtraction(logits, mask):
    actual = masked_softmax(logits, mask)
    assert not actual.available
    assert actual.logsumexp is None
    assert logsumexp(logits, mask) is None
    assert np.all(actual.probabilities == 0.0)
    assert np.all(np.isneginf(actual.log_probabilities))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_internal_nonfinite_cannot_be_hidden(bad):
    for function in (softplus, sigmoid, scaled_norm):
        with pytest.raises(NumericError):
            function([bad])
    with pytest.raises(NumericError):
        compensated_sum([bad])
    with pytest.raises(NumericError):
        last_group_mean([bad], [False], [1])
    with pytest.raises(NumericError):
        last_group_max([bad], [False])
    with pytest.raises(NumericError):
        masked_softmax([bad])
    if bad != -np.inf:
        with pytest.raises(NumericError):
            masked_softmax([bad], [False])


def test_logit_structure_and_explicit_exclusion():
    with pytest.raises(InputValidationError):
        masked_softmax([[1.0]])
    with pytest.raises(InputValidationError):
        masked_softmax([1.0], [1])
    with pytest.raises(InputValidationError):
        masked_softmax([1.0], [True, False])
    with pytest.raises(NumericError):
        log_softmax([-1e308, 1e308])
    result = masked_softmax([3.0, -np.inf], [True, False])
    np.testing.assert_array_equal(result.probabilities, [1.0, 0.0])
    np.testing.assert_array_equal(result.log_probabilities, [0.0, -np.inf])


@pytest.mark.parametrize(
    "value,reason",
    [
        (None, "vector_absent"),
        ([0.0, 0.0], "vector_zero"),
        ([1e-12, 0.0], "vector_zero"),
        ([1e-300, 0.0], "vector_zero"),
        ([1.0], "vector_invalid"),
        ([1.0, np.nan], "vector_invalid"),
        ([np.inf, 0.0], "vector_invalid"),
        (["1", "2"], "vector_invalid"),
        ([True, False], "vector_invalid"),
        ([True, 2.0], "vector_invalid"),
        ([1 + 2j, 0j], "vector_invalid"),
        ([[1.0, 2.0]], "vector_invalid"),
        ([[1.0], [2.0, 3.0]], "vector_invalid"),
    ],
)
def test_b05_external_vector_reasons(value, reason):
    result = normalize_external_vector(value, 2)
    assert not result.mask
    assert result.reason == reason
    np.testing.assert_array_equal(result.vector, np.zeros(2))


def test_b05_large_vectors_cancellation_and_independent_modalities():
    values = np.array([1e200, -1e200])
    original = values.copy()
    result = normalize_external_vector(values, 2)
    assert result.mask and result.reason is None
    np.testing.assert_allclose(result.vector, [1 / math.sqrt(2), -1 / math.sqrt(2)])
    np.testing.assert_array_equal(values, original)
    assert scaled_norm(values) == pytest.approx(math.sqrt(2) * 1e200)
    cancelled = normalize_external_vector(result.vector + (-result.vector), 2)
    assert cancelled.reason == "vector_zero"
    invalid_other_modality = normalize_external_vector([np.nan, 0.0], 2)
    assert not invalid_other_modality.mask and result.mask
    huge = normalize_external_vector([np.finfo(float).max] * 4, 4)
    np.testing.assert_array_equal(huge.vector, [0.5] * 4)
    with pytest.raises(NumericError):
        scaled_norm([np.finfo(float).max] * 4)
    assert normalize_external_vector([np.nextafter(1e-12, np.inf), 0.0], 2).mask
    assert scaled_norm([]) == 0.0


@pytest.mark.parametrize(
    "dimension,epsilon",
    [(0, 1e-12), (-1, 1e-12), (True, 1e-12), (2, 0.0), (2, np.nan), (2, np.inf)],
)
def test_invalid_vector_configuration_is_not_missing_modality(dimension, epsilon):
    with pytest.raises(InputValidationError):
        normalize_external_vector(None, dimension, epsilon=epsilon)


def test_compensated_sum_retains_small_residual_and_checks_overflow():
    assert compensated_sum([1e16, 1.0, -1e16]) == 1.0
    assert compensated_sum([]) == 0.0
    largest = np.finfo(float).max
    assert compensated_sum([largest, largest, -largest]) == largest
    assert compensated_sum([largest, largest, -largest, -largest, 1e-100]) == 1e-100
    smallest = float(np.nextafter(0.0, 1.0))
    assert compensated_sum([largest, largest, -largest, -largest, smallest]) == smallest
    with pytest.raises(NumericError):
        compensated_sum([largest, largest])


def test_b01_int64_boundaries_and_explicit_units():
    assert checked_add(INT64_MAX, 0) == INT64_MAX
    assert checked_subtract(INT64_MIN, 0) == INT64_MIN
    assert checked_add(INT64_MIN, INT64_MAX) == -1
    assert seconds_to_milliseconds(1800) == 1_800_000
    assert seconds_to_milliseconds(-1) == -1000
    assert checked_subtract(1001, 1000) == 1
    assert checked_add(1000, 1) > 1000
    assert counter_add(2, -2) == 0
    for function, args in [
        (checked_add, (INT64_MAX, 1)),
        (checked_add, (INT64_MIN, -1)),
        (checked_subtract, (INT64_MAX, -1)),
        (checked_subtract, (INT64_MIN, 1)),
        (checked_multiply, (INT64_MIN, -1)),
        (checked_multiply, (INT64_MAX, 2)),
        (seconds_to_milliseconds, (INT64_MAX,)),
        (counter_add, (INT64_MAX, 1)),
        (counter_add, (0, -1)),
    ]:
        with pytest.raises(NumericError):
            function(*args)


@pytest.mark.parametrize("bad", [True, 1.5, 1.0, "1", np.int64(1), INT64_MAX + 1])
def test_checked_arithmetic_does_not_coerce_input(bad):
    with pytest.raises(InputValidationError):
        checked_add(bad, 0)
    with pytest.raises(InputValidationError):
        seconds_to_milliseconds(bad)


def test_b01_masked_last_group_examples():
    values, counts = [1.0, 0.0], [9, 1]
    assert last_group_mean(values, [True, True], counts).value == 0.5
    assert last_group_mean(values, [True, True], counts, semantics="multiset_v1").value == 0.9
    for semantics in ("set_v1", "multiset_v1"):
        one = last_group_mean(values, [True, False], counts, semantics=semantics)
        assert one.value == 1.0 and one.mask
        neither = last_group_mean(values, [False, False], counts, semantics=semantics)
        assert neither.value == 0.0 and not neither.mask
    assert last_group_max(values, [True, True]).value == 1.0
    assert last_group_max([100.0, -3.0], [False, True]).value == -3.0
    assert not last_group_max(values, [False, False]).mask
    actual = last_group_mean([1e308, 1e308], [True, True], [9, 1], semantics="multiset_v1")
    assert actual.value == 1e308 and actual.mask
    assert last_group_mean([1.0], [True], [INT64_MAX], semantics="multiset_v1").value == 1.0
    with pytest.raises(NumericError):
        last_group_mean(values, [True, True], [INT64_MAX, 1], semantics="multiset_v1")
    assert not last_group_mean([], [], []).mask
    assert not last_group_max([], []).mask


def test_maximal_group_and_extreme_weights_remain_representable():
    largest = np.finfo(float).max
    for counts in ([1, 49], [INT64_MAX - 1024, 1024], [1] * 1000, list(range(1, 1001))):
        values = [largest] * len(counts)
        mask = [True] * len(counts)
        for semantics in ("set_v1", "multiset_v1"):
            result = last_group_mean(values, mask, counts, semantics=semantics)
            assert result.mask and result.value == largest
            assert last_group_max(values, mask).value == largest


@pytest.mark.parametrize(
    "counts,semantics",
    [([0], "set_v1"), ([-1], "set_v1"), ([True], "set_v1"), ([], "set_v1"), ([1], "unknown")],
)
def test_invalid_multiplicities(counts, semantics):
    with pytest.raises(InputValidationError):
        last_group_mean([1.0], [True], counts, semantics=semantics)


def test_tie_order_exact_utf8_and_nearly_equal_scores():
    assert sort_ids(["é", "e\u0301", "Я", "A"]) == ("A", "e\u0301", "é", "Я")
    higher = float(np.nextafter(1.0, np.inf))
    assert sort_scored_ids([("A", 1.0), ("Я", higher), ("é", 1.0)]) == (
        ("Я", higher),
        ("A", 1.0),
        ("é", 1.0),
    )
    assert sort_scored_ids([("b", -0.0), ("a", 0.0)]) == (("a", 0.0), ("b", -0.0))
    with pytest.raises(NumericError):
        sort_scored_ids([("a", np.nan)])
    with pytest.raises(InputValidationError):
        sort_ids(["\ud800"])
