"""Контракт производных: состав групп, типы и связь с Dataset."""

from types import SimpleNamespace

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.ranking.objective import make_objective, session_lambdas
from tests.reference.e4_objective_oracle import direct_derivatives


@pytest.mark.parametrize("dtype,atol,rtol", [(np.float64, 1e-12, 1e-10), (np.float32, 1e-7, 1e-6)])
def test_multiple_groups_match_direct_pair_sums(dtype, atol, rtol):
    labels = [1, 0, 1, 0, 0, 0, 1, 1, 1]
    scores = [0.0, 0.0, 0.0, 0.0, 3.0, -2.0, 1.0, 0.0, -1.0]
    skus = ["я", "é", "a", "Ω", "same", "b", "same", "c", "d"]
    groups = [4, 2, 3]
    expected = direct_derivatives(scores, labels, groups, skus)
    actual = session_lambdas(scores, labels, groups, skus, dtype=dtype)
    for values, reference in zip(actual, expected, strict=True):
        assert values.dtype == dtype
        assert values.flags.c_contiguous
        np.testing.assert_allclose(values, reference, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(values[4:], np.zeros(5))


def test_empty_table_has_empty_derivatives():
    for values in session_lambdas([], [], [], []):
        assert values.shape == (0,)
        assert values.dtype == np.float32


@pytest.mark.parametrize(
    "scores,labels,groups,skus,error",
    [
        ([[0.0, 0.0]], [1, 0], [2], ["a", "b"], InputValidationError),
        ([0.0], [1, 0], [2], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [[1, 0]], [2], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 2], [2], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0.5], [2], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [1], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [0, 2], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [-1, 3], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [2.0], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [True, True], ["a", "b"], InputValidationError),
        ([0.0, 0.0], [1, 0], [2], ["a"], InputValidationError),
        ([0.0, 0.0], [1, 0], [2], ["a", "a"], InputValidationError),
        ([0.0, 0.0], [1, 0], [2], ["a", ""], InputValidationError),
        ([0.0, 0.0], [1, 0], [2], ["a", "\ud800"], InputValidationError),
        ([np.nan, 0.0], [1, 0], [2], ["a", "b"], NumericError),
        ([np.inf, 0.0], [1, 0], [2], ["a", "b"], NumericError),
        ([-np.inf, 0.0], [1, 0], [2], ["a", "b"], NumericError),
        ([0.0, 0.0], [1, np.nan], [2], ["a", "b"], NumericError),
        ([1e308, -1e308], [1, 0], [2], ["a", "b"], NumericError),
    ],
)
def test_invalid_derivative_inputs_fail(scores, labels, groups, skus, error):
    with pytest.raises(error):
        session_lambdas(scores, labels, groups, skus)


@pytest.mark.parametrize("dtype", [np.float16, np.int32, "bad"])
def test_only_fp32_or_fp64_outputs_are_allowed(dtype):
    with pytest.raises(InputValidationError):
        session_lambdas([0.0, 0.0], [1, 0], [2], ["a", "b"], dtype=dtype)


def dataset(*, labels=(1, 0), groups=(2,), weight=None, init_score=None):
    return SimpleNamespace(
        get_label=lambda: np.asarray(labels),
        get_group=lambda: np.asarray(groups),
        get_weight=lambda: weight,
        get_init_score=lambda: init_score,
    )


@pytest.mark.parametrize(
    "data",
    [
        dataset(labels=(0, 1)),
        dataset(groups=(1, 1)),
        dataset(weight=np.ones(2)),
        dataset(init_score=np.zeros(2)),
    ],
)
def test_callback_rejects_other_groups_labels_or_external_weights(data):
    callback = make_objective([1, 0], [2], ["a", "b"])
    with pytest.raises(InputValidationError):
        callback(np.zeros(2), data)


def test_callback_captures_inputs_and_checks_dataset_each_time():
    labels, groups, skus = [1, 0], [2], ["a", "b"]
    callback = make_objective(labels, groups, skus)
    labels.reverse()
    groups[:] = [1, 1]
    skus.reverse()
    data = dataset()
    expected = session_lambdas([0.0, 0.0], [1, 0], [2], ["a", "b"])
    for actual, reference in zip(callback(np.zeros(2), data), expected, strict=True):
        assert actual.dtype == np.float32 and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, reference)
    data.get_weight = lambda: np.ones(2)
    with pytest.raises(InputValidationError):
        callback(np.zeros(2), data)
