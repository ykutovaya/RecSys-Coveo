"""Производные: независимая сумма, сдвиг и перестановка; 1000 случаев, seed 17."""

import numpy as np
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.ranking.objective import session_lambdas
from tests.reference.e4_objective_oracle import direct_derivatives


@seed(17)
@given(
    st.lists(
        st.lists(st.tuples(st.integers(-20, 20), st.integers(0, 1)), min_size=1, max_size=30),
        min_size=1,
        max_size=4,
    ),
    st.integers(-100, 100),
)
def test_pair_sums_are_shift_and_within_group_permutation_invariant(groups, shift):
    sizes = [len(group) for group in groups]
    scores = np.array([score for group in groups for score, _ in group], dtype=np.float64)
    labels = np.array([label for group in groups for _, label in group])
    skus = [f"sku-{i:03}" for i in range(len(scores))]
    actual = session_lambdas(scores, labels, sizes, skus, dtype=np.float64)
    expected = direct_derivatives(scores, labels, sizes, skus)
    shifted = session_lambdas(scores + shift, labels, sizes, skus, dtype=np.float64)
    permutation, offset = [], 0
    for size in sizes:
        permutation.extend(range(offset + size - 1, offset - 1, -1))
        assert abs(float(actual[0][offset : offset + size].sum())) <= 1e-12
        offset += size
    permuted = session_lambdas(
        scores[permutation],
        labels[permutation],
        sizes,
        [skus[i] for i in permutation],
        dtype=np.float64,
    )
    for index in range(2):
        np.testing.assert_allclose(actual[index], expected[index], atol=1e-12, rtol=1e-10)
        np.testing.assert_array_equal(shifted[index], actual[index])
        np.testing.assert_allclose(
            permuted[index], actual[index][permutation], atol=1e-12, rtol=1e-10
        )
    assert np.all(actual[1] >= 0.0)
