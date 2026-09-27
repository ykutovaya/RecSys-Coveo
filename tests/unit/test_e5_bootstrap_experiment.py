"""B12: дисперсия повторённых знаков и заранее закреплённая проверка покрытия."""

from itertools import product

import numpy as np
import pytest

from coveo_recsys.evaluation.intervals import (
    _bootstrap_repetitions,
    _calendar_sums,
    nearest_rank,
)
from tests.reference.e5_bootstrap_experiment import (
    COVERAGE_BLOCK_CELLS,
    COVERAGE_EXPERIMENTS,
    binomial_acceptance,
    repeated_sign_experiments,
)


@pytest.mark.parametrize("multiplicity", [1, 2, 5])
def test_repeated_independent_signs_have_block_variance_not_iid_variance(multiplicity):
    blocks = 8
    means = []
    for signs in product((-1, 1), repeat=blocks):
        outcomes = [sign for sign in signs for _ in range(multiplicity)]
        counts, numerators, _ = _calendar_sums(
            list(range(len(outcomes))), outcomes, 0, multiplicity, blocks, (-1, 0, 1)
        )
        means.append(int(numerators.sum()) / int(counts.sum()))
    # Полный перебор равновероятных исходов: дисперсия распределения, ddof=0.
    actual = float(np.var(means))
    assert actual == 1 / blocks
    iid_variance = 1 / (blocks * multiplicity)
    assert actual == multiplicity * iid_variance


def test_moving_block_interval_covers_zero_with_fixed_binomial_tolerance():
    covered = 0
    for signs in repeated_sign_experiments():
        repeats = _bootstrap_repetitions(signs, [1] * len(signs), COVERAGE_BLOCK_CELLS)
        lower, upper = nearest_rank(repeats, 0.025), nearest_rank(repeats, 0.975)
        covered += lower <= 0 <= upper
    lower_count, upper_count = binomial_acceptance(COVERAGE_EXPERIMENTS, 0.95, 0.99)
    assert lower_count <= covered <= upper_count, (
        f"Покрытие {covered}/{COVERAGE_EXPERIMENTS}; "
        f"заранее заданный 99%-допуск [{lower_count}, {upper_count}]"
    )
