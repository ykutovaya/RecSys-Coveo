"""План B12 закреплён до опыта; генератор входов не зависит от рабочего bootstrap.

128 опытов, 400 независимых равновероятных знаков в каждом, 4 соседних
копии каждого знака. Блок bootstrap содержит 40 ячеек; штатные B=2000,
seed=17 и alpha=.05 задаёт проверяемый модуль. Принимается центральный
99%-диапазон Bin(128,.95). Допуск относится к частоте покрытия, а не
к требованию сделать каждый отдельный интервал шире IID-интервала.
"""

import random
from decimal import Decimal, localcontext

COVERAGE_EXPERIMENTS = 128
INDEPENDENT_SIGNS = 400
SIGN_MULTIPLICITY = 4
COVERAGE_BLOCK_CELLS = 40
INPUT_SEED = 17


def repeated_sign_experiments():
    rng = random.Random(INPUT_SEED)
    for _ in range(COVERAGE_EXPERIMENTS):
        yield tuple(
            sign
            for sign in (rng.choice((-1, 1)) for _ in range(INDEPENDENT_SIGNS))
            for _ in range(SIGN_MULTIPLICITY)
        )


def binomial_acceptance(trials, probability, confidence):
    """Точные дискретные квантили биномиального закона, Decimal-рекурсия."""
    with localcontext() as context:
        context.prec = 70
        p, level = Decimal(str(probability)), Decimal(str(confidence))
        lower_probability, upper_probability = (1 - level) / 2, (1 + level) / 2
        mass = (1 - p) ** trials
        cumulative = mass
        lower = 0 if cumulative >= lower_probability else None
        for successes in range(trials + 1):
            if successes:
                mass *= Decimal(trials - successes + 1) / successes * p / (1 - p)
                cumulative += mass
            if lower is None and cumulative >= lower_probability:
                lower = successes
            if cumulative >= upper_probability:
                assert lower is not None
                return lower, successes
    raise AssertionError("Не получены биномиальные границы при заданной точности")
