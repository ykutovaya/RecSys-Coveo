"""Буквальные календарные суммы и выборка индексов, независимо от реализации."""

from decimal import Decimal, localcontext
from math import ceil

import numpy as np


def repetitions(numerators, counts, length, repetitions=2000):
    rng = np.random.Generator(np.random.PCG64(17))
    values = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(counts) - length + 1, size=ceil(len(counts) / length))
        indices = [cell for start in starts for cell in range(start, start + length)][: len(counts)]
        denominator = sum(counts[cell] for cell in indices)
        values.append(
            sum(numerators[cell] for cell in indices) / denominator if denominator else float("nan")
        )
    return np.array(values)


def grid(times, outcomes, start, step, cells):
    groups = [
        [
            value
            for time, value in zip(times, outcomes, strict=True)
            if start + j * step <= time < start + (j + 1) * step
        ]
        for j in range(cells)
    ]
    return [len(group) for group in groups], [sum(group) for group in groups]


def quantile(values, probability):
    return sorted(values)[ceil(probability * len(values)) - 1]


def mc_ranks(n, probability):
    """Decimal-рекурсия биномиальных вероятностей, не логарифмическая реализация."""
    with localcontext() as context:
        context.prec = 60
        p = Decimal(str(probability))
        reverse = p > Decimal("0.5")
        if reverse:
            p = 1 - p
        mass = (1 - p) ** n
        masses = [mass]
        for k in range(n):
            mass *= Decimal(n - k) / (k + 1) * p / (1 - p)
            masses.append(mass)
        if reverse:
            masses.reverse()
        cumulative = Decimal(0)
        ranks = []
        for index, mass in enumerate(masses):
            cumulative += mass
            if not ranks and cumulative >= Decimal("0.025"):
                ranks.append(index)
            if cumulative >= Decimal("0.975"):
                ranks.append(index + 1)
                return tuple(ranks)
    raise AssertionError("Недостаточная точность независимого эталона")
