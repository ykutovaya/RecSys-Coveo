"""Независимые эталоны модальностей: Decimal-норма и явный словарь A_G."""

import math
from decimal import Decimal, localcontext


def vector(raw, dimension, epsilon=1e-12):
    if raw is None:
        return [0.0] * dimension, False, "vector_absent"
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != dimension
        or any(type(value) not in (int, float) for value in raw)
        or any(not math.isfinite(value) for value in raw)
    ):
        return [0.0] * dimension, False, "vector_invalid"
    with localcontext() as context:
        context.prec = 100
        coordinates = [Decimal.from_float(float(value)) for value in raw]
        norm = sum(value * value for value in coordinates).sqrt()
        if norm <= Decimal.from_float(epsilon):
            return [0.0] * dimension, False, "vector_zero"
        return [float(value / norm) for value in coordinates], True, None


def categories(skus, records):
    by_sku = {record.sku: record for record in records}
    known = sorted(
        {by_sku[sku].category for sku in skus if sku in by_sku and by_sku[sku].category},
        key=lambda value: value.encode("utf-8"),
    )
    codes = [
        known.index(by_sku[sku].category) + 1 if sku in by_sku and by_sku[sku].category else 0
        for sku in skus
    ]
    return tuple(known), codes
