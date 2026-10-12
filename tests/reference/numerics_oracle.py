"""Прямые формулы в Decimal, независимо от numerics и NumPy/SciPy-примитивов."""

from decimal import Decimal, localcontext


def decimal(value):
    return Decimal.from_float(float(value))


def softplus(value):
    with localcontext() as context:
        context.prec = 100
        return float((Decimal(1) + decimal(value).exp()).ln())


def sigmoid(value):
    with localcontext() as context:
        context.prec = 100
        return float(Decimal(1) / (Decimal(1) + (-decimal(value)).exp()))


def normalize(values):
    with localcontext() as context:
        context.prec = 100
        vector = [decimal(value) for value in values]
        norm = sum((value * value for value in vector), Decimal(0)).sqrt()
        if norm == 0:
            return 0.0, [0.0 for _ in vector]
        return float(norm), [float(value / norm) for value in vector]


def log_distribution(values, mask):
    with localcontext() as context:
        context.prec = 100
        selected = [decimal(value) for value, include in zip(values, mask) if include]
        if not selected:
            return None, [0.0 for _ in values], [float("-inf") for _ in values]
        maximum = max(selected)
        total = sum(((value - maximum).exp() for value in selected), Decimal(0))
        logs = [(value - maximum) - total.ln() for value in selected]
        iterator = iter(logs)
        all_logs = [float(next(iterator)) if include else float("-inf") for include in mask]
        iterator = iter(logs)
        probabilities = [float(next(iterator).exp()) if include else 0.0 for include in mask]
        return float(maximum + total.ln()), probabilities, all_logs


def exact_sum(values):
    with localcontext() as context:
        # Полный диапазон конечных FP64, включая компенсацию max + subnormal - max.
        context.prec = 1200
        return float(sum((decimal(value) for value in values), Decimal(0)))


def group(values, mask, counts, semantics):
    with localcontext() as context:
        context.prec = 1200
        pairs = [
            (decimal(value), 1 if semantics == "set_v1" else count)
            for value, include, count in zip(values, mask, counts)
            if include
        ]
        if not pairs:
            return 0.0, 0.0, False
        mean = sum((value * count for value, count in pairs), Decimal(0)) / sum(
            count for _, count in pairs
        )
        return float(mean), float(max(value for value, _ in pairs)), True
