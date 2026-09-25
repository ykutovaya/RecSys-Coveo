"""Устойчивые FP64-примитивы (sec:ops-numerics).

Элементные функции возвращают ndarray, в том числе нульмерный для скаляра.
Нормировка логитов одномерная. Пустая опора имеет available=False и LSE=None;
-inf в log_probabilities обозначает только явно исключённые позиции.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike

from coveo_recsys.contracts import InputValidationError, NumericError

from ._validation import BoolArray, FloatArray, bool_mask, finite_array, finite_scalar, real_array


def softplus(values: ArrayLike) -> FloatArray:
    x = finite_array(values)
    with np.errstate(under="ignore"):
        return np.asarray(np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x))))


def sigmoid(values: ArrayLike) -> FloatArray:
    x = finite_array(values)
    result = np.empty_like(x)
    positive = x >= 0.0
    with np.errstate(under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exponential = np.exp(x[~positive])
        result[~positive] = exponential / (1.0 + exponential)
    return result


def compensated_sum(values: Iterable[float]) -> float:
    """Сумма с компенсацией через частичные суммы math.fsum.

    При переполнении промежуточной суммы точный рациональный резерв сохраняет
    остаток после компенсации; непредставимый итог вызывает NumericError.
    """
    terms = [finite_scalar(x) for x in values]
    try:
        result = math.fsum(terms)
    except OverflowError:
        try:
            result = float(sum((Fraction.from_float(x) for x in terms), Fraction()))
        except OverflowError as exc:
            raise NumericError("Сумма не представима в FP64") from exc
    if not math.isfinite(result):
        raise NumericError("Сумма не представима в FP64")
    return result


@dataclass(frozen=True, slots=True)
class SoftmaxResult:
    probabilities: FloatArray
    log_probabilities: FloatArray
    logsumexp: float | None
    available: bool


def _prepare_logits(logits: ArrayLike, mask: ArrayLike | None) -> tuple[FloatArray, BoolArray]:
    values = real_array(logits)
    if values.ndim != 1:
        raise InputValidationError("Логиты должны быть одномерными")
    active = bool_mask(mask, values.shape)
    # -inf разрешён лишь вместе с явной выключенной маской. NaN/+inf — ошибка
    # даже вне опоры: маска не превращает внутренний отказ в отсутствие данных.
    if (
        np.any(np.isnan(values))
        or np.any(np.isposinf(values))
        or np.any(np.isneginf(values) & active)
    ):
        raise NumericError("Неконечный логит вне явного маркера исключения")
    return values, active


def _centered(values: FloatArray) -> tuple[FloatArray, float, float]:
    maximum = float(np.max(values))
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - maximum
        total = compensated_sum(np.exp(shifted))
    return shifted, maximum, math.log(total)


def logsumexp(logits: ArrayLike, mask: ArrayLike | None = None) -> float | None:
    """LSE по маске; None для пустой опоры, включая пустой массив."""
    values, active = _prepare_logits(logits, mask)
    if not np.any(active):
        return None
    _, maximum, log_total = _centered(values[active])
    return finite_scalar(maximum + log_total)


def masked_softmax(logits: ArrayLike, mask: ArrayLike | None = None) -> SoftmaxResult:
    """Вероятности и центрированные log-вероятности, без log(округлённого нуля)."""
    values, active = _prepare_logits(logits, mask)
    probabilities = np.zeros(values.shape, dtype=np.float64)
    log_probabilities = np.full(values.shape, -np.inf, dtype=np.float64)
    if not np.any(active):
        return SoftmaxResult(probabilities, log_probabilities, None, False)
    shifted, maximum, log_total = _centered(values[active])
    with np.errstate(over="ignore", under="ignore"):
        logs = shifted - log_total
        probabilities[active] = np.exp(logs)
    if not np.all(np.isfinite(logs)):
        raise NumericError("Центрированные log-вероятности не представимы в FP64")
    log_probabilities[active] = logs
    return SoftmaxResult(probabilities, log_probabilities, finite_scalar(maximum + log_total), True)


def log_softmax(logits: ArrayLike, mask: ArrayLike | None = None) -> FloatArray:
    """Log-вероятности; пустая опора возвращает только маркеры -inf.

    Для явного признака отсутствия источника использовать masked_softmax.
    """
    return masked_softmax(logits, mask).log_probabilities
