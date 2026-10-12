"""Проверки численного ядра; маска не скрывает внутренний NaN/Inf."""

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from coveo_recsys.contracts import InputValidationError, NumericError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def real_array(value: ArrayLike) -> FloatArray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise InputValidationError("Ожидались вещественные числовые значения")
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError("Неверный числовой массив") from exc


def finite_array(value: ArrayLike) -> FloatArray:
    result = real_array(value)
    if not np.all(np.isfinite(result)):
        raise NumericError("Неконечное внутреннее значение")
    return result


def finite_scalar(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InputValidationError("Ожидалось вещественное число")
    try:
        result = float(value)
    except OverflowError as exc:
        raise NumericError("Число не представимо в FP64") from exc
    if not np.isfinite(result):
        raise NumericError("Неконечное внутреннее значение")
    return result


def bool_mask(mask: ArrayLike | None, shape: tuple[int, ...]) -> BoolArray:
    if mask is None:
        return np.ones(shape, dtype=np.bool_)
    try:
        result = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("Неверная маска") from exc
    if (result.dtype.kind != "b" and result.size != 0) or result.shape != shape:
        raise InputValidationError("Маска должна быть булевой и совпадать по форме")
    return np.asarray(result, dtype=np.bool_)
