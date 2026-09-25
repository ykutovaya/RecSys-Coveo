"""Агрегаты полной последней группы (eq:model-last-group-mean).

Каждая позиция — один различный SKU. Кратности уже очищены от транспортных
повторов вызывающей стороной. Недоступные значения не вычисляются: передаётся
конечный заполнитель и маска False, а не скрытый внутренний NaN/Inf.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from coveo_recsys.contracts import InputValidationError, NumericError, require_int64

from ._validation import BoolArray, FloatArray, bool_mask, finite_array
from .integers import counter_add
from .stable import compensated_sum

MultiplicitySemantics = Literal["set_v1", "multiset_v1"]


@dataclass(frozen=True, slots=True)
class MaskedValue:
    value: float
    mask: bool


def _inputs(values: ArrayLike, mask: ArrayLike) -> tuple[FloatArray, BoolArray]:
    data = finite_array(values)
    if data.ndim != 1:
        raise InputValidationError("Значения группы должны быть одномерными")
    return data, bool_mask(mask, data.shape)


def last_group_mean(
    values: ArrayLike,
    mask: ArrayLike,
    multiplicities: Sequence[int],
    *,
    semantics: MultiplicitySemantics = "set_v1",
) -> MaskedValue:
    data, active = _inputs(values, mask)
    if semantics not in ("set_v1", "multiset_v1"):
        raise InputValidationError("Неизвестная семантика кратностей")
    if len(multiplicities) != data.size:
        raise InputValidationError("Кратности не совпадают с формой группы")
    weights: list[int] = []
    for count, is_active in zip(multiplicities, active, strict=True):
        require_int64(count, nonnegative=True)
        if count == 0:
            raise InputValidationError("Кратность различного товара должна быть положительной")
        if is_active:
            weights.append(1 if semantics == "set_v1" else count)
    total = 0
    for weight in weights:
        total = counter_add(total, weight)
    if total == 0:
        return MaskedValue(0.0, False)
    selected = data[active]
    scale = float(np.max(np.abs(selected)))
    if scale == 0.0:
        return MaskedValue(0.0, True)
    # Нормировка ДО умножения на большие значения предотвращает переполнение
    # числителя, даже когда математическое среднее представимо.
    with np.errstate(under="ignore"):
        scaled_mean = compensated_sum(
            (float(value) / scale) * (weight / total)
            for value, weight in zip(selected, weights, strict=True)
        )
    result = scaled_mean * scale
    if not np.isfinite(result):
        raise NumericError("Среднее не представимо в FP64")
    return MaskedValue(result, True)


def last_group_max(values: ArrayLike, mask: ArrayLike) -> MaskedValue:
    """Максимум только доступных значений, независимо от кратностей."""
    data, active = _inputs(values, mask)
    if not np.any(active):
        return MaskedValue(0.0, False)
    return MaskedValue(float(np.max(data[active])), True)
