"""Раздельная подготовка внешних модальностей без переполнения нормы."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from coveo_recsys.contracts import InputValidationError, NumericError, require_int64

from ._validation import FloatArray, finite_array, finite_scalar, real_array
from .stable import compensated_sum

VectorReason = Literal["vector_absent", "vector_invalid", "vector_zero"]


@dataclass(frozen=True, slots=True)
class NormalizedVector:
    vector: FloatArray
    mask: bool
    reason: VectorReason | None


def _components(vector: FloatArray) -> tuple[float, FloatArray, float]:
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0, np.zeros_like(vector), 0.0
    with np.errstate(under="ignore"):
        scaled = vector / scale
        radius = math.sqrt(compensated_sum(scaled * scaled))
    return scale, scaled, radius


def scaled_norm(vector: ArrayLike) -> float:
    """Евклидова норма конечного 1D-вектора; пустая норма равна нулю.

    Если сама норма не представима, возникает NumericError. Нормирование
    внешней модальности обходится без произведения scale*radius.
    """
    values = finite_array(vector)
    if values.ndim != 1:
        raise InputValidationError("Ожидался одномерный вектор")
    scale, _, radius = _components(values)
    return finite_scalar(scale * radius)


def normalize_external_vector(
    vector: ArrayLike | None,
    expected_dim: int,
    *,
    epsilon: float = 1e-12,
) -> NormalizedVector:
    """Непригодный внешний вход даёт нулевой вектор и отдельную причину.

    Ошибки expected_dim/epsilon — ошибки обязательной конфигурации, не пропуск
    модальности. Не использовать эту функцию для маскировки внутренних ошибок.
    """
    require_int64(expected_dim, nonnegative=True)
    if expected_dim == 0:
        raise InputValidationError("Ожидаемая размерность должна быть положительной")
    try:
        threshold = finite_scalar(epsilon)
    except NumericError as exc:
        raise InputValidationError("Порог нормы должен быть конечным") from exc
    if threshold <= 0.0:
        raise InputValidationError("Порог нормы должен быть положительным")

    def unavailable(reason: VectorReason) -> NormalizedVector:
        return NormalizedVector(np.zeros(expected_dim, dtype=np.float64), False, reason)

    if vector is None:
        return unavailable("vector_absent")
    # NumPy иначе превращает [True, 2.0] в числовой массив до проверки типа.
    if isinstance(vector, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in vector):
        return unavailable("vector_invalid")
    try:
        values = real_array(vector)
    except InputValidationError:
        return unavailable("vector_invalid")
    if values.shape != (expected_dim,) or not np.all(np.isfinite(values)):
        return unavailable("vector_invalid")
    scale, scaled, radius = _components(values)
    if scale == 0.0 or scale <= threshold / radius:
        return unavailable("vector_zero")
    return NormalizedVector(scaled / radius, True, None)
