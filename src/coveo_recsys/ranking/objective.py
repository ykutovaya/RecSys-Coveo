"""session_lambdas_v1: полные ранги групп, парные веса и диагональные производные."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from coveo_recsys.contracts import InputValidationError, NumericError, id_bytes, require_int64
from coveo_recsys.numerics import compensated_sum, sigmoid
from coveo_recsys.numerics._validation import finite_array

if TYPE_CHECKING:
    from lightgbm import Dataset

DerivativeArray = NDArray[np.float32] | NDArray[np.float64]
Derivatives = tuple[DerivativeArray, DerivativeArray]


def _layout(
    labels: ArrayLike, group_sizes: Sequence[int], skus: Sequence[str]
) -> tuple[NDArray[np.float64], tuple[int, ...], tuple[bytes, ...]]:
    targets = finite_array(labels)
    if targets.ndim != 1 or np.any((targets != 0) & (targets != 1)):
        raise InputValidationError("Метки должны быть одномерными и бинарными")
    sizes = []
    for size in group_sizes:
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise InputValidationError("Размер группы должен быть целым")
        checked = require_int64(int(size), nonnegative=True)
        if checked == 0:
            raise InputValidationError("Dataset содержит только непустые группы")
        sizes.append(checked)
    if sum(sizes) != targets.size or isinstance(skus, (str, bytes)) or len(skus) != targets.size:
        raise InputValidationError("Размеры групп, меток и SKU не совпадают")
    keys = tuple(id_bytes(sku) for sku in skus)
    offset = 0
    for size in sizes:
        if len(set(keys[offset : offset + size])) != size:
            raise InputValidationError("SKU внутри группы не должны повторяться")
        offset += size
    return targets.copy(), tuple(sizes), keys


def _derivatives(
    scores: ArrayLike,
    labels: NDArray[np.float64],
    sizes: tuple[int, ...],
    keys: tuple[bytes, ...],
    dtype: DTypeLike,
) -> Derivatives:
    values = finite_array(scores)
    if values.ndim != 1 or values.shape != labels.shape:
        raise InputValidationError("Баллы должны совпадать по форме с метками")
    try:
        output_type = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("Производные поддерживают только FP32 и FP64") from exc
    if output_type not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise InputValidationError("Производные поддерживают только FP32 и FP64")
    gradient = np.zeros(values.size, dtype=np.float64)
    hessian = np.zeros(values.size, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 22, dtype=np.float64))
    offset = 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for size in sizes:
                stop = offset + size
                positive = np.flatnonzero(labels[offset:stop] == 1) + offset
                negative = np.flatnonzero(labels[offset:stop] == 0) + offset
                if positive.size and negative.size:
                    order = sorted(range(offset, stop), key=lambda i: (-values[i], keys[i]))
                    rank_discounts = np.zeros(size, dtype=np.float64)
                    for rank, index in enumerate(order[:20]):
                        rank_discounts[index - offset] = discounts[rank]
                    idcg = compensated_sum(discounts[: min(20, positive.size)])
                    pair_weights = np.abs(
                        rank_discounts[positive - offset, None] - rank_discounts[negative - offset]
                    ) / (idcg * int(positive.size) * int(negative.size))
                    differences = values[positive, None] - values[negative]
                    pair_gradient = pair_weights * sigmoid(-differences)
                    # exp(-abs(d)) сохраняет малую кривизну и для отрицательного d.
                    exponential = np.exp(-np.abs(differences))
                    pair_hessian = pair_weights * exponential / (1.0 + exponential) ** 2
                    gradient[positive] = -pair_gradient.sum(axis=1, dtype=np.float64)
                    gradient[negative] = pair_gradient.sum(axis=0, dtype=np.float64)
                    hessian[positive] = pair_hessian.sum(axis=1, dtype=np.float64)
                    hessian[negative] = pair_hessian.sum(axis=0, dtype=np.float64)
                offset = stop
            result = (
                np.ascontiguousarray(gradient, dtype=output_type),
                np.ascontiguousarray(hessian, dtype=output_type),
            )
    except FloatingPointError as exc:
        raise NumericError("Баллы или производные не представимы в выбранном типе") from exc
    if not all(np.all(np.isfinite(array)) for array in result):
        raise NumericError("Неконечные производные ранжировщика")
    return result


def session_lambdas(
    scores: ArrayLike,
    labels: ArrayLike,
    group_sizes: Sequence[int],
    skus: Sequence[str],
    *,
    dtype: DTypeLike = np.float32,
) -> Derivatives:
    """Вес сессии 1, все положительно-отрицательные пары, K=20 и beta=1.

    FP64 доступен для проверки накопленных сумм; LightGBM получает FP32.
    Пустая таблица и группы без пар возвращают нули, не признак успешного обучения.
    """
    labels_array, sizes, keys = _layout(labels, group_sizes, skus)
    return _derivatives(scores, labels_array, sizes, keys, dtype)


def make_objective(
    labels: ArrayLike, group_sizes: Sequence[int], skus: Sequence[str]
) -> Callable[[ArrayLike, "Dataset"], Derivatives]:
    """Связывает производные с порядком строк; Dataset не задаёт повторных весов."""
    labels_array, sizes, keys = _layout(labels, group_sizes, skus)

    def session_lambdas_v1(scores: ArrayLike, dataset: "Dataset") -> Derivatives:
        if dataset.get_weight() is not None or dataset.get_init_score() is not None:
            raise InputValidationError("Dataset.weight и Dataset.init_score должны быть None")
        data_labels, data_groups = dataset.get_label(), dataset.get_group()
        if (
            data_labels is None
            or data_groups is None
            or not np.array_equal(data_labels, labels_array)
            or not np.array_equal(data_groups, sizes)
        ):
            raise InputValidationError("Метки или группы Dataset изменились")
        return _derivatives(scores, labels_array, sizes, keys, np.float32)

    return session_lambdas_v1
