"""Матрица модели без присоединения целей и изменения признаков Э3."""

import numpy as np
from numpy.typing import NDArray

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import FeatureBatch


def feature_matrix(
    batch: FeatureBatch,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    feature_schema_sha256: str,
) -> NDArray[np.float64]:
    if (
        batch.feature_schema_sha256 != feature_schema_sha256
        or batch.numeric_columns != numeric_columns
        or batch.categorical_columns != categorical_columns
        or categorical_columns != ("category",)
        or len(set(numeric_columns + categorical_columns))
        != len(numeric_columns + categorical_columns)
    ):
        raise InputValidationError("Схема или порядок признаков не соответствует модели")
    if any(not 0 <= value < 2**31 for row in batch.categorical for value in row):
        raise InputValidationError("Категория должна быть неотрицательной int32")
    rows = len(batch.skus)
    numeric = np.asarray(batch.numeric, dtype=np.float64).reshape(rows, len(numeric_columns))
    categorical = np.asarray(batch.categorical, dtype=np.float64).reshape(rows, 1)
    result = np.ascontiguousarray(np.concatenate((numeric, categorical), axis=1))
    if not np.isfinite(result).all():
        raise InputValidationError("Неконечные признаки ранжировщика")
    return result
