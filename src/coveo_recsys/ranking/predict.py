"""Основная выдача по сырым баллам; резерв здесь не ранжируется."""

import numpy as np

from coveo_recsys.contracts import InputValidationError, NumericError, id_bytes
from coveo_recsys.contracts.records import FeatureBatch

from .load import RankerModel
from .matrix import feature_matrix


def predict(
    model: RankerModel, batch: FeatureBatch, *, catalog_sha256: str
) -> tuple[tuple[str, float], ...]:
    summary = model.summary
    if catalog_sha256 != summary["identity"]["catalog_sha256"]:
        raise InputValidationError("Каталог запроса не соответствует модели")
    if batch.skus != tuple(sorted(batch.skus, key=id_bytes)):
        raise InputValidationError("Кандидаты должны следовать порядку UTF-8")
    matrix = feature_matrix(
        batch,
        tuple(summary["numeric_columns"]),
        tuple(summary["categorical_columns"]),
        summary["feature_schema_sha256"],
    )
    if not batch.skus:
        return ()
    scores = np.asarray(
        model.booster.predict(
            matrix, raw_score=True, num_iteration=summary["actual_tree_count"], num_threads=1
        ),
        dtype=np.float64,
    )
    if scores.shape != (len(batch.skus),):
        raise InputValidationError("Число баллов не соответствует кандидатам")
    if not np.isfinite(scores).all():
        raise NumericError("Неконечные баллы ранжировщика")
    order = sorted(
        range(len(scores)), key=lambda index: (-scores[index], id_bytes(batch.skus[index]))
    )
    return tuple((batch.skus[index], float(scores[index])) for index in order[:20])
