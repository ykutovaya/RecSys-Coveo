"""Обучающие группы из уже допущенных таблиц B, без повторного отбора кандидатов."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from coveo_recsys.contracts.common import (
    InputValidationError,
    OutcomeStatus,
    id_bytes,
    require_int64,
)
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.records import Branch, CandidateResult, FeatureBatch
from coveo_recsys.contracts.training import SupervisionRecord, TargetRecord

from .matrix import feature_matrix


@dataclass(frozen=True, slots=True)
class TrainingTable:
    data: NDArray[np.float64]
    labels: NDArray[np.float32]
    group_sizes: tuple[int, ...]
    skus: tuple[str, ...]
    group_keys: tuple[tuple[str, int, str], ...]
    feature_schema_sha256: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    statistics: dict[str, int]
    supervision: tuple[SupervisionRecord, ...]


def build_training_table(
    examples: Iterable[tuple[PreparedRequest, CandidateResult, FeatureBatch, TargetRecord]],
    *,
    feature_schema_sha256: str,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    max_candidates: int,
    max_rows: int,
    max_memory_bytes: int,
    max_disk_bytes: int,
) -> TrainingTable:
    """Допуск периода B проверяет вызывающий код: сами записи периода не содержат.

    Оценка до выделения матрицы: N_B+ * M строк, матрица FP64 и две копии
    (преобразование/обучение), бины Dataset (2 байта), метки, производные,
    ссылки строк и сохранённые парные веса.
    Это плановая оценка, а не замена контроля памяти процесса в команде обучения.
    """
    for limit in (max_candidates, max_rows, max_memory_bytes, max_disk_bytes):
        require_int64(limit, nonnegative=True)
        if limit == 0:
            raise InputValidationError("Пределы ресурсов должны быть положительными")
    columns = numeric_columns + categorical_columns
    if (
        not re.fullmatch(r"[0-9a-f]{64}", feature_schema_sha256)
        or categorical_columns != ("category",)
        or len(set(columns)) != len(columns)
    ):
        raise InputValidationError("Несогласованная схема обучающей таблицы")
    for name in columns:
        id_bytes(name)

    stats = dict.fromkeys(
        (
            "requests",
            "target_sessions",
            "unknown_sessions",
            "empty_candidate_groups",
            "zero_pair_groups",
            "groups",
            "rows",
            "positive_rows",
            "negative_rows",
            "pair_count",
            "target_skus_outside_candidates",
        ),
        0,
    )
    selected: list[tuple[PreparedRequest, CandidateResult, FeatureBatch, TargetRecord]] = []
    seen_requests: set[str] = set()
    seen_sessions: set[str] = set()
    identifier_bytes = 0
    for item in examples:
        if len(item) != 4 or not all(
            isinstance(record, kind)
            for record, kind in zip(
                item, (PreparedRequest, CandidateResult, FeatureBatch, TargetRecord), strict=True
            )
        ):
            raise InputValidationError("Нужны запрос, кандидаты, признаки и отдельная запись цели")
        request, candidates, features, target = item
        if any(record.request_id != request.request_id for record in item[1:]):
            raise InputValidationError("Несогласованные ключи запроса")
        if request.ordinal != 1:
            raise InputValidationError("Обучение допускает только первый запрос сессии")
        if request.request_id in seen_requests or request.session_id in seen_sessions:
            raise InputValidationError("Повтор запроса или сессии в обучении")
        seen_requests.add(request.request_id)
        seen_sessions.add(request.session_id)
        if candidates.branch != Branch.PRIMARY:
            raise InputValidationError("Обучение требует основных кандидатов, не резерв")
        if len(candidates.skus) > max_candidates:
            raise InputValidationError("Превышено число кандидатов; усечение запрещено")
        if candidates.skus != features.skus or candidates.skus != tuple(
            sorted(candidates.skus, key=id_bytes)
        ):
            raise InputValidationError(
                "Состав или UTF-8-порядок кандидатов и признаков различается"
            )
        if (
            features.feature_schema_sha256 != feature_schema_sha256
            or features.numeric_columns != numeric_columns
            or features.categorical_columns != categorical_columns
        ):
            raise InputValidationError("Схема или порядок признаков не соответствует обучению")
        if any(not 0 <= value < 2**31 for row in features.categorical for value in row):
            raise InputValidationError("Категория должна быть неотрицательной int32")
        stats["requests"] += 1
        identifier_bytes += len(id_bytes(request.session_id)) + len(id_bytes(request.request_id))
        if target.status == OutcomeStatus.UNKNOWN:
            stats["unknown_sessions"] += 1
            continue
        stats["target_sessions"] += 1
        purchases = set(target.purchases)
        stats["target_skus_outside_candidates"] += len(purchases.difference(candidates.skus))
        if not candidates.skus:
            stats["empty_candidate_groups"] += 1
            continue
        positive = sum(sku in purchases for sku in candidates.skus)
        negative = len(candidates.skus) - positive
        pairs = positive * negative
        stats["positive_rows"] += positive
        stats["negative_rows"] += negative
        stats["pair_count"] += pairs
        stats["zero_pair_groups"] += int(pairs == 0)
        stats["groups"] += 1
        stats["rows"] += len(candidates.skus)
        identifier_bytes += sum(len(id_bytes(sku)) for sku in candidates.skus)
        selected.append(item)

    upper = stats["target_sessions"] * max_candidates
    width = len(columns)
    stats["row_upper_bound"] = upper
    stats["estimated_memory_bytes"] = (
        upper * (26 * width + 44)
        + stats["pair_count"] * 8
        + stats["requests"] * 256
        + identifier_bytes
    )
    stats["estimated_disk_bytes"] = upper * (8 * width + 4) + identifier_bytes + len(selected) * 64
    if (
        upper > max_rows
        or stats["estimated_memory_bytes"] > max_memory_bytes
        or stats["estimated_disk_bytes"] > max_disk_bytes
    ):
        raise InputValidationError("Превышен бюджет строк или ресурсов обучающей таблицы")

    selected.sort(key=lambda x: (id_bytes(x[0].session_id), x[0].t_q, id_bytes(x[0].request_id)))
    data = np.empty((stats["rows"], width), dtype=np.float64)
    labels = np.empty(stats["rows"], dtype=np.float32)
    group_sizes: list[int] = []
    skus: list[str] = []
    group_keys: list[tuple[str, int, str]] = []
    supervision: list[SupervisionRecord] = []
    offset = 0
    for request, candidates, features, target in selected:
        count = len(candidates.skus)
        purchases = set(target.purchases)
        group_labels = tuple(int(sku in purchases) for sku in candidates.skus)
        data[offset : offset + count] = feature_matrix(
            features, numeric_columns, categorical_columns, feature_schema_sha256
        )
        labels[offset : offset + count] = group_labels
        positives = sum(group_labels)
        pairs = positives * (count - positives)
        supervision.append(
            SupervisionRecord(
                request.request_id,
                request.session_id,
                group_labels,
                1,
                1.0,
                (1.0 / max(1, pairs),) * pairs,
            )
        )
        group_sizes.append(count)
        skus.extend(candidates.skus)
        group_keys.append((request.session_id, request.t_q, request.request_id))
        offset += count
    return TrainingTable(
        data,
        labels,
        tuple(group_sizes),
        tuple(skus),
        tuple(group_keys),
        feature_schema_sha256,
        numeric_columns,
        categorical_columns,
        stats,
        tuple(supervision),
    )
