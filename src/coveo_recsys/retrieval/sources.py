"""Точные списки источников и минимальная расчётная опора."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from coveo_recsys.contracts.common import (
    InputValidationError,
    NumericError,
    id_bytes,
    require_int64,
)
from coveo_recsys.contracts.historical import RELATIONS
from coveo_recsys.contracts.records import SourceEvidence, SourceResult
from coveo_recsys.contracts.retrieval import SOURCES

from .scores import QueryScores


def exact_top_m(
    skus: Sequence[str],
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    m: int,
    *,
    block_size: int = 1024,
) -> tuple[int, ...]:
    """Индексы точного Top_m: убывание балла, затем UTF-8 ID."""
    keys = tuple(id_bytes(sku) for sku in skus)
    if len(set(keys)) != len(keys):
        raise InputValidationError("Повтор SKU в точном отборе")
    order = np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)
    return _top_m_in_order(order, values, mask, m, block_size=block_size)


def _top_m_in_order(
    sku_order: NDArray[np.intp],
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    m: int,
    *,
    block_size: int,
) -> tuple[int, ...]:
    """Порядок неизменного каталога проверен; баллы и маска проверяются каждый раз."""
    require_int64(m, nonnegative=True)
    if require_int64(block_size, nonnegative=True) == 0:
        raise InputValidationError("Порция должна быть положительной")
    if values.shape != (len(sku_order),) or mask.shape != values.shape or mask.dtype != np.bool_:
        raise InputValidationError("Баллы и маска не соответствуют каталогу")
    if not np.isfinite(values[mask]).all():
        raise NumericError("Неконечный измеримый балл отбора")
    if m == 0:
        return ()
    eligible = sku_order[mask[sku_order]]
    ordered = np.argsort(-np.asarray(values[eligible], dtype=np.float64), kind="stable")
    return tuple(int(i) for i in eligible[ordered[:m]])


def source_results(scores: QueryScores) -> tuple[SourceResult, ...]:
    """Попадание в список определяется квотой, не измеримостью балла."""
    skus = scores.index.package.catalog.skus
    results = []
    for source, quota in zip(SOURCES, scores.config.quotas, strict=True):
        mask = scores.eligible & scores.masks[source]
        if source in RELATIONS:
            mask &= scores.values[source] > 0
        indices = _top_m_in_order(
            scores.index._sku_order,
            scores.values[source],
            mask,
            quota,
            block_size=scores.config.block_size,
        )
        reason = scores.reasons[source]
        results.append(
            SourceResult(
                source,
                tuple(skus[i] for i in indices),
                tuple(float(scores.values[source][i]) for i in indices),
                reason is None,
                quota,
                reason,
            )
        )
    return tuple(results)


def source_evidence(
    results: Sequence[SourceResult], scores: QueryScores
) -> tuple[SourceEvidence, ...]:
    evidence = []
    package = scores.index.package
    for result in results:
        for sku, value in zip(result.skus, result.scores, strict=True):
            if value is None:
                raise InputValidationError("Расчётная опора требует измеримого балла")
            anchor, support, context_id, space = None, None, None, None
            if result.source in RELATIONS:
                target = scores.index.positions[sku]
                contributions = scores.contributions[result.source]
                available = [
                    key for key, targets in contributions.items() if targets.get(target, 0.0) > 0
                ]
                if not available:
                    raise InputValidationError("Нет опоры положительного графового балла")
                anchor = min(
                    available, key=lambda key: (-contributions[key][target], id_bytes(key))
                )
                edge = scores.index.edge(result.source, anchor, target)
                if edge is None:
                    raise InputValidationError("Графовая опора отсутствует в пакете")
                support = edge[1]
            elif result.source in {"description", "image", "search"}:
                modality = "description" if result.source == "search" else result.source
                dimension = getattr(package.content, modality).vectors.shape[1]
                space = f"{modality}:{package.input_hashes['metadata_sha256']}:{dimension}"
                if result.source == "search":
                    if scores.context.prefix.search is None:
                        raise InputValidationError("Поисковая опора не имеет контекста")
                    context_id = scores.context.prefix.search.block_id
            evidence.append(
                SourceEvidence(sku, result.source, anchor, value, support, context_id, space)
            )
    return tuple(evidence)
