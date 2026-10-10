"""Общие баллы отбора и признаков; без файлов, целей и часов процесса."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from coveo_recsys.contracts import InputValidationError, NumericError, id_bytes
from coveo_recsys.contracts.historical import (
    HISTORICAL_VERSION,
    RELATIONS,
    HistoricalPackage,
    MetadataRecord,
)
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.records import DetailHistoryItem, RequestContext
from coveo_recsys.contracts.retrieval import (
    BASE_CONFIG_ID,
    NO_METADATA_CONFIG_ID,
    NO_METADATA_SOURCES,
    SOURCES,
    SUPPORTED_CONFIG_IDS,
    RetrievalConfig,
    validate_retrieval_profile,
)
from coveo_recsys.numerics import (
    checked_subtract,
    compensated_sum,
    masked_softmax,
    normalize_external_vector,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ItemContent:
    description: FloatArray
    image: FloatArray
    description_mask: bool
    image_mask: bool
    category: str | None
    price_bucket: int | None


class RetrievalIndex:
    """Однократная привязка проверенного пакета и метаданных наблюдаемых SKU."""

    def __init__(
        self,
        package: HistoricalPackage,
        catalog_sha256: str,
        observed: Mapping[str, MetadataRecord] | None = None,
        *,
        config_id: str = BASE_CONFIG_ID,
    ) -> None:
        if config_id not in SUPPORTED_CONFIG_IDS:
            raise InputValidationError("Неизвестный профиль отбора")
        if package.schema_version != HISTORICAL_VERSION:
            raise InputValidationError("Неизвестная версия исторического пакета")
        if re.fullmatch(r"[0-9a-f]{64}", catalog_sha256) is None:
            raise InputValidationError("Нужен хеш закреплённого каталога")
        self.package = package
        self.config_id = config_id
        self.catalog_sha256 = catalog_sha256
        keys = tuple(id_bytes(sku) for sku in package.catalog.skus)
        if len(set(keys)) != len(keys):
            raise InputValidationError("Повтор SKU в точном отборе")
        self._sku_order = np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)
        self._sku_order.flags.writeable = False
        self.positions = {sku: i for i, sku in enumerate(package.catalog.skus)}
        self._rows: dict[str, ItemContent] = {}
        self._edges: dict[str, dict[str, dict[int, tuple[float, int]]]] = {}
        count = len(self.positions)
        if config_id == BASE_CONFIG_ID:
            content = package.content
            for modality in (content.description, content.image):
                if (
                    modality.vectors.ndim != 2
                    or modality.vectors.shape[0] != count
                    or modality.mask.shape != (count,)
                ):
                    raise InputValidationError("Форма модальностей не соответствует каталогу")
                if not np.isfinite(modality.vectors).all():
                    raise NumericError("Неконечные значения внутри пакета")
            for i, sku in enumerate(package.catalog.skus):
                code = int(content.category_codes[i])
                if not 0 <= code <= len(content.categories):
                    raise InputValidationError("Неизвестный код категории")
                self._rows[sku] = ItemContent(
                    content.description.vectors[i],
                    content.image.vectors[i],
                    bool(content.description.mask[i]),
                    bool(content.image.mask[i]),
                    content.categories[code - 1] if code else None,
                    int(content.price_bucket[i]) if content.price_mask[i] else None,
                )
        if tuple(g.relation for g in package.graphs.relations) != RELATIONS:
            raise InputValidationError("Неверный состав графов")
        for graph in package.graphs.relations:
            if not all(
                np.isfinite(a).all()
                for a in (graph.weights, graph.probabilities, graph.rho, graph.row_sums)
            ):
                raise NumericError("Неконечный граф")
            edges = {}
            for i, sku in enumerate(package.catalog.skus):
                start, end = int(graph.indptr[i]), int(graph.indptr[i + 1])
                edges[sku] = {
                    int(graph.indices[k]): (
                        float(graph.rho[i] * graph.probabilities[k]),
                        int(graph.support[k]),
                    )
                    for k in range(start, end)
                }
            self._edges[graph.relation] = edges
        if (
            package.statistics.popularity.shape != (count,)
            or not np.isfinite(package.statistics.popularity).all()
        ):
            raise NumericError("Некорректная популярность")
        if config_id == BASE_CONFIG_ID:
            for sku, record in (observed or {}).items():
                id_bytes(sku)
                if sku != record.sku or sku in self.positions:
                    raise InputValidationError(
                        "Внешние метаданные не заменяют исторический каталог"
                    )
                description = normalize_external_vector(
                    record.description, content.description.vectors.shape[1]
                )
                image = normalize_external_vector(record.image, content.image.vectors.shape[1])
                self._rows[sku] = ItemContent(
                    description.vector,
                    image.vector,
                    description.mask,
                    image.mask,
                    record.category or None,
                    record.price_bucket,
                )

    def row(self, sku: str) -> ItemContent | None:
        return self._rows.get(sku)

    def edge(self, relation: str, sku: str, target_index: int) -> tuple[float, int] | None:
        return self._edges[relation].get(sku, {}).get(target_index)


def bind_context(prepared: PreparedRequest, index: RetrievalIndex) -> RequestContext:
    context = RequestContext(
        prepared.request_id,
        prepared.session_id,
        prepared.trigger_block_id,
        prepared.t_q,
        prepared.a_q,
        prepared.ordinal,
        prepared.prefix,
        index.catalog_sha256,
    )
    _check_context(context, index)
    return context


def _check_context(context: RequestContext, index: RetrievalIndex) -> None:
    if context.catalog_sha256 != index.catalog_sha256:
        raise InputValidationError("Контекст закреплён за другим каталогом")
    if index.package.snapshot_time_ms >= context.t_q:
        raise InputValidationError("Исторический снимок должен предшествовать запросу")


@dataclass(frozen=True)
class QueryScores:
    context: RequestContext
    index: RetrievalIndex
    config: RetrievalConfig
    values: dict[str, FloatArray]
    masks: dict[str, BoolArray]
    reasons: dict[str, str | None]
    eligible: BoolArray
    history: tuple[DetailHistoryItem, ...]
    weights: FloatArray
    history_available: dict[str, bool]
    search_age_ms: int
    contributions: dict[str, dict[str, dict[int, float]]]


def _history(
    context: RequestContext, config: RetrievalConfig
) -> tuple[tuple[DetailHistoryItem, ...], FloatArray]:
    ordered = sorted(
        context.prefix.detail_history, key=lambda item: (-item.event_time_ms, id_bytes(item.sku))
    )
    remaining = config.history_limit
    selected = []
    for item in ordered:
        if not remaining:
            break
        count = min(remaining, item.count)
        selected.append(replace(item, count=count))
        remaining -= count
        checked_subtract(context.t_q, item.event_time_ms)
    selected.sort(key=lambda item: (item.event_time_ms, id_bytes(item.sku)))
    if not selected:
        return (), np.zeros(0, dtype=np.float64)
    newest = max(item.event_time_ms for item in selected)
    logits = [
        -checked_subtract(newest, item.event_time_ms) / config.tau_history_ms + math.log(item.count)
        for item in selected
    ]
    return tuple(selected), masked_softmax(logits).probabilities


def _dot_blocks(matrix: FloatArray, vector: FloatArray, block_size: int) -> FloatArray:
    values = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, len(values), block_size):
        # Одна и та же редукция координат, независимо от размера порции.
        values[start : start + block_size] = np.sum(
            matrix[start : start + block_size] * vector, axis=1
        )
    if not np.isfinite(values).all():
        raise NumericError("Неконечное внутреннее сходство")
    return values


def score_request(
    context: RequestContext,
    index: RetrievalIndex,
    config: RetrievalConfig = RetrievalConfig(),
    source_runtime: Mapping[str, str] | None = None,
) -> QueryScores:
    _check_context(context, index)
    validate_retrieval_profile(index.config_id, config)
    runtime = dict(source_runtime or {})
    active_sources = NO_METADATA_SOURCES if index.config_id == NO_METADATA_CONFIG_ID else SOURCES
    if any(
        key not in active_sources or not isinstance(reason, str) or not reason
        for key, reason in runtime.items()
    ):
        raise InputValidationError("Неизвестный источник или пустая причина отключения")
    package = index.package
    count = len(package.catalog.skus)
    values = {source: np.zeros(count, dtype=np.float64) for source in SOURCES}
    masks = {source: np.zeros(count, dtype=np.bool_) for source in SOURCES}
    reasons: dict[str, str | None] = dict.fromkeys(SOURCES)
    purchased = set(context.prefix.purchased_skus)
    eligible = np.array([sku not in purchased for sku in package.catalog.skus], dtype=np.bool_)
    history, weights = _history(context, config)
    contributions: dict[str, dict[str, dict[int, float]]] = {}
    for relation in RELATIONS:
        contributions[relation] = {}
        masks[relation][:] = bool(history)
        reasons[relation] = None if history else "empty_history"
        grouped_weights: dict[str, list[float]] = {}
        for item, weight in zip(history, weights, strict=True):
            grouped_weights.setdefault(item.sku, []).append(float(weight))
        for sku, terms in grouped_weights.items():
            alpha = compensated_sum(terms)
            entries = index._edges[relation].get(sku, {})
            weighted = {j: alpha * strength for j, (strength, _) in entries.items()}
            contributions[relation][sku] = weighted
        for weighted in contributions[relation].values():
            for j, value in weighted.items():
                values[relation][j] += value
    history_available = {}
    modalities = ("description", "image") if index.config_id == BASE_CONFIG_ID else ()
    for name in modalities:
        modality = getattr(package.content, name)
        dimension = modality.vectors.shape[1]
        vector_terms = []
        for item, weight in zip(history, weights, strict=True):
            row = index.row(item.sku)
            if row is not None and getattr(row, name + "_mask"):
                vector_terms.append(float(weight) * getattr(row, name))
        vector = np.array(
            [compensated_sum(term[k] for term in vector_terms) for k in range(dimension)],
            dtype=np.float64,
        )
        if not np.isfinite(vector).all():
            raise NumericError("Неконечное представление истории")
        normalized = normalize_external_vector(vector, dimension, epsilon=config.vector_epsilon)
        history_available[name] = normalized.mask
        if normalized.mask:
            masks[name] = modality.mask.copy()
            values[name] = _dot_blocks(modality.vectors, normalized.vector, config.block_size)
        else:
            reasons[name] = "empty_history" if not history else "history_vector_unavailable"
    search_age = 0
    search = context.prefix.search if index.config_id == BASE_CONFIG_ID else None
    if index.config_id == NO_METADATA_CONFIG_ID:
        pass
    elif search is None:
        reasons["search"] = "search_absent"
    elif package.metadata_contract["query_description_compatibility"] not in {
        "publisher_statement",
        "confirmed",
    }:
        reasons["search"] = "incompatible_query_space"
    elif checked_subtract(context.t_q, search.event_time_ms) > config.search_max_age_ms:
        reasons["search"] = "search_expired"
    elif search.payload.reason is not None:
        reasons["search"] = search.payload.reason
    else:
        normalized = normalize_external_vector(
            search.payload.vector,
            package.content.description.vectors.shape[1],
            epsilon=config.vector_epsilon,
        )
        if normalized.mask:
            search_age = checked_subtract(context.t_q, search.event_time_ms)
            values["search"] = _dot_blocks(
                package.content.description.vectors, normalized.vector, config.block_size
            )
            masks["search"] = package.content.description.mask.copy()
        else:
            reasons["search"] = normalized.reason
    for sku, timestamp in context.prefix.last_views:
        if sku in index.positions:
            j = index.positions[sku]
            values["recent"][j] = -checked_subtract(context.t_q, timestamp) / 1000.0
            masks["recent"][j] = True
    if not context.prefix.last_views:
        reasons["recent"] = "empty_history"
    values["popularity"] = package.statistics.popularity.copy()
    masks["popularity"][:] = True
    for source, reason in runtime.items():
        values[source][:] = 0.0
        masks[source][:] = False
        reasons[source] = reason
        if source == "search":
            search_age = 0
    for source in SOURCES:
        if not np.isfinite(values[source]).all():
            raise NumericError("Неконечный балл источника")
        values[source][~masks[source]] = 0.0
    return QueryScores(
        context,
        index,
        config,
        values,
        masks,
        reasons,
        eligible,
        history,
        weights,
        history_available,
        search_age,
        contributions,
    )
