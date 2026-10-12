"""Отдельная резервная ветвь: последние просмотры, затем популярность."""

import numpy as np

from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    RequestContext,
    SourceEvidence,
    SourceResult,
)
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.numerics import checked_subtract

from .scores import RetrievalIndex, _check_context
from .sources import _top_m_in_order


def fallback(
    context: RequestContext,
    index: RetrievalIndex,
    config: RetrievalConfig = RetrievalConfig(),
) -> CandidateResult:
    _check_context(context, index)
    catalog = index.package.catalog.skus
    eligible = np.ones(len(catalog), dtype=np.bool_)
    for sku in context.prefix.purchased_skus:
        position = index.positions.get(sku)
        if position is not None:
            eligible[position] = False
    recent = np.zeros(len(catalog), dtype=np.float64)
    recent_mask = np.zeros(len(catalog), dtype=np.bool_)
    for sku, timestamp in context.prefix.last_views:
        if sku in index.positions:
            i = index.positions[sku]
            recent[i] = -checked_subtract(context.t_q, timestamp) / 1000.0
            recent_mask[i] = True
    selected = _top_m_in_order(
        index._sku_order,
        recent,
        eligible & recent_mask,
        config.max_candidates,
        block_size=config.block_size,
    )
    recent_reason = None if context.prefix.last_views else "empty_history"
    sources = [
        SourceResult(
            "recent",
            tuple(catalog[i] for i in selected),
            tuple(float(recent[i]) for i in selected),
            recent_reason is None,
            config.max_candidates,
            recent_reason,
        )
    ]
    remaining = eligible.copy()
    remaining[list(selected)] = False
    popularity = index.package.statistics.popularity
    fill = _top_m_in_order(
        index._sku_order,
        popularity,
        remaining,
        config.max_candidates - len(selected),
        block_size=config.block_size,
    )
    sources.append(
        SourceResult(
            "popularity",
            tuple(catalog[i] for i in fill),
            tuple(float(popularity[i]) for i in fill),
            True,
            config.max_candidates - len(selected),
            None,
        )
    )
    skus = tuple(catalog[i] for i in (*selected, *fill))
    origins = tuple((sku, (source.source,)) for source in sources for sku in source.skus)
    evidence = tuple(
        SourceEvidence(sku, source.source, None, float(value), None, None, None)
        for source in sources
        for sku, value in zip(source.skus, source.scores, strict=True)
        if value is not None
    )
    return CandidateResult(
        context.request_id,
        tuple(sources),
        skus,
        origins,
        Branch.FALLBACK,
        (f"recent:{recent_reason}",) if recent_reason is not None else (),
        evidence,
    )
