"""Объединение источников и отдельное заполнение популярностью."""

from collections.abc import Mapping
from dataclasses import replace

from coveo_recsys.contracts.common import InputValidationError, id_bytes
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    RequestContext,
    SourceEvidence,
)
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig

from .scores import QueryScores, RetrievalIndex, score_request
from .sources import _top_m_in_order, source_evidence, source_results


def retrieve(
    context: RequestContext,
    index: RetrievalIndex,
    config: RetrievalConfig = RetrievalConfig(),
    source_runtime: Mapping[str, str] | None = None,
    *,
    scores: QueryScores | None = None,
) -> CandidateResult:
    if scores is None:
        scores = score_request(context, index, config, source_runtime)
    elif (
        source_runtime is not None
        or scores.context != context
        or scores.index is not index
        or scores.config != config
    ):
        raise InputValidationError("Предварительные баллы не соответствуют запросу или параметрам")
    results = source_results(scores)
    origins: dict[str, list[str]] = {}
    for result in results:
        for sku in result.skus:
            origins.setdefault(sku, []).append(result.source)
    fill_mask = scores.eligible.copy()
    for sku in origins:
        fill_mask[index.positions[sku]] = False
    catalog, popularity = index.package.catalog.skus, index.package.statistics.popularity
    fill = _top_m_in_order(
        index._sku_order,
        popularity,
        fill_mask,
        config.max_candidates - len(origins),
        block_size=config.block_size,
    )
    evidence = list(source_evidence(results, scores))
    for i in fill:
        sku = catalog[i]
        origins[sku] = ["fill"]
        evidence.append(SourceEvidence(sku, "fill", None, float(popularity[i]), None, None, None))
    skus = tuple(sorted(origins, key=id_bytes))
    return CandidateResult(
        context.request_id,
        results,
        skus,
        tuple((sku, tuple(origins[sku])) for sku in skus),
        Branch.PRIMARY,
        tuple(f"{r.source}:{r.reason}" for r in results if r.reason is not None),
        tuple(evidence),
    )


def remove_source(result: CandidateResult, source: str) -> CandidateResult:
    """B08: удалить только данное происхождение, не повторяя заполнение."""
    if source not in {*SOURCES, "fill"}:
        raise InputValidationError("Неизвестное происхождение")
    origins = tuple(
        (sku, tuple(name for name in names if name != source))
        for sku, names in result.origins
        if any(name != source for name in names)
    )
    retained = {sku for sku, _ in origins}
    return replace(
        result,
        sources=tuple(
            replace(item, skus=(), scores=(), available=False, reason="removed")
            if item.source == source
            else item
            for item in result.sources
        ),
        skus=tuple(sku for sku in result.skus if sku in retained),
        origins=origins,
        reasons=(*result.reasons, f"{source}:removed"),
        evidence=tuple(e for e in result.evidence if e.source != source and e.sku in retained),
    )
