"""Локальное ideal_deterministic: готовые зависимости, без целей, файлов и часов."""

from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

import numpy as np

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.common import require_int64
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestContext,
    RequestResult,
    SourceEvidence,
    SourceResult,
)
from coveo_recsys.contracts.retrieval import SOURCES
from coveo_recsys.features import FeatureEngine
from coveo_recsys.ranking.load import RankerModel
from coveo_recsys.ranking.predict import predict
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.fallback import fallback
from coveo_recsys.retrieval.scores import _check_context, score_request
from coveo_recsys.retrieval.sources import _top_m_in_order, source_results


def finalize_result(
    context: RequestContext,
    candidates: CandidateResult | None,
    ranked: tuple[tuple[str, float], ...],
    *,
    package_id: str,
    trace_ref: str,
    delivery: DeliveryStatus = DeliveryStatus.DELIVERED,
    reasons: tuple[str, ...] = (),
) -> RequestResult:
    """Зафиксировать выбранную ветвь; недоставка очищает только выдачу.

    В локальном запуске trace_ref ссылается на его явную запись результата,
    а elapsed_ms заполняется измерением вызывающей команды после расчёта.
    """
    if not package_id or not trace_ref:
        raise InputValidationError("Нужны идентификатор пакета и локальная ссылка результата")
    if candidates is None:
        candidates = CandidateResult(context.request_id, (), (), (), Branch.NONE, reasons)
    if candidates.request_id != context.request_id:
        raise InputValidationError("Кандидаты другого запроса")
    if candidates.branch == Branch.NONE:
        if candidates.skus or ranked:
            raise InputValidationError("Без выбранной ветви кандидаты и выдача пусты")
        delivery = DeliveryStatus.NOT_ATTEMPTED
    if (
        len(ranked) > 20
        or len({sku for sku, _ in ranked}) != len(ranked)
        or not {sku for sku, _ in ranked} <= set(candidates.skus)
        or set(candidates.skus) & set(context.prefix.purchased_skus)
    ):
        raise InputValidationError("Выдача или кандидаты нарушают допустимость")
    delivered = ranked if delivery == DeliveryStatus.DELIVERED else ()
    return RequestResult(
        context.request_id,
        package_id,
        candidates,
        tuple(sku for sku, _ in delivered),
        tuple(score for _, score in delivered),
        (),
        delivery,
        reasons,
        0,
        trace_ref,
    )


def _popularity(
    context: RequestContext, engine: FeatureEngine, limit: int | None = None
) -> CandidateResult:
    index, config = engine.index, engine.config
    catalog = index.package.catalog.skus
    purchased = set(context.prefix.purchased_skus)
    eligible = np.array([sku not in purchased for sku in catalog], dtype=np.bool_)
    popularity = index.package.statistics.popularity
    limit = config.max_candidates if limit is None else limit
    positions = _top_m_in_order(
        index._sku_order,
        popularity,
        eligible,
        limit,
        block_size=config.block_size,
    )
    skus = tuple(catalog[position] for position in positions)
    values = tuple(float(popularity[position]) for position in positions)
    source = SourceResult("popularity", skus, values, True, limit, None)
    return CandidateResult(
        context.request_id,
        (source,),
        skus,
        tuple((sku, ("popularity",)) for sku in skus),
        Branch.PRIMARY,
        (),
        tuple(
            SourceEvidence(sku, "popularity", None, value, None, None, None)
            for sku, value in zip(skus, values, strict=True)
        ),
    )


def _short_lists(sources: tuple[SourceResult, ...], depth: int) -> dict[str, Any]:
    return {
        source.source: {
            "skus": source.skus[:depth] if source.available else None,
            "depth": depth,
            "truncated": len(source.skus) > depth if source.available else None,
            "reason": source.reason,
        }
        for source in sources
    }


def recommend(
    context: RequestContext,
    engine: FeatureEngine,
    model: RankerModel | None,
    *,
    policy: str = "GCS+Rk",
    package_id: str,
    trace_ref: str,
    diagnostics: MutableMapping[str, Any] | None = None,
    diagnostic_depth: int = 50,
) -> RequestResult:
    """Первый запрос, одна из GCS+Rk/P/R; None явно выбирает резерв GCS+Rk.

    Ошибки совместимости или вычислений передаются вызывающему коду. Реальный
    запуск GCS+Rk обязан заранее загрузить и проверить готовую модель.
    """
    if policy not in {"GCS+Rk", "P", "R"}:
        raise InputValidationError("Неизвестная локальная политика")
    if context.ordinal != 1 or context.t_q != context.a_q:
        raise InputValidationError("Нужен первый запрос в режиме ideal_deterministic")
    if require_int64(diagnostic_depth, nonnegative=True) == 0:
        raise InputValidationError("Глубина диагностики должна быть положительной")
    _check_context(context, engine.index)
    diagnostic_sources: tuple[SourceResult, ...] = ()
    features = None
    if policy == "GCS+Rk" and model is not None:
        scores = score_request(context, engine.index, engine.config)
        candidates = retrieve(context, engine.index, engine.config, scores=scores)
        features = engine.build(context, candidates, scores=scores)
        ranked = predict(model, features, catalog_sha256=context.catalog_sha256)
        reasons = candidates.reasons
        if diagnostics is not None:
            diagnostic_config = replace(
                engine.config,
                quotas=(diagnostic_depth + 1,) * len(SOURCES),
                max_candidates=(diagnostic_depth + 1) * len(SOURCES),
            )
            diagnostic_sources = source_results(replace(scores, config=diagnostic_config))
    else:
        if policy == "P":
            candidates = _popularity(context, engine)
        else:
            candidates = fallback(context, engine.index, engine.config)
            if policy == "R":
                candidates = replace(candidates, branch=Branch.PRIMARY)
        values = {item.sku: item.score for item in candidates.evidence}
        ranked = tuple((sku, values[sku]) for sku in candidates.skus[:20])
        reasons = candidates.reasons
        if policy == "GCS+Rk":
            reasons = (*reasons, "model_unavailable")
        if diagnostics is not None:
            diagnostic_sources = _popularity(context, engine, diagnostic_depth + 1).sources
            if policy != "P":
                diagnostic_config = replace(
                    engine.config,
                    max_candidates=max(diagnostic_depth + 1, engine.config.max_candidates),
                )
                recent = fallback(context, engine.index, diagnostic_config).sources[0]
                diagnostic_sources = (recent, *diagnostic_sources)
    result = finalize_result(
        context,
        candidates,
        ranked,
        package_id=package_id,
        trace_ref=trace_ref,
        reasons=reasons,
    )
    if diagnostics is not None:
        diagnostics["source_lists"] = _short_lists(diagnostic_sources, diagnostic_depth)
        diagnostics["features"] = features
    return result
