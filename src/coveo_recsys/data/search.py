"""Последний поисковый блок: неоднозначность до нормировки, без отката назад."""

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.records import (
    CanonicalBlock,
    EventType,
    SearchContext,
    SearchPayload,
)
from coveo_recsys.numerics import normalize_external_vector


def resolve_search_context(block: CanonicalBlock, config: TaskConfig) -> SearchContext | None:
    """Вернуть None только для блока без поиска.

    Пригодный контекст содержит единичное направление. Различимость задаётся
    исходными FP64-координатами, а не этим направлением. Кратности одного
    значения не создают неоднозначность. Если причин несколько, основная
    выбирается по UTF-8; все исходные причины остаются в CanonicalBlock.
    """
    has_search = any(event.event_type == EventType.SEARCH for event in block.events)
    if has_search != bool(block.search_payloads):
        raise InputValidationError("Поисковые события и нагрузки блока не согласованы")
    if not has_search:
        return None

    vectors: set[tuple[float, ...]] = set()
    reasons: set[str] = set()
    directions: dict[tuple[float, ...], tuple[float, ...]] = {}
    for payload in block.search_payloads:
        if payload.reason is not None:
            reasons.add(payload.reason)
            continue
        normalized = normalize_external_vector(payload.vector, config.query_dimension)
        if not normalized.mask:
            assert normalized.reason is not None
            reasons.add(normalized.reason)
            continue
        assert payload.vector is not None
        # Равенство tuple[float] уже отождествляет -0 и +0.
        vectors.add(payload.vector)
        directions[payload.vector] = tuple(float(x) for x in normalized.vector)

    if reasons:
        resolved = SearchPayload(None, min(reasons, key=lambda value: value.encode("utf-8")))
    elif len(vectors) != 1:
        resolved = SearchPayload(None, "search_ambiguous")
    else:
        resolved = SearchPayload(directions[next(iter(vectors))], None)
    return SearchContext(block.block_id, block.event_time_ms, block.available_time_ms, resolved)
