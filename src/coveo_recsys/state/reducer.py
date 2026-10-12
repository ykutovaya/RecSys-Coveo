"""Один reducer для хронологического продолжения и восстановления среза."""

from collections.abc import Iterable
from dataclasses import replace

from coveo_recsys.contracts import INT64_MIN, InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.records import (
    CanonicalBlock,
    CartItem,
    CartState,
    DetailHistoryItem,
    EventType,
    PrefixSnapshot,
    StateTransition,
)
from coveo_recsys.data.search import resolve_search_context
from coveo_recsys.numerics import counter_add


def empty_prefix(source_namespace: str, session_id: str, config: TaskConfig) -> PrefixSnapshot:
    """Начало восстановления, до первого видимого блока."""
    return PrefixSnapshot(
        session_id,
        0,
        INT64_MIN,
        INT64_MIN,
        (),
        (),
        None,
        source_namespace=source_namespace,
        event_semantics=config.event_semantics,
    )


def _history_tail(
    history: tuple[DetailHistoryItem, ...],
    limit: int,
) -> tuple[DetailHistoryItem, ...]:
    # Сначала новые времена; внутри равного времени — меньший UTF-8 SKU.
    ordered = sorted(history, key=lambda item: id_bytes(item.sku))
    ordered.sort(key=lambda item: item.event_time_ms, reverse=True)
    result = []
    remaining = limit
    for item in ordered:
        if remaining == 0:
            break
        retained = min(remaining, item.count)
        result.append(replace(item, count=retained))
        remaining -= retained
    return tuple(sorted(result, key=lambda item: (item.event_time_ms, id_bytes(item.sku))))


def reduce_state(
    state: PrefixSnapshot,
    block: CanonicalBlock,
    config: TaskConfig,
) -> StateTransition:
    """Продолжить состояние строго следующим временем полного блока.

    Поздний/повторный блок нельзя прибавить к уже накопленным агрегатам:
    вызывающий сценарий восстанавливает новый срез через build_prefix.
    receipt_id здесь лишь ссылка на блок, не подтверждение сервисного приёма.
    """
    if (state.session_id, state.source_namespace) != (block.session_id, block.source_namespace):
        raise InputValidationError("Чужая сессия или пространство источника")
    if state.event_semantics != config.event_semantics:
        raise InputValidationError("Нельзя менять семантику накопленного состояния")
    if block.event_time_ms < state.event_cutoff_ms or (
        block.event_time_ms == state.event_cutoff_ms and state.version != 0
    ):
        raise InputValidationError("Блок требует восстановления хронологического среза")

    counters = {(kind, sku): count for kind, sku, count in state.counters}
    if len(counters) != len(state.counters):
        raise InputValidationError("Повтор ключа счётчика в префиксе")
    last_views = dict(state.last_views)
    cart = {item.sku: item for item in state.cart}
    if len(last_views) != len(state.last_views) or len(cart) != len(state.cart):
        raise InputValidationError("Повтор товара в агрегатах префикса")
    purchased = set(state.purchased_skus)
    raw_counts = dict(block.raw_multiplicities)
    details: list[DetailHistoryItem] = []
    cart_actions: dict[str, set[EventType]] = {}
    for event in block.events:
        count = raw_counts[event.event_id] if config.event_semantics == "multiset_v1" else 1
        key = (event.event_type.value, event.sku)
        counters[key] = counter_add(counters.get(key, 0), count)
        if event.event_type == EventType.DETAIL:
            assert event.sku is not None
            details.append(DetailHistoryItem(event.sku, block.event_time_ms, count))
            last_views[event.sku] = block.event_time_ms
        elif event.event_type == EventType.PURCHASE:
            assert event.sku is not None
            purchased.add(event.sku)
        elif event.event_type in (EventType.ADD, EventType.REMOVE):
            assert event.sku is not None
            cart_actions.setdefault(event.sku, set()).add(event.event_type)

    for sku, actions in cart_actions.items():
        previous = cart.get(sku)
        if len(actions) == 2:
            cart[sku] = CartItem(
                sku,
                CartState.UNKNOWN,
                block.event_time_ms,
                previous.last_unambiguous_ms if previous else None,
            )
        else:
            value = CartState.ADD if EventType.ADD in actions else CartState.REMOVE
            cart[sku] = CartItem(sku, value, block.event_time_ms, block.event_time_ms)

    latest_detail = tuple(sorted(details, key=lambda item: id_bytes(item.sku)))
    search = resolve_search_context(block, config)
    ordered_counters = tuple(
        (kind, sku, count)
        for (kind, sku), count in sorted(
            counters.items(),
            key=lambda item: (item[0][0], (item[0][1] or "").encode("utf-8")),
        )
    )
    snapshot = PrefixSnapshot(
        state.session_id,
        counter_add(state.version, 1),
        block.event_time_ms,
        max(state.availability_cutoff_ms, block.available_time_ms),
        (*state.blocks, block)[-config.block_history_limit :],
        tuple(cart[sku] for sku in sorted(cart, key=id_bytes)),
        search if search is not None else state.search,
        source_namespace=state.source_namespace,
        detail_history=_history_tail(
            (*state.detail_history, *latest_detail), config.detail_history_limit
        ),
        full_last_detail=latest_detail or state.full_last_detail,
        counters=ordered_counters,
        last_views=tuple(sorted(last_views.items(), key=lambda item: id_bytes(item[0]))),
        purchased_skus=tuple(sorted(purchased, key=id_bytes)),
        event_semantics=config.event_semantics,
        first_event_time_ms=(
            block.event_time_ms if state.first_event_time_ms is None else state.first_event_time_ms
        ),
    )
    return StateTransition(
        state.session_id,
        state.version,
        snapshot,
        tuple(event.event_id for event in block.events),
        block.block_id,
    )


def build_prefix(
    blocks: Iterable[CanonicalBlock],
    *,
    source_namespace: str,
    session_id: str,
    event_cutoff_ms: int,
    availability_cutoff_ms: int,
    config: TaskConfig,
) -> PrefixSnapshot:
    """Срез полного журнала одной сессии; ограниченного хвоста недостаточно.

    Весь блок становится видимым атомарно. После фильтрации по обоим часам
    тот же reducer воспроизводит хронологию; прежние снимки не изменяются.
    """
    require_int64(event_cutoff_ms)
    require_int64(availability_cutoff_ms)
    state = empty_prefix(source_namespace, session_id, config)
    visible = []
    for block in blocks:
        if (block.source_namespace, block.session_id) != (source_namespace, session_id):
            raise InputValidationError("В журнале среза обнаружена чужая сессия")
        if (
            block.event_time_ms <= event_cutoff_ms
            and block.available_time_ms <= availability_cutoff_ms
        ):
            visible.append(block)
    for block in sorted(visible, key=lambda item: item.event_time_ms):
        state = reduce_state(state, block, config).next_snapshot
    return replace(
        state, event_cutoff_ms=event_cutoff_ms, availability_cutoff_ms=availability_cutoff_ms
    )
