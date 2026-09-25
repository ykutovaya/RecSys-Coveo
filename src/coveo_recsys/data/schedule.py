"""Расписание по готовым видимым префиксам, без состояния и будущих целей."""

import hashlib
import json
from collections.abc import Iterable

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparedRequest, ScheduleCursor, TaskConfig
from coveo_recsys.contracts.records import EventType, PrefixSnapshot
from coveo_recsys.numerics import checked_add, counter_add


def advance_schedule(
    cursor: ScheduleCursor,
    prefix: PrefixSnapshot,
    trigger_block_id: str,
    config: TaskConfig,
) -> tuple[ScheduleCursor, PreparedRequest | None]:
    """Обработать один новый блок; повторные вызовы исключает вызывающий сценарий."""
    if (cursor.source_namespace, cursor.session_id) != (prefix.source_namespace, prefix.session_id):
        raise InputValidationError("Курсор и префикс относятся к разным сессиям")
    if (
        cursor.request_count > config.max_requests
        or prefix.event_semantics != config.event_semantics
    ):
        raise InputValidationError("Курсор или префикс несовместим с конфигурацией")
    triggers = [
        block
        for block in prefix.blocks
        if block.block_id == trigger_block_id and block.event_time_ms == prefix.event_cutoff_ms
    ]
    if len(triggers) != 1:
        raise InputValidationError("Нет единственного полного запускающего блока")
    if (
        cursor.first_request_time_ms is not None
        and prefix.event_cutoff_ms < cursor.first_request_time_ms
    ):
        raise InputValidationError("Расписание нельзя продолжать прошлым временем")
    if prefix.purchased_skus or cursor.request_count == config.max_requests:
        return cursor, None

    actions = {event.event_type for event in triggers[0].events}
    first_time = cursor.first_request_time_ms
    if first_time is None:
        if EventType.DETAIL not in actions:
            return cursor, None
        first_time = prefix.event_cutoff_ms
    elif not actions.intersection((EventType.DETAIL, EventType.SEARCH)):
        return cursor, None
    if prefix.event_cutoff_ms > checked_add(first_time, config.horizon_ms):
        return cursor, None

    ordinal = counter_add(cursor.request_count, 1)
    identity = [
        "prepared_request_v1",
        prefix.source_namespace,
        prefix.session_id,
        trigger_block_id,
        ordinal,
    ]
    request_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    request = PreparedRequest(
        request_id,
        prefix.session_id,
        trigger_block_id,
        prefix.event_cutoff_ms,
        prefix.availability_cutoff_ms,
        ordinal,
        prefix,
    )
    return ScheduleCursor(prefix.source_namespace, prefix.session_id, first_time, ordinal), request


def make_schedule(
    prefixes: Iterable[PrefixSnapshot], config: TaskConfig
) -> tuple[PreparedRequest, ...]:
    """Историческое расписание одной сессии: каждый новый блок ровно один раз.

    Поток префиксов дочитывается после прекращения рекомендаций. Производитель
    сохраняет последующие события независимо от результата этой функции.
    """
    requests = []
    cursor: ScheduleCursor | None = None
    previous_time: int | None = None
    for prefix in prefixes:
        if previous_time is not None and prefix.event_cutoff_ms <= previous_time:
            raise InputValidationError("Повторный блок или неверный порядок префиксов")
        if not prefix.blocks or prefix.blocks[-1].event_time_ms != prefix.event_cutoff_ms:
            raise InputValidationError("Расписание требует префикс после нового полного блока")
        if cursor is None:
            cursor = ScheduleCursor(prefix.source_namespace, prefix.session_id)
        cursor, request = advance_schedule(cursor, prefix, prefix.blocks[-1].block_id, config)
        if request is not None:
            requests.append(request)
        previous_time = prefix.event_cutoff_ms
    return tuple(requests)
