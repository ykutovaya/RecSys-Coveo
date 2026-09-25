"""Полнота исторического наблюдения; конец файла не является свидетельством."""

from collections.abc import Sequence
from dataclasses import dataclass

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import Coverage, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.numerics import checked_add


@dataclass(frozen=True, slots=True)
class Observation:
    coverage_complete: bool
    session_complete: bool
    close_ms: int | None
    target_end_ms: int | None
    mature: bool
    reason: str | None


def coverage_reason(coverage: Coverage, start_ms: int, end_ms: int) -> str | None:
    """Проверить свидетельство непрерывности полуоткрытого [start_ms,end_ms)."""
    if start_ms >= end_ms:
        raise InputValidationError("Непустой интервал покрытия обязателен")
    if coverage.confirmed_at_ms is None:
        return "coverage_unconfirmed"
    if start_ms < coverage.start_ms or end_ms > coverage.end_ms:
        return "coverage_range"
    if any(start < end_ms and end > start_ms for start, end in coverage.gaps):
        return "coverage_gap"
    return None


def validate_session(blocks: Sequence[CanonicalBlock], coverage: Coverage) -> None:
    """Одна полная наблюдаемая группа сессии, строго по времени, только a=t.

    При задержках Coverage не содержит последовательности продвижения границы
    полноты. Этот режим явно отклоняется, а не выдаёт недоказанное закрытие.
    Ограничение памяти и группировку обеспечивает вызывающий сценарий.
    """
    if not blocks:
        raise InputValidationError("Нужна непустая сессия")
    identity = (blocks[0].source_namespace, blocks[0].session_id)
    if coverage.source_namespace != identity[0]:
        raise InputValidationError("Покрытие другого пространства источника")
    previous: int | None = None
    block_ids: set[str] = set()
    for block in blocks:
        if (block.source_namespace, block.session_id) != identity:
            raise InputValidationError("Смешаны сессии или пространства источника")
        if previous is not None and block.event_time_ms <= previous:
            raise InputValidationError("Требуются полные блоки в строгом временном порядке")
        if block.block_id in block_ids:
            raise InputValidationError("Повтор ID блока")
        if block.available_time_ms != block.event_time_ms or any(
            event.available_time_ms != event.event_time_ms for event in block.events
        ):
            raise InputValidationError("Зрелость при задержках требует истории границы полноты")
        block_ids.add(block.block_id)
        previous = block.event_time_ms


def observe_session(
    blocks: Sequence[CanonicalBlock],
    first_request_ms: int | None,
    coverage: Coverage,
    config: TaskConfig,
) -> Observation:
    """Различать зрелость окна цели и закрытие всей сессии.

    До t0 включается весь наблюдаемый префикс: его разрыв может скрыть покупку
    или более ранний подходящий detail. Для длинной сессии разрыв после H+g
    не отменяет зрелость окна, но запрещает утверждение о полноте всей сессии.
    confirmed_at_ms удостоверяет наличие свидетельства; в ideal a=t момент
    закрытия — last+g, не время получения свидетельства.
    """
    validate_session(blocks, coverage)
    first, last = blocks[0].event_time_ms, blocks[-1].event_time_ms
    close_boundary = checked_add(last, config.gap_ms)
    close_ms = close_boundary if coverage_reason(coverage, last, close_boundary) is None else None
    session_complete = coverage_reason(coverage, first, close_boundary) is None
    if first_request_ms is None:
        return Observation(False, session_complete, close_ms, None, False, "no_request")
    if not first <= first_request_ms <= last:
        raise InputValidationError("Первый запрос вне наблюдаемой сессии")
    horizon = checked_add(first_request_ms, config.horizon_ms)
    window_end = checked_add(horizon, config.gap_ms)
    reason = coverage_reason(coverage, first, window_end)
    complete = reason is None
    return Observation(
        complete,
        session_complete,
        close_ms,
        min(horizon, last) if complete else None,
        complete,
        reason,
    )
