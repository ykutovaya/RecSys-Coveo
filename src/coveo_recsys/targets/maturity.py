"""Окно зарегистрированных событий без предположения о полноте телеметрии."""

from collections.abc import Sequence
from dataclasses import dataclass

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import Coverage, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.numerics import checked_add


@dataclass(frozen=True, slots=True)
class Observation:
    last_event_ms: int
    target_end_ms: int | None


def validate_session(blocks: Sequence[CanonicalBlock], coverage: Coverage) -> None:
    """Одна непустая сессия, полные блоки в строгом порядке, историческое a=t."""
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
            raise InputValidationError(
                "Подготовка зарегистрированных целей поддерживает только a=t"
            )
        block_ids.add(block.block_id)
        previous = block.event_time_ms


def observe_session(
    blocks: Sequence[CanonicalBlock],
    first_request_ms: int | None,
    coverage: Coverage,
    config: TaskConfig,
) -> Observation:
    """Ограничить цель горизонтом и последним опубликованным событием."""
    validate_session(blocks, coverage)
    first, last = blocks[0].event_time_ms, blocks[-1].event_time_ms
    if first_request_ms is None:
        return Observation(last, None)
    if not first <= first_request_ms <= last:
        raise InputValidationError("Первый запрос вне наблюдаемой сессии")
    return Observation(last, min(checked_add(first_request_ms, config.horizon_ms), last))
