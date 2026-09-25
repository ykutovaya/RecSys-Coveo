"""Принадлежность целой сессии, допуск и отбор без чтения покупок."""

import hashlib
import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.preparation import PERIOD_NAMES, Period, TaskConfig
from coveo_recsys.numerics import checked_add


@dataclass(frozen=True, slots=True)
class Admission:
    period: str | None
    eligible: bool
    admitted: bool
    reason: str | None


def validate_periods(periods: Sequence[Period]) -> None:
    if tuple(p.name for p in periods) != PERIOD_NAMES or any(
        a.end_ms > b.start_ms for a, b in zip(periods, periods[1:])
    ):
        raise InputValidationError("Нужны семь упорядоченных непересекающихся периодов")


def period_for(first_request_ms: int | None, periods: Sequence[Period]) -> Period | None:
    validate_periods(periods)
    if first_request_ms is None:
        return None
    require_int64(first_request_ms)
    return next((p for p in periods if p.start_ms <= first_request_ms < p.end_ms), None)


def admit(
    first_request_ms: int | None,
    periods: Sequence[Period],
    config: TaskConfig,
    *,
    last_event_ms: int | None,
) -> Admission:
    """Допуск по календарю и опубликованному хвосту, независимо от покупок и покрытия."""
    period = period_for(first_request_ms, periods)
    if first_request_ms is None:
        return Admission(None, False, False, "no_request")
    if last_event_ms is None:
        raise InputValidationError("Не указано последнее наблюдаемое событие")
    require_int64(last_event_ms)
    if last_event_ms < first_request_ms:
        raise InputValidationError("Последнее наблюдаемое событие раньше первого запроса")
    if period is None:
        return Admission(None, False, False, "outside_periods")
    boundary = checked_add(checked_add(first_request_ms, config.horizon_ms), config.gap_ms)
    if boundary >= period.end_ms:
        return Admission(period.name, True, False, "window_boundary")
    if period.name == "A_G" and checked_add(last_event_ms, config.gap_ms) >= period.end_ms:
        return Admission(period.name, True, False, "late_A_G_tail")
    return Admission(period.name, True, True, None)


def session_key(session_id: str, seed: int = 17) -> tuple[bytes, bytes]:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise InputValidationError("seed должен быть uint64")
    encoded = id_bytes(session_id)
    payload = seed.to_bytes(8, "big") + len(encoded).to_bytes(8, "big") + encoded
    return hashlib.sha256(payload).digest(), encoded


def sample_sessions(
    session_ids: Iterable[str], n: int, *, seed: int = 17, max_sessions: int = 100_000
) -> tuple[str, ...]:
    """ID одного пространства и периода; вызывающий сохраняет все их строки."""
    require_int64(n, nonnegative=True)
    require_int64(max_sessions, nonnegative=True)
    if max_sessions == 0:
        raise InputValidationError("Нулевой предел числа сессий")
    session_key("validate_seed", seed)
    seen: set[str] = set()
    for identifier in session_ids:
        id_bytes(identifier)
        if identifier in seen:
            raise InputValidationError("Повтор ID в отборе сессий")
        seen.add(identifier)
        if len(seen) > max_sessions:
            raise InputValidationError("Число сессий превысило объявленный предел")
    return tuple(heapq.nsmallest(n, seen, key=lambda s: session_key(s, seed)))
