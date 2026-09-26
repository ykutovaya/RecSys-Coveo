"""Контракты пакетного контура. Не импортируются обслуживанием."""

from dataclasses import dataclass, field

from .common import InputValidationError, OutcomeStatus, id_bytes, require_int64
from .records import Record


@dataclass(frozen=True, slots=True)
class SessionRecord(Record):
    session_id: str
    admission_ms: int | None
    period: str | None
    admitted: bool
    exclusion_reason: str | None
    source_namespace: str = field(kw_only=True)
    last_event_ms: int = field(kw_only=True)
    target_end_ms: int | None = None
    request_count: int = 0
    telemetry_complete: bool | None = None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        require_int64(self.request_count, nonnegative=True)
        if self.telemetry_complete is not None:
            raise InputValidationError("Полнота телеметрии остаётся неизвестной")
        if self.request_count not in (0, 1) or (self.request_count == 0) != (
            self.admission_ms is None
        ):
            raise InputValidationError("Несогласованный первый запрос")
        if self.admitted != (self.exclusion_reason is None) or (
            self.admitted
            and (self.admission_ms is None or self.target_end_ms is None or self.period is None)
        ):
            raise InputValidationError("Несогласованный календарный допуск")
        if self.admission_ms is not None and self.admission_ms > self.last_event_ms:
            raise InputValidationError("Первый запрос позже последнего события")
        if self.target_end_ms is not None and (
            self.admission_ms is None
            or not self.admission_ms <= self.target_end_ms <= self.last_event_ms
        ):
            raise InputValidationError("Неверные границы целевого окна")


@dataclass(frozen=True, slots=True)
class TargetRecord(Record):
    request_id: str
    purchases: tuple[str, ...]
    status: OutcomeStatus
    horizon_ms: int
    reason: str | None = None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.horizon_ms, nonnegative=True)
        if self.status == OutcomeStatus.UNKNOWN:
            if self.purchases or not self.reason:
                raise InputValidationError("Неизвестный исход требует пустую цель и причину")
        elif not self.purchases or self.reason is not None:
            raise InputValidationError("Зарегистрированная цель требует непустой набор покупок")
        if len(self.purchases) != len(set(self.purchases)):
            raise InputValidationError("Покупки должны быть различными")


@dataclass(frozen=True, slots=True)
class SupervisionRecord(Record):
    request_id: str
    group_id: str
    labels: tuple[int, ...]
    session_request_count: int
    session_weight: float
    pair_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.session_request_count != 1 or any(y not in (0, 1) for y in self.labels):
            raise InputValidationError("Неверные метки или число запросов")
        pairs = sum(self.labels) * (len(self.labels) - sum(self.labels))
        if (
            self.session_weight != 1.0
            or len(self.pair_weights) != pairs
            or any(w != 1.0 / max(1, pairs) for w in self.pair_weights)
        ):
            raise InputValidationError("Веса не соответствуют observed_purchases_v1")


@dataclass(frozen=True, slots=True)
class SessionEvaluation(Record):
    session_id: str
    b_s: int | None
    v_s: int | None
    u_s: int | None
    hit_s: int | None
    first_hit_ms: int | None
    mature: bool
    calendar_cell: int
    slices: tuple[str, ...]
    unknown_reason: str | None = None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        values = (self.hit_s, self.u_s, self.v_s, self.b_s)
        if any(x is None for x in values) and not self.unknown_reason:
            raise InputValidationError("Неизвестный счётчик требует причину")
        known = [x for x in values if x is not None]
        if any(x not in (0, 1) for x in known) or known != sorted(known):
            raise InputValidationError("Нарушено Hit ≤ U ≤ V ≤ b")
        if not self.mature and known:
            raise InputValidationError("Незрелые итоговые компоненты не должны быть известны")


@dataclass(frozen=True, slots=True)
class MetricResult(Record):
    value: float | None
    status: str
    reason: str | None
    unit: str
    denominator: int
    counters: tuple[tuple[str, int], ...]
    interval: tuple[float, float] | None
    interval_valid: bool
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.denominator, nonnegative=True)
        for _, count in self.counters:
            require_int64(count, nonnegative=True)
        if self.status not in {"passed", "failed", "insufficient", "not_run"}:
            raise InputValidationError("Неизвестный статус метрики")
        if self.value is None and not self.reason:
            raise InputValidationError("Нет причины отсутствия метрики")
        if self.interval is not None and self.interval[0] > self.interval[1]:
            raise InputValidationError("Границы интервала переставлены")
        if self.interval_valid and self.interval is None:
            raise InputValidationError("Пригодный интервал должен быть задан")
