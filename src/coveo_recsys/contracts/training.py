"""Контракты пакетного контура. Не импортируются обслуживанием."""

from dataclasses import dataclass, field

from .common import InputValidationError, OutcomeStatus, id_bytes, require_int64
from .records import Record


@dataclass(frozen=True, slots=True)
class SessionRecord(Record):
    session_id: str
    admission_ms: int | None
    close_ms: int | None
    coverage_complete: bool
    period: str | None
    mature: bool
    exclusion_reason: str | None
    source_namespace: str = field(kw_only=True)
    last_event_ms: int | None = None
    target_end_ms: int | None = None
    request_count: int = 0

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        require_int64(self.request_count, nonnegative=True)
        if self.mature and (not self.coverage_complete or self.admission_ms is None):
            raise InputValidationError("Зрелость окна требует начала и подтверждённого покрытия")
        if (
            self.admission_ms is not None
            and self.last_event_ms is not None
            and self.admission_ms > self.last_event_ms
        ):
            raise InputValidationError("Первый запрос позже последнего события")
        if self.target_end_ms is not None and (
            self.admission_ms is None
            or self.last_event_ms is None
            or not self.admission_ms <= self.target_end_ms <= self.last_event_ms
        ):
            raise InputValidationError("Неверные границы целевого окна")
        if (
            self.close_ms is not None
            and self.last_event_ms is not None
            and self.close_ms < self.last_event_ms
        ):
            raise InputValidationError("Закрытие раньше последнего события")


@dataclass(frozen=True, slots=True)
class TargetRecord(Record):
    request_id: str
    purchases: tuple[str, ...] | None
    mature: bool
    status: OutcomeStatus
    horizon_ms: int
    b_s: int | None
    reason: str | None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.horizon_ms, nonnegative=True)
        if self.status == OutcomeStatus.UNKNOWN:
            if self.purchases is not None or self.b_s is not None or not self.reason:
                raise InputValidationError("Неизвестный исход требует null и причину")
        elif not self.mature or self.b_s not in (0, 1) or self.purchases is None:
            raise InputValidationError("Известный исход требует зрелость и цели")
        if self.purchases is not None and len(self.purchases) != len(set(self.purchases)):
            raise InputValidationError("Покупки должны быть различными")
        if self.purchases and self.b_s == 0:
            raise InputValidationError("Покупка в целях несовместима с b_s=0")


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
        if self.session_request_count <= 0 or any(y not in (0, 1) for y in self.labels):
            raise InputValidationError("Неверные метки или число запросов")
        if self.session_weight < 0 or any(w < 0 for w in self.pair_weights):
            raise InputValidationError("Отрицательный обучающий вес")


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
