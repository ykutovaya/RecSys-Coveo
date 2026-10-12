"""Контракты подготовки без каталога, будущих целей и обучающих весов."""

from dataclasses import dataclass

from .common import InputValidationError, id_bytes
from .records import PrefixSnapshot, Record

PERIOD_NAMES = ("A_G", "A_N", "B", "C_smoke", "C_select", "E", "D")
PREPARATION_VERSION = "3"
EVALUATION_PROTOCOL = "observed_purchases_v1"


@dataclass(frozen=True, slots=True)
class TaskConfig(Record):
    horizon_ms: int = 86_400_000
    gap_ms: int = 1_800_000
    max_requests: int = 1
    detail_history_limit: int = 20
    block_history_limit: int = 20
    query_dimension: int = 2
    event_semantics: str = "set_v1"
    evaluation_protocol: str = EVALUATION_PROTOCOL

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.max_requests != 1 or self.evaluation_protocol != EVALUATION_PROTOCOL:
            raise InputValidationError("observed_purchases_v1 требует ровно один первый запрос")
        if min(
            self.horizon_ms,
            self.gap_ms,
            self.max_requests,
            self.detail_history_limit,
            self.block_history_limit,
            self.query_dimension,
        ) <= 0 or self.event_semantics not in {"set_v1", "multiset_v1"}:
            raise InputValidationError("Неверная конфигурация подготовки")


@dataclass(frozen=True, slots=True)
class PreparedRequest(Record):
    request_id: str
    session_id: str
    trigger_block_id: str
    t_q: int
    a_q: int
    ordinal: int
    prefix: PrefixSnapshot

    @property
    def source_namespace(self) -> str:
        return self.prefix.source_namespace

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if (
            self.ordinal != 1
            or self.prefix.session_id != self.session_id
            or self.prefix.event_cutoff_ms != self.t_q
            or self.prefix.availability_cutoff_ms != self.a_q
        ):
            raise InputValidationError("Несогласованный подготовленный запрос")
        if not any(
            b.block_id == self.trigger_block_id and b.event_time_ms == self.t_q
            for b in self.prefix.blocks
        ):
            raise InputValidationError("Нет запускающего блока")


@dataclass(frozen=True, slots=True)
class ScheduleCursor(Record):
    source_namespace: str
    session_id: str
    first_request_time_ms: int | None = None
    request_count: int = 0

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if self.request_count not in (0, 1) or (self.request_count == 0) != (
            self.first_request_time_ms is None
        ):
            raise InputValidationError("Несогласованный курсор расписания")


@dataclass(frozen=True, slots=True)
class Coverage(Record):
    """Метаданные наблюдаемого диапазона; не условие допуска зарегистрированных целей."""

    source_namespace: str
    start_ms: int
    end_ms: int
    gaps: tuple[tuple[int, int], ...]
    evidence: str
    confirmed_at_ms: int | None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if self.start_ms >= self.end_ms or not self.evidence:
            raise InputValidationError("Неверные границы или отсутствует основание покрытия")
        previous = self.start_ms
        for start, end in self.gaps:
            if not self.start_ms <= start < end <= self.end_ms or start < previous:
                raise InputValidationError("Разрывы должны быть упорядочены и не пересекаться")
            previous = end


@dataclass(frozen=True, slots=True)
class Period(Record):
    name: str
    start_ms: int
    end_ms: int
    sample_size: int

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.name not in PERIOD_NAMES or self.start_ms >= self.end_ms or self.sample_size < 0:
            raise InputValidationError("Неверный период или объём подвыборки")


@dataclass(frozen=True, slots=True)
class PreparationPlan(Record):
    """План подготовки; техническая выборка не даёт допуска к обучению."""

    scope: str
    source_namespace: str
    task: TaskConfig
    periods: tuple[Period, ...]
    coverage: Coverage
    seed: int = 17
    chunk_size: int = 4096
    chunk_bytes: int = 16_777_216
    fan_in: int = 32
    max_group_events: int = 100_000
    max_group_bytes: int = 16_777_216
    max_session_blocks: int = 100_000
    max_session_bytes: int = 16_777_216
    max_sessions: int = 100_000
    diagnostic_sessions: int = 0
    max_seconds: int = 1800
    max_memory_bytes: int = 536_870_912
    max_temp_bytes: int = 268_435_456

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if self.scope not in {"synthetic", "actual"} or self.seed != 17:
            raise InputValidationError("Неверная область плана или seed != 17")
        if not 0 <= self.diagnostic_sessions <= self.max_sessions:
            raise InputValidationError("Неверный размер технической выборки")
        if self.coverage.source_namespace != self.source_namespace:
            raise InputValidationError("Покрытие относится к другому пространству источника")
        if tuple(p.name for p in self.periods) != PERIOD_NAMES or any(
            a.end_ms > b.start_ms for a, b in zip(self.periods, self.periods[1:])
        ):
            raise InputValidationError("Нужны семь упорядоченных непересекающихся периодов")
        if (
            min(
                self.chunk_size,
                self.chunk_bytes,
                self.max_group_events,
                self.max_group_bytes,
                self.max_session_blocks,
                self.max_session_bytes,
                self.max_sessions,
                self.max_seconds,
                self.max_memory_bytes,
                self.max_temp_bytes,
            )
            <= 0
            or self.fan_in < 2
        ):
            raise InputValidationError("Неверные ограничения ресурсов подготовки")
