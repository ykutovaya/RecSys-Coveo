"""Неизменяемые структуры доступного префикса и результата запроса."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .common import InputValidationError, id_bytes, require_int64


@dataclass(frozen=True, slots=True)
class Record:
    def __post_init__(self) -> None:
        from .serialization import validate_record

        validate_record(self)


class EventType(StrEnum):
    DETAIL = "detail"
    ADD = "add"
    REMOVE = "remove"
    PURCHASE = "purchase"
    SEARCH = "search"
    PAGE = "page"


class Branch(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    NONE = "none"


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True, slots=True)
class SearchPayload(Record):
    # Некорректные внешние векторы уже представлены отдельной причиной.
    vector: tuple[float, ...] | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class SearchContext(Record):
    block_id: str
    event_time_ms: int
    available_time_ms: int
    payload: SearchPayload


@dataclass(frozen=True, slots=True)
class RawEvent(Record):
    source_namespace: str
    session_id: str
    event_id: str
    event_time_ms: int
    available_time_ms: int
    event_type: EventType
    sku: str | None
    search: SearchPayload | None
    provenance: str
    transport_id: str | None = None
    origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if not self.provenance or any(not origin for origin in self.origins):
            raise InputValidationError("Не задано происхождение события")
        if (
            self.event_type == EventType.SEARCH
            and self.sku is not None
            or self.event_type != EventType.SEARCH
            and self.search is not None
        ):
            raise InputValidationError("Поисковая нагрузка не соответствует типу события")
        if (
            self.event_type
            in {EventType.DETAIL, EventType.ADD, EventType.REMOVE, EventType.PURCHASE}
            and not self.sku
        ):
            raise InputValidationError("Товарное событие требует SKU")


@dataclass(frozen=True, slots=True)
class CanonicalBlock(Record):
    block_id: str
    session_id: str
    event_time_ms: int
    available_time_ms: int
    events: tuple[RawEvent, ...]
    raw_multiplicities: tuple[tuple[str, int], ...]
    search_payloads: tuple[SearchPayload, ...]
    search_multiplicities: tuple[int, ...] = ()

    @property
    def source_namespace(self) -> str:
        return self.events[0].source_namespace

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if not self.events:
            raise InputValidationError("Пустой канонический блок")
        if any(
            e.session_id != self.session_id
            or e.event_time_ms != self.event_time_ms
            or e.source_namespace != self.source_namespace
            or e.available_time_ms > self.available_time_ms
            for e in self.events
        ):
            raise InputValidationError("События не согласованы с блоком")
        ids = [(e.source_namespace, e.event_id) for e in self.events]
        if len(ids) != len(set(ids)):
            raise InputValidationError("Повтор ID в каноническом блоке")
        logical_keys = [(e.event_type, e.sku) for e in self.events]
        if len(logical_keys) != len(set(logical_keys)):
            raise InputValidationError("Повтор канонического действия; используйте кратность")
        if (
            len(self.raw_multiplicities) != len(self.events)
            or {key for key, _ in self.raw_multiplicities} != {e.event_id for e in self.events}
            or any(n <= 0 for _, n in self.raw_multiplicities)
        ):
            raise InputValidationError("Кратность должна быть положительной")
        if len(self.search_multiplicities) != len(self.search_payloads) or any(
            n <= 0 for n in self.search_multiplicities
        ):
            raise InputValidationError("Несогласованные кратности поиска")
        raw_counts = dict(self.raw_multiplicities)
        search_count = sum(
            raw_counts[e.event_id] for e in self.events if e.event_type == EventType.SEARCH
        )
        if search_count != sum(self.search_multiplicities):
            raise InputValidationError("Число поисков не соответствует нагрузкам")


class CartState(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CartItem(Record):
    sku: str
    state: CartState
    updated_at_ms: int | None
    last_unambiguous_ms: int | None = None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.state != CartState.UNKNOWN and (
            self.updated_at_ms is None or self.last_unambiguous_ms != self.updated_at_ms
        ):
            raise InputValidationError("Однозначная корзина требует времени этого действия")
        if self.last_unambiguous_ms is not None and (
            self.updated_at_ms is None or self.last_unambiguous_ms > self.updated_at_ms
        ):
            raise InputValidationError("Будущее однозначное действие корзины")
        if (
            self.state == CartState.UNKNOWN
            and self.last_unambiguous_ms is not None
            and self.last_unambiguous_ms == self.updated_at_ms
        ):
            raise InputValidationError("Конфликтная группа не может быть последней однозначной")


@dataclass(frozen=True, slots=True)
class DetailHistoryItem(Record):
    sku: str
    event_time_ms: int
    count: int

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.count <= 0:
            raise InputValidationError("Число просмотров должно быть положительным")


@dataclass(frozen=True, slots=True)
class PrefixSnapshot(Record):
    session_id: str
    version: int
    event_cutoff_ms: int
    availability_cutoff_ms: int
    blocks: tuple[CanonicalBlock, ...]
    cart: tuple[CartItem, ...]
    search: SearchContext | None
    source_namespace: str = field(kw_only=True)
    detail_history: tuple[DetailHistoryItem, ...] = ()
    full_last_detail: tuple[DetailHistoryItem, ...] = ()
    counters: tuple[tuple[str, str | None, int], ...] = ()
    last_views: tuple[tuple[str, int], ...] = ()
    purchased_skus: tuple[str, ...] = ()
    event_semantics: str = "set_v1"
    first_event_time_ms: int | None = None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        require_int64(self.version, nonnegative=True)
        if any(
            b.session_id != self.session_id
            or b.event_time_ms > self.event_cutoff_ms
            or b.source_namespace != self.source_namespace
            or b.available_time_ms > self.availability_cutoff_ms
            for b in self.blocks
        ):
            raise InputValidationError("Будущий или чужой блок в префиксе")
        times = [b.event_time_ms for b in self.blocks]
        if bool(times) != (self.first_event_time_ms is not None):
            raise InputValidationError("Начало истории обязательно для непустого префикса")
        if self.first_event_time_ms is not None and (
            self.first_event_time_ms > self.event_cutoff_ms
            or any(t < self.first_event_time_ms for t in times)
        ):
            raise InputValidationError("Неверное начало видимой истории")
        if times != sorted(set(times)):
            raise InputValidationError("Порядок временных групп нарушен")
        if any(
            c.updated_at_ms is not None and c.updated_at_ms > self.event_cutoff_ms
            for c in self.cart
        ):
            raise InputValidationError("Будущее состояние корзины")
        if self.event_semantics not in {"set_v1", "multiset_v1"}:
            raise InputValidationError("Неизвестная семантика событий")
        if (
            any(
                v.event_time_ms > self.event_cutoff_ms
                for v in (*self.detail_history, *self.full_last_detail)
            )
            or any(t > self.event_cutoff_ms for _, t in self.last_views)
            or any(n < 0 for _, _, n in self.counters)
        ):
            raise InputValidationError("Несогласованные агрегаты префикса")
        if any(
            c.last_unambiguous_ms is not None
            and (c.updated_at_ms is None or c.last_unambiguous_ms > c.updated_at_ms)
            for c in self.cart
        ):
            raise InputValidationError("Неверное время однозначного действия корзины")
        if self.search is not None and (
            self.search.event_time_ms > self.event_cutoff_ms
            or self.search.available_time_ms > self.availability_cutoff_ms
        ):
            raise InputValidationError("Будущий поисковый контекст")
        for ids in (
            self.purchased_skus,
            tuple(sku for sku, _ in self.last_views),
            tuple(item.sku for item in self.cart),
        ):
            if len(ids) != len(set(ids)):
                raise InputValidationError("Повтор ID в агрегате префикса")
            for identifier in ids:
                id_bytes(identifier)
        observed_purchases = {
            e.sku for b in self.blocks for e in b.events if e.event_type == EventType.PURCHASE
        }
        if not observed_purchases <= set(self.purchased_skus):
            raise InputValidationError("Покупка сохранённого блока отсутствует в полном агрегате")
        keys = [(kind, sku) for kind, sku, _ in self.counters]
        if len(keys) != len(set(keys)):
            raise InputValidationError("Повтор ключа полного счётчика")
        for kind, sku in keys:
            EventType(kind)
            if sku is not None:
                id_bytes(sku)


@dataclass(frozen=True, slots=True)
class RequestContext(Record):
    request_id: str
    session_id: str
    trigger_block_id: str
    t_q: int
    a_q: int
    ordinal: int
    prefix: PrefixSnapshot
    catalog_sha256: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.ordinal <= 0:
            raise InputValidationError("Номер запроса должен быть положительным")
        if (
            self.prefix.session_id != self.session_id
            or self.prefix.event_cutoff_ms != self.t_q
            or self.prefix.availability_cutoff_ms != self.a_q
        ):
            raise InputValidationError("Контекст не соответствует отсечениям префикса")
        if not any(
            b.block_id == self.trigger_block_id and b.event_time_ms == self.t_q
            for b in self.prefix.blocks
        ):
            raise InputValidationError("Нет запускающего блока")


@dataclass(frozen=True, slots=True)
class SourceResult(Record):
    source: str
    skus: tuple[str, ...]
    scores: tuple[float | None, ...]
    available: bool
    quota: int
    reason: str | None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.quota, nonnegative=True)
        if len(self.skus) != len(self.scores) or len(self.skus) != len(set(self.skus)):
            raise InputValidationError("Несогласованный список источника")
        if len(self.skus) > self.quota or (not self.available and self.skus):
            raise InputValidationError("Нарушена квота или доступность источника")


@dataclass(frozen=True, slots=True)
class CandidateResult(Record):
    request_id: str
    sources: tuple[SourceResult, ...]
    skus: tuple[str, ...]
    origins: tuple[tuple[str, tuple[str, ...]], ...]
    branch: Branch
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if len(self.skus) != len(set(self.skus)):
            raise InputValidationError("Повтор SKU в кандидатах")


@dataclass(frozen=True, slots=True)
class FeatureBatch(Record):
    request_id: str
    skus: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric: tuple[tuple[float, ...], ...]
    categorical: tuple[tuple[int, ...], ...]
    masks: tuple[tuple[bool, ...], ...]
    feature_schema_sha256: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        n = len(self.skus)
        if len(set(self.skus)) != n or any(
            len(x) != n for x in (self.numeric, self.categorical, self.masks)
        ):
            raise InputValidationError("Несогласованные строки признаков")
        for matrix, width in (
            (self.numeric, len(self.numeric_columns)),
            (self.categorical, len(self.categorical_columns)),
            (self.masks, len(self.numeric_columns) + len(self.categorical_columns)),
        ):
            if any(len(row) != width for row in matrix):
                raise InputValidationError("Неверная ширина матрицы")


@dataclass(frozen=True, slots=True)
class Explanation(Record):
    sku: str
    branch: Branch
    package_id: str
    template_id: str
    sources: tuple[str, ...]
    support: tuple[tuple[str, float], ...]
    public_text: str


@dataclass(frozen=True, slots=True)
class RequestResult(Record):
    request_id: str
    package_id: str
    candidates: CandidateResult
    delivered_skus: tuple[str, ...]
    scores: tuple[float, ...]
    explanations: tuple[Explanation, ...]
    delivery: DeliveryStatus
    reasons: tuple[str, ...]
    elapsed_ms: int
    trace_ref: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.elapsed_ms, nonnegative=True)
        if self.request_id != self.candidates.request_id:
            raise InputValidationError("Кандидаты другого запроса")
        if (
            len(self.scores) != len(self.delivered_skus)
            or len(set(self.delivered_skus)) != len(self.delivered_skus)
            or not set(self.delivered_skus) <= set(self.candidates.skus)
        ):
            raise InputValidationError("Несогласованная выдача")
        if self.delivery != DeliveryStatus.DELIVERED and self.delivered_skus:
            raise InputValidationError("Недоставленный ответ не содержит выдачу")


@dataclass(frozen=True, slots=True)
class StateTransition(Record):
    session_id: str
    expected_version: int
    next_snapshot: PrefixSnapshot
    accepted_event_ids: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        require_int64(self.expected_version, nonnegative=True)
        if self.next_snapshot.session_id != self.session_id:
            raise InputValidationError("Переход содержит снимок чужой сессии")


class StateStore(Protocol):
    def read(self, session_id: str) -> PrefixSnapshot | None: ...
    def compare_and_swap(self, transition: StateTransition) -> bool: ...


class StateReducer(Protocol):
    def reduce(self, state: PrefixSnapshot, block: CanonicalBlock) -> StateTransition: ...
