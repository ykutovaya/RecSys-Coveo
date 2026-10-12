"""Статическая проверка нового sealed_block_v1, не подтверждение приёма.

Байтовый профиль: один UTF-8 JSON-объект с девятью нормативными полями.
Хешируемая нагрузка — точный фрагмент значения ``events``, от ``[`` до ``]``
включительно, без внешних пробелов. Размер ограничивает весь исходный конверт.
Повторные ключи запрещены; эквивалентная пересериализация не заменяет эти байты.

Событие содержит source_namespace, session_id, event_id, event_time_ms,
event_type, sku. Для search обязателен query_vector (null означает отсутствие),
допустимы product_skus_hash/clicked_skus_hash. Поисковые списки проверяются,
но не включаются в RawEvent. Время доступности и происхождение задаёт проверяющий.

Нет аутентификации, квитанций, проверки уже закрытого времени, ограничений
скорости/сессии или изменения состояния. Подтверждённый повтор должен обходить
проверку нового временного окна в будущем сервисном адаптере. Исторический
импорт не вызывает эту функцию и не наследует транспортные пределы.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from coveo_recsys.artifacts import sha256_bytes
from coveo_recsys.contracts.common import (
    InputValidationError,
    ValidationIssue,
    id_bytes,
    require_int64,
)
from coveo_recsys.contracts.records import EventType, RawEvent, SearchPayload
from coveo_recsys.data.io import vector_payload

_ENVELOPE_FIELDS = {
    "source_namespace",
    "session_id",
    "event_time_ms",
    "block_id",
    "event_count",
    "event_ids",
    "events",
    "payload_sha256",
    "closed",
}
_EVENT_FIELDS = {"source_namespace", "session_id", "event_id", "event_time_ms", "event_type", "sku"}
_SEARCH_FIELDS = {"query_vector", "product_skus_hash", "clicked_skus_hash"}
_SPACE = " \t\r\n"


@dataclass(frozen=True, slots=True)
class EnvelopeLimits:
    max_request_bytes: int = 1_048_576
    max_sku_utf8_bytes: int = 256
    max_transport_events: int = 100
    max_search_list_items: int = 1000
    late_event_limit_ms: int = 1_800_000

    def __post_init__(self) -> None:
        for value in (
            self.max_request_bytes,
            self.max_sku_utf8_bytes,
            self.max_transport_events,
            self.max_search_list_items,
        ):
            require_int64(value)
            if value <= 0:
                raise InputValidationError("Предел должен быть положительным")
        require_int64(self.late_event_limit_ms, nonnegative=True)


@dataclass(frozen=True, slots=True)
class SealedBlock:
    source_namespace: str
    session_id: str
    event_time_ms: int
    block_id: str
    events: tuple[RawEvent, ...]
    payload_sha256: str
    envelope_sha256: str
    envelope_bytes: bytes
    decision_time_ms: int


@dataclass(frozen=True, slots=True)
class EnvelopeValidation:
    block: SealedBlock | None
    issues: tuple[ValidationIssue, ...]

    @property
    def static_valid(self) -> bool:
        return self.block is not None and not self.issues


class _Rejected(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        self.issue = ValidationIssue(path, code, message)
        super().__init__(message)


def _event_time(
    event_time_ms: int, decision_time_ms: int, calendar: tuple[int, int], limits: EnvelopeLimits
) -> None:
    _require(calendar[0] <= event_time_ms < calendar[1], "event_time_ms", "Время вне календаря")
    _require(event_time_ms <= decision_time_ms, "event_time_ms", "Будущее", "timestamp_future")
    _require(
        decision_time_ms - event_time_ms <= limits.late_event_limit_ms,
        "event_time_ms",
        "Запоздание",
        "timestamp_late",
    )


def block_time_issues(
    block: SealedBlock, calendar: tuple[int, int], limits: EnvelopeLimits = EnvelopeLimits()
) -> tuple[ValidationIssue, ...]:
    """Окно нового блока; вызывается только после проверки квитанции повтора."""
    try:
        _event_time(block.event_time_ms, block.decision_time_ms, calendar, limits)
    except _Rejected as exc:
        return (exc.issue,)
    return ()


def inspect_sealed_block(
    envelope: bytes,
    *,
    decision_time_ms: int,
    expected_namespace: str,
    query_dimension: int,
    limits: EnvelopeLimits = EnvelopeLimits(),
) -> EnvelopeValidation:
    """Статическая проверка для сервиса: не приём и не проверка нового окна времени."""
    try:
        if type(envelope) is not bytes or len(envelope) > limits.max_request_bytes:
            return EnvelopeValidation(None, (ValidationIssue("$", "input_limit", "Размер входа"),))
        payload = _events_bytes(envelope.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return EnvelopeValidation(None, (ValidationIssue("$", "invalid_input", "Неверный вход"),))
    return _validate_sealed_block(
        envelope,
        payload,
        decision_time_ms=decision_time_ms,
        allowed_event_time_range=(-(2**63), 2**63 - 1),
        expected_namespace=expected_namespace,
        query_dimension=query_dimension,
        limits=limits,
        check_time=False,
    )


def validate_sealed_block(
    envelope: bytes,
    payload_bytes: bytes,
    *,
    decision_time_ms: int,
    allowed_event_time_range: tuple[int, int],
    expected_namespace: str,
    query_dimension: int,
    limits: EnvelopeLimits = EnvelopeLimits(),
) -> EnvelopeValidation:
    """Прежняя полная проверка нового конверта; статический успех не равен приёму."""
    return _validate_sealed_block(
        envelope,
        payload_bytes,
        decision_time_ms=decision_time_ms,
        allowed_event_time_range=allowed_event_time_range,
        expected_namespace=expected_namespace,
        query_dimension=query_dimension,
        limits=limits,
    )


def _require(condition: bool, path: str, message: str, code: str = "invalid_input") -> None:
    if not condition:
        raise _Rejected(code, path, message)


def _transport_int(token: str) -> int | float:
    # Такое целое заведомо вне FP64 и int64. Не отключаем защитный предел
    # Python для int(): сохраняем непригодность до контекстной проверки поля.
    if len(token.lstrip("-")) > 310:
        return -float("inf") if token.startswith("-") else float("inf")
    return int(token)


def _transport_json(envelope: bytes) -> Any:
    """Синтаксис JSON строгий; пригодность чисел зависит от поля.

    1e999 — число JSON, но непригодная координата FP64. Оно доходит до
    проверки query_vector; обязательные поля ниже имеют строгие типы.
    Не-JSON токены NaN/Infinity и повторные ключи запрещены везде.
    """

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("Повтор JSON-ключа")
            values[key] = value
        return values

    def invalid_constant(value: str) -> Any:
        raise ValueError("Недопустимая константа JSON")

    return json.loads(
        envelope.decode("utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=invalid_constant,
        parse_int=_transport_int,
    )


def _events_bytes(text: str) -> bytes:
    """Извлечь исходный фрагмент; полный JSON отдельно проверяется строго."""
    decoder = json.JSONDecoder(parse_int=_transport_int)
    position = text.index("{") + 1
    while position < len(text):
        while position < len(text) and text[position] in _SPACE:
            position += 1
        if position == len(text) or text[position] == "}":
            break
        key, position = decoder.raw_decode(text, position)
        while position < len(text) and text[position] in _SPACE:
            position += 1
        position += 1  # Двоеточие уже проверено _transport_json.
        while position < len(text) and text[position] in _SPACE:
            position += 1
        start = position
        _, position = decoder.raw_decode(text, position)
        if key == "events":
            return text[start:position].encode("utf-8")
        while position < len(text) and text[position] in _SPACE:
            position += 1
        if position == len(text) or text[position] == "}":
            break
        position += 1  # Запятая перед следующим полем.
    raise ValueError("Отсутствует обязательное поле events")


def _sku(value: Any, path: str, limits: EnvelopeLimits) -> None:
    _require(
        len(id_bytes(value)) <= limits.max_sku_utf8_bytes,
        path,
        "SKU превышает предел UTF-8",
        "input_limit",
    )


def _event(
    value: Any,
    index: int,
    header: dict[str, Any],
    decision_time_ms: int,
    limits: EnvelopeLimits,
    query_dimension: int,
) -> RawEvent:
    path = f"events[{index}]"
    _require(isinstance(value, dict), path, "Событие должно быть объектом")
    _require(_EVENT_FIELDS <= value.keys(), path, "Отсутствуют обязательные поля события")
    kind = EventType(value["event_type"])
    allowed = _EVENT_FIELDS | (_SEARCH_FIELDS if kind == EventType.SEARCH else set())
    _require(value.keys() <= allowed, path, "Неизвестные поля события")
    _require(
        kind != EventType.SEARCH or "query_vector" in value,
        path,
        "Для поиска требуется query_vector, допускается null",
    )
    for name in ("source_namespace", "session_id", "event_id"):
        id_bytes(value[name])
    require_int64(value["event_time_ms"])
    for name in ("source_namespace", "session_id", "event_time_ms"):
        _require(value[name] == header[name], f"{path}.{name}", "Событие не соответствует конверту")
    _require(
        value["event_id"] == header["event_ids"][index],
        f"{path}.event_id",
        "Порядок event_ids не соответствует событиям",
    )
    if value["sku"] is not None:
        _sku(value["sku"], f"{path}.sku", limits)
    search = None
    if kind == EventType.SEARCH:
        _require(value["sku"] is None, f"{path}.sku", "Поиск не является товарным событием")
        for name in ("product_skus_hash", "clicked_skus_hash"):
            if name not in value:
                continue
            values = value[name]
            _require(isinstance(values, list), f"{path}.{name}", "Ожидался поисковый список")
            _require(
                len(values) <= limits.max_search_list_items,
                f"{path}.{name}",
                "Поисковый список превышает предел",
                "input_limit",
            )
            for sku in values:
                _sku(sku, f"{path}.{name}", limits)
        try:
            query_json = json.dumps(value["query_vector"], ensure_ascii=False, allow_nan=False)
        except (ValueError, OverflowError):
            search = SearchPayload(None, "vector_invalid")
        else:
            search = vector_payload(query_json, query_dimension)
    origin = f"sealed_block_v1:{header['_wire_sha256']}:event={index}"
    return RawEvent(
        header["source_namespace"],
        header["session_id"],
        value["event_id"],
        header["event_time_ms"],
        decision_time_ms,
        kind,
        value["sku"],
        search,
        origin,
        value["event_id"],
        (origin,),
    )


def _validate_sealed_block(
    envelope: bytes,
    payload_bytes: bytes,
    *,
    decision_time_ms: int,
    allowed_event_time_range: tuple[int, int],
    expected_namespace: str,
    query_dimension: int,
    limits: EnvelopeLimits = EnvelopeLimits(),
    check_time: bool = True,
) -> EnvelopeValidation:
    """Проверить новый полный блок; при любой ошибке block=None, без части событий.

    decision_time_ms — внешнее серверное время предполагаемого атомарного решения,
    диапазон времени — [start, end). Неверная доверенная конфигурация даёт исключение;
    неверный конверт — отрицательный результат. Успех не удостоверяет отправителя,
    полноту сессии и фактическое принятие; RawEvent лишь подготовлен к этому решению.
    """
    require_int64(decision_time_ms)
    if type(allowed_event_time_range) is not tuple or len(allowed_event_time_range) != 2:
        raise InputValidationError("Нужен полуоткрытый календарный диапазон")
    start, end = (require_int64(v) for v in allowed_event_time_range)
    if start >= end:
        raise InputValidationError("Пустой календарный диапазон")
    id_bytes(expected_namespace)
    require_int64(query_dimension)
    if query_dimension <= 0 or not isinstance(limits, EnvelopeLimits):
        raise InputValidationError("Неверная размерность или пределы")
    try:
        _require(
            type(envelope) is bytes and type(payload_bytes) is bytes, "$", "Нужны точные байты"
        )
        _require(
            len(envelope) <= limits.max_request_bytes,
            "$",
            "Конверт превышает предел",
            "input_limit",
        )
        header = _transport_json(envelope)
        _require(
            isinstance(header, dict) and header.keys() == _ENVELOPE_FIELDS,
            "$",
            "Неверный набор полей sealed_block_v1",
        )
        _require(header["closed"] is True, "closed", "Разрешён только закрытый блок")
        for name in ("source_namespace", "session_id", "block_id"):
            id_bytes(header[name])
        _require(
            header["source_namespace"] == expected_namespace,
            "source_namespace",
            "Неожиданное пространство источника",
        )
        time_ms = require_int64(header["event_time_ms"])
        if check_time:
            _event_time(time_ms, decision_time_ms, (start, end), limits)
        count = require_int64(header["event_count"])
        _require(
            isinstance(header["events"], list) and isinstance(header["event_ids"], list),
            "events",
            "События и ID должны быть массивами",
        )
        _require(
            1 <= len(header["events"]) <= limits.max_transport_events,
            "events",
            "Неверное число сырых событий",
            "input_limit",
        )
        _require(
            count == len(header["events"]) == len(header["event_ids"]),
            "event_count",
            "Число событий не соответствует обоим массивам",
        )
        for event_id in header["event_ids"]:
            id_bytes(event_id)
        _require(len(set(header["event_ids"])) == count, "event_ids", "Повтор ID внутри блока")
        actual_payload = _events_bytes(envelope.decode("utf-8"))
        _require(payload_bytes == actual_payload, "events", "Не совпадают точные байты нагрузки")
        digest = header["payload_sha256"]
        _require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "payload_sha256",
            "Ожидался SHA-256 в нижнем регистре",
        )
        _require(sha256_bytes(payload_bytes) == digest, "payload_sha256", "Неверный хеш нагрузки")
        header["_wire_sha256"] = sha256_bytes(envelope)
        events = tuple(
            _event(v, i, header, decision_time_ms, limits, query_dimension)
            for i, v in enumerate(header["events"])
        )
        block = SealedBlock(
            header["source_namespace"],
            header["session_id"],
            time_ms,
            header["block_id"],
            events,
            digest,
            header["_wire_sha256"],
            envelope,
            decision_time_ms,
        )
        return EnvelopeValidation(block, ())
    except _Rejected as exc:
        return EnvelopeValidation(None, (exc.issue,))
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        return EnvelopeValidation(
            None,
            (
                ValidationIssue(
                    "$",
                    "invalid_input",
                    "Конверт не соответствует байтовому профилю sealed_block_v1",
                ),
            ),
        )
