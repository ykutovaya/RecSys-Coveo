"""Строгая сериализация без pickle, нормализации ID и потери int64."""

import dataclasses
import json
import math
import re
import types
from enum import Enum
from functools import cache
from typing import Any, Callable, TypeVar, cast, get_args, get_origin, get_type_hints

from .common import CONTRACT_VERSION, InputValidationError, id_bytes, require_int64

T = TypeVar("T")


_type_hints: Callable[[Any], dict[str, Any]] = cache(get_type_hints)


def record_version(record_type: type[Any]) -> str:
    # Версия подготовки меняется без изменения неизменённых событий и численных записей.
    return (
        "3"
        if record_type.__name__
        in {
            "TaskConfig",
            "PreparationPlan",
            "PreparedRequest",
            "ScheduleCursor",
            "SessionRecord",
            "TargetRecord",
            "CandidateResult",
            "SourceEvidence",
            "SupervisionRecord",
        }
        else CONTRACT_VERSION
    )


def _convert(value: Any, expected: Any, *, decode: bool = False) -> Any:
    if expected is int:
        return require_int64(value)
    if expected is float:
        try:
            finite = math.isfinite(value) if type(value) in (int, float) else False
        except OverflowError:
            finite = False
        if type(value) not in ((int, float) if decode else (float,)) or not finite:
            raise InputValidationError("Ожидалось конечное число FP64")
        return float(value)
    if expected is str or expected is bool:
        if type(value) is not expected:
            raise InputValidationError(f"Ожидалось {expected}")
        if expected is str:
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise InputValidationError("Строка содержит некорректный UTF-8") from exc
        return value
    origin, args = get_origin(expected), get_args(expected)
    if origin is types.UnionType:
        for candidate in args:
            try:
                return _convert(value, candidate, decode=decode)
            except (ValueError, TypeError):
                pass
        raise InputValidationError(f"Значение не соответствует {expected}")
    if expected is type(None):
        if value is not None:
            raise InputValidationError("Ожидалось null")
        return None
    if origin is tuple:
        if type(value) is not (list if decode else tuple):
            raise InputValidationError("Ожидалась неизменяемая последовательность")
        if len(args) == 2 and args[1] is Ellipsis:
            if decode:
                return tuple(_convert(v, args[0], decode=True) for v in value)
            for item in value:
                _convert(item, args[0])
            return value
        if len(value) != len(args):
            raise InputValidationError("Неверное число элементов")
        if decode:
            return tuple(_convert(v, t, decode=True) for v, t in zip(value, args, strict=True))
        for item, item_type in zip(value, args, strict=True):
            _convert(item, item_type)
        return value
    if dataclasses.is_dataclass(expected):
        if decode:
            hints = _type_hints(expected)
            if not isinstance(value, dict) or set(value) != set(hints):
                raise InputValidationError("Неизвестные или отсутствующие поля")
            factory = cast(Any, expected)
            return factory(**{k: _convert(value[k], t, decode=True) for k, t in hints.items()})
        if type(value) is not expected:
            raise InputValidationError("Неверный тип записи")
        return value
    if isinstance(expected, type) and issubclass(expected, Enum):
        if decode:
            return expected(value)
        if type(value) is not expected:
            raise InputValidationError("Неверный тип перечисления")
        return value
    if type(value) is not expected:
        raise InputValidationError(f"Ожидалось {expected}")
    return value


def validate_record(value: Any) -> None:
    for name, expected in _type_hints(type(value)).items():
        item = getattr(value, name)
        _convert(item, expected)
        if item is not None and (name.endswith("_id") or name == "sku"):
            id_bytes(item)
        if item is not None and (
            name.endswith("_ids") or name in {"skus", "delivered_skus", "purchases"}
        ):
            for identifier in item:
                id_bytes(identifier)
        if name.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", item):
            raise InputValidationError("Ожидался SHA-256 в нижнем регистре")


def _record_data(value: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}


def to_json(record: Any) -> bytes:
    if not dataclasses.is_dataclass(record):
        raise InputValidationError("Ожидалась запись")
    validate_record(record)
    return json.dumps(
        {
            "contract_version": record_version(type(record)),
            "type": type(record).__name__,
            "data": record,
        },
        default=_record_data,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError(f"Повтор JSON ключа: {key}")
        result[key] = value
    return result


def from_json(payload: bytes, record_type: type[T]) -> T:
    try:
        envelope = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        if (
            set(envelope) != {"contract_version", "type", "data"}
            or envelope["contract_version"] != record_version(record_type)
            or envelope["type"] != record_type.__name__
        ):
            raise InputValidationError("Несовместимый конверт записи")
        return cast(T, _convert(envelope["data"], record_type, decode=True))
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise InputValidationError(str(exc)) from exc


def record_schema(record_type: type[Any]) -> dict[str, Any]:
    """JSON Schema конверта; смысловые ограничения проверяет конструктор записи."""

    def describe(expected: Any) -> dict[str, Any]:
        origin, args = get_origin(expected), get_args(expected)
        if origin is types.UnionType:
            return {"anyOf": [describe(t) for t in args]}
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return {"type": "array", "items": describe(args[0])}
            return {
                "type": "array",
                "prefixItems": [describe(t) for t in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        if dataclasses.is_dataclass(expected):
            hints = _type_hints(expected)
            return {
                "type": "object",
                "additionalProperties": False,
                "required": list(hints),
                "properties": {k: describe(t) for k, t in hints.items()},
            }
        if isinstance(expected, type) and issubclass(expected, Enum):
            return {"type": "string", "enum": [x.value for x in expected]}
        if expected is int:
            return {"type": "integer", "minimum": -(2**63), "maximum": 2**63 - 1}
        return {
            "type": {str: "string", float: "number", bool: "boolean", type(None): "null"}[expected]
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["contract_version", "type", "data"],
        "properties": {
            "contract_version": {"const": record_version(record_type)},
            "type": {"const": record_type.__name__},
            "data": describe(record_type),
        },
    }
