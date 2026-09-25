"""Строгая сериализация без pickle, нормализации ID и потери int64."""

import dataclasses
import json
import math
import re
import types
from enum import Enum
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

from .common import CONTRACT_VERSION, InputValidationError, id_bytes, require_int64

T = TypeVar("T")


def _convert(value: Any, expected: Any, *, decode: bool = False) -> Any:
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
            return tuple(_convert(v, args[0], decode=decode) for v in value)
        if len(value) != len(args):
            raise InputValidationError("Неверное число элементов")
        return tuple(_convert(v, t, decode=decode) for v, t in zip(value, args, strict=True))
    if dataclasses.is_dataclass(expected):
        if decode:
            if not isinstance(value, dict) or set(value) != set(get_type_hints(expected)):
                raise InputValidationError("Неизвестные или отсутствующие поля")
            factory = cast(Any, expected)
            return factory(
                **{
                    k: _convert(value[k], t, decode=True)
                    for k, t in get_type_hints(expected).items()
                }
            )
        if type(value) is not expected:
            raise InputValidationError("Неверный тип записи")
        return value
    if isinstance(expected, type) and issubclass(expected, Enum):
        if decode:
            return expected(value)
        if type(value) is not expected:
            raise InputValidationError("Неверный тип перечисления")
        return value
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
    if type(value) is not expected:
        raise InputValidationError(f"Ожидалось {expected}")
    if expected is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise InputValidationError("Строка содержит некорректный UTF-8") from exc
    return value


def validate_record(value: Any) -> None:
    for name, expected in get_type_hints(type(value)).items():
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


def to_json(record: Any) -> bytes:
    if not dataclasses.is_dataclass(record):
        raise InputValidationError("Ожидалась запись")
    validate_record(record)
    return json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "type": type(record).__name__,
            "data": dataclasses.asdict(cast(Any, record)),
        },
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
            or envelope["contract_version"] != CONTRACT_VERSION
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
            hints = get_type_hints(expected)
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
            "contract_version": {"const": CONTRACT_VERSION},
            "type": {"const": record_type.__name__},
            "data": describe(record_type),
        },
    }
