"""Записи общих таблиц выдачи и последующего присоединения целей."""

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from .common import InputValidationError
from .records import Record, RequestResult
from .training import SessionEvaluation

POLICIES = ("GCS+Rk", "P", "R")


def _diagnostic_json(payload: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InputValidationError("Повтор ключа диагностики")
            result[key] = value
        return result

    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise InputValidationError("Неконечное значение диагностики")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    try:
        value = json.loads(payload, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise InputValidationError("Диагностика должна быть JSON-объектом")
        finite(value)
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise InputValidationError(str(exc)) from exc


class _ParsedDiagnostic:
    # Это слот, а не поле dataclass: схема, wire JSON и равенство не меняются.
    __slots__ = ("_parsed_json",)

    @property
    def parsed_json(self) -> dict[str, Any]:
        """Строго проверенный словарь записи; потребители используют его только для чтения."""
        return cast(dict[str, Any], object.__getattribute__(self, "_parsed_json"))


@dataclass(frozen=True, slots=True)
class ReplayRecord(Record, _ParsedDiagnostic):
    session_id: str
    policy: str
    result: RequestResult
    diagnostics_json: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.policy not in POLICIES:
            raise InputValidationError("Незапланированная политика")
        object.__setattr__(self, "_parsed_json", _diagnostic_json(self.diagnostics_json))


@dataclass(frozen=True, slots=True)
class EvaluationRecord(Record, _ParsedDiagnostic):
    session_id: str
    policy: str
    evaluation: SessionEvaluation
    details_json: str

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.policy not in POLICIES or self.evaluation.session_id != self.session_id:
            raise InputValidationError("Несовместимая сессионная оценка")
        object.__setattr__(self, "_parsed_json", _diagnostic_json(self.details_json))
