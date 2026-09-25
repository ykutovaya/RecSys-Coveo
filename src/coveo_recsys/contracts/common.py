"""Единицы, ошибки и результаты проверок; только стандартная библиотека."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
CONTRACT_VERSION = "2"


class InputValidationError(ValueError):
    """Ошибка обязательного входа, не пропуск модальности."""


class NumericError(ArithmeticError):
    """Неконечный внутренний результат или переполнение."""


def require_int64(value: int, *, nonnegative: bool = False) -> int:
    if type(value) is not int or not INT64_MIN <= value <= INT64_MAX:
        raise InputValidationError("Ожидалось целое int64")
    if nonnegative and value < 0:
        raise InputValidationError("Счётчик не может быть отрицательным")
    return value


def id_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise InputValidationError("ID должен быть непустой строкой UTF-8")
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise InputValidationError("ID содержит некорректный UTF-8") from exc


class Stage(StrEnum):
    PREFLIGHT = "preflight"
    POST_TRAINING = "post_training"
    POST_LOAD = "post_load"
    PRE_SELECTION = "pre_selection"
    PRE_D = "pre_D"
    POST_D = "post_D"
    PRE_AB = "pre_AB"
    POST_AB = "post_AB"


class TestStatus(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    NOT_USED = "not_used"


class OutcomeStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    code: str
    message: str
    layer: str = "semantic"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    structure_valid: bool
    semantic_valid: bool
    results_complete: bool
    gate_passed: bool
    issues: tuple[ValidationIssue, ...] = ()


class Clock(Protocol):
    def now_ms(self) -> int: ...


class ArtifactReader(Protocol):
    def read_bytes(self, relative_path: str) -> bytes: ...
