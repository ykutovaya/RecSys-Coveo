"""Общие контракты. Обучающие метки доступны только в contracts.training."""

from .common import (
    CONTRACT_VERSION,
    INT64_MAX,
    INT64_MIN,
    ArtifactReader,
    Clock,
    InputValidationError,
    NumericError,
    OutcomeStatus,
    Stage,
    TestStatus,
    ValidationIssue,
    ValidationResult,
    id_bytes,
    require_int64,
)

__all__ = [
    "CONTRACT_VERSION",
    "INT64_MAX",
    "INT64_MIN",
    "ArtifactReader",
    "Clock",
    "InputValidationError",
    "NumericError",
    "OutcomeStatus",
    "Stage",
    "TestStatus",
    "ValidationIssue",
    "ValidationResult",
    "id_bytes",
    "require_int64",
]
