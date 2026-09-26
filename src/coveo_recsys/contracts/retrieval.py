"""Фиксированные источники и пределы точного отбора Э3."""

import math
from dataclasses import dataclass

from .common import InputValidationError, require_int64
from .records import Record

SOURCES = (
    "detail_detail",
    "detail_add",
    "detail_purchase",
    "description",
    "image",
    "search",
    "recent",
    "popularity",
)


@dataclass(frozen=True, slots=True)
class RetrievalConfig(Record):
    quotas: tuple[int, ...] = (50, 50, 50, 40, 30, 50, 20, 10)
    max_candidates: int = 300
    history_limit: int = 20
    tau_history_ms: float = 300_000.0
    search_max_age_ms: int = 600_000
    vector_epsilon: float = 1e-12
    block_size: int = 1024

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        for value in (
            self.max_candidates,
            self.history_limit,
            self.search_max_age_ms,
            self.block_size,
        ):
            if require_int64(value, nonnegative=True) == 0:
                raise InputValidationError("Предел должен быть положительным")
        if (
            len(self.quotas) != len(SOURCES)
            or any(q < 0 for q in self.quotas)
            or sum(self.quotas) > self.max_candidates
        ):
            raise InputValidationError("Квоты не соответствуют источникам и общему бюджету")
        if any(not math.isfinite(v) or v <= 0 for v in (self.tau_history_ms, self.vector_epsilon)):
            raise InputValidationError("Неверное затухание или порог нормы")
        if self.vector_epsilon != 1e-12:
            raise InputValidationError("Базовая конфигурация требует epsilon=1e-12")
