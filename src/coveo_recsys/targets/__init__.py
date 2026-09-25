"""Пакетные цели Э1, физически отдельные от входов обслуживания."""

from .labels import prepare_targets
from .maturity import Observation, observe_session

__all__ = [
    "Observation",
    "observe_session",
    "prepare_targets",
]
