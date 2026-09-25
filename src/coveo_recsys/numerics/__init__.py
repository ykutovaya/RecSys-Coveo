"""Чистые численные примитивы Э0-03; не реализация полного B01/B05."""

from .groups import MaskedValue, MultiplicitySemantics, last_group_max, last_group_mean
from .integers import (
    checked_add,
    checked_multiply,
    checked_subtract,
    counter_add,
    seconds_to_milliseconds,
)
from .ordering import sort_ids, sort_scored_ids
from .stable import (
    SoftmaxResult,
    compensated_sum,
    log_softmax,
    logsumexp,
    masked_softmax,
    sigmoid,
    softplus,
)
from .vectors import NormalizedVector, normalize_external_vector, scaled_norm

__all__ = [
    "MaskedValue",
    "MultiplicitySemantics",
    "NormalizedVector",
    "SoftmaxResult",
    "checked_add",
    "checked_multiply",
    "checked_subtract",
    "compensated_sum",
    "counter_add",
    "last_group_max",
    "last_group_mean",
    "log_softmax",
    "logsumexp",
    "masked_softmax",
    "normalize_external_vector",
    "scaled_norm",
    "seconds_to_milliseconds",
    "sigmoid",
    "softplus",
    "sort_ids",
    "sort_scored_ids",
]
