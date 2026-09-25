"""Независимые формулы допуска и буквальное кодирование ключа отбора."""

import hashlib
import struct


def key(identifier, seed=17):
    raw = identifier.encode("utf-8")
    return hashlib.sha256(struct.pack(">QQ", seed, len(raw)) + raw).digest(), raw


def admitted(t0, period, horizon, gap, coverage, complete, close):
    return (
        period.start_ms <= t0 < period.end_ms
        and t0 + horizon + gap < period.end_ms
        and coverage
        and (period.name != "A_G" or (complete and close is not None and close < period.end_ms))
    )
