from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.preparation import PERIOD_NAMES, Period, TaskConfig
from coveo_recsys.data.periods import (
    admit,
    period_for,
    sample_sessions,
    session_key,
    validate_periods,
)
from tests.reference.e1_periods_oracle import key

PERIODS = tuple(Period(name, i * 100, (i + 1) * 100, 2) for i, name in enumerate(PERIOD_NAMES))
CONFIG = TaskConfig(horizon_ms=20, gap_ms=10)


def decision(t0, **kwargs):
    values = dict(coverage_complete=True, session_complete=True, close_ms=None)
    values.update(kwargs)
    return admit(t0, PERIODS, CONFIG, **values)


def test_exact_seven_half_open_parts_and_no_split_session():
    validate_periods(PERIODS)
    assert period_for(None, PERIODS) is None
    assert period_for(-1, PERIODS) is None
    assert period_for(700, PERIODS) is None
    for i, period in enumerate(PERIODS):
        assert period_for(i * 100, PERIODS) == period
        assert period_for(i * 100 + 99, PERIODS) == period
    for parts in (PERIODS[:-1], PERIODS[::-1], (replace(PERIODS[0], end_ms=101), *PERIODS[1:])):
        with pytest.raises(InputValidationError):
            validate_periods(parts)


def test_strict_maturity_and_A_G_full_session():
    assert decision(269).admitted
    assert decision(270).reason == "maturity_boundary"
    assert decision(271).reason == "maturity_boundary"
    assert decision(0, close_ms=99).admitted
    assert not decision(0, close_ms=100).admitted
    assert not decision(0, close_ms=99, session_complete=False).admitted
    assert not decision(0, close_ms=None).admitted
    # Длинная B-сессия с подтверждённым целевым окном ещё может быть открыта.
    assert decision(200, session_complete=False, close_ms=None).admitted
    assert decision(200, coverage_complete=False).reason == "coverage_incomplete"
    assert decision(None).reason == "no_request"
    assert decision(-1).reason == "outside_periods"
    assert decision(100).period == "A_N"  # Не переносится в B.


def test_checked_times_and_explicit_proofs():
    with pytest.raises(InputValidationError):
        decision(0, coverage_complete=1)
    with pytest.raises(InputValidationError):
        decision(200, close_ms=199)
    distant = (*PERIODS[:-1], Period("D", 600, 2**63 - 1, 2))
    with pytest.raises(NumericError):
        admit(
            2**63 - 2,
            distant,
            CONFIG,
            coverage_complete=True,
            session_complete=False,
            close_ms=None,
        )


def test_sampling_exact_bytes_nested_and_no_purchase_argument():
    ids = ("A", "a", "é", "é", "商品", "s1")
    assert [session_key(s) for s in ids] == [key(s) for s in ids]
    expected = tuple(sorted(ids, key=key))
    assert sample_sessions(ids, 3) == expected[:3]
    assert sample_sessions(reversed(ids), 4)[:3] == expected[:3]
    assert sample_sessions(ids, 0) == ()
    assert sample_sessions(ids, 999) == expected
    for values in (("s", "s"), ("",)):
        with pytest.raises(InputValidationError):
            sample_sessions(values, 1)
    with pytest.raises(InputValidationError):
        sample_sessions(ids, 1, max_sessions=2)
    for seed in (-1, 2**64, True):
        with pytest.raises(InputValidationError):
            session_key("s", seed)
