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
    values = dict(last_event_ms=t0)
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


@pytest.mark.parametrize("t0, allowed", [(269, True), (270, False), (271, False)])
def test_strict_target_window_boundary(t0, allowed):
    result = decision(t0)
    assert result.eligible
    assert result.admitted is allowed
    assert result.reason == (None if allowed else "window_boundary")


@pytest.mark.parametrize("last_event_ms, allowed", [(89, True), (90, False), (91, False)])
def test_A_G_full_observed_tail(last_event_ms, allowed):
    result = decision(0, last_event_ms=last_event_ms)
    assert result.eligible
    assert result.admitted is allowed
    assert result.reason == (None if allowed else "late_A_G_tail")


def test_non_A_G_admission_does_not_require_tail_within_period():
    assert decision(200, last_event_ms=700).admitted


def test_no_request_outside_period_and_first_request_membership():
    assert decision(None).reason == "no_request"
    assert decision(-1).reason == "outside_periods"
    assert decision(100).period == "A_N"  # Не переносится в B.


@pytest.mark.parametrize("last_event_ms", [None, 199, True, 2**63])
def test_last_observation_must_exist_and_not_precede_request(last_event_ms):
    with pytest.raises(InputValidationError):
        decision(200, last_event_ms=last_event_ms)


@pytest.mark.parametrize("tail_overflow", [False, True])
def test_checked_admission_times(tail_overflow):
    distant = (*PERIODS[:-1], Period("D", 600, 2**63 - 1, 2))
    if tail_overflow:
        distant = tuple(
            Period(name, 2**63 - 101 + i, 2**63 - 100 + i, 2) for i, name in enumerate(PERIOD_NAMES)
        )
        # Широкий A_G: окно запроса помещается, но last_event + g переполняется.
        distant = (Period("A_G", 0, distant[1].start_ms, 2), *distant[1:])
    with pytest.raises(NumericError):
        admit(
            0 if tail_overflow else 2**63 - 2,
            distant,
            CONFIG,
            last_event_ms=2**63 - 2,
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
