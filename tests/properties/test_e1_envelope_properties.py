from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.data.envelope import validate_sealed_block
from tests.reference.e1_envelope_oracle import (
    event,
    expected_time_reason,
    expected_transport_count,
    packet,
)


@seed(17)
@given(time=st.integers(-(2**63), 2**63 - 1), delta=st.integers(-1_800_002, 2))
def test_exact_time_window_against_oracle(time, delta):
    decision = max(-(2**63), min(2**63 - 1, time - delta))
    calendar = (-(2**63), 2**63 - 1)
    wire, payload = packet([event(time_ms=time)], overrides={"event_time_ms": time})
    result = validate_sealed_block(
        wire,
        payload,
        decision_time_ms=decision,
        allowed_event_time_range=calendar,
        expected_namespace="source",
        query_dimension=2,
    )
    reason = expected_time_reason(time, decision, calendar)
    assert result.static_valid == (reason is None)
    if reason:
        assert result.block is None and result.issues[0].code == reason


@seed(17)
@given(count=st.integers(0, 102), first_id=st.integers(0, 100_000), compact=st.booleans())
def test_transport_size_not_logical_size(count, first_id, compact):
    wire, payload = packet([event(str(first_id + i)) for i in range(count)], compact=compact)
    result = validate_sealed_block(
        wire,
        payload,
        decision_time_ms=1000,
        allowed_event_time_range=(0, 2000),
        expected_namespace="source",
        query_dimension=2,
    )
    assert result.static_valid == expected_transport_count(count)
    if result.static_valid:
        assert len(result.block.events) == count
