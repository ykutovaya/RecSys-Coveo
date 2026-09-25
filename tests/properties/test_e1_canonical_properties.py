from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts.records import EventType, RawEvent
from coveo_recsys.data.canonical import canonicalize
from tests.reference.e1_data_oracle import expected_counts


@seed(17)
@given(
    st.lists(
        st.tuples(st.integers(0, 5), st.sampled_from(["A", "B", "é", "é"])), min_size=1, max_size=15
    )
)
def test_canonical_counts_and_permutations(rows):
    events = [
        RawEvent("n", "s", str(i), t, t, EventType.DETAIL, sku, None, str(i), str(i))
        for i, (t, sku) in enumerate(rows)
    ]
    expected = expected_counts(events)
    blocks = list(canonicalize(events, chunk_size=3, fan_in=2))
    reverse = list(canonicalize(reversed(events), chunk_size=5, fan_in=2))
    actual = {
        (e.source_namespace, e.session_id, e.event_time_ms, e.event_type.value, e.sku): n
        for b in blocks
        for e, (_, n) in zip(b.events, b.raw_multiplicities, strict=True)
    }
    assert actual == expected
    assert [b.block_id for b in blocks] == [b.block_id for b in reverse]
