"""Свойства Э1: 1000 случаев на свойство, seed 17, независимые эталоны."""

from dataclasses import replace

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.data.schedule import make_schedule
from coveo_recsys.data.search import resolve_search_context
from coveo_recsys.state import build_prefix, empty_prefix, reduce_state
from tests.reference.e1_state_oracle import prefix as prefix_oracle
from tests.reference.e1_state_oracle import schedule as schedule_oracle
from tests.reference.e1_state_oracle import search as search_oracle

NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
COORDINATE = st.floats(-1e200, 1e200, allow_nan=False, allow_infinity=False)
PAYLOAD = st.one_of(
    st.tuples(st.lists(COORDINATE, min_size=1, max_size=3).map(tuple), st.none()),
    st.tuples(st.none(), st.sampled_from(["vector_absent", "vector_invalid", "vector_zero"])),
)


@st.composite
def search_blocks(draw):
    payloads = draw(st.lists(PAYLOAD, min_size=1, max_size=8))
    counts = draw(st.lists(st.integers(1, 100), min_size=len(payloads), max_size=len(payloads)))
    permutation = draw(st.permutations(tuple(range(len(payloads)))))
    return payloads, counts, permutation


@NORMATIVE
@seed(17)
@given(search_blocks())
def test_search_resolution_against_decimal_and_permutations(case):
    values, counts, permutation = case
    payloads = tuple(SearchPayload(vector, reason) for vector, reason in values)
    event = RawEvent("n", "s", "e", 0, 1, EventType.SEARCH, None, payloads[0], "fixture")
    block = CanonicalBlock("b", "s", 0, 1, (event,), (("e", sum(counts)),), payloads, tuple(counts))
    changed = CanonicalBlock(
        "b",
        "s",
        0,
        1,
        (event,),
        (("e", sum(counts)),),
        tuple(payloads[i] for i in permutation),
        tuple(counts[i] for i in permutation),
    )
    first = resolve_search_context(block, TaskConfig())
    second = resolve_search_context(changed, TaskConfig())
    expected, reason = search_oracle(values, 2)
    assert first is not None and second is not None
    assert first.payload.reason == second.payload.reason == reason
    if expected is None:
        assert first.payload.vector is second.payload.vector is None
    else:
        np.testing.assert_allclose(first.payload.vector, expected, atol=1e-10, rtol=1e-8)
        np.testing.assert_array_equal(first.payload.vector, second.payload.vector)


ACTIONS = [
    ("detail", "A"),
    ("detail", "é"),
    ("detail", "я"),
    ("add", "A"),
    ("remove", "A"),
    ("purchase", "B"),
    ("search", None),
    ("page", None),
]


@st.composite
def state_cases(draw):
    times = sorted(draw(st.lists(st.integers(-20, 60), min_size=1, max_size=10, unique=True)))
    config = TaskConfig(
        event_semantics=draw(st.sampled_from(["set_v1", "multiset_v1"])),
        detail_history_limit=draw(st.integers(1, 8)),
        block_history_limit=draw(st.integers(1, 8)),
        horizon_ms=draw(st.integers(1, 30)),
    )
    blocks = []
    for time in times:
        available = time + draw(st.integers(0, 15))
        actions = draw(st.lists(st.sampled_from(ACTIONS), min_size=1, max_size=5, unique=True))
        query = draw(
            st.sampled_from(
                [
                    SearchPayload((1.0, 0.0), None),
                    SearchPayload((0.0, 1.0), None),
                    SearchPayload(None, "vector_invalid"),
                ]
            )
        )
        events = tuple(
            RawEvent(
                "n",
                "s",
                f"{time}:{i}",
                time,
                available,
                EventType(kind),
                sku,
                query if kind == "search" else None,
                f"fixture:{time}:{i}",
            )
            for i, (kind, sku) in enumerate(actions)
        )
        counts = draw(st.lists(st.integers(1, 5), min_size=len(events), max_size=len(events)))
        search_present = any(event.event_type == EventType.SEARCH for event in events)
        payloads = (query,) if search_present else ()
        search_counts = tuple(
            count
            for event, count in zip(events, counts, strict=True)
            if event.event_type == EventType.SEARCH
        )
        blocks.append(
            CanonicalBlock(
                str(time),
                "s",
                time,
                available,
                events,
                tuple((event.event_id, count) for event, count in zip(events, counts, strict=True)),
                payloads,
                search_counts,
            )
        )
    return blocks, config, draw(st.integers(-21, 76)), draw(st.integers(-21, 76))


def assert_matches_oracle(actual, expected):
    assert actual.version == expected["version"]
    assert actual.first_event_time_ms == expected["first_event_time_ms"]
    assert actual.blocks == expected["blocks"]
    assert (
        tuple((item.sku, item.event_time_ms, item.count) for item in actual.detail_history)
        == expected["detail_history"]
    )
    assert (
        tuple((item.sku, item.event_time_ms, item.count) for item in actual.full_last_detail)
        == expected["full_last_detail"]
    )
    assert {(kind, sku): count for kind, sku, count in actual.counters} == expected["counts"]
    assert dict(actual.last_views) == expected["last_views"]
    assert set(actual.purchased_skus) == expected["purchased"]
    assert {
        item.sku: (item.state.value, item.updated_at_ms, item.last_unambiguous_ms)
        for item in actual.cart
    } == expected["cart"]
    if expected["search"] is None:
        assert actual.search is None
    else:
        block_id, time, available, direction, reason = expected["search"]
        assert actual.search is not None
        assert (
            actual.search.block_id,
            actual.search.event_time_ms,
            actual.search.available_time_ms,
        ) == (block_id, time, available)
        assert actual.search.payload.reason == reason
        if direction is None:
            assert actual.search.payload.vector is None
        else:
            np.testing.assert_allclose(
                actual.search.payload.vector, direction, atol=1e-10, rtol=1e-8
            )


@NORMATIVE
@seed(17)
@given(state_cases())
def test_visible_slice_against_full_journal_oracle(case):
    blocks, config, time, available = case
    actual = build_prefix(
        reversed(blocks),
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=time,
        availability_cutoff_ms=available,
        config=config,
    )
    expected = prefix_oracle(blocks, time, available, config)
    assert_matches_oracle(actual, expected)
    assert actual.event_cutoff_ms == time and actual.availability_cutoff_ms == available
    visible = [
        block
        for block in blocks
        if block.event_time_ms <= time and block.available_time_ms <= available
    ]
    assert actual == build_prefix(
        visible,
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=time,
        availability_cutoff_ms=available,
        config=config,
    )


@NORMATIVE
@seed(17)
@given(state_cases())
def test_incremental_state_and_schedule_against_independent_oracle(case):
    original, config, _, _ = case
    blocks = [
        replace(
            block,
            available_time_ms=block.event_time_ms,
            events=tuple(
                replace(event, available_time_ms=block.event_time_ms) for event in block.events
            ),
        )
        for block in original
    ]
    state = empty_prefix("n", "s", config)
    snapshots = []
    for block in blocks:
        before = state
        transition = reduce_state(state, block, config)
        assert transition.expected_version == before.version
        assert transition.receipt_id == block.block_id
        state = transition.next_snapshot
        snapshots.append(state)
    expected = prefix_oracle(blocks, blocks[-1].event_time_ms, blocks[-1].event_time_ms, config)
    assert_matches_oracle(state, expected)
    requests = make_schedule(snapshots, config)
    assert [
        (request.t_q, request.trigger_block_id, request.ordinal) for request in requests
    ] == schedule_oracle(blocks, config)
    assert len({request.request_id for request in requests}) == len(requests)
    for request in requests:
        prefix = request.prefix
        assert prefix.event_cutoff_ms == request.t_q
        assert all(
            block.event_time_ms <= request.t_q and block.available_time_ms <= request.a_q
            for block in prefix.blocks
        )
