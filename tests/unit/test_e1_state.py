import csv
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.contracts import INT64_MAX, INT64_MIN, InputValidationError, NumericError
from coveo_recsys.contracts.preparation import ScheduleCursor, TaskConfig
from coveo_recsys.contracts.records import (
    CanonicalBlock,
    CartState,
    DetailHistoryItem,
    EventType,
    RawEvent,
    SearchPayload,
)
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.io import iter_events, load_profile
from coveo_recsys.data.schedule import advance_schedule, make_schedule
from coveo_recsys.state import build_prefix, empty_prefix, reduce_state


def block(time, actions, *, available=None, counts=None, namespace="n", session="s", query=None):
    available = time if available is None else available
    events = tuple(
        RawEvent(
            namespace,
            session,
            f"{time}:{index}",
            time,
            available,
            EventType(kind),
            sku,
            query if kind == "search" else None,
            f"fixture:{time}:{index}",
        )
        for index, (kind, sku) in enumerate(actions)
    )
    payloads = (
        (query or SearchPayload(None, "vector_absent"),)
        if any(event.event_type == EventType.SEARCH for event in events)
        else ()
    )
    return CanonicalBlock(
        f"{namespace}:{session}:{time}",
        session,
        time,
        available,
        events,
        tuple((event.event_id, (counts or [1] * len(events))[i]) for i, event in enumerate(events)),
        payloads,
        (1,) if payloads else (),
    )


def prefixes(blocks, config=TaskConfig()):
    state = empty_prefix("n", "s", config)
    result = []
    for value in blocks:
        state = reduce_state(state, value, config).next_snapshot
        result.append(state)
    return result


def test_independent_histories_and_unbounded_prefix_aggregates():
    values = [
        block(0, [("detail", "A")]),
        *[
            block(time, [("search", None)], query=SearchPayload((1.0, 0.0), None))
            for time in range(1, 26)
        ],
        block(26, [("detail", "B")]),
    ]
    all_prefixes = prefixes(values)
    previous = to_json(all_prefixes[0])
    final = all_prefixes[-1]
    assert len(final.blocks) == 20 and final.blocks[0].event_time_ms == 7
    assert final.detail_history == (DetailHistoryItem("A", 0, 1), DetailHistoryItem("B", 26, 1))
    assert final.full_last_detail == (DetailHistoryItem("B", 26, 1),)
    assert ("search", None, 25) in final.counters
    assert final.last_views == (("A", 0), ("B", 26))
    assert to_json(all_prefixes[0]) == previous
    assert from_json(to_json(final), type(final)) == final


def test_first_visible_event_survives_trimming_and_replay():
    values = [block(time, [("page", None)]) for time in range(26)]
    values.append(block(100, [("detail", "A")]))
    state = prefixes(values)[-1]
    assert state.first_event_time_ms == 0
    assert state.blocks[0].event_time_ms > 0
    (request,) = make_schedule(prefixes(values), TaskConfig())
    assert request.t_q - request.prefix.first_event_time_ms == 100
    delayed = replace(
        values[0],
        available_time_ms=101,
        events=tuple(replace(event, available_time_ms=101) for event in values[0].events),
    )
    partial = build_prefix(
        [delayed, *values[1:]],
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=100,
        availability_cutoff_ms=100,
        config=TaskConfig(),
    )
    assert partial.first_event_time_ms == 1
    complete = build_prefix(
        [delayed, *values[1:]],
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=100,
        availability_cutoff_ms=101,
        config=TaskConfig(),
    )
    assert complete.first_event_time_ms == 0


@pytest.mark.parametrize(
    "semantics,expected", [("set_v1", (("A", 1), ("B", 1))), ("multiset_v1", (("A", 9), ("B", 1)))]
)
def test_full_group_and_raw_counts_survive_history_trimming(semantics, expected):
    config = TaskConfig(event_semantics=semantics, detail_history_limit=5)
    value = block(0, [("detail", "A"), ("detail", "B")], counts=[9, 1])
    final = prefixes([value], config)[0]
    assert tuple((item.sku, item.count) for item in final.full_last_detail) == expected
    assert sum(item.count for item in final.detail_history) == (2 if semantics == "set_v1" else 5)
    if semantics == "multiset_v1":
        assert final.detail_history == (DetailHistoryItem("A", 0, 5),)
    assert ("detail", "A", expected[0][1]) in final.counters


def test_history_time_ties_use_utf8_sku_not_event_id():
    value = block(0, [("detail", "я"), ("detail", "A"), ("detail", "é")])
    snapshot = prefixes([value], TaskConfig(detail_history_limit=2))[0]
    assert tuple(item.sku for item in snapshot.detail_history) == ("A", "é")
    assert len(snapshot.full_last_detail) == 3


def test_cart_is_ternary_and_preserves_last_unambiguous_time():
    values = [
        block(0, [("detail", "A")]),
        block(1, [("add", "A")]),
        block(2, [("add", "A"), ("remove", "A")]),
        block(3, [("remove", "A")]),
        block(4, [("remove", "A")]),
    ]
    states = prefixes(values)
    assert states[0].cart == ()  # Ненаблюдавшийся товар имеет состояние unknown.
    assert states[1].cart[0].state == CartState.ADD
    assert states[2].cart[0].state == CartState.UNKNOWN
    assert states[2].cart[0].updated_at_ms == 2 and states[2].cart[0].last_unambiguous_ms == 1
    assert states[4].cart[0].state == CartState.REMOVE
    assert states[4].cart[0].updated_at_ms == states[4].cart[0].last_unambiguous_ms == 4


def test_latest_unavailable_search_never_restores_older_context():
    values = [
        block(0, [("detail", "A")]),
        block(1, [("search", None)], query=SearchPayload((1.0, 0.0), None)),
        block(2, [("search", None)]),
        block(3, [("page", None)]),
    ]
    states = prefixes(values)
    assert states[1].search.payload.vector == (1.0, 0.0)
    assert states[2].search.payload == SearchPayload(None, "vector_absent")
    assert states[3].search == states[2].search
    assert [request.t_q for request in make_schedule(states, TaskConfig())] == [0, 1, 2]


def test_search_results_and_clicks_cannot_change_computational_context(tmp_path):
    profile = load_profile(Path(__file__).resolve().parents[2] / "configs/synthetic_csv.json")
    path = tmp_path / "search.csv"
    variants = [('["A"]', '["A"]'), ('["future","different"]', "[]")]
    prepared, origins = [], []
    for results, clicks in variants:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "session_id",
                    "event_id",
                    "server_timestamp_epoch_ms",
                    "query_vector",
                    "product_skus_hash",
                    "clicked_skus_hash",
                ]
            )
            writer.writerow(["s", "search:1", 1, "[3,4]", results, clicks])
            writer.writerow(["s", "search:2", 2, "null", results, clicks])
        searches = list(iter_events(path, "search", profile, batch_size=1))
        origins.append(tuple(event.provenance for event in searches))
        detail = RawEvent(
            profile.namespace, "s", "detail", 0, 0, EventType.DETAIL, "A", None, "fixture:detail"
        )
        blocks = list(canonicalize([detail, *searches], temp_dir=tmp_path, chunk_size=1))
        state = empty_prefix(profile.namespace, "s", TaskConfig())
        snapshots = []
        for value in blocks:
            state = reduce_state(state, value, TaskConfig()).next_snapshot
            snapshots.append(state)
        requests = make_schedule(snapshots, TaskConfig())
        comparable = []
        for request in requests:
            # Хеши исходных байтов закономерно меняются; расчёт их не использует.
            computational_blocks = tuple(
                replace(
                    item,
                    events=tuple(
                        replace(event, provenance="audit_omitted", origins=())
                        for event in item.events
                    ),
                )
                for item in request.prefix.blocks
            )
            comparable.append(
                replace(request, prefix=replace(request.prefix, blocks=computational_blocks))
            )
        prepared.append(tuple(comparable))
    assert origins[0] != origins[1]
    assert prepared[0] == prepared[1]
    assert [request.t_q for request in prepared[0]] == [0, 1, 2]
    assert prepared[0][-1].prefix.search.payload.reason == "vector_absent"


def test_two_clocks_full_blocks_and_immutable_old_slice():
    delayed = block(10, [("detail", "A"), ("purchase", "B")], available=30)
    timely = block(20, [("detail", "C")])
    values = [delayed, timely]
    old = build_prefix(
        values,
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=20,
        availability_cutoff_ms=29,
        config=TaskConfig(),
    )
    serialized = to_json(old)
    current = build_prefix(
        reversed(values),
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=20,
        availability_cutoff_ms=30,
        config=TaskConfig(),
    )
    event_limited = build_prefix(
        values,
        source_namespace="n",
        session_id="s",
        event_cutoff_ms=9,
        availability_cutoff_ms=100,
        config=TaskConfig(),
    )
    assert old.last_views == (("C", 20),) and old.purchased_skus == ()
    assert current.last_views == (("A", 10), ("C", 20)) and current.purchased_skus == ("B",)
    assert event_limited.version == 0 and event_limited.blocks == ()
    assert to_json(old) == serialized
    with pytest.raises(InputValidationError):
        reduce_state(old, delayed, TaskConfig())


def test_request_boundaries_one_per_block_and_stop_after_purchase():
    config = TaskConfig(horizon_ms=10)
    values = [
        block(-1, [("search", None)]),
        block(0, [("detail", "A"), ("search", None)]),
        block(1, [("page", None)]),
        block(9, [("detail", "B")]),
        block(10, [("search", None)]),
        block(11, [("detail", "C")]),
        block(12, [("purchase", "A")]),
        block(13, [("detail", "D")]),
    ]
    states = prefixes(values, config)
    requests = make_schedule(states, config)
    assert [request.t_q for request in requests] == [0, 9, 10]
    assert [request.ordinal for request in requests] == [1, 2, 3]
    assert ("detail", "D", 1) in states[-1].counters
    assert all(not hasattr(request, "catalog_sha256") for request in requests)
    assert make_schedule(states, config) == requests


@pytest.mark.parametrize(
    "values",
    [
        [block(0, [("detail", "A"), ("purchase", "A")]), block(1, [("detail", "B")])],
        [block(0, [("purchase", "A")]), block(1, [("detail", "B")])],
    ],
)
def test_purchase_before_or_in_first_detail_excludes_requests(values):
    assert make_schedule(prefixes(values), TaskConfig()) == ()


def test_stop_limit_keeps_processing_later_events():
    values = [block(time, [("detail", "A")]) for time in range(12)]
    values += [block(12, [("purchase", "B")]), block(13, [("detail", "C")])]
    states = prefixes(values)
    requests = make_schedule(iter(states), TaskConfig())
    assert len(requests) == 10 and states[-1].version == 14
    assert states[-1].purchased_skus == ("B",)


def test_scope_semantics_order_and_trigger_rejections():
    config = TaskConfig()
    state = prefixes([block(0, [("detail", "A")])])[0]
    for bad in (
        block(1, [("detail", "A")], namespace="other"),
        block(1, [("detail", "A")], session="other"),
        block(0, [("detail", "A")]),
    ):
        with pytest.raises(InputValidationError):
            reduce_state(state, bad, config)
    with pytest.raises(InputValidationError):
        reduce_state(
            state, block(1, [("detail", "A")]), replace(config, event_semantics="multiset_v1")
        )
    with pytest.raises(InputValidationError):
        make_schedule([state, state], config)
    with pytest.raises(InputValidationError):
        advance_schedule(ScheduleCursor("n", "s"), state, "missing", config)
    with pytest.raises(InputValidationError):
        build_prefix(
            [block(0, [("detail", "A")], namespace="other")],
            source_namespace="n",
            session_id="s",
            event_cutoff_ms=1,
            availability_cutoff_ms=1,
            config=config,
        )


def test_checked_counter_version_horizon_and_extreme_first_timestamp():
    config = TaskConfig()
    initial = empty_prefix("n", "s", config)
    earliest = reduce_state(initial, block(INT64_MIN, [("detail", "A")]), config).next_snapshot
    assert earliest.event_cutoff_ms == INT64_MIN
    state = prefixes([block(0, [("detail", "A")])])[0]
    for inflated in (
        replace(state, version=INT64_MAX),
        replace(state, counters=(("detail", "A", INT64_MAX),)),
    ):
        with pytest.raises(NumericError):
            reduce_state(inflated, block(1, [("detail", "A")]), config)
    too_late = prefixes([block(INT64_MAX, [("detail", "A")])])[0]
    with pytest.raises(NumericError):
        make_schedule([too_late], config)


def test_empty_input_and_past_context_are_independent_of_suffix():
    assert make_schedule([], TaskConfig()) == ()
    base = [block(0, [("detail", "A")])]
    states = prefixes(base)
    (request,) = make_schedule(states, TaskConfig())
    first_bytes = to_json(request)
    continuation = base + [block(1, [("purchase", "B")]), block(2, [("search", None)])]
    assert make_schedule(prefixes(continuation), TaskConfig())[0] == request
    assert to_json(request) == first_bytes
