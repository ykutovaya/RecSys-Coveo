from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError, NumericError, OutcomeStatus
from coveo_recsys.contracts.preparation import PERIOD_NAMES, Coverage, Period, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.data.schedule import make_schedule
from coveo_recsys.state import empty_prefix, reduce_state
from coveo_recsys.targets import observe_session, prepare_targets
from tests.reference.e1_targets_oracle import reference_targets

CONFIG = TaskConfig(horizon_ms=20, gap_ms=10)
PERIODS = tuple(Period(name, i * 100, (i + 1) * 100, 100) for i, name in enumerate(PERIOD_NAMES))
COVERAGE = Coverage("n", -1000, 1000, (), "explicit synthetic completeness", 2000)


def make_history(rows, config=CONFIG, *, namespace="n", session="s"):
    """Только подготовка входов: ожидаемые значения вычисляет отдельный эталон."""
    groups = {}
    for time, kind, sku in rows:
        counts = groups.setdefault(time, {})
        counts[kind, sku] = counts.get((kind, sku), 0) + 1
    blocks = []
    for time, counts in sorted(groups.items()):
        events = tuple(
            RawEvent(
                namespace,
                session,
                f"{time}:{i}",
                time,
                time,
                EventType(kind),
                sku,
                SearchPayload(None, "vector_absent") if kind == "search" else None,
                f"fixture:{time}:{i}",
            )
            for i, (kind, sku) in enumerate(counts)
        )
        searches = sum(n for (kind, _), n in counts.items() if kind == "search")
        blocks.append(
            CanonicalBlock(
                f"{namespace}:{session}:{time}",
                session,
                time,
                time,
                events,
                tuple(
                    (e.event_id, count) for e, count in zip(events, counts.values(), strict=True)
                ),
                (SearchPayload(None, "vector_absent"),) if searches else (),
                (searches,) if searches else (),
            )
        )
    state = empty_prefix(namespace, session, config)
    prefixes = []
    for block in blocks:
        state = reduce_state(state, block, config).next_snapshot
        prefixes.append(state)
    return tuple(blocks), make_schedule(prefixes, config)


def prepare(rows, coverage=COVERAGE, config=CONFIG, periods=PERIODS):
    blocks, requests = make_history(rows, config)
    return prepare_targets(blocks, requests, coverage, periods, config)


def assert_oracle(rows, coverage=COVERAGE, config=CONFIG):
    expected = reference_targets(
        rows,
        [(p.name, p.start_ms, p.end_ms) for p in PERIODS],
        config.horizon_ms,
        config.gap_ms,
    )
    blocks, requests = make_history(rows, config)
    session, targets = prepare_targets(blocks, requests, coverage, PERIODS, config)
    assert [q.t_q for q in requests] == expected["requests"]
    assert session.request_count == len(expected["requests"])
    assert session.target_end_ms == expected["tau"]
    assert session.period == expected["period"]
    assert session.admitted == expected["admitted"]
    assert session.telemetry_complete is None
    assert [target.purchases for target in targets] == expected["targets"]
    assert all(
        target.status == (OutcomeStatus.KNOWN if target.purchases else OutcomeStatus.UNKNOWN)
        for target in targets
    )
    return session, targets


def test_targets_include_all_distinct_registered_skus_and_right_boundary():
    rows = [
        (200, "detail", "A"),
        (201, "search", None),
        (205, "purchase", "first"),
        (220, "purchase", "not_in_I_G"),
        (220, "purchase", "not_in_I_G"),
        (220, "purchase", "é"),
        (221, "purchase", "late"),
    ]
    session, (target,) = assert_oracle(rows)
    assert session.target_end_ms == 220 and session.request_count == 1
    assert target.purchases == ("first", "not_in_I_G", "é")
    assert target.horizon_ms == 20 and target.reason is None


@pytest.mark.parametrize("last,expected", [(200, 200), (205, 205), (220, 220), (221, 220)])
def test_last_observed_event_limits_target_window(last, expected):
    rows = [(200, "detail", "A"), (last, "search", None)]
    session, _ = assert_oracle(rows)
    assert session.last_event_ms == last and session.target_end_ms == expected


@pytest.mark.parametrize("time,expected", [(219, ("A",)), (220, ("A",)), (221, ())])
def test_purchase_horizon_boundary(time, expected):
    _, (target,) = assert_oracle([(200, "detail", "A"), (time, "purchase", "A")])
    assert target.purchases == expected


@pytest.mark.parametrize(
    "rows",
    [
        [(200, "purchase", "A"), (201, "detail", "B")],
        [(200, "detail", "A"), (200, "purchase", "A")],
        [(200, "page", None), (201, "search", None)],
    ],
)
def test_no_request_is_not_negative_outcome(rows):
    session, targets = assert_oracle(rows)
    assert session.request_count == 0 and session.admission_ms is None
    assert session.exclusion_reason == "no_request" and not session.admitted
    assert targets == ()


@pytest.mark.parametrize("purchase", [False, True])
@pytest.mark.parametrize(
    "coverage",
    [
        replace(COVERAGE, confirmed_at_ms=None),
        replace(COVERAGE, start_ms=201, end_ms=202),
        replace(COVERAGE, gaps=((200, 211),)),
    ],
)
def test_telemetry_metadata_neither_blocks_registered_goals_nor_confirms_negatives(
    purchase, coverage
):
    rows = [
        (200, "detail", "A"),
        (210, "purchase" if purchase else "page", "B" if purchase else None),
    ]
    session, (target,) = assert_oracle(rows, coverage)
    assert session.admitted and session.telemetry_complete is None
    assert target.purchases == (("B",) if purchase else ())
    assert target.reason == (None if purchase else "no_registered_purchase")


@pytest.mark.parametrize("last,admitted", [(89, True), (90, False), (91, False)])
def test_A_G_tail_strict_boundary_excludes_entire_session(last, admitted):
    session, (target,) = assert_oracle(
        [(0, "detail", "A"), (10, "purchase", "B"), (last, "search", None)]
    )
    assert session.admitted is admitted
    assert session.exclusion_reason == (None if admitted else "late_A_G_tail")
    assert target.purchases == (("B",) if admitted else ())


def test_late_tail_in_B_does_not_extend_target_or_reject_session():
    session, (target,) = assert_oracle(
        [(200, "detail", "A"), (220, "purchase", "B"), (310, "search", None)]
    )
    assert session.admitted and session.target_end_ms == 220
    assert target.purchases == ("B",)


@pytest.mark.parametrize("t0,admitted", [(269, True), (270, False), (271, False)])
def test_strict_calendar_admission_boundary(t0, admitted):
    session, (target,) = assert_oracle([(t0, "detail", "A"), (t0 + 1, "purchase", "B")])
    assert session.admitted is admitted
    assert session.exclusion_reason == (None if admitted else "window_boundary")
    assert target.purchases == (("B",) if admitted else ())


def test_outside_periods_is_excluded_even_with_registered_purchase():
    session, (target,) = assert_oracle([(-1, "detail", "A"), (1, "purchase", "B")])
    assert not session.admitted and session.period is None
    assert session.exclusion_reason == "outside_periods"
    assert target.purchases == () and target.status == OutcomeStatus.UNKNOWN


def test_later_requests_are_not_generated_but_full_history_is_read():
    rows = [(200 + i, "detail", str(i)) for i in range(15)] + [
        (218, "purchase", "outside_catalog"),
        (250, "page", None),
    ]
    session, targets = assert_oracle(rows)
    assert session.request_count == len(targets) == 1 and session.last_event_ms == 250
    assert targets[0].purchases == ("outside_catalog",)


def test_targets_and_future_events_do_not_mutate_first_request():
    blocks, requests = make_history([(200, "detail", "A"), (210, "purchase", "B")])
    before = to_json(requests[0])
    first = prepare_targets(blocks, requests, COVERAGE, PERIODS, CONFIG)
    changed, changed_requests = make_history(
        [(200, "detail", "A"), (205, "search", None), (210, "purchase", "C")]
    )
    second = prepare_targets(changed, changed_requests, COVERAGE, PERIODS, CONFIG)
    assert to_json(requests[0]) == before == to_json(changed_requests[0])
    assert first[1][0].purchases != second[1][0].purchases
    assert first[0].request_count == second[0].request_count == 1
    assert not hasattr(requests[0].prefix, "target_end_ms")


@pytest.mark.parametrize("semantics", ["set_v1", "multiset_v1"])
def test_multiplicity_does_not_weight_targets(semantics):
    config = replace(CONFIG, event_semantics=semantics)
    _, (target,) = assert_oracle(
        [(200, "detail", "A")] * 3 + [(210, "purchase", "B")] * 5, config=config
    )
    assert target.purchases == ("B",)


def test_mixed_unsorted_duplicate_or_delayed_blocks_refused():
    blocks, requests = make_history([(200, "detail", "A"), (210, "purchase", "B")])
    other, _ = make_history([(210, "page", None)], namespace="other")
    delayed = replace(
        blocks[1],
        available_time_ms=211,
        events=tuple(replace(e, available_time_ms=211) for e in blocks[1].events),
    )
    for values in [
        (),
        blocks[::-1],
        (*blocks, blocks[-1]),
        (blocks[0], other[0]),
        (blocks[0], delayed),
        (blocks[0], replace(blocks[1], block_id=blocks[0].block_id)),
    ]:
        with pytest.raises(InputValidationError):
            prepare_targets(values, requests, COVERAGE, PERIODS, CONFIG)
    with pytest.raises(InputValidationError):
        prepare_targets(
            blocks, requests, replace(COVERAGE, source_namespace="other"), PERIODS, CONFIG
        )


def test_incompatible_missing_or_later_requests_refused():
    blocks, requests = make_history([(200, "detail", "A"), (201, "detail", "B")])
    _, late = make_history([(201, "detail", "B")])
    delayed_prefix = replace(requests[0].prefix, availability_cutoff_ms=201)
    invalid = [
        (),
        late,
        (requests[0], requests[0]),
        (replace(requests[0], a_q=201, prefix=delayed_prefix),),
    ]
    for values in invalid:
        with pytest.raises(InputValidationError):
            prepare_targets(blocks, values, COVERAGE, PERIODS, CONFIG)
    with pytest.raises(InputValidationError):
        prepare_targets(
            blocks, requests, COVERAGE, PERIODS, replace(CONFIG, event_semantics="multiset_v1")
        )


def test_target_horizon_int64_arithmetic_has_no_wraparound():
    blocks, _ = make_history([(2**63 - 2, "page", None)])
    with pytest.raises(NumericError):
        observe_session(blocks, blocks[0].event_time_ms, COVERAGE, CONFIG)
