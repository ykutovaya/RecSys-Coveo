from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError, NumericError, OutcomeStatus
from coveo_recsys.contracts.preparation import PERIOD_NAMES, Coverage, Period, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.data.schedule import make_schedule
from coveo_recsys.state import empty_prefix, reduce_state
from coveo_recsys.targets import coverage_reason, observe_session, prepare_targets
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
        coverage.start_ms,
        coverage.end_ms,
        coverage.gaps,
        coverage.confirmed_at_ms is not None,
        config.max_requests,
    )
    blocks, requests = make_history(rows, config)
    session, targets = prepare_targets(blocks, requests, coverage, PERIODS, config)
    assert [q.t_q for q in requests] == expected["requests"]
    assert session.request_count == len(expected["requests"])
    assert session.close_ms == expected["close_ms"]
    assert session.mature == expected["mature"]
    assert session.target_end_ms == expected["tau"]
    assert session.period == expected["period"]
    assert [target.purchases for target in targets] == expected["targets"]
    assert all(target.b_s == expected["b_s"] for target in targets)
    assert all(
        target.status == (OutcomeStatus.KNOWN if expected["admitted"] else OutcomeStatus.UNKNOWN)
        for target in targets
    )
    return session, targets


def test_targets_include_right_boundary_unknown_catalog_sku_and_distinct_purchases():
    rows = [
        (200, "detail", "A"),
        (201, "search", None),
        (220, "purchase", "not_in_I_G"),
        (220, "purchase", "not_in_I_G"),
        (220, "purchase", "é"),
        (221, "purchase", "late"),
    ]
    session, targets = assert_oracle(rows)
    assert session.target_end_ms == 220 and session.request_count == 2
    assert [t.purchases for t in targets] == [("not_in_I_G", "é")] * 2
    assert all(t.horizon_ms == 20 and t.b_s == 1 and t.reason is None for t in targets)


def test_last_event_limits_tau_and_confirmation_time_does_not_replace_close():
    session, targets = assert_oracle([(200, "detail", "A"), (205, "purchase", "B")])
    assert session.target_end_ms == 205 and session.close_ms == 215
    assert COVERAGE.confirmed_at_ms == 2000 and session.close_ms != COVERAGE.confirmed_at_ms
    assert targets[0].purchases == ("B",)


def test_request_at_horizon_has_known_empty_target_with_reason():
    rows = [(200, "detail", "A"), (220, "search", None), (221, "purchase", "A")]
    session, targets = assert_oracle(rows)
    assert session.request_count == 2
    assert all(
        t.purchases == () and t.b_s == 0 and t.reason == "no_future_purchase" for t in targets
    )


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
    assert session.exclusion_reason == "no_request" and not session.mature
    assert targets == ()


@pytest.mark.parametrize("purchase", [False, True])
@pytest.mark.parametrize(
    "coverage,reason",
    [
        (replace(COVERAGE, confirmed_at_ms=None), "coverage_unconfirmed"),
        (replace(COVERAGE, end_ms=229), "coverage_range"),
        (replace(COVERAGE, gaps=((210, 211),)), "coverage_gap"),
    ],
)
def test_incomplete_buying_and_nonbuying_sessions_remain_unknown(purchase, coverage, reason):
    rows = [
        (200, "detail", "A"),
        (201, "purchase" if purchase else "page", "B" if purchase else None),
    ]
    session, targets = assert_oracle(rows, coverage)
    assert not session.mature and session.target_end_ms is None
    assert session.exclusion_reason == reason
    assert all(
        t.status == OutcomeStatus.UNKNOWN
        and t.purchases is None
        and t.b_s is None
        and t.reason == reason
        for t in targets
    )


def test_gap_before_first_detail_can_hide_purchase_or_earlier_request():
    session, targets = assert_oracle(
        [(200, "page", None), (210, "detail", "A")], replace(COVERAGE, gaps=((201, 209),))
    )
    assert session.admission_ms == 210 and session.exclusion_reason == "coverage_gap"
    assert not session.mature and targets[0].b_s is None


def test_coverage_end_equality_and_half_open_gap_boundaries():
    rows = [(200, "detail", "A")]
    assert prepare(rows, replace(COVERAGE, start_ms=200, end_ms=230))[0].mature
    assert not prepare(rows, replace(COVERAGE, start_ms=201))[0].mature
    for gaps, complete in [
        (((199, 200),), True),
        (((230, 240),), True),
        (((229, 230),), False),
        (((200, 201),), False),
    ]:
        assert prepare(rows, replace(COVERAGE, gaps=gaps))[0].mature is complete


def test_long_B_window_mature_without_full_session_closure():
    rows = [(200, "detail", "A"), (220, "purchase", "B"), (290, "page", None)]
    session, targets = assert_oracle(rows, replace(COVERAGE, end_ms=230))
    assert session.mature and session.coverage_complete and session.close_ms is None
    assert session.target_end_ms == 220 and session.exclusion_reason is None
    assert targets[0].status == OutcomeStatus.KNOWN and targets[0].purchases == ("B",)


@pytest.mark.parametrize("offset,excluded", [(0, True), (200, False)])
def test_gap_after_window_disqualifies_only_full_session_A_G(offset, excluded):
    rows = [(offset, "detail", "A"), (offset + 20, "purchase", "B"), (offset + 70, "page", None)]
    session, targets = assert_oracle(rows, replace(COVERAGE, gaps=((offset + 40, offset + 50),)))
    assert session.mature and session.close_ms == offset + 80
    assert (session.exclusion_reason == "session_not_closed_in_A_G") is excluded
    assert (targets[0].status == OutcomeStatus.UNKNOWN) is excluded


def test_A_G_close_exact_period_boundary_is_excluded_but_window_stays_mature():
    rows = [(0, "detail", "A"), (90, "page", None)]
    session, targets = assert_oracle(rows)
    assert session.close_ms == 100 and session.mature
    assert session.exclusion_reason == "session_not_closed_in_A_G"
    assert targets[0].purchases is None


@pytest.mark.parametrize("t0,admitted", [(269, True), (270, False), (271, False)])
def test_strict_period_admission_is_separate_from_observed_maturity(t0, admitted):
    session, targets = assert_oracle([(t0, "detail", "A")])
    assert session.mature
    assert (session.exclusion_reason is None) is admitted
    assert (targets[0].status == OutcomeStatus.KNOWN) is admitted


def test_outside_periods_has_unknown_targets_even_with_full_observation():
    session, targets = assert_oracle([(-1, "detail", "A"), (1, "purchase", "B")])
    assert (
        session.mature and session.period is None and session.exclusion_reason == "outside_periods"
    )
    assert targets[0].purchases is None


def test_all_original_requests_retained_after_cap_and_purchase():
    rows = [(200 + i, "detail", str(i)) for i in range(15)] + [
        (218, "purchase", "outside_catalog"),
        (250, "page", None),
    ]
    session, targets = assert_oracle(rows)
    assert session.request_count == len(targets) == 10 and session.last_event_ms == 250
    assert all(t.purchases == ("outside_catalog",) for t in targets)


def test_targets_and_future_count_do_not_mutate_past_input():
    blocks, requests = make_history([(200, "detail", "A"), (210, "purchase", "B")])
    before = to_json(requests[0])
    first = prepare_targets(blocks, requests, COVERAGE, PERIODS, CONFIG)
    changed, changed_requests = make_history(
        [(200, "detail", "A"), (205, "search", None), (210, "purchase", "C")]
    )
    second = prepare_targets(changed, changed_requests, COVERAGE, PERIODS, CONFIG)
    assert to_json(requests[0]) == before == to_json(changed_requests[0])
    assert first[1][0].purchases != second[1][0].purchases
    assert first[0].request_count == 1 and second[0].request_count == 2
    assert not hasattr(requests[0].prefix, "target_end_ms")


@pytest.mark.parametrize("semantics", ["set_v1", "multiset_v1"])
def test_multiplicity_does_not_weight_targets(semantics):
    config = replace(CONFIG, event_semantics=semantics)
    _, targets = assert_oracle(
        [(200, "detail", "A")] * 3 + [(210, "purchase", "B")] * 5, config=config
    )
    assert targets[0].purchases == ("B",) and targets[0].b_s == 1


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


def test_incompatible_or_partial_numbering_of_requests_refused():
    blocks, requests = make_history([(200, "detail", "A"), (201, "search", None)])
    delayed_prefix = replace(requests[0].prefix, availability_cutoff_ms=201)
    invalid = [
        (requests[1],),
        (requests[0], requests[0]),
        requests[::-1],
        (replace(requests[0], a_q=201, prefix=delayed_prefix),),
    ]
    for values in invalid:
        with pytest.raises(InputValidationError):
            prepare_targets(blocks, values, COVERAGE, PERIODS, CONFIG)
    with pytest.raises(InputValidationError):
        prepare_targets(blocks, requests, COVERAGE, PERIODS, replace(CONFIG, max_requests=1))
    with pytest.raises(InputValidationError):
        prepare_targets(
            blocks, requests, COVERAGE, PERIODS, replace(CONFIG, event_semantics="multiset_v1")
        )


def test_exact_int64_arithmetic_has_no_wraparound():
    blocks, _ = make_history([(2**63 - 2, "page", None)])
    with pytest.raises(NumericError):
        observe_session(blocks, None, replace(COVERAGE, end_ms=2**63 - 1), CONFIG)
    with pytest.raises(InputValidationError):
        coverage_reason(COVERAGE, 10, 10)
