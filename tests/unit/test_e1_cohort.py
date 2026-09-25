from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PERIOD_NAMES, Period, TaskConfig
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.targets import CohortRecord, historical_cohort, prepare_targets
from tests.reference.e1_targets_oracle import reference_cohort
from tests.unit.test_e1_targets import CONFIG, COVERAGE, PERIODS, make_history


def cohort_case(rows, *, coverage=COVERAGE, config=CONFIG, periods=PERIODS):
    blocks, requests = make_history(rows, config)
    session, _ = prepare_targets(blocks, requests, coverage, periods, config)
    result = historical_cohort(blocks, session, coverage, periods, config)
    expected = reference_cohort(
        rows,
        [(p.name, p.start_ms, p.end_ms) for p in periods],
        config.horizon_ms,
        config.gap_ms,
        coverage.start_ms,
        coverage.end_ms,
        coverage.gaps,
        coverage.confirmed_at_ms is not None,
        config.max_requests,
    )
    assert {
        r.sku: (r.anchor_ms, r.target_end_ms, r.mature, r.purchased) for r in result
    } == expected
    assert len({r.sku for r in result}) == len(result)
    assert [r.sku for r in result] == sorted(expected, key=lambda sku: sku.encode("utf-8"))
    return session, result


def test_B09_zero_five_ten_minutes_anchor_does_not_shift():
    config = TaskConfig()
    periods = tuple(
        Period(name, i * 200_000_000, (i + 1) * 200_000_000, 100)
        for i, name in enumerate(PERIOD_NAMES)
    )
    coverage = replace(COVERAGE, end_ms=1_400_000_001, confirmed_at_ms=1_400_000_001)
    _, records = cohort_case(
        [(0, "detail", "A"), (300_000, "purchase", "A"), (600_000, "detail", "A")],
        coverage=coverage,
        config=config,
        periods=periods,
    )
    assert len(records) == 1 and records[0].anchor_ms == 0 and records[0].purchased == 1
    assert records[0].target_end_ms == 86_400_000


@pytest.mark.parametrize("purchase_time,expected", [(9, 0), (10, 0), (11, 1), (30, 1), (31, 0)])
def test_same_item_purchase_interval_is_strict_left_inclusive_right(purchase_time, expected):
    _, records = cohort_case(
        [(0, "detail", "seed"), (10, "detail", "A"), (purchase_time, "purchase", "A")]
    )
    item = next(r for r in records if r.sku == "A")
    assert item.anchor_ms == 10 and item.target_end_ms == 30 and item.purchased == expected
    assert item.reason == (None if expected else "no_purchase_in_anchor_window")


@pytest.mark.parametrize("anchor,expected", [(69, True), (70, False), (71, False)])
def test_anchor_maturity_strictly_before_A_G_end_independent_of_session(anchor, expected):
    session, records = cohort_case([(0, "detail", "seed"), (anchor, "detail", "A")])
    item = next(r for r in records if r.sku == "A")
    assert session.mature and session.exclusion_reason is None
    assert item.mature is expected
    if not expected:
        assert item.purchased is None and item.reason == "anchor_maturity_boundary"


def test_each_pair_has_own_window_even_after_recommendations_stopped():
    config = replace(CONFIG, max_requests=1)
    rows = [
        (0, "detail", "seed"),
        (5, "purchase", "seed"),
        (60, "detail", "A"),
        (70, "detail", "A"),
        (75, "purchase", "A"),
    ]
    session, records = cohort_case(rows, config=config)
    assert session.request_count == 1 and session.target_end_ms == 20
    item = next(r for r in records if r.sku == "A")
    assert (item.anchor_ms, item.target_end_ms, item.purchased) == (60, 80, 1)


def test_pair_requires_own_coverage_beyond_session_window_and_closed_end():
    rows = [(0, "detail", "seed"), (60, "detail", "A")]
    session, records = cohort_case(rows, coverage=replace(COVERAGE, end_ms=89))
    assert session.mature and session.close_ms == 70 and session.exclusion_reason is None
    item = next(r for r in records if r.sku == "A")
    assert not item.mature and item.purchased is None and item.reason == "coverage_range"
    _, records = cohort_case(rows, coverage=replace(COVERAGE, end_ms=90))
    assert all(r.mature for r in records)


def test_gap_after_closed_session_but_inside_anchor_window_is_unknown():
    rows = [(0, "detail", "seed"), (60, "detail", "A")]
    session, records = cohort_case(rows, coverage=replace(COVERAGE, gaps=((80, 85),)))
    assert session.exclusion_reason is None
    item = next(r for r in records if r.sku == "A")
    assert item.purchased is None and item.reason == "coverage_gap"


@pytest.mark.parametrize(
    "coverage", [replace(COVERAGE, confirmed_at_ms=None), replace(COVERAGE, gaps=((40, 50),))]
)
def test_full_session_A_G_coverage_remains_required(coverage):
    _, records = cohort_case([(0, "detail", "A"), (60, "page", None)], coverage=coverage)
    assert all(not r.mature and r.purchased is None and r.reason for r in records)


def test_other_period_and_noneligible_session_cannot_supply_historical_pairs():
    for rows in [
        [(200, "detail", "A"), (210, "purchase", "A")],
        [(0, "purchase", "A"), (1, "detail", "A")],
    ]:
        _, records = cohort_case(rows)
        assert records and all(
            not r.mature and r.purchased is None and r.reason == "outside_A_G" for r in records
        )


def test_no_detail_has_empty_cohort_and_purchase_only_sku_is_not_a_pair():
    _, records = cohort_case([(0, "page", None), (1, "purchase", "only_purchase")])
    assert records == ()
    _, records = cohort_case([(0, "detail", "A"), (5, "purchase", "only_purchase")])
    assert [(r.sku, r.purchased) for r in records] == [("A", 0)]


def test_repeated_views_and_purchases_make_one_pair_and_no_cross_item_match():
    rows = (
        [(0, "detail", "A")] * 10
        + [(10, "detail", "A")] * 5
        + [(15, "purchase", "A")] * 10
        + [(5, "detail", "B"), (15, "purchase", "C")]
    )
    _, records = cohort_case(rows)
    assert [(r.sku, r.anchor_ms, r.purchased) for r in records] == [("A", 0, 1), ("B", 5, 0)]


def test_other_session_cannot_supply_purchase_and_record_must_match():
    blocks, requests = make_history([(0, "detail", "A")])
    session, _ = prepare_targets(blocks, requests, COVERAGE, PERIODS, CONFIG)
    other, _ = make_history([(5, "purchase", "A")], session="other")
    with pytest.raises(InputValidationError):
        historical_cohort((*blocks, *other), session, COVERAGE, PERIODS, CONFIG)
    for bad in [
        replace(session, source_namespace="other"),
        replace(session, period="B"),
        replace(session, mature=False),
        replace(session, exclusion_reason="fake"),
    ]:
        with pytest.raises(InputValidationError):
            historical_cohort(blocks, bad, COVERAGE, PERIODS, CONFIG)


def test_cohort_contract_roundtrip_and_unknown_is_never_zero():
    _, records = cohort_case([(0, "detail", "A"), (70, "detail", "B")])
    for record in records:
        assert from_json(to_json(record), CohortRecord) == record
    invalid = next(r for r in records if not r.mature)
    with pytest.raises(InputValidationError):
        replace(invalid, purchased=0)
    with pytest.raises(InputValidationError):
        replace(invalid, reason=None)
