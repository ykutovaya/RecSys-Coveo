from dataclasses import fields, replace
from pathlib import Path

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import (
    Coverage,
    Period,
    PreparationPlan,
    PreparedRequest,
    TaskConfig,
)
from coveo_recsys.contracts.records import CartItem, CartState
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord


def test_preparation_configuration_roundtrip():
    config = TaskConfig()
    assert config.max_requests == 1
    assert config.evaluation_protocol == "observed_purchases_v1"
    assert from_json(to_json(config), TaskConfig) == config
    assert not {"catalog_sha256", "b_s", "session_request_count", "targets", "mature"} & {
        item.name for item in fields(PreparedRequest)
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_ms": True},
        {"max_requests": 0},
        {"max_requests": 2},
        {"evaluation_protocol": "legacy"},
        {"query_dimension": -1},
        {"event_semantics": "unknown"},
    ],
)
def test_bad_preparation_config(kwargs):
    with pytest.raises(InputValidationError):
        TaskConfig(**kwargs)


def test_coverage_is_explicit_and_ordered():
    coverage = Coverage("synthetic", 0, 100, ((10, 20),), "fixture declaration", 100)
    assert from_json(to_json(coverage), Coverage) == coverage
    for gaps in (((20, 30), (10, 15)), ((0, 101),), ((10, 10),)):
        with pytest.raises(InputValidationError):
            Coverage("synthetic", 0, 100, gaps, "fixture declaration", 100)
    with pytest.raises(InputValidationError):
        Coverage("synthetic", 0, 100, (), "", None)
    with pytest.raises(InputValidationError):
        Period("D", 0, 10, -1)


def test_cart_and_maturity_invariants():
    with pytest.raises(InputValidationError):
        CartItem("A", CartState.ADD, 1, None)
    with pytest.raises(InputValidationError):
        CartItem("A", CartState.REMOVE, 2, 1)
    with pytest.raises(InputValidationError):
        CartItem("A", CartState.UNKNOWN, 2, 2)
    assert CartItem("A", CartState.UNKNOWN, 2, 1).last_unambiguous_ms == 1
    session = SessionRecord(
        "s",
        0,
        "B",
        True,
        None,
        source_namespace="n",
        last_event_ms=200,
        target_end_ms=100,
        request_count=1,
    )
    assert session.telemetry_complete is None and session.admitted
    for changes in (
        {"telemetry_complete": True},
        {"admission_ms": None},
        {"request_count": 2},
        {"target_end_ms": 201},
        {"exclusion_reason": "excluded"},
    ):
        with pytest.raises(InputValidationError):
            replace(session, **changes)


def test_contradictory_canonical_and_prefix_records_rejected():
    from tests.unit.test_e1_state import block, prefixes

    detail = block(0, [("detail", "A")])
    duplicate = replace(detail.events[0], event_id="another")
    with pytest.raises(InputValidationError):
        replace(
            detail,
            events=(*detail.events, duplicate),
            raw_multiplicities=(*detail.raw_multiplicities, ("another", 1)),
        )
    search = block(0, [("search", None)])
    with pytest.raises(InputValidationError):
        replace(search, search_multiplicities=(100,))
    purchase = block(1, [("purchase", "B")])
    prefix = prefixes([detail, purchase])[-1]
    for changes in (
        {"purchased_skus": ()},
        {"purchased_skus": ("B", "B")},
        {"last_views": (("", 0),)},
        {"counters": (("detail", "A", 1), ("detail", "A", 2))},
    ):
        with pytest.raises(InputValidationError):
            replace(prefix, **changes)
    with pytest.raises(InputValidationError):
        SessionRecord(
            "s",
            10,
            "B",
            True,
            None,
            source_namespace="n",
            last_event_ms=5,
            target_end_ms=3,
        )


def test_versioned_preparation_plan():
    path = Path(__file__).resolve().parents[2] / "configs/synthetic_preparation.json"
    plan = from_json(path.read_bytes(), PreparationPlan)
    assert from_json(to_json(plan), PreparationPlan) == plan
    assert replace(plan, scope="actual", diagnostic_sessions=0).scope == "actual"
    old = to_json(plan).replace(b'"contract_version":"3"', b'"contract_version":"2"')
    with pytest.raises(InputValidationError):
        from_json(old, PreparationPlan)
    for changes in (
        {"seed": 18},
        {"periods": plan.periods[:-1]},
        {"max_sessions": 0},
        {"fan_in": 1},
        {"source_namespace": "foreign"},
    ):
        with pytest.raises(InputValidationError):
            replace(plan, **changes)
