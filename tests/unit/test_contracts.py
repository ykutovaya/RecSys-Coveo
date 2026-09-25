import dataclasses
import json

import jsonschema
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts import CONTRACT_VERSION, InputValidationError, OutcomeStatus, id_bytes
from coveo_recsys.contracts.records import (
    CanonicalBlock,
    CartItem,
    CartState,
    EventType,
    PrefixSnapshot,
    RawEvent,
    RequestContext,
    SearchPayload,
    StateTransition,
)
from coveo_recsys.contracts.serialization import from_json, record_schema, to_json
from coveo_recsys.contracts.training import MetricResult, SessionEvaluation, TargetRecord


def context(time_ms=2**63 - 1, sku="e\u0301"):
    event = RawEvent(
        "fixture", "сессия", "event", time_ms, time_ms, EventType.DETAIL, sku, None, "synthetic:1"
    )
    block = CanonicalBlock(
        "block", "сессия", time_ms, time_ms, (event,), ((event.event_id, 1),), ()
    )
    prefix = PrefixSnapshot(
        "сессия",
        1,
        time_ms,
        time_ms,
        (block,),
        (),
        None,
        source_namespace="fixture",
        first_event_time_ms=time_ms,
    )
    return RequestContext("request", "сессия", "block", time_ms, time_ms, 1, prefix, "a" * 64)


def test_roundtrip_and_immutable_prefix():
    item = context()
    assert from_json(to_json(item), RequestContext) == item
    jsonschema.Draft202012Validator(record_schema(RequestContext)).validate(
        json.loads(to_json(item))
    )
    assert item.prefix.blocks[0].events[0].sku.encode("utf-8") == b"e\xcc\x81"
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.prefix.version = 2
    assert not {"targets", "labels", "b_s", "session_request_count"} & {
        f.name for f in dataclasses.fields(RequestContext)
    }


@pytest.mark.parametrize("bad", [True, 1.5, 2**63, -(2**63) - 1, "1700000000000"])
def test_invalid_time(bad):
    with pytest.raises(InputValidationError):
        context(bad)


def test_no_future_or_mutable_prefix():
    item = context(100)
    with pytest.raises(InputValidationError):
        dataclasses.replace(item.prefix, event_cutoff_ms=99)
    with pytest.raises(InputValidationError):
        dataclasses.replace(item.prefix, availability_cutoff_ms=99)
    with pytest.raises(InputValidationError):
        dataclasses.replace(item.prefix, blocks=list(item.prefix.blocks))
    with pytest.raises(InputValidationError):
        dataclasses.replace(item.prefix, cart=(CartItem("sku", CartState.ADD, 101, 101),))


def test_partial_measurements_keep_known_values():
    item = SessionEvaluation("s", 1, 1, None, 0, None, True, 0, (), "trace_incomplete")
    assert from_json(to_json(item), SessionEvaluation) == item
    with pytest.raises(InputValidationError):
        dataclasses.replace(item, unknown_reason=None)


def test_unknown_is_not_zero():
    unknown = TargetRecord("r", None, False, OutcomeStatus.UNKNOWN, 100, None, "immature")
    assert from_json(to_json(unknown), TargetRecord) == unknown
    with pytest.raises(InputValidationError):
        dataclasses.replace(unknown, b_s=0)
    with pytest.raises(InputValidationError):
        dataclasses.replace(unknown, purchases=())


def test_empty_target_id_and_metric_consistency():
    with pytest.raises(InputValidationError):
        TargetRecord("r", ("",), True, OutcomeStatus.KNOWN, 100, 1, None)
    with pytest.raises(InputValidationError):
        MetricResult(0.0, "passed", None, "per_1000", 1, (), None, True, ())
    with pytest.raises(InputValidationError):
        MetricResult(0.0, "passed", None, "per_1000", 1, (("n", -1),), None, False, ())


def test_review_counterexamples():
    with pytest.raises(InputValidationError):
        TargetRecord("q", ("sku",), True, OutcomeStatus.KNOWN, 100, 0, None)
    with pytest.raises(InputValidationError):
        SessionEvaluation("s", None, 0, 0, 0, None, False, 0, (), "immature")
    with pytest.raises(InputValidationError):
        CartItem(chr(0xD800), CartState.UNKNOWN, None)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "type": "SearchPayload",
        "data": {"vector": [10**400], "reason": None},
    }
    with pytest.raises(InputValidationError):
        from_json(json.dumps(payload).encode(), SearchPayload)
    item = context()
    with pytest.raises(InputValidationError):
        StateTransition("other", 0, item.prefix, (), "receipt")
    with pytest.raises(InputValidationError):
        StateTransition(item.session_id, -1, item.prefix, (), "receipt")


def test_reject_injected_training_field_and_duplicates():
    document = json.loads(to_json(context()))
    document["data"]["labels"] = [1]
    with pytest.raises(InputValidationError):
        from_json(json.dumps(document).encode(), RequestContext)
    with pytest.raises(InputValidationError):
        from_json(b'{"type":"RequestContext","type":"TargetRecord"}', RequestContext)


@seed(17)
@given(
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1),
)
def test_property_lossless_time_and_utf8(time_ms, sku):
    item = context(time_ms, sku)
    actual = from_json(to_json(item), RequestContext)
    assert actual.t_q == time_ms
    assert actual.prefix.blocks[0].events[0].sku.encode("utf-8") == sku.encode("utf-8")


def test_id_order_is_exact_utf8():
    assert sorted(["é", "e\u0301", "Я", "A"], key=id_bytes) == ["A", "e\u0301", "é", "Я"]
