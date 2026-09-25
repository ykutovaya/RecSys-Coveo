import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from coveo_recsys.contracts.common import InputValidationError
from coveo_recsys.data.envelope import EnvelopeLimits, validate_sealed_block
from tests.reference.e1_envelope_oracle import event, expected_time_reason, packet


def validate(wire, payload, **options):
    arguments = {
        "decision_time_ms": 1000,
        "allowed_event_time_range": (-(2**62), 2**62),
        "expected_namespace": "source",
        "query_dimension": 2,
    }
    arguments.update(options)
    return validate_sealed_block(wire, payload, **arguments)


def test_static_success_is_immutable_not_receipt_or_session_closure():
    wire, payload = packet()
    result = validate(wire, payload)
    assert result.static_valid and result.issues == ()
    block = result.block
    assert block is not None and block.envelope_bytes is wire
    assert block.envelope_sha256 == hashlib.sha256(wire).hexdigest()
    assert block.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert block.events[0].transport_id == "e0"
    assert block.events[0].available_time_ms == 1000
    assert not hasattr(result, "accepted") and not hasattr(block, "coverage_complete")
    with pytest.raises(FrozenInstanceError):
        block.block_id = "changed"


@pytest.mark.parametrize(
    "change",
    [
        {"closed": False},
        {"closed": 1},
        {"event_count": 0},
        {"event_count": True},
        {"event_count": 1.0},
        {"event_count": 2},
        {"event_ids": []},
        {"event_ids": ["other"]},
        {"event_ids": "e0"},
        {"event_ids": [None]},
        {"event_ids": [[]]},
        {"source_namespace": "wrong"},
        {"source_namespace": ""},
        {"session_id": ""},
        {"block_id": ""},
        {"block_id": "\ud800"},
        {"payload_sha256": "x"},
        {"payload_sha256": "a" * 64},
        {"unknown": 1},
    ],
)
def test_rejects_invalid_header(change):
    if "\ud800" in change.values():
        wire, payload = packet()
        wire = wire.replace(b'"block_id":"block"', b'"block_id":"\\ud800"')
    else:
        wire, payload = packet(overrides=change)
    result = validate(wire, payload)
    assert not result.static_valid and result.block is None and result.issues


@pytest.mark.parametrize("time_ms", [True, 1.0, "1000", None, -(2**63) - 1, 2**63])
def test_timestamp_is_exact_int64(time_ms):
    wire, payload = packet(overrides={"event_time_ms": time_ms})
    assert validate(wire, payload).block is None


@pytest.mark.parametrize(
    "time_ms,decision_ms,calendar",
    [
        (1000, 1000, (1000, 1001)),
        (1000, 1000, (999, 1000)),
        (1000, 1000, (1001, 1002)),
        (1000, 1_801_000, (0, 2_000_000)),
        (1000, 1_801_001, (0, 2_000_000)),
        (1001, 1000, (0, 2000)),
        (-(2**63), 2**63 - 1, (-(2**63), 2**63 - 1)),
    ],
)
def test_time_boundaries(time_ms, decision_ms, calendar):
    wire, payload = packet([event(time_ms=time_ms)], overrides={"event_time_ms": time_ms})
    result = validate(
        wire, payload, decision_time_ms=decision_ms, allowed_event_time_range=calendar
    )
    expected = expected_time_reason(time_ms, decision_ms, calendar)
    assert result.static_valid == (expected is None)
    if expected is not None:
        assert result.issues[0].code == expected


@pytest.mark.parametrize(
    "change",
    [
        {"source_namespace": "other"},
        {"session_id": "other"},
        {"event_time_ms": 999},
        {"event_time_ms": True},
        {"event_id": ""},
        {"sku": None},
        {"sku": 5},
        {"event_type": "bad"},
        {"event_type": []},
        {"available_time_ms": 999},
        {"provenance": "trusted"},
        {"query_vector": [1.0, 2.0]},
    ],
)
def test_invalid_second_event_rejects_entire_block(change):
    bad = event("e1")
    bad.update(change)
    wire, payload = packet([event(), bad])
    result = validate(wire, payload)
    assert result.block is None and not result.static_valid


def test_duplicate_ids_differ_from_duplicate_semantics():
    assert validate(*packet([event(), event()])).block is None
    result = validate(*packet([event(), event("e1")]))
    assert result.static_valid and len(result.block.events) == 2


@pytest.mark.parametrize("count,valid", [(0, False), (1, True), (100, True), (101, False)])
def test_raw_count_limit_before_canonicalization(count, valid):
    # Все события семантически совпадают: логический размер не отменяет сырой предел.
    result = validate(*packet([event(str(i)) for i in range(count)]))
    assert result.static_valid is valid


@pytest.mark.parametrize(
    "sku,valid", [("a" * 256, True), ("a" * 257, False), ("я" * 128, True), ("я" * 129, False)]
)
def test_sku_limit_counts_utf8_bytes(sku, valid):
    assert validate(*packet([event(sku=sku)])).static_valid is valid


def test_size_includes_header_and_whitespace_not_only_events():
    wire, payload = packet()
    limits = EnvelopeLimits(max_request_bytes=len(wire))
    assert validate(wire, payload, limits=limits).static_valid
    assert not validate(wire + b" ", payload, limits=limits).static_valid
    oversized = wire + b" " * (1_048_577 - len(wire))
    assert validate(oversized, payload).issues[0].code == "input_limit"


def test_exact_payload_not_equivalent_reserialization():
    compact, compact_payload = packet()
    pretty, pretty_payload = packet(compact=False)
    assert validate(compact, compact_payload).static_valid
    assert validate(pretty, pretty_payload).static_valid
    assert not validate(pretty, compact_payload).static_valid
    changed = pretty.replace(
        hashlib.sha256(pretty_payload).hexdigest().encode(),
        hashlib.sha256(compact_payload).hexdigest().encode(),
    )
    assert not validate(changed, pretty_payload).static_valid
    assert validate(compact, b" " + compact_payload).block is None


def test_payload_span_handles_field_order_escaped_key_and_nested_strings():
    wire, payload = packet([event(sku='ключ"events":[]')])
    header = wire[: -(len(payload) + len(b',"events":') + 1)] + b"}"
    # Переносим events в начало; исходный строковый SKU не является полем верхнего уровня.
    changed = b'{"\\u0065vents" : \n' + payload + b"," + header[1:]
    assert validate(changed, payload).static_valid


@pytest.mark.parametrize(
    "wire",
    [
        b"null",
        b"[]",
        b"{",
        b"\xff",
        b"{}",
        b'{"x":NaN}',
        b'{"x":1e999}',
        b'{"x":1,"x":2}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_malformed_json_returns_rejection(wire):
    result = validate(wire, b"[]")
    assert result.block is None and result.issues[0].code == "invalid_input"


def test_duplicate_nested_keys_and_missing_fields_rejected():
    wire, payload = packet()
    duplicate = payload.replace(b'"sku":"A"', b'"sku":"A","sku":"B"')
    changed = wire.replace(payload, duplicate).replace(
        hashlib.sha256(payload).hexdigest().encode(), hashlib.sha256(duplicate).hexdigest().encode()
    )
    assert not validate(changed, duplicate).static_valid
    assert not validate(wire.replace(b'"closed":true,', b""), payload).static_valid


@pytest.mark.parametrize(
    "query,reason",
    [
        (None, "vector_absent"),
        ([], "vector_invalid"),
        ([1], "vector_invalid"),
        ([0, 0], "vector_zero"),
        ([True, 2], "vector_invalid"),
        ({"x": 1}, "vector_invalid"),
    ],
)
def test_bad_search_vector_is_masked_without_rejecting_block(query, reason):
    result = validate(*packet([event(kind="search", sku=None, query_vector=query)]))
    assert result.static_valid
    search = result.block.events[0].search
    assert search.vector is None and search.reason
    assert search.reason == reason


@pytest.mark.parametrize(
    "number,valid",
    [
        (b"1e999", True),
        (b"-1e999", True),
        (b"9" * 5000, True),
        (b"-" + b"9" * 5000, True),
        (b"NaN", False),
        (b"Infinity", False),
        (b"-Infinity", False),
    ],
)
def test_external_vector_overflow_keeps_other_events_but_non_json_tokens_reject(number, valid):
    wire, payload = packet(
        [event(), event("search", kind="search", sku=None, query_vector=[12345, 0])]
    )
    changed_payload = payload.replace(b"12345", number)
    changed_wire = wire.replace(payload, changed_payload).replace(
        hashlib.sha256(payload).hexdigest().encode(),
        hashlib.sha256(changed_payload).hexdigest().encode(),
    )
    result = validate(changed_wire, changed_payload)
    assert result.static_valid is valid
    if valid:
        assert len(result.block.events) == 2
        assert result.block.events[0].sku == "A"
        assert result.block.events[1].search.vector is None
        assert result.block.events[1].search.reason == "vector_invalid"
    else:
        assert result.block is None


@pytest.mark.parametrize(
    "field",
    [
        "event_time_ms",
        "event_count",
        "source_namespace",
        "session_id",
        "block_id",
        "closed",
        "payload_sha256",
    ],
)
def test_overflow_in_required_header_field_still_rejects(field):
    wire, payload = packet()
    fields = json.loads(wire)
    value = json.dumps(fields[field], separators=(",", ":")).encode()
    changed_wire = wire.replace(
        b'"' + field.encode() + b'":' + value, b'"' + field.encode() + b'":1e999', 1
    )
    assert not validate(changed_wire, payload).static_valid


@pytest.mark.parametrize("field", ["event_count", "event_time_ms"])
def test_huge_integer_in_required_header_field_is_not_masked(field):
    wire, payload = packet()
    fields = json.loads(wire)
    value = str(fields[field]).encode()
    changed_wire = wire.replace(
        b'"' + field.encode() + b'":' + value, b'"' + field.encode() + b'":' + b"9" * 5000, 1
    )
    assert not validate(changed_wire, payload).static_valid


def test_huge_integer_event_time_with_correct_digest_rejects_whole_block():
    wire, payload = packet([event(), event("e1")])
    changed_payload = payload.replace(b'"event_time_ms":1000', b'"event_time_ms":' + b"9" * 5000, 1)
    changed_wire = wire.replace(payload, changed_payload).replace(
        hashlib.sha256(payload).hexdigest().encode(),
        hashlib.sha256(changed_payload).hexdigest().encode(),
    )
    assert validate(changed_wire, changed_payload).block is None


def test_search_keeps_original_coordinates_and_discards_only_checked_lists():
    rows = [
        event(
            "e0",
            kind="search",
            sku=None,
            query_vector=[3, -0.0],
            product_skus_hash=["a"] * 1000,
            clicked_skus_hash=[],
        ),
        event("e1", kind="search", sku=None, query_vector=[6, 0.0]),
    ]
    result = validate(*packet(rows))
    assert result.static_valid
    assert result.block.events[0].search.vector == (3.0, 0.0)
    assert result.block.events[1].search.vector == (6.0, 0.0)
    assert not hasattr(result.block.events[0], "product_skus_hash")
    rows[0]["product_skus_hash"].append("a")
    assert not validate(*packet(rows)).static_valid


@pytest.mark.parametrize("values", [None, {}, "[]", [None], ["a" * 257]])
def test_search_list_structure_and_sku_limits(values):
    wire, payload = packet(
        [event(kind="search", sku=None, query_vector=[1, 2], clicked_skus_hash=values)]
    )
    assert not validate(wire, payload).static_valid


def test_search_requires_explicit_payload_and_no_product_sku():
    assert not validate(*packet([event(kind="search", sku=None)])).static_valid
    assert not validate(*packet([event(kind="search", query_vector=[1, 2])])).static_valid


def test_sender_cannot_supply_availability_and_new_late_block_is_not_retry():
    wire, payload = packet([event(available_time_ms=1000)])
    assert not validate(wire, payload).static_valid
    wire, payload = packet()
    first = validate(wire, payload, decision_time_ms=1100)
    assert first.block.events[0].available_time_ms == 1100
    assert validate(wire, payload, decision_time_ms=1_801_001).issues[0].code == "timestamp_late"


@pytest.mark.parametrize(
    "arguments",
    [
        {"query_dimension": 0},
        {"query_dimension": True},
        {"decision_time_ms": 1.0},
        {"expected_namespace": ""},
        {"allowed_event_time_range": (1, 1)},
        {"allowed_event_time_range": [0, 2]},
    ],
)
def test_invalid_trusted_configuration_raises(arguments):
    with pytest.raises(InputValidationError):
        validate(*packet(), **arguments)


def test_invalid_limits_and_nonbyte_input():
    with pytest.raises(InputValidationError):
        EnvelopeLimits(max_request_bytes=0)
    with pytest.raises(InputValidationError):
        EnvelopeLimits(late_event_limit_ms=-1)
    wire, payload = packet()
    assert not validate(bytearray(wire), payload).static_valid
    assert not validate(wire, json.loads(payload)).static_valid
