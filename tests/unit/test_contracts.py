import builtins
import dataclasses
import json

import jsonschema
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts import (
    CONTRACT_VERSION,
    InputValidationError,
    OutcomeStatus,
    id_bytes,
    serialization,
)
from coveo_recsys.contracts.records import (
    CanonicalBlock,
    CartItem,
    CartState,
    EventType,
    FeatureBatch,
    PrefixSnapshot,
    RawEvent,
    RequestContext,
    SearchPayload,
    SourceEvidence,
    StateTransition,
)
from coveo_recsys.contracts.serialization import from_json, record_schema, record_version, to_json
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


@pytest.mark.parametrize("expected", [str | None, int | None, SearchPayload | None])
@pytest.mark.parametrize("decode", [False, True])
def test_allowed_none_does_not_try_other_union_members(monkeypatch, expected, decode):
    convert = serialization._convert
    attempted = []

    def checked(value, candidate, **kwargs):
        attempted.append(candidate)
        return convert(value, candidate, **kwargs)

    monkeypatch.setattr(serialization, "_convert", checked)
    monkeypatch.setattr(
        serialization,
        "_type_shape",
        lambda expected: pytest.fail("Допустимый null не требует разбора типа"),
    )
    assert convert(None, expected, decode=decode) is None
    assert attempted == []


@pytest.mark.parametrize("decode", [False, True])
def test_union_without_none_rejects_null(decode):
    with pytest.raises(InputValidationError):
        serialization._convert(None, int | str, decode=decode)


def test_record_metadata_is_reused_but_values_are_checked(monkeypatch):
    item = SourceEvidence("sku", "popularity", None, 0.5, None, None, None)
    from_json(to_json(item), SourceEvidence)

    def repeated_inspection(*args):
        pytest.fail("Повторный разбор неизменного описания записи")

    with monkeypatch.context() as patched:
        patched.setattr(serialization, "get_origin", repeated_inspection)
        patched.setattr(serialization, "get_args", repeated_inspection)
        patched.setattr(serialization.dataclasses, "fields", repeated_inspection)
        changed = dataclasses.replace(item, sku="другой", score=-0.0)
        encoded = to_json(changed)
        assert from_json(encoded, SourceEvidence) == changed
        with pytest.raises(InputValidationError):
            dataclasses.replace(changed, score=float("nan"))
        with pytest.raises(InputValidationError):
            dataclasses.replace(changed, sku="")
    expected = json.dumps(
        {
            "contract_version": record_version(SourceEvidence),
            "type": "SourceEvidence",
            "data": dataclasses.asdict(changed),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert encoded == expected


def test_record_construction_does_not_repeat_validator_import(monkeypatch):
    original = builtins.__import__
    imports = []

    def observed(name, *args, **kwargs):
        if name == "serialization":
            imports.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observed)
    SourceEvidence("sku", "popularity", None, 0.5, None, None, None)
    with pytest.raises(InputValidationError):
        SourceEvidence("sku", "popularity", None, float("inf"), None, None, None)
    assert imports == []


def test_serialization_does_not_repeat_python_field_reads(monkeypatch):
    item = SourceEvidence("sku", "popularity", None, 0.5, None, None, None)
    expected = to_json(item)
    reads = []

    def observed(obj, name):
        reads.append(name)
        return getattr(obj, name)

    monkeypatch.setattr(serialization, "getattr", observed, raising=False)
    assert to_json(item) == expected
    assert reads == [field.name for field in dataclasses.fields(item)]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_serialization_handles_empty_and_single_field_records(count):
    record_type = dataclasses.make_dataclass(
        "SmallRecord", [(f"value_{i}", float) for i in range(count)], frozen=True
    )
    item = record_type(*([-0.0] * count))
    expected = json.dumps(
        {
            "contract_version": record_version(record_type),
            "type": record_type.__name__,
            "data": dataclasses.asdict(item),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert to_json(item) == expected
    assert from_json(expected, record_type) == item


@pytest.mark.parametrize(
    "item",
    [
        context(),
        SearchPayload((0.0, -0.0, 1e-300, 1e300), "поиск"),
        FeatureBatch(
            "request",
            ("e\u0301", "é"),
            ("score",),
            ("category",),
            ((-0.0,), (0.125,)),
            ((-(2**63),), (2**63 - 1,)),
            ((False, True), (True, False)),
            "a" * 64,
        ),
        TargetRecord("request", ("sku",), OutcomeStatus.KNOWN, 100),
    ],
)
def test_json_bytes_match_asdict_encoding(item):
    expected = json.dumps(
        {
            "contract_version": record_version(type(item)),
            "type": type(item).__name__,
            "data": dataclasses.asdict(item),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert to_json(item) == expected
    assert from_json(expected, type(item)) == item


@pytest.mark.parametrize("bad", [True, 1, float("nan"), float("inf"), "1"])
def test_nested_float_validation_does_not_coerce_values(bad):
    with pytest.raises(InputValidationError):
        SearchPayload((bad,), None)


def test_serialization_revalidates_values_after_an_earlier_success():
    item = SearchPayload((0.5,), None)
    to_json(item)
    object.__setattr__(item, "vector", (True,))
    with pytest.raises(InputValidationError):
        to_json(item)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", 1])
def test_nested_float_json_accepts_only_finite_numbers(value):
    payload = json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "type": "SearchPayload",
            "data": {"vector": [value], "reason": None},
        }
    ).encode()
    if type(value) is int:
        result = from_json(payload, SearchPayload)
        assert result == SearchPayload((1.0,), None)
        assert type(result.vector[0]) is float
    else:
        with pytest.raises(InputValidationError):
            from_json(payload, SearchPayload)


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
    unknown = TargetRecord("r", (), OutcomeStatus.UNKNOWN, 100, "no_registered_purchase")
    assert from_json(to_json(unknown), TargetRecord) == unknown
    with pytest.raises(InputValidationError):
        dataclasses.replace(unknown, status=OutcomeStatus.KNOWN)
    with pytest.raises(InputValidationError):
        dataclasses.replace(unknown, purchases=("sku",))


def test_empty_target_id_and_metric_consistency():
    with pytest.raises(InputValidationError):
        TargetRecord("r", ("",), OutcomeStatus.KNOWN, 100)
    with pytest.raises(InputValidationError):
        MetricResult(0.0, "passed", None, "per_1000", 1, (), None, True, ())
    with pytest.raises(InputValidationError):
        MetricResult(0.0, "passed", None, "per_1000", 1, (("n", -1),), None, False, ())


def test_review_counterexamples():
    with pytest.raises(InputValidationError):
        TargetRecord("q", (), OutcomeStatus.KNOWN, 100)
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
