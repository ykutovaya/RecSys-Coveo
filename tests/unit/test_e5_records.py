"""Общие таблицы Э5 сохраняют выдачу и оценку раздельно."""

import hashlib
import json
import weakref
from dataclasses import fields, replace

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts import evaluation as records
from coveo_recsys.contracts.serialization import from_json, record_schema, to_json
from coveo_recsys.contracts.training import SessionEvaluation
from tests.reference.e3_cases import request
from tests.unit.test_e5_metrics import result


def diagnostic_record(kind, payload):
    req = request()
    if issubclass(kind, records.ReplayRecord):
        return kind(req.session_id, "P", result(req), payload)
    return kind(
        req.session_id,
        "P",
        SessionEvaluation(req.session_id, None, None, None, None, None, True, 0, (), "unknown"),
        payload,
    )


def test_evaluation_records_roundtrip_without_target_in_answers(tmp_path):
    from coveo_recsys.contracts.evaluation import EvaluationRecord, ReplayRecord
    from coveo_recsys.data.prepared import TableWriter, iter_sessions

    req = request()
    answer = ReplayRecord(req.session_id, "P", result(req), '{"source_lists":{}}')
    score = EvaluationRecord(
        req.session_id,
        "P",
        SessionEvaluation(
            req.session_id, None, None, None, None, None, True, 0, (), "no_registered_purchase"
        ),
        '{"request_id":"q"}',
    )
    for record in (answer, score):
        assert from_json(to_json(record), type(record)) == record
        path = tmp_path / (type(record).__name__ + ".parquet")
        with TableWriter(path, type(record)) as writer:
            writer.append(req.session_id, (record,))
        assert list(iter_sessions(path, type(record))) == [(req.session_id, (record,))]
    assert b'"purchases"' not in to_json(answer)


@pytest.mark.parametrize("kind", [records.ReplayRecord, records.EvaluationRecord])
@pytest.mark.parametrize(
    "payload",
    [
        "null",
        "[]",
        "false",
        "1",
        '"value"',
        "{",
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
        '{"outer":[-1e999]}',
        '{"x":1,"x":2}',
        '{"outer":[{"x":1,"x":2}]}',
    ],
)
def test_diagnostic_payload_requires_unambiguous_finite_json_object(kind, payload):
    with pytest.raises(InputValidationError):
        diagnostic_record(kind, payload)


def test_evaluation_record_rejects_foreign_session_or_policy():
    from coveo_recsys.contracts.evaluation import EvaluationRecord

    score = SessionEvaluation("s", None, None, None, None, None, True, 0, (), "unknown")
    record = EvaluationRecord("s", "P", score, "{}")
    with pytest.raises(InputValidationError):
        replace(record, session_id="other")
    with pytest.raises(InputValidationError):
        replace(record, policy="unplanned")


@pytest.mark.parametrize("kind", [records.ReplayRecord, records.EvaluationRecord])
def test_diagnostic_json_is_parsed_once_for_the_lifetime_of_one_record(monkeypatch, kind):
    payload = '{"nested":[1,{"value":2.5}],"ok":true}'
    loads = records.json.loads
    calls = []

    def counted(value, *args, **kwargs):
        calls.append(value)
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(records.json, "loads", counted)
    record = diagnostic_record(kind, payload)
    first = record.parsed_json
    assert first is record.parsed_json
    assert first == {"nested": [1, {"value": 2.5}], "ok": True}
    to_json(record)
    assert calls == [payload]


@pytest.mark.parametrize(
    "kind,wire_hash,schema_hash",
    [
        (
            records.ReplayRecord,
            "f28d08fcc6fd71c10dd6d7117a82105214e3f7303675a1f5677d7c0af43193ea",
            "55643ae0f65f030997b927a3f2ec84c128c2a23c407a10c55f5223cc0d26a55d",
        ),
        (
            records.EvaluationRecord,
            "6cbc58cfa2e6e9f4840b50fcffbecf3f6e5ce2e457a7222a333a06e48c0c1c85",
            "21b47a4128330c93e812db91b3e4e64573dcd76959fb54e03b7014cf70a92a3c",
        ),
    ],
)
def test_diagnostic_cache_preserves_original_wire_schema_equality_and_slots(
    kind, wire_hash, schema_hash
):
    record = diagnostic_record(kind, '{"nested":[1,{"value":2.5}],"ok":true}')
    encoded = to_json(record)
    schema = json.dumps(record_schema(kind), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == wire_hash
    assert hashlib.sha256(schema).hexdigest() == schema_hash
    restored = from_json(encoded, kind)
    assert restored.parsed_json is not record.parsed_json
    assert restored == record and hash(restored) == hash(record)
    assert len(fields(record)) == 4 and not hasattr(record, "__dict__")
    assert b"_parsed_json" not in encoded + schema
    # Даже служебное состояние не участвует в равенстве и сериализации.
    object.__setattr__(restored, "_parsed_json", {"independent_cache": True})
    assert restored == record and hash(restored) == hash(record)
    assert to_json(restored) == encoded


@pytest.mark.parametrize("kind", [records.ReplayRecord, records.EvaluationRecord])
def test_replacing_json_revalidates_and_does_not_share_the_old_cache(kind):
    record = diagnostic_record(kind, '{"x":1}')
    field = "diagnostics_json" if kind is records.ReplayRecord else "details_json"
    changed = replace(record, **{field: '{"x":2}'})
    assert record.parsed_json == {"x": 1}
    assert changed.parsed_json == {"x": 2}
    assert changed.parsed_json is not record.parsed_json
    with pytest.raises(InputValidationError):
        replace(record, **{field: '{"x":1e999}'})


@pytest.mark.parametrize("kind", [records.ReplayRecord, records.EvaluationRecord])
@pytest.mark.parametrize("where", ["envelope", "data"])
def test_private_cache_is_not_an_accepted_wire_field(kind, where):
    envelope = json.loads(to_json(diagnostic_record(kind, "{}")))
    target = envelope if where == "envelope" else envelope["data"]
    target["_parsed_json"] = {}
    with pytest.raises(InputValidationError):
        from_json(json.dumps(envelope).encode(), kind)


@pytest.mark.parametrize("kind", [records.ReplayRecord, records.EvaluationRecord])
def test_per_instance_cache_does_not_retain_previous_records(kind):
    class ObservableRecord(kind):
        __slots__ = ("__weakref__",)

    references = []
    for index in range(30):
        record = diagnostic_record(ObservableRecord, json.dumps({"index": index}))
        assert record.parsed_json == {"index": index}
        references.append(weakref.ref(record))
        assert all(reference() is None for reference in references[:-1])
    del record
    assert all(reference() is None for reference in references)
