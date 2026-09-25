import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import RawEvent
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.io import (
    DEFAULT_MAX_BATCH_CHARS,
    FILES,
    inspect_inputs,
    iter_events,
    load_profile,
    row_batches,
    timestamp,
    vector_payload,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/synthetic_csv.json"
FIXTURES = ROOT / "tests/fixtures/e1"


@pytest.mark.parametrize("batch_size", [1, 2, 7, 1000])
def test_batches_preserve_rows_and_exact_ids(batch_size):
    profile = load_profile(PROFILE)
    events = list(
        iter_events(FIXTURES / FILES["browsing"], "browsing", profile, batch_size=batch_size)
    )
    assert len(events) == 9
    assert events[0].transport_id == "e1"
    assert events[0].event_time_ms == events[0].available_time_ms == 1000
    assert from_json(to_json(events[0]), RawEvent) == events[0]
    report = inspect_inputs(
        {k: FIXTURES / v for k, v in FILES.items()}, profile, batch_size=batch_size
    )
    assert report["files"]["browsing"]["rows"] == 9
    assert not report["files"]["browsing"]["coverage_confirmed"]
    assert report["files"]["browsing"]["serialized_event_bytes"] == sum(
        len(to_json(e)) + 1 for e in events
    )
    assert report["disk_plan"]["input_bytes"] == sum(
        (FIXTURES / name).stat().st_size for name in FILES.values()
    )
    assert report["disk_plan"]["sorting_spill_estimate_bytes"] == 3 * sum(
        item["serialized_event_bytes"] for item in report["files"].values()
    )
    assert report["disk_plan"]["estimate_only"] is True
    assert report["memory_plan"]["max_batch_chars"] == DEFAULT_MAX_BATCH_CHARS


@pytest.mark.parametrize(
    "payload,reason",
    [
        ("", "vector_absent"),
        ("[0,0]", "vector_zero"),
        ("[NaN,0]", "vector_invalid"),
        ("[true,1]", "vector_invalid"),
        ("[1]", "vector_invalid"),
        ("[1e999,0]", "vector_invalid"),
    ],
)
def test_external_vector_markers(payload, reason):
    assert vector_payload(payload, 2).reason == reason
    assert vector_payload("[2,-0.0]", 2).vector == (2.0, 0.0)


@pytest.mark.parametrize(
    "body",
    [
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\ns,e,1.5,detail,A\n",
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\ns,e,9223372036854775808,detail,A\n",
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\ns,e,0,detail,\n",
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\ns,e,0,detail,A,extra\n",
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\ns,e,0,unknown,A\n",
        "session_id,event_id,event_id\ns,e,e\n",
    ],
)
def test_malformed_rows_fail(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(InputValidationError):
        list(iter_events(path, "browsing", load_profile(PROFILE)))


def test_technical_id_is_not_transport_confirmation(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\n"
        "001,,1,detail,е́\n001,,1,detail,е́\n",
        encoding="utf-8",
    )
    rows = list(iter_events(path, "browsing", load_profile(PROFILE)))
    assert rows[0].event_id != rows[1].event_id
    assert rows[0].transport_id is rows[1].transport_id is None
    assert rows[0].session_id == "001"
    envelope = json.loads(to_json(rows[0]))
    envelope["contract_version"] = "1"
    with pytest.raises(InputValidationError):
        from_json(json.dumps(envelope).encode(), RawEvent)


def test_profile_is_explicit_and_synthetic(tmp_path):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["scope"] = "actual"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_profile(path)


def test_profile_cannot_diverge_from_hash():
    profile = load_profile(PROFILE)
    with pytest.raises(TypeError):
        profile.dimensions["query"] = 3
    with pytest.raises(TypeError):
        profile.columns["browsing"]["session_id"] = "other"
    assert vector_payload("[100000000000000000000,0]", 2) == vector_payload("[1e20,0]", 2)
    assert vector_payload(" null ", 2).reason == "vector_absent"
    assert vector_payload("[" * 1200 + "0" + "]" * 1200, 2).reason == "vector_invalid"


def test_readers_do_not_share_field_limit(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\n"
        "s,e,1,detail,A\ns,f,2,detail," + "A" * 50 + "\n",
        encoding="utf-8",
    )
    profile = load_profile(PROFILE)
    original = csv.field_size_limit()
    wide = row_batches(path, "browsing", replace(profile, max_field_chars=100), batch_size=1)
    narrow = row_batches(path, "browsing", replace(profile, max_field_chars=32), batch_size=1)
    next(wide)
    next(narrow)
    assert len(next(wide)[0][1]["sku"]) == 50
    assert csv.field_size_limit() == original
    with pytest.raises(InputValidationError):
        next(narrow)
    wide.close()
    assert csv.field_size_limit() == original
    with pytest.raises(InputValidationError):
        timestamp("9" * 4301)


@pytest.mark.parametrize("batch_size", [1, 2, 1000])
def test_batch_character_budget_preserves_whole_rows(tmp_path, batch_size):
    path = tmp_path / "rows.csv"
    rows = [["s", f"e{i}", str(i), "detail", "я\n" + "A" * (i + 1)] for i in range(6)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["session_id", "event_id", "server_timestamp_epoch_ms", "product_action", "sku"]
        )
        writer.writerows(rows)
    profile = load_profile(PROFILE)
    budget = max(sum(map(len, row)) for row in rows) * 2
    batches = list(
        row_batches(path, "browsing", profile, batch_size=batch_size, max_batch_chars=budget)
    )
    assert all(
        len(batch) <= batch_size and sum(sum(map(len, row.values())) for _, row in batch) <= budget
        for batch in batches
    )
    expected = [
        item for batch in row_batches(path, "browsing", profile, batch_size=1) for item in batch
    ]
    assert [item for batch in batches for item in batch] == expected
    assert [row["sku"] for batch in batches for _, row in batch] == [row[-1] for row in rows]
    if batch_size == 1000:
        assert len(batches) > 1
    limited = list(
        iter_events(path, "browsing", profile, batch_size=batch_size, max_batch_chars=budget)
    )
    assert limited == list(iter_events(path, "browsing", profile, batch_size=1))


def test_batch_limit_boundary_and_oversized_row_are_explicit(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "session_id,event_id,server_timestamp_epoch_ms,product_action,sku\n"
        "s,e,0,detail,A\ns,f,1,detail,AB\n",
        encoding="utf-8",
    )
    profile = load_profile(PROFILE)
    original = csv.field_size_limit()
    reader = row_batches(path, "browsing", profile, batch_size=1000, max_batch_chars=10)
    assert next(reader)[0][1]["sku"] == "A"  # Ровно десять символов полей.
    assert csv.field_size_limit() == original
    with pytest.raises(InputValidationError, match="предел символов порции"):
        next(reader)
    assert csv.field_size_limit() == original
    reader.close()
    assert csv.field_size_limit() == original
    with pytest.raises(InputValidationError):
        list(row_batches(path, "browsing", replace(profile, max_row_chars=9), max_batch_chars=100))


@pytest.mark.parametrize("budget", [0, -1, True, 2**63])
def test_invalid_batch_character_limit(budget):
    with pytest.raises(InputValidationError):
        list(
            row_batches(
                FIXTURES / FILES["browsing"],
                "browsing",
                load_profile(PROFILE),
                max_batch_chars=budget,
            )
        )


def test_inspection_records_selected_batch_budget_and_keeps_disk_plan():
    profile = load_profile(PROFILE)
    paths = {key: FIXTURES / name for key, name in FILES.items()}
    ordinary = inspect_inputs(paths, profile)
    limited = inspect_inputs(paths, profile, batch_size=2, max_batch_chars=256)
    assert limited["files"] == ordinary["files"]
    assert limited["disk_plan"] == ordinary["disk_plan"]
    assert limited["memory_plan"]["max_batch_chars"] == 256
    assert limited["memory_plan"]["max_pending_row_chars"] == 256
    assert limited["memory_plan"]["batch_char_measure"] == "sum_of_decoded_field_lengths"
