import os
import sys

import pyarrow.parquet as pq
import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.contracts.training import SessionRecord
from coveo_recsys.data.prepared import (
    FORMAT_VERSION,
    REPORT,
    TableWriter,
    iter_sessions,
    read_report,
    table_info,
    write_report,
)


def block(sid, time=1):
    return SessionRecord(
        sid,
        time,
        "A_G",
        True,
        None,
        source_namespace="source",
        last_event_ms=time,
        target_end_ms=time,
        request_count=1,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_batches_preserve_whole_sessions(tmp_path, batch_size):
    path = tmp_path / "events.parquet"
    sessions = [
        (sid, tuple(block(sid, t) for t in range(1, n + 1)))
        for sid, n in [("a", 3), ("b", 1), ("я", 5)]
    ]
    with TableWriter(path, SessionRecord, batch_rows=2) as writer:
        for sid, records in sessions:
            writer.append(sid, records)
    assert list(iter_sessions(path, SessionRecord, batch_size=batch_size)) == sessions


def test_incomplete_table_is_not_published(tmp_path):
    path = tmp_path / "events.parquet"
    with pytest.raises(KeyboardInterrupt), TableWriter(path, SessionRecord) as writer:
        writer.append("a", (block("a"),))
        raise KeyboardInterrupt
    assert list(tmp_path.iterdir()) == []


def test_table_repeat_does_not_overwrite_changed_result(tmp_path):
    path = tmp_path / "events.parquet"
    for _ in range(2):
        with TableWriter(path, SessionRecord) as writer:
            writer.append("a", (block("a"),))
    before = path.read_bytes()
    with pytest.raises(InputValidationError), TableWriter(path, SessionRecord) as writer:
        writer.append("a", (block("a", 2),))
    assert path.read_bytes() == before


@pytest.mark.parametrize("sid", ["a", "0"])
def test_writer_rejects_repeated_or_unordered_session(tmp_path, sid):
    with (
        pytest.raises(InputValidationError),
        TableWriter(tmp_path / "x.parquet", SessionRecord) as writer,
    ):
        writer.append("a", (block("a"),))
        writer.append(sid, (block(sid),))


@pytest.mark.parametrize("limits", [{"max_session_records": 1}, {"max_session_bytes": 1}])
def test_reader_enforces_whole_session_budget(tmp_path, limits):
    path = tmp_path / "events.parquet"
    with TableWriter(path, SessionRecord) as writer:
        writer.append("a", (block("a"), block("a", 2)))
    with pytest.raises(InputValidationError):
        list(iter_sessions(path, SessionRecord, **limits))


def test_writer_rejects_foreign_session_key(tmp_path):
    with (
        pytest.raises(InputValidationError),
        TableWriter(tmp_path / "x.parquet", SessionRecord) as writer,
    ):
        writer.append("a", (block("b"),))


def test_writer_flushes_before_crossing_byte_budget(tmp_path):
    path = tmp_path / "events.parquet"
    record = block("a")
    size = len(to_json(record)) + 1
    with TableWriter(path, SessionRecord, batch_bytes=2 * size - 1) as writer:
        writer.append("a", (record, record))
    with pq.ParquetFile(path) as parquet:
        assert parquet.metadata.num_row_groups == 2


def test_reader_limits_each_batch_to_one_row_group(tmp_path, monkeypatch):
    path = tmp_path / "events.parquet"
    records = tuple(block("a", time) for time in (1, 2, 3))
    with TableWriter(path, SessionRecord, batch_rows=1) as writer:
        writer.append("a", records)
    opened = []
    original = pq.ParquetFile.iter_batches

    def batches(parquet, *args, **kwargs):
        assert len(kwargs.get("row_groups", [])) == 1
        opened.extend(kwargs["row_groups"])
        return original(parquet, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    assert list(iter_sessions(path, SessionRecord)) == [("a", records)]
    assert opened == [0, 1, 2]


@pytest.fixture
def prepared_report(tmp_path):
    path = tmp_path / "preparation/sessions.parquet"
    with TableWriter(path, SessionRecord) as writer:
        writer.append("a", (block("a"),))
    report = dict(
        format_version=FORMAT_VERSION,
        contract_version="3",
        evaluation_protocol="observed_purchases_v1",
        status="completed",
        source_sha256="a" * 64,
        inputs={name: "b" * 64 for name in ("browsing", "search", "products", "profile", "plan")},
        profile_json="{\n}",
        plan_json="{\n}",
        scope="synthetic",
        run_id="small",
        real_preflight=False,
        telemetry_complete=None,
        tables={"preparation/sessions.parquet": table_info(path, SessionRecord)},
    )
    for name in ("profile", "plan"):
        report["inputs"][name] = sha256_bytes(report[f"{name}_json"].encode("utf-8"))
    write_report(tmp_path, report)
    assert read_report(tmp_path) == report
    return tmp_path, report


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "invalid"),
        ("inputs", {}),
        ("inputs", {"browsing": "invalid"}),
        ("scope", "unverified"),
        ("run_id", ""),
        ("run_id", None),
        ("real_preflight", True),
        ("real_preflight", None),
    ],
)
def test_report_rejects_incomplete_identity(prepared_report, field, value):
    root, report = prepared_report
    report[field] = value
    (root / REPORT).write_bytes(canonical_json_bytes(report))
    with pytest.raises(InputValidationError):
        read_report(root)


@pytest.mark.parametrize("name", ["profile", "plan"])
def test_report_embedded_settings_match_exact_input_bytes(prepared_report, name):
    root, report = prepared_report
    report[f"{name}_json"] = "{}"
    (root / REPORT).write_bytes(canonical_json_bytes(report))
    with pytest.raises(InputValidationError):
        read_report(root, verify_tables=False)


@pytest.mark.parametrize("same_content", [False, True])
def test_concurrent_publication_cannot_replace_result(tmp_path, monkeypatch, same_content):
    original = os.link
    publications = []
    payload = {"result": "ours"}
    expected = canonical_json_bytes(payload) if same_content else b"concurrent result"

    def concurrent(temporary, destination):
        destination.write_bytes(expected)
        publications.append(destination)
        return original(temporary, destination)

    monkeypatch.setattr(os, "link", concurrent)
    if same_content:
        write_report(tmp_path, payload)
    else:
        with pytest.raises(InputValidationError):
            write_report(tmp_path, payload)
    assert publications == [tmp_path / REPORT]
    assert (tmp_path / REPORT).read_bytes() == expected
    assert len(list((tmp_path / "reports").iterdir())) == 1


def test_final_guard_rejects_report_after_serialization(tmp_path):
    def reject():
        assert list((tmp_path / "reports").glob("*.tmp"))
        raise InputValidationError("Исчерпан бюджет")

    with pytest.raises(InputValidationError, match="Исчерпан бюджет"):
        write_report(tmp_path, {"result": "new"}, before_publish=reject)
    assert list((tmp_path / "reports").iterdir()) == []


@pytest.mark.parametrize("writer", ["table", "report"])
@pytest.mark.parametrize("position", ["root", "parent", "file"])
def test_writer_rejects_symlink_before_creating_files(tmp_path, writer, position):
    root = tmp_path / "run"
    path = root / ("canonical/A_G.parquet" if writer == "table" else REPORT)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "keep"
    target.write_bytes(b"unchanged")
    link = {"root": root, "parent": path.parent, "file": path}[position]
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(
            target if position == "file" else outside, target_is_directory=position != "file"
        )
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("windows_symlink_unavailable: WinError 1314")
        raise
    with pytest.raises(InputValidationError):
        if writer == "table":
            with TableWriter(path, SessionRecord) as table:
                table.append("a", (block("a"),))
        else:
            write_report(root, {"result": "new"})
    assert list(outside.iterdir()) == [target]
    assert target.read_bytes() == b"unchanged"
