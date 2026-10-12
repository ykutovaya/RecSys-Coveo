"""Проверка готового входа Э2 на малой существующей подготовке Э1."""

import csv
import io
import json
import shutil
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.cli.preparation import Preparation
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.historical import HistoricalConfig
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord
from coveo_recsys.data.metadata import open_metadata_source
from coveo_recsys.historical.build import build_historical

ROOT = Path(__file__).resolve().parents[2]
REPORT = "reports/preparation.json"


@pytest.fixture(scope="module")
def prepared_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2-input")
    inputs = root / "input"
    shutil.copytree(ROOT / "tests/fixtures/e1", inputs)
    products = inputs / "sku_to_content.csv"
    products.write_text(
        products.read_text(encoding="utf-8").replace(",low", ",1").replace(",high", ",2"),
        encoding="utf-8",
    )
    # Малый CSV Э1; уменьшенный H оставляет опубликованную покупку за окном цели.
    base = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    plan_path = root / "plan.json"
    plan_path.write_bytes(
        to_json(
            replace(
                base,
                task=replace(base.task, horizon_ms=1000),
                periods=tuple(
                    replace(p, sample_size=p.sample_size if p.name == "A_G" else 0)
                    for p in base.periods
                ),
            )
        )
    )
    run = root / "prepared"
    Preparation(LocalArtifactStore(run), ROOT, "e2-input").prepare(
        inputs,
        ROOT / "configs/synthetic_csv.json",
        plan_path,
        batch_size=2,
        max_batch_chars=256,
    )
    return run


@pytest.fixture
def prepared(prepared_source, tmp_path):
    run = tmp_path / "prepared"
    shutil.copytree(prepared_source, run)
    return run


def open_input(prepared, audit):
    from coveo_recsys.data.historical import open_historical_input

    return open_historical_input(
        prepared,
        report_sha256=sha256_bytes((prepared / REPORT).read_bytes()),
        audit=audit.append,
    )


def rewrite(prepared, path, *, value=None, payload=None):
    """Перепубликовать малый повреждённый пример с согласованными новыми хешами."""
    content = canonical_json_bytes(value) if value is not None else payload
    assert content is not None
    (prepared / path).write_bytes(content)
    if path != REPORT:
        report = json.loads((prepared / REPORT).read_bytes())
        report["tables"][path].update(
            sha256=sha256_bytes(content),
            bytes=len(content),
            rows=pq.read_metadata(prepared / path).num_rows,
        )
        (prepared / REPORT).write_bytes(canonical_json_bytes(report))


def rewrite_records(prepared, path, kind, change):
    table = pq.read_table(prepared / path)
    records = [from_json(v, kind) for v in table["record_json"].to_pylist()]
    records = change(records)
    result = pa.Table.from_arrays(
        [pa.array([r.session_id for r in records]), pa.array([to_json(r) for r in records])],
        schema=table.schema,
    )
    buffer = io.BytesIO()
    pq.write_table(result, buffer)
    rewrite(prepared, path, payload=buffer.getvalue())


def test_full_input_is_repeatable_keeps_unknown_and_late_purchases(prepared):
    before = {p.relative_to(prepared): p.stat().st_mtime_ns for p in prepared.rglob("*")}
    audit = []
    source = open_input(prepared, audit)
    first = tuple(source.iter_sessions())
    assert first == tuple(source.iter_sessions())
    assert source.session_count == 2
    assert {s[0].session_id for s in first} == {"s1", "s2"}
    assert sum(len(s) for s in first) == 8
    assert any(
        e.event_type == "purchase" and b.event_time_ms > 2000
        for s in first
        for b in s
        for e in b.events
    )
    assert source.snapshot_time_ms == 200000000
    assert source.input_hashes["products"]
    assert source.input_hashes["preparation_code"]
    assert all(item["allowed"] for item in audit)
    assert sum(item.get("sessions", 0) == 2 for item in audit) >= 2
    assert len(audit) <= 4, "Журнал агрегируется по проходу, а не по сессиям"
    assert before == {p.relative_to(prepared): p.stat().st_mtime_ns for p in prepared.rglob("*")}


@pytest.mark.parametrize("status", ["failed", "interrupted", "materialized"])
def test_input_requires_completed_publication(prepared, status):
    value = json.loads((prepared / REPORT).read_bytes())
    value["status"] = status
    rewrite(prepared, REPORT, value=value)
    audit = []
    with pytest.raises(ValueError):
        open_input(prepared, audit)
    assert audit[-1]["allowed"] is False


def test_report_scope_must_match_profile_and_plan(prepared):
    value = json.loads((prepared / REPORT).read_bytes())
    value["scope"] = "actual"
    rewrite(prepared, REPORT, value=value)
    with pytest.raises(ValueError):
        tuple(open_input(prepared, []).iter_sessions())


def test_current_builder_code_does_not_replace_preparation_identity(prepared, monkeypatch):
    monkeypatch.setattr("coveo_recsys.verification.provenance.source_sha256", lambda *_: "f" * 64)
    source = open_input(prepared, [])
    assert len(tuple(source.iter_sessions())) == 2
    assert source.input_hashes["preparation_code"] != "f" * 64


@pytest.mark.parametrize("field,value", [("period", "B"), ("source_namespace", "foreign")])
def test_session_register_does_not_grant_foreign_access(prepared, field, value):
    rewrite_records(
        prepared,
        "preparation/sessions.parquet",
        SessionRecord,
        lambda rows: [replace(rows[0], **{field: value}), *rows[1:]],
    )
    with pytest.raises(ValueError):
        tuple(open_input(prepared, []).iter_sessions())


def test_selection_and_session_register_must_agree(prepared):
    value = json.loads((prepared / REPORT).read_bytes())
    value["sampled"]["selection"]["A_G"] = ["foreign", "s2"]
    rewrite(prepared, REPORT, value=value)
    with pytest.raises(ValueError):
        tuple(open_input(prepared, []).iter_sessions())


@pytest.mark.parametrize(
    "mutation", ["omitted_block", "omitted_source_row", "foreign_origin", "duplicate_origin"]
)
def test_canonical_rows_are_complete_and_keep_source_provenance(prepared, mutation):
    path = "canonical/A_G.parquet"

    def change(rows):
        if mutation == "omitted_block":
            return rows[:-1]
        block = rows[0]
        event = block.events[0]
        origin = event.origins[0]
        origins = (
            (origin.replace(origin.split(":")[1], "e" * 64),)
            if mutation == "foreign_origin"
            else event.origins[1:]
            if mutation == "omitted_source_row"
            else (origin, origin)
        )
        altered = replace(event, provenance=origins[0], origins=origins)
        return [replace(block, events=(altered, *block.events[1:])), *rows[1:]]

    rewrite_records(prepared, path, CanonicalBlock, change)
    audit = []
    with pytest.raises(ValueError):
        tuple(open_input(prepared, audit).iter_sessions())
    assert audit[-1]["allowed"] is False


def test_repeated_read_detects_changed_canonical_payload(prepared):
    audit = []
    source = open_input(prepared, audit)
    tuple(source.iter_sessions())
    (prepared / "canonical/A_G.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        tuple(source.iter_sessions())
    assert audit[-1]["allowed"] is False


def test_historical_reader_never_opens_prefixes_or_targets(prepared):
    for name in ("prefixes", "targets"):
        (prepared / f"datasets/A_G/{name}.parquet").write_bytes(b"not readable")
    assert len(tuple(open_input(prepared, []).iter_sessions())) == 2


def test_table_changed_during_iteration_is_rejected(prepared):
    audit = []
    iterator = open_input(prepared, audit).iter_sessions()
    next(iterator)
    path = prepared / "preparation/sessions.parquet"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        tuple(iterator)
    assert audit[-1]["allowed"] is False


def test_absent_root_is_not_created(tmp_path):
    from coveo_recsys.data.historical import open_historical_input

    missing = tmp_path / "absent"
    audit = []
    with pytest.raises((InputValidationError, OSError)):
        open_historical_input(
            missing,
            report_sha256="0" * 64,
            audit=audit.append,
        )
    assert not missing.exists()
    assert audit[-1]["allowed"] is False


@pytest.mark.parametrize("period_name", ["A_N", "B", "C_smoke", "C_select", "E", "D"])
def test_later_period_changes_with_new_pins_preserve_historical_tables(
    prepared_source, tmp_path, period_name
):
    source = prepared_source
    inputs = tmp_path / "changed-input"
    shutil.copytree(source.parent / "input", inputs)
    report = json.loads((source / REPORT).read_bytes())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(report["plan_json"], encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(report["profile_json"], encoding="utf-8")
    plan = from_json(report["plan_json"].encode(), PreparationPlan)
    period = next(p for p in plan.periods if p.name == period_name)
    sid = f"later-{period_name}"
    with (inputs / "browsing_train.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for index, (action, sku) in enumerate(
            [("detail", "A"), ("add", "outside-A_G"), ("purchase", "C")]
        ):
            writer.writerow([sid, f"{sid}-{index}", period.start_ms + index + 100, action, sku])
    changed_root = tmp_path / "changed-prepared"
    preparation = Preparation(LocalArtifactStore(changed_root), ROOT, f"e2-{period_name}")
    result = preparation.prepare(inputs, profile_path, plan_path, batch_size=2, max_batch_chars=256)
    baseline = open_input(prepared_source, [])
    changed = open_input(changed_root, [])
    assert baseline.input_hashes["browsing"] != changed.input_hashes["browsing"]
    assert result["period_counts"][period_name]["eligible"] == 1
    assert result["selection"][period_name] == []
    profile = json.loads(report["profile_json"])
    metadata = open_metadata_source(
        inputs / "sku_to_content.csv", profile, expected_sha256=changed.input_hashes["products"]
    )
    config = HistoricalConfig(baseline.snapshot_time_ms, graph_min_support=1)
    before = build_historical(baseline, metadata, config, work_dir=tmp_path / "before")
    after = build_historical(changed, metadata, config, work_dir=tmp_path / "after")

    def same(left, right):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif is_dataclass(left):
            for field in fields(left):
                same(getattr(left, field.name), getattr(right, field.name))
        else:
            assert left == right

    same(before.catalog, after.catalog)
    for original, rebuilt in zip(before.graphs.relations, after.graphs.relations, strict=True):
        same(original, rebuilt)
    same(before.statistics, after.statistics)
    same(before.content, after.content)
