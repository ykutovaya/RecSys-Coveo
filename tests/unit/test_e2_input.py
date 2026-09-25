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
    result = Preparation(LocalArtifactStore(run), ROOT, "e2-input").prepare(
        inputs,
        ROOT / "configs/synthetic_csv.json",
        plan_path,
        batch_size=2,
        max_batch_chars=256,
    )
    return run, result["execution_path"]


@pytest.fixture
def prepared(prepared_source, tmp_path):
    source, execution = prepared_source
    run = tmp_path / "prepared"
    shutil.copytree(source, run)
    return run, execution


def open_input(prepared, audit):
    from coveo_recsys.data.historical import open_historical_input

    root, execution = prepared
    store = LocalArtifactStore(root)
    descriptor = store.describe(REPORT)
    return open_historical_input(
        root,
        report_sha256=descriptor.sha256,
        report_descriptor_sha256=descriptor.descriptor_sha256,
        execution_path=execution,
        execution_descriptor_sha256=store.describe(execution).descriptor_sha256,
        audit=audit.append,
    )


def rewrite(prepared, path, *, value=None, payload=None, descriptor_changes=None):
    """Перепубликовать малый повреждённый пример с согласованными новыми хешами."""
    root, _ = prepared
    descriptors = {
        str(p.relative_to(root)).replace("\\", "/")[: -len(".artifact.json")]: json.loads(
            p.read_bytes()
        )
        for p in root.rglob("*.artifact.json")
    }
    updated = {}

    def visit(name):
        if name in updated:
            return updated[name]
        descriptor = descriptors[name]
        dependencies = {k: visit(k) for k in descriptor["artifact_dependencies"]}
        content = (root / name).read_bytes()
        if name == path:
            if value is not None:
                content = canonical_json_bytes(value)
            if payload is not None:
                content = payload
            descriptor.update(descriptor_changes or {})
        if name == REPORT:
            report = json.loads(content)
            report["artifacts"] = {k: visit(k) for k in report["artifacts"]}
            content = canonical_json_bytes(report)
        descriptor.update(
            artifact_dependencies=dependencies,
            sha256=sha256_bytes(content),
            size_bytes=len(content),
        )
        encoded = canonical_json_bytes(descriptor)
        digest = sha256_bytes(encoded)
        (root / name).write_bytes(content)
        (root / (name + ".artifact.json")).write_bytes(encoded)
        (root / (name + ".artifact.sha256")).write_text(digest + "\n", encoding="ascii")
        (root / ".objects" / descriptor["sha256"]).write_bytes(content)
        updated[name] = digest
        return digest

    for name in descriptors:
        visit(name)


def rewrite_records(prepared, path, kind, change):
    store = LocalArtifactStore(prepared[0])
    table = store.read_table(path)
    records = [from_json(v, kind) for v in table["record_json"].to_pylist()]
    records = change(records)
    result = pa.Table.from_arrays([pa.array([to_json(r) for r in records])], schema=table.schema)
    buffer = io.BytesIO()
    pq.write_table(result, buffer)
    rewrite(prepared, path, payload=buffer.getvalue())


def test_full_input_is_repeatable_keeps_unknown_and_late_purchases(prepared):
    before = {p.relative_to(prepared[0]): p.stat().st_mtime_ns for p in prepared[0].rglob("*")}
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
    assert before == {
        p.relative_to(prepared[0]): p.stat().st_mtime_ns for p in prepared[0].rglob("*")
    }


@pytest.mark.parametrize("status", ["failed", "interrupted", "prepared"])
def test_materialized_report_requires_completed_execution(prepared, status):
    root, execution = prepared
    value = json.loads((root / execution).read_bytes())
    value["status"] = status
    rewrite(prepared, execution, value=value)
    audit = []
    with pytest.raises(ValueError):
        open_input(prepared, audit)
    assert audit[-1]["allowed"] is False


@pytest.mark.parametrize("field,value", [("code_sha256", "e" * 64), ("scope", "actual")])
def test_foreign_producer_or_scope_is_rejected(prepared, field, value):
    rewrite(prepared, "preparation/sessions.parquet", descriptor_changes={field: value})
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


def test_selection_index_and_session_register_must_agree(prepared):
    path = "preparation/selection.json"
    value = json.loads((prepared[0] / path).read_bytes())
    value["selection"]["A_G"] = ["foreign", "s2"]
    rewrite(prepared, path, value=value)
    with pytest.raises(ValueError):
        tuple(open_input(prepared, []).iter_sessions())


@pytest.mark.parametrize("mutation", ["omitted_block", "foreign_origin", "duplicate_origin"])
def test_canonical_rows_are_complete_and_keep_source_provenance(prepared, mutation):
    index = json.loads((prepared[0] / "canonical/index.json").read_bytes())
    path = index[0]["path"]

    def change(rows):
        if mutation == "omitted_block":
            return rows[:-1]
        block = rows[0]
        event = block.events[0]
        origin = event.origins[0]
        origins = (
            (origin.replace(origin.split(":")[1], "e" * 64),)
            if mutation == "foreign_origin"
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
    index = json.loads((prepared[0] / "canonical/index.json").read_bytes())
    (prepared[0] / index[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        tuple(source.iter_sessions())
    assert audit[-1]["allowed"] is False


def test_absent_root_is_not_created(tmp_path):
    from coveo_recsys.data.historical import open_historical_input

    missing = tmp_path / "absent"
    audit = []
    with pytest.raises((InputValidationError, OSError)):
        open_historical_input(
            missing,
            report_sha256="0" * 64,
            report_descriptor_sha256="0" * 64,
            execution_path="preparation/executions/absent.json",
            execution_descriptor_sha256="0" * 64,
            audit=audit.append,
        )
    assert not missing.exists()
    assert audit[-1]["allowed"] is False


@pytest.mark.parametrize("period_name", ["A_N", "B", "C_smoke", "C_select", "E", "D"])
def test_later_period_changes_with_new_pins_preserve_historical_tables(
    prepared_source, tmp_path, period_name
):
    source, _ = prepared_source
    inputs = tmp_path / "changed-input"
    shutil.copytree(source.parent / "input", inputs)
    plan_path = source / "inputs/plan.json"
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
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
    result = preparation.prepare(
        inputs, source / "inputs/profile.json", plan_path, batch_size=2, max_batch_chars=256
    )
    baseline = open_input(prepared_source, [])
    changed = open_input((changed_root, result["execution_path"]), [])
    assert baseline.input_hashes["browsing"] != changed.input_hashes["browsing"]
    assert result["period_counts"][period_name]["eligible"] == 1
    assert result["selection"][period_name] == []
    profile = json.loads((source / "inputs/profile.json").read_bytes())
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
