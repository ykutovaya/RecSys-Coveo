"""Малые придуманные CSV фактической схемы; не требуют данных Coveo."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, sha256_file
from coveo_recsys.cli.preparation import Preparation
from coveo_recsys.contracts import InputValidationError, OutcomeStatus
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.periods import session_key

ROOT = Path(__file__).resolve().parents[2]
REPORT = "reports/preparation.json"


@pytest.fixture
def actual_inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    profile = json.loads((ROOT / "configs/coveo_supply_v1.json").read_bytes())
    vector = json.dumps([1.0] + [0.0] * 49)
    selected = sorted(("session-a", "session-b", "session-c"), key=session_key)[:2]
    rows = {
        "browsing_train.csv": [
            [sid, "event_product", "detail", "sku-a", 100, "url"]
            for sid in ("session-a", "session-b", "session-c")
        ]
        + [
            [selected[0], "event_product", "purchase", "sku-b", 120, "url"],
            [selected[0], "pageview", "", "", 210_000_000, "url"],
            ["closed-period", "event_product", "detail", "sku-b", 1_300_000_000, "url"],
        ],
        "search_train.csv": [[selected[1], vector, "['sku-b']", "['sku-a']", 150]],
        "sku_to_content.csv": [["sku-a", vector, "category", vector, "3.0"]],
    }
    for name, values in rows.items():
        path = source / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(profile["files"][name]["columns"])
            writer.writerows(values)
        profile["files"][name]["expected_sha256"] = sha256_file(path)
        profile["files"][name]["expected_size_bytes"] = path.stat().st_size
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    base = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    plan = replace(
        base,
        scope="actual",
        source_namespace="coveo-sigir-1.0.0",
        task=replace(base.task, query_dimension=50),
        coverage=replace(base.coverage, source_namespace="coveo-sigir-1.0.0", confirmed_at_ms=None),
        diagnostic_sessions=2,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(to_json(plan))
    return source, profile_path, plan_path, selected


def run(inputs, destination, *, batch_size=1):
    source, profile, plan, _ = inputs
    return Preparation(LocalArtifactStore(destination), ROOT, "actual-test").prepare(
        source, profile, plan, batch_size=batch_size, max_batch_chars=4096
    )


def records(store, path, kind):
    return tuple(from_json(v, kind) for v in store.read_table(path)["record_json"].to_pylist())


def test_actual_unknown_targets_survive_without_admitted_dataset(actual_inputs, tmp_path):
    destination = tmp_path / "run"
    result = run(actual_inputs, destination)
    store = LocalArtifactStore(destination)
    assert result["status"] == "prepared_diagnostic"
    assert result["scope"] == "actual" and not result["real_preflight"]
    assert result["datasets"] == [] and all(not ids for ids in result["selection"].values())
    assert result["canonical_sessions"] == 2
    assert set(result["technical_selection"]) == set(actual_inputs[3])
    sessions = records(store, "preparation/sessions.parquet", SessionRecord)
    assert result["request_count"] == sum(s.request_count for s in sessions) > 0
    assert all(not s.mature and not s.coverage_complete for s in sessions)
    blocks, targets = [], []
    access = DatasetAccess(
        store, code_sha256=result["source_sha256"], run_id="actual-test", scope="actual"
    )
    for dataset in result["diagnostic_datasets"]:
        blocks.extend(records(store, dataset["canonical_path"], CanonicalBlock))
        targets.extend(records(store, dataset["targets"], TargetRecord))
        requests = records(store, dataset["prefixes"], PreparedRequest)
        assert len(requests) == dataset["requests"]
        with pytest.raises(InputValidationError):
            access.read(dataset["targets"], purpose="labels")
    assert any(b.event_time_ms == 210_000_000 for b in blocks), "Нельзя обрезать хвост сессии"
    assert len(targets) == result["request_count"]
    assert all(
        t.status == OutcomeStatus.UNKNOWN and t.b_s is None and t.purchases is None and t.reason
        for t in targets
    )
    assert not (destination / "datasets").exists() and not (destination / "cohort").exists()
    assert all(
        store.describe(p.relative_to(destination).as_posix()[:-14]).scope == "actual"
        for p in destination.rglob("*.artifact.json")
    )


def test_actual_resume_and_batch_size_preserve_records(actual_inputs, tmp_path, monkeypatch):
    complete = run(actual_inputs, tmp_path / "complete", batch_size=3)
    publish = LocalArtifactStore.publish_json

    def interrupt(self, path, *args, **kwargs):
        if path == REPORT:
            raise KeyboardInterrupt
        return publish(self, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalArtifactStore, "publish_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run(actual_inputs, tmp_path / "resumed")
    assert not (tmp_path / "resumed" / REPORT).exists()
    resumed = run(actual_inputs, tmp_path / "resumed", batch_size=2)
    repeated = run(actual_inputs, tmp_path / "resumed", batch_size=3)
    assert complete["technical_selection"] == resumed["technical_selection"]
    assert repeated["report_sha256"] == resumed["report_sha256"]
    first, second = (LocalArtifactStore(tmp_path / name) for name in ("complete", "resumed"))
    for item in complete["diagnostic_datasets"]:
        for key in ("canonical_path", "prefixes", "targets"):
            assert first.read_bytes(item[key]) == second.read_bytes(item[key])


def test_interruption_after_materialization_has_no_success_marker(
    actual_inputs, tmp_path, monkeypatch
):
    original = Preparation.progress

    def interrupt(self, step, **fields):
        if step == "prepared":
            raise KeyboardInterrupt
        return original(self, step, **fields)

    with monkeypatch.context() as patch:
        patch.setattr(Preparation, "progress", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run(actual_inputs, tmp_path / "run")
    store = LocalArtifactStore(tmp_path / "run")
    report = json.loads(store.read_bytes(REPORT))
    assert report["status"] == "diagnostic_materialized"
    attempts = [
        json.loads(p.read_bytes())
        for p in (store.root / "preparation/executions").glob("*.json")
        if not p.name.endswith(".artifact.json")
    ]
    assert len(attempts) == 1 and attempts[0]["status"] == "interrupted"
    assert attempts[0]["reads"] and attempts[0]["elapsed_seconds"] > 0
    resumed = run(actual_inputs, tmp_path / "run")
    assert resumed["status"] == "prepared_diagnostic"
    assert resumed["execution"]["status"] == "completed"


@pytest.mark.parametrize("field", ["max_seconds", "max_memory_bytes", "max_temp_bytes"])
def test_actual_budget_exceeded_is_failure(actual_inputs, tmp_path, field):
    plan_path = actual_inputs[2]
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
    plan_path.write_bytes(to_json(replace(plan, **{field: 1})))
    if field == "max_seconds":
        # Проверяем тот же контроль без зависимости от скорости машины.
        preparation = Preparation(LocalArtifactStore(tmp_path / "run"), ROOT, "actual-test")
        preparation.started -= 2
        with pytest.raises(InputValidationError, match="бюджет"):
            preparation.prepare(*actual_inputs[:3], batch_size=1, max_batch_chars=4096)
    else:
        with pytest.raises(InputValidationError, match="бюджет"):
            run(actual_inputs, tmp_path / "run")
    assert not (tmp_path / "run" / REPORT).exists()


@pytest.mark.parametrize("mutation", ["input", "profile", "plan", "code", "artifact"])
def test_actual_reuse_rejects_changed_identity(actual_inputs, tmp_path, monkeypatch, mutation):
    destination = tmp_path / "run"
    result = run(actual_inputs, destination)
    if mutation in {"input", "profile", "plan"}:
        path = {
            "input": actual_inputs[0] / "browsing_train.csv",
            "profile": actual_inputs[1],
            "plan": actual_inputs[2],
        }[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "code":
        monkeypatch.setattr("coveo_recsys.cli.preparation.source_sha256", lambda _: "f" * 64)
    else:
        path = destination / result["diagnostic_datasets"][0]["targets"]
        path.write_bytes(b"corrupted")
    with pytest.raises((ValueError, OSError)):
        run(actual_inputs, destination)
