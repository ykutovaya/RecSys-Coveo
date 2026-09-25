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
            [selected[1], "event_product", "purchase", "sku-b", 170, "url"],
            [selected[1], "event_product", "purchase", "sku-b", 180, "url"],
            [selected[1], "event_product", "purchase", "sku-c", 180, "url"],
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
        periods=tuple(replace(p, sample_size=2 if p.name == "A_G" else 0) for p in base.periods),
        diagnostic_sessions=0,
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


def test_actual_selection_uses_admission_and_keeps_unknown_empty_targets(
    actual_inputs, tmp_path, monkeypatch
):
    def no_repeated_supply_audit(*args, **kwargs):
        raise AssertionError("Подготовка не повторяет полный аудит поставки")

    monkeypatch.setattr("coveo_recsys.cli.preparation.inspect_supply", no_repeated_supply_audit)
    destination = tmp_path / "run"
    result = run(actual_inputs, destination)
    store = LocalArtifactStore(destination)
    assert result["status"] == "prepared"
    assert result["scope"] == "actual" and not result["real_preflight"]
    excluded = actual_inputs[3][0]
    expected = sorted({"session-a", "session-b", "session-c"} - {excluded}, key=session_key)
    assert result["selection"]["A_G"] == expected
    assert all(not ids for p, ids in result["selection"].items() if p != "A_G")
    assert result["canonical_sessions"] == 2
    assert result["exclusions"]["late_A_G_tail"] == 1
    assert result["period_counts"]["A_G"]["admitted"] == 2
    sessions = records(store, "preparation/sessions.parquet", SessionRecord)
    assert result["request_count"] == sum(s.request_count for s in sessions) > 0
    assert all(s.admitted and s.telemetry_complete is None for s in sessions)
    blocks, targets = [], []
    access = DatasetAccess(
        store, code_sha256=result["source_sha256"], run_id="actual-test", scope="actual"
    )
    for dataset in result["datasets"]:
        blocks.extend(records(store, dataset["canonical_path"], CanonicalBlock))
        targets.extend(records(store, dataset["targets"], TargetRecord))
        requests = records(store, dataset["prefixes"], PreparedRequest)
        assert len(requests) == dataset["requests"]
        assert access.read(dataset["targets"], purpose="labels")
    assert all(b.session_id != excluded for b in blocks), "Поздний хвост исключает всю сессию"
    assert len(targets) == result["request_count"]
    assert sorted(t.purchases for t in targets) == [(), ("sku-b", "sku-c")]
    assert any(t.status == OutcomeStatus.UNKNOWN and t.reason for t in targets if not t.purchases)
    assert result["N_obs"] == 2 and result["N_plus"] == 1
    assert result["composition"]["detail"]["sessions"] == 2
    assert result["composition"]["target_window"]["sessions_by_target_size"] == {"0": 1, "2": 1}
    assert not (destination / "diagnostic").exists() and not (destination / "cohort").exists()
    assert all(
        store.describe(p.relative_to(destination).as_posix()[:-14]).scope == "actual"
        for p in destination.rglob("*.artifact.json")
    )


def test_composition_counts_canonical_actions_sessions_and_target_window(actual_inputs, tmp_path):
    source, profile_path, plan_path, selected = actual_inputs
    positive = selected[1]
    other = ({"session-a", "session-b", "session-c"} - set(selected)).pop()
    extra = [
        [positive, "event_product", "detail", "sku-a", 100, "url"],  # set_v1 повтор
        [positive, "event_product", "detail", "sku-a", 110, "url"],
        [positive, "event_product", "purchase", "sku-d", 90_000_000, "url"],  # вне H
        [other, "event_product", "detail", "sku-a", 110, "url"],
        [other, "event_product", "detail", "sku-b", 111, "url"],
        [other, "event_product", "purchase", "sku-b", 120, "url"],
        ["fourth", "event_product", "detail", "sku-a", 100, "url"],
        ["fourth", "event_product", "add", "sku-d", 110, "url"],
        ["fourth", "event_product", "purchase", "sku-b", 130, "url"],
    ]
    path = source / "browsing_train.csv"
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(extra)
    profile = json.loads(profile_path.read_bytes())
    profile["files"][path.name].update(
        expected_sha256=sha256_file(path), expected_size_bytes=path.stat().st_size
    )
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
    plan_path.write_bytes(
        to_json(
            replace(
                plan,
                periods=tuple(
                    replace(p, sample_size=10) if p.name == "A_G" else p for p in plan.periods
                ),
            )
        )
    )
    result = run(actual_inputs, tmp_path / "run")
    assert (result["N_eligible"], result["N_admitted"], result["N_obs"], result["N_plus"]) == (
        4,
        3,
        3,
        3,
    )
    assert result["selection_shortfalls"] == {
        "A_G": {"requested": 10, "admitted": 3, "selected": 3, "reason": "fewer_admitted_sessions"}
    }
    assert result["canonical_blocks"] == 13
    assert result["composition"] == {
        "event_semantics": "set_v1",
        "source_rows": {"browsing_train.csv": 14, "search_train.csv": 1},
        "product_skus": 4,
        "detail": {
            "events": 6,
            "sessions": 3,
            "skus": 2,
            "skus_in_at_least_2_sessions": 1,
            "skus_in_at_least_3_sessions": 1,
        },
        "purchase": {
            "events": 6,
            "sessions": 3,
            "skus": 3,
            "skus_in_at_least_2_sessions": 1,
            "skus_in_at_least_3_sessions": 1,
        },
        "repeated_detail_session_sku_pairs": 2,
        "target_window": {
            "purchase_events": 5,
            "skus": 2,
            "sessions_by_target_size": {"1": 2, "2": 1},
        },
    }


@pytest.mark.parametrize("step", ["canonical_session", "prepared_session", "reading"])
def test_frequent_budget_checks_scan_only_temporary_files(tmp_path, monkeypatch, step):
    preparation = Preparation(LocalArtifactStore(tmp_path / "run"), ROOT, "budget-test")
    plan = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    preparation.plan = replace(plan, max_temp_bytes=10)
    temporary = preparation.store.root / ".prepare-test" / "nested"
    temporary.mkdir(parents=True)
    chunk = temporary / "chunk"
    chunk.write_bytes(b"0123456789")
    original = Path.rglob

    def temporary_only(path, pattern):
        assert any(part.startswith(".prepare-") for part in path.parts)
        return original(path, pattern)

    monkeypatch.setattr(Path, "rglob", temporary_only)
    preparation.progress(step, sessions=1)
    assert preparation.temp_peak == 10
    chunk.write_bytes(b"01234567890")
    with pytest.raises(InputValidationError, match="бюджет"):
        preparation.progress(step, sessions=2)


def test_missing_published_row_is_not_unknown_telemetry(actual_inputs, tmp_path, monkeypatch):
    from coveo_recsys.cli import preparation as module

    original = module.iter_events

    def missing(*args, **kwargs):
        for event in original(*args, **kwargs):
            if event.event_time_ms != 170:
                yield event

    monkeypatch.setattr(module, "iter_events", missing)
    with pytest.raises(InputValidationError, match="строк"):
        run(actual_inputs, tmp_path / "run")
    assert not (tmp_path / "run" / REPORT).exists()


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
    assert complete["selection"] == resumed["selection"]
    assert repeated["report_sha256"] == resumed["report_sha256"]
    first, second = (LocalArtifactStore(tmp_path / name) for name in ("complete", "resumed"))
    for item in complete["datasets"]:
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
    assert report["status"] == "materialized"
    attempts = [
        json.loads(p.read_bytes())
        for p in (store.root / "preparation/executions").glob("*.json")
        if not p.name.endswith(".artifact.json")
    ]
    assert len(attempts) == 1 and attempts[0]["status"] == "interrupted"
    assert attempts[0]["reads"] and attempts[0]["elapsed_seconds"] > 0
    resumed = run(actual_inputs, tmp_path / "run")
    assert resumed["status"] == "prepared"
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
        path = destination / result["datasets"][0]["targets"]
        path.write_bytes(b"corrupted")
    with pytest.raises((ValueError, OSError)):
        run(actual_inputs, destination)
