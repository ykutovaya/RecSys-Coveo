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
from coveo_recsys.data.prepared import iter_sessions, read_report

ROOT = Path(__file__).resolve().parents[2]
REPORT = "reports/preparation.json"


@pytest.fixture
def actual_inputs(tmp_path, request):
    source = tmp_path / "input"
    source.mkdir()
    profile = json.loads((ROOT / "configs/coveo_supply_v1.json").read_bytes())
    base = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    period, sample_size = getattr(request, "param", ("A_G", 2))
    offset = next(p.start_ms for p in base.periods if p.name == period)
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
        if name != "sku_to_content.csv":
            timestamp_index = profile["files"][name]["columns"].index("server_timestamp_epoch_ms")
            for row in values:
                if row[0] != "closed-period":
                    row[timestamp_index] += offset
        path = source / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(profile["files"][name]["columns"])
            writer.writerows(values)
        profile["files"][name]["expected_sha256"] = sha256_file(path)
        profile["files"][name]["expected_size_bytes"] = path.stat().st_size
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan = replace(
        base,
        scope="actual",
        source_namespace="coveo-sigir-1.0.0",
        task=replace(base.task, query_dimension=50),
        coverage=replace(base.coverage, source_namespace="coveo-sigir-1.0.0", confirmed_at_ms=None),
        periods=tuple(
            replace(p, sample_size=sample_size if p.name == period else 0) for p in base.periods
        ),
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
    return tuple(record for _, rows in iter_sessions(store.root / path, kind) for record in rows)


def test_preparation_uses_only_common_tables_and_no_session_artifacts(actual_inputs, tmp_path):
    destination = tmp_path / "run"
    result = run(actual_inputs, destination)
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*.parquet")} == {
        "canonical/A_G.parquet",
        "datasets/A_G/prefixes.parquet",
        "datasets/A_G/targets.parquet",
        "preparation/sessions.parquet",
    }
    assert not (destination / ".objects").exists()
    assert not (destination / "preparation/progress").exists()
    assert not list(destination.rglob("*.artifact.*"))
    assert {"datasets", "artifacts"}.isdisjoint(result)


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
    blocks = records(store, "canonical/A_G.parquet", CanonicalBlock)
    targets = records(store, "datasets/A_G/targets.parquet", TargetRecord)
    requests = records(store, "datasets/A_G/prefixes.parquet", PreparedRequest)
    access = DatasetAccess(
        store.root, code_sha256=result["source_sha256"], run_id="actual-test", scope="actual"
    )
    assert len(requests) == result["request_count"]
    assert (
        tuple(
            row
            for _, rows in access.read(
                "datasets/A_G/targets.parquet", TargetRecord, purpose="labels"
            )
            for row in rows
        )
        == targets
    )
    assert all(b.session_id != excluded for b in blocks), "Поздний хвост исключает всю сессию"
    assert len(targets) == result["request_count"]
    assert sorted(t.purchases for t in targets) == [(), ("sku-b", "sku-c")]
    assert any(t.status == OutcomeStatus.UNKNOWN and t.reason for t in targets if not t.purchases)
    assert result["N_obs"] == 2 and result["N_plus"] == 1
    assert result["composition"]["detail"]["sessions"] == 2
    assert result["composition"]["target_window"]["sessions_by_target_size"] == {"0": 1, "2": 1}
    assert not (destination / "diagnostic").exists() and not (destination / "cohort").exists()
    assert read_report(destination)["status"] == "completed"


@pytest.mark.parametrize("actual_inputs", [("B", 3)], indirect=True)
def test_actual_b_prepares_full_sessions_with_first_requests_and_separate_targets(
    actual_inputs, tmp_path
):
    destination = tmp_path / "run"
    result = run(actual_inputs, destination)
    store = LocalArtifactStore(destination)
    plan = from_json(actual_inputs[2].read_bytes(), PreparationPlan)
    period = next(p for p in plan.periods if p.name == "B")
    assert result["selection"]["B"] == sorted(
        ("session-a", "session-b", "session-c"), key=session_key
    )
    assert all(not ids for name, ids in result["selection"].items() if name != "B")
    assert set(result["tables"]) == {
        "preparation/sessions.parquet",
        "canonical/B.parquet",
        "datasets/B/prefixes.parquet",
        "datasets/B/targets.parquet",
    }
    sessions = records(store, "preparation/sessions.parquet", SessionRecord)
    assert all(s.period == "B" and s.admitted and s.telemetry_complete is None for s in sessions)
    assert sorted(s.last_event_ms for s in sessions) == [
        period.start_ms + 100,
        period.start_ms + 180,
        period.start_ms + 210_000_000,
    ]
    blocks = records(store, "canonical/B.parquet", CanonicalBlock)
    assert sum(len(event.origins) for block in blocks for event in block.events) == 9
    assert max(block.event_time_ms for block in blocks) > period.end_ms
    requests = records(store, "datasets/B/prefixes.parquet", PreparedRequest)
    assert len(requests) == len(sessions) == result["request_count"] == 3
    assert all(
        request.ordinal == 1 and request.t_q == period.start_ms + 100 for request in requests
    )
    assert all(
        len(request.prefix.blocks) == 1
        and all(event.event_type == "detail" for event in request.prefix.blocks[0].events)
        for request in requests
    )
    targets = records(store, "datasets/B/targets.parquet", TargetRecord)
    assert {target.request_id for target in targets} == {request.request_id for request in requests}
    assert sorted(target.purchases for target in targets) == [(), ("sku-b",), ("sku-b", "sku-c")]
    empty_target = next(target for target in targets if not target.purchases)
    assert empty_target.status == OutcomeStatus.UNKNOWN and empty_target.reason


@pytest.mark.parametrize("period", ["A_N", "C_smoke", "C_select", "E", "D"])
def test_actual_preparation_rejects_nonzero_closed_period(actual_inputs, tmp_path, period):
    plan_path = actual_inputs[2]
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
    plan_path.write_bytes(
        to_json(
            replace(
                plan,
                periods=tuple(
                    replace(p, sample_size=1) if p.name == period else p for p in plan.periods
                ),
            )
        )
    )
    destination = tmp_path / "run"
    with pytest.raises(
        InputValidationError,
        match="C_smoke требует post_training, C_select — pre_selection, D — pre_D",
    ):
        run(actual_inputs, destination)
    assert not (destination / REPORT).exists()
    assert not list(destination.rglob("*.parquet"))


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


def test_temporary_budget_includes_report_before_publication(tmp_path):
    preparation = Preparation(LocalArtifactStore(tmp_path / "run"), ROOT, "budget-test")
    plan = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    preparation.plan = replace(plan, max_temp_bytes=10)
    temporary = preparation.store.root / "reports/.preparation.json.test.tmp"
    temporary.parent.mkdir()
    temporary.write_bytes(b"0123456789")
    preparation._check_budget(force=True)
    assert preparation.temp_peak == 10
    temporary.write_bytes(b"01234567890")
    with pytest.raises(InputValidationError, match="бюджет"):
        preparation._check_budget(force=True)
    assert not (preparation.store.root / REPORT).exists()


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
    from coveo_recsys.cli import preparation as module

    complete = run(actual_inputs, tmp_path / "complete", batch_size=3)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(module, "write_report", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run(actual_inputs, tmp_path / "resumed")
    assert not (tmp_path / "resumed" / REPORT).exists()
    resumed = run(actual_inputs, tmp_path / "resumed", batch_size=2)
    repeated = run(actual_inputs, tmp_path / "resumed", batch_size=3)
    assert complete["selection"] == resumed["selection"]
    assert repeated["report_sha256"] == resumed["report_sha256"]
    for path in complete["tables"]:
        assert (tmp_path / "complete" / path).read_bytes() == (
            tmp_path / "resumed" / path
        ).read_bytes()


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
    assert not (tmp_path / "run" / REPORT).exists()
    assert (tmp_path / "run/preparation/identity.json").is_file()
    assert not (tmp_path / "run/preparation/executions").exists()
    resumed = run(actual_inputs, tmp_path / "run")
    assert resumed["status"] == "prepared"
    assert resumed["execution"]["status"] == "completed"
    assert read_report(tmp_path / "run")["status"] == "completed"
    assert not (tmp_path / "run/preparation/identity.json").exists()


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
    run(actual_inputs, destination)
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
        path = destination / "datasets/A_G/targets.parquet"
        path.write_bytes(b"corrupted")
    with pytest.raises((ValueError, OSError)):
        run(actual_inputs, destination)


@pytest.mark.parametrize("mutation", ["input", "profile", "plan", "code"])
def test_incomplete_preparation_keeps_identity_before_recalculation(
    actual_inputs, tmp_path, monkeypatch, mutation
):
    destination = tmp_path / "run"
    progress = Preparation.progress

    def interrupt(self, step, **fields):
        progress(self, step, **fields)
        if step == "prepared":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Preparation, "progress", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run(actual_inputs, destination)
    before = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in destination.rglob("*.parquet")}
    assert len(before) == 4 and not (destination / REPORT).exists()
    if mutation == "code":
        monkeypatch.setattr("coveo_recsys.cli.preparation.source_sha256", lambda _: "f" * 64)
    else:
        path = {
            "input": actual_inputs[0] / "browsing_train.csv",
            "profile": actual_inputs[1],
            "plan": actual_inputs[2],
        }[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(InputValidationError):
        run(actual_inputs, destination)
    assert not (destination / REPORT).exists()
    assert all(
        (sha256_file(path), path.stat().st_mtime_ns) == value for path, value in before.items()
    )
