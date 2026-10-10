"""Искусственная подготовка через CLI; не допуск Э1-10 и не маршрут с моделью."""

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coveo_recsys.artifacts import LocalArtifactStore, sha256_file
from coveo_recsys.cli import main
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PERIOD_NAMES, PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.io import FILES
from coveo_recsys.data.prepared import iter_sessions, read_report

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/e1"
PROFILE = ROOT / "configs/synthetic_csv.json"
PLAN = ROOT / "configs/synthetic_preparation.json"
REPORT = "reports/preparation.json"


def arguments(
    action, run_dir, *, input_dir=FIXTURES, plan=PLAN, batch_size=1024, max_batch_chars=4_194_304
):
    values = [
        "--project-root",
        str(ROOT),
        "data",
        action,
        "--input-dir",
        str(input_dir),
        "--profile",
        str(PROFILE),
        "--run-id",
        "integration_e1",
        "--run-dir",
        str(run_dir),
        "--batch-size",
        str(batch_size),
        "--max-batch-chars",
        str(max_batch_chars),
    ]
    if action == "prepare":
        values.extend(["--plan", str(plan)])
    return values


def invoke(capsys, values, *, success=True):
    capsys.readouterr()
    code = main(values)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert (code == 0) is success, (code, result, captured.err)
    if not success:
        assert result["status"] == "failed"
    return result


def records(store, path, record_type):
    with pq.ParquetFile(store.root / path) as parquet:
        assert parquet.schema_arrow.names == ["session_id", "record_json"]
        assert parquet.schema_arrow.field("record_json").type == pa.binary()
    return tuple(
        record for _, rows in iter_sessions(store.root / path, record_type) for record in rows
    )


def dataset_records(store, report):
    result = {}
    for period, ids in report["selection"].items():
        if not ids:
            continue
        blocks = dict(iter_sessions(store.root / f"canonical/{period}.parquet", CanonicalBlock))
        requests = dict(
            iter_sessions(store.root / f"datasets/{period}/prefixes.parquet", PreparedRequest)
        )
        targets = dict(
            iter_sessions(store.root / f"datasets/{period}/targets.parquet", TargetRecord)
        )
        for sid in ids:
            result[period, sid] = (blocks[sid], requests[sid], targets[sid])
    return result


def file_fingerprints(root):
    return {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def copy_inputs(destination):
    destination.mkdir()
    for filename in FILES.values():
        shutil.copyfile(FIXTURES / filename, destination / filename)
    return destination


@pytest.mark.parametrize("defect", ["product_columns", "product_id", "unselected_search"])
def test_synthetic_prepare_validates_rows_outside_selected_events(tmp_path, capsys, defect):
    input_dir = copy_inputs(tmp_path / "input")
    if defect.startswith("product_"):
        path = input_dir / FILES["products"]
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        if defect == "product_columns":
            rows[2].pop()
        else:
            rows[2][0] = ""
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
    else:
        with (input_dir / FILES["search"]).open("a", encoding="utf-8", newline="") as stream:
            # s3 исключается из отбора: первая покупка одновременна первому detail.
            csv.writer(stream).writerow(["s3", "q-invalid", "3000", "[1,0]", "[", "[]"])
    run_dir = tmp_path / "run"
    invoke(capsys, arguments("prepare", run_dir, input_dir=input_dir), success=False)
    assert not (run_dir / REPORT).exists()
    assert not list(run_dir.rglob("*.parquet"))


def test_progress_reports_throttled_json_and_hashing_to_stderr(tmp_path, capsys, monkeypatch):
    from coveo_recsys.cli import preparation as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        module, "_peak_memory", lambda: {"peak_process_memory_bytes": 1234, "memory_method": "test"}
    )
    preparation = module.Preparation(LocalArtifactStore(tmp_path / "run"), ROOT, "progress-test")
    preparation.progress("canonical_session", sessions=1)
    now[0] = 109.999
    preparation.progress("prepared_session", sessions=2)
    now[0] = 110.0
    preparation.progress("reading", table="browsing", rows=10)
    preparation.progress("selection_complete", sessions=2)
    preparation.progress("failed", reason="test_failure")
    now[0] = 120.0
    preparation._input_hashes(FIXTURES)
    captured = capsys.readouterr()
    assert captured.out == ""
    entries = [json.loads(line) for line in captured.err.splitlines()]
    assert [item["step"] for item in entries] == [
        "canonical_session",
        "reading",
        "selection_complete",
        "failed",
        "hashing",
    ]
    assert all(item["peak_process_memory_bytes"] == 1234 for item in entries)
    assert entries[-1]["bytes"] == (FIXTURES / FILES["browsing"]).stat().st_size
    assert all(
        set(item)
        <= {
            "step",
            "elapsed_seconds",
            "peak_process_memory_bytes",
            "memory_method",
            "sessions",
            "table",
            "rows",
            "reason",
            "bytes",
        }
        for item in entries
    )
    assert preparation.last_progress["table"] == "products"


def assert_prefix_only(request):
    forbidden = {
        "targets",
        "target",
        "labels",
        "purchases",
        "b_s",
        "future_counts",
        "session_request_count",
        "session_weight",
        "pair_weights",
        "target_end_ms",
        "close_ms",
        "mature",
        "request_count",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            assert value.get("type") != "TargetRecord"
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(json.loads(to_json(request)))
    assert request.a_q == request.t_q
    assert all(
        block.event_time_ms <= request.t_q and block.available_time_ms <= request.a_q
        for block in request.prefix.blocks
    )


def test_cli_inspect_then_prepare_has_exact_fixture_results(tmp_path, capsys):
    run_dir = tmp_path / "run"
    inspection = invoke(capsys, arguments("inspect", run_dir, batch_size=2, max_batch_chars=256))
    assert inspection["files"]["browsing"]["rows"] == 9
    assert inspection["files"]["search"]["rows"] == 2
    assert inspection["files"]["products"]["rows"] == 2
    assert all(not value["coverage_confirmed"] for value in inspection["files"].values())
    assert inspection["memory_plan"]["max_batch_chars"] == 256
    assert inspection["disk_plan"]["estimate_only"] is True

    result = invoke(capsys, arguments("prepare", run_dir, batch_size=2, max_batch_chars=256))
    assert result["status"] == "prepared" and result["scope"] == "synthetic"
    assert result["real_preflight"] is False
    assert (result["canonical_sessions"], result["canonical_blocks"], result["request_count"]) == (
        2,
        8,
        2,
    )
    assert result["selection"]["A_G"] == ["s1", "s2"]
    assert all(result["selection"][name] == [] for name in PERIOD_NAMES[1:])
    assert result["exclusions"] == {"no_request": 1}
    assert result["report_path"] == REPORT
    store = LocalArtifactStore(run_dir)
    stored = read_report(run_dir)
    assert stored == {
        **{
            key: value
            for key, value in result.items()
            if key not in {"report_path", "report_sha256", "status", "execution", "execution_path"}
        },
        "status": "completed",
    }
    assert result["report_sha256"] == sha256_file(run_dir / REPORT)
    assert set(result["tables"]) == {
        "canonical/A_G.parquet",
        "preparation/sessions.parquet",
        "datasets/A_G/prefixes.parquet",
        "datasets/A_G/targets.parquet",
    }
    canonical = dict(iter_sessions(run_dir / "canonical/A_G.parquet", CanonicalBlock))
    assert {sid: len(rows) for sid, rows in canonical.items()} == {"s1": 5, "s2": 3}
    sessions = {
        row.session_id: row for row in records(store, "preparation/sessions.parquet", SessionRecord)
    }
    assert set(sessions) == {"s1", "s2"}
    for session_id, last in [("s1", 5000), ("s2", 4000)]:
        session = sessions[session_id]
        assert (
            session.admission_ms,
            session.last_event_ms,
            session.target_end_ms,
            session.request_count,
        ) == (1000, last, last, 1)
        assert (
            session.admitted
            and session.telemetry_complete is None
            and session.exclusion_reason is None
        )

    prepared = dataset_records(store, result)
    assert set(prepared) == {("A_G", "s1"), ("A_G", "s2")}
    for (_, session_id), (blocks, requests, targets) in prepared.items():
        assert len(blocks) == (5 if session_id == "s1" else 3)
        assert [request.t_q for request in requests] == [1000]
        assert [request.ordinal for request in requests] == [1]
        assert {target.request_id for target in targets} == {
            request.request_id for request in requests
        }
        assert len(targets) == len(requests) == 1
        for request in requests:
            assert_prefix_only(request)
            assert request.prefix.first_event_time_ms == 1000
            assert not request.prefix.purchased_skus
            assert all(
                event.sku not in {"B", "C"}
                for block in request.prefix.blocks
                for event in block.events
            )
        assert requests[0].prefix.counters == (("detail", "A", 1),)
        assert requests[0].prefix.search is None and not requests[0].prefix.cart
        for target in targets:
            assert target.status.value == ("known" if session_id == "s1" else "unknown")
            assert target.horizon_ms == 86_400_000
            assert target.purchases == (("B",) if session_id == "s1" else ())
            assert target.reason == (None if session_id == "s1" else "no_registered_purchase")
    assert not (run_dir / "cohort").exists()


def test_batch_and_external_sort_chunk_sizes_do_not_change_records(tmp_path, capsys):
    expected = None
    for index, (batch_size, budget, chunk_size) in enumerate(
        [(1, 128, 1), (7, 256, 2), (1024, 4_194_304, 4096)]
    ):
        plan_path = tmp_path / f"plan-{index}.json"
        plan_path.write_bytes(
            to_json(replace(from_json(PLAN.read_bytes(), PreparationPlan), chunk_size=chunk_size))
        )
        run_dir = tmp_path / f"run-{index}"
        result = invoke(
            capsys,
            arguments(
                "prepare", run_dir, plan=plan_path, batch_size=batch_size, max_batch_chars=budget
            ),
        )
        actual = (
            result["selection"],
            result["exclusions"],
            result["period_counts"],
            dataset_records(LocalArtifactStore(run_dir), result),
        )
        if expected is None:
            expected = actual
        else:
            assert actual == expected


def test_compatible_repeat_preserves_artifact_identity(tmp_path, capsys):
    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir)
    first = invoke(capsys, values)
    before = file_fingerprints(run_dir)
    repeated = invoke(capsys, values)
    assert repeated["report_sha256"] == first["report_sha256"]
    assert file_fingerprints(run_dir) == before
    assert len(before) == 5  # Четыре общие таблицы и один итоговый отчёт.


def test_rejected_nested_inspect_preserves_preparation_identity(tmp_path, monkeypatch):
    from coveo_recsys.cli import preparation as module

    store = LocalArtifactStore(tmp_path / "run")
    preparation = module.Preparation(store, ROOT, "integration_e1")
    publish = module.write_report
    rejected = False

    def nested_before_report(root, report, **kwargs):
        nonlocal rejected
        with pytest.raises(InputValidationError):
            preparation.inspect(FIXTURES, PROFILE, batch_size=1024, max_batch_chars=4_194_304)
        rejected = True
        return publish(root, report, **kwargs)

    monkeypatch.setattr(module, "write_report", nested_before_report)
    result = preparation.prepare(
        FIXTURES, PROFILE, PLAN, batch_size=1024, max_batch_chars=4_194_304
    )
    assert rejected and result["status"] == "prepared"
    assert preparation.artifacts == {}
    assert read_report(store.root)["tables"] == result["tables"]


def test_resume_after_report_failure_reuses_completed_artifacts(tmp_path, capsys, monkeypatch):
    from coveo_recsys.cli import preparation as module

    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir)
    publish = module.write_report

    def fail_report(*args, **kwargs):
        raise OSError("injected_before_preparation_report")

    monkeypatch.setattr(module, "write_report", fail_report)
    failed = invoke(capsys, values, success=False)
    assert "injected_before_preparation_report" in failed["message"]
    assert not (run_dir / REPORT).exists()
    before = {p: value for p, value in file_fingerprints(run_dir).items() if p.endswith(".parquet")}
    assert len(before) == 4
    monkeypatch.setattr(module, "write_report", publish)
    resumed = invoke(capsys, values)
    assert resumed["status"] == "prepared" and resumed["request_count"] == 2
    assert all(file_fingerprints(run_dir)[path] == value for path, value in before.items())
    assert not (run_dir / "preparation/identity.json").exists()


@pytest.mark.parametrize("mutation", ["input", "plan", "code", "artifact"])
def test_incompatible_or_corrupted_checkpoint_is_rejected(tmp_path, capsys, monkeypatch, mutation):
    input_dir = copy_inputs(tmp_path / "input")
    plan_path = tmp_path / "plan.json"
    shutil.copyfile(PLAN, plan_path)
    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir, input_dir=input_dir, plan=plan_path)
    invoke(capsys, values)
    old_report = (run_dir / REPORT).read_bytes()
    if mutation == "input":
        with (input_dir / FILES["browsing"]).open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["s1", "new", "6000", "detail", "B"])
    elif mutation == "plan":
        plan = from_json(plan_path.read_bytes(), PreparationPlan)
        plan_path.write_bytes(to_json(replace(plan, chunk_size=plan.chunk_size + 1)))
    elif mutation == "code":
        monkeypatch.setattr("coveo_recsys.cli.preparation.source_sha256", lambda _: "f" * 64)
    else:
        path = run_dir / "datasets/A_G/prefixes.parquet"
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    invoke(capsys, values, success=False)
    assert (run_dir / REPORT).read_bytes() == old_report


@pytest.mark.parametrize("mutation", ["input", "plan", "profile", "code"])
def test_reuse_rechecks_dependencies_after_last_progress(tmp_path, capsys, monkeypatch, mutation):
    from coveo_recsys.cli.preparation import Preparation

    input_dir = copy_inputs(tmp_path / "input")
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    shutil.copyfile(PLAN, plan_path)
    shutil.copyfile(PROFILE, profile_path)
    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir, input_dir=input_dir, plan=plan_path)
    values[values.index("--profile") + 1] = str(profile_path)
    invoke(capsys, values)
    old_report = (run_dir / REPORT).read_bytes()
    progress = Preparation.progress
    changed = False

    def change_after_reuse(self, step, **fields):
        nonlocal changed
        progress(self, step, **fields)
        if step != "prepared_reused":
            return
        changed = True
        if mutation == "input":
            with (input_dir / FILES["browsing"]).open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["s1", "late_change", "6000", "detail", "B"])
        elif mutation == "plan":
            plan = from_json(plan_path.read_bytes(), PreparationPlan)
            plan_path.write_bytes(to_json(replace(plan, chunk_size=plan.chunk_size + 1)))
        elif mutation == "profile":
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["max_row_chars"] -= 1
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
        else:
            monkeypatch.setattr("coveo_recsys.cli.preparation.source_sha256", lambda _: "f" * 64)

    monkeypatch.setattr(Preparation, "progress", change_after_reuse)
    invoke(capsys, values, success=False)
    assert changed  # Изменение действительно произошло после чтения готовой контрольной точки.
    assert (run_dir / REPORT).read_bytes() == old_report
    assert (run_dir / REPORT).read_bytes() == old_report


@pytest.mark.parametrize("path", ["canonical/A_G.parquet", "datasets/A_G/targets.parquet"])
@pytest.mark.parametrize("mode", ["prepare", "reuse"])
def test_prepare_final_verification_rejects_late_artifact_corruption(
    tmp_path, capsys, monkeypatch, mode, path
):
    from coveo_recsys.cli.preparation import Preparation

    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir)
    if mode == "reuse":
        invoke(capsys, values)
    progress = Preparation.progress
    changed = False

    def corrupt_after_final_progress(self, step, **fields):
        nonlocal changed
        progress(self, step, **fields)
        if step != ("prepared_reused" if mode == "reuse" else "prepared"):
            return
        target = self.store.root / path
        data = target.read_bytes()
        target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        changed = True

    monkeypatch.setattr(Preparation, "progress", corrupt_after_final_progress)
    invoke(capsys, values, success=False)
    assert changed, "Инъекция должна сработать после последней записи progress"
    if mode == "prepare":
        assert not (run_dir / REPORT).exists()


def test_seven_period_datasets_enforce_purpose_and_log_direct_path_attempts(tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copyfile(FIXTURES / FILES["products"], input_dir / FILES["products"])
    with (input_dir / FILES["browsing"]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["session_id", "event_id", "server_timestamp_epoch_ms", "product_action", "sku"]
        )
        for index, period in enumerate(PERIOD_NAMES):
            start = index * 200_000_000
            writer.writerow([f"s_{period}", f"view_{period}", start + 1000, "detail", "A"])
            writer.writerow([f"s_{period}", f"purchase_{period}", start + 2000, "purchase", "B"])
    (input_dir / FILES["search"]).write_text(
        "session_id,event_id,server_timestamp_epoch_ms,query_vector,product_skus_hash,clicked_skus_hash\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    result = invoke(capsys, arguments("prepare", run_dir, input_dir=input_dir, batch_size=1))
    assert result["canonical_sessions"] == result["request_count"] == 6
    assert result["canonical_blocks"] == 12 and result["exclusions"] == {}
    assert set(result["selection"]) == set(PERIOD_NAMES)
    audit = []
    access = DatasetAccess(
        run_dir, code_sha256=result["source_sha256"], run_id="integration_e1", audit=audit.append
    )
    for period in PERIOD_NAMES:
        assert result["selection"][period] == ([] if period == "D" else [f"s_{period}"])
        for kind, record_type, purpose in [
            ("prefixes", PreparedRequest, "prefix"),
            ("targets", TargetRecord, "labels"),
        ]:
            path = f"datasets/{period}/{kind}.parquet"
            if period in {"A_G", "B", "C_smoke"}:
                assert tuple(access.read(path, record_type, purpose=purpose)) == tuple(
                    iter_sessions(run_dir / path, record_type)
                )
            else:
                assert (run_dir / path).is_file() is (period != "D")
                with pytest.raises(InputValidationError):
                    tuple(access.read(path, record_type, purpose=purpose))
    with pytest.raises(InputValidationError):
        tuple(access.read("datasets/B/targets.parquet", TargetRecord, purpose="prefix"))
    assert len(audit) == 15 and sum(log["allowed"] for log in audit) == 6
    assert all(log["boundary"] == "local_routing_only" for log in audit)
    assert {(log["path"], log["purpose"]) for log in audit if not log["allowed"]} == {
        *(
            (f"datasets/{period}/{kind}.parquet", purpose)
            for period in ("A_N", "C_select", "E", "D")
            for kind, purpose in (("prefixes", "prefix"), ("targets", "labels"))
        ),
        ("datasets/B/targets.parquet", "prefix"),
    }
    assert not (run_dir / "access").exists()
