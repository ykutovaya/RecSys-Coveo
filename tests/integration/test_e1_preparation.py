"""Искусственная подготовка через CLI; не допуск Э1-10 и не маршрут с моделью."""

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

from coveo_recsys.artifacts import ArtifactError, LocalArtifactStore, sha256_bytes
from coveo_recsys.cli import main
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PERIOD_NAMES, PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.io import FILES

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
    table = store.read_table(path)
    assert table.column_names == ["record_json"]
    assert table.schema.field("record_json").type == pa.binary()
    return tuple(from_json(value, record_type) for value in table.column("record_json").to_pylist())


def dataset_records(store, report):
    result = {}
    for dataset in report["datasets"]:
        result[dataset["period"], dataset["session_id"]] = (
            records(store, dataset["canonical_path"], CanonicalBlock),
            records(store, dataset["prefixes"], PreparedRequest),
            records(store, dataset["targets"], TargetRecord),
        )
    return result


def descriptor_hashes(store):
    paths = (
        str(path.relative_to(store.root)).replace("\\", "/")[: -len(".artifact.json")]
        for path in store.root.rglob("*.artifact.json")
    )
    return {path: store.describe(path).descriptor_sha256 for path in paths}


def copy_inputs(destination):
    destination.mkdir()
    for filename in FILES.values():
        shutil.copyfile(FIXTURES / filename, destination / filename)
    return destination


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
    stored = json.loads(store.read_bytes(REPORT))
    assert stored == {
        **{
            key: value
            for key, value in result.items()
            if key not in {"report_path", "report_sha256", "status", "execution", "execution_path"}
        },
        "status": "materialized",
    }
    assert result["report_sha256"] == store.describe(REPORT).sha256
    for path, digest in result["artifacts"].items():
        descriptor = store.describe(path)
        assert descriptor.descriptor_sha256 == digest
        assert descriptor.scope == "synthetic" and descriptor.run_id == "integration_e1"

    canonical_index = json.loads(store.read_bytes("canonical/index.json"))
    assert {
        row["session_id"]: len(records(store, row["path"], CanonicalBlock))
        for row in canonical_index
    } == {"s1": 5, "s2": 3}
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
    values = arguments("prepare", tmp_path / "run")
    first = invoke(capsys, values)
    store = LocalArtifactStore(tmp_path / "run")
    before = descriptor_hashes(store)
    repeated = invoke(capsys, values)
    assert repeated["report_sha256"] == first["report_sha256"]
    after = descriptor_hashes(store)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(
        path.startswith(("preparation/progress/", "preparation/executions/"))
        for path in after.keys() - before.keys()
    )


def test_rejected_nested_inspect_preserves_outer_report_dependencies(tmp_path, monkeypatch):
    from coveo_recsys.cli.preparation import Preparation

    store = LocalArtifactStore(tmp_path / "run")
    preparation = Preparation(store, ROOT, "integration_e1")
    publish = store.publish_json
    expected_dependencies = {}
    rejected = False

    def nested_before_report(relative_path, value, **kwargs):
        nonlocal rejected, expected_dependencies
        if relative_path == REPORT:
            expected_dependencies = dict(preparation.artifacts)
            with pytest.raises(ArtifactError):
                preparation.inspect(FIXTURES, PROFILE, batch_size=1024, max_batch_chars=4_194_304)
            rejected = True
        return publish(relative_path, value, **kwargs)

    monkeypatch.setattr(store, "publish_json", nested_before_report)
    result = preparation.prepare(
        FIXTURES, PROFILE, PLAN, batch_size=1024, max_batch_chars=4_194_304
    )
    assert rejected and result["status"] == "prepared"
    assert {"canonical/index.json", "preparation/selection.json"} <= expected_dependencies.keys()
    target = result["datasets"][0]["targets"]
    assert target in expected_dependencies
    assert store.describe(REPORT).artifact_dependencies == expected_dependencies
    assert preparation.artifacts == result["artifacts"] == expected_dependencies
    assert json.loads(store.read_bytes(REPORT))["artifacts"] == expected_dependencies

    # Сохранённая цепочка обязана обнаружить последующее повреждение выбранной цели.
    monkeypatch.setattr(store, "publish_json", publish)
    path = store.root / target
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ArtifactError):
        Preparation(store, ROOT, "integration_e1").prepare(
            FIXTURES, PROFILE, PLAN, batch_size=1024, max_batch_chars=4_194_304
        )


def test_resume_after_report_failure_reuses_completed_artifacts(tmp_path, capsys, monkeypatch):
    values = arguments("prepare", tmp_path / "run")
    publish = LocalArtifactStore.publish_json

    def fail_report(self, relative_path, value, **kwargs):
        if relative_path == REPORT:
            raise OSError("injected_before_preparation_report")
        return publish(self, relative_path, value, **kwargs)

    monkeypatch.setattr(LocalArtifactStore, "publish_json", fail_report)
    failed = invoke(capsys, values, success=False)
    assert "injected_before_preparation_report" in failed["message"]
    store = LocalArtifactStore(tmp_path / "run")
    assert not (store.root / REPORT).exists()
    before = descriptor_hashes(store)
    assert any(path.endswith(".parquet") for path in before)
    monkeypatch.setattr(LocalArtifactStore, "publish_json", publish)
    resumed = invoke(capsys, values)
    assert resumed["status"] == "prepared"
    assert all(store.describe(path).descriptor_sha256 == digest for path, digest in before.items())
    assert resumed["request_count"] == 2


@pytest.mark.parametrize("mutation", ["input", "plan", "code", "artifact"])
def test_incompatible_or_corrupted_checkpoint_is_rejected(tmp_path, capsys, monkeypatch, mutation):
    input_dir = copy_inputs(tmp_path / "input")
    plan_path = tmp_path / "plan.json"
    shutil.copyfile(PLAN, plan_path)
    run_dir = tmp_path / "run"
    values = arguments("prepare", run_dir, input_dir=input_dir, plan=plan_path)
    result = invoke(capsys, values)
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
        path = run_dir / result["datasets"][0]["prefixes"]
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
    assert LocalArtifactStore(run_dir).read_bytes(REPORT) == old_report


@pytest.mark.parametrize("component", ["payload", "descriptor", "object"])
@pytest.mark.parametrize("node", ["shared", "own"])
@pytest.mark.parametrize("mode", ["prepare", "reuse"])
def test_prepare_final_verification_rejects_late_artifact_corruption(
    tmp_path,
    capsys,
    monkeypatch,
    mode,
    node,
    component,
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
        relative = "canonical/index.json" if node == "shared" else REPORT
        descriptor = self.store.describe(relative)
        if component == "descriptor":
            path = self.store.root / (relative + ".artifact.json")
            data = json.loads(path.read_bytes())
            data["metadata"]["changed_after_verification"] = True
            encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
            path.write_bytes(encoded)
            (self.store.root / (relative + ".artifact.sha256")).write_text(
                sha256_bytes(encoded) + "\n", encoding="ascii"
            )
        else:
            path = self.store.root / (
                f".objects/{descriptor.sha256}" if component == "object" else relative
            )
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        changed = True

    monkeypatch.setattr(Preparation, "progress", corrupt_after_final_progress)
    capsys.readouterr()
    code = main(values)
    result = json.loads(capsys.readouterr().out)
    assert changed, "Инъекция должна сработать после последней записи progress"
    assert code != 0 and result["status"] == "failed", (
        f"{mode}/{node}/{component}: успех после позднего повреждения артефакта"
    )


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
    assert result["canonical_sessions"] == result["request_count"] == 7
    assert result["canonical_blocks"] == 14 and result["exclusions"] == {}
    datasets = {row["period"]: row for row in result["datasets"]}
    assert set(datasets) == set(PERIOD_NAMES)
    store = LocalArtifactStore(run_dir)
    access = DatasetAccess(
        store, code_sha256=store.describe(REPORT).code_sha256, run_id="integration_e1"
    )
    for period, dataset in datasets.items():
        assert result["selection"][period] == [f"s_{period}"]
        assert dataset["requests"] == 1
        assert "cohort" not in dataset
        for kind, purpose in [("prefixes", "prefix"), ("targets", "labels")]:
            path = dataset[kind]
            if period in {"A_G", "B", "C_smoke"}:
                assert access.read(path, purpose=purpose) == store.read_bytes(path)
            else:
                assert (run_dir / path).is_file()  # Запрет маршрута, а не отсутствие файла.
                with pytest.raises(InputValidationError):
                    access.read(path, purpose=purpose)
    with pytest.raises(InputValidationError):
        access.read(datasets["B"]["targets"], purpose="prefix")
    log_paths = [
        path
        for path in (run_dir / "access").glob("*.json")
        if not path.name.endswith(".artifact.json")
    ]
    logs = [
        json.loads(store.read_bytes(path.relative_to(run_dir).as_posix())) for path in log_paths
    ]
    assert len(logs) == 15 and sum(log["allowed"] for log in logs) == 6
    assert all(log["boundary"] == "local_routing_only" for log in logs)
    assert {(log["path"], log["purpose"]) for log in logs if not log["allowed"]} == {
        *(
            (datasets[period][kind], purpose)
            for period in ("A_N", "C_select", "E", "D")
            for kind, purpose in (("prefixes", "prefix"), ("targets", "labels"))
        ),
        (datasets["B"]["targets"], "prefix"),
    }
