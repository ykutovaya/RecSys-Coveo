"""Малый полный путь Э5 на синтетической готовой модели и трёх сессиях."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.contracts import OutcomeStatus
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from coveo_recsys.state import build_prefix
from tests.integration.test_e3_pipeline import e3_settings
from tests.integration.test_e4_pipeline import e4_settings
from tests.reference.e3_cases import request
from tests.unit.test_e2_input import prepared_source  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["GCS+Rk", "P", "R"]


@pytest.fixture(scope="module")
def replay_template(prepared_source, tmp_path_factory):  # noqa: F811
    from coveo_recsys.cli.training import run_training

    root = tmp_path_factory.mktemp("e5-template")
    feature_settings, training_source, historical = e3_settings.__wrapped__(prepared_source, root)
    training_settings, _, features = e4_settings.__wrapped__(
        (feature_settings, training_source, historical), root
    )
    model = root / "model"
    run_training(ROOT, training_settings, model, "e5-producer")
    training_report = json.loads((training_source / "reports/preparation.json").read_bytes())
    plan = from_json(training_report["plan_json"].encode(), PreparationPlan)
    plan = replace(
        plan,
        periods=tuple(
            replace(p, sample_size=3 if p.name == "C_smoke" else 0) for p in plan.periods
        ),
    )
    plan_path = root / "c-plan.json"
    plan_path.write_bytes(to_json(plan))
    source = root / "evaluation"
    tables = {}
    with (
        TableWriter(source / "datasets/C_smoke/prefixes.parquet", PreparedRequest) as prefixes,
        TableWriter(source / "canonical/C_smoke.parquet", CanonicalBlock) as canonical,
        TableWriter(source / "datasets/C_smoke/targets.parquet", TargetRecord) as targets,
    ):
        for i in range(3):
            sid, qid, instant = f"e{i}", f"eq{i}", 600001000 + i * 2000
            events = ((instant, "detail", "A"),) * (i + 1)
            if i != 1:
                events += ((instant + 500, "purchase", "C" if i == 0 else "outside"),)
            base = request(events, time=instant)
            blocks = tuple(
                replace(
                    block,
                    session_id=sid,
                    events=tuple(
                        replace(event, session_id=sid, source_namespace=plan.source_namespace)
                        for event in block.events
                    ),
                )
                for block in base.prefix.blocks
            )
            prefix = replace(
                base.prefix, session_id=sid, source_namespace=plan.source_namespace, blocks=blocks
            )
            prepared = replace(base, request_id=qid, session_id=sid, prefix=prefix)
            prefixes.append(sid, (prepared,))
            full_blocks = tuple(
                replace(
                    block,
                    session_id=sid,
                    events=tuple(
                        replace(event, session_id=sid, source_namespace=plan.source_namespace)
                        for event in block.events
                    ),
                )
                for block in request(events).prefix.blocks
            )
            canonical.append(sid, full_blocks)
            target = TargetRecord(
                qid, ("C",) if i == 0 else ("outside",), OutcomeStatus.KNOWN, plan.task.horizon_ms
            )
            if i == 1:
                target = TargetRecord(
                    qid, (), OutcomeStatus.UNKNOWN, plan.task.horizon_ms, "no_registered_purchase"
                )
            targets.append(sid, (target,))
    for path, kind, role in (
        ("datasets/C_smoke/prefixes.parquet", PreparedRequest, "prefix"),
        ("datasets/C_smoke/targets.parquet", TargetRecord, "targets"),
        ("canonical/C_smoke.parquet", CanonicalBlock, "canonical"),
    ):
        tables[path] = table_info(source / path, kind, period="C_smoke", role=role)
    report = dict(training_report)
    report.update(
        run_id="e5-input",
        tables=tables,
        plan_json=to_json(plan).decode(),
        selection={p.name: ["e0", "e1", "e2"] if p.name == "C_smoke" else [] for p in plan.periods},
    )
    report["inputs"] = {**report["inputs"], "plan": sha256_bytes(to_json(plan))}
    report.update(
        N_obs=3,
        N_plus=2,
        request_count=3,
        exclusions={},
        period_counts={
            p.name: {
                "admitted": 3 if p.name == "C_smoke" else 0,
                "eligible": 3 if p.name == "C_smoke" else 0,
                "selected": 3 if p.name == "C_smoke" else 0,
            }
            for p in plan.periods
        },
    )
    (source / "reports").mkdir()
    (source / "reports/preparation.json").write_bytes(canonical_json_bytes(report))
    estimate_names = [*POLICIES, "GCS+Rk-P", "GCS+Rk-R"]
    evaluation_plan = {
        "period": "C_smoke",
        "policies": POLICIES,
        "k": 20,
        "diagnostic_depth": 2,
        "intervals": {
            "grid_step_ms": 1981990,
            "block_cells": 2,
            "sensitivity_block_cells": [1, 3],
            "mc_error_method": "binomial_order_statistic_95pct",
            "dependence": {name: {"supported": None, "assumptions": []} for name in estimate_names},
            "half_width_fraction": dict.fromkeys(estimate_names),
        },
    }
    configuration = json.loads(training_settings.read_bytes())
    settings = {
        "preparation": {
            "path": str(source),
            "report_sha256": sha256_file(source / "reports/preparation.json"),
        },
        "training_preparation": configuration["preparation"],
        "training_features": configuration["features"],
        "historical": configuration["historical"],
        "model": {"path": str(model), "report_sha256": sha256_file(model / "reports/ranker.json")},
        "preparation_plan": str(plan_path),
        "feature_schema": "configs/feature_schema.json",
        "recipe": "configs/ranker_recipe.json",
        "retrieval": {},
        "evaluation_plan": evaluation_plan,
        "limits": {
            "max_requests": 3,
            "max_observed_skus": 100,
            "batch_size": 2,
            "max_seconds": 300,
            "max_memory_bytes": 2147483648,
            "max_result_bytes": 67108864,
        },
    }
    return settings, {
        "source": source,
        "training_source": training_source,
        "historical": historical,
        "model": model,
        "features": features,
        "plan": plan_path,
    }


@pytest.fixture
def replay_settings(replay_template, tmp_path):
    settings, roots = replay_template
    path = tmp_path / "replay.json"
    path.write_bytes(canonical_json_bytes(settings))
    return path, roots


def test_replay_three_policies_unknown_and_targets_after_saved_answers(
    replay_settings, tmp_path, monkeypatch
):
    from coveo_recsys.cli import replay
    from coveo_recsys.contracts.evaluation import EvaluationRecord, ReplayRecord
    from coveo_recsys.data.access import DatasetAccess

    settings, _ = replay_settings
    destination = tmp_path / "result"
    original_read = DatasetAccess.read
    target_reads = []

    def checked_read(self, path, kind, *, purpose):
        if purpose == "labels":
            saved = list(iter_sessions(destination / "replay/answers.parquet", ReplayRecord))
            assert len(saved) == 3 and all(len(items) == 3 for _, items in saved)
            target_reads.append(path)
        return original_read(self, path, kind, purpose=purpose)

    monkeypatch.setattr(DatasetAccess, "read", checked_read)
    result = replay.run_replay(ROOT, settings, destination, "e5")
    assert result["status"] == "completed" and result["requests"] == 3
    assert target_reads == ["datasets/C_smoke/targets.parquet"]
    answers = list(iter_sessions(destination / "replay/answers.parquet", ReplayRecord))
    assert [sid for sid, _ in answers] == ["e0", "e1", "e2"]
    assert all([row.policy for row in items] == POLICIES for _, items in answers)
    sessions = dict(iter_sessions(destination / "replay/sessions.parquet", EvaluationRecord))
    assert all(row.evaluation.hit_s is None for row in sessions["e1"])
    assert all(row.evaluation.hit_s == 0 for row in sessions["e2"])
    for _, items in answers:
        for row in items:
            diagnostics = json.loads(row.diagnostics_json)
            assert "features" not in diagnostics
            assert (
                diagnostics["feature_rows"] is None
                if row.policy != "GCS+Rk"
                else diagnostics["feature_rows"] > 0
            )


def test_completed_replay_repeat_is_read_only_without_new_recommendations(
    replay_settings, tmp_path, monkeypatch
):
    from coveo_recsys.cli import replay

    settings, roots = replay_settings
    destination = tmp_path / "result"
    result = replay.run_replay(ROOT, settings, destination, "e5")
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for base in (destination, roots["model"], roots["historical"])
        for path in base.rglob("*")
        if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Повтор не запускает выдачу")

    monkeypatch.setattr(replay, "recommend", forbidden)
    assert replay.run_replay(ROOT, settings, destination, "e5") == result
    assert before == {path: (sha256_file(path), path.stat().st_mtime_ns) for path in before}


def test_interruption_does_not_publish_completed_run_and_retry_succeeds(replay_settings, tmp_path):
    from coveo_recsys.cli.replay import run_replay

    settings, _ = replay_settings
    destination = tmp_path / "result"

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_replay(ROOT, settings, destination, "e5", before_publish=interrupt)
    assert not (destination / "reports/replay.json").exists()
    assert run_replay(ROOT, settings, destination, "e5")["status"] == "completed"


def test_report_uses_only_saved_results_and_can_restore_markdown(
    replay_settings, tmp_path, monkeypatch
):
    from coveo_recsys.cli import replay

    settings, _ = replay_settings
    destination = tmp_path / "result"
    result = replay.run_replay(ROOT, settings, destination, "e5")
    markdown = destination / "reports/replay.md"
    content = markdown.read_bytes()
    markdown.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Отчёт читает только сохранённые результаты")

    monkeypatch.setattr(replay, "recommend", forbidden)
    monkeypatch.setattr(replay, "load_ranker", forbidden)
    monkeypatch.setattr(replay, "assert_source_binding", forbidden)
    assert replay.read_report(destination) == result
    assert markdown.read_bytes() == content
    markdown.write_bytes(b"foreign")
    with pytest.raises(ValueError):
        replay.read_report(destination)


def pilot_ids(count):
    def key(sid):
        encoded = sid.encode("utf-8")
        payload = (17).to_bytes(8, "big") + len(encoded).to_bytes(8, "big") + encoded
        return hashlib.sha256(payload).digest(), encoded

    return sorted(("e0", "e1", "e2"), key=key)[:count]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_explicit_pilot_uses_hash_subset_for_all_policies_and_metrics(
    replay_settings, tmp_path, monkeypatch, count
):
    from coveo_recsys.cli import replay
    from coveo_recsys.contracts.evaluation import EvaluationRecord, ReplayRecord

    settings_path, roots = replay_settings
    settings = json.loads(settings_path.read_bytes())
    settings.update(pilot_sessions=count)
    settings["limits"]["max_requests"] = count
    settings_path.write_bytes(canonical_json_bytes(settings))
    expected = pilot_ids(count)
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for root in (roots["source"], roots["model"], roots["historical"])
        for path in root.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "pilot"
    calls = []
    original = replay.recommend

    def record(context, *args, **kwargs):
        calls.append((context.session_id, kwargs["policy"]))
        return original(context, *args, **kwargs)

    monkeypatch.setattr(replay, "recommend", record)
    result = replay.run_replay(ROOT, settings_path, destination, "pilot")
    assert result["requests"] == count
    assert result["identity"]["pilot_sessions"] == count
    assert result["identity"]["selected_ids"] == expected
    assert result["identity"]["selected_count"] == count
    assert result["identity"]["selected_ids_sha256"] == sha256_bytes(canonical_json_bytes(expected))
    assert set(calls) == {(sid, policy) for sid in expected for policy in POLICIES}
    assert len(calls) == count * 3
    for filename, kind in (("answers", ReplayRecord), ("sessions", EvaluationRecord)):
        rows = dict(iter_sessions(destination / f"replay/{filename}.parquet", kind))
        assert set(rows) == set(expected)
        assert all([row.policy for row in items] == POLICIES for items in rows.values())
    known = len(set(expected) & {"e0", "e2"})
    hit = int("e0" in expected)
    assert result["data"]["prepared_selected_size"] == 3
    assert result["data"]["selected_size"] == count
    assert result["data"]["evaluation_scope"] == "pilot"
    for value in result["policies"].values():
        assert value["counts"]["N_obs"] == count
        assert value["counts"]["N"] == known
        assert value["metrics"]["Hit@20"]["value"] == (hit / known if known else None)
    assert before == {path: (sha256_file(path), path.stat().st_mtime_ns) for path in before}

    def forbidden(*args, **kwargs):
        raise AssertionError("Повтор пилота не пересчитывает рекомендации")

    monkeypatch.setattr(replay, "recommend", forbidden)
    assert replay.run_replay(ROOT, settings_path, destination, "pilot") == result
    assert replay.read_report(destination) == result


def test_pilot_collects_metadata_only_for_selected_prefixes(replay_settings, tmp_path):
    from coveo_recsys.cli.replay import run_replay

    settings_path, roots = replay_settings
    source = tmp_path / "prepared-with-external-sku"
    shutil.copytree(roots["source"], source)
    expected = pilot_ids(2)
    excluded = next(sid for sid in ("e0", "e1", "e2") if sid not in expected)
    path = "datasets/C_smoke/prefixes.parquet"
    rows = dict(iter_sessions(source / path, PreparedRequest))
    item = rows[excluded][0]
    block = replace(
        item.prefix.blocks[0],
        events=tuple(
            replace(event, sku="outside-prefix") for event in item.prefix.blocks[0].events
        ),
    )
    prefix = build_prefix(
        (block,),
        source_namespace=item.source_namespace,
        session_id=excluded,
        event_cutoff_ms=item.t_q,
        availability_cutoff_ms=item.a_q,
        config=from_json((roots["plan"]).read_bytes(), PreparationPlan).task,
    )
    rows[excluded] = (replace(item, prefix=prefix),)
    (source / path).unlink()
    with TableWriter(source / path, PreparedRequest) as writer:
        for sid, items in rows.items():
            writer.append(sid, items)
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"][path] = table_info(
        source / path, PreparedRequest, period="C_smoke", role="prefix"
    )
    canonical_path = "canonical/C_smoke.parquet"
    canonical_rows = dict(iter_sessions(source / canonical_path, CanonicalBlock))
    canonical_rows[excluded] = (block, *canonical_rows[excluded][1:])
    (source / canonical_path).unlink()
    with TableWriter(source / canonical_path, CanonicalBlock) as writer:
        for sid, blocks in canonical_rows.items():
            writer.append(sid, blocks)
    report["tables"][canonical_path] = table_info(
        source / canonical_path, CanonicalBlock, period="C_smoke", role="canonical"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(settings_path.read_bytes())
    settings.update(
        pilot_sessions=2,
        preparation={"path": str(source), "report_sha256": sha256_file(report_path)},
    )
    settings["limits"]["max_requests"] = 2
    settings["limits"]["max_observed_skus"] = 1
    settings_path.write_bytes(canonical_json_bytes(settings))
    assert run_replay(ROOT, settings_path, tmp_path / "pilot", "pilot")["requests"] == 2
