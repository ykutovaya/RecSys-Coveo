"""Допуск, происхождение производителей и подмены зависимостей Э5."""

# ruff: noqa: F811

import json
import shutil
from dataclasses import replace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from coveo_recsys.state import build_prefix
from tests.integration.test_e5_replay_pipeline import (  # noqa: F401
    ROOT,
    prepared_source,
    replay_settings,
    replay_template,
)


def test_old_model_producer_is_preserved_by_new_consumer(replay_settings, tmp_path, monkeypatch):
    from coveo_recsys.cli import replay

    settings, roots = replay_settings
    original = json.loads((roots["model"] / "reports/ranker.json").read_bytes())
    monkeypatch.setattr(replay, "source_sha256", lambda *_: "f" * 64)
    result = replay.run_replay(ROOT, settings, tmp_path / "result", "new-consumer")
    assert result["identity"]["source_sha256"] == "f" * 64
    assert result["identity"]["model_identity"] == original["identity"]
    assert (
        sha256_file(roots["model"] / "reports/ranker.json")
        == json.loads(settings.read_bytes())["model"]["report_sha256"]
    )


@pytest.mark.parametrize("period", ["A_G", "A_N", "B", "E"])
def test_closed_period_is_rejected_before_loading_inputs(tmp_path, period):
    from coveo_recsys.cli.replay import run_replay

    with pytest.raises(ValueError):
        run_replay(ROOT, tmp_path / "absent.json", tmp_path / "result", "e5", period=period)


@pytest.mark.parametrize("name", ["model", "schema", "recipe", "plan", "retrieval", "features"])
def test_substituted_pinned_input_is_rejected_before_prefix_access(
    replay_settings, tmp_path, monkeypatch, name
):
    from coveo_recsys.cli.replay import run_replay
    from coveo_recsys.data.access import DatasetAccess

    path, roots = replay_settings
    settings = json.loads(path.read_bytes())
    if name == "model":
        model = tmp_path / "damaged-model"
        shutil.copytree(roots["model"], model)
        (model / "models/ranker.txt").write_bytes(b"invalid")
        settings["model"]["path"] = str(model)
    elif name == "retrieval":
        settings["retrieval"]["history_limit"] = 19
    elif name == "features":
        settings["training_features"]["report_sha256"] = "f" * 64
    else:
        key = {"schema": "feature_schema", "recipe": "recipe", "plan": "preparation_plan"}[name]
        original = ROOT / settings[key]
        replacement = tmp_path / f"{name}.json"
        replacement.write_bytes(original.read_bytes() + b" ")
        settings[key] = str(replacement)
    path.write_bytes(canonical_json_bytes(settings))

    def forbidden(*args, **kwargs):
        raise AssertionError("Отказ должен предшествовать чтению C_smoke")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    with pytest.raises(ValueError):
        run_replay(ROOT, path, tmp_path / "result", "e5")


def test_completed_output_damage_and_foreign_run_id_are_rejected(replay_settings, tmp_path):
    from coveo_recsys.cli.replay import run_replay

    settings, _ = replay_settings
    destination = tmp_path / "result"
    run_replay(ROOT, settings, destination, "e5")
    with pytest.raises(ValueError):
        run_replay(ROOT, settings, destination, "other")
    (destination / "replay/answers.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        run_replay(ROOT, settings, destination, "e5")


def test_limits_refuse_the_whole_dataset_without_truncating(replay_settings, tmp_path):
    from coveo_recsys.cli.replay import run_replay

    path, _ = replay_settings
    settings = json.loads(path.read_bytes())
    settings["limits"]["max_requests"] = 2
    path.write_bytes(canonical_json_bytes(settings))
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="запрос"):
        run_replay(ROOT, path, destination, "e5")
    assert not (destination / "reports/replay.json").exists()


def test_evaluation_admission_runs_before_opening_any_c_table(
    replay_settings, tmp_path, monkeypatch
):
    from coveo_recsys.cli import admission
    from coveo_recsys.cli.replay import run_replay
    from coveo_recsys.data.access import DatasetAccess

    settings, _ = replay_settings
    calls = []

    def denied(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("B12 gate denied")

    def forbidden(*args, **kwargs):
        raise AssertionError("Таблицы C открыты до допуска")

    monkeypatch.setattr(admission, "require_replay_admission", denied)
    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    with pytest.raises(ValueError, match="B12"):
        run_replay(ROOT, settings, tmp_path / "result", "e5")
    assert len(calls) == 1 and calls[0]["evaluate"] is True


@pytest.mark.parametrize("damaged", ["settings", "canonical", "model"])
def test_changed_dependency_before_publication_is_rejected(replay_settings, tmp_path, damaged):
    from coveo_recsys.cli.replay import run_replay

    settings_path, roots = replay_settings
    settings = json.loads(settings_path.read_bytes())
    if damaged == "settings":
        path = settings_path
    elif damaged == "canonical":
        copied = tmp_path / "copied-source"
        shutil.copytree(roots["source"], copied)
        settings["preparation"]["path"] = str(copied)
        path = copied / "canonical/C_smoke.parquet"
    else:
        copied = tmp_path / "copied-model"
        shutil.copytree(roots["model"], copied)
        settings["model"]["path"] = str(copied)
        path = copied / "models/ranker.txt"
    settings_path.write_bytes(canonical_json_bytes(settings))

    def corrupt():
        path.write_bytes(path.read_bytes() + b"changed")

    destination = tmp_path / "result"
    with pytest.raises(ValueError):
        run_replay(ROOT, settings_path, destination, "e5", before_publish=corrupt)
    assert not (destination / "reports/replay.json").exists()


def test_late_request_is_rejected_before_targets_open(replay_settings, tmp_path, monkeypatch):
    from coveo_recsys.cli.replay import run_replay
    from coveo_recsys.data.access import DatasetAccess

    settings_path, roots = replay_settings
    copied = tmp_path / "late-input"
    shutil.copytree(roots["source"], copied)
    report_path = copied / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = from_json(report["plan_json"].encode(), PreparationPlan)
    period = next(period for period in plan.periods if period.name == "C_smoke")
    late = period.end_ms - plan.task.horizon_ms - plan.task.gap_ms
    prefixes_path = "datasets/C_smoke/prefixes.parquet"
    blocks_path = "canonical/C_smoke.parquet"
    requests = dict(iter_sessions(copied / prefixes_path, PreparedRequest))
    blocks = dict(iter_sessions(copied / blocks_path, CanonicalBlock))
    delta = late - requests["e0"][0].t_q
    blocks["e0"] = tuple(
        replace(
            block,
            event_time_ms=block.event_time_ms + delta,
            available_time_ms=block.available_time_ms + delta,
            events=tuple(
                replace(
                    event,
                    event_time_ms=event.event_time_ms + delta,
                    available_time_ms=event.available_time_ms + delta,
                )
                for event in block.events
            ),
        )
        for block in blocks["e0"]
    )
    prefix = build_prefix(
        blocks["e0"],
        source_namespace=plan.source_namespace,
        session_id="e0",
        event_cutoff_ms=late,
        availability_cutoff_ms=late,
        config=plan.task,
    )
    requests["e0"] = (replace(requests["e0"][0], t_q=late, a_q=late, prefix=prefix),)
    for path, kind, rows, role in (
        (prefixes_path, PreparedRequest, requests, "prefix"),
        (blocks_path, CanonicalBlock, blocks, "canonical"),
    ):
        (copied / path).unlink()
        with TableWriter(copied / path, kind) as writer:
            for sid, items in rows.items():
                writer.append(sid, items)
        report["tables"][path] = table_info(copied / path, kind, period="C_smoke", role=role)
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(settings_path.read_bytes())
    settings["preparation"] = {"path": str(copied), "report_sha256": sha256_file(report_path)}
    settings_path.write_bytes(canonical_json_bytes(settings))
    original = DatasetAccess.read

    def guarded(self, path, kind, *, purpose):
        assert purpose != "labels", "Неверное время запроса должно отклоняться до целей"
        return original(self, path, kind, purpose=purpose)

    monkeypatch.setattr(DatasetAccess, "read", guarded)
    with pytest.raises(ValueError):
        run_replay(ROOT, settings_path, tmp_path / "result", "e5")


@pytest.mark.parametrize("size", [0, -1, True, None, 1.5, "2", 4])
def test_invalid_pilot_size_is_rejected_before_table_access(
    replay_settings, tmp_path, monkeypatch, size
):
    from coveo_recsys.cli.replay import run_replay
    from coveo_recsys.data.access import DatasetAccess

    settings_path, _ = replay_settings
    settings = json.loads(settings_path.read_bytes())
    settings["pilot_sessions"] = size
    settings_path.write_bytes(canonical_json_bytes(settings))

    def forbidden(*args, **kwargs):
        raise AssertionError("Размер пилота проверяется до чтения таблиц")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    with pytest.raises(ValueError):
        run_replay(ROOT, settings_path, tmp_path / "pilot", "pilot")


def test_pilot_budget_does_not_silently_reduce_explicit_size(replay_settings, tmp_path):
    from coveo_recsys.cli.replay import run_replay

    settings_path, _ = replay_settings
    settings = json.loads(settings_path.read_bytes())
    settings["pilot_sessions"] = 2
    settings["limits"]["max_requests"] = 1
    settings_path.write_bytes(canonical_json_bytes(settings))
    with pytest.raises(ValueError, match="запрос"):
        run_replay(ROOT, settings_path, tmp_path / "pilot", "pilot")


def test_pilot_still_validates_full_source_request_count(replay_settings, tmp_path, monkeypatch):
    from coveo_recsys.cli.replay import run_replay
    from coveo_recsys.data.access import DatasetAccess

    settings_path, roots = replay_settings
    source = tmp_path / "wrong-full-count"
    shutil.copytree(roots["source"], source)
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"]["datasets/C_smoke/prefixes.parquet"]["rows"] = 4
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(settings_path.read_bytes())
    settings.update(
        pilot_sessions=1,
        preparation={"path": str(source), "report_sha256": sha256_file(report_path)},
    )
    settings["limits"]["max_requests"] = 1
    settings_path.write_bytes(canonical_json_bytes(settings))

    def forbidden(*args, **kwargs):
        raise AssertionError("Полный счётчик должен проверяться до фильтра и чтения")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    with pytest.raises(ValueError, match="Число первых запросов"):
        run_replay(ROOT, settings_path, tmp_path / "pilot", "pilot")
