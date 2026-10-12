"""Настройки малого воспроизведения не используют прежние результаты."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.evaluation.reporting import validate_evaluation_plan
from coveo_recsys.verification import reproduce

ROOT = Path(__file__).resolve().parents[2]


def test_helper_help_is_utf8_in_redirected_windows_terminal():
    result = subprocess.run(
        [sys.executable, "-m", "coveo_recsys.verification.reproduce", "--help"],
        capture_output=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONIOENCODING="cp1251"),
    )
    assert result.returncode == 0
    assert "Настройки и проверки" in result.stdout


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "clone"
    (root / "configs").mkdir(parents=True)
    for name in ("coveo_preparation.json", "coveo_historical.json"):
        shutil.copyfile(ROOT / "configs" / name, root / "configs" / name)
    inputs = tmp_path / "csv"
    inputs.mkdir()
    for name in reproduce.CSV_FILES:
        (inputs / name).write_text("header\n", encoding="utf-8")
    monkeypatch.setattr(reproduce, "assert_source_binding", lambda _: None)
    return root, root / "runs/small", inputs


def initialize(workspace):
    root, work, inputs = workspace
    reproduce.initialize(
        root,
        work,
        inputs,
        ag_sessions=20,
        b_sessions=10,
        smoke_sessions=5,
        max_seconds=60,
        max_memory_bytes=2**30,
        max_temp_bytes=2**30,
    )
    return root, work, inputs


def test_plans_change_only_sizes_and_resource_limits(workspace):
    root, work, _ = initialize(workspace)
    original = json.loads((root / "configs/coveo_preparation.json").read_text())["data"]
    for period, count in (("A_G", 20), ("B", 10), ("C_smoke", 5)):
        raw = (work / f"{period}-plan.json").read_bytes()
        plan = from_json(raw, PreparationPlan)
        current = json.loads(raw)["data"]
        assert plan.diagnostic_sessions == 0
        assert sum(p.sample_size for p in plan.periods) == count
        assert next(p.sample_size for p in plan.periods if p.name == period) == count
        for key in ("task", "coverage", "seed", "source_namespace"):
            assert current[key] == original[key]
        assert [(p.start_ms, p.end_ms) for p in plan.periods] == [
            (p["start_ms"], p["end_ms"]) for p in original["periods"]
        ]
    validate_evaluation_plan(reproduce.smoke_evaluation_plan(), plan)


@pytest.mark.parametrize("target", [".", "configs/new", "../outside", "runs"])
def test_output_is_a_new_child_of_runs(workspace, target):
    root, _, inputs = workspace
    with pytest.raises((InputValidationError, FileExistsError)):
        reproduce.initialize(
            root,
            root / target,
            inputs,
            ag_sessions=20,
            b_sessions=10,
            smoke_sessions=5,
            max_seconds=60,
            max_memory_bytes=2**30,
            max_temp_bytes=2**30,
        )


def test_existing_directory_is_not_reinitialized(workspace):
    root, work, inputs = initialize(workspace)
    snapshot = {p.name: p.read_bytes() for p in work.iterdir()}
    with pytest.raises(FileExistsError):
        initialize(workspace)
    assert snapshot == {p.name: p.read_bytes() for p in work.iterdir()}


def test_historical_settings_pin_new_preparation_not_old_run(workspace):
    root, work, inputs = initialize(workspace)
    reports = work / "a_g/reports"
    reports.mkdir(parents=True)
    (reports / "preparation.json").write_text('{"status":"completed"}')
    destination = reproduce.make_config(root, work, inputs, "historical", max_result_bytes=2**30)
    config = json.loads(destination.read_bytes())
    assert config["preparation"]["path"] == "runs/small/a_g"
    assert config["metadata_path"] == (inputs / "sku_to_content.csv").as_posix()
    assert config["config"]["graph_min_support"] == 3
    assert config["config"]["max_seconds"] == 60
    assert not (root / "runs/coveo-ag50000-compact-v1").exists()
    (reports / "preparation.json").write_text('{"status":"completed","changed":true}')
    with pytest.raises(InputValidationError, match="измен"):
        reproduce.make_config(root, work, inputs, "historical", max_result_bytes=2**30)


def test_incomplete_input_does_not_produce_config(workspace):
    root, work, inputs = initialize(workspace)
    reports = work / "a_g/reports"
    reports.mkdir(parents=True)
    (reports / "preparation.json").write_text('{"status":"interrupted"}')
    with pytest.raises(InputValidationError, match="заверш"):
        reproduce.make_config(root, work, inputs, "historical", max_result_bytes=2**30)
    assert not (work / "historical.json").exists()


def test_replay_requires_full_independent_data_evidence(workspace, monkeypatch):
    from coveo_recsys.artifacts import sha256_file
    from coveo_recsys.verification.provenance import source_sha256

    root, work, inputs = initialize(workspace)
    reports = work / "c_smoke/reports"
    reports.mkdir(parents=True)
    (reports / "preparation.json").write_text('{"status":"completed"}')
    # Этих трёх полей недостаточно: нужны правила, счётчики, план и связи входов.
    (reports / "data-check.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "current_code_sha256": source_sha256(root),
                "bindings": {"report_sha256": sha256_file(reports / "preparation.json")},
            }
        )
    )
    (work / "evaluation-preparation.json").write_text('{"preparation":null}')
    with pytest.raises((InputValidationError, ValueError)):
        reproduce.make_config(root, work, inputs, "replay", max_result_bytes=2**30)
    assert not (work / "replay.json").exists()
