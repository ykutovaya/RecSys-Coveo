"""Вариант без метаданных использует существующие команды B."""

from pathlib import Path

import pytest

from coveo_recsys.cli import main
from coveo_recsys.contracts.retrieval import NO_METADATA_CONFIG_ID


def test_evaluate_calls_shared_runner_only_for_d(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr("coveo_recsys.cli.replay.run_replay", run)
    assert (
        main(["evaluate", "--config", "final.json", "--run-dir", "runs/final", "--run-id", "final"])
        == 0
    )
    args, kwargs = calls[0]
    assert args[1:] == (Path("final.json"), Path("runs/final"), "final")
    assert kwargs == {"period": "D"}


@pytest.mark.parametrize("step", ["features", "train"])
def test_metadata_command_calls_existing_b_stage(monkeypatch, step):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "completed"}

    module, function = (
        ("features", "run_features") if step == "features" else ("training", "run_training")
    )
    monkeypatch.setattr(f"coveo_recsys.cli.{module}.{function}", run)
    assert (
        main(
            [
                "sensitivity",
                "metadata",
                "--step",
                step,
                "--config",
                "variant.json",
                "--run-dir",
                "runs/variant",
                "--run-id",
                "variant",
            ]
        )
        == 0
    )
    args, kwargs = calls[0]
    assert args[1:] == (Path("variant.json"), Path("runs/variant"), "variant")
    expected = {"required_config_id": NO_METADATA_CONFIG_ID}
    if step == "features":
        expected["period"] = "B"
    assert kwargs == expected


@pytest.mark.parametrize("period", ["C_select", "D", "E", "A_N"])
def test_metadata_command_does_not_open_other_periods(period):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "sensitivity",
                "metadata",
                "--step",
                "features",
                "--config",
                "variant.json",
                "--run-dir",
                "runs/variant",
                "--run-id",
                "variant",
                "--period",
                period,
            ]
        )
    assert exc.value.code == 2
