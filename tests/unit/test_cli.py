import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from coveo_recsys.cli import _parser, main


@pytest.mark.parametrize(
    "arguments",
    [
        ["verify", "register", "--run-id", "run", "--run-dir", "runs/test"],
        [
            "manifest",
            "freeze",
            "--manifest",
            "manifest.json",
            "--run-dir",
            "runs/test",
            "--stage",
            "preflight",
            "--snapshot-id",
            "snapshot",
        ],
    ],
)
def test_removed_registration_and_snapshot_commands_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["train", "serve", "evaluate"])
def test_future_commands_fail_explicitly(command):
    result = subprocess.run([sys.executable, "-m", "coveo_recsys", command], capture_output=True)
    assert result.returncode == 2


def test_historical_build_has_explicit_inputs():
    args = _parser().parse_args(
        [
            "historical",
            "build",
            "--config",
            "configs/coveo_historical.json",
            "--run-dir",
            "runs/e2",
            "--run-id",
            "e2",
        ]
    )
    assert args.action == "build"
    assert str(args.config).replace("\\", "/") == "configs/coveo_historical.json"


def test_features_build_has_explicit_period_and_inputs():
    args = _parser().parse_args(
        [
            "features",
            "build",
            "--config",
            "features.json",
            "--run-dir",
            "runs/e3",
            "--run-id",
            "e3",
            "--period",
            "B",
        ]
    )
    assert args.action == "build" and args.period == "B"
    assert str(args.config) == "features.json"


def test_serve_requires_explicit_config_and_local_run_directory():
    args = _parser().parse_args(
        ["serve", "--config", "configs/coveo_service.json", "--run-dir", "runs/service"]
    )
    assert args.command == "serve"
    assert args.host == "127.0.0.1" and args.port == 8000
    assert args.config.as_posix() == "configs/coveo_service.json"


@pytest.mark.parametrize(
    "field,value",
    [("timeout_ms", value) for value in (True, 0, -1, 1.5, "1000", None, 100, 2**63)]
    + [("reserve_ms", value) for value in (True, 0, -1, 1.5, "100", None, 1000, 1001, 2**63)],
)
def test_serve_rejects_invalid_timing_before_loading_package(tmp_path, monkeypatch, field, value):
    from coveo_recsys.cli import serving

    settings = json.loads(
        (Path(__file__).parents[2] / "configs/coveo_service.json").read_text(encoding="utf-8")
    )
    settings.update(timeout_ms=1000, reserve_ms=100)
    settings[field] = value
    config = tmp_path / "service.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("COVEO_LOCAL_TOKEN", "test-secret")
    monkeypatch.setattr(serving, "load_serving_package", lambda *_: pytest.fail("package read"))
    with pytest.raises(ValueError, match="timeout_ms|reserve_ms"):
        serving.run_serve(tmp_path, config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_environment_command(capsys):
    assert main(["environment"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "development_environment"
    assert result["libraries"]["lightgbm"] == "4.6.0"
    assert result["load_measurements"] is None


def test_no_action_is_not_success():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "content",
    [
        '{"stage": "preflight", "stage": "post_D"}',
        '{"value": NaN}',
        '{"value": [1e999]}',
        "[]",
    ],
)
def test_manifest_command_rejects_invalid_json_before_validation(tmp_path, capsys, content):
    path = tmp_path / "manifest.json"
    path.write_text(content, encoding="utf-8")
    assert (
        main(
            [
                "manifest",
                "validate",
                "--manifest",
                str(path),
                "--run-dir",
                str(tmp_path / "run"),
                "--stage",
                "preflight",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and result["reason"] == "ValueError"


def test_redirected_cli_output_is_utf8(tmp_path):
    environment = dict(os.environ, PYTHONIOENCODING="cp1251")
    unsupported = subprocess.run(
        [sys.executable, "-m", "coveo_recsys", "select"],
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    assert unsupported.returncode == 2
    assert "Команда select ещё не реализована" in unsupported.stderr
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "coveo_recsys",
            "manifest",
            "init",
            "--run-id",
            "",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    assert invalid.returncode == 2
    report = json.loads(invalid.stdout)
    assert report["status"] == "failed"
    assert any("а" <= char <= "я" for char in report["message"].lower())


def test_train_ranker_parser_has_explicit_inputs():
    args = _parser().parse_args(
        [
            "train",
            "ranker",
            "--config",
            "training.json",
            "--run-dir",
            "runs/example",
            "--run-id",
            "example",
        ]
    )
    assert args.action == "ranker" and args.config.name == "training.json"


def test_evaluation_commands_have_explicit_period_and_saved_run():
    args = _parser().parse_args(
        [
            "replay",
            "--config",
            "replay.json",
            "--period",
            "C_smoke",
            "--run-dir",
            "runs/smoke",
            "--run-id",
            "smoke",
        ]
    )
    assert args.period == "C_smoke" and args.config.name == "replay.json"
    assert _parser().parse_args(["report", "--run", "runs/smoke"]).run.name == "smoke"
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "replay",
                "--config",
                "replay.json",
                "--period",
                "D",
                "--run-dir",
                "runs/d",
                "--run-id",
                "d",
            ]
        )
