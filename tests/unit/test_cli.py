import json
import os
import subprocess
import sys

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
        [sys.executable, "-m", "coveo_recsys", "train"],
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    assert unsupported.returncode == 2
    assert "Команда train ещё не реализована" in unsupported.stderr
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
