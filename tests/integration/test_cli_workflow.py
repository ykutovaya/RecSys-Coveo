import json
from pathlib import Path

from coveo_recsys.cli import main


def test_one_manifest_without_snapshots_and_blocked_incomplete_preflight(tmp_path, capsys):
    run_dir = tmp_path / "run"
    arguments = ["manifest", "init", "--run-id", "test_e0", "--run-dir", str(run_dir)]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["manifest_version"] == "2.8"
    manifest_path = run_dir / "manifest.json"
    assert sorted(path.name for path in run_dir.iterdir()) == ["manifest.json"]
    original = manifest_path.read_bytes()
    assert main(arguments) != 0
    capsys.readouterr()
    assert manifest_path.read_bytes() == original
    assert (
        main(
            [
                "manifest",
                "validate",
                "--manifest",
                str(manifest_path),
                "--run-dir",
                str(run_dir),
                "--stage",
                "preflight",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["gate_passed"] is False and result["issues"]
    assert sorted(path.name for path in Path(run_dir).iterdir()) == ["manifest.json"]
