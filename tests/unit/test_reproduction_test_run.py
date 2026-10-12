import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.verification.provenance import source_inventory
from coveo_recsys.verification.run_tests import run_tests


@pytest.fixture
def test_root(tmp_path, monkeypatch):
    path = tmp_path / "src/example.py"
    path.parent.mkdir()
    path.write_text("value = 1\n", encoding="utf-8")
    old = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(path, (old, old))
    checked = []
    monkeypatch.setattr("coveo_recsys.verification.run_tests.assert_source_binding", checked.append)
    monkeypatch.setattr(
        "coveo_recsys.verification.run_tests.stage_test_rows", lambda *args, **kwargs: []
    )
    return tmp_path, checked


def junit(path, outcome=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests="1",
        failures=str(int(outcome == "failure")),
        errors="0",
        skipped=str(int(outcome == "skipped")),
        timestamp=datetime.now(UTC).isoformat(),
    )
    case = ET.SubElement(suite, "testcase", classname="tests.unit.test_rule", name="test_rule")
    if outcome:
        ET.SubElement(case, outcome, message="Фактический исход")
    path.write_bytes(ET.tostring(root))


@pytest.mark.parametrize("relative", [False, True])
def test_tests_are_run_once_with_current_source_and_unmodified_skip(
    test_root, monkeypatch, relative
):
    root, checked = test_root
    output = root / "runs/checks"
    before = source_inventory(root)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k omit_normative_test")
    monkeypatch.setenv("REPRODUCTION_TEST_ENV", "retained")

    def execute(command, *, cwd, env):
        assert command == [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not coveo and not load",
            f"--junitxml={output / 'pytest.xml'}",
            "--hypothesis-show-statistics",
        ]
        assert cwd == root
        assert "PYTEST_ADDOPTS" not in env
        assert env["REPRODUCTION_TEST_ENV"] == "retained"
        assert json.loads((output / "source-inventory.json").read_bytes()) == before
        junit(output / "pytest.xml", "skipped")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", execute)
    result = run_tests(root, output.relative_to(root) if relative else output)
    assert checked == [root]
    assert result["status"] == "passed" and result["exit_code"] == 0
    assert result["source_sha256"] == sha256_bytes(canonical_json_bytes(before))
    assert result["junit_sha256"] == sha256_file(output / "pytest.xml")
    assert result["test_evidence"]["counts"]["skipped"] == 1
    assert result["test_evidence"]["counts"]["passed"] == 0
    assert json.loads((output / "result.json").read_bytes()) == result


@pytest.mark.parametrize("fault", ["pytest", "source", "missing_junit", "invalid_junit", "start"])
def test_failure_is_saved_and_cannot_become_success(test_root, monkeypatch, fault):
    root, _ = test_root
    output = root / "runs/checks"

    def execute(command, *, cwd, env):
        if fault == "start":
            raise OSError("pytest unavailable")
        if fault == "source":
            (root / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        if fault == "invalid_junit":
            (output / "pytest.xml").write_bytes(b"not XML")
        elif fault != "missing_junit":
            junit(output / "pytest.xml", "failure" if fault == "pytest" else None)
        return SimpleNamespace(returncode=int(fault == "pytest"))

    monkeypatch.setattr(subprocess, "run", execute)
    result = run_tests(root, output)
    assert result["status"] == "failed" and result["error"]
    assert result["exit_code"] == (None if fault == "start" else int(fault == "pytest"))
    assert json.loads((output / "result.json").read_bytes()) == result
    assert (output / "source-inventory.json").exists()


@pytest.mark.parametrize("failed_stage", [None, "preflight", "post_training"])
def test_normative_coverage_is_checked_before_data_commands(test_root, monkeypatch, failed_stage):
    root, _ = test_root
    output = root / "runs/checks"
    called = []

    def execute(command, *, cwd, env):
        junit(output / "pytest.xml", "skipped" if failed_stage else None)
        return SimpleNamespace(returncode=0)

    def coverage(manifest, stage, evidence, references, *, operation=None):
        assert manifest["scope"] == "actual"
        assert not any(manifest["modules"].values())
        assert not any(
            manifest[f"{name}_extension"]["enabled"] for name in ("neural", "sid", "ann")
        )
        assert references
        called.append((stage, operation))
        if stage == failed_stage:
            assert evidence.summary["counts"]["skipped"] == 1
            raise ValueError(f"Неполное исполнение {stage}")
        return []

    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setattr("coveo_recsys.verification.run_tests.stage_test_rows", coverage)
    result = run_tests(root, output)
    assert result["exit_code"] == 0
    assert result["status"] == ("failed" if failed_stage else "passed")
    assert called == [("preflight", None)] + (
        [] if failed_stage == "preflight" else [("post_training", "evaluate")]
    )
    if failed_stage:
        assert "Неполное исполнение" in result["error"]


@pytest.mark.parametrize(
    "destination",
    [
        ".",
        "src/checks",
        "tests/checks",
        "configs/checks",
        "schemas/checks",
        ".git/checks",
        "../outside",
        "runs/existing",
    ],
)
def test_output_must_be_new_inside_root_and_outside_source_inventory(
    test_root, monkeypatch, destination
):
    root, _ = test_root
    existing = root / "runs/existing"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("pytest must not run"))
    with pytest.raises((ValueError, FileExistsError)):
        run_tests(root, root / destination)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (existing / "result.json").exists()
