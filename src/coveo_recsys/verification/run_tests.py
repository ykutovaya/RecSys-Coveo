"""Один обычный запуск pytest с привязкой JUnit к проверенному исходному коду."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.config import init_manifest
from coveo_recsys.verification.provenance import assert_source_binding, source_inventory
from coveo_recsys.verification.test_evidence import read_test_evidence, stage_test_rows


def run_tests(root: Path, output_dir: Path) -> dict[str, Any]:
    """Сохранить фактический исход; passed здесь не заменяет допуск стадии."""
    root = root.resolve()
    assert_source_binding(root)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if any(part.is_symlink() for part in (output_dir, *output_dir.parents)):
        raise ValueError("Каталог результатов тестов не должен содержать символические ссылки")
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root) or output_dir == root:
        raise ValueError("Нужен отдельный каталог результатов внутри project-root")
    if output_dir.relative_to(root).parts[0] in {"src", "tests", "configs", "schemas", ".git"}:
        raise ValueError("Результаты тестов нельзя сохранять в исходники или .git")
    before = source_inventory(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source-inventory.json").write_bytes(canonical_json_bytes(before))
    junit = output_dir / "pytest.xml"
    result: dict[str, Any] = {
        "status": "failed",
        "exit_code": None,
        "source_sha256": sha256_bytes(canonical_json_bytes(before)),
        "junit_sha256": None,
    }
    try:
        environment = os.environ.copy()
        environment.pop("PYTEST_ADDOPTS", None)
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not coveo and not load",
                f"--junitxml={junit}",
                "--hypothesis-show-statistics",
            ],
            cwd=root,
            env=environment,
        )
        result["exit_code"] = process.returncode
        evidence = read_test_evidence(root, junit, before)
        result["test_evidence"] = evidence.summary
        if process.returncode != 0:
            raise ValueError(f"pytest завершился с кодом {process.returncode}")
        counts = evidence.summary["counts"]
        if counts["failed"] or counts["errors"]:
            raise ValueError("JUnit содержит неуспешные тесты при нулевом коде pytest")
        manifest = init_manifest("reproduction-test-check", scope="actual")
        references = "Проверка состава тестов до обработки CSV; фактический допуск не выдан"
        stage_test_rows(manifest, "preflight", evidence, references)
        stage_test_rows(manifest, "post_training", evidence, references, operation="evaluate")
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            result["exit_code"] = 130
        result["error"] = f"{type(exc).__name__}: {exc}"
    if junit.is_file():
        result["junit_sha256"] = sha256_file(junit)
    (output_dir / "result.json").write_bytes(canonical_json_bytes(result))
    return result
