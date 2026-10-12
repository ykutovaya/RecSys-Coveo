import hashlib
import json

import pytest

from coveo_recsys.verification.provenance import (
    assert_source_binding,
    source_inventory,
    source_sha256,
)


def test_generated_results_do_not_change_code_hash(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_bytes(b"x = 1\n")
    expected_inventory = {"src/a.py": hashlib.sha256(b"x = 1\n").hexdigest()}
    expected_bytes = (
        json.dumps(
            expected_inventory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert source_inventory(tmp_path) == expected_inventory
    before = source_sha256(tmp_path)
    assert before == hashlib.sha256(expected_bytes).hexdigest()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/report.json").write_text('{"status":"passed"}')
    assert source_sha256(tmp_path) == before
    (tmp_path / "src/a.py").write_bytes(b"x = 2\n")
    assert source_sha256(tmp_path) != before
    assert list(source_inventory(tmp_path)) == ["src/a.py"]


def test_packaging_exclusions_are_part_of_source_hash(tmp_path):
    before = source_sha256(tmp_path)
    (tmp_path / ".gitignore").write_text("/data/\n", encoding="utf-8")
    after = source_sha256(tmp_path)
    assert after != before
    assert ".gitignore" in source_inventory(tmp_path)
    (tmp_path / ".gitignore").write_text("data\n", encoding="utf-8")
    assert source_sha256(tmp_path) != after


def test_other_checkout_cannot_be_reported_as_executed_source(tmp_path):
    package = tmp_path / "src/coveo_recsys"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Другой пакет\n", encoding="utf-8")
    with pytest.raises(ValueError, match="project-root"):
        assert_source_binding(tmp_path)
