"""Подготовка C_smoke не открывает строки до связанного post_training."""

from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.cli.preparation import Preparation
from coveo_recsys.contracts import InputValidationError
from tests.integration.test_actual_preparation import actual_inputs  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.parametrize("actual_inputs", [("C_smoke", 3)], indirect=True)


def test_actual_smoke_without_admission_reads_no_raw_rows(actual_inputs, tmp_path, monkeypatch):  # noqa: F811
    source, profile, plan, _ = actual_inputs
    preparation = Preparation(LocalArtifactStore(tmp_path / "run"), ROOT, "c")
    monkeypatch.setattr(preparation, "_input_hashes", lambda *_: pytest.fail("Доступ до допуска"))
    with pytest.raises(InputValidationError):
        preparation.prepare(source, profile, plan, batch_size=2, max_batch_chars=4096)


@pytest.mark.parametrize("revoke", [False, True])
def test_smoke_admission_checked_before_reading_and_publication(
    actual_inputs,  # noqa: F811
    tmp_path,
    monkeypatch,
    revoke,
):
    from coveo_recsys.cli import admission

    source, profile, plan, _ = actual_inputs
    settings = tmp_path / "replay.json"
    settings.write_text("{}", encoding="utf-8")
    destination = tmp_path / "run"
    preparation = Preparation(LocalArtifactStore(destination), ROOT, "c")
    calls = []

    def gate(project_root, settings_path, current_plan, *, profile_sha256):
        assert project_root == ROOT and settings_path == settings
        assert len(profile_sha256) == 64
        assert [p.name for p in current_plan.periods if p.sample_size] == ["C_smoke"]
        calls.append(len(preparation.reads))
        if revoke and len(calls) > 1:
            raise InputValidationError("Допуск изменился")
        return "a" * 64

    monkeypatch.setattr(admission, "require_evaluation_preparation", gate, raising=False)

    def run():
        return preparation.prepare(
            source,
            profile,
            plan,
            evaluation_config=settings,
            batch_size=2,
            max_batch_chars=4096,
        )

    if revoke:
        with pytest.raises(InputValidationError, match="Допуск изменился"):
            run()
        assert not (destination / "reports/preparation.json").exists()
    else:
        result = run()
        assert result["selection"]["C_smoke"]
        assert result["evaluation_admission"]["manifest_sha256"] == "a" * 64
    assert calls[0] == 0 and len(calls) == 2 and calls[1] > 0
