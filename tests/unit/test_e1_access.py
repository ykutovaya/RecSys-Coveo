import sys
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.access import DatasetAccess


@pytest.fixture
def prepared_access(tmp_path, monkeypatch):
    """Проверяем маршрут отдельно от чтения и проверки общих таблиц."""
    plan = from_json(Path("configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    report = {
        "format_version": "preparation_tables_v1",
        "status": "completed",
        "source_sha256": "a" * 64,
        "run_id": "sample",
        "scope": "synthetic",
        "evaluation_protocol": "observed_purchases_v1",
        "telemetry_complete": None,
        "plan_json": to_json(plan).decode("utf-8"),
        "tables": {},
    }
    for period in ("A_G", "B", "C_smoke"):
        for filename, kind, role in (
            ("prefixes", "PreparedRequest", "prefix"),
            ("targets", "TargetRecord", "targets"),
        ):
            report["tables"][f"datasets/{period}/{filename}.parquet"] = {
                "sha256": "b" * 64,
                "bytes": 100,
                "rows": 3,
                "record_type": kind,
                "period": period,
                "data_role": role,
                "admitted": True,
            }
    checked, opened, audit = [], [], []

    def verify(root, value, paths=None):
        assert root == tmp_path and value is report
        checked.extend(paths)

    def sessions(path, kind, **kwargs):
        opened.append((path, kind))
        yield "s1", ("r1", "r2")
        yield "s2", ("r3",)

    monkeypatch.setattr(
        "coveo_recsys.data.access.read_report", lambda *a, **k: report, raising=False
    )
    monkeypatch.setattr("coveo_recsys.data.access.verify_tables", verify, raising=False)
    monkeypatch.setattr("coveo_recsys.data.access.iter_sessions", sessions, raising=False)
    return tmp_path, report, checked, opened, audit


@pytest.mark.parametrize("period", ["A_G", "B", "C_smoke"])
@pytest.mark.parametrize(
    "filename,purpose,kind",
    [("prefixes", "prefix", PreparedRequest), ("targets", "labels", TargetRecord)],
)
def test_access_reads_whole_sessions_without_file_journal(
    prepared_access, period, filename, purpose, kind
):
    root, _, checked, opened, audit = prepared_access
    path = f"datasets/{period}/{filename}.parquet"
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample", audit=audit.append)
    result = access.read(path, kind, purpose=purpose)
    assert checked == [path] and opened == []
    assert list(result) == [("s1", ("r1", "r2")), ("s2", ("r3",))]
    assert checked == [path, path]
    assert opened == [(root / path, kind)]
    assert len(audit) == 1 and audit[0]["allowed"] is True
    assert not list(root.iterdir())


@pytest.mark.parametrize(
    "path,purpose,kind",
    [
        ("datasets/B/prefixes.parquet", "labels", TargetRecord),
        ("datasets/B/targets.parquet", "prefix", PreparedRequest),
        ("datasets/B/prefixes.parquet", "prefix", TargetRecord),
        *[
            (f"datasets/{period}/targets.parquet", "labels", TargetRecord)
            for period in ("D", "E", "A_N", "C_select")
        ],
        ("datasets/B/../D/targets.parquet", "labels", TargetRecord),
        ("datasets//B/prefixes.parquet", "prefix", PreparedRequest),
        ("datasets\\B\\prefixes.parquet", "prefix", PreparedRequest),
        ("/datasets/B/prefixes.parquet", "prefix", PreparedRequest),
        ("C:/datasets/B/prefixes.parquet", "prefix", PreparedRequest),
        ("datasets/B/prefixes/s.parquet", "prefix", PreparedRequest),
    ],
)
def test_access_rejects_closed_or_misdirected_paths(prepared_access, path, purpose, kind):
    root, _, checked, opened, audit = prepared_access
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample", audit=audit.append)
    with pytest.raises(InputValidationError):
        access.read(path, kind, purpose=purpose)
    assert not checked and not opened
    assert len(audit) == 1 and audit[0]["allowed"] is False


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("report", "format_version", "old"),
        ("report", "status", "materialized"),
        ("report", "source_sha256", "c" * 64),
        ("report", "run_id", "other"),
        ("report", "scope", "actual"),
        ("report", "evaluation_protocol", "old"),
        ("report", "telemetry_complete", True),
        ("table", "period", "D"),
        ("table", "record_type", "TargetRecord"),
        ("table", "data_role", "targets"),
        ("table", "admitted", False),
    ],
)
def test_access_binds_completed_table_to_its_route(prepared_access, section, key, value):
    root, report, checked, opened, audit = prepared_access
    path = "datasets/B/prefixes.parquet"
    target = report if section == "report" else report["tables"][path]
    target[key] = value
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample", audit=audit.append)
    with pytest.raises(InputValidationError):
        access.read(path, PreparedRequest, purpose="prefix")
    assert not checked and not opened
    assert audit[0]["allowed"] is False
    assert audit[0]["reason"] == "InputValidationError"


def test_diagnostic_cannot_be_opened_under_dataset_path(prepared_access):
    root, report, checked, opened, _ = prepared_access
    plan = from_json(report["plan_json"].encode("utf-8"), PreparationPlan)
    report["plan_json"] = to_json(replace(plan, diagnostic_sessions=1)).decode("utf-8")
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample")
    with pytest.raises(InputValidationError):
        access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix")
    assert not checked and not opened


def test_access_does_not_reuse_a_previous_report(prepared_access):
    root, report, _, _, _ = prepared_access
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample")
    path = "datasets/B/prefixes.parquet"
    assert list(access.read(path, PreparedRequest, purpose="prefix"))
    report["status"] = "materialized"
    with pytest.raises(InputValidationError):
        access.read(path, PreparedRequest, purpose="prefix")


def test_failed_table_verification_is_not_granted(prepared_access, monkeypatch):
    root, _, _, opened, audit = prepared_access

    def corrupted(*args, **kwargs):
        raise InputValidationError("Контрольная сумма таблицы не совпала")

    monkeypatch.setattr("coveo_recsys.data.access.verify_tables", corrupted)
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample", audit=audit.append)
    with pytest.raises(InputValidationError):
        access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix")
    assert not opened
    assert audit[0]["allowed"] is False


def test_changed_table_during_read_is_not_granted(prepared_access, monkeypatch):
    root, _, _, _, audit = prepared_access
    checks = 0

    def changed_after_read(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise InputValidationError("Таблица изменилась при чтении")

    monkeypatch.setattr("coveo_recsys.data.access.verify_tables", changed_after_read)
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample", audit=audit.append)
    with pytest.raises(InputValidationError):
        list(access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix"))
    assert audit[0]["allowed"] is False


def test_access_rejects_link_outside_root(prepared_access, tmp_path):
    root, _, checked, opened, _ = prepared_access
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (root / "datasets").mkdir()
    try:
        (root / "datasets/B").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("windows_symlink_unavailable: WinError 1314")
        raise
    access = DatasetAccess(root, code_sha256="a" * 64, run_id="sample")
    with pytest.raises(InputValidationError):
        access.read("datasets/B/prefixes.parquet", PreparedRequest, purpose="prefix")
    assert not checked and not opened
