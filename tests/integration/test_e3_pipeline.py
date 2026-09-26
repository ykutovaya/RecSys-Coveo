"""Пакетная команда Э3: публикация, повтор и допуск тестовых префиксов."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.cli.historical import run_historical
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CandidateResult, FeatureBatch
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from tests.reference.e3_cases import request as small_request
from tests.unit.test_e2_input import prepared_source  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def e3_settings(prepared_source, tmp_path):  # noqa: F811
    source = tmp_path / "prepared"
    shutil.copytree(prepared_source, source)
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = from_json(report["plan_json"].encode(), PreparationPlan)
    plan = replace(
        plan, periods=tuple(replace(p, sample_size=1) if p.name == "B" else p for p in plan.periods)
    )
    report["plan_json"] = to_json(plan).decode()
    from coveo_recsys.artifacts import sha256_bytes

    report["inputs"]["plan"] = sha256_bytes(report["plan_json"].encode())
    prepared = small_request(((400001000, "detail", "A"),))
    prefix_path = "datasets/B/prefixes.parquet"
    with TableWriter(source / prefix_path, PreparedRequest) as writer:
        writer.append("s", (prepared,))
    report["tables"][prefix_path] = table_info(
        source / prefix_path, PreparedRequest, period="B", role="prefix"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    historical = tmp_path / "historical"
    hist_settings = tmp_path / "historical.json"
    hist_settings.write_bytes(
        canonical_json_bytes(
            {
                "preparation": {
                    "path": str(prepared_source),
                    "report_sha256": sha256_file(prepared_source / "reports/preparation.json"),
                },
                "metadata_path": str(prepared_source.parent / "input/sku_to_content.csv"),
                "config": {"snapshot_time_ms": 200000000},
            }
        )
    )
    run_historical(ROOT, hist_settings, historical, "e3-history")
    settings = {
        "preparation": {"path": str(source), "report_sha256": sha256_file(report_path)},
        "historical": {
            "path": str(historical),
            "package_sha256": sha256_file(historical / "historical/package.json"),
        },
        "feature_schema": "configs/feature_schema.json",
        "retrieval": {},
        "limits": {
            "max_requests": 10,
            "batch_size": 2,
            "max_observed_skus": 100,
            "max_seconds": 300,
            "max_memory_bytes": 2147483648,
            "max_result_bytes": 67108864,
        },
    }
    path = tmp_path / "features.json"
    path.write_bytes(canonical_json_bytes(settings))
    return path, source, historical


def test_features_command_roundtrip_and_repeat(e3_settings, tmp_path, capsys):
    from coveo_recsys.cli import main
    from coveo_recsys.cli.features import run_features

    settings, _, _ = e3_settings
    destination = tmp_path / "features"
    result = run_features(ROOT, settings, destination, "features", period="B")
    assert result["status"] == "completed" and result["requests"] == 1
    candidates = list(iter_sessions(destination / "features/candidates.parquet", CandidateResult))[
        0
    ][1][0]
    batch = list(iter_sessions(destination / "features/values.parquet", FeatureBatch))[0][1][0]
    assert batch.skus == candidates.skus and batch.skus
    before = {p: sha256_file(p) for p in destination.rglob("*") if p.is_file()}
    assert run_features(ROOT, settings, destination, "features", period="B") == result
    assert before == {p: sha256_file(p) for p in before}
    assert not {"target", "label", "session_weight"} & set(batch.numeric_columns)
    capsys.readouterr()
    assert (
        main(
            [
                "--project-root",
                str(ROOT),
                "features",
                "build",
                "--config",
                str(settings),
                "--run-dir",
                str(destination),
                "--run-id",
                "features",
                "--period",
                "B",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result
    assert all(read["purpose"] == "prefix" for read in result["reads"])


@pytest.mark.parametrize("period", ["A_G", "C_smoke", "C_select", "E", "D"])
def test_feature_command_refuses_other_periods_before_input_read(tmp_path, period):
    from coveo_recsys.cli.features import run_features

    with pytest.raises(ValueError, match="период"):
        run_features(ROOT, tmp_path / "missing.json", tmp_path / "output", "x", period=period)


def test_actual_b_is_refused_before_table_access(e3_settings, monkeypatch, tmp_path):
    from coveo_recsys.cli.features import run_features
    from coveo_recsys.data.access import DatasetAccess

    settings, source, _ = e3_settings
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = replace(from_json(report["plan_json"].encode(), PreparationPlan), scope="actual")
    report.update(scope="actual", plan_json=to_json(plan).decode())
    from coveo_recsys.artifacts import sha256_bytes

    report["inputs"]["plan"] = sha256_bytes(report["plan_json"].encode())
    report_path.write_bytes(canonical_json_bytes(report))
    body = json.loads(settings.read_bytes())
    body["preparation"]["report_sha256"] = sha256_file(report_path)
    settings.write_bytes(canonical_json_bytes(body))

    def forbidden(*args, **kwargs):
        pytest.fail("Прочитана таблица реального B")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    with pytest.raises(ValueError, match="Э1-10"):
        run_features(ROOT, settings, tmp_path / "out", "x", period="B")


def test_interrupted_features_do_not_publish_ready_result(e3_settings, tmp_path):
    from coveo_recsys.cli.features import run_features

    settings, _, _ = e3_settings
    destination = tmp_path / "out"

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_features(ROOT, settings, destination, "x", period="B", before_publish=interrupt)
    assert not (destination / "reports/features.json").exists()
    assert run_features(ROOT, settings, destination, "x", period="B")["status"] == "completed"


def test_repeated_features_check_metadata_used_by_outside_sku(e3_settings, tmp_path):  # noqa: F811
    from coveo_recsys.cli.features import run_features

    settings, source, _ = e3_settings
    prefix_path = source / "datasets/B/prefixes.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    from coveo_recsys.data.prepared import TABLE_SCHEMA

    prepared = small_request(((400001000, "detail", "outside"),))
    pq.write_table(
        pa.Table.from_pylist(
            [{"session_id": "s", "record_json": to_json(prepared)}], schema=TABLE_SCHEMA
        ),
        prefix_path,
    )
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"]["datasets/B/prefixes.parquet"] = table_info(
        prefix_path, PreparedRequest, period="B", role="prefix"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    metadata_path = tmp_path / "metadata.csv"
    history_settings = json.loads((tmp_path / "historical.json").read_bytes())
    shutil.copyfile(history_settings["metadata_path"], metadata_path)
    configuration = json.loads(settings.read_bytes())
    configuration["metadata_path"] = str(metadata_path)
    configuration["preparation"]["report_sha256"] = sha256_file(report_path)
    settings.write_bytes(canonical_json_bytes(configuration))
    destination = tmp_path / "features"
    run_features(ROOT, settings, destination, "features", period="B")
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="метаданных"):
        run_features(ROOT, settings, destination, "features", period="B")
