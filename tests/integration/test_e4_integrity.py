"""Отказы обучения, целостность входов и изоляция будущих периодов."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.cli.training import run_training
from coveo_recsys.contracts import InputValidationError, OutcomeStatus
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import TABLE_SCHEMA, iter_sessions, table_info
from tests.integration.test_e3_pipeline import e3_settings  # noqa: F401
from tests.integration.test_e4_pipeline import ROOT, e4_settings  # noqa: F401
from tests.unit.test_e2_input import prepared_source  # noqa: F401


def _replace_targets(configuration, source, features_root, rows):
    """Меняем только тестовые цели и согласованно обновляем привязки входов."""
    relative = "datasets/B/targets.parquet"
    path = source / relative
    pq.write_table(
        pa.Table.from_pylist(
            [{"session_id": sid, "record_json": to_json(record)} for sid, record in rows],
            schema=TABLE_SCHEMA,
        ),
        path,
    )
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"][relative] = table_info(path, TargetRecord, period="B", role="targets")
    report_path.write_bytes(canonical_json_bytes(report))
    preparation_hash = sha256_file(report_path)
    features_path = features_root / "reports/features.json"
    features = json.loads(features_path.read_bytes())
    features["identity"]["preparation_sha256"] = preparation_hash
    features_path.write_bytes(canonical_json_bytes(features))
    settings = json.loads(configuration.read_bytes())
    settings["preparation"]["report_sha256"] = preparation_hash
    settings["features"]["report_sha256"] = sha256_file(features_path)
    configuration.write_bytes(canonical_json_bytes(settings))


@pytest.mark.parametrize("empty_target", [False, True], ids=["outside-candidates", "unknown"])
def test_no_pairs_is_insufficient_without_publishing_a_model(
    e4_settings,  # noqa: F811
    tmp_path,
    capsys,
    empty_target,
):
    from coveo_recsys.cli import main

    configuration, source, features_root = e4_settings
    rows = [
        (
            sid,
            replace(
                records[0],
                purchases=() if empty_target else ("outside",),
                status=OutcomeStatus.UNKNOWN if empty_target else OutcomeStatus.KNOWN,
                reason="no_registered_purchase" if empty_target else None,
            ),
        )
        for sid, records in iter_sessions(source / "datasets/B/targets.parquet", TargetRecord)
    ]
    _replace_targets(configuration, source, features_root, rows)
    destination = tmp_path / "ranker"
    result = run_training(ROOT, configuration, destination, "no-pairs")
    assert result["status"] == "insufficient" and result["groups"]["pair_count"] == 0
    assert result["groups"]["rows"] == (0 if empty_target else 480)
    assert result["groups"]["unknown_sessions"] == (160 if empty_target else 0)
    assert not (destination / "models/ranker.txt").exists()
    capsys.readouterr()
    exit_code = main(
        [
            "--project-root",
            str(ROOT),
            "train",
            "ranker",
            "--config",
            str(configuration),
            "--run-dir",
            str(destination),
            "--run-id",
            "no-pairs",
        ]
    )
    assert exit_code != 0
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient"


def test_missing_target_record_is_not_silently_replaced_with_unknown(
    e4_settings,  # noqa: F811
    tmp_path,
):
    configuration, source, features_root = e4_settings
    rows = [
        (sid, records[0])
        for sid, records in iter_sessions(source / "datasets/B/targets.parquet", TargetRecord)
    ]
    _replace_targets(configuration, source, features_root, rows[1:])
    destination = tmp_path / "ranker"
    with pytest.raises(InputValidationError, match="цел|ключ|сесс|строк"):
        run_training(ROOT, configuration, destination, "missing-target")
    assert not (destination / "reports/ranker.json").exists()
    assert not (destination / "models/ranker.txt").exists()


@pytest.mark.parametrize(
    ("limit", "value"),
    [("max_requests", 159), ("max_rows", 47999), ("max_memory_bytes", 1), ("max_result_bytes", 1)],
)
def test_resource_limits_refuse_training_without_truncation(
    e4_settings,  # noqa: F811
    tmp_path,
    limit,
    value,
):
    configuration, _, _ = e4_settings
    settings = json.loads(configuration.read_bytes())
    settings["limits"][limit] = value
    configuration.write_bytes(canonical_json_bytes(settings))
    destination = tmp_path / "ranker"
    with pytest.raises(InputValidationError, match="ресурс|бюджет|строк|запрос"):
        run_training(ROOT, configuration, destination, "budget")
    assert not (destination / "reports/ranker.json").exists()
    assert not (destination / "models/ranker.txt").exists()


def test_completed_training_cannot_be_reused_after_input_corruption(
    e4_settings,  # noqa: F811
    tmp_path,
):
    configuration, source, features_root = e4_settings
    settings = json.loads(configuration.read_bytes())
    recipe, schema = tmp_path / "recipe.json", tmp_path / "schema.json"
    recipe.write_bytes((ROOT / settings["recipe"]).read_bytes())
    schema.write_bytes((ROOT / settings["feature_schema"]).read_bytes())
    settings.update(recipe=str(recipe), feature_schema=str(schema))
    configuration.write_bytes(canonical_json_bytes(settings))
    destination = tmp_path / "ranker"
    run_training(ROOT, configuration, destination, "integrity")
    original = {path: sha256_file(path) for path in destination.rglob("*") if path.is_file()}
    for path in (
        source / "datasets/B/targets.parquet",
        features_root / "features/values.parquet",
        features_root / "reports/features.json",
        source / "reports/preparation.json",
        recipe,
        schema,
    ):
        payload = path.read_bytes()
        try:
            path.write_bytes(payload + (b"\n" if path.suffix == ".json" else b"\x00"))
            with pytest.raises((ValueError, OSError)):
                run_training(ROOT, configuration, destination, "integrity")
            assert original == {output: sha256_file(output) for output in original}
        finally:
            path.write_bytes(payload)


def test_changes_outside_b_do_not_change_training_table_model_or_recipe(
    e4_settings,  # noqa: F811
    tmp_path,
):
    configuration, source, _ = e4_settings
    before = run_training(ROOT, configuration, tmp_path / "before", "before")
    for period in ("C_smoke", "C_select", "E", "D"):
        for name in ("prefixes", "targets"):
            path = source / f"datasets/{period}/{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"forbidden future data; not a parquet file")
    after = run_training(ROOT, configuration, tmp_path / "after", "after")
    assert before["model_sha256"] == after["model_sha256"]
    assert before["groups"] == after["groups"]
    assert before["recipe"] == after["recipe"]
    assert all(read["path"].startswith("datasets/B/") for read in after["reads"])
