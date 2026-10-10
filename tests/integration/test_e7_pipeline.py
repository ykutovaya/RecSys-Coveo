"""Команда B10.metadata на синтетических таблицах; реальные запуски запрещены."""

import json
from dataclasses import replace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import FeatureBatch
from coveo_recsys.contracts.retrieval import NO_METADATA_CONFIG_ID
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.prepared import iter_sessions
from tests.integration.test_e3_pipeline import ROOT, e3_settings  # noqa: F401
from tests.integration.test_e4_pipeline import e4_settings  # noqa: F401
from tests.reference.e3_cases import request
from tests.unit.test_e2_input import prepared_source, rewrite_records  # noqa: F401


def _command(step, settings, destination):
    from coveo_recsys.cli import main

    return main(
        [
            "--project-root",
            str(ROOT),
            "sensitivity",
            "metadata",
            "--step",
            step,
            "--config",
            str(settings),
            "--run-dir",
            str(destination),
            "--run-id",
            "e7",
        ]
    )


@pytest.fixture
def no_metadata_settings(e4_settings, tmp_path):  # noqa: F811
    path, source, _ = e4_settings
    training = json.loads(path.read_bytes())
    training["feature_schema"] = "configs/feature_schema_no_metadata.json"
    training["recipe"] = "configs/ranker_recipe_no_metadata.json"
    feature_root = tmp_path / "no-metadata-features"
    training["features"] = {"path": str(feature_root), "report_sha256": "0" * 64}
    features = {
        "preparation": training["preparation"],
        "historical": training["historical"],
        "feature_schema": training["feature_schema"],
        "retrieval": {"quotas": [50, 50, 50, 0, 0, 0, 20, 10]},
        "limits": {
            "max_requests": 160,
            "batch_size": 2,
            "max_observed_skus": 100,
            "max_seconds": 300,
            "max_memory_bytes": 2147483648,
            "max_result_bytes": 67108864,
        },
    }
    features_path, training_path = tmp_path / "e7-features.json", tmp_path / "e7-training.json"
    features_path.write_bytes(canonical_json_bytes(features))
    training_path.write_bytes(canonical_json_bytes(training))
    return features_path, training_path, source, feature_root


def _refresh_preparation(source, *settings_paths):
    digest = sha256_file(source / "reports/preparation.json")
    for path in settings_paths:
        settings = json.loads(path.read_bytes())
        settings["preparation"]["report_sha256"] = digest
        path.write_bytes(canonical_json_bytes(settings))


def _bind_features(training_path, feature_root):
    settings = json.loads(training_path.read_bytes())
    settings["features"]["report_sha256"] = sha256_file(feature_root / "reports/features.json")
    training_path.write_bytes(canonical_json_bytes(settings))


def test_metadata_cli_trains_own_b_with_external_sku_and_without_metadata_source(
    no_metadata_settings, tmp_path, capsys
):
    from coveo_recsys.ranking.load import load_ranker
    from coveo_recsys.ranking.predict import predict

    features_path, training_path, source, feature_root = no_metadata_settings

    def add_external_anchor(rows):
        first = rows[0]
        prefix = request(((400000999, "detail", "outside"), (400001000, "detail", "A"))).prefix
        blocks = tuple(
            replace(
                block,
                session_id=first.session_id,
                events=tuple(replace(event, session_id=first.session_id) for event in block.events),
            )
            for block in prefix.blocks
        )
        rows[0] = replace(first, prefix=replace(prefix, session_id=first.session_id, blocks=blocks))
        return rows

    rewrite_records(source, "datasets/B/prefixes.parquet", PreparedRequest, add_external_anchor)
    _refresh_preparation(source, features_path, training_path)
    assert "metadata_path" not in json.loads(features_path.read_bytes())
    capsys.readouterr()
    assert _command("features", features_path, feature_root) == 0
    features = json.loads(capsys.readouterr().out)
    assert features["status"] == "completed" and features["requests"] == 160
    assert features["identity"]["config_id"] == NO_METADATA_CONFIG_ID
    assert all(read["purpose"] == "prefix" for read in features["reads"])
    batch = next(iter_sessions(feature_root / "features/values.parquet", FeatureBatch))[1][0]
    assert batch.skus == ("A", "B", "C") and batch.categorical_columns == ()
    _bind_features(training_path, feature_root)
    model_root = tmp_path / "no-metadata-ranker"
    assert _command("train", training_path, model_root) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed" and result["groups"]["pair_count"] == 320
    assert result["config_id"] == NO_METADATA_CONFIG_ID
    assert result["identity"]["config_id"] == NO_METADATA_CONFIG_ID
    assert result["categorical_columns"] == []
    model = load_ranker(model_root)
    ranked = predict(model, batch, catalog_sha256=result["identity"]["catalog_sha256"])
    assert ranked[0][0] == "C" and ranked[0][1] > ranked[-1][1]
    assert _command("features", features_path, feature_root) == 0
    assert json.loads(capsys.readouterr().out) == features
    assert _command("train", training_path, model_root) == 0
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("step", ["features", "train"])
def test_metadata_cli_refuses_base_schema_before_reading_b(
    e4_settings,  # noqa: F811
    tmp_path,
    monkeypatch,
    capsys,
    step,
):
    from coveo_recsys.data.access import DatasetAccess

    training_path, _, _ = e4_settings
    settings_path = tmp_path / "features.json" if step == "features" else training_path

    def forbidden(*args, **kwargs):
        pytest.fail("Чужой профиль не должен читать B")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    capsys.readouterr()
    destination = tmp_path / "foreign-schema"
    assert _command(step, settings_path, destination) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert "профил" in failure["message"].lower() or "схем" in failure["message"].lower()
    assert not (destination / "reports").exists()


def test_metadata_cli_cannot_open_actual_b_without_admission(
    e3_settings,  # noqa: F811
    tmp_path,
    monkeypatch,
    capsys,
):
    from coveo_recsys.data.access import DatasetAccess

    settings_path, source, _ = e3_settings
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = replace(from_json(report["plan_json"].encode(), PreparationPlan), scope="actual")
    report.update(scope="actual", plan_json=to_json(plan).decode())
    report["inputs"]["plan"] = sha256_bytes(report["plan_json"].encode())
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(settings_path.read_bytes())
    settings["feature_schema"] = "configs/feature_schema_no_metadata.json"
    settings["retrieval"]["quotas"] = [50, 50, 50, 0, 0, 0, 20, 10]
    settings_path.write_bytes(canonical_json_bytes(settings))
    _refresh_preparation(source, settings_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Недопущенный actual не должен читать таблицы B")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    capsys.readouterr()
    destination = tmp_path / "no-admission"
    assert _command("features", settings_path, destination) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed" and "Э1-10" in failure["message"]
    assert not (destination / "reports").exists()


@pytest.mark.parametrize("config_id", ["gcs_rk_v2_7", "unregistered-profile"])
def test_training_refuses_rehashed_feature_report_with_foreign_profile(
    no_metadata_settings, tmp_path, monkeypatch, capsys, config_id
):
    from coveo_recsys.data.access import DatasetAccess

    features_path, training_path, _, feature_root = no_metadata_settings
    assert _command("features", features_path, feature_root) == 0
    report_path = feature_root / "reports/features.json"
    report = json.loads(report_path.read_bytes())
    report["identity"]["config_id"] = config_id
    report_path.write_bytes(canonical_json_bytes(report))
    _bind_features(training_path, feature_root)

    def forbidden(*args, **kwargs):
        pytest.fail("Подменённый профиль отчёта не должен читать B")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    capsys.readouterr()
    destination = tmp_path / "foreign-feature-profile"
    assert _command("train", training_path, destination) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert "несовместим" in failure["message"].lower() or "профил" in failure["message"].lower()
    assert not (destination / "reports").exists()


def test_metadata_cli_refuses_metadata_path_before_reading_b(
    no_metadata_settings, tmp_path, monkeypatch, capsys
):
    from coveo_recsys.data.access import DatasetAccess

    features_path, _, _, feature_root = no_metadata_settings
    settings = json.loads(features_path.read_bytes())
    settings["metadata_path"] = str(tmp_path / "must-not-be-opened.csv")
    features_path.write_bytes(canonical_json_bytes(settings))

    def forbidden(*args, **kwargs):
        pytest.fail("metadata_path запрещён до чтения B")

    monkeypatch.setattr(DatasetAccess, "read", forbidden)
    capsys.readouterr()
    assert _command("features", features_path, feature_root) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed" and "metadata_path" in failure["message"]
    assert not (feature_root / "reports").exists()
