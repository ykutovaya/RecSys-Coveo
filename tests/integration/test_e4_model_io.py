"""Чтение штатной модели LightGBM и отказ на незавершённом/повреждённом наборе."""

import copy

import lightgbm as lgb
import numpy as np
import pytest

from coveo_recsys.artifacts import ArtifactError, canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts.records import FeatureBatch


@pytest.fixture(scope="module")
def model_payload():
    matrix = np.tile([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], (10, 1))
    dataset = lgb.Dataset(
        matrix,
        label=np.tile([0.0, 0.0, 1.0], 10),
        feature_name=["x", "category"],
        categorical_feature=["category"],
    )
    booster = lgb.train(
        {
            "objective": "regression",
            "num_threads": 1,
            "learning_rate": 0.05,
            "min_data_in_leaf": 1,
            "num_leaves": 3,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=2,
    )
    payload = booster.model_to_string().encode("utf-8")
    summary = {
        "format_version": "ranker_v1",
        "status": "completed",
        "identity": {
            "catalog_sha256": "b" * 64,
            "code_sha256": "c" * 64,
            "feature_schema_sha256": "a" * 64,
        },
        "model_sha256": sha256_bytes(payload),
        "feature_schema_sha256": "a" * 64,
        "numeric_columns": ["x"],
        "categorical_columns": ["category"],
        "library_version": "4.6.0",
        "actual_tree_count": booster.num_trees(),
    }
    return payload, summary, booster


def write_model(root, payload, summary):
    (root / "models").mkdir()
    (root / "reports").mkdir()
    (root / "models/ranker.txt").write_bytes(payload)
    (root / "reports/ranker.json").write_bytes(canonical_json_bytes(summary))


def test_native_round_trip_preserves_scores_and_accepts_declared_old_producer(
    tmp_path, model_payload
):
    from coveo_recsys.ranking.load import load_ranker
    from coveo_recsys.ranking.predict import predict

    payload, summary, original = model_payload
    write_model(tmp_path, payload, summary)
    loaded = load_ranker(tmp_path, expected_identity=summary["identity"])
    features = FeatureBatch(
        "q",
        ("a", "b", "c"),
        ("x",),
        ("category",),
        ((0.0,), (1.0,), (2.0,)),
        ((1,), (2,), (3,)),
        ((True, True),) * 3,
        "a" * 64,
    )
    expected = original.predict(
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        raw_score=True,
        num_iteration=summary["actual_tree_count"],
        num_threads=1,
    )
    result = dict(predict(loaded, features, catalog_sha256="b" * 64))
    np.testing.assert_allclose(
        [result[sku] for sku in features.skus], expected, atol=1e-10, rtol=1e-8
    )
    assert loaded.summary == summary


@pytest.mark.parametrize(
    "damage",
    [
        "missing_summary",
        "missing_model",
        "status",
        "format",
        "version",
        "schema",
        "schema_binding",
        "catalog",
        "column_order",
        "tree_count",
        "tree_bool",
        "checksum",
        "json_duplicate",
        "model_syntax",
        "identity",
    ],
)
def test_incomplete_corrupt_or_incompatible_models_are_rejected(tmp_path, model_payload, damage):
    from coveo_recsys.ranking.load import load_ranker

    payload, original_summary, _ = model_payload
    summary = copy.deepcopy(original_summary)
    expected_identity = summary["identity"].copy()
    changes = {
        "status": ("status", "insufficient"),
        "format": ("format_version", "future"),
        "version": ("library_version", "4.0.0"),
        "schema": ("feature_schema_sha256", "bad"),
        "schema_binding": ("feature_schema_sha256", "d" * 64),
        "column_order": ("numeric_columns", ["other"]),
        "tree_count": ("actual_tree_count", 100),
        "tree_bool": ("actual_tree_count", True),
    }
    if damage in changes:
        key, value = changes[damage]
        summary[key] = value
    elif damage == "catalog":
        summary["identity"]["catalog_sha256"] = "bad"
    elif damage == "checksum":
        payload += b"changed"
    elif damage == "model_syntax":
        payload = b"not a model\n"
        summary["model_sha256"] = sha256_bytes(payload)
    elif damage == "identity":
        expected_identity["code_sha256"] = "d" * 64
    write_model(tmp_path, payload, summary)
    if damage == "missing_summary":
        (tmp_path / "reports/ranker.json").unlink()
    elif damage == "missing_model":
        (tmp_path / "models/ranker.txt").unlink()
    elif damage == "json_duplicate":
        (tmp_path / "reports/ranker.json").write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_ranker(tmp_path, expected_identity=expected_identity)


def test_load_of_absent_root_does_not_create_directory(tmp_path):
    from coveo_recsys.ranking.load import load_ranker

    absent = tmp_path / "absent"
    with pytest.raises(ArtifactError):
        load_ranker(absent)
    assert not absent.exists()


@pytest.mark.parametrize("damage", ["profile", "unknown_profile", "identity_profile", "recipe"])
def test_model_profile_and_recipe_cannot_contradict_schema(tmp_path, model_payload, damage):
    from coveo_recsys.ranking.load import load_ranker
    from tests.reference.e4_cases import recipe

    payload, original, _ = model_payload
    summary = copy.deepcopy(original)
    if damage == "profile":
        summary["config_id"] = "graph_recent_popularity_no_provided_metadata"
    elif damage == "unknown_profile":
        summary["config_id"] = "unregistered"
    elif damage == "identity_profile":
        summary["identity"]["config_id"] = "graph_recent_popularity_no_provided_metadata"
    else:
        summary["recipe"] = recipe()
        summary["recipe"]["training"]["num_boost_round"] = 1
    write_model(tmp_path, payload, summary)
    with pytest.raises(ArtifactError):
        load_ranker(tmp_path)
