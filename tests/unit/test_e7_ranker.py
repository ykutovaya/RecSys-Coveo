"""Две закреплённые схемы ранжировщика; без вычислений на данных Coveo."""

import json
from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.artifacts import (
    ArtifactError,
    canonical_json_bytes,
    sha256_bytes,
)
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.config.resources import resource_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.ranking.matrix import feature_matrix
from tests.reference.e4_cases import informative_table, recipe
from tests.unit.test_e4_model import CATALOG_SHA, SCHEMA_SHA, batch, model
from tests.unit.test_e4_table import build, example

BASE = "gcs_rk_v2_7"
NO_METADATA = "graph_recent_popularity_no_provided_metadata"


@pytest.fixture
def fixed_no_metadata_schema():
    return json.loads(resource_bytes("configs/feature_schema_no_metadata.json"))


def without_category(features):
    return replace(
        features,
        categorical_columns=(),
        categorical=((),) * len(features.skus),
        masks=tuple(row[:-1] for row in features.masks),
    )


@pytest.mark.parametrize("skus", [(), ("a", "b", "я")])
def test_numeric_only_matrix_has_no_dummy_category(skus):
    features = without_category(batch(skus))
    actual = feature_matrix(features, features.numeric_columns, (), SCHEMA_SHA)
    np.testing.assert_array_equal(actual, np.asarray(features.numeric).reshape(len(skus), 1))
    assert actual.dtype == np.float64 and actual.flags.c_contiguous


@pytest.mark.parametrize("skus", [(), ("a", "b")])
def test_numeric_only_table_preserves_rows_and_supervision(skus):
    request, candidates, features, target = example(skus=skus)
    original = build([(request, candidates, features, target)])
    numeric_only = without_category(features)
    actual = build([(request, candidates, numeric_only, target)], categorical_columns=())
    np.testing.assert_array_equal(actual.data, original.data[:, :-1])
    np.testing.assert_array_equal(actual.labels, original.labels)
    assert actual.supervision == original.supervision
    assert actual.group_keys == original.group_keys
    assert actual.group_sizes == original.group_sizes
    assert actual.categorical_columns == ()


@pytest.mark.parametrize("columns", [("other",), ("category", "other")])
@pytest.mark.parametrize("consumer", ["matrix", "table"])
def test_arbitrary_categorical_columns_do_not_add_profiles(columns, consumer):
    if consumer == "table":
        with pytest.raises(InputValidationError):
            build([], categorical_columns=columns)
    else:
        features = batch()
        features = replace(
            features,
            categorical_columns=columns,
            categorical=((1,) * len(columns),) * len(features.skus),
            masks=((True,) * (len(features.numeric_columns) + len(columns)),) * len(features.skus),
        )
        with pytest.raises(InputValidationError):
            feature_matrix(features, features.numeric_columns, columns, SCHEMA_SHA)


@pytest.mark.parametrize("config_id", [BASE, NO_METADATA])
def test_each_profile_accepts_only_its_exact_recipe(config_id):
    from coveo_recsys.ranking.train import validate_recipe

    settings = recipe()
    settings["dataset"]["categorical_feature"] = [] if config_id == NO_METADATA else ["category"]
    validate_recipe(settings, config_id=config_id)


@pytest.mark.parametrize("config_id", [BASE, NO_METADATA, "unregistered"])
@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "num_boost_round", 1),
        ("model", "learning_rate", 0.1),
        ("dataset", "categorical_feature", ["other_category"]),
    ],
)
def test_profile_does_not_allow_new_recipes(config_id, section, key, value):
    from coveo_recsys.ranking.train import validate_recipe

    settings = recipe()
    settings["dataset"]["categorical_feature"] = [] if config_id == NO_METADATA else ["category"]
    settings[section][key] = value
    with pytest.raises(InputValidationError):
        validate_recipe(settings, config_id=config_id)


@pytest.mark.parametrize("config_id", [BASE, NO_METADATA])
def test_training_rejects_table_from_other_profile_before_lightgbm(monkeypatch, config_id):
    from coveo_recsys.ranking import train

    table, settings = informative_table(groups=1), recipe()
    if config_id == BASE:
        table = replace(table, data=table.data[:, :-1], categorical_columns=())
    else:
        settings["dataset"]["categorical_feature"] = []
    monkeypatch.setattr(train.lgb, "Dataset", lambda *a, **kw: pytest.fail("Dataset created"))
    with pytest.raises(InputValidationError):
        train.fit_ranker(table, settings, config_id=config_id)


@pytest.mark.parametrize("damage", ["name", "order", "hash", "semantics"])
@pytest.mark.parametrize("consumer", ["train", "load"])
def test_no_metadata_requires_fixed_numeric_columns_and_full_schema(
    monkeypatch, tmp_path, fixed_no_metadata_schema, damage, consumer
):
    from coveo_recsys.ranking import load, train

    schema = fixed_no_metadata_schema
    columns = tuple(column["name"] for column in schema["columns"])
    digest = sha256_bytes(canonical_json_bytes(schema))
    if damage == "name":
        columns = ("price_bucket",) + columns[1:]
    elif damage == "order":
        columns = tuple(reversed(columns))
    elif damage == "hash":
        digest = "a" * 64
    else:
        schema["event_semantics"] = "multiset_v1"
        digest = sha256_bytes(canonical_json_bytes(schema))
    settings = get_ranker_recipe(NO_METADATA)
    if consumer == "train":
        original = informative_table(groups=1)
        numeric = np.zeros((3, len(columns)), dtype=np.float64)
        numeric[:, 0] = original.data[:, 0]
        table = replace(
            original,
            data=numeric,
            numeric_columns=columns,
            categorical_columns=(),
            feature_schema_sha256=digest,
        )
        monkeypatch.setattr(train.lgb, "Dataset", lambda *a, **kw: pytest.fail("Dataset created"))
        with pytest.raises(InputValidationError, match="схем"):
            train.fit_ranker(table, settings, config_id=NO_METADATA)
    else:
        payload = b"synthetic model; schema must be rejected before LightGBM"
        summary = {
            "format_version": "ranker_v1",
            "status": "completed",
            "library_version": "4.6.0",
            "config_id": NO_METADATA,
            "recipe": settings,
            "identity": {"catalog_sha256": CATALOG_SHA, "feature_schema_sha256": digest},
            "feature_schema_sha256": digest,
            "numeric_columns": list(columns),
            "categorical_columns": [],
            "actual_tree_count": 1,
            "model_sha256": sha256_bytes(payload),
        }
        (tmp_path / "reports").mkdir()
        (tmp_path / "models").mkdir()
        (tmp_path / "reports/ranker.json").write_bytes(canonical_json_bytes(summary))
        (tmp_path / "models/ranker.txt").write_bytes(payload)
        monkeypatch.setattr(load.lgb, "Booster", lambda *a, **kw: pytest.fail("Booster created"))
        with pytest.raises(ArtifactError, match="схем"):
            load.load_ranker(tmp_path)


@pytest.mark.parametrize("column_container", [list, tuple])
def test_numeric_only_prediction_preserves_raw_scores_and_order(
    fixed_no_metadata_schema, column_container
):
    from coveo_recsys.ranking.predict import predict

    columns = tuple(column["name"] for column in fixed_no_metadata_schema["columns"])
    digest = sha256_bytes(canonical_json_bytes(fixed_no_metadata_schema))
    features = replace(
        without_category(batch()),
        numeric_columns=columns,
        numeric=tuple((float(i),) + (0.0,) * (len(columns) - 1) for i in range(3)),
        masks=((True,) * len(columns),) * 3,
        feature_schema_sha256=digest,
    )
    loaded = model([0.0, 1.0, 1.0])
    loaded.summary["identity"]["feature_schema_sha256"] = digest
    loaded.summary.update(
        config_id=NO_METADATA,
        numeric_columns=list(columns),
        categorical_columns=column_container(),
        feature_schema_sha256=digest,
    )
    actual = predict(loaded, features, catalog_sha256=CATALOG_SHA)
    assert actual == (("b", 1.0), ("я", 1.0), ("a", 0.0))
    matrix, _ = loaded.booster.calls[0]
    assert matrix.shape == (3, 54)


@pytest.mark.parametrize("config_id", [None, BASE, NO_METADATA, "unregistered"])
def test_prediction_rejects_profile_schema_mismatch_before_booster(config_id):
    from coveo_recsys.ranking.predict import predict

    loaded, features = model([0.0, 1.0, 1.0]), batch()
    if config_id is not None:
        loaded.summary["config_id"] = config_id
    if config_id != NO_METADATA:
        loaded.summary["categorical_columns"] = []
        features = without_category(features)
    with pytest.raises(InputValidationError):
        predict(loaded, features, catalog_sha256=CATALOG_SHA)
    assert loaded.booster.calls == []
