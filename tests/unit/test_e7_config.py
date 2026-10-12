"""Два закреплённых рецепта; метаданные не меняют бюджет обучения."""

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, strict_json_loads
from coveo_recsys.config import init_manifest, validate_manifest
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.config.resources import resource_bytes
from coveo_recsys.config.schema import validate_feature_schema
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import BASE_CONFIG_ID, NO_METADATA_CONFIG_ID


def test_metadata_recipe_changes_only_categorical_columns():
    baseline = get_ranker_recipe()
    variant = get_ranker_recipe(NO_METADATA_CONFIG_ID)
    assert baseline["dataset"]["categorical_feature"] == ["category"]
    assert variant["dataset"]["categorical_feature"] == []
    variant["dataset"]["categorical_feature"] = ["category"]
    assert variant == baseline == get_ranker_recipe(BASE_CONFIG_ID)
    assert get_ranker_recipe(NO_METADATA_CONFIG_ID)["dataset"]["categorical_feature"] == []


def test_unknown_recipe_profile_rejected():
    with pytest.raises(InputValidationError):
        get_ranker_recipe("arbitrary")


@pytest.mark.parametrize("change", [None, "category", "formula", "columns"])
def test_no_metadata_schema_has_a_fixed_layout(change):
    schema = strict_json_loads(resource_bytes("configs/feature_schema_no_metadata.json"))
    if change == "category":
        schema["columns"][0]["dtype"] = "category"
    elif change == "formula":
        schema["columns"][0]["formula"] = "other"
    elif change == "columns":
        schema["columns"].pop()
    assert bool(validate_feature_schema(schema)) == (change is not None)


@pytest.mark.parametrize("change", [None, "quotas", "graph", "ranker"])
def test_no_metadata_manifest_keeps_fixed_recipe(tmp_path, change):
    manifest = init_manifest("metadata")
    manifest["base_model"]["config_id"] = NO_METADATA_CONFIG_ID
    for source in ("description", "image", "search"):
        manifest["base_model"]["quotas"][source] = 0
    manifest["ranker"]["resolved_parameters"] = get_ranker_recipe(NO_METADATA_CONFIG_ID)
    if change == "quotas":
        manifest["base_model"]["quotas"]["search"] = 1
    elif change == "graph":
        manifest["base_model"]["graph_min_sessions"] += 1
    elif change == "ranker":
        manifest["ranker"]["num_boost_round"] += 1
    result = validate_manifest(manifest, "preflight", LocalArtifactStore(tmp_path))
    fixed_issues = [i for i in result.issues if i.code == "baseline_changed"]
    assert bool(fixed_issues) == (change is not None)
