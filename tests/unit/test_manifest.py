from copy import deepcopy

from jsonschema import Draft202012Validator

from coveo_recsys.config import init_manifest, load_registry, load_schema, structure_issues


def test_template_is_structurally_valid_not_runnable() -> None:
    manifest = init_manifest("example")
    assert not structure_issues(manifest)
    assert manifest["status"] == "draft"
    assert manifest["dataset"]["version"] is None
    assert manifest["verification"]["math_status"] == "not_run"
    Draft202012Validator.check_schema(load_schema())
    Draft202012Validator.check_schema(load_schema("feature"))


def test_schema_rejects_types_enums_unknown_fields_and_nonfinite() -> None:
    manifest = init_manifest("example")
    for path, value in [("stage", "wrong"), ("status", "passed"), ("unknown", 1)]:
        changed = deepcopy(manifest)
        changed[path] = value
        assert structure_issues(changed)
    for key, value in [
        ("property_case_count", True),
        ("math_status", "not_used"),
        ("business_status", "unknown"),
    ]:
        changed = deepcopy(manifest)
        changed["verification"][key] = value
        assert structure_issues(changed)
    manifest["inference"]["alpha"] = float("nan")
    assert structure_issues(manifest)


def test_registry_is_data_only_complete_and_fresh() -> None:
    registry = load_registry()
    assert len(registry["normative_ids"]) == 30
    assert len(set(registry["normative_ids"])) == 30
    assert registry["own_ids"] == ["EXPL01"]
    assert "checks" not in registry
    assert registry["implemented_checks"]
    registry["normative_ids"].clear()
    assert len(load_registry()["normative_ids"]) == 30
