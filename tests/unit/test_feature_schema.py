from copy import deepcopy

from coveo_recsys.config import validate_feature_schema


def feature_schema_example() -> dict:
    return {
        "schema_version": "1",
        "config_id": "synthetic",
        "event_semantics": "set_v1",
        "columns": [
            {
                "name": "count",
                "position": 0,
                "dtype": "float64",
                "formula": "synthetic_identity",
                "mask": None,
                "transformation": "identity",
                "fit_period": "none",
                "provenance": {
                    "entity": "request",
                    "sources": ["prefix"],
                    "available_at": "a_q",
                    "version": "1",
                    "missing_rule": "zero",
                },
            }
        ],
    }


def test_feature_contract() -> None:
    example = feature_schema_example()
    assert not validate_feature_schema(example)
    for key, value in [
        ("position", 1),
        ("name", "target"),
        ("dtype", "string"),
        ("mask", "missing"),
        ("fit_period", "D"),
    ]:
        changed = deepcopy(example)
        changed["columns"][0][key] = value
        assert validate_feature_schema(changed)
    duplicated = deepcopy(example)
    duplicated["columns"].append(deepcopy(duplicated["columns"][0]))
    assert validate_feature_schema(duplicated)
