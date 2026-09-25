from pathlib import Path

from coveo_recsys.config import load_registry

ROOT = Path(__file__).resolve().parents[2]


def test_implemented_checks_keep_partial_coverage_and_existing_test_paths():
    registry = load_registry()
    assert set(registry) == {"registry_version", "normative_ids", "own_ids", "implemented_checks"}
    checks = registry["implemented_checks"]
    assert set(checks) == {
        "B01.data",
        "B02.context",
        "B03.data",
        "B05.numerics",
        "B06",
        "B09",
        "B10.periods",
        "B13.preparation",
    }
    for check in checks.values():
        assert check["test_id"] in registry["normative_ids"]
        assert check["paths"] and check["reference_paths"]
        assert all((ROOT / name).is_file() for name in check["paths"] + check["reference_paths"])
    for name in ("B02.context", "B03.data", "B05.numerics", "B10.periods"):
        assert checks[name]["complete_test"] is False
        assert checks[name]["remaining"]
