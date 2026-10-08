from pathlib import Path

from coveo_recsys.config import load_registry

ROOT = Path(__file__).resolve().parents[2]


def test_implemented_checks_keep_partial_coverage_and_existing_test_paths():
    registry = load_registry()
    assert set(registry) == {"registry_version", "normative_ids", "own_ids", "implemented_checks"}
    checks = registry["implemented_checks"]
    assert set(checks) == {
        "B01.data",
        "B01.features",
        "B02.context",
        "B02.features",
        "B03.data",
        "B03.service",
        "B04.graph",
        "B05.content",
        "B05.features",
        "B05.numerics",
        "B05.ranker",
        "B06",
        "B07.retrieval",
        "B07.evaluation",
        "B08.retrieval",
        "B08.evaluation",
        "B09",
        "B10.periods",
        "B10.features",
        "B10.ranker",
        "B10.evaluation",
        "B11",
        "B12",
        "B13.preparation",
        "B13.service",
        "B14.local",
        "B15.local",
        "EXPL01",
    }
    for check in checks.values():
        assert check["test_id"] in registry["normative_ids"] + registry["own_ids"]
        assert check["paths"] and check["reference_paths"]
        assert all((ROOT / name).is_file() for name in check["paths"] + check["reference_paths"])
    for name in (
        "B02.context",
        "B03.data",
        "B06",
        "B09",
        "B10.periods",
        "B01.features",
        "B02.features",
        "B07.retrieval",
        "B08.retrieval",
        "B10.features",
        "B10.ranker",
        "B10.evaluation",
    ):
        assert checks[name]["complete_test"] is False
        assert checks[name]["remaining"]
    for name in ("B05.numerics", "B05.content", "B05.features", "B05.ranker"):
        assert checks[name]["complete_test"] is True
