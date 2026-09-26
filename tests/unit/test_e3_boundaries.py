"""Границы знания и зависимостей Э3; без чтения реальных периодов."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.historical import MetadataRecord
from coveo_recsys.contracts.retrieval import RetrievalConfig
from tests.reference.e3_cases import CATALOG_HASH, package, request
from tests.reference.e3_retrieval_oracle import retrieval_oracle


@pytest.mark.parametrize("snapshot,allowed", [(1999, True), (2000, False), (2001, False)])
def test_snapshot_must_strictly_precede_event_time_not_availability(snapshot, allowed):
    from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context

    index = RetrievalIndex(replace(package(), snapshot_time_ms=snapshot), CATALOG_HASH)
    prepared = request(available_time=3000)
    if allowed:
        assert bind_context(prepared, index).t_q == 2000
    else:
        with pytest.raises(InputValidationError):
            bind_context(prepared, index)


def test_already_bound_context_cannot_silently_change_catalog():
    from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request

    index = RetrievalIndex(package(), CATALOG_HASH)
    context = replace(bind_context(request(), index), catalog_sha256="b" * 64)
    with pytest.raises(InputValidationError):
        score_request(context, index)


def test_historical_package_with_unknown_version_is_rejected():
    from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context

    with pytest.raises(InputValidationError):
        index = RetrievalIndex(replace(package(), schema_version="historical_future"), CATALOG_HASH)
        bind_context(request(), index)


def test_observed_unknown_item_changes_history_but_not_historical_catalog_or_dictionary():
    from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request

    historical = package()
    metadata = {"outside": MetadataRecord("outside", [0.0, 2.0], None, "new/path", 4)}
    index = RetrievalIndex(historical, CATALOG_HASH, metadata)
    context = bind_context(request(((2000, "detail", "outside"),)), index)
    config = RetrievalConfig()
    actual = score_request(context, index, config)
    expected = retrieval_oracle(context, historical, config, observed=metadata)
    assert historical.catalog.skus == ("a", "b", "c", "я")
    assert historical.content.categories == ("root/a", "root/b")
    assert "outside" not in index.positions
    assert actual.history_available["description"]
    assert not actual.history_available["image"]
    for source in ("description", "image", "detail_detail"):
        assert actual.masks[source].tolist() == [
            expected["masks"][source][sku] for sku in historical.catalog.skus
        ]
        np.testing.assert_allclose(
            actual.values[source],
            [expected["values"][source][sku] for sku in historical.catalog.skus],
            rtol=1e-8,
            atol=1e-10,
        )


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_internal_package_is_an_error_not_missing_modality(coordinate):
    from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request

    historical = package()
    historical.content.description.vectors[0, 0] = coordinate
    with pytest.raises((InputValidationError, NumericError)):
        index = RetrievalIndex(historical, CATALOG_HASH)
        score_request(bind_context(request(), index), index)


def test_retrieval_and_feature_kernel_do_not_import_future_targets_or_io():
    root = Path(__file__).resolve().parents[2] / "src/coveo_recsys"
    paths = [*(root / "retrieval").glob("*.py"), root / "features/engine.py"]
    assert paths and (root / "features/engine.py").exists()
    forbidden = {
        "targets",
        "training",
        "data",
        "cli",
        "artifacts",
        "verification",
        "time",
        "datetime",
        "subprocess",
        "requests",
        "httpx",
        "socket",
        "lightgbm",
        "torch",
        "sqlite3",
    }
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                names.extend((node.module or "") + "." + entry.name for entry in node.names)
            for name in names:
                assert not (set(name.split(".")) & forbidden), (path, name)
