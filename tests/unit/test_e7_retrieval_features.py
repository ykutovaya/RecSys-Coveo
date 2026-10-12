"""Два фиксированных профиля и отсутствие чтения метаданных в Э7."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.historical import (
    RELATIONS,
    GraphBuildResult,
    GraphRelation,
    HistoricalCatalog,
    HistoricalStatistics,
)
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request

BASE = "gcs_rk_v2_7"
VARIANT = "graph_recent_popularity_no_provided_metadata"
QUOTAS = (50, 50, 50, 0, 0, 0, 20, 10)
ALLOWED_SOURCES = ("detail_detail", "detail_add", "detail_purchase", "recent", "popularity")
CONFIG = RetrievalConfig(quotas=QUOTAS)


def expected_columns():
    result = []
    for source in ALLOWED_SOURCES:
        result.extend(
            (
                f"origin_{source}",
                f"score_{source}",
                f"score_{source}_mask",
                f"rank_{source}",
                f"rank_{source}_mask",
            )
        )
    result.append("origin_fill")
    for relation in RELATIONS:
        result.extend(
            (
                f"last_{relation}",
                f"last_{relation}_mask",
                f"support_{relation}",
                f"support_{relation}_mask",
            )
        )
    result.extend(
        (
            "detail_count",
            "detail_age",
            "detail_age_mask",
            "cart_add",
            "cart_add_mask",
            "cart_remove",
            "cart_remove_mask",
            "cart_age",
            "cart_age_mask",
            "session_event_count",
            "session_distinct_items",
            "session_add_count",
            "session_duration",
            "session_duration_mask",
            "historical_purchase_count",
            "cart_unknown",
        )
    )
    return tuple(result)


def schema(config_id=VARIANT):
    document = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))
    document["config_id"] = config_id
    if config_id == VARIANT:
        allowed = set(expected_columns())
        document["columns"] = [c for c in document["columns"] if c["name"] in allowed]
        for i, column in enumerate(document["columns"]):
            column["position"] = i
    return document


class MetadataReadGuard:
    def __init__(self, wrapped, forbidden):
        self._wrapped = wrapped
        self._forbidden = forbidden

    def __getattr__(self, name):
        if name in self._forbidden:
            pytest.fail(f"Вариант без метаданных прочитал {name}")
        return getattr(self._wrapped, name)


class ObservedReadGuard(dict):
    def __bool__(self):
        pytest.fail("Вариант без метаданных обратился к внешним метаданным SKU истории")

    def items(self):
        pytest.fail("Вариант без метаданных прочитал внешние метаданные SKU истории")


def test_metadata_variant_never_reads_excluded_inputs_or_item_rows(monkeypatch):
    historical = MetadataReadGuard(package(), {"content", "metadata_contract", "input_hashes"})
    index = RetrievalIndex(historical, CATALOG_HASH, ObservedReadGuard(), config_id=VARIANT)
    prepared = request(
        ((2000, "detail", "outside"), (3000, "detail", "a")), query=(1.0, 0.0), search_time=1500
    )
    context = bind_context(prepared, index)
    object.__setattr__(context, "prefix", MetadataReadGuard(context.prefix, {"search"}))

    def forbidden_row(*args):
        pytest.fail("Вариант без метаданных запросил ItemContent")

    monkeypatch.setattr(index, "row", forbidden_row)
    scores = score_request(context, index, CONFIG)
    selected = retrieve(context, index, CONFIG, scores=scores)
    actual = FeatureEngine(index, schema(), CONFIG).build(context, selected, scores=scores)
    assert actual.numeric_columns == expected_columns()
    assert actual.categorical_columns == ()
    assert actual.categorical == ((),) * len(actual.skus)
    assert all(len(values) == len(expected_columns()) for values in actual.numeric)
    assert "outside" not in actual.skus
    assert index.config_id == VARIANT
    assert not any("outside" == sku for sku in index.positions)


def test_disabled_sources_are_empty_configuration_choices_not_runtime_failures():
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    context = bind_context(request(), index)
    scores = score_request(context, index, CONFIG)
    selected = retrieve(context, index, CONFIG, scores=scores)
    assert tuple(source.source for source in selected.sources) == SOURCES
    for source in selected.sources[3:6]:
        assert source.quota == 0 and source.skus == source.scores == ()
        assert source.available and source.reason is None
        assert not scores.masks[source.source].any()
        assert not scores.values[source.source].any()
    assert not any(e.source in {"description", "image", "search"} for e in selected.evidence)


@pytest.mark.parametrize("profile", ["unknown", "", "graph_only"])
def test_unknown_profile_is_rejected(profile):
    with pytest.raises(InputValidationError):
        RetrievalIndex(package(), CATALOG_HASH, config_id=profile)


@pytest.mark.parametrize("index_profile,schema_profile", [(BASE, VARIANT), (VARIANT, BASE)])
def test_schema_cannot_cross_profile_boundary(index_profile, schema_profile):
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=index_profile)
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema(schema_profile), CONFIG)


@pytest.mark.parametrize("change", ["missing", "extra", "order", "mask", "category"])
def test_variant_schema_is_exact_not_a_recipe(change):
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    document = schema()
    if change == "missing":
        document["columns"].pop()
    elif change == "extra":
        document["columns"].append(schema(BASE)["columns"][-1])
    elif change == "order":
        document["columns"][0], document["columns"][1] = (
            document["columns"][1],
            document["columns"][0],
        )
    elif change == "mask":
        document["columns"][1]["mask"] = None
    else:
        document["columns"][-1]["dtype"] = "category"
    with pytest.raises(InputValidationError):
        FeatureEngine(index, document, CONFIG)


@pytest.mark.parametrize("config", [RetrievalConfig(), replace(CONFIG, max_candidates=301)])
def test_variant_enforces_fixed_source_quotas_and_candidate_budget(config):
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    with pytest.raises(InputValidationError):
        score_request(bind_context(request(), index), index, config)


@pytest.mark.parametrize("config", [RetrievalConfig(), replace(CONFIG, max_candidates=301)])
def test_variant_constructor_rejects_invalid_config_before_any_scoring(config, monkeypatch):
    from coveo_recsys.features import engine as feature_engine

    def forbidden_scoring(*args, **kwargs):
        pytest.fail("Проверка профиля не должна рассчитывать запрос")

    monkeypatch.setattr(feature_engine, "score_request", forbidden_scoring)
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema(), config)


def test_retrieval_profile_validation_rejects_unknown_profile():
    from coveo_recsys.contracts.retrieval import validate_retrieval_profile

    with pytest.raises(InputValidationError):
        validate_retrieval_profile("unknown", CONFIG)


@pytest.mark.parametrize("source", ["description", "image", "search"])
def test_runtime_failure_cannot_relabel_excluded_source(source):
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    with pytest.raises(InputValidationError):
        score_request(bind_context(request(), index), index, CONFIG, {source: "backend_timeout"})


@pytest.mark.parametrize("purchased", [(), ("a",), ("outside",), ("a", "b", "c", "я")])
def test_variant_preserves_purchase_filter_and_empty_primary(purchased):
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    context = bind_context(
        request(((2000, "detail", "a"),) + tuple((2000, "purchase", sku) for sku in purchased)),
        index,
    )
    selected = retrieve(context, index, CONFIG)
    assert selected.skus == tuple(sku for sku in package().catalog.skus if sku not in purchased)
    batch = FeatureEngine(index, schema(), CONFIG).build(context, selected)
    assert batch.skus == selected.skus


def test_variant_uses_full_aggregates_with_compacted_history():
    prepared = request(
        ((2000, "detail", "b"), (3000, "detail", "b"), (4000, "add", "b"), (5000, "detail", "a"))
    )
    prepared = replace(
        prepared,
        prefix=replace(
            prepared.prefix,
            blocks=prepared.prefix.blocks[-1:],
            detail_history=prepared.prefix.detail_history[-1:],
        ),
    )
    index = RetrievalIndex(package(), CATALOG_HASH, config_id=VARIANT)
    context = bind_context(prepared, index)
    batch = FeatureEngine(index, schema(), CONFIG).build(context, retrieve(context, index, CONFIG))
    row = dict(zip(batch.numeric_columns, batch.numeric[batch.skus.index("b")], strict=True))
    assert row["detail_count"] == math.log1p(2)
    assert row["session_event_count"] == math.log1p(4)
    assert row["session_duration"] == math.log1p(3)


@pytest.mark.parametrize("catalog_size", [20, 299, 300, 301, 360])
def test_fixed_quotas_deduplicate_and_fill_to_common_budget(catalog_size):
    historical = package()
    skus = tuple(f"sku{i:03d}" for i in range(catalog_size))
    graphs = []
    for relation in RELATIONS:
        neighbors = min(60, catalog_size - 1)
        indptr = np.array([0] + [neighbors] * catalog_size, dtype=np.int64)
        row_sums = np.array([float(neighbors)] + [0.0] * (catalog_size - 1))
        graphs.append(
            GraphRelation(
                relation,
                indptr,
                np.arange(1, neighbors + 1, dtype=np.int64),
                np.ones(neighbors),
                np.full(neighbors, 1.0 / neighbors),
                np.full(neighbors, 3, dtype=np.int64),
                row_sums,
                (row_sums > 0).astype(np.float64),
            )
        )
    historical = replace(
        historical,
        catalog=HistoricalCatalog(skus),
        graphs=GraphBuildResult(tuple(graphs), {}),
        statistics=HistoricalStatistics(
            np.ones(catalog_size, dtype=np.int64),
            np.zeros(catalog_size, dtype=np.int64),
            np.full(catalog_size, 1.0 / catalog_size),
            1,
        ),
    )
    index = RetrievalIndex(historical, CATALOG_HASH, config_id=VARIANT)
    context = bind_context(
        request(
            tuple((2000, "detail", sku) for sku in skus[:25]) + ((2000, "purchase", skus[-1]),)
        ),
        index,
    )
    actual = retrieve(context, index, CONFIG)
    expected_size = min(catalog_size - 1, 300)
    assert actual.skus == skus[:expected_size]
    assert len(actual.skus) == len(set(actual.skus))
    sources = {source.source: source for source in actual.sources}
    assert sources["recent"].skus == skus[: min(20, catalog_size - 1)]
    assert sources["popularity"].skus == skus[: min(10, catalog_size - 1)]
    for relation in RELATIONS:
        assert len(sources[relation].skus) == min(50, catalog_size - 2)
    assert all(source.quota == quota for source, quota in zip(actual.sources, QUOTAS, strict=True))
    origins = dict(actual.origins)
    assert origins[skus[1]][:3] == RELATIONS
    if expected_size > 51:
        assert all(origins[sku] == ("fill",) for sku in actual.skus[51:])
