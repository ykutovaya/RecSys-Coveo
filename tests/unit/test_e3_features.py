"""Признаки Э3: семейства формул, доступность и согласованность входа."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.records import Branch, CandidateResult, RequestContext, SourceResult
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval.scores import RetrievalIndex, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


def schema():
    return json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))


def bind(prepared):
    return RequestContext(
        prepared.request_id,
        prepared.session_id,
        prepared.trigger_block_id,
        prepared.t_q,
        prepared.a_q,
        prepared.ordinal,
        prepared.prefix,
        CATALOG_HASH,
    )


def candidates(index, context, skus=None):
    skus = index.package.catalog.skus if skus is None else skus
    scores = score_request(context, index)
    sources = tuple(
        SourceResult(
            source,
            skus if source == "popularity" else (),
            tuple(float(index.package.statistics.popularity[index.positions[sku]]) for sku in skus)
            if source == "popularity"
            else (),
            scores.reasons[source] is None,
            quota,
            scores.reasons[source],
        )
        for source, quota in zip(SOURCES, RetrievalConfig().quotas, strict=True)
    )
    return CandidateResult(
        context.request_id,
        sources,
        skus,
        tuple((sku, ("popularity",)) for sku in skus),
        Branch.PRIMARY,
        (),
    )


def build(prepared=None, *, skus=None, package_value=None, config=RetrievalConfig()):
    prepared = request() if prepared is None else prepared
    context = bind(prepared)
    index = RetrievalIndex(package() if package_value is None else package_value, CATALOG_HASH)
    result = FeatureEngine(index, schema(), config).build(context, candidates(index, context, skus))
    return result, index, context


def row(batch, sku):
    i = batch.skus.index(sku)
    names = batch.numeric_columns + batch.categorical_columns
    values = batch.numeric[i] + batch.categorical[i]
    return dict(zip(names, zip(values, batch.masks[i], strict=True), strict=True))


def test_schema_layout_hash_and_zero_rows_are_stable():
    actual, index, context = build()
    document = schema()
    assert actual.numeric_columns + actual.categorical_columns == tuple(
        c["name"] for c in document["columns"]
    )
    assert len(actual.numeric_columns) == 92 and actual.categorical_columns == ("category",)
    canonical = (
        json.dumps(
            document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    assert actual.feature_schema_sha256 == hashlib.sha256(canonical).hexdigest()
    assert all(math.isfinite(value) for values in actual.numeric for value in values)
    empty = FeatureEngine(index, document).build(context, candidates(index, context, ()))
    assert empty.skus == empty.numeric == empty.categorical == empty.masks == ()
    assert empty.numeric_columns == actual.numeric_columns


def test_measurable_source_score_exists_outside_source_quota_but_rank_does_not():
    actual, index, context = build()
    values = row(actual, "b")
    assert values["origin_detail_detail"] == (0.0, True)
    assert values["rank_detail_detail"] == (0.0, False)
    assert values["rank_detail_detail_mask"] == (0.0, True)
    assert values["score_detail_detail"] == pytest.approx((0.25, True))
    assert values["rank_popularity"] == pytest.approx((1 / 3, True))
    scores = score_request(context, index, RetrievalConfig())
    reused = FeatureEngine(index, schema()).build(
        context, candidates(index, context), scores=scores, check_deadline=lambda: None
    )
    assert reused == actual


@pytest.mark.parametrize("expired_at", ["start", "checked", 1, 2, 3, 4, "batch"])
def test_expired_feature_deadline_stops_work_without_returning_partial_batch(
    monkeypatch, expired_at
):
    from coveo_recsys.features import engine as feature_engine

    context = bind(request())
    index = RetrievalIndex(package(), CATALOG_HASH)
    engine = FeatureEngine(index, schema())
    selected = candidates(index, context, ("a", "b", "c", "я"))
    scores = score_request(context, index)
    expired = expired_at == "start"
    failure = RuntimeError("deadline reached")
    checked, rows, batches = [], [], []
    original_check, original_row = engine._check, index.row
    original_batch = feature_engine.FeatureBatch
    anchor_count = len(context.prefix.full_last_detail)

    def check_deadline():
        if expired:
            raise failure

    def validate(*args):
        nonlocal expired
        checked.append(True)
        original_check(*args)
        if expired_at == "checked":
            expired = True

    def read_row(sku):
        nonlocal expired
        rows.append(sku)
        result = original_row(sku)
        if len(rows) - anchor_count == expired_at:
            expired = True
        return result

    def make_batch(*args):
        nonlocal expired
        result = original_batch(*args)
        batches.append(result)
        if expired_at == "batch":
            expired = True
        return result

    def unexpected_scoring(*args):
        pytest.fail("Просроченный запрос не должен начинать расчёт общих баллов")

    monkeypatch.setattr(engine, "_check", validate)
    monkeypatch.setattr(index, "row", read_row)
    monkeypatch.setattr(feature_engine, "FeatureBatch", make_batch)
    monkeypatch.setattr(feature_engine, "score_request", unexpected_scoring)
    with pytest.raises(RuntimeError) as caught:
        engine.build(
            context,
            selected,
            scores=None if expired_at == "start" else scores,
            check_deadline=check_deadline,
        )
    assert caught.value is failure
    assert bool(checked) == (expired_at != "start")
    assert len(batches) == (expired_at == "batch")
    if isinstance(expired_at, int):
        assert rows[anchor_count:] == list(selected.skus[:expired_at])
    elif expired_at in {"start", "checked"}:
        assert rows == []


@pytest.mark.parametrize("semantics,expected", [("set_v1", 0.5), ("multiset_v1", 0.8)])
def test_last_group_means_use_all_items_and_declared_multiplicities(semantics, expected):
    prepared = request(((2000, "detail", "a"),) * 4 + ((2000, "detail", "b"),), semantics=semantics)
    document = schema()
    document["event_semantics"] = semantics
    context = bind(prepared)
    index = RetrievalIndex(package(), CATALOG_HASH)
    actual = FeatureEngine(index, document, RetrievalConfig(history_limit=1)).build(
        context, candidates(index, context)
    )
    for name in ("last_description", "last_image"):
        assert row(actual, "a")[name] == pytest.approx((expected, True))
    assert row(actual, "c")["last_detail_detail"] == pytest.approx((0.125, True))
    assert row(actual, "c")["support_detail_detail"] == pytest.approx((math.log1p(3), True))


def test_full_prefix_statistics_survive_shortened_retained_history():
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
    actual, _, _ = build(prepared)
    values = row(actual, "b")
    for name, value in {
        "detail_count": 2,
        "detail_age": 2,
        "session_event_count": 4,
        "session_distinct_items": 2,
        "session_add_count": 1,
        "session_duration": 3,
        "cart_age": 1,
    }.items():
        assert values[name] == pytest.approx((math.log1p(value), True))
    assert values["cart_add"] == (1.0, True)
    assert values["cart_unknown"] == (0.0, True)
    assert values["historical_purchase_count"] == pytest.approx((math.log1p(2), True))


def test_conflicting_cart_preserves_age_of_last_unambiguous_action():
    actual, _, _ = build(
        request(
            ((2000, "add", "b"), (3000, "add", "b"), (3000, "remove", "b"), (4000, "detail", "a"))
        )
    )
    values = row(actual, "b")
    assert values["cart_add"] == values["cart_remove"] == (0.0, False)
    assert values["cart_unknown"] == (1.0, True)
    assert values["cart_age"] == pytest.approx((math.log1p(2), True))


def test_category_and_price_compare_full_last_group_without_changing_dictionary():
    actual, _, _ = build(request(((2000, "detail", "a"), (2000, "detail", "c"))))
    values = row(actual, "b")
    assert values["category"] == (1, True)
    assert values["category_similarity"] == values["category_match"] == (1.0, True)
    assert values["last_price_equal"] == values["last_price_sign"] == (0.0, True)
    assert values["last_price_absolute"] == (1.0, True)
    missing = row(actual, "я")
    for name in (
        "category_similarity",
        "category_match",
        "price_bucket",
        "last_price_equal",
        "last_price_sign",
        "last_price_absolute",
        "last_description",
    ):
        assert missing[name] == (0.0, False)
        assert missing[name + "_mask"] == (0.0, True)
    assert missing["category"] == (0, True)
    assert missing["item_description_available"] == (0.0, True)
    assert missing["last_image"][1] is True


@pytest.mark.parametrize(
    "query,age,available",
    [((1.0, 0.0), 1000, True), ((1.0, 0.0), 600001, False), (None, 1000, False)],
)
def test_search_presence_and_usable_search_age_are_distinct(query, age, available):
    actual, _, _ = build(request(((2000 + age, "detail", "a"),), query=query, search_time=2000))
    values = row(actual, "b")
    assert values["session_has_search"] == (1.0, True)
    assert values["search_age"] == pytest.approx(
        (math.log1p(age / 1000) if available else 0.0, available)
    )


@pytest.mark.parametrize(
    "change", ["request", "order", "outside", "origins", "purchased", "catalog"]
)
def test_incompatible_candidates_or_context_are_rejected(change):
    prepared = (
        request(((2000, "purchase", "c"), (3000, "detail", "a")))
        if change == "purchased"
        else request()
    )
    context = bind(prepared)
    index = RetrievalIndex(package(), CATALOG_HASH)
    selected = candidates(index, context)
    if change == "request":
        selected = replace(selected, request_id="other")
    elif change == "order":
        selected = replace(selected, skus=tuple(reversed(selected.skus)))
    elif change == "outside":
        selected = replace(selected, skus=("outside",))
    elif change == "origins":
        selected = replace(selected, origins=())
    elif change == "catalog":
        context = replace(context, catalog_sha256="b" * 64)
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema()).build(context, selected)


@pytest.mark.parametrize("change", ["name", "position", "dtype", "transformation", "semantics"])
def test_incompatible_feature_schema_is_rejected(change):
    document = schema()
    if change == "semantics":
        document["event_semantics"] = "multiset_v1"
    else:
        document["columns"][0][change] = {
            "name": "unknown",
            "position": 1,
            "dtype": "category",
            "transformation": "log1p",
        }[change]
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind(request())
    with pytest.raises(InputValidationError):
        FeatureEngine(index, document).build(context, candidates(index, context))


def test_score_context_and_config_cannot_be_reused_for_another_request():
    actual, index, context = build()
    scores = score_request(context, index, RetrievalConfig())
    other = replace(context, request_id="another")
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema()).build(other, candidates(index, other), scores=scores)
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema(), RetrievalConfig(history_limit=1)).build(
            context, candidates(index, context), scores=scores
        )


def test_nonfinite_shared_score_is_not_hidden_by_a_mask():
    _, index, context = build()
    scores = score_request(context, index, RetrievalConfig())
    scores.values["popularity"][0] = np.nan
    with pytest.raises(NumericError):
        FeatureEngine(index, schema()).build(context, candidates(index, context), scores=scores)


@pytest.mark.parametrize("change", ["score", "reason", "mask"])
def test_source_results_must_match_shared_scores(change):
    _, index, context = build()
    scores = score_request(context, index)
    selected = candidates(index, context)
    if change == "score":
        last = replace(selected.sources[-1], scores=(0.999,) + selected.sources[-1].scores[1:])
        selected = replace(selected, sources=selected.sources[:-1] + (last,))
    elif change == "reason":
        first = replace(selected.sources[0], available=False, reason="unexpected")
        selected = replace(selected, sources=(first,) + selected.sources[1:])
    else:
        scores.masks["popularity"][0] = False
    with pytest.raises(InputValidationError):
        FeatureEngine(index, schema()).build(context, selected, scores=scores)


@pytest.mark.parametrize("strength,support", [(0.09, 3), (0.125, 3), (0.3, 9)])
def test_last_graph_support_belongs_to_maximum_with_utf8_anchor_tie(strength, support):
    historical = package()
    graph = historical.graphs.relations[0]
    rho, counts = graph.rho.copy(), graph.support.copy()
    rho[1], counts[-1] = strength, 9
    graph = replace(graph, rho=rho, support=counts)
    historical = replace(
        historical,
        graphs=replace(historical.graphs, relations=(graph,) + historical.graphs.relations[1:]),
    )
    actual, _, _ = build(
        request(((2000, "detail", "b"), (2000, "detail", "a"))), package_value=historical
    )
    assert row(actual, "c")["support_detail_detail"] == pytest.approx((math.log1p(support), True))
