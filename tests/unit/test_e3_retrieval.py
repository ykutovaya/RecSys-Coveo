"""Правила точного отбора, происхождений и отдельного резерва Э3."""

from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts.common import InputValidationError, NumericError
from coveo_recsys.contracts.historical import GraphRelation
from coveo_recsys.contracts.records import Branch, RequestContext
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.retrieval import retrieve, retrieve_fallback
from coveo_recsys.retrieval.candidates import remove_source
from coveo_recsys.retrieval.scores import RetrievalIndex, score_request
from coveo_recsys.retrieval.sources import exact_top_m, source_results
from tests.reference.e3_cases import CATALOG_HASH, package, request


def context(prepared=None):
    prepared = request() if prepared is None else prepared
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


def config(*, maximum=3, **quotas):
    return RetrievalConfig(quotas=tuple(quotas.get(s, 0) for s in SOURCES), max_candidates=maximum)


@pytest.mark.parametrize("size", [1, 2, 4, 50])
@pytest.mark.parametrize("limit", [0, 1, 3, 10])
def test_exact_top_m_resolves_ties_across_blocks_and_keeps_negative_scores(size, limit):
    ids = ("я", "é", "z", "a", "b")
    actual = exact_top_m(
        ids,
        np.array([-1.0, 0.0, 0.0, 0.0, 99.0]),
        np.array([1, 1, 1, 1, 0], dtype=bool),
        limit,
        block_size=size,
    )
    assert tuple(ids[i] for i in actual) == ("a", "z", "é", "я")[:limit]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_top_m_rejects_nonfinite_measurable_scores(value):
    with pytest.raises(NumericError):
        exact_top_m(("a",), np.array([value]), np.array([True]), 1)


@pytest.mark.parametrize(
    "ids,values,mask,limit,block",
    [
        (("a", "a"), [1.0, 2.0], [True, True], 1, 1),
        (("",), [1.0], [True], 1, 1),
        (("\ud800",), [1.0], [True], 1, 1),
        (("a",), [1.0, 2.0], [True, True], 1, 1),
        (("a",), [1.0], [True, False], 1, 1),
        (("a",), [1.0], [1], 1, 1),
        (("a",), [1.0], [True], True, 1),
        (("a",), [1.0], [True], -1, 1),
        (("a",), [1.0], [True], 1, 0),
        (("a",), [1.0], [True], 1, True),
    ],
)
def test_top_m_preserves_input_checks(ids, values, mask, limit, block):
    with pytest.raises(InputValidationError):
        exact_top_m(ids, np.array(values), np.array(mask), limit, block_size=block)


@pytest.mark.parametrize("limit", [0, 1, 2, 10])
def test_top_m_masks_nonfinite_values_and_preserves_utf8_nul_suffix_order(limit):
    ids = ("a\0", "a", "я", "z")
    actual = exact_top_m(
        ids, np.array([-0.0, 0.0, np.nan, np.inf]), np.array([True, True, False, False]), limit
    )
    assert tuple(ids[i] for i in actual) == ("a", "a\0")[:limit]
    with pytest.raises(NumericError):
        exact_top_m(("a",), np.array([np.nan]), np.array([True]), limit)


def test_sources_reuse_validated_catalog_order_without_reencoding_ids(monkeypatch):
    import coveo_recsys.retrieval.sources as source_module

    idx, ctx, cfg = RetrievalIndex(package(), CATALOG_HASH), context(), config(description=1)
    calls = []
    original = source_module.id_bytes

    def counted(sku):
        calls.append(sku)
        return original(sku)

    monkeypatch.setattr(source_module, "id_bytes", counted)
    expected = retrieve(ctx, idx, cfg), retrieve_fallback(ctx, idx, cfg)
    assert (retrieve(ctx, idx, cfg), retrieve_fallback(ctx, idx, cfg)) == expected
    assert calls == []


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_cached_catalog_does_not_skip_score_validation_for_zero_quota(value):
    idx, ctx, cfg = RetrievalIndex(package(), CATALOG_HASH), context(), config()
    scores = score_request(ctx, idx, cfg)
    scores.values["popularity"][0] = value
    with pytest.raises(NumericError):
        source_results(scores)


def test_source_lists_apply_quotas_without_losing_measurable_scores():
    idx, ctx, cfg = (
        RetrievalIndex(package(), CATALOG_HASH),
        context(),
        config(detail_detail=1, description=1),
    )
    scores = score_request(ctx, idx, cfg)
    sources = {s.source: s for s in source_results(scores)}
    assert sources["detail_detail"].skus == ("b",)
    assert sources["description"].skus == ("a",)
    assert scores.masks["detail_detail"][idx.positions["c"]]
    assert scores.values["detail_detail"][idx.positions["c"]] > 0
    assert sources["detail_add"].skus == ()
    assert sources["detail_add"].available


def test_zero_graph_score_is_measurable_but_not_a_graph_candidate():
    idx = RetrievalIndex(package(), CATALOG_HASH)
    sources = source_results(
        score_request(context(request(((2000, "detail", "c"),))), idx, config(detail_detail=2))
    )
    graph = sources[0]
    assert graph.available and graph.reason is None and graph.skus == ()


def test_union_preserves_every_origin_and_fill_is_not_popularity_quota():
    result = retrieve(
        context(),
        RetrievalIndex(package(), CATALOG_HASH),
        config(detail_detail=1, detail_purchase=1),
    )
    assert result.branch == Branch.PRIMARY and result.skus == ("a", "b", "c")
    assert dict(result.origins) == {
        "a": ("fill",),
        "b": ("detail_detail", "detail_purchase"),
        "c": ("fill",),
    }
    assert next(s for s in result.sources if s.source == "popularity").skus == ()


def test_fill_survives_unavailable_initial_popularity_without_fictitious_source_score():
    result = retrieve(
        context(),
        RetrievalIndex(package(), CATALOG_HASH),
        config(popularity=1),
        {"popularity": "unavailable"},
    )
    pop = next(s for s in result.sources if s.source == "popularity")
    assert not pop.available and pop.reason == "unavailable" and pop.skus == ()
    assert result.skus == ("a", "b", "c")
    assert all(origins == ("fill",) for _, origins in result.origins)


def test_fallback_has_own_recent_order_then_popularity_and_excludes_purchases():
    ctx = context(
        request(
            (
                (2000, "detail", "b"),
                (3000, "detail", "a"),
                (3000, "detail", "c"),
                (3000, "purchase", "a"),
            )
        )
    )
    idx, cfg = RetrievalIndex(package(), CATALOG_HASH), config()
    primary, reserve = retrieve(ctx, idx, cfg), retrieve_fallback(ctx, idx, cfg)
    assert primary.skus == ("b", "c", "я")
    assert reserve.branch == Branch.FALLBACK and reserve.skus == ("c", "b", "я")
    assert dict(reserve.origins) == {"b": ("recent",), "c": ("recent",), "я": ("popularity",)}


def test_fallback_does_not_inherit_recent_quota_or_limited_history():
    ctx = context(request(((2000, "detail", "c"), (3000, "detail", "b"), (4000, "detail", "a"))))
    result = retrieve_fallback(
        ctx, RetrievalIndex(package(), CATALOG_HASH), replace(config(), history_limit=1)
    )
    assert result.skus == ("a", "b", "c")
    assert all(origins == ("recent",) for _, origins in result.origins)


@pytest.mark.parametrize("purchased", [(), ("a",), ("outside",), ("a", "b", "c", "я")])
def test_fallback_purchase_mask_uses_existing_positions(purchased):
    idx, cfg = RetrievalIndex(package(), CATALOG_HASH), config()
    ctx = context(
        request(((2000, "detail", "a"),) + tuple((2000, "purchase", sku) for sku in purchased))
    )
    expected = retrieve_fallback(ctx, idx, cfg)

    class IndexedCatalog(tuple):
        def __iter__(self):
            pytest.fail("Повторный обход всего каталога для маски покупок")

    # Данные прежние; запрещён только обход всех SKU после построения индекса.
    object.__setattr__(idx.package.catalog, "skus", IndexedCatalog(idx.package.catalog.skus))
    assert retrieve_fallback(ctx, idx, cfg) == expected


@pytest.mark.parametrize(
    "fn,branch", [(retrieve, Branch.PRIMARY), (retrieve_fallback, Branch.FALLBACK)]
)
def test_all_purchased_catalog_preserves_chosen_empty_branch(fn, branch):
    ctx = context(request(tuple((2000, "purchase", sku) for sku in package().catalog.skus)))
    result = fn(ctx, RetrievalIndex(package(), CATALOG_HASH), config())
    assert result.branch == branch and result.skus == () and result.origins == ()


def test_graph_evidence_uses_total_contribution_of_repeated_sku():
    ctx = context(request(((2000, "detail", "b"), (3000, "detail", "b"), (3000, "detail", "a"))))
    result = retrieve(ctx, RetrievalIndex(package(), CATALOG_HASH), config(detail_detail=2))
    evidence = next(e for e in result.evidence if e.sku == "c" and e.source == "detail_detail")
    assert evidence.anchor_sku == "b" and evidence.support == 3
    assert (
        evidence.score == next(s for s in result.sources if s.source == "detail_detail").scores[0]
    )


def test_equal_graph_contributions_choose_utf8_anchor():
    original = package()
    graph = GraphRelation(
        "detail_detail",
        np.array([0, 1, 2, 2, 2], dtype=np.int64),
        np.array([2, 2], dtype=np.int64),
        np.ones(2),
        np.ones(2),
        np.array([3, 3], dtype=np.int64),
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([1 / 11, 1 / 11, 0.0, 0.0]),
    )
    altered = replace(
        original, graphs=replace(original.graphs, relations=(graph, *original.graphs.relations[1:]))
    )
    ctx = context(request(((2000, "detail", "b"), (2000, "detail", "a"))))
    result = retrieve(
        ctx, RetrievalIndex(altered, CATALOG_HASH), config(maximum=1, detail_detail=1)
    )
    assert result.evidence[0].anchor_sku == "a"


def test_search_evidence_names_used_context_and_space():
    ctx = context(request(query=(1.0, 0.0), search_time=2000))
    result = retrieve(ctx, RetrievalIndex(package(), CATALOG_HASH), config(search=1))
    evidence = next(e for e in result.evidence if e.source == "search")
    assert evidence.context_id == ctx.prefix.search.block_id
    assert evidence.space_version


def test_fixed_source_removal_preserves_other_origins_without_refilling():
    result = retrieve(
        context(),
        RetrievalIndex(package(), CATALOG_HASH),
        config(detail_detail=1, detail_purchase=1),
    )
    removed = remove_source(result, "detail_detail")
    assert removed.skus == result.skus
    assert dict(removed.origins)["b"] == ("detail_purchase",)
    removed = remove_source(removed, "detail_purchase")
    assert removed.skus == ("a", "c")
    assert all(e.source not in {"detail_detail", "detail_purchase"} for e in removed.evidence)


def test_precomputed_scores_cannot_be_reused_for_different_request():
    idx, ctx, cfg = RetrievalIndex(package(), CATALOG_HASH), context(), config()
    scores = score_request(ctx, idx, cfg)
    with pytest.raises(InputValidationError):
        retrieve(replace(ctx, request_id="other"), idx, cfg, scores=scores)


def test_precomputed_scores_and_runtime_override_cannot_be_combined():
    idx, ctx, cfg = RetrievalIndex(package(), CATALOG_HASH), context(), config()
    scores = score_request(ctx, idx, cfg)
    with pytest.raises(InputValidationError):
        retrieve(ctx, idx, cfg, {"description": "disabled"}, scores=scores)
