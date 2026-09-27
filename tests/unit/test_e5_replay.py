"""Локальная выдача Э5 использует ядро и отделяет выбор ветви от доставки."""

import builtins
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.records import Branch, DeliveryStatus
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.ranking.load import RankerModel
from coveo_recsys.ranking.predict import predict
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


class FeatureBooster:
    def __init__(self, *, invalid=False):
        self.invalid = invalid
        self.calls = []

    def predict(self, matrix, **kwargs):
        self.calls.append(matrix.copy())
        return np.full(len(matrix), np.nan) if self.invalid else matrix[:, 0]


def setup(prepared=None, *, invalid=False, historical=None):
    index = RetrievalIndex(package() if historical is None else historical, CATALOG_HASH)
    schema = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))
    engine = FeatureEngine(index, schema)
    model = RankerModel(
        FeatureBooster(invalid=invalid),
        {
            "identity": {"catalog_sha256": CATALOG_HASH},
            "feature_schema_sha256": engine.schema_sha256,
            "numeric_columns": list(engine.numeric_columns),
            "categorical_columns": list(engine.categorical_columns),
            "actual_tree_count": 1,
        },
    )
    return bind_context(request() if prepared is None else prepared, index), engine, model


def test_primary_matches_direct_core_and_reuses_the_same_scores(monkeypatch):
    from coveo_recsys.evaluation import replay

    context, engine, model = setup()
    scores = score_request(context, engine.index, engine.config)
    candidates = retrieve(context, engine.index, engine.config, scores=scores)
    features = engine.build(context, candidates, scores=scores)
    ranked = predict(model, features, catalog_sha256=CATALOG_HASH)
    calls = []

    def recorded(*args, **kwargs):
        calls.append(args)
        return score_request(*args, **kwargs)

    monkeypatch.setattr(replay, "score_request", recorded)
    actual = replay.recommend(context, engine, model, package_id="model", trace_ref="local:q")
    assert actual.candidates == candidates
    assert tuple(zip(actual.delivered_skus, actual.scores, strict=True)) == ranked
    np.testing.assert_array_equal(model.booster.calls[0], model.booster.calls[1])
    assert len(calls) == 1
    assert actual.explanations == ()
    assert actual.elapsed_ms == 0
    assert actual.trace_ref == "local:q"
    assert actual.delivery == DeliveryStatus.DELIVERED


@pytest.mark.parametrize("policy, expected", [("P", ("a", "c", "я")), ("R", ("c", "a", "я"))])
def test_controls_share_eligibility_and_keep_cart_items(policy, expected):
    from coveo_recsys.evaluation.replay import recommend

    prepared = request(
        (
            (2000, "detail", "a"),
            (2100, "purchase", "b"),
            (2200, "add", "a"),
            (3000, "detail", "c"),
            (3000, "detail", "outside"),
        )
    )
    context, engine, model = setup(prepared)
    actual = recommend(context, engine, model, policy=policy, package_id="model", trace_ref="q")
    assert actual.delivered_skus == expected
    assert actual.candidates.skus == expected
    assert actual.candidates.branch == Branch.PRIMARY
    assert model.booster.calls == []


def test_missing_model_selects_an_independent_fallback():
    from coveo_recsys.evaluation.replay import recommend

    context, engine, _ = setup(request(((2000, "detail", "c"),)))
    actual = recommend(context, engine, None, package_id="model", trace_ref="q")
    assert actual.candidates.branch == Branch.FALLBACK
    assert actual.delivered_skus == ("c", "a", "b", "я")
    assert "model_unavailable" in actual.reasons
    assert all(set(origins) <= {"recent", "popularity"} for _, origins in actual.candidates.origins)


@pytest.mark.parametrize("policy", ["GCS+Rk", "P", "R"])
def test_empty_eligible_catalog_is_a_successful_empty_answer(policy):
    from coveo_recsys.evaluation.replay import recommend

    prepared = request(
        tuple((2000, "purchase", sku) for sku in ("a", "b", "c", "я")) + ((3000, "detail", "a"),)
    )
    context, engine, model = setup(prepared)
    diagnostics = {}
    actual = recommend(
        context,
        engine,
        model,
        policy=policy,
        package_id="model",
        trace_ref="q",
        diagnostics=diagnostics,
    )
    assert actual.delivered_skus == actual.candidates.skus == ()
    assert actual.delivery == DeliveryStatus.DELIVERED
    assert actual.candidates.branch == Branch.PRIMARY
    assert model.booster.calls == []
    assert diagnostics["source_lists"]["popularity"]["skus"] == ()
    assert diagnostics["source_lists"]["popularity"]["truncated"] is False


@pytest.mark.parametrize("failure", ["schema", "numeric", "catalog"])
def test_incompatible_or_invalid_model_is_not_hidden_by_fallback(failure):
    from coveo_recsys.evaluation.replay import recommend

    context, engine, model = setup(invalid=failure == "numeric")
    if failure == "schema":
        model.summary["feature_schema_sha256"] = "e" * 64
    elif failure == "catalog":
        model.summary["identity"]["catalog_sha256"] = "e" * 64
    with pytest.raises((InputValidationError, NumericError)):
        recommend(context, engine, model, package_id="model", trace_ref="q")


@pytest.mark.parametrize("case", ["policy", "ordinal", "availability"])
def test_replay_rejects_another_protocol_or_mode(case):
    from coveo_recsys.evaluation.replay import recommend

    context, engine, model = setup(request(available_time=3000) if case == "availability" else None)
    if case == "ordinal":
        context = replace(context, ordinal=2)
    with pytest.raises(InputValidationError):
        recommend(
            context,
            engine,
            model,
            policy="other" if case == "policy" else "GCS+Rk",
            package_id="model",
            trace_ref="q",
        )


@pytest.mark.parametrize("delivery", [DeliveryStatus.FAILED, DeliveryStatus.NOT_ATTEMPTED])
def test_failed_delivery_preserves_selected_candidates(delivery):
    from coveo_recsys.evaluation.replay import finalize_result

    context, engine, _ = setup()
    candidates = retrieve(context, engine.index)
    actual = finalize_result(
        context,
        candidates,
        (("b", 2.0),),
        package_id="model",
        trace_ref="q",
        delivery=delivery,
        reasons=("delivery_failure",),
    )
    assert actual.candidates == candidates
    assert actual.delivered_skus == actual.scores == ()
    assert actual.delivery == delivery
    assert actual.reasons == ("delivery_failure",)


def test_no_selected_branch_has_no_candidates_and_is_not_attempted():
    from coveo_recsys.evaluation.replay import finalize_result

    context, _, _ = setup()
    actual = finalize_result(
        context, None, (), package_id="model", trace_ref="q", reasons=("no_ready_branch",)
    )
    assert actual.candidates.branch == Branch.NONE
    assert actual.candidates.skus == actual.delivered_skus == ()
    assert actual.delivery == DeliveryStatus.NOT_ATTEMPTED


def test_diagnostics_measure_before_quota_and_distinguish_unknown_lists():
    from coveo_recsys.evaluation.replay import recommend

    context, engine, model = setup()
    engine.config = RetrievalConfig(quotas=(0,) * 8, max_candidates=1)
    diagnostics = {}
    actual = recommend(
        context,
        engine,
        model,
        package_id="model",
        trace_ref="q",
        diagnostics=diagnostics,
        diagnostic_depth=1,
    )
    lists = diagnostics["source_lists"]
    assert lists["detail_detail"] == {
        "skus": ("b",),
        "depth": 1,
        "truncated": True,
        "reason": None,
    }
    assert lists["detail_purchase"] == {
        "skus": ("b",),
        "depth": 1,
        "truncated": False,
        "reason": None,
    }
    assert lists["search"]["skus"] is None
    assert lists["search"]["reason"]
    assert lists["search"]["truncated"] is None
    assert actual.candidates.sources[0].skus == ()
    assert diagnostics["features"].skus == actual.candidates.skus
    assert len(model.booster.calls) == 1


@pytest.mark.parametrize("policy", ["P", "R"])
def test_control_diagnostics_cover_unlimited_popularity_and_recent_before_quota(policy):
    from coveo_recsys.evaluation.replay import recommend

    context, engine, model = setup(request(((2000, "detail", "c"),)))
    engine.config = RetrievalConfig(quotas=(0,) * 8, max_candidates=1)
    diagnostics = {}
    recommend(
        context,
        engine,
        model,
        policy=policy,
        package_id="model",
        trace_ref="q",
        diagnostics=diagnostics,
        diagnostic_depth=2,
    )
    assert diagnostics["source_lists"]["popularity"] == {
        "skus": ("a", "b"),
        "depth": 2,
        "truncated": True,
        "reason": None,
    }
    assert diagnostics["features"] is None
    if policy == "R":
        assert diagnostics["source_lists"]["recent"]["skus"] == ("c",)


@pytest.mark.parametrize("policy", ["GCS+Rk", "P", "R"])
def test_recommend_needs_no_files_or_clocks_and_ignores_future_purchases(monkeypatch, policy):
    from coveo_recsys.evaluation.replay import recommend

    context, engine, model = setup()
    future = request(((2000, "detail", "a"), (3000, "purchase", "b")), time=2000)
    future_context = bind_context(future, engine.index)

    def forbidden(*args, **kwargs):
        raise AssertionError("Файлы и часы не входят в чистую выдачу")

    with monkeypatch.context() as patch:
        for owner, name in (
            (builtins, "open"),
            (Path, "read_text"),
            (Path, "read_bytes"),
            (time, "time"),
            (time, "monotonic"),
            (time, "perf_counter"),
        ):
            patch.setattr(owner, name, forbidden)
        before = recommend(context, engine, model, policy=policy, package_id="model", trace_ref="q")
        after = recommend(
            future_context, engine, model, policy=policy, package_id="model", trace_ref="q"
        )
    assert before == after


@pytest.mark.parametrize("policy", ["GCS+Rk", "P", "R"])
def test_empty_historical_catalog_does_not_invent_candidates(policy):
    from coveo_recsys.evaluation.replay import recommend

    original = package()
    graphs = tuple(
        replace(
            graph,
            indptr=np.zeros(1, dtype=np.int64),
            indices=graph.indices[:0],
            weights=graph.weights[:0],
            probabilities=graph.probabilities[:0],
            support=graph.support[:0],
            row_sums=graph.row_sums[:0],
            rho=graph.rho[:0],
        )
        for graph in original.graphs.relations
    )

    def empty_modality(modality):
        return replace(modality, vectors=modality.vectors[:0], mask=modality.mask[:0], reasons=())

    historical = replace(
        original,
        catalog=replace(original.catalog, skus=()),
        graphs=replace(original.graphs, relations=graphs),
        statistics=replace(
            original.statistics,
            detail_counts=original.statistics.detail_counts[:0],
            purchase_counts=original.statistics.purchase_counts[:0],
            popularity=original.statistics.popularity[:0],
        ),
        content=replace(
            original.content,
            description=empty_modality(original.content.description),
            image=empty_modality(original.content.image),
            category_codes=original.content.category_codes[:0],
            price_bucket=original.content.price_bucket[:0],
            price_mask=original.content.price_mask[:0],
        ),
    )
    context, engine, model = setup(historical=historical)
    actual = recommend(context, engine, model, policy=policy, package_id="model", trace_ref="q")
    assert actual.delivered_skus == actual.candidates.skus == ()
    assert actual.delivery == DeliveryStatus.DELIVERED
    assert model.booster.calls == []
