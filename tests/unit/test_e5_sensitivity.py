"""Диагностика кратностей меняет префикс и схему, но не A_G и не модель."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


def inputs(rows):
    prepared = request(rows)
    index = RetrievalIndex(package(), CATALOG_HASH)
    schema = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))
    baseline = FeatureEngine(index, schema)
    schema["event_semantics"] = "multiset_v1"
    multiset = FeatureEngine(index, schema)
    context = bind_context(prepared, index)
    scores = score_request(context, index)
    candidates = retrieve(context, index, scores=scores)
    features = baseline.build(context, candidates, scores=scores)
    return prepared, baseline, multiset, candidates, features


def compare(prepared, baseline, multiset, candidates, features, *, blocks=None):
    from coveo_recsys.evaluation.sensitivity import compare_multiplicities

    return compare_multiplicities(
        prepared,
        prepared.prefix.blocks if blocks is None else blocks,
        TaskConfig(),
        baseline,
        multiset,
        base_candidates=candidates,
        base_features=features,
    )


@pytest.mark.parametrize(
    "rows",
    [
        ((2000, "detail", "a"),),
        ((2000, "detail", "a"), (2000, "detail", "b")),
        ((1500, "detail", "a"), (2000, "detail", "a")),
        ((1500, "add", "a"), (2000, "remove", "a"), (2000, "detail", "b")),
    ],
)
def test_unchanged_prefix_reuses_values_with_an_explicit_diagnostic_schema(monkeypatch, rows):
    from coveo_recsys.evaluation import sensitivity

    values = inputs(rows)

    def unexpected(*args, **kwargs):
        pytest.fail("Для одинаковых префиксов повторный расчёт не нужен")

    monkeypatch.setattr(sensitivity, "score_request", unexpected)
    monkeypatch.setattr(sensitivity, "retrieve", unexpected)
    monkeypatch.setattr(values[2], "build", unexpected)
    actual = compare(*values)
    assert actual == {
        "request_id": values[0].request_id,
        "scope": "prefix_sensitivity_fixed_historical",
        "set_schema_sha256": values[4].feature_schema_sha256,
        "multiset_schema_sha256": values[2].schema_sha256,
        "affected": False,
        "prefix_changed": False,
        "candidate_changed": False,
        "common_candidates": len(values[3].skus),
        "changed_features": [],
        "set_candidate_skus": values[3].skus,
        "multiset_candidate_skus": values[3].skus,
    }
    assert actual["set_schema_sha256"] != actual["multiset_schema_sha256"]


def test_repeated_events_change_counts_and_means_but_keep_graph_support_and_purchase_set():
    values = inputs(
        ((2000, "detail", "a"),) * 4 + ((2000, "detail", "b"),) + ((2000, "purchase", "c"),) * 2
    )
    graph_support = [
        relation.support.copy() for relation in values[1].index.package.graphs.relations
    ]
    actual = compare(*values)
    changes = {item["name"]: item for item in actual["changed_features"]}
    assert actual["affected"] is True
    assert actual["prefix_changed"] is True
    assert actual["candidate_changed"] is False
    assert changes["detail_count"]["changed_rows"] == 1
    assert changes["detail_count"]["max_abs_delta"] == pytest.approx(math.log(5 / 2))
    assert changes["last_description"]["max_abs_delta"] == pytest.approx(0.3)
    assert not any(name.startswith("support_") for name in changes)
    assert "c" not in actual["multiset_candidate_skus"]
    assert values[0].prefix.purchased_skus == ("c",)
    for relation, before in zip(
        values[1].index.package.graphs.relations, graph_support, strict=True
    ):
        np.testing.assert_array_equal(relation.support, before)


def test_future_multiplicities_are_ignored_in_reconstruction():
    values = inputs(((2000, "detail", "a"),))
    future = request(((3000, "detail", "c"),) * 8 + ((4000, "purchase", "b"),))
    assert compare(*values, blocks=values[0].prefix.blocks + future.prefix.blocks) == compare(
        *values
    )


def test_multiplicities_can_change_candidates_without_applying_a_model():
    prepared, baseline, multiset, _, _ = inputs(
        ((2000, "detail", "a"),) + ((2000, "detail", "b"),) * 4
    )
    config = RetrievalConfig(quotas=(0, 0, 0, 1, 0, 0, 0, 0), max_candidates=1)
    baseline.config = multiset.config = config
    context = bind_context(prepared, baseline.index)
    scores = score_request(context, baseline.index, config)
    candidates = retrieve(context, baseline.index, config, scores=scores)
    features = baseline.build(context, candidates, scores=scores)
    actual = compare(prepared, baseline, multiset, candidates, features)
    assert actual["set_candidate_skus"] == ("a",)
    assert actual["multiset_candidate_skus"] == ("b",)
    assert actual["candidate_changed"] is True
    assert actual["common_candidates"] == 0
    assert actual["changed_features"] == []


@pytest.mark.parametrize("case", ["tail", "schema", "historical", "features"])
def test_incomplete_or_incompatible_reconstruction_is_rejected(case):
    prepared, baseline, multiset, candidates, features = inputs(
        ((1500, "add", "a"), (2000, "detail", "b"))
    )
    blocks = prepared.prefix.blocks
    if case == "tail":
        blocks = blocks[-1:]
    elif case == "schema":
        multiset = baseline
    elif case == "historical":
        _, _, multiset, _, _ = inputs(((2000, "detail", "a"),))
    else:
        features = replace(features, request_id="other")
    with pytest.raises(InputValidationError):
        compare(prepared, baseline, multiset, candidates, features, blocks=blocks)
