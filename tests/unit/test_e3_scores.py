from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


def test_history_scores_keep_counts_and_regularization():
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(
        request(
            ((2000, "detail", "a"), (2000, "detail", "a"), (2000, "detail", "b")),
            semantics="multiset_v1",
        ),
        index,
    )
    scores = score_request(context, index)
    np.testing.assert_allclose(scores.weights, [2 / 3, 1 / 3])
    assert scores.values["detail_detail"][1] == pytest.approx(1 / 6)
    assert scores.values["detail_detail"][2] == pytest.approx(1 / 12 + 1 / 33)


def test_configured_history_tail_is_not_full_last_group():
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(request(((2000, "detail", "b"), (2000, "detail", "a"))), index)
    scores = score_request(context, index, replace(RetrievalConfig(), history_limit=1))
    assert [i.sku for i in scores.history] == ["a"]
    assert len(context.prefix.full_last_detail) == 2


def test_distant_history_is_normalized_before_exponentiation():
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(request(((2000, "detail", "a"), (10**12, "page", None))), index)
    scores = score_request(context, index)
    assert scores.weights.tolist() == [1.0]
    assert scores.values["detail_detail"][1] == pytest.approx(0.25)


@pytest.mark.parametrize("age,available", [(599999, True), (600000, True), (600001, False)])
def test_search_age_boundary(age, available):
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(
        request(((10000 + age, "detail", "a"),), query=(1.0, 0.0), search_time=10000), index
    )
    scores = score_request(context, index)
    assert bool(scores.masks["search"][0]) is available


def test_runtime_failure_cannot_be_an_arbitrary_zero_score():
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(request(), index)
    scores = score_request(context, index, source_runtime={"description": "backend_timeout"})
    assert not scores.masks["description"].any()
    assert scores.reasons["description"] == "backend_timeout"
    with pytest.raises(InputValidationError):
        score_request(context, index, source_runtime={"typo": "disabled"})
