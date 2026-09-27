"""Будущее, перестановки и разбиение запросов не меняют локальную выдачу."""

import json
from pathlib import Path

import numpy as np
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.evaluation.replay import recommend
from coveo_recsys.evaluation.sensitivity import compare_multiplicities
from coveo_recsys.features import FeatureEngine
from coveo_recsys.ranking.load import RankerModel
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


def engine():
    schema = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))
    return FeatureEngine(RetrievalIndex(package(), CATALOG_HASH), schema)


class DeterministicBooster:
    def predict(self, matrix, **kwargs):
        return np.sum(matrix, axis=1)


@given(
    st.lists(
        st.tuples(
            st.integers(2001, 9000),
            st.sampled_from(["detail", "purchase", "add", "remove"]),
            st.sampled_from(["a", "b", "c", "я", "outside"]),
        ),
        max_size=20,
    ),
    st.sampled_from(["GCS+Rk", "P", "R"]),
)
@seed(17)
def test_future_events_and_purchases_never_change_the_reply(future, policy):
    selected = engine()
    model = RankerModel(
        DeterministicBooster(),
        {
            "identity": {"catalog_sha256": CATALOG_HASH},
            "feature_schema_sha256": selected.schema_sha256,
            "numeric_columns": list(selected.numeric_columns),
            "categorical_columns": list(selected.categorical_columns),
            "actual_tree_count": 1,
        },
    )
    original = request(((2000, "detail", "a"),))
    with_future = request(((2000, "detail", "a"), *future), time=2000)
    actual = [
        recommend(
            bind_context(item, selected.index),
            selected,
            model,
            policy=policy,
            package_id="model",
            trace_ref="local:q",
        )
        for item in (original, with_future)
    ]
    assert actual[0] == actual[1]


@given(
    st.permutations(("a", "b", "c", "я")),
    st.sampled_from(["P", "R"]),
    st.integers(2000, 2**31),
    st.integers(0, 86_400_000),
)
@seed(17)
def test_simultaneous_views_and_popularity_ties_use_utf8_order(order, policy, viewed_at, delay):
    selected = engine()
    selected.index.package.statistics.popularity[:] = 0.25
    prepared = request(
        (*((viewed_at, "detail", sku) for sku in order), (viewed_at + delay, "add", "outside"))
    )
    actual = recommend(
        bind_context(prepared, selected.index),
        selected,
        None,
        policy=policy,
        package_id="model",
        trace_ref="local:q",
    )
    assert actual.delivered_skus == ("a", "b", "c", "я")
    assert actual.scores == ((0.25 if policy == "P" else -delay / 1000),) * 4


@given(
    st.integers(1, 8),
    st.sampled_from(["P", "R"]),
    st.lists(
        st.lists(
            st.tuples(
                st.integers(2000, 9000),
                st.sampled_from(["detail", "purchase", "add", "remove"]),
                st.sampled_from(["a", "b", "c", "я", "outside"]),
            ),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=8,
    ),
)
@seed(17)
def test_processing_chunks_does_not_carry_state_between_requests(chunk_size, policy, histories):
    selected = engine()
    contexts = [bind_context(request(rows), selected.index) for rows in histories]

    def run(items):
        return [
            recommend(
                context, selected, None, policy=policy, package_id="model", trace_ref="local:q"
            )
            for context in items
        ]

    chunks = [
        run(contexts[start : start + chunk_size]) for start in range(0, len(contexts), chunk_size)
    ]
    assert [result for chunk in chunks for result in chunk] == run(contexts)


@given(
    st.lists(
        st.tuples(
            st.integers(1001, 2000),
            st.sampled_from(["detail", "purchase", "add", "remove"]),
            st.sampled_from(["a", "b", "c", "я", "outside"]),
        ),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
@seed(17)
def test_unit_multiplicities_reuse_exact_full_recalculation(rows):
    baseline = engine()
    schema = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf-8"))
    schema["event_semantics"] = "multiset_v1"
    multiset = FeatureEngine(baseline.index, schema)
    prepared = request(rows)
    sets, matrices = [], []
    for selected, item in (
        (baseline, prepared),
        (multiset, request(rows, semantics="multiset_v1")),
    ):
        context = bind_context(item, selected.index)
        scores = score_request(context, selected.index, selected.config)
        candidates = retrieve(context, selected.index, selected.config, scores=scores)
        sets.append(candidates)
        matrices.append(selected.build(context, candidates, scores=scores))
    assert sets[0] == sets[1]
    for field in ("numeric", "categorical", "masks", "skus"):
        assert getattr(matrices[0], field) == getattr(matrices[1], field)
    actual = compare_multiplicities(
        prepared,
        prepared.prefix.blocks,
        TaskConfig(),
        baseline,
        multiset,
        base_candidates=sets[0],
        base_features=matrices[0],
    )
    assert actual["affected"] is False
    assert actual["changed_features"] == []
    assert actual["common_candidates"] == len(sets[1].skus)
    assert actual["multiset_candidate_skus"] == sets[1].skus
    assert actual["multiset_schema_sha256"] == matrices[1].feature_schema_sha256
