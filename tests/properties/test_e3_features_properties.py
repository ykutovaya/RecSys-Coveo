"""1000 случаев, seed17: независимые признаки и перестановка временной группы."""

from dataclasses import replace

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts.historical import MetadataRecord
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request
from tests.reference.e3_features_oracle import feature_oracle
from tests.unit.test_e3_features import bind, row, schema

NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
EVENTS = st.lists(
    st.tuples(
        st.integers(0, 7),
        st.sampled_from(("detail", "add", "remove", "purchase")),
        st.sampled_from(("a", "b", "c", "я", "outside")),
    ),
    min_size=0,
    max_size=15,
)


@NORMATIVE
@seed(17)
@given(
    EVENTS,
    st.sampled_from(("set_v1", "multiset_v1")),
    st.integers(1, 5),
    st.integers(0, 2),
    st.integers(0, 3),
    st.sampled_from((None, "description", "search", "detail_detail")),
    st.booleans(),
)
def test_all_feature_columns_match_independent_scalar_oracle(
    events, semantics, limit, quota, query, disabled_source, has_observed_metadata
):
    rows = tuple((2000 + tick * 1000, kind, sku) for tick, kind, sku in events) + (
        (9000, "detail", "a"),
    )
    vector = (None, (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0))[query]
    prepared = request(rows, semantics=semantics, query=vector, search_time=3000)
    context = bind(prepared)
    historical = package()
    config = RetrievalConfig(
        quotas=(quota,) * 8, max_candidates=max(1, 8 * quota), history_limit=limit, block_size=2
    )
    observed = (
        {"outside": MetadataRecord("outside", (1.0, 1.0), (-1.0, 0.0), "root/new", 4)}
        if has_observed_metadata
        else None
    )
    runtime = {disabled_source: "unavailable"} if disabled_source is not None else None
    index = RetrievalIndex(historical, CATALOG_HASH, observed)
    scores = score_request(context, index, config, source_runtime=runtime)
    selected = retrieve(context, index, config, scores=scores)
    document = schema()
    document["event_semantics"] = semantics
    engine = FeatureEngine(index, document, config)
    actual = engine.build(context, selected, scores=scores)
    expected = feature_oracle(
        context, historical, config, observed=observed, source_runtime=runtime
    )
    assert actual.skus == tuple(expected)
    for sku in actual.skus:
        values = row(actual, sku)
        assert values.keys() == expected[sku].keys()
        assert [pair[1] for pair in values.values()] == [expected[sku][name][1] for name in values]
        np.testing.assert_allclose(
            [pair[0] for pair in values.values()],
            [expected[sku][name][0] for name in values],
            rtol=1e-8,
            atol=1e-10,
        )
    if runtime is None:
        assert engine.build(context, selected) == actual
    shuffled = replace(
        context,
        prefix=replace(
            context.prefix, full_last_detail=tuple(reversed(context.prefix.full_last_detail))
        ),
    )
    shuffled_scores = score_request(shuffled, index, config, source_runtime=runtime)
    assert engine.build(shuffled, selected, scores=shuffled_scores) == actual
