"""1000 случаев, seed17: разрешённые признаки Э7 по независимому эталону Э3."""

from dataclasses import replace

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request
from tests.reference.e3_features_oracle import feature_oracle
from tests.reference.e3_retrieval_oracle import retrieval_oracle
from tests.unit.test_e7_retrieval_features import (
    ALLOWED_SOURCES,
    CONFIG,
    VARIANT,
    expected_columns,
    schema,
)


@settings(max_examples=1000, deadline=None, database=None, print_blob=True)
@seed(17)
@given(
    events=st.lists(
        st.tuples(
            st.integers(0, 5),
            st.sampled_from(("detail", "add", "remove", "purchase")),
            st.sampled_from(("a", "b", "c", "я", "outside")),
        ),
        max_size=12,
    ),
    semantics=st.sampled_from(("set_v1", "multiset_v1")),
    history_limit=st.integers(1, 20),
    block_size=st.sampled_from((1, 2, 40)),
    failed_source=st.sampled_from((None, *ALLOWED_SOURCES)),
)
def test_allowed_features_scores_and_candidates_match_independent_oracle(
    events, semantics, history_limit, block_size, failed_source
):
    historical = package()
    rows = tuple((2000 + tick * 1000, kind, sku) for tick, kind, sku in events) + (
        (9000, "detail", "a"),
    )
    prepared = request(rows, semantics=semantics, query=(0.0, 1.0), search_time=2500)
    index = RetrievalIndex(historical, CATALOG_HASH, config_id=VARIANT)
    context = bind_context(prepared, index)
    config = replace(CONFIG, history_limit=history_limit, block_size=block_size)
    runtime = {failed_source: "unavailable"} if failed_source is not None else None
    scores = score_request(context, index, config, runtime)
    selected = retrieve(context, index, config, scores=scores)
    document = schema()
    document["event_semantics"] = semantics
    actual = FeatureEngine(index, document, config).build(context, selected, scores=scores)
    expected = feature_oracle(context, historical, config, source_runtime=runtime)
    expected_selection = retrieval_oracle(context, historical, config, source_runtime=runtime)
    assert actual.skus == tuple(expected)
    assert actual.numeric_columns == expected_columns()
    assert actual.categorical_columns == ()
    for source in selected.sources:
        assert source.skus == tuple(expected_selection["sources"][source.source])
    for source in ALLOWED_SOURCES:
        np.testing.assert_allclose(
            scores.values[source],
            [expected_selection["values"][source][sku] for sku in historical.catalog.skus],
            rtol=1e-8,
            atol=1e-10,
        )
        assert scores.masks[source].tolist() == [
            expected_selection["masks"][source][sku] for sku in historical.catalog.skus
        ]
    for sku, values, masks in zip(actual.skus, actual.numeric, actual.masks, strict=True):
        assert masks == tuple(expected[sku][name][1] for name in actual.numeric_columns)
        np.testing.assert_allclose(
            values,
            [expected[sku][name][0] for name in actual.numeric_columns],
            rtol=1e-8,
            atol=1e-10,
        )
