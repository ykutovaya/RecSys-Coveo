"""Метрики и Delta_fixed: независимый эталон на 1000 примерах, seed 17."""

import math
from dataclasses import replace

from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.evaluation.metrics import aggregate_metrics, evaluate_session
from tests.reference.e3_cases import request
from tests.reference.e5_metrics_oracle import session_oracle, source_oracle
from tests.unit.test_e5_metrics import result, target


@seed(17)
@given(
    st.sets(st.integers(0, 7)),
    st.sets(st.integers(0, 7)),
    st.lists(st.integers(0, 7), unique=True),
    st.sets(st.integers(0, 9), min_size=1),
    st.integers(1, 20),
    st.lists(st.sampled_from((("a",), ("b",), ("a", "b"))), min_size=8, max_size=8),
)
def test_session_chain_metrics_and_source_loss_match_exhaustive_oracle(
    catalog_ids, candidate_ids, ranking_ids, target_ids, k, all_origins
):
    catalog = tuple(str(i) for i in sorted(catalog_ids))
    candidates = tuple(str(i) for i in sorted(candidate_ids & catalog_ids))
    delivered = tuple(str(i) for i in ranking_ids if str(i) in candidates)[:k]
    purchases = tuple(str(i) for i in sorted(target_ids))
    origins = {sku: all_origins[int(sku)] for sku in candidates}
    req = request()
    response = result(req, candidates, delivered, origins=origins)
    evaluation, details = evaluate_session(
        req, response, target(req, purchases), catalog_skus=catalog, k=k
    )
    expected = session_oracle(catalog, (), candidates, delivered, purchases, k)
    assert (evaluation.v_s, evaluation.u_s, evaluation.hit_s) == (
        expected["v"],
        expected["u"],
        expected["hit"],
    )
    assert tuple(details["losses"].values()) == expected["losses"]
    assert sum(details["losses"].values()) == 1 - evaluation.hit_s
    assert details["recall"] == expected["recall"]
    assert abs(details["ndcg"] - expected["ndcg"]) < 1e-14
    for source, row in details["sources"].items():
        assert row["delta_fixed"] == source_oracle(origins, purchases, source)


@seed(17)
@given(
    st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), max_size=256
    )
)
def test_streaming_quality_slices_and_overlap_preserve_fsum(values):
    req = request()
    evaluation, template = evaluate_session(
        req,
        result(req, ("b",), ("b",), origins={"b": ("a", "b")}),
        target(req),
        catalog_skus=("b",),
    )

    def rows():
        for index, value in enumerate(values):
            sid = f"sum-{index}"
            yield (
                replace(evaluation, session_id=sid),
                {
                    **template,
                    "session_id": sid,
                    "recall": value,
                    "ndcg": value,
                    "slice_metrics": {
                        "custom": {"hit": 1, "v": 1, "u": 1, "recall": value, "ndcg": value},
                    },
                    "overlaps": {"a|b": {"value": value, "reason": None}},
                },
            )

    report = aggregate_metrics(rows(), n_eligible=len(values))
    expected = math.fsum(values) / len(values) if values else None
    assert report["metrics"]["Recall@20"]["value"] == expected
    assert report["metrics"]["NDCG@20"]["value"] == expected
    if values:
        assert report["slices"]["custom"]["metrics"]["Recall@20"]["value"] == expected
        assert report["overlaps"]["a|b"]["value"] == expected
