"""Точный отбор: 1000 примеров, seed 17, сжатие контрпримеров."""

import numpy as np
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.retrieval import retrieve, retrieve_fallback
from coveo_recsys.retrieval.scores import RetrievalIndex
from coveo_recsys.retrieval.sources import exact_top_m
from tests.reference.e3_cases import CATALOG_HASH, package, request
from tests.reference.e3_retrieval_oracle import retrieval_oracle
from tests.unit.test_e3_retrieval import context


@seed(17)
@given(
    st.lists(st.tuples(st.integers(-10, 10), st.booleans()), max_size=80),
    st.integers(0, 90),
    st.integers(1, 20),
)
def test_blocked_top_m_equals_complete_order_and_is_permutation_invariant(rows, quota, block):
    ids = tuple(f"товар-{i:03}" for i in range(len(rows)))
    values = np.array([x for x, _ in rows], dtype=np.float64)
    mask = np.array([m for _, m in rows], dtype=bool)
    expected = tuple(
        sorted(
            (i for i, (_, ok) in enumerate(rows) if ok),
            key=lambda i: (-rows[i][0], ids[i].encode("utf-8")),
        )
    )[:quota]
    actual = exact_top_m(ids, values, mask, quota, block_size=block)
    assert actual == expected
    reverse = exact_top_m(ids[::-1], values[::-1], mask[::-1], quota, block_size=block)
    assert tuple(ids[i] for i in actual) == tuple(ids[::-1][i] for i in reverse)


@seed(17)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(("detail", "add", "purchase")), st.sampled_from(("a", "b", "c", "я"))
        ),
        max_size=6,
    ),
    st.lists(st.integers(0, 1), min_size=8, max_size=8),
    st.one_of(st.none(), st.sampled_from(SOURCES)),
)
def test_source_union_and_reserve_match_independent_full_catalog(events, quotas, disabled):
    rows = ((2000, "detail", "a"),) + tuple(
        (3000 + i, action, sku) for i, (action, sku) in enumerate(events)
    )
    ctx = context(request(rows, query=(0.0, 1.0), search_time=rows[-1][0]))
    historical = package()
    idx = RetrievalIndex(historical, CATALOG_HASH)
    cfg = RetrievalConfig(quotas=tuple(quotas), max_candidates=max(1, sum(quotas)))
    runtime = {disabled: "disabled"} if disabled is not None else None
    expected = retrieval_oracle(ctx, historical, cfg, source_runtime=runtime)
    actual = retrieve(ctx, idx, cfg, runtime)
    assert actual.skus == expected["main"]
    assert dict(actual.origins) == expected["origins"]
    assert {r.source: r.skus for r in actual.sources} == expected["sources"]
    assert {r.source: r.available for r in actual.sources} == expected["available"]
    assert retrieve_fallback(ctx, idx, cfg).skus == expected["fallback"]
