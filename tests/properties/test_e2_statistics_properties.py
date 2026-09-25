"""Hypothesis 6.131.9: 1000 примеров, seed 17, сохранение минимального контрпримера."""

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts.historical import HistoricalCatalog, HistoricalConfig
from coveo_recsys.historical.statistics import build_statistics
from tests.reference.e2_cases import history
from tests.reference.e2_statistics_oracle import statistics

ATOL, RTOL = 1e-10, 1e-8
NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
ROW = st.tuples(
    st.sampled_from(["s1", "s2", "s3", "s4"]),
    st.integers(0, 100_000_000),
    st.sampled_from(["detail", "purchase", "add", "remove"]),
    st.sampled_from(["a", "b", "я"]),
)


@NORMATIVE
@seed(17)
@given(st.lists(ROW, max_size=40), st.sampled_from([1e-10, 1.0, 13.0, 1e20]))
def test_session_pairs_oracle_and_invariance_to_order_and_repeated_events(rows, eta):
    config = HistoricalConfig(snapshot_time_ms=200_000_000, popularity_smoothing=eta)
    catalog = HistoricalCatalog(("a", "b", "я"))
    expected = statistics(rows, catalog.skus, eta)
    source = history(rows, snapshot=config.snapshot_time_ms)
    actual = build_statistics(source, catalog, config)
    assert actual.detail_counts.tolist() == expected[0]
    assert actual.purchase_counts.tolist() == expected[1]
    assert actual.session_count == expected[3]
    np.testing.assert_allclose(actual.popularity, expected[2], atol=ATOL, rtol=RTOL)
    assert np.isclose(np.sum(actual.popularity), 1.0, atol=ATOL, rtol=RTOL)
    assert np.all(actual.popularity > 0)
    changed = build_statistics(
        history(list(reversed(rows)) + rows, snapshot=config.snapshot_time_ms), catalog, config
    )
    np.testing.assert_array_equal(actual.detail_counts, changed.detail_counts)
    np.testing.assert_array_equal(actual.purchase_counts, changed.purchase_counts)
    np.testing.assert_array_equal(actual.popularity, changed.popularity)
    assert actual.session_count == changed.session_count
