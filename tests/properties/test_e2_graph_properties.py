"""B04: полный перебор и независимость результата от порядка и порций."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts.historical import HistoricalCatalog, HistoricalConfig
from coveo_recsys.historical.graph import build_graph
from tests.reference.e2_graph_oracle import brute_force_graph, make_input


@st.composite
def histories(draw):
    event = st.tuples(
        st.integers(0, 1000),
        st.sampled_from(["detail", "add", "remove", "purchase"]),
        st.sampled_from(["a", "b", "é", "я"]),
    )
    sessions = draw(st.lists(st.lists(event, max_size=9), max_size=5))
    config = HistoricalConfig(
        snapshot_time_ms=800,
        graph_min_support=draw(st.integers(1, 3)),
        graph_max_group_distance=draw(st.integers(1, 4)),
        graph_window_ms=draw(st.integers(1, 1000)),
        graph_max_neighbors=draw(st.integers(1, 3)),
        buffer_records=3,
        merge_fan_in=2,
    )
    return sessions, config


@settings(max_examples=1000, deadline=None, database=None, print_blob=True)
@seed(17)
@given(histories())
def test_graph_matches_oracle_and_is_invariant_to_permutations_and_chunks(case):
    specifications, config = case
    history = make_input(specifications, snapshot_time_ms=config.snapshot_time_ms)
    catalog = HistoricalCatalog(
        tuple(
            sorted(
                {sku for session in specifications for _, _, sku in session},
                key=lambda sku: sku.encode("utf-8"),
            )
        )
    )
    expected = brute_force_graph(history, config)
    permuted = make_input(
        [list(reversed(session)) for session in reversed(specifications)],
        snapshot_time_ms=config.snapshot_time_ms,
    )
    with TemporaryDirectory() as directory:
        first = build_graph(history, catalog, config, work_dir=Path(directory))
        second = build_graph(
            permuted,
            catalog,
            replace(config, buffer_records=17, merge_fan_in=3),
            work_dir=Path(directory),
        )
    for graph, reordered in zip(first.relations, second.relations, strict=True):
        for name in ("indptr", "indices", "weights", "probabilities", "support", "row_sums", "rho"):
            np.testing.assert_array_equal(getattr(graph, name), getattr(reordered, name))
        actual_edges = {}
        for source_index, source in enumerate(catalog.skus):
            start, end = graph.indptr[source_index : source_index + 2]
            assert end - start <= config.graph_max_neighbors
            for edge in range(start, end):
                target = catalog.skus[graph.indices[edge]]
                assert source != target
                assert 0 < graph.weights[edge] <= graph.support[edge]
                actual_edges[source, target] = (
                    graph.weights[edge],
                    graph.support[edge],
                    graph.probabilities[edge],
                    graph.row_sums[source_index],
                    graph.rho[source_index],
                )
            if start == end:
                assert graph.row_sums[source_index] == graph.rho[source_index] == 0
            else:
                assert abs(sum(graph.probabilities[start:end]) - 1) < 1e-12
                np.testing.assert_allclose(
                    graph.rho[source_index] * graph.probabilities[start:end],
                    graph.weights[start:end]
                    / (graph.row_sums[source_index] + config.graph_regularization),
                    rtol=1e-12,
                    atol=1e-15,
                )
        assert actual_edges.keys() == expected[graph.relation].keys()
        for key, values in actual_edges.items():
            np.testing.assert_allclose(
                values, expected[graph.relation][key], rtol=1e-12, atol=1e-15
            )
