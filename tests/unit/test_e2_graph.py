"""Правила B04: направление, поддержка, разреживание и пределы ресурсов."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.historical import HistoricalCatalog, HistoricalConfig
from coveo_recsys.historical.graph import build_graph
from tests.reference.e2_graph_oracle import make_input


def catalog_of(history):
    return HistoricalCatalog(
        tuple(
            sorted(
                {
                    event.sku
                    for session in history.iter_sessions()
                    for block in session
                    for event in block.events
                    if event.sku is not None
                },
                key=lambda sku: sku.encode("utf-8"),
            )
        )
    )


def compute(tmp_path, specifications, **settings):
    config = HistoricalConfig(snapshot_time_ms=10_000_000, graph_min_support=1, **settings)
    history = make_input(specifications, snapshot_time_ms=config.snapshot_time_ms)
    catalog = catalog_of(history)
    return build_graph(history, catalog, config, work_dir=tmp_path), catalog


def row(result, catalog, source, relation="detail_detail"):
    graph = next(graph for graph in result.relations if graph.relation == relation)
    index = catalog.skus.index(source)
    start, end = graph.indptr[index : index + 2]
    return graph, index, slice(start, end)


@pytest.mark.parametrize("kind", ["detail", "add", "purchase"])
@pytest.mark.parametrize("delay,expected", [(0, 0), (1, 1), (1_800_000, 1), (1_800_001, 0)])
def test_direction_and_time_boundaries(tmp_path, kind, delay, expected):
    result, catalog = compute(tmp_path, [[(0, "detail", "a"), (delay, kind, "b")]])
    graph, _, selected = row(result, catalog, "a", f"detail_{kind}")
    assert graph.indices[selected].size == expected
    assert sum(len(item.indices) for item in result.relations) == expected
    if expected:
        assert graph.weights[selected].tolist() == [math.exp(-delay / 600_000)]


@pytest.mark.parametrize("distance,expected", [(1, 1), (10, 1), (11, 0)])
def test_product_groups_without_detail_count_toward_distance(tmp_path, distance, expected):
    specification = [(0, "detail", "a")]
    specification.extend((index, "remove", "x") for index in range(1, distance))
    specification.append((distance, "purchase", "b"))
    result, catalog = compute(tmp_path, [specification])
    graph, _, selected = row(result, catalog, "a", "detail_purchase")
    assert graph.indices[selected].size == expected


def test_nonproduct_groups_do_not_change_distance(tmp_path):
    specification = [(0, "detail", "a"), (30, "add", "b")]
    specification.extend((index, "page", None) for index in range(1, 30))
    result, catalog = compute(tmp_path, [specification])
    graph, _, selected = row(result, catalog, "a", "detail_add")
    assert graph.support[selected].tolist() == [1]


def test_loops_are_excluded_for_every_relation(tmp_path):
    result, _ = compute(
        tmp_path, [[(0, "detail", "a"), (1, "detail", "a"), (2, "add", "a"), (3, "purchase", "a")]]
    )
    assert all(len(graph.indices) == 0 for graph in result.relations)


@pytest.mark.parametrize("target_time,expected", [(9, 1), (10, 0), (11, 0)])
def test_snapshot_is_strict_and_future_events_are_ignored(tmp_path, target_time, expected):
    config = HistoricalConfig(snapshot_time_ms=10, graph_min_support=1)
    history = make_input([[(0, "detail", "a"), (target_time, "add", "b")]], snapshot_time_ms=10)
    result = build_graph(history, HistoricalCatalog(("a", "b")), config, work_dir=tmp_path)
    assert len(result.relations[1].indices) == expected


@pytest.mark.parametrize("session_count", [2, 3, 4])
def test_maximum_contribution_and_support_count_distinct_sessions(tmp_path, session_count):
    history = make_input(
        [
            [(0, "detail", "a"), (10, "add", "b"), (20, "detail", "a"), (21, "add", "b")]
            for _ in range(session_count)
        ]
        + [[(0, "detail", "a"), (1, "add", "c")]] * 2
    )
    catalog = catalog_of(history)
    config = HistoricalConfig(snapshot_time_ms=history.snapshot_time_ms, graph_max_neighbors=1)
    result = build_graph(history, catalog, config, work_dir=tmp_path)
    graph, _, selected = row(result, catalog, "a", "detail_add")
    expected = int(session_count >= 3)
    assert graph.indices[selected].size == expected
    if expected:
        assert catalog.skus[graph.indices[selected][0]] == "b"
        assert graph.support[selected].tolist() == [session_count]
        assert graph.weights[selected][0] == math.fsum(
            [math.exp(-1 / config.graph_tau_ms)] * session_count
        )


def test_support_threshold_precedes_neighbor_pruning(tmp_path):
    history = make_input(
        [[(0, "detail", "a"), (600_000, "add", "supported")]] * 3
        + [[(0, "detail", "a"), (1, "add", "heavier")]] * 2
    )
    catalog = catalog_of(history)
    config = HistoricalConfig(history.snapshot_time_ms, graph_max_neighbors=1)
    result = build_graph(history, catalog, config, work_dir=tmp_path)
    graph, _, selected = row(result, catalog, "a", "detail_add")
    assert [catalog.skus[index] for index in graph.indices[selected]] == ["supported"]


@pytest.mark.parametrize("neighbor_count", [99, 100, 101])
def test_top_100_ties_follow_utf8_and_normalize_kept_weights(tmp_path, neighbor_count):
    neighbors = [f"item{index:03d}" for index in range(neighbor_count - 2)] + ["é", "я"]
    specification = [(0, "detail", "source")] + [(1, "add", sku) for sku in reversed(neighbors)]
    result, catalog = compute(tmp_path, [specification] * 3)
    graph, index, selected = row(result, catalog, "source", "detail_add")
    expected = sorted(neighbors, key=lambda sku: sku.encode("utf-8"))[:100]
    assert [catalog.skus[item] for item in graph.indices[selected]] == expected
    total = math.fsum([3 * math.exp(-1 / 600_000)] * len(expected))
    assert graph.row_sums[index] == total
    np.testing.assert_allclose(graph.probabilities[selected], 1 / len(expected), atol=1e-15)
    assert graph.rho[index] == total / (total + 10)
    assert result.metrics["relations"]["detail_add"]["edges_after_support"] == neighbor_count
    assert result.metrics["relations"]["detail_add"]["edges_after_pruning"] == len(expected)


def test_empty_rows_and_empty_catalog_are_zero(tmp_path):
    config = HistoricalConfig(snapshot_time_ms=10)
    for catalog in (HistoricalCatalog(()), HistoricalCatalog(("only",))):
        result = build_graph(
            make_input([], snapshot_time_ms=10), catalog, config, work_dir=tmp_path
        )
        for graph in result.relations:
            assert graph.indptr.tolist() == [0] * (len(catalog.skus) + 1)
            assert graph.indices.size == graph.weights.size == graph.support.size == 0
            assert graph.row_sums.tolist() == graph.rho.tolist() == [0.0] * len(catalog.skus)


def test_underflow_rejects_existing_support_before_filtering(tmp_path):
    with localcontext() as context:
        context.prec = 80
        assert (-Decimal(800)).exp() > 0
    history = make_input([[(0, "detail", "a"), (800, "add", "b")]])
    config = HistoricalConfig(snapshot_time_ms=history.snapshot_time_ms, graph_tau_ms=1)
    with pytest.raises(NumericError, match="затухан|FP64|нул"):
        build_graph(history, catalog_of(history), config, work_dir=tmp_path)


@pytest.mark.parametrize(
    "limit,value",
    [("max_pairs", 2), ("max_group_size", 2), ("max_temp_bytes", 1), ("max_result_bytes", 1)],
)
def test_resource_limit_fails_without_truncating_group(tmp_path, limit, value):
    history = make_input([[(0, "detail", "a")] + [(1, "add", str(i)) for i in range(3)]])
    config = replace(HistoricalConfig(snapshot_time_ms=history.snapshot_time_ms), **{limit: value})
    with pytest.raises(InputValidationError, match="предел|лимит|бюджет"):
        build_graph(history, catalog_of(history), config, work_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("after_spill", [False, True])
def test_external_budget_callback_is_called_and_exception_propagates(tmp_path, after_spill):
    def deny():
        if not after_spill or list(tmp_path.rglob("*.bin")):
            raise RuntimeError("time budget exhausted")

    history = make_input([[(0, "detail", "a"), (1, "add", "b")]])
    with pytest.raises(RuntimeError, match="time budget exhausted"):
        build_graph(
            history,
            catalog_of(history),
            HistoricalConfig(history.snapshot_time_ms),
            work_dir=tmp_path,
            check_budget=deny,
        )
    assert list(tmp_path.iterdir()) == []


def test_snapshot_config_must_match_verified_input(tmp_path):
    history = make_input([[(0, "detail", "a")]], snapshot_time_ms=10)
    with pytest.raises(InputValidationError, match="срез|сним|границ"):
        build_graph(history, catalog_of(history), HistoricalConfig(11), work_dir=tmp_path)


def test_catalog_does_not_silently_lose_historical_sku(tmp_path):
    history = make_input([[(0, "detail", "a"), (1, "add", "missing")]])
    with pytest.raises(InputValidationError, match="каталог"):
        build_graph(
            history,
            HistoricalCatalog(("a",)),
            HistoricalConfig(history.snapshot_time_ms),
            work_dir=tmp_path,
        )


def test_temporary_storage_and_metrics_are_bounded(tmp_path):
    history = make_input([[(0, "detail", "a")] + [(1, "add", str(i)) for i in range(30)]] * 3)
    config = HistoricalConfig(history.snapshot_time_ms, buffer_records=2, merge_fan_in=2)
    result = build_graph(history, catalog_of(history), config, work_dir=tmp_path)
    assert result.metrics["candidate_pairs"] == 90
    assert result.metrics["max_group_size"] == 30
    assert 0 < result.metrics["temp_peak_bytes"] <= config.max_temp_bytes
    assert result.metrics["buffer_peak_records"] <= config.buffer_records
    assert result.metrics["merge_peak_readers"] <= config.merge_fan_in
    assert list(tmp_path.iterdir()) == []
