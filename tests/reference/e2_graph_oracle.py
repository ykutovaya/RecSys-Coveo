"""Независимый полный перебор B04 и построение малых канонических примеров."""

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import TypeAlias

from coveo_recsys.contracts.historical import HistoricalConfig, HistoricalInput, Session
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent

EventSpec: TypeAlias = tuple[int, str, str | None]
EdgeValues: TypeAlias = tuple[float, int, float, float, float]
GraphOracle: TypeAlias = dict[str, dict[tuple[str, str], EdgeValues]]
PRODUCT_TYPES = {"detail", "add", "remove", "purchase"}


def make_input(
    specifications: Sequence[Sequence[EventSpec]], *, snapshot_time_ms: int = 10_000_000
) -> HistoricalInput:
    sessions: list[Session] = []
    for session_index, specification in enumerate(specifications):
        session_id = f"s{session_index}"
        groups: dict[int, dict[tuple[str, str | None], int]] = defaultdict(dict)
        for time_ms, kind, sku in specification:
            counts = groups[time_ms]
            counts[kind, sku] = counts.get((kind, sku), 0) + 1
        blocks = []
        for time_ms, counts in sorted(groups.items()):
            events = tuple(
                RawEvent(
                    "fixture",
                    session_id,
                    f"{session_id}:{time_ms}:{index}",
                    time_ms,
                    time_ms,
                    EventType(kind),
                    sku,
                    None,
                    "artificial",
                )
                for index, (kind, sku) in enumerate(counts)
            )
            blocks.append(
                CanonicalBlock(
                    f"{session_id}:{time_ms}",
                    session_id,
                    time_ms,
                    time_ms,
                    events,
                    tuple(
                        (event.event_id, count)
                        for event, count in zip(events, counts.values(), strict=True)
                    ),
                    (),
                )
            )
        sessions.append(tuple(blocks))
    return HistoricalInput(
        lambda: iter(sessions), len(sessions), snapshot_time_ms, "fixture", {}, "artificial"
    )


def brute_force_graph(history: HistoricalInput, config: HistoricalConfig) -> GraphOracle:
    """Перебор всех событий без ограниченного окна, порций и кода графовой ветви."""
    contributions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for session in history.iter_sessions():
        events = [
            event
            for block in session
            for event in block.events
            if event.event_type.value in PRODUCT_TYPES
            and event.event_time_ms < config.snapshot_time_ms
        ]
        group_times = sorted({event.event_time_ms for event in events})
        group_number = {time_ms: index for index, time_ms in enumerate(group_times)}
        delays: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for source in events:
            for target in events:
                delay = target.event_time_ms - source.event_time_ms
                distance = group_number[target.event_time_ms] - group_number[source.event_time_ms]
                if (
                    source.event_type.value == "detail"
                    and target.event_type.value in {"detail", "add", "purchase"}
                    and source.sku != target.sku
                    and 0 < delay <= config.graph_window_ms
                    and 0 < distance <= config.graph_max_group_distance
                ):
                    assert source.sku is not None and target.sku is not None
                    key = (f"detail_{target.event_type.value}", source.sku, target.sku)
                    delays[key].append(delay)
        for key, allowed_delays in delays.items():
            contributions[key].append(
                max(math.exp(-delay / config.graph_tau_ms) for delay in allowed_delays)
            )
    rows: dict[tuple[str, str], list[tuple[str, float, int]]] = defaultdict(list)
    for (relation, source, target), weights in contributions.items():
        if len(weights) >= config.graph_min_support:
            rows[relation, source].append((target, math.fsum(weights), len(weights)))
    result: GraphOracle = {name: {} for name in ("detail_detail", "detail_add", "detail_purchase")}
    for (relation, source), neighbors in rows.items():
        kept = sorted(neighbors, key=lambda row: (-row[1], row[0].encode("utf-8")))[
            : config.graph_max_neighbors
        ]
        total = math.fsum(weight for _, weight, _ in kept)
        for target, weight, support in kept:
            result[relation][source, target] = (
                weight,
                support,
                weight / total,
                total,
                total / (total + config.graph_regularization),
            )
    return result
