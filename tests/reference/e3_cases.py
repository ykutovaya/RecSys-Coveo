"""Малый пакет и запросы Э3; не эталон расчёта баллов или признаков."""

from collections import defaultdict

import numpy as np

from coveo_recsys.contracts.historical import (
    RELATIONS,
    GraphBuildResult,
    GraphRelation,
    HistoricalCatalog,
    HistoricalContent,
    HistoricalPackage,
    HistoricalStatistics,
    Modality,
)
from coveo_recsys.contracts.preparation import PreparedRequest, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.state import build_prefix

CATALOG_HASH = "a" * 64


def package():
    skus = ("a", "b", "c", "я")
    graphs = []
    for relation, rows in zip(
        RELATIONS,
        (((0, 1, 4.0, 4), (0, 2, 2.0, 3), (1, 2, 1.0, 3)), ((0, 2, 3.0, 3),), ((0, 1, 5.0, 4),)),
        strict=True,
    ):
        indptr, indices, weights, support, sums = [0], [], [], [], []
        for i in range(4):
            selected = [edge for edge in rows if edge[0] == i]
            indices.extend(edge[1] for edge in selected)
            weights.extend(edge[2] for edge in selected)
            support.extend(edge[3] for edge in selected)
            sums.append(sum(edge[2] for edge in selected))
            indptr.append(len(indices))
        probabilities = [w / sums[i] for i in range(4) for w in weights[indptr[i] : indptr[i + 1]]]
        graphs.append(
            GraphRelation(
                relation,
                np.array(indptr, dtype=np.int64),
                np.array(indices, dtype=np.int64),
                np.array(weights),
                np.array(probabilities),
                np.array(support, dtype=np.int64),
                np.array(sums),
                np.array([s / (s + 10) for s in sums]),
            )
        )
    content = HistoricalContent(
        Modality(
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]]),
            np.array([True, True, True, False]),
            (None, None, None, "vector_absent"),
        ),
        Modality(
            np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]),
            np.ones(4, dtype=bool),
            (None,) * 4,
        ),
        ("root/a", "root/b"),
        np.array([1, 1, 2, 0], dtype=np.int64),
        np.array([1, 2, 3, 0], dtype=np.int64),
        np.array([True, True, True, False]),
    )
    return HistoricalPackage(
        HistoricalCatalog(skus),
        GraphBuildResult(tuple(graphs), {}),
        HistoricalStatistics(
            np.array([8, 6, 3, 0]),
            np.array([3, 2, 1, 0]),
            np.array([9 / 21, 7 / 21, 4 / 21, 1 / 21]),
            10,
        ),
        content,
        1000,
        {"fixture": "b" * 64, "metadata_sha256": "b" * 64, "metadata_profile_sha256": "c" * 64},
        {
            "description_dimension": 2,
            "image_dimension": 2,
            "query_description_compatibility": "publisher_statement",
            "static_availability": "unknown",
        },
    )


def request(
    rows=((2000, "detail", "a"),),
    *,
    semantics="set_v1",
    query=None,
    search_time=None,
    time=None,
    available_time=None,
):
    grouped = defaultdict(list)
    for i, (timestamp, action, sku) in enumerate(rows):
        grouped[timestamp].append(
            RawEvent(
                "fixture",
                "s",
                str(i),
                timestamp,
                timestamp,
                EventType(action),
                sku,
                None,
                "fixture",
            )
        )
    if search_time is not None:
        grouped[search_time].append(
            RawEvent(
                "fixture",
                "s",
                "search",
                search_time,
                search_time,
                EventType.SEARCH,
                None,
                SearchPayload(query, None if query is not None else "vector_absent"),
                "fixture",
            )
        )
    blocks = []
    for timestamp, events in sorted(grouped.items()):
        unique = {}
        counts = defaultdict(int)
        for event in events:
            key = (event.event_type, event.sku)
            unique.setdefault(key, event)
            counts[unique[key].event_id] += 1
        payloads = tuple(e.search for e in unique.values() if e.event_type == EventType.SEARCH)
        multiplicities = tuple(
            counts[e.event_id] for e in unique.values() if e.event_type == EventType.SEARCH
        )
        blocks.append(
            CanonicalBlock(
                str(timestamp),
                "s",
                timestamp,
                timestamp,
                tuple(unique.values()),
                tuple(counts.items()),
                payloads,
                multiplicities,
            )
        )
    tq = max(grouped) if time is None else time
    aq = tq if available_time is None else available_time
    prefix = build_prefix(
        blocks,
        event_cutoff_ms=tq,
        availability_cutoff_ms=aq,
        config=TaskConfig(event_semantics=semantics),
        source_namespace="fixture",
        session_id="s",
    )
    trigger = next(b.block_id for b in prefix.blocks if b.event_time_ms == tq)
    return PreparedRequest("q", "s", trigger, tq, aq, 1, prefix)
