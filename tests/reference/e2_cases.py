"""Малые полные истории для тестов Э2; не эталон вычислений."""

from collections import defaultdict

from coveo_recsys.contracts.historical import HistoricalInput, MetadataRecord, MetadataSource
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent


def history(rows, *, snapshot=10_000_000):
    groups = defaultdict(list)
    for i, (sid, timestamp, action, sku) in enumerate(rows):
        groups[sid, timestamp].append(
            RawEvent(
                "fixture",
                sid,
                str(i),
                timestamp,
                timestamp,
                EventType(action),
                sku,
                None,
                "fixture",
            )
        )
    sessions = defaultdict(list)
    for (sid, timestamp), events in sorted(groups.items()):
        unique = {(e.event_type, e.sku): e for e in events}
        selected = tuple(unique.values())
        sessions[sid].append(
            CanonicalBlock(
                f"{sid}:{timestamp}",
                sid,
                timestamp,
                timestamp,
                selected,
                tuple((e.event_id, 1) for e in selected),
                (),
            )
        )
    values = tuple(tuple(blocks) for blocks in sessions.values())
    return HistoricalInput(
        lambda: iter(values), len(values), snapshot, "fixture", {"fixture": "a" * 64}, "synthetic"
    )


def metadata():
    rows = {
        "a": MetadataRecord("a", [3.0, 4.0], [1.0, 0.0], "category", 3),
        "b": MetadataRecord("b", None, [0.0, 1.0], "category", None),
    }
    return MetadataSource(
        lambda skus: (rows[sku] for sku in skus if sku in rows),
        2,
        2,
        {"metadata": "b" * 64},
        "publisher_statement",
    )
