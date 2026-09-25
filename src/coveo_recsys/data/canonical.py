"""Двухпроходная внешняя сортировка: транспортные повторы, затем полные группы."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import tempfile
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.numerics import checked_add


def _search_key(value: SearchPayload | None) -> tuple[Any, ...]:
    if value is None:
        return ("absent",)
    return (value.reason or "", tuple(0.0 if x == 0 else x for x in (value.vector or ())))


def _meaning(event: RawEvent) -> tuple[Any, ...]:
    return (
        event.source_namespace,
        event.session_id,
        event.event_time_ms,
        event.event_type.value,
        event.sku,
        _search_key(event.search),
    )


def _identity(event: RawEvent) -> tuple[Any, ...]:
    return (
        id_bytes(event.source_namespace),
        event.transport_id is None,
        id_bytes(event.transport_id or event.event_id),
    )


def _time_key(event: RawEvent) -> tuple[Any, ...]:
    return (
        id_bytes(event.source_namespace),
        id_bytes(event.session_id),
        event.event_time_ms,
        event.event_type.value,
        (event.sku or "").encode("utf-8"),
        event.available_time_ms,
        event.event_id.encode("utf-8"),
    )


def _external_sort(
    events: Iterable[RawEvent],
    key: Callable[[RawEvent], tuple[Any, ...]],
    directory: Path,
    *,
    chunk_size: int,
    chunk_bytes: int,
    fan_in: int,
) -> Generator[RawEvent, None, None]:
    directory.mkdir()
    chunks: list[Path] = []
    batch: list[RawEvent] = []
    size = 0

    def write_chunk() -> None:
        path = directory / f"chunk-{len(chunks)}.jsonl"
        with path.open("wb") as stream:
            for event in sorted(batch, key=key):
                stream.write(to_json(event) + b"\n")
        chunks.append(path)

    for event in events:
        event_bytes = len(to_json(event))
        if event_bytes > chunk_bytes:
            raise InputValidationError("Одна строка превышает бюджет внешней сортировки")
        if batch and (len(batch) >= chunk_size or size + event_bytes > chunk_bytes):
            write_chunk()
            batch, size = [], 0
        batch.append(event)
        size += event_bytes
    if batch:
        write_chunk()
    generation = 0
    while len(chunks) > 1:
        merged = []
        for offset in range(0, len(chunks), fan_in):
            path = directory / f"merge-{generation}-{offset}.jsonl"
            sources = chunks[offset : offset + fan_in]
            with ExitStack() as stack, path.open("wb") as output:
                streams = [stack.enter_context(p.open("rb")) for p in sources]
                iterators = [(from_json(line, RawEvent) for line in stream) for stream in streams]
                for event in heapq.merge(*iterators, key=key):
                    output.write(to_json(event) + b"\n")
            for source in sources:
                source.unlink()
            merged.append(path)
        chunks = merged
        generation += 1
    if chunks:
        with chunks[0].open("rb") as stream:
            for line in stream:
                yield from_json(line, RawEvent)


def _deduplicate(
    events: Iterable[RawEvent], max_group_bytes: int
) -> Generator[RawEvent, None, None]:
    for _, group in itertools.groupby(events, key=_identity):
        first = next(group)
        if first.transport_id is None:
            yield first
            yield from group
            continue
        origins = set(first.origins or (first.provenance,))
        available = first.available_time_ms
        origin_bytes = sum(len(x.encode("utf-8")) for x in origins)
        for repeated in group:
            if _meaning(first) != _meaning(repeated):
                raise InputValidationError("Один транспортный ID соответствует разным событиям")
            available = min(available, repeated.available_time_ms)
            for origin in repeated.origins or (repeated.provenance,):
                if origin not in origins:
                    origin_bytes += len(origin.encode("utf-8"))
                    if origin_bytes > max_group_bytes:
                        raise InputValidationError("Происхождение повторов превышает бюджет")
                    origins.add(origin)
        ordered = tuple(sorted(origins, key=lambda x: x.encode("utf-8")))
        yield replace(first, available_time_ms=available, provenance=ordered[0], origins=ordered)


def _make_block(rows: list[RawEvent]) -> CanonicalBlock:
    first = rows[0]
    groups: dict[tuple[str, str | None], list[RawEvent]] = {}
    for row in rows:
        groups.setdefault((row.event_type.value, row.sku), []).append(row)
    events, multiplicities = [], []
    search_values: dict[tuple[Any, ...], tuple[SearchPayload, int]] = {}
    for (kind, sku), values in sorted(
        groups.items(), key=lambda kv: (kv[0][0], (kv[0][1] or "").encode())
    ):
        payload = None
        if kind == EventType.SEARCH:
            for row in values:
                value = row.search or SearchPayload(None, "vector_absent")
                key = _search_key(value)
                old_count = search_values.get(key, (value, 0))[1]
                search_values[key] = (value, checked_add(old_count, 1))
            payload = (
                next(iter(search_values.values()))[0]
                if len(search_values) == 1
                else SearchPayload(None, "search_ambiguous")
            )
        logical_id = hashlib.sha256(
            canonical_json_bytes(
                [
                    first.source_namespace,
                    first.session_id,
                    first.event_time_ms,
                    kind,
                    sku,
                ]
            )
        ).hexdigest()
        origins = tuple(
            sorted(
                {origin for row in values for origin in (row.origins or (row.provenance,))},
                key=lambda x: x.encode("utf-8"),
            )
        )
        events.append(
            RawEvent(
                first.source_namespace,
                first.session_id,
                logical_id,
                first.event_time_ms,
                max(v.available_time_ms for v in values),
                EventType(kind),
                sku,
                payload,
                origins[0],
                None,
                origins,
            )
        )
        multiplicities.append((logical_id, require_int64(len(values))))
    searches = sorted(search_values.items(), key=lambda x: x[0])
    payloads = tuple(v[0] for _, v in searches)
    counts = tuple(v[1] for _, v in searches)
    available = max(v.available_time_ms for v in rows)
    semantic = {
        "namespace": first.source_namespace,
        "session": first.session_id,
        "time": first.event_time_ms,
        "available": available,
        "events": [(_meaning(e), n) for e, (_, n) in zip(events, multiplicities, strict=True)],
        "search": [(_search_key(p), n) for p, n in zip(payloads, counts, strict=True)],
    }
    block_id = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return CanonicalBlock(
        block_id,
        first.session_id,
        first.event_time_ms,
        available,
        tuple(events),
        tuple(multiplicities),
        payloads,
        counts,
    )


def canonicalize(
    events: Iterable[RawEvent],
    *,
    temp_dir: Path | None = None,
    chunk_size: int = 4096,
    chunk_bytes: int = 16_777_216,
    fan_in: int = 32,
    max_group_events: int = 100_000,
    max_group_bytes: int = 67_108_864,
) -> Generator[CanonicalBlock, None, None]:
    for value in (chunk_size, chunk_bytes, fan_in, max_group_events, max_group_bytes):
        require_int64(value)
        if value <= 0:
            raise InputValidationError("Бюджеты сортировки должны быть положительными")
    if fan_in < 2:
        raise InputValidationError("Нужно не менее двух входов слияния")
    with (
        tempfile.TemporaryDirectory(prefix="coveo-sort-", dir=temp_dir) as directory,
        ExitStack() as resources,
    ):
        root = Path(directory)
        identity_order = _external_sort(
            events,
            _identity,
            root / "identity",
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            fan_in=fan_in,
        )
        resources.callback(identity_order.close)
        deduplicated = _deduplicate(identity_order, max_group_bytes)
        resources.callback(deduplicated.close)
        time_order = _external_sort(
            deduplicated,
            _time_key,
            root / "time",
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            fan_in=fan_in,
        )
        resources.callback(time_order.close)
        for _, group in itertools.groupby(
            time_order,
            key=lambda e: (
                e.source_namespace,
                e.session_id,
                e.event_time_ms,
            ),
        ):
            rows: list[RawEvent] = []
            size = 0
            for event in group:
                size += len(to_json(event))
                if len(rows) >= max_group_events or size > max_group_bytes:
                    raise InputValidationError(
                        "Полная временная группа превышает объявленный бюджет"
                    )
                rows.append(event)
            yield _make_block(rows)
