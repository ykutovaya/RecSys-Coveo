"""Искусственные байты и независимые эталоны границ, без проверяемого кода."""

import hashlib
import json


def event(identifier="e0", *, time_ms=1000, kind="detail", sku="A", **fields):
    return {
        "source_namespace": "source",
        "session_id": "session",
        "event_id": identifier,
        "event_time_ms": time_ms,
        "event_type": kind,
        "sku": sku,
        **fields,
    }


def packet(events=None, *, overrides=None, compact=True):
    events = [event()] if events is None else events
    separators = (",", ":") if compact else (", ", ": ")
    payload = json.dumps(events, ensure_ascii=False, separators=separators, allow_nan=False).encode(
        "utf-8"
    )
    fields = {
        "source_namespace": "source",
        "session_id": "session",
        "event_time_ms": 1000,
        "block_id": "block",
        "event_count": len(events),
        "event_ids": [e.get("event_id") for e in events],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "closed": True,
    }
    fields.update(overrides or {})
    header = json.dumps(fields, ensure_ascii=False, separators=separators, allow_nan=False).encode(
        "utf-8"
    )
    return header[:-1] + b',"events":' + payload + b"}", payload


def expected_time_reason(time_ms, decision_ms, calendar, late_ms=1_800_000):
    # Независимо проверяем включение каждого конца интервалов.
    if time_ms < calendar[0] or time_ms >= calendar[1]:
        return "invalid_input"
    if time_ms > decision_ms:
        return "timestamp_future"
    if decision_ms > time_ms + late_ms:
        return "timestamp_late"
    return None


def expected_transport_count(count, maximum=100):
    return 0 < count <= maximum
