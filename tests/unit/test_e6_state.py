"""Приём проверяется по транспортным байтам, не через готовые переходы ядра."""

import hashlib
import json
from dataclasses import replace

import pytest

from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.service_state import StateError
from coveo_recsys.state.service import IntakeService
from coveo_recsys.state.store import FileStateStore, MemoryStateStore


def packet(time=1000, *, block="b1", namespace="n", session="s", events=None):
    rows = events if events is not None else [("e1", "detail", "A")]
    body = [
        {
            "source_namespace": namespace,
            "session_id": session,
            "event_time_ms": time,
            "event_id": eid,
            "event_type": kind,
            "sku": sku,
        }
        for eid, kind, sku in rows
    ]
    payload = json.dumps(body, separators=(",", ":")).encode()
    header = {
        "source_namespace": namespace,
        "session_id": session,
        "event_time_ms": time,
        "block_id": block,
        "event_count": len(body),
        "event_ids": [row[0] for row in rows],
        "closed": True,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return json.dumps(header, separators=(",", ":")).encode()[:-1] + b',"events":' + payload + b"}"


def service(store=None):
    return IntakeService(
        store or MemoryStateStore(), TaskConfig(), {"secret": "n", "other": "m"}, 0, 10_000_000
    )


def accept(app, wire, now=1000, *, complete=True, token="secret"):
    return app.accept(
        wire, token=token, now_ms=now, check_deadline=lambda: None, stream_complete=complete
    )


def prefix(app, block="b1", *, at=1000, namespace="n"):
    return app.prefix(namespace, "s", block, a_q=at, check_deadline=lambda: None)


def test_acceptance_is_atomic_receipt_and_transport_mapping():
    app = service()
    result = accept(app, packet())
    assert result.status == "accepted" and result.reason is None
    receipt = result.receipt
    assert receipt.event_time_ms == receipt.decision_time_ms == 1000
    assert receipt.block_id == "b1" and receipt.canonical_block_id != "b1"
    request = prefix(app)
    assert request.trigger_block_id == receipt.canonical_block_id
    assert request.prefix.source_namespace == "n"
    assert request.t_q == request.a_q == 1000 and request.ordinal == 1
    assert app.receipt("n", "s", "b1") == receipt
    assert app.namespace_for("secret") == "n"


def test_confirmed_repeat_after_lateness_keeps_state_and_original_availability():
    app = service()
    first = accept(app, packet())
    before = app.store.read("n", "s")
    repeat = accept(app, packet(), now=2_000_000)
    assert repeat.status == "duplicate" and repeat.receipt == first.receipt
    assert repeat.decision_time_ms == 2_000_000
    assert app.store.read("n", "s") == before


def test_header_formatting_is_not_payload_conflict():
    app = service()
    wire = packet()
    accept(app, wire)
    repeat = accept(app, b" \n" + wire + b"  ", now=2_000_000)
    assert repeat.status == "duplicate"


def test_namespaces_and_unauthorized_sender_are_isolated():
    app = service()
    accept(app, packet())
    accept(app, packet(namespace="m"), token="other")
    before = app.store.read("n", "s")
    result = accept(app, packet(block="evil"), token="other")
    assert result.status == "rejected" and result.reason == "namespace_forbidden"
    assert app.store.read("n", "s") == before
    assert prefix(app, namespace="m").prefix.source_namespace == "m"
    with pytest.raises(StateError, match="unauthorized"):
        app.namespace_for("bad")


@pytest.mark.parametrize(
    "wire",
    [packet(events=[("e2", "detail", "B")]), packet(block="b2", events=[("e2", "detail", "B")])],
)
def test_closed_time_conflict_preserves_history_and_quarantines(wire):
    app = service()
    accept(app, packet())
    result = accept(app, wire)
    assert result.reason == "closed_block_conflict" and result.receipt is None
    state = app.store.read("n", "s")
    assert len(state.blocks) == 1 and state.quarantine
    with pytest.raises(StateError, match="state_incomplete"):
        prefix(app)


@pytest.mark.parametrize("excess", [-1, 0, 1], ids=["below-limit", "at-limit", "above-limit"])
def test_quarantine_total_utf8_bytes_are_bounded_without_losing_history(excess):
    app = service()
    accept(app, packet())
    accepted = app.store.read("n", "s")
    limit = 1_048_576
    wires = []
    for index, size in enumerate((limit // 2, limit // 2 + excess)):
        wire = packet(block=f"bad{index}", events=[(f"bad{index}", "detail", "B")])
        wire = wire.replace(f'"bad{index}"'.encode(), f'"плохой{index}"'.encode())
        # Исходные байты нагрузки изменились; отказ не должен сохранить сверх лимита.
        wire = b" " * (size - len(wire)) + wire
        wires.append(wire)
        assert accept(app, wire).reason == "invalid_input"
    state = app.store.read("n", "s")
    assert state.quarantine == tuple(w.decode() for w in wires[: 1 if excess > 0 else 2])
    assert sum(len(text.encode("utf-8")) for text in state.quarantine) <= limit
    assert state.blocks == accepted.blocks and state.snapshot == accepted.snapshot
    assert state.last_arrival_ms == accepted.last_arrival_ms
    assert state.gap_reasons == ("invalid_input",) and not state.stream_complete
    assert accept(app, wires[-1]).reason == "invalid_input"
    assert app.store.read("n", "s").quarantine == state.quarantine
    with pytest.raises(StateError, match="state_incomplete"):
        prefix(app)


def test_quarantine_stays_bounded_across_unique_errors_and_rate_rejections():
    app = service()
    accepted_wire = packet(events=[(f"e{i}", "detail", "A") for i in range(100)])
    assert accept(app, accepted_wire).status == "accepted"
    before = app.store.read("n", "s")
    limit = 1_048_576
    for index in range(8):
        wire = packet(block=f"bad{index}", events=[(f"bad{index}", "detail", "B")])
        reason = "rate_rejected"
        if index % 2:
            wire = wire.replace(b'"closed":true', b'"closed":false')
            reason = "invalid_input"
        wire = b" " * (limit // 2 - len(wire)) + wire
        assert accept(app, wire).reason == reason
        assert accept(app, wire).reason == reason
        state = app.store.read("n", "s")
        assert sum(len(text.encode("utf-8")) for text in state.quarantine) <= limit
        assert state.blocks == before.blocks and state.snapshot == before.snapshot
        assert not state.stream_complete
    assert len(state.quarantine) == 2
    assert state.gap_reasons == ("invalid_input", "rate_rejected")
    repeat = accept(app, accepted_wire)
    assert repeat.reason == "duplicate_rate_rejected"
    assert app.store.read("n", "s") == state


def test_known_events_under_new_block_id_do_not_create_block():
    app = service()
    accept(app, packet())
    before = app.store.read("n", "s")
    result = accept(app, packet(block="alias"))
    assert result.status == "duplicate" and result.reason == "known_events"
    assert app.store.read("n", "s") == before


def test_mixed_ids_are_never_partially_accepted():
    app = service()
    accept(app, packet())
    result = accept(
        app, packet(block="mixed", events=[("e1", "detail", "A"), ("e2", "detail", "B")])
    )
    assert result.reason == "closed_block_conflict"
    assert len(app.store.read("n", "s").blocks) == 1


@pytest.mark.parametrize(
    "event_time,now,reason", [(1001, 1000, "timestamp_future"), (1, 1_800_002, "timestamp_late")]
)
def test_rejected_new_trusted_time_marks_gap_without_advancing_last_arrival(
    event_time, now, reason
):
    app = service()
    result = accept(app, packet(event_time), now=now)
    assert result.reason == reason
    state = app.store.read("n", "s")
    assert not state.blocks and state.last_arrival_ms is None and state.gap_reasons


def test_closed_does_not_declare_stream_complete():
    app = service()
    accept(app, packet(), complete=False)
    with pytest.raises(StateError, match="state_incomplete"):
        prefix(app)
    assert app.first_trigger("n", "s", "b1", a_q=1000).block_id == "b1"


def test_late_block_replays_event_order_without_changing_previous_request():
    app = service()
    accept(app, packet(1000, events=[("e1", "detail", "A")]))
    old = prefix(app)
    accept(app, packet(500, block="late", events=[("late-e", "add", "B")]), now=1100)
    later = prefix(app, at=1100)
    assert old.prefix.cart == () and later.prefix.cart[0].sku == "B"
    assert old.a_q == 1000 and later.a_q == 1100
    assert app.store.read("n", "s").revision == 2


def test_later_trigger_cannot_skip_first_eligible_request():
    app = service()
    accept(app, packet())
    accept(app, packet(1100, block="b2", events=[("e2", "detail", "B")]), now=1100)
    with pytest.raises(StateError, match="not_first_request"):
        prefix(app, "b2", at=1100)
    with pytest.raises(StateError, match="not_first_request"):
        app.first_trigger("n", "s", "b2", a_q=1100)


def test_first_trigger_has_no_replay_and_obeys_availability(monkeypatch):
    app = service()
    accept(app, packet(), now=1100)

    def forbidden(*args, **kwargs):
        raise AssertionError("Первый trigger не требует восстановления префикса")

    monkeypatch.setattr("coveo_recsys.state.service.reduce_state", forbidden)
    assert app.first_trigger("n", "s", "b1", a_q=1100).event_time_ms == 1000
    with pytest.raises(StateError, match="trigger_not_available"):
        app.first_trigger("n", "s", "b1", a_q=1099)


def test_chronological_admission_does_not_replay_all_raw_events():
    app = service()
    for i in range(11):
        wire = packet(
            1000 + 10000 * i, block=f"b{i}", events=[(f"e{i}-{j}", "add", "A") for j in range(100)]
        )
        assert accept(app, wire, now=1000 + 10000 * i).status == "accepted"
    # 1100 транспортных событий свёрнуты в 11 логических; не 1100 событий сессии.
    assert sum(len(b.canonical.events) for b in app.store.read("n", "s").blocks) == 11


@pytest.mark.parametrize("count,accepted", [(1000, True), (1001, False)])
def test_session_logical_limit_without_partial_admission(count, accepted):
    app = service()
    for start in range(0, count, 100):
        events = [(f"e{j}", "add", f"sku{j}") for j in range(start, min(count, start + 100))]
        result = accept(
            app,
            packet(1000 + start * 1000, block=f"b{start}", events=events),
            now=1000 + start * 1000,
        )
    assert (result.status == "accepted") is accepted
    if not accepted:
        assert result.reason == "session_event_limit"
    assert sum(len(b.canonical.events) for b in app.store.read("n", "s").blocks) == 1000


def test_late_replay_limit_marks_gap_without_accepting_new_block():
    app = service()
    for i in range(11):
        wire = packet(
            1000 + 10000 * i, block=f"b{i}", events=[(f"e{i}-{j}", "add", "A") for j in range(100)]
        )
        assert accept(app, wire, now=1000 + 10000 * i).status == "accepted"
    result = accept(app, packet(900, block="late", events=[("late", "detail", "B")]), now=111000)
    assert result.reason == "state_replay_limit"
    state = app.store.read("n", "s")
    assert len(state.blocks) == 11 and "state_replay_limit" in state.gap_reasons


@pytest.mark.parametrize("journal_bytes,allowed", [(1_048_576, True), (1_048_577, False)])
def test_replay_byte_limit_uses_full_wire_not_only_logical_events(journal_bytes, allowed):
    app = service()
    for i, size in enumerate((journal_bytes // 2, journal_bytes - journal_bytes // 2)):
        wire = packet(
            1000 + i, block=f"b{i}", events=[(f"e{i}", "add" if i == 0 else "detail", "A")]
        )
        wire = b" " * (size - len(wire)) + wire
        assert accept(app, wire, now=1000 + i).status == "accepted"
    if allowed:
        assert prefix(app, "b1", at=1001).prefix.cart[0].sku == "A"
    else:
        with pytest.raises(StateError, match="state_replay_limit"):
            prefix(app, "b1", at=1001)


def test_old_clock_cannot_accept_an_event_with_backdated_availability():
    app = service()
    accept(app, packet(), now=2000)
    before = app.store.read("n", "s")
    result = accept(app, packet(1100, block="next", events=[("e2", "detail", "B")]), now=1500)
    assert result.reason == "state_clock_regressed"
    assert app.store.read("n", "s").blocks == before.blocks


def test_nonfinite_search_coordinate_stays_masked_not_rejected():
    app = service()
    body = [
        {
            "source_namespace": "n",
            "session_id": "s",
            "event_time_ms": 1000,
            "event_id": "search",
            "event_type": "search",
            "sku": None,
            "query_vector": ["marker", 0],
        }
    ]
    payload = json.dumps(body, separators=(",", ":")).encode().replace(b'"marker"', b"1e999")
    header = {
        "source_namespace": "n",
        "session_id": "s",
        "event_time_ms": 1000,
        "block_id": "search",
        "event_count": 1,
        "event_ids": ["search"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "closed": True,
    }
    wire = json.dumps(header).encode()[:-1] + b',"events":' + payload + b"}"
    assert accept(app, wire).status == "accepted"
    assert app.store.read("n", "s").blocks[0].canonical.events[0].search.vector is None


def test_incompatible_task_state_has_safe_error_not_reinterpretation():
    app = service()
    accept(app, packet())
    changed = IntakeService(
        app.store, TaskConfig(event_semantics="multiset_v1"), {"secret": "n"}, 0, 10_000_000
    )
    with pytest.raises(StateError, match="^state_incompatible$"):
        prefix(changed)
    result = accept(changed, packet(1100, block="new", events=[("e2", "add", "B")]), now=1100)
    assert result.reason == "state_incompatible"


def test_purchase_in_trigger_has_no_eligible_request():
    app = service()
    accept(app, packet(events=[("e1", "detail", "A"), ("e2", "purchase", "B")]))
    with pytest.raises(StateError, match="request_not_eligible"):
        prefix(app)


def test_receipt_survives_absent_operational_snapshot_and_prefix_rebuilds():
    app = service()
    receipt = accept(app, packet()).receipt
    state = app.store.read("n", "s")
    assert app.store.compare_and_swap(
        "n", "s", state.revision, replace(state, revision=state.revision + 1, snapshot=None)
    )
    assert app.receipt("n", "s", "b1") == receipt
    assert prefix(app).prefix.detail_history[0].sku == "A"


def test_repeats_consume_raw_rate_limit_without_creating_gap():
    app = service()
    wire = packet(events=[(f"e{i}", "detail", "A") for i in range(100)])
    first = accept(app, wire)
    before = app.store.read("n", "s")
    repeated = accept(app, wire)
    assert repeated.reason == "duplicate_rate_rejected"
    assert repeated.receipt == first.receipt and app.store.read("n", "s") == before
    assert accept(app, wire, now=11000).status == "duplicate"


def test_expired_deadline_cannot_publish_any_state():
    app = service()

    def expired():
        raise StateError("deadline_exceeded")

    with pytest.raises(StateError, match="deadline_exceeded"):
        app.accept(packet(), token="secret", now_ms=1000, check_deadline=expired)
    assert app.store.read("n", "s") is None


@pytest.mark.parametrize("conflicts,expected", [(1, "accepted"), (2, "state_conflict")])
def test_only_one_cas_retry_is_allowed(conflicts, expected):
    class ConflictingStore(MemoryStateStore):
        attempts = 0

        def compare_and_swap(self, *args, **kwargs):
            self.attempts += 1
            return (
                False if self.attempts <= conflicts else super().compare_and_swap(*args, **kwargs)
            )

    store = ConflictingStore()
    result = accept(service(store), packet())
    assert result.reason == expected if expected == "state_conflict" else result.status == expected
    assert store.attempts == 2


def test_file_replay_preserves_receipts_prefix_and_detects_corruption(tmp_path):
    path = tmp_path / "state.json"
    first = service(FileStateStore(path))
    receipt = accept(first, packet()).receipt
    restored = service(FileStateStore(path))
    assert restored.receipt("n", "s", "b1") == receipt
    assert prefix(restored) == prefix(first)
    assert accept(restored, packet(), now=2_000_000).status == "duplicate"
    path.write_text('{"unexpected":true}', encoding="utf-8")
    with pytest.raises(StateError, match="state_incompatible"):
        FileStateStore(path)
