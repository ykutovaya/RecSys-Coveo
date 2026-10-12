import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from coveo_recsys.contracts.service_state import StateError, StoredSession
from coveo_recsys.state.store import FileStateStore, MemoryStateStore
from tests.unit.test_e6_state import accept, packet, service


def test_concurrent_cas_accepts_exactly_one_revision():
    store = MemoryStateStore()
    state = StoredSession("n", "s", 1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.compare_and_swap("n", "s", 0, state), range(2)))
    assert sorted(results) == [False, True]
    assert store.read("n", "s") == state


@pytest.mark.parametrize("namespace,revision", [("foreign", 1), ("n", 0), ("n", 2)])
def test_cas_rejects_wrong_namespace_or_revision(namespace, revision):
    store = MemoryStateStore()
    with pytest.raises(StateError, match="state_incompatible"):
        store.compare_and_swap("n", "s", 0, StoredSession(namespace, "s", revision))
    assert store.read("n", "s") is None


def test_file_write_failure_keeps_original_memory_and_disk(tmp_path, monkeypatch):
    store = FileStateStore(tmp_path / "state.json")
    initial = StoredSession("n", "s", 1)
    assert store.compare_and_swap("n", "s", 0, initial)
    original = store.path.read_bytes()

    def failed(*args):
        raise OSError("Do not expose private details")

    monkeypatch.setattr("coveo_recsys.state.store.os.replace", failed)
    with pytest.raises(StateError, match="^state_unavailable$"):
        store.compare_and_swap("n", "s", 1, replace(initial, revision=2))
    assert store.read("n", "s") == initial and store.path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize("operation", ["cas", "charge"])
def test_memory_checks_deadline_before_publication(operation):
    store = MemoryStateStore()
    initial = StoredSession("n", "s", 1)
    assert store.compare_and_swap("n", "s", 0, initial)
    assert store.charge("n", "s", 1, 0)
    buckets = dict(store._buckets)

    def expired():
        raise StateError("deadline_exceeded")

    with pytest.raises(StateError, match="^deadline_exceeded$"):
        if operation == "cas":
            store.compare_and_swap(
                "n", "s", 1, replace(initial, revision=2), check_deadline=expired
            )
        else:
            store.charge("n", "s", 100, 1000, check_deadline=expired)
    assert store.read("n", "s") == initial and store._buckets == buckets


@pytest.mark.parametrize("operation", ["cas", "charge"])
def test_slow_fsync_cannot_publish_after_deadline(tmp_path, monkeypatch, operation):
    store = FileStateStore(tmp_path / "state.json")
    initial = StoredSession("n", "s", 1)
    assert store.compare_and_swap("n", "s", 0, initial)
    assert store.charge("n", "s", 1, 0)
    original, buckets = store.path.read_bytes(), dict(store._buckets)
    clock = [0.0]
    original_fsync = os.fsync

    def delayed_fsync(descriptor):
        original_fsync(descriptor)
        clock[0] = 0.100

    def check_deadline():
        if clock[0] >= 0.010:
            raise StateError("deadline_exceeded")

    monkeypatch.setattr("coveo_recsys.state.store.os.fsync", delayed_fsync)
    with pytest.raises(StateError, match="^deadline_exceeded$"):
        if operation == "cas":
            store.compare_and_swap(
                "n", "s", 1, replace(initial, revision=2), check_deadline=check_deadline
            )
        else:
            store.charge("n", "s", 100, 1000, check_deadline=check_deadline)
    assert store.read("n", "s") == initial and store._buckets == buckets
    assert store.path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize("phase", ["charge", "accept", "reject"])
def test_intake_passes_original_deadline_to_every_publication(tmp_path, monkeypatch, phase):
    store = FileStateStore(tmp_path / "state.json")
    app = service(store)
    first = accept(app, packet()).receipt
    initial = store.read("n", "s")
    wire = packet(1000 if phase == "reject" else 1100, block="b2", events=[("e2", "detail", "B")])
    clock, sync_count = [0.0], [0]
    before_expired_write = []
    original_fsync = os.fsync

    def delayed_fsync(descriptor):
        sync_count[0] += 1
        original_fsync(descriptor)
        if sync_count[0] == (1 if phase == "charge" else 2):
            before_expired_write.append((store.path.read_bytes(), dict(store._buckets)))
            clock[0] = 0.100

    def check_deadline():
        if clock[0] >= 0.010:
            raise StateError("deadline_exceeded")

    monkeypatch.setattr("coveo_recsys.state.store.os.fsync", delayed_fsync)
    with pytest.raises(StateError, match="^deadline_exceeded$"):
        app.accept(wire, token="secret", now_ms=1100, check_deadline=check_deadline)
    assert before_expired_write
    assert store.path.read_bytes() == before_expired_write[0][0]
    assert store._buckets == before_expired_write[0][1]
    assert store.read("n", "s") == initial
    assert store.receipt("n", "s", "b1") == first
    assert store.receipt("n", "s", "b2") is None
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_file_limit_is_checked_before_publication(tmp_path):
    store = FileStateStore(tmp_path / "state.json", max_bytes=100)
    with pytest.raises(StateError, match="state_replay_limit"):
        store.compare_and_swap("n", "s", 0, StoredSession("n", "s", 1))
    assert store.read("n", "s") is None and not store.path.exists()


def test_file_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"format":"local_state_v1","format":"local_state_v1"}', encoding="utf8")
    with pytest.raises(StateError, match="state_incompatible"):
        FileStateStore(path)


def test_deep_file_json_is_safe_incompatible_error(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf8")
    with pytest.raises(StateError, match="^state_incompatible$"):
        FileStateStore(path)


def test_file_rejects_symlink(tmp_path):
    actual = tmp_path / "actual.json"
    actual.write_text("{}", encoding="utf8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(actual)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("windows_symlink_unavailable: WinError 1314")
        raise
    with pytest.raises(StateError, match="state_incompatible"):
        FileStateStore(link)


def test_rate_bucket_uses_raw_events_and_millisecond_boundaries():
    store = MemoryStateStore()
    assert store.charge("n", "s", 100, 0)
    assert not store.charge("n", "s", 1, 99)
    assert store.charge("n", "s", 1, 100)
    assert store.charge("m", "s", 100, 100)
    assert not store.charge("n", "s", 1, 99)


def test_cas_cannot_replace_or_remove_accepted_history():
    store = MemoryStateStore()
    app = service(store)
    receipt = accept(app, packet()).receipt
    state = store.read("n", "s")
    with pytest.raises(StateError, match="state_incompatible"):
        store.compare_and_swap(
            "n", "s", state.revision, replace(state, revision=state.revision + 1, blocks=())
        )
    assert store.receipt("n", "s", "b1") == receipt and store.read("n", "s") == state


@pytest.mark.parametrize("file_backed", [False, True])
def test_health_probe_does_not_write_or_change_user_state(tmp_path, file_backed):
    store = FileStateStore(tmp_path / "state.json") if file_backed else MemoryStateStore()
    assert store.check_available() is None
    assert list(tmp_path.iterdir()) == []
    app = service(store)
    accept(app, packet())
    before = store.read("n", "s")
    raw = store.path.read_bytes() if file_backed else None
    modified = store.path.stat().st_mtime_ns if file_backed else None
    assert store.check_available() is None
    assert store.read("n", "s") == before
    if file_backed:
        assert store.path.read_bytes() == raw and store.path.stat().st_mtime_ns == modified


def test_health_probe_detects_removed_state_file_without_recreating(tmp_path):
    path = tmp_path / "state.json"
    store = FileStateStore(path)
    accept(service(store), packet())
    path.unlink()
    with pytest.raises(StateError, match="state_unavailable"):
        store.check_available()
    assert not path.exists()
