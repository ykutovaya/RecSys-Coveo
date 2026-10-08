"""Тестовые память и файловое воспроизведение; не рабочая СУБД."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

from coveo_recsys.contracts.common import id_bytes, require_int64
from coveo_recsys.contracts.serialization import _unique_object, from_json, to_json
from coveo_recsys.contracts.service_state import BlockReceipt, StateError, StoredSession


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _invalid_constant(value: str) -> None:
    raise StateError("state_incompatible")


class MemoryStateStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[tuple[str, str], StoredSession] = {}
        self._receipts: dict[tuple[str, str, str], BlockReceipt] = {}
        self._buckets: dict[tuple[str, str], tuple[int, int]] = {}

    def check_available(self) -> None:
        """Проверка адаптера без чтения или изменения пользовательской истории."""
        with self._lock:
            pass

    def read(self, namespace: str, session_id: str) -> StoredSession | None:
        with self._lock:
            return self._sessions.get((namespace, session_id))

    def receipt(self, namespace: str, session_id: str, block_id: str) -> BlockReceipt | None:
        with self._lock:
            return self._receipts.get((namespace, session_id, block_id))

    def _persist(
        self,
        sessions: dict[tuple[str, str], StoredSession],
        buckets: dict[tuple[str, str], tuple[int, int]],
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> None:
        if check_deadline is not None:
            check_deadline()

    def compare_and_swap(
        self,
        namespace: str,
        session_id: str,
        expected_revision: int,
        next_session: StoredSession,
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> bool:
        key = namespace, session_id
        id_bytes(namespace)
        id_bytes(session_id)
        require_int64(expected_revision, nonnegative=True)
        if (next_session.source_namespace, next_session.session_id) != key or (
            next_session.revision != expected_revision + 1
        ):
            raise StateError("state_incompatible")
        with self._lock:
            previous = self._sessions.get(key)
            if (previous.revision if previous else 0) != expected_revision:
                return False
            if (
                previous is not None
                and next_session.blocks[: len(previous.blocks)] != previous.blocks
            ):
                raise StateError("state_incompatible")
            sessions = {**self._sessions, key: next_session}
            self._persist(sessions, self._buckets, check_deadline=check_deadline)
            self._sessions = sessions
            for block in next_session.blocks:
                self._receipts[(*key, block.receipt.block_id)] = block.receipt
            return True

    def charge(
        self,
        namespace: str,
        session_id: str,
        cost: int,
        now_ms: int,
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> bool:
        require_int64(cost, nonnegative=True)
        require_int64(now_ms)
        if cost < 1:
            raise StateError("invalid_input")
        key = namespace, session_id
        with self._lock:
            tokens, previous = self._buckets.get(key, (100_000, now_ms))
            refill = min(100_000, tokens + max(0, now_ms - previous) * 10)
            allowed = refill >= cost * 1000
            buckets = {
                **self._buckets,
                key: (refill - cost * 1000 if allowed else refill, max(previous, now_ms)),
            }
            self._persist(self._sessions, buckets, check_deadline=check_deadline)
            self._buckets = buckets
            return allowed


class FileStateStore(MemoryStateStore):
    """Один проверяемый JSON-снимок; память меняется только после атомарной записи."""

    def __init__(self, path: Path, *, max_bytes: int = 16_777_216) -> None:
        super().__init__()
        require_int64(max_bytes, nonnegative=True)
        if max_bytes == 0:
            raise StateError("invalid_input")
        self.path, self.max_bytes = path.absolute(), max_bytes
        self._check_path()
        self._published = self.path.exists()
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > max_bytes:
                raise StateError("state_replay_limit")
            with self.path.open("rb") as stream:
                payload = stream.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise StateError("state_replay_limit")
            value = json.loads(
                payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
            if (
                not isinstance(value, dict)
                or set(value) != {"format", "body", "sha256"}
                or (
                    value["format"] != "local_state_v1"
                    or hashlib.sha256(_json_bytes(value["body"])).hexdigest() != value["sha256"]
                )
            ):
                raise StateError("state_incompatible")
            body = value["body"]
            if not isinstance(body, dict) or set(body) != {"sessions", "buckets"}:
                raise StateError("state_incompatible")
            for encoded in body["sessions"]:
                session = from_json(_json_bytes(encoded), StoredSession)
                key = session.source_namespace, session.session_id
                if key in self._sessions:
                    raise StateError("state_incompatible")
                self._sessions[key] = session
                for block in session.blocks:
                    self._receipts[(*key, block.receipt.block_id)] = block.receipt
            for ns, sid, tokens, timestamp in body["buckets"]:
                id_bytes(ns)
                id_bytes(sid)
                require_int64(tokens, nonnegative=True)
                require_int64(timestamp)
                if tokens > 100_000 or (ns, sid) in self._buckets:
                    raise StateError("state_incompatible")
                self._buckets[ns, sid] = tokens, timestamp
        except OSError as exc:
            raise StateError("state_unavailable") from exc
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            if isinstance(exc, StateError):
                raise
            raise StateError("state_incompatible") from exc

    def _check_path(self) -> None:
        if any(part.is_symlink() for part in (self.path, *self.path.parents)):
            raise StateError("state_incompatible")

    def check_available(self) -> None:
        """Проверить доступ без тестовой записи; это не гарантия следующего fsync."""
        with self._lock:
            self._check_path()
            try:
                directory = self.path.parent
                while not directory.exists() and directory != directory.parent:
                    directory = directory.parent
                if not directory.is_dir() or not os.access(directory, os.R_OK | os.W_OK):
                    raise StateError("state_unavailable")
                if self._published and not self.path.is_file():
                    raise StateError("state_unavailable")
                if self.path.exists():
                    if not self.path.is_file() or not os.access(self.path, os.R_OK | os.W_OK):
                        raise StateError("state_unavailable")
                    if self.path.stat().st_size > self.max_bytes:
                        raise StateError("state_replay_limit")
                    with self.path.open("rb") as stream:
                        stream.read(1)
            except OSError as exc:
                raise StateError("state_unavailable") from exc

    def _persist(
        self,
        sessions: dict[tuple[str, str], StoredSession],
        buckets: dict[tuple[str, str], tuple[int, int]],
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> None:
        if check_deadline is not None:
            check_deadline()
        self._check_path()
        body: dict[str, Any] = {
            "sessions": [json.loads(to_json(sessions[key])) for key in sorted(sessions)],
            "buckets": [[*key, *buckets[key]] for key in sorted(buckets)],
        }
        payload = _json_bytes(
            {
                "format": "local_state_v1",
                "body": body,
                "sha256": hashlib.sha256(_json_bytes(body)).hexdigest(),
            }
        )
        if len(payload) > self.max_bytes:
            raise StateError("state_replay_limit")
        temporary: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_path()
            if check_deadline is not None:
                check_deadline()
            os.replace(temporary, self.path)
            temporary = None
            self._published = True
        except OSError as exc:
            raise StateError("state_unavailable") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
