"""Приём полных блоков на локальных тестовых адаптерах."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from coveo_recsys.contracts.common import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.preparation import PreparedRequest, ScheduleCursor, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, PrefixSnapshot
from coveo_recsys.contracts.service_state import (
    Acceptance,
    BlockReceipt,
    StateError,
    StateStore,
    StoredBlock,
    StoredSession,
)
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.envelope import (
    EnvelopeLimits,
    SealedBlock,
    _transport_json,
    block_time_issues,
    inspect_sealed_block,
)
from coveo_recsys.data.schedule import advance_schedule

from .reducer import build_prefix, empty_prefix, reduce_state

MAX_QUARANTINE_BYTES = 1_048_576  # Сумма UTF-8 байтов конвертов одной сессии.


class IntakeService:
    def __init__(
        self,
        store: StateStore,
        task: TaskConfig,
        owners: Mapping[str, str],
        calendar_start_ms: int,
        calendar_end_ms: int,
    ) -> None:
        self.store, self.task = store, task
        self.owners = dict(owners)
        self.calendar = calendar_start_ms, calendar_end_ms
        require_int64(calendar_start_ms)
        require_int64(calendar_end_ms)
        if calendar_start_ms >= calendar_end_ms or not isinstance(task, TaskConfig):
            raise InputValidationError("Неверный календарь или задача")
        for token, namespace in self.owners.items():
            id_bytes(token)
            id_bytes(namespace)
        self.limits = EnvelopeLimits()

    def namespace_for(self, token: str) -> str:
        if token not in self.owners:
            raise StateError("unauthorized")
        return self.owners[token]

    def receipt(
        self, namespace: str, session_id: str, trigger_block_id: str
    ) -> BlockReceipt | None:
        return self.store.receipt(namespace, session_id, trigger_block_id)

    def first_trigger(
        self,
        namespace: str,
        session_id: str,
        trigger_block_id: str,
        *,
        a_q: int | None = None,
        check_deadline: Callable[[], None] | None = None,
    ) -> BlockReceipt:
        """Первый допустимый trigger по журналу; без восстановления и фиксации запроса."""
        if a_q is not None:
            require_int64(a_q)
        if check_deadline is not None:
            check_deadline()
        receipt = self.store.receipt(namespace, session_id, trigger_block_id)
        if receipt is None:
            raise StateError("trigger_not_accepted")
        if a_q is not None and receipt.decision_time_ms > a_q:
            raise StateError("trigger_not_available")
        state = self.store.read(namespace, session_id)
        if state is None:
            raise StateError("retry_unverifiable")
        if state.task != self.task:
            raise StateError("state_incompatible")
        for item in sorted(state.blocks, key=lambda b: b.receipt.event_time_ms):
            if check_deadline is not None:
                check_deadline()
            if a_q is not None and item.receipt.decision_time_ms > a_q:
                continue
            actions = {event.event_type for event in item.canonical.events}
            if EventType.PURCHASE in actions:
                break
            if EventType.DETAIL in actions:
                if item.receipt.block_id != trigger_block_id:
                    raise StateError("not_first_request")
                return receipt
        raise StateError("request_not_eligible")

    def accept(
        self,
        envelope: bytes,
        *,
        token: str,
        now_ms: int,
        check_deadline: Callable[[], None],
        stream_complete: bool = False,
    ) -> Acceptance:
        check_deadline()
        require_int64(now_ms)
        if type(stream_complete) is not bool:
            raise InputValidationError("Полнота задаётся явным bool")
        try:
            namespace = self.namespace_for(token)
        except StateError as exc:
            return Acceptance("rejected", exc.code, None, now_ms)
        # Заголовок здесь нужен лишь для безопасной адресации/стоимости попытки.
        # Все поля и точные байты повторно проверяются общим валидатором конверта.
        header: Any = None
        if type(envelope) is bytes and len(envelope) <= self.limits.max_request_bytes:
            try:
                header = _transport_json(envelope)
            except (ValueError, TypeError, UnicodeError, RecursionError):
                pass
        if not isinstance(header, dict):
            return Acceptance("rejected", "invalid_input", None, now_ms)
        if header.get("source_namespace") != namespace:
            return Acceptance("rejected", "namespace_forbidden", None, now_ms)
        session_id = header.get("session_id")
        if not isinstance(session_id, str):
            return Acceptance("rejected", "invalid_input", None, now_ms)
        try:
            id_bytes(session_id)
        except (ValueError, TypeError):
            return Acceptance("rejected", "invalid_input", None, now_ms)
        raw = header.get("events")
        cost = max(1, len(raw)) if isinstance(raw, list) else 1
        try:
            allowed = self.store.charge(
                namespace, session_id, cost, now_ms, check_deadline=check_deadline
            )
            check_deadline()
            inspected = inspect_sealed_block(
                envelope,
                decision_time_ms=now_ms,
                expected_namespace=namespace,
                query_dimension=self.task.query_dimension,
                limits=self.limits,
            )
            if inspected.block is None:
                return self._reject(
                    namespace,
                    session_id,
                    envelope,
                    inspected.issues[0].code,
                    now_ms,
                    check_deadline,
                )
            block = inspected.block
            assert isinstance(raw, list)
            receipt = self.store.receipt(namespace, session_id, block.block_id)
            if receipt is not None and self._same_receipt(receipt, block):
                return Acceptance(
                    "duplicate" if allowed else "rejected",
                    None if allowed else "duplicate_rate_rejected",
                    receipt,
                    now_ms,
                )
            raw_identity = json.loads(envelope, parse_int=str, parse_float=str)["events"]
            fingerprints = tuple(
                (
                    row["event_id"],
                    hashlib.sha256(
                        json.dumps(
                            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                )
                for row in raw_identity
            )
            for _ in range(2):
                check_deadline()
                previous = self.store.read(namespace, session_id)
                current = previous or StoredSession(namespace, session_id, 0, task=self.task)
                if current.task != self.task:
                    return Acceptance("rejected", "state_incompatible", None, now_ms)
                old_blocks = {b.receipt.block_id: b for b in current.blocks}
                # Повторная сверка после конкурентного принятия, не новый приём.
                existing = old_blocks.get(block.block_id)
                if existing is not None and self._same_receipt(existing.receipt, block):
                    return Acceptance(
                        "duplicate" if allowed else "rejected",
                        None if allowed else "duplicate_rate_rejected",
                        existing.receipt,
                        now_ms,
                    )
                known = {eid: value for b in current.blocks for eid, value in b.event_fingerprints}
                if (
                    existing is None
                    and fingerprints
                    and all(known.get(eid) == value for eid, value in fingerprints)
                ):
                    return Acceptance(
                        "duplicate" if allowed else "rejected",
                        "known_events" if allowed else "duplicate_rate_rejected",
                        None,
                        now_ms,
                    )
                if not allowed:
                    return self._reject(
                        namespace, session_id, envelope, "rate_rejected", now_ms, check_deadline
                    )
                closed = any(b.receipt.event_time_ms == block.event_time_ms for b in current.blocks)
                if existing is not None or closed:
                    return self._reject(
                        namespace,
                        session_id,
                        envelope,
                        "closed_block_conflict",
                        now_ms,
                        check_deadline,
                    )
                if any(eid in known for eid, _ in fingerprints):
                    return self._reject(
                        namespace, session_id, envelope, "event_id_conflict", now_ms, check_deadline
                    )
                issues = block_time_issues(block, self.calendar, self.limits)
                if issues:
                    return self._reject(
                        namespace, session_id, envelope, issues[0].code, now_ms, check_deadline
                    )
                canonical = tuple(canonicalize(block.events))
                if len(canonical) != 1:
                    return self._reject(
                        namespace, session_id, envelope, "invalid_input", now_ms, check_deadline
                    )
                logical_count = sum(len(b.canonical.events) for b in current.blocks)
                if logical_count + len(canonical[0].events) > 1000:
                    return self._reject(
                        namespace,
                        session_id,
                        envelope,
                        "session_event_limit",
                        now_ms,
                        check_deadline,
                    )
                revision = current.revision + 1
                receipt = BlockReceipt(
                    namespace,
                    session_id,
                    block.block_id,
                    block.event_time_ms,
                    len(block.events),
                    tuple(row["event_id"] for row in raw),
                    block.payload_sha256,
                    canonical[0].block_id,
                    now_ms,
                    revision,
                )
                stored = StoredBlock(receipt, canonical[0], envelope.decode("utf-8"), fingerprints)
                journal = (*current.blocks, stored)
                previous_snapshot = current.snapshot
                if not current.blocks:
                    previous_snapshot = empty_prefix(namespace, session_id, self.task)
                try:
                    if previous_snapshot is not None and (
                        previous_snapshot.event_cutoff_ms < block.event_time_ms
                    ):
                        snapshot = reduce_state(
                            previous_snapshot, canonical[0], self.task
                        ).next_snapshot
                    else:
                        snapshot = self._build(
                            journal,
                            namespace,
                            session_id,
                            max(b.receipt.event_time_ms for b in journal),
                            now_ms,
                            check_deadline,
                        )
                except StateError as exc:
                    if exc.code != "state_replay_limit":
                        raise
                    return self._reject(
                        namespace, session_id, envelope, exc.code, now_ms, check_deadline
                    )
                if current.last_arrival_ms is not None and now_ms < current.last_arrival_ms:
                    return self._reject(
                        namespace,
                        session_id,
                        envelope,
                        "state_clock_regressed",
                        now_ms,
                        check_deadline,
                    )
                updated = replace(
                    current,
                    revision=revision,
                    blocks=journal,
                    snapshot=snapshot,
                    stream_complete=stream_complete and not current.gap_reasons,
                    last_arrival_ms=now_ms,
                )
                check_deadline()
                if self.store.compare_and_swap(
                    namespace,
                    session_id,
                    current.revision,
                    updated,
                    check_deadline=check_deadline,
                ):
                    return Acceptance("accepted", None, receipt, now_ms)
            return Acceptance("rejected", "state_conflict", None, now_ms)
        except StateError as exc:
            if exc.code == "deadline_exceeded":
                raise
            return Acceptance("rejected", exc.code, None, now_ms)

    @staticmethod
    def _same_receipt(receipt: BlockReceipt, block: SealedBlock) -> bool:
        return (
            receipt.source_namespace == block.source_namespace
            and receipt.session_id == block.session_id
            and receipt.block_id == block.block_id
            and receipt.event_time_ms == block.event_time_ms
            and receipt.event_count == len(block.events)
            and receipt.event_ids == tuple(e.transport_id for e in block.events)
            and receipt.payload_sha256 == block.payload_sha256
        )

    def _reject(
        self,
        namespace: str,
        session_id: str,
        envelope: bytes,
        reason: str,
        now_ms: int,
        check_deadline: Callable[[], None],
    ) -> Acceptance:
        for _ in range(2):
            check_deadline()
            current = self.store.read(namespace, session_id) or StoredSession(
                namespace, session_id, 0, task=self.task
            )
            if current.task != self.task:
                return Acceptance("rejected", "state_incompatible", None, now_ms)
            quarantine = current.quarantine
            if len(envelope) <= self.limits.max_request_bytes:
                text = envelope.decode("utf-8", errors="replace")
                if text not in quarantine:
                    size = sum(len(item.encode("utf-8")) for item in quarantine)
                    if size + len(text.encode("utf-8")) <= MAX_QUARANTINE_BYTES:
                        quarantine = (*quarantine, text)
            updated = replace(
                current,
                revision=current.revision + 1,
                stream_complete=False,
                gap_reasons=tuple(sorted({*current.gap_reasons, reason})),
                quarantine=quarantine,
            )
            check_deadline()
            if self.store.compare_and_swap(
                namespace,
                session_id,
                current.revision,
                updated,
                check_deadline=check_deadline,
            ):
                return Acceptance("rejected", reason, None, now_ms)
        return Acceptance("rejected", "state_conflict", None, now_ms)

    def _visible(
        self,
        blocks: tuple[StoredBlock, ...],
        t_q: int,
        a_q: int,
        check_deadline: Callable[[], None],
    ) -> tuple[CanonicalBlock, ...]:
        visible = []
        events = size = 0
        for block in blocks:
            check_deadline()
            if block.receipt.event_time_ms <= t_q and block.receipt.decision_time_ms <= a_q:
                events += block.receipt.event_count
                size += len(block.envelope_json.encode("utf-8"))
                if events > 1000 or size > 1_048_576:
                    raise StateError("state_replay_limit")
                visible.append(block.canonical)
        return tuple(sorted(visible, key=lambda b: b.event_time_ms))

    def _build(
        self,
        blocks: tuple[StoredBlock, ...],
        namespace: str,
        session_id: str,
        t_q: int,
        a_q: int,
        check_deadline: Callable[[], None],
    ) -> PrefixSnapshot:
        visible = self._visible(blocks, t_q, a_q, check_deadline)
        check_deadline()
        return build_prefix(
            visible,
            source_namespace=namespace,
            session_id=session_id,
            event_cutoff_ms=t_q,
            availability_cutoff_ms=a_q,
            config=self.task,
        )

    def prefix(
        self,
        namespace: str,
        session_id: str,
        trigger_block_id: str,
        *,
        a_q: int,
        check_deadline: Callable[[], None],
    ) -> PreparedRequest:
        check_deadline()
        require_int64(a_q)
        self.first_trigger(
            namespace, session_id, trigger_block_id, a_q=a_q, check_deadline=check_deadline
        )
        state = self.store.read(namespace, session_id)
        receipt = self.store.receipt(namespace, session_id, trigger_block_id)
        if receipt is None:
            raise StateError("trigger_not_accepted")
        if state is None:
            raise StateError("retry_unverifiable")
        if not state.stream_complete or state.gap_reasons:
            raise StateError("state_incomplete")
        if receipt.decision_time_ms > a_q:
            raise StateError("trigger_not_available")
        visible = self._visible(state.blocks, receipt.event_time_ms, a_q, check_deadline)
        snapshot = empty_prefix(namespace, session_id, self.task)
        cursor = ScheduleCursor(namespace, session_id)
        first: PreparedRequest | None = None
        for block in visible:
            check_deadline()
            snapshot = reduce_state(snapshot, block, self.task).next_snapshot
            current = replace(snapshot, availability_cutoff_ms=a_q)
            if first is None:
                cursor, first = advance_schedule(cursor, current, block.block_id, self.task)
        if first is None:
            raise StateError("request_not_eligible")
        if first.trigger_block_id != receipt.canonical_block_id:
            raise StateError("not_first_request")
        check_deadline()
        return first
