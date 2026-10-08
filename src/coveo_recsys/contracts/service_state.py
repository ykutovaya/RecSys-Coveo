"""Локальный приём: квитанции и журнал отделены от результатов рекомендаций."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .common import InputValidationError, id_bytes, require_int64
from .preparation import TaskConfig
from .records import CanonicalBlock, PrefixSnapshot, Record


class StateError(InputValidationError):
    """Безопасный машинный код, без входной нагрузки и секретов."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BlockReceipt(Record):
    source_namespace: str
    session_id: str
    block_id: str
    event_time_ms: int
    event_count: int
    event_ids: tuple[str, ...]
    payload_sha256: str
    canonical_block_id: str
    decision_time_ms: int
    state_version: int

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if self.event_count != len(self.event_ids) or not 1 <= self.event_count <= 100:
            raise InputValidationError("Несогласованная квитанция")
        if len(set(self.event_ids)) != self.event_count or self.state_version <= 0:
            raise InputValidationError("Повтор ID или неверная версия квитанции")
        if self.decision_time_ms < self.event_time_ms:
            raise InputValidationError("Квитанция события из будущего")


@dataclass(frozen=True, slots=True)
class StoredBlock(Record):
    receipt: BlockReceipt
    canonical: CanonicalBlock
    envelope_json: str
    event_fingerprints: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        receipt, block = self.receipt, self.canonical
        if (
            (receipt.source_namespace, receipt.session_id, receipt.event_time_ms)
            != (block.source_namespace, block.session_id, block.event_time_ms)
            or receipt.canonical_block_id != block.block_id
            or receipt.decision_time_ms != block.available_time_ms
            or tuple(eid for eid, _ in self.event_fingerprints) != receipt.event_ids
        ):
            raise InputValidationError("Квитанция не соответствует сохранённому блоку")


@dataclass(frozen=True, slots=True)
class StoredSession(Record):
    source_namespace: str
    session_id: str
    revision: int
    blocks: tuple[StoredBlock, ...] = ()
    snapshot: PrefixSnapshot | None = None
    stream_complete: bool = False
    gap_reasons: tuple[str, ...] = ()
    last_arrival_ms: int | None = None
    quarantine: tuple[str, ...] = ()
    transformation_version: str = "service_state_v1"
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        require_int64(self.revision, nonnegative=True)
        key = self.source_namespace, self.session_id
        if self.transformation_version != "service_state_v1":
            raise InputValidationError("Несовместимое состояние")
        if (
            self.snapshot is not None
            and (self.snapshot.source_namespace, self.snapshot.session_id) != key
        ):
            raise InputValidationError("Чужой снимок")
        if self.snapshot is not None and self.snapshot.event_semantics != self.task.event_semantics:
            raise InputValidationError("Несовместимая семантика снимка")
        if any((b.receipt.source_namespace, b.receipt.session_id) != key for b in self.blocks):
            raise InputValidationError("Чужой блок")
        if len({b.receipt.block_id for b in self.blocks}) != len(self.blocks) or len(
            {b.receipt.event_time_ms for b in self.blocks}
        ) != len(self.blocks):
            raise InputValidationError("Повтор закрытого блока")
        marks = [eid for b in self.blocks for eid in b.receipt.event_ids]
        if len(set(marks)) != len(marks) or any(
            b.receipt.state_version > self.revision for b in self.blocks
        ):
            raise InputValidationError("Повтор событий или неверная ревизия")


@dataclass(frozen=True, slots=True)
class Acceptance(Record):
    status: str
    reason: str | None
    receipt: BlockReceipt | None
    decision_time_ms: int

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        if self.status not in {"accepted", "duplicate", "rejected"}:
            raise InputValidationError("Неизвестный результат приёма")
        if self.status == "accepted" and (self.receipt is None or self.reason is not None):
            raise InputValidationError("Принятие требует квитанцию")


class StateStore(Protocol):
    def check_available(self) -> None: ...
    def read(self, namespace: str, session_id: str) -> StoredSession | None: ...
    def receipt(self, namespace: str, session_id: str, block_id: str) -> BlockReceipt | None: ...
    def compare_and_swap(
        self,
        namespace: str,
        session_id: str,
        expected_revision: int,
        next_session: StoredSession,
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> bool: ...
    def charge(
        self,
        namespace: str,
        session_id: str,
        cost: int,
        now_ms: int,
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> bool: ...
