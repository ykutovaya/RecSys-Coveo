"""Общие таблицы подготовки: порционное чтение и публикация завершённого набора."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, TypeVar
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from coveo_recsys.artifacts import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.preparation import EVALUATION_PROTOCOL, PREPARATION_VERSION
from coveo_recsys.contracts.records import Record
from coveo_recsys.contracts.serialization import from_json, to_json

FORMAT_VERSION = "preparation_tables_v1"
REPORT = "reports/preparation.json"
TABLE_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("record_json", pa.binary(), nullable=False),
    ]
)
R = TypeVar("R", bound=Record)


def _reject_links(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise InputValidationError("Символьные ссылки в подготовке запрещены")


def _publish(temporary: Path, destination: Path) -> None:
    _reject_links(destination)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        _reject_links(destination)
        if sha256_file(temporary) != sha256_file(destination):
            raise InputValidationError("Повтор не может заменить отличающийся результат")
    temporary.unlink()


class TableWriter:
    def __init__(
        self,
        path: Path,
        record_type: type[Record],
        *,
        batch_rows: int = 4096,
        batch_bytes: int = 8_388_608,
    ) -> None:
        if batch_rows < 1 or batch_bytes < 1:
            raise InputValidationError("Некорректная порция записи")
        self.path, self.kind = path, record_type
        self.batch_rows, self.batch_bytes = batch_rows, batch_bytes
        _reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_links(path)
        self.temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        self.writer = pq.ParquetWriter(self.temporary, TABLE_SCHEMA, compression="zstd")
        self.rows: list[dict[str, Any]] = []
        self.size = 0
        self.previous: bytes | None = None

    def __enter__(self) -> TableWriter:
        return self

    def append(self, session_id: str, records: Sequence[Record]) -> None:
        key = id_bytes(session_id)
        if not records or (self.previous is not None and key <= self.previous):
            raise InputValidationError("Пустая, повторная или неупорядоченная сессия")
        self.previous = key
        for record in records:
            if (
                type(record) is not self.kind
                or getattr(record, "session_id", session_id) != session_id
            ):
                raise InputValidationError("Запись другого типа или сессии")
            payload = to_json(record)
            record_bytes = len(payload) + len(key)
            if self.rows and self.size + record_bytes > self.batch_bytes:
                self._flush()
            self.rows.append({"session_id": session_id, "record_json": payload})
            self.size += record_bytes
            if len(self.rows) >= self.batch_rows or self.size >= self.batch_bytes:
                self._flush()

    def _flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=TABLE_SCHEMA))
            self.rows.clear()
            self.size = 0

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                self._flush()
            self.writer.close()
            if exc_type is None:
                _publish(self.temporary, self.path)
        finally:
            self.writer.close()
            self.temporary.unlink(missing_ok=True)


def iter_sessions(
    path: Path,
    kind: type[R],
    *,
    batch_size: int = 4096,
    max_session_records: int = 1_000_000,
    max_session_bytes: int = 268_435_456,
) -> Iterator[tuple[str, tuple[R, ...]]]:
    if min(batch_size, max_session_records, max_session_bytes) < 1:
        raise InputValidationError("Некорректный бюджет чтения")
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.equals(TABLE_SCHEMA):
            raise InputValidationError("Несовместимая схема общей таблицы")
        current: str | None = None
        records: list[R] = []
        size = 0
        # Группа ограничена порцией записи либо одной крупной записью; сессию не режем.
        batches = (
            batch
            for group in range(parquet.num_row_groups)
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group])
        )
        for batch in batches:
            for sid, payload in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            ):
                key = id_bytes(sid)
                if current is not None and sid != current:
                    if key <= id_bytes(current):
                        raise InputValidationError("Повторная или неупорядоченная сессия")
                    yield current, tuple(records)
                    records, size = [], 0
                current = sid
                if not isinstance(payload, bytes):
                    raise InputValidationError("Пустая запись общей таблицы")
                size += len(payload)
                if len(records) >= max_session_records or size > max_session_bytes:
                    raise InputValidationError("Полная сессия превышает бюджет чтения")
                record = from_json(payload, kind)
                if getattr(record, "session_id", sid) != sid:
                    raise InputValidationError("Ключ таблицы не соответствует записи")
                records.append(record)
        if current is not None:
            yield current, tuple(records)


def table_info(
    path: Path, kind: type[Record], period: str | None = None, role: str | None = None
) -> dict[str, Any]:
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.equals(TABLE_SCHEMA):
            raise InputValidationError("Несовместимая схема общей таблицы")
        rows = parquet.metadata.num_rows
    return dict(
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        rows=rows,
        record_type=kind.__name__,
        period=period,
        data_role=role,
        **({"admitted": True} if role in {"prefix", "targets"} else {}),
    )


def _table_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        str(path) != relative
        or "\\" in relative
        or path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(
            r"(?:canonical/(?:A_G|A_N|B|C_smoke|C_select|E|D)|datasets/(?:A_G|A_N|B|C_smoke|C_select|E|D)/(?:prefixes|targets)|preparation/sessions)\.parquet",
            relative,
        )
    ):
        raise InputValidationError("Недопустимый путь таблицы подготовки")
    destination = root / relative
    _reject_links(destination)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise InputValidationError("Таблица вне каталога подготовки")
    return destination


def verify_tables(root: Path, report: dict[str, Any], paths: Sequence[str] | None = None) -> None:
    for relative in report["tables"] if paths is None else paths:
        path = _table_path(root, relative)
        info = report["tables"][relative]
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise InputValidationError("Содержимое таблицы подготовки изменилось")
        with pq.ParquetFile(path) as parquet:
            if (
                not parquet.schema_arrow.equals(TABLE_SCHEMA)
                or parquet.metadata.num_rows != info["rows"]
            ):
                raise InputValidationError("Схема или число строк таблицы изменилось")


def read_report(
    root: Path, report_sha256: str | None = None, *, verify_tables: bool = True
) -> dict[str, Any]:
    path = root / REPORT
    _reject_links(path)
    content = path.read_bytes()
    if report_sha256 is not None and sha256_bytes(content) != report_sha256:
        raise InputValidationError("Отчёт подготовки не совпал с закреплённым хешем")
    report: dict[str, Any] = strict_json_loads(content)
    if (
        not isinstance(report, dict)
        or report.get("format_version") != FORMAT_VERSION
        or report.get("status") != "completed"
        or report.get("contract_version") != PREPARATION_VERSION
        or report.get("evaluation_protocol") != EVALUATION_PROTOCOL
        or report.get("telemetry_complete", False) is not None
        or not isinstance(report.get("tables"), dict)
        or not report["tables"]
    ):
        raise InputValidationError("Нет совместимого завершённого набора подготовки")
    inputs = report.get("inputs")
    if (
        not isinstance(report.get("source_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", report["source_sha256"]) is None
        or not isinstance(inputs, dict)
        or set(inputs) != {"browsing", "search", "products", "profile", "plan"}
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in inputs.values()
        )
        or report.get("scope") not in ("actual", "synthetic")
        or not isinstance(report.get("run_id"), str)
        or not report["run_id"]
        or report.get("real_preflight") is not False
    ):
        raise InputValidationError("Неполные сведения о происхождении подготовки")
    for name in ("profile", "plan"):
        encoded = report.get(f"{name}_json")
        if not isinstance(encoded, str) or sha256_bytes(encoded.encode("utf-8")) != inputs[name]:
            raise InputValidationError("Встроенные настройки не совпали с хешами входов")
    for relative, info in report["tables"].items():
        _table_path(root, relative)
        if (
            not isinstance(info, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256")))
            or any(type(info.get(k)) is not int or info[k] < 0 for k in ("bytes", "rows"))
        ):
            raise InputValidationError("Некорректные сведения о таблице")
    if verify_tables:
        _verify_tables(root, report)
    return report


_verify_tables = verify_tables


def write_report(
    root: Path,
    report: dict[str, Any],
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    path = root / REPORT
    _reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(report))
        if before_publish:
            before_publish()
        _publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
