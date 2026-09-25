"""Порционное чтение искусственного и фактического форматов по явному профилю."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from coveo_recsys.artifacts import sha256_file, strict_json_loads
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.records import EventType, RawEvent, SearchPayload
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.numerics import normalize_external_vector

FILES = {
    "browsing": "browsing_train.csv",
    "search": "search_train.csv",
    "products": "sku_to_content.csv",
}
FIELDS = {
    "browsing": {"session_id", "event_id", "timestamp", "action", "sku"},
    "search": {"session_id", "event_id", "timestamp", "query", "results", "clicks"},
    "products": {"sku", "description", "image", "category", "price_bucket"},
}
_ACTUAL_COLUMNS = {
    "browsing": {
        "session_id": "session_id_hash",
        "event_type": "event_type",
        "action": "product_action",
        "sku": "product_sku_hash",
        "timestamp": "server_timestamp_epoch_ms",
        "url": "hashed_url",
    },
    "search": {
        "session_id": "session_id_hash",
        "query": "query_vector",
        "clicks": "clicked_skus_hash",
        "results": "product_skus_hash",
        "timestamp": "server_timestamp_epoch_ms",
    },
    "products": {
        "sku": "product_sku_hash",
        "description": "description_vector",
        "category": "category_hash",
        "image": "image_vector",
        "price_bucket": "price_bucket",
    },
}
_CSV_LOCK = threading.Lock()
DEFAULT_MAX_BATCH_CHARS = 4_194_304


def _input_json(content: bytes) -> Any:
    try:
        return strict_json_loads(content)
    except RecursionError as exc:
        raise InputValidationError("Превышена вложенность JSON") from exc


@dataclass(frozen=True)
class CsvProfile:
    namespace: str
    columns: Mapping[str, Mapping[str, str]]
    dimensions: Mapping[str, int]
    sha256: str
    scope: str = "synthetic"
    max_field_chars: int = 1_048_576
    max_row_chars: int = 4_194_304
    expected_files: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    event_types: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "columns",
            MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.columns.items()}),
        )
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(
            self,
            "expected_files",
            MappingProxyType(
                {
                    table: MappingProxyType(
                        {
                            key: tuple(value) if key == "columns" else value
                            for key, value in item.items()
                        }
                    )
                    for table, item in self.expected_files.items()
                }
            ),
        )


def load_profile(path: Path) -> CsvProfile:
    content = path.read_bytes()
    data = _input_json(content)
    return parse_profile(data, sha256=hashlib.sha256(content).hexdigest())


def parse_profile(data: Any, *, sha256: str) -> CsvProfile:
    """Проверить уже прочитанный профиль без изменения правил его формата."""
    if isinstance(data, dict) and data.get("schema_version") == "coveo_supply_v1":
        from .supply import _profile_files

        files = _profile_files(data)
        dimensions = data.get("dimensions")
        if not isinstance(dimensions, dict):
            raise InputValidationError("Фактический профиль должен задавать размерности векторов")
        return CsvProfile(
            namespace="coveo-sigir-1.0.0",
            columns=_ACTUAL_COLUMNS,
            dimensions={
                key: dimensions[key + "_vector"] for key in ("query", "description", "image")
            },
            sha256=sha256,
            scope="actual",
            expected_files={table: files[filename] for table, filename in FILES.items()},
            event_types=tuple(data.get("event_type_values", ("event_product", "pageview"))),
            actions=tuple(
                data.get("product_action_values", ("", "detail", "add", "purchase", "remove"))
            ),
        )
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "schema_version",
            "scope",
            "source_namespace",
            "columns",
            "dimensions",
            "max_field_chars",
            "max_row_chars",
        }
        or data["schema_version"] != "1"
        or data["scope"] != "synthetic"
    ):
        raise InputValidationError("Поддерживается только явный искусственный профиль версии 1")
    id_bytes(data["source_namespace"])
    if not isinstance(data["columns"], dict) or set(data["columns"]) != set(FILES):
        raise InputValidationError("Нужны схемы всех трёх таблиц")
    for table, fields in FIELDS.items():
        mapping = data["columns"][table]
        if not isinstance(mapping, dict) or set(mapping) != fields:
            raise InputValidationError(f"Неверное соответствие столбцов {table}")
        if any(not isinstance(v, str) or not v for v in mapping.values()) or len(
            set(mapping.values())
        ) != len(mapping):
            raise InputValidationError("Повтор или пустое имя столбца")
    if not isinstance(data["dimensions"], dict) or set(data["dimensions"]) != {
        "query",
        "description",
        "image",
    }:
        raise InputValidationError("Нужны три независимые размерности")
    for value in (*data["dimensions"].values(), data["max_field_chars"], data["max_row_chars"]):
        require_int64(value)
        if value <= 0:
            raise InputValidationError("Размер должен быть положительным")
    return CsvProfile(
        data["source_namespace"],
        data["columns"],
        data["dimensions"],
        sha256,
        data["scope"],
        data["max_field_chars"],
        data["max_row_chars"],
    )


def timestamp(value: str) -> int:
    if len(value) > 20 or not re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        raise InputValidationError("Время: нужны целые миллисекунды без округления")
    return require_int64(int(value))


def vector_payload(value: str, dimension: int) -> SearchPayload:
    if not value or value == "null":
        return SearchPayload(None, "vector_absent")
    try:
        raw = _input_json(value.encode("utf-8"))
        if raw is None:
            return SearchPayload(None, "vector_absent")
        if not isinstance(raw, list) or any(type(v) not in (int, float) for v in raw):
            return SearchPayload(None, "vector_invalid")
        raw = [float(v) for v in raw]
        normalized = normalize_external_vector(raw, dimension)
        if not normalized.mask:
            return SearchPayload(None, normalized.reason)
        # Канонизация сравнивает исходные координаты, не единичное направление.
        return SearchPayload(tuple(0.0 if float(v) == 0 else float(v) for v in raw), None)
    except (ValueError, TypeError, OverflowError):
        return SearchPayload(None, "vector_invalid")


def row_batches(
    path: Path,
    table: str,
    profile: CsvProfile,
    *,
    batch_size: int = 1024,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    progress: Callable[[str, int], None] | None = None,
) -> Generator[tuple[tuple[int, dict[str, str]], ...], None, None]:
    """Целые строки, ограниченные числом и суммой длин декодированных полей."""
    require_int64(batch_size)
    require_int64(max_batch_chars)
    if batch_size <= 0 or max_batch_chars <= 0 or table not in FILES:
        raise InputValidationError("Неверная таблица или размер порции")
    mapping = profile.columns[table]

    physical_lines: Any = None

    def next_row(reader: Any) -> list[str] | None:
        if physical_lines is not None:
            result: list[str] | None = physical_lines.next_record(
                reader, max_field_chars=profile.max_field_chars
            )
            return result
        with _CSV_LOCK:
            previous_limit = csv.field_size_limit(profile.max_field_chars)
            try:
                return next(reader, None)
            finally:
                csv.field_size_limit(previous_limit)

    try:
        with path.open("rb") as binary:
            if profile.scope == "actual":
                from .supply import _PhysicalLines

                physical_lines = _PhysicalLines(
                    binary,
                    FILES[table],
                    hashlib.sha256(),
                    4096,
                    profile.max_row_chars,
                    float("inf"),
                )
                reader = csv.reader(physical_lines, strict=True)
            else:
                stream = io.TextIOWrapper(binary, encoding="utf-8", newline="")
                reader = csv.reader(stream, strict=True)
            header = next_row(reader)
            if (
                header is None
                or len(header) != len(set(header))
                or set(header) != set(mapping.values())
                or (
                    profile.scope == "actual"
                    and tuple(header) != profile.expected_files[table]["columns"]
                )
            ):
                raise InputValidationError(f"Несовместимая схема {table}")
            batch: list[tuple[int, dict[str, str]]] = []
            batch_chars = 0
            total_rows = 0
            while (row := next_row(reader)) is not None:
                row_chars = sum(map(len, row))
                if (
                    len(row) != len(header)
                    or row_chars > profile.max_row_chars
                    or (profile.scope == "actual" and any("\x00" in value for value in row))
                ):
                    raise InputValidationError(f"Неверная строка {table}:{reader.line_num}")
                if row_chars > max_batch_chars:
                    raise InputValidationError(
                        f"Строка превышает предел символов порции {table}:{reader.line_num}"
                    )
                if batch and batch_chars + row_chars > max_batch_chars:
                    if progress is not None:
                        progress(table, total_rows)
                    yield tuple(batch)
                    batch = []
                    batch_chars = 0
                physical = dict(zip(header, row, strict=True))
                batch.append((reader.line_num, {k: physical[v] for k, v in mapping.items()}))
                batch_chars += row_chars
                total_rows += 1
                if len(batch) == batch_size or batch_chars == max_batch_chars:
                    if progress is not None:
                        progress(table, total_rows)
                    yield tuple(batch)
                    batch = []
                    batch_chars = 0
            if batch:
                if progress is not None:
                    progress(table, total_rows)
                yield tuple(batch)
    except (csv.Error, UnicodeError) as exc:
        raise InputValidationError(f"Повреждённый CSV {table}: {exc}") from exc


def iter_events(
    path: Path,
    table: str,
    profile: CsvProfile,
    *,
    batch_size: int = 1024,
    file_sha256: str | None = None,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    session_ids: frozenset[str] | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> Iterator[RawEvent]:
    if table not in {"browsing", "search"}:
        raise InputValidationError("Таблица не содержит события")
    digest = file_sha256 or sha256_file(path)
    _check_expected_input(path, table, profile, digest)
    for batch in row_batches(
        path,
        table,
        profile,
        batch_size=batch_size,
        max_batch_chars=max_batch_chars,
        progress=progress,
    ):
        for line, row in batch:
            id_bytes(row["session_id"])
            if session_ids is not None and row["session_id"] not in session_ids:
                continue
            time_ms = timestamp(row["timestamp"])
            origin = f"{FILES[table]}:{digest}:line={line}"
            transport = row.get("event_id") or None
            event_id = transport or hashlib.sha256(origin.encode("utf-8")).hexdigest()
            if table == "search":
                # R/C проверяются по профилю, но не становятся датированными событиями.
                for field in ("results", "clicks"):
                    if profile.scope == "actual":
                        from .supply import _list_kind

                        _list_kind(row[field], FILES[table], profile.columns[table][field], line)
                        continue
                    values = _input_json((row[field] or "[]").encode("utf-8"))
                    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                        raise InputValidationError("Поисковый список должен содержать строки")
                event_type, sku = EventType.SEARCH, None
                search = vector_payload(row["query"], profile.dimensions["query"])
            else:
                if profile.scope == "actual" and (
                    row["event_type"] not in profile.event_types
                    or row["action"] not in profile.actions
                ):
                    raise InputValidationError(f"Несовместимый тип или действие browsing:{line}")
                try:
                    event_type = EventType(row["action"] or "page")
                except ValueError as exc:
                    raise InputValidationError("Неизвестное товарное действие") from exc
                if event_type == EventType.SEARCH:
                    raise InputValidationError("Поиск должен поступать из таблицы поиска")
                sku, search = row["sku"] or None, None
            yield RawEvent(
                profile.namespace,
                row["session_id"],
                event_id,
                time_ms,
                time_ms,
                event_type,
                sku,
                search,
                origin,
                transport,
                (origin,),
            )


def _check_expected_input(path: Path, table: str, profile: CsvProfile, digest: str) -> None:
    expected = profile.expected_files.get(table)
    if expected is not None and (
        digest != expected["expected_sha256"]
        or path.stat().st_size != expected["expected_size_bytes"]
    ):
        raise InputValidationError(f"Вход не соответствует профилю: {FILES[table]}")


def inspect_inputs(
    paths: Mapping[str, Path],
    profile: CsvProfile,
    *,
    batch_size: int = 1024,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    if set(paths) != set(FILES) or len({p.resolve() for p in paths.values()}) != 3:
        raise InputValidationError("Нужны три разных входных файла")
    result: dict[str, Any] = {
        "schema_version": "1",
        "scope": profile.scope,
        "profile_sha256": profile.sha256,
        "files": {},
    }
    for table, path in paths.items():
        before = sha256_file(path)
        _check_expected_input(path, table, profile, before)
        size = path.stat().st_size
        count, earliest, latest = 0, None, None
        serialized_bytes = 0
        invalid_vectors: dict[str, int] = {}
        if table != "products":
            for event in iter_events(
                path,
                table,
                profile,
                batch_size=batch_size,
                file_sha256=before,
                max_batch_chars=max_batch_chars,
                progress=progress,
            ):
                count += 1
                serialized_bytes += len(to_json(event)) + 1
                earliest = (
                    event.event_time_ms if earliest is None else min(earliest, event.event_time_ms)
                )
                latest = event.event_time_ms if latest is None else max(latest, event.event_time_ms)
                if event.search is not None and event.search.reason:
                    key = event.search.reason
                    invalid_vectors[key] = invalid_vectors.get(key, 0) + 1
        else:
            for batch in row_batches(
                path,
                table,
                profile,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
                progress=progress,
            ):
                for line, row in batch:
                    count += 1
                    id_bytes(row["sku"])
                    if profile.scope == "actual":
                        from .supply import _price_bucket

                        _price_bucket(row["price_bucket"], FILES[table], line)
                    for modality in ("description", "image"):
                        payload = vector_payload(row[modality], profile.dimensions[modality])
                        if payload.reason:
                            key = modality + ":" + payload.reason
                            invalid_vectors[key] = invalid_vectors.get(key, 0) + 1
        if before != sha256_file(path) or size != path.stat().st_size:
            raise InputValidationError("Вход изменился во время чтения")
        result["files"][table] = {
            "name": FILES[table],
            "sha256": before,
            "size_bytes": size,
            "rows": count,
            "observed_min_ms": earliest,
            "observed_max_ms": latest,
            "coverage_confirmed": False,
            "vector_reasons": invalid_vectors,
            "serialized_event_bytes": serialized_bytes,
        }
    result["memory_plan"] = {
        "batch_rows": batch_size,
        "max_field_chars": profile.max_field_chars,
        "max_row_chars": profile.max_row_chars,
        "max_batch_chars": max_batch_chars,
        "batch_char_measure": "sum_of_decoded_field_lengths",
        "max_pending_row_chars": min(profile.max_row_chars, max_batch_chars),
        "grouping": "external_merge_sort",
        "note": "Диапазон времён не подтверждает полноту; нужен отдельный реестр покрытия",
    }
    result["disk_plan"] = {
        "input_bytes": sum(item["size_bytes"] for item in result["files"].values()),
        "sorting_spill_estimate_bytes": 3
        * sum(item["serialized_event_bytes"] for item in result["files"].values()),
        "estimate_only": True,
        "not_included": ["prepared_datasets", "artifact_copies", "filesystem_overhead"],
    }
    return result
