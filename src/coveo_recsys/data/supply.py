"""Потоковая проверка исходных CSV Coveo без подготовки сессий и целей."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO

from coveo_recsys.contracts import InputValidationError, require_int64
from coveo_recsys.data.io import _CSV_LOCK
from coveo_recsys.numerics import normalize_external_vector

_HEADERS: dict[str, tuple[str, ...]] = {
    "browsing_train.csv": (
        "session_id_hash",
        "event_type",
        "product_action",
        "product_sku_hash",
        "server_timestamp_epoch_ms",
        "hashed_url",
    ),
    "search_train.csv": (
        "session_id_hash",
        "query_vector",
        "clicked_skus_hash",
        "product_skus_hash",
        "server_timestamp_epoch_ms",
    ),
    "sku_to_content.csv": (
        "product_sku_hash",
        "description_vector",
        "category_hash",
        "image_vector",
        "price_bucket",
    ),
}
_VECTOR_COLUMNS = {
    "search_train.csv": ("query_vector",),
    "sku_to_content.csv": ("description_vector", "image_vector"),
}
_SEARCH_LIST_COLUMNS = ("clicked_skus_hash", "product_skus_hash")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EPOCH_MS = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_PRICE = re.compile(r"[+]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,3})?\Z")
_ACTIONS = {"", "detail", "add", "purchase", "remove"}
_EVENT_TYPES = {"pageview", "event", "event_product"}
_MAX_VECTOR_DIM = 4096
_PROFILE_KEYS = {
    "schema_version",
    "scope",
    "release",
    "encoding",
    "delimiter",
    "quotechar",
    "timestamp_unit",
    "files",
    "coverage_confirmed",
    "dimensions",
    "source_url",
    "header_policy",
    "timestamp_shifted_whole_weeks",
    "event_type_values",
    "product_action_values",
    "price_bucket",
    "search_list_format",
    "vector_format",
    "query_description_compatibility",
    "delivery_event_id",
}
_DOCUMENTED_VALUES: dict[str, Any] = {
    "source_url": "https://github.com/coveooss/SIGIR-ecom-data-challenge",
    "header_policy": "exact_order",
    "timestamp_shifted_whole_weeks": True,
    "price_bucket": "integral_decimal_or_int",
    "search_list_format": "python_single_quoted_string_list",
    "vector_format": "json_finite_numeric_array",
    "query_description_compatibility": "publisher_statement",
    "delivery_event_id": "not_present",
}


class SupplyScanError(InputValidationError):
    """Безопасная ошибка: код, имя таблицы, столбец и номер строки."""

    def __init__(
        self, code: str, file: str = "profile", column: str | None = None, line: int | None = None
    ) -> None:
        self.code, self.file, self.column, self.line = code, file, column, line
        parts = [code, file]
        if column is not None:
            parts.append(column)
        if line is not None:
            parts.append(f"line={line}")
        super().__init__(": ".join(parts))


@dataclass(frozen=True, slots=True)
class SupplyLimits:
    batch_size: int = 1024
    max_batch_chars: int = 4_194_304
    max_header_bytes: int = 4096
    max_record_bytes: int = 1_048_576
    max_field_chars: int = 1_048_576
    max_seconds: float = 7200.0
    max_temp_bytes: int = 67_108_864

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "max_batch_chars",
            "max_header_bytes",
            "max_record_bytes",
            "max_field_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SupplyScanError("invalid_limit", column=name)
        if type(self.max_temp_bytes) is not int or self.max_temp_bytes < 0:
            raise SupplyScanError("invalid_limit", column="max_temp_bytes")
        if (
            type(self.max_seconds) not in (int, float)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise SupplyScanError("invalid_limit", column="max_seconds")


def _parse_epoch_ms(text: str) -> int:
    """Разобрать целое число миллисекунд UTC без округления и вывода значения."""
    if not isinstance(text, str) or len(text) > 20 or _EPOCH_MS.fullmatch(text) is None:
        raise InputValidationError("Неверное время: ожидаются целые миллисекунды int64")
    try:
        return require_int64(int(text))
    except (InputValidationError, ValueError):
        raise InputValidationError("Время вне диапазона int64") from None


def _profile_files(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(profile, Mapping)
        or set(profile) - _PROFILE_KEYS
        or any(
            profile.get(key) != expected
            for key, expected in (
                ("schema_version", "coveo_supply_v1"),
                ("scope", "actual"),
                ("release", "1.0.0"),
                ("encoding", "utf-8"),
                ("delimiter", ","),
                ("quotechar", '"'),
                ("timestamp_unit", "epoch_ms_utc"),
            )
        )
        or profile.get("coverage_confirmed") is not False
    ):
        raise SupplyScanError("unsupported_profile")
    if any(
        key in profile and profile[key] != expected for key, expected in _DOCUMENTED_VALUES.items()
    ):
        raise SupplyScanError("unsupported_profile_claim")
    for key, allowed in (("event_type_values", _EVENT_TYPES), ("product_action_values", _ACTIONS)):
        values = profile.get(key)
        if values is not None and (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values) <= allowed
        ):
            raise SupplyScanError("invalid_profile_values", column=key)
    files = profile.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_HEADERS):
        raise SupplyScanError("invalid_profile_files")
    result: dict[str, dict[str, Any]] = {}
    for filename, expected_columns in _HEADERS.items():
        item = files[filename]
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "columns",
                "expected_size_bytes",
                "expected_sha256",
            }
            or item.get("columns") != list(expected_columns)
        ):
            raise SupplyScanError("invalid_profile_columns", filename)
        size = item.get("expected_size_bytes")
        digest = item.get("expected_sha256")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise SupplyScanError("invalid_profile_identity", filename)
        result[filename] = dict(item)
    dimensions = profile.get("dimensions")
    if dimensions is not None:
        if (
            not isinstance(dimensions, Mapping)
            or set(dimensions)
            != {
                "query_vector",
                "description_vector",
                "image_vector",
            }
            or any(
                type(value) is not int or not 0 < value <= _MAX_VECTOR_DIM
                for value in dimensions.values()
            )
        ):
            raise SupplyScanError("invalid_profile_dimensions")
    return result


def _progress(
    callback: Callable[[str, int, int], None] | None, filename: str, rows: int, size: int
) -> None:
    if callback is None:
        return
    try:
        callback(filename, rows, size)
    except Exception:
        raise SupplyScanError("progress_failed", filename) from None


def _check_time(deadline: float, filename: str) -> None:
    if time.monotonic() > deadline:
        raise SupplyScanError("time_limit", filename)


class _PhysicalLines:
    """Подаёт csv.reader строки, ограничивая байты логической записи до разбора."""

    def __init__(
        self,
        stream: BinaryIO,
        filename: str,
        digest: Any,
        header_limit: int,
        record_limit: int,
        deadline: float,
    ) -> None:
        self.stream, self.filename, self.digest = stream, filename, digest
        self.header_limit, self.record_limit, self.deadline = header_limit, record_limit, deadline
        self.record_bytes = 0
        self.total_bytes = 0
        self.header = True

    def __iter__(self) -> Iterator[str]:
        while True:
            _check_time(self.deadline, self.filename)
            cap = self.header_limit if self.header else self.record_limit
            remaining = cap - self.record_bytes
            if remaining < 0:
                raise SupplyScanError("record_limit", self.filename)
            raw = self.stream.readline(remaining + 1)
            if not raw:
                return
            if len(raw) > remaining:
                raise SupplyScanError(
                    "header_limit" if self.header else "record_limit", self.filename
                )
            self.record_bytes += len(raw)
            self.total_bytes += len(raw)
            self.digest.update(raw)
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise SupplyScanError("invalid_utf8", self.filename) from None

    def next_record(self, reader: Any, *, max_field_chars: int) -> list[str] | None:
        with _CSV_LOCK:
            previous = csv.field_size_limit(max_field_chars)
            try:
                row = next(reader, None)
            except csv.Error:
                raise SupplyScanError("invalid_csv", self.filename, line=reader.line_num) from None
            finally:
                csv.field_size_limit(previous)
        self.record_bytes = 0
        self.header = False
        return row


def _list_kind(text: str, filename: str, column: str, line: int) -> str:
    """Проверить литерал списка строк без исполнения и сохранения SKU."""
    if text == "":
        return "missing"
    length = len(text)
    index = 0

    def spaces(position: int) -> int:
        while position < length and text[position].isspace():
            position += 1
        return position

    index = spaces(index)
    if index >= length or text[index] != "[":
        raise SupplyScanError("invalid_search_list", filename, column, line)
    index = spaces(index + 1)
    if index < length and text[index] == "]":
        if spaces(index + 1) != length:
            raise SupplyScanError("invalid_search_list", filename, column, line)
        return "explicit_empty"
    while index < length:
        if text[index] != "'":
            raise SupplyScanError("invalid_search_list", filename, column, line)
        index += 1
        content = 0
        while index < length:
            char = text[index]
            if char == "\\":
                if index + 1 >= length or text[index + 1] not in "\\'":
                    raise SupplyScanError("invalid_search_list", filename, column, line)
                index += 2
                content += 1
                continue
            if char == "'":
                index += 1
                break
            if char in "\r\n":
                raise SupplyScanError("invalid_search_list", filename, column, line)
            index += 1
            content += 1
        else:
            raise SupplyScanError("invalid_search_list", filename, column, line)
        if not content:
            raise SupplyScanError("invalid_search_list", filename, column, line)
        index = spaces(index)
        if index >= length:
            break
        if text[index] == "]":
            if spaces(index + 1) == length:
                return "nonempty"
            break
        if text[index] != ",":
            break
        index = spaces(index + 1)
        if index < length and text[index] == "]":
            if spaces(index + 1) == length:
                return "nonempty"
            break
    raise SupplyScanError("invalid_search_list", filename, column, line)


def _vector_reason(value: str, dimension: int | None) -> tuple[str, int | None]:
    if value == "" or value == "null":
        return "vector_absent", dimension
    try:
        raw = json.loads(value, parse_constant=lambda _: None)
        if raw is None:
            return "vector_absent" if value == "null" else "vector_invalid", dimension
        if (
            not isinstance(raw, list)
            or not 0 < len(raw) <= _MAX_VECTOR_DIM
            or (dimension is not None and len(raw) != dimension)
            or any(type(item) not in (int, float) for item in raw)
        ):
            return "vector_invalid", dimension
        values = [float(item) for item in raw]
        if any(not math.isfinite(item) for item in values):
            return "vector_invalid", dimension
        observed = dimension or len(values)
        result = normalize_external_vector(values, observed, epsilon=1e-12)
        if not result.mask:
            return result.reason or "vector_invalid", dimension
        return "valid", observed
    except (ValueError, TypeError, OverflowError, RecursionError):
        return "vector_invalid", dimension


def _price_bucket(value: str, filename: str, line: int) -> None:
    if value == "":
        return
    if len(value) > 64 or _PRICE.fullmatch(value) is None:
        raise SupplyScanError("invalid_price_bucket", filename, "price_bucket", line)
    try:
        bucket = Decimal(value)
        if not bucket.is_finite() or bucket < 0 or bucket != bucket.to_integral_value():
            raise ValueError
    except (InvalidOperation, ValueError):
        raise SupplyScanError("invalid_price_bucket", filename, "price_bucket", line) from None


def _row_summary(
    filename: str,
    row: Mapping[str, str],
    line: int,
    dimensions: dict[str, int | None],
    reasons: dict[str, dict[str, int]],
    search_lists: dict[str, dict[str, int]],
    event_types: set[str],
    actions: set[str],
) -> int | None:
    key = "product_sku_hash" if filename == "sku_to_content.csv" else "session_id_hash"
    if not row[key]:
        raise SupplyScanError("missing_id", filename, key, line)
    if filename == "browsing_train.csv":
        if row["event_type"] not in event_types:
            raise SupplyScanError("invalid_event_type", filename, "event_type", line)
        action = row["product_action"]
        if action not in actions:
            raise SupplyScanError("invalid_product_action", filename, "product_action", line)
        if action and not row["product_sku_hash"]:
            raise SupplyScanError("missing_action_sku", filename, "product_sku_hash", line)
    elif filename == "search_train.csv":
        for column in _SEARCH_LIST_COLUMNS:
            kind = _list_kind(row[column], filename, column, line)
            if kind in {"missing", "explicit_empty"}:
                search_lists[column][kind] += 1
    else:
        _price_bucket(row["price_bucket"], filename, line)
    for column in _VECTOR_COLUMNS.get(filename, ()):
        reason, dimensions[column] = _vector_reason(row[column], dimensions[column])
        reasons[column][reason] += 1
    if filename == "sku_to_content.csv":
        return None
    try:
        return _parse_epoch_ms(row["server_timestamp_epoch_ms"])
    except InputValidationError:
        raise SupplyScanError(
            "invalid_timestamp", filename, "server_timestamp_epoch_ms", line
        ) from None


def inspect_supply(
    input_dir: Path,
    profile: Mapping[str, Any],
    *,
    limits: SupplyLimits,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Проверить все исходные строки; результат не даёт допуска к подготовке."""
    if not isinstance(limits, SupplyLimits):
        raise SupplyScanError("invalid_limits")
    profile_files = _profile_files(profile)
    expected_dimensions = profile.get("dimensions") or {}
    event_types = set(profile.get("event_type_values", ("event_product", "pageview")))
    actions = set(profile.get("product_action_values", _ACTIONS))
    deadline = time.monotonic() + limits.max_seconds
    paths = {filename: input_dir / filename for filename in _HEADERS}
    identities: dict[str, tuple[int, int, int, int]] = {}
    for filename, path in paths.items():
        try:
            stat = path.stat()
        except OSError:
            raise SupplyScanError("file_unavailable", filename) from None
        if not path.is_file() or stat.st_size != profile_files[filename]["expected_size_bytes"]:
            raise SupplyScanError("size_mismatch", filename)
        identities[filename] = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)

    files: dict[str, Any] = {}
    for filename, columns in _HEADERS.items():
        _check_time(deadline, filename)
        digest = hashlib.sha256()
        rows = 0
        lowest: int | None = None
        highest: int | None = None
        dimensions: dict[str, int | None] = {
            column: expected_dimensions.get(column) for column in _VECTOR_COLUMNS.get(filename, ())
        }
        reasons = {
            column: {
                key: 0
                for key in (
                    "valid",
                    "vector_absent",
                    "vector_invalid",
                    "vector_zero",
                )
            }
            for column in dimensions
        }
        search_lists = (
            {column: {"missing": 0, "explicit_empty": 0} for column in _SEARCH_LIST_COLUMNS}
            if filename == "search_train.csv"
            else {}
        )
        batch_rows = 0
        batch_chars = 0
        last_progress = time.monotonic()
        _progress(progress, filename, 0, 0)
        try:
            with paths[filename].open("rb") as stream:
                lines = _PhysicalLines(
                    stream,
                    filename,
                    digest,
                    limits.max_header_bytes,
                    limits.max_record_bytes,
                    deadline,
                )
                reader = csv.reader(lines, delimiter=",", quotechar='"', strict=True)
                header = lines.next_record(reader, max_field_chars=limits.max_field_chars)
                if header != list(columns):
                    raise SupplyScanError("header_mismatch", filename, line=reader.line_num)
                while (
                    values := lines.next_record(reader, max_field_chars=limits.max_field_chars)
                ) is not None:
                    _check_time(deadline, filename)
                    if len(values) != len(columns):
                        raise SupplyScanError("row_width", filename, line=reader.line_num)
                    if any(
                        len(value) > limits.max_field_chars or "\x00" in value for value in values
                    ):
                        raise SupplyScanError("field_limit", filename, line=reader.line_num)
                    chars = sum(len(value) for value in values)
                    if chars > limits.max_batch_chars:
                        raise SupplyScanError("batch_limit", filename, line=reader.line_num)
                    if (
                        batch_rows >= limits.batch_size
                        or batch_chars + chars > limits.max_batch_chars
                    ):
                        batch_rows, batch_chars = 0, 0
                    batch_rows += 1
                    batch_chars += chars
                    row = dict(zip(columns, values, strict=True))
                    timestamp = _row_summary(
                        filename,
                        row,
                        reader.line_num,
                        dimensions,
                        reasons,
                        search_lists,
                        event_types,
                        actions,
                    )
                    if timestamp is not None:
                        lowest = timestamp if lowest is None else min(lowest, timestamp)
                        highest = timestamp if highest is None else max(highest, timestamp)
                    rows += 1
                    now = time.monotonic()
                    if now - last_progress >= 10:
                        _progress(progress, filename, rows, lines.total_bytes)
                        last_progress = now
                size = lines.total_bytes
        except OSError:
            raise SupplyScanError("file_unavailable", filename) from None
        if rows == 0:
            raise SupplyScanError("empty_file", filename)
        if (
            size != profile_files[filename]["expected_size_bytes"]
            or digest.hexdigest() != profile_files[filename]["expected_sha256"]
        ):
            raise SupplyScanError("content_mismatch", filename)
        files[filename] = {
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "rows": rows,
            "min_timestamp_ms": lowest,
            "max_timestamp_ms": highest,
            "timestamp_status": "not_applicable"
            if filename == "sku_to_content.csv"
            else "observed_range_only",
            "vector_dimensions": dimensions,
            "vector_reasons": reasons,
            "search_lists": search_lists,
        }
        _progress(progress, filename, rows, size)

    for filename, path in paths.items():
        try:
            stat = path.stat()
        except OSError:
            raise SupplyScanError("file_changed", filename) from None
        if (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino) != identities[filename]:
            raise SupplyScanError("file_changed", filename)
    status = (
        "confirmed"
        if all(
            files[filename]["vector_reasons"][column]["valid"] > 0
            for filename, columns in _VECTOR_COLUMNS.items()
            for column in columns
        )
        else "unknown"
    )
    return {
        "scan_complete": True,
        "format_status": status,
        "coverage_confirmed": False,
        "coverage_status": "unknown",
        "coverage_reason": "no_independent_telemetry_evidence",
        "unknown_intervals": "not_determined",
        "preflight_ready": False,
        "real_prepare_ready": False,
        "files": files,
    }
