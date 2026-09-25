"""Независимые искусственные входы для проверки поставки Coveo.

Рецепт: csv.writer кодирует три таблицы в UTF-8; hashlib считает точные байты.
Содержимое не заимствовано из поставки. Эталонные размеры и хеши вычисляются
из записанных байтов, а не через проверяемый читатель.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any

BROWSING = "browsing_train.csv"
SEARCH = "search_train.csv"
PRODUCTS = "sku_to_content.csv"

HEADERS = {
    BROWSING: [
        "session_id_hash",
        "event_type",
        "product_action",
        "product_sku_hash",
        "server_timestamp_epoch_ms",
        "hashed_url",
    ],
    SEARCH: [
        "session_id_hash",
        "query_vector",
        "clicked_skus_hash",
        "product_skus_hash",
        "server_timestamp_epoch_ms",
    ],
    PRODUCTS: [
        "product_sku_hash",
        "description_vector",
        "category_hash",
        "image_vector",
        "price_bucket",
    ],
}

ROWS = {
    BROWSING: [
        ["0001", "event_product", "detail", "sku-A", "0", ""],
        ["сессия", "pageview", "", "", "1", ""],
        ["0001", "event_product", "purchase", "sku-B", "2", ""],
    ],
    SEARCH: [
        ["0001", "[3,4]", "['sku-B']", "['sku-A']", "1"],
        ["сессия", "[5,12]", "", "[]", "2"],
    ],
    PRODUCTS: [
        ["sku-A", "[3,4]", "cat", "[1,2,3]", "3.0"],
        ["sku-B", "[5,12]", "", "[0,0,2]", ""],
    ],
}


def csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def synthetic_supply(
    root: Path,
    *,
    headers: dict[str, list[str]] | None = None,
    rows: dict[str, list[list[str]]] | None = None,
    raw: dict[str, bytes] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Создать три маленьких CSV и профиль с независимыми SHA-256."""
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    selected_headers = headers or HEADERS
    selected_rows = rows or ROWS
    raw = raw or {}
    files: dict[str, dict[str, Any]] = {}
    for name in (BROWSING, SEARCH, PRODUCTS):
        content = raw.get(name, csv_bytes(selected_headers[name], selected_rows[name]))
        (input_dir / name).write_bytes(content)
        files[name] = {
            "columns": selected_headers[name],
            "expected_size_bytes": len(content),
            "expected_sha256": hashlib.sha256(content).hexdigest(),
        }
    return input_dir, {
        "schema_version": "coveo_supply_v1",
        "scope": "actual",
        "release": "1.0.0",
        "encoding": "utf-8",
        "delimiter": ",",
        "quotechar": '"',
        "timestamp_unit": "epoch_ms_utc",
        "price_bucket": "integral_decimal_or_int",
        "files": files,
        "coverage_confirmed": False,
    }
