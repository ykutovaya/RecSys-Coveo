"""Повторяемое чтение закреплённой поставки метаданных."""

import re
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from coveo_recsys.artifacts import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.historical import MetadataRecord, MetadataSource

from .io import FILES, _check_expected_input, parse_profile, row_batches
from .supply import _price_bucket


def _vector(value: str) -> Any:
    if not value:
        return None
    try:
        return strict_json_loads(value)
    except (ValueError, RecursionError):
        # Сохраняем непригодность, не подменяя её отсутствием. Общая нормировка
        # проверит координаты, размерность и настраиваемый порог уже в ядре.
        return value


def open_metadata_source(
    path: Path, profile: dict[str, Any], *, expected_sha256: str
) -> MetadataSource:
    """Открыть поставку; каждый проход проверяет хеш до выдачи выбранных записей.

    CSV читается порциями. В памяти сохраняются лишь запрошенные записи;
    новые товары не образуют общий словарь и не расширяют каталог A_G.
    """
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise InputValidationError("Метаданные требуют закреплённый SHA-256")
    profile_sha256 = sha256_bytes(canonical_json_bytes(profile))
    parsed = parse_profile(profile, sha256=profile_sha256)
    path = path.resolve()
    size = path.stat().st_size

    def verify() -> None:
        digest = sha256_file(path)
        if digest != expected_sha256 or path.stat().st_size != size:
            raise InputValidationError("Хеш или размер поставки метаданных изменился")
        _check_expected_input(path, "products", parsed, digest)

    verify()

    def read(skus: Sequence[str]) -> Iterator[MetadataRecord]:
        requested = set(skus)
        for sku in requested:
            id_bytes(sku)
        verify()
        rows: dict[str, MetadataRecord] = {}
        for batch in row_batches(path, "products", parsed):
            for line, row in batch:
                sku = row["sku"]
                if sku not in requested:
                    continue
                if sku in rows:
                    raise InputValidationError("Повтор SKU в поставке метаданных")
                _price_bucket(row["price_bucket"], FILES["products"], line)
                price = (
                    require_int64(int(Decimal(row["price_bucket"])), nonnegative=True)
                    if row["price_bucket"]
                    else None
                )
                category = row["category"] or None
                if category is not None:
                    id_bytes(category)
                rows[sku] = MetadataRecord(
                    sku, _vector(row["description"]), _vector(row["image"]), category, price
                )
        verify()
        yield from rows.values()

    return MetadataSource(
        read,
        parsed.dimensions["description"],
        parsed.dimensions["image"],
        MappingProxyType(
            {"metadata_sha256": expected_sha256, "metadata_profile_sha256": profile_sha256}
        ),
        profile.get("query_description_compatibility", "unknown"),
        static_availability="unknown",
    )
