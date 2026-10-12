"""Раздельные модальности и словарь исторического каталога."""

from collections.abc import Sequence

import numpy as np

from coveo_recsys.contracts import InputValidationError, NumericError, id_bytes, require_int64
from coveo_recsys.contracts.historical import (
    HistoricalCatalog,
    HistoricalConfig,
    HistoricalContent,
    MetadataSource,
    Modality,
)
from coveo_recsys.numerics import normalize_external_vector


def build_content(
    catalog: HistoricalCatalog, metadata: MetadataSource, config: HistoricalConfig
) -> HistoricalContent:
    """Рассчитать модальности только каталога и закрепить его словарь категорий."""
    return _content(catalog.skus, metadata, config, categories=None)


def read_observed_content(
    skus: Sequence[str],
    metadata: MetadataSource,
    config: HistoricalConfig,
    *,
    categories: tuple[str, ...],
) -> HistoricalContent:
    """Прочитать наблюдаемые SKU в общем порядке, сохраняя словарь A_G."""
    ordered = HistoricalCatalog(tuple(skus)).skus
    if categories != tuple(sorted(set(categories), key=id_bytes)):
        raise InputValidationError("Словарь категорий требует уникальные метки в порядке UTF-8")
    return _content(ordered, metadata, config, categories=categories)


def _content(
    skus: tuple[str, ...],
    metadata: MetadataSource,
    config: HistoricalConfig,
    *,
    categories: tuple[str, ...] | None,
) -> HistoricalContent:
    dimensions = (metadata.description_dimension, metadata.image_dimension)
    for dimension in dimensions:
        if require_int64(dimension, nonnegative=True) == 0:
            raise InputValidationError("Ожидаемая размерность должна быть положительной")

    count = len(skus)
    positions = {sku: index for index, sku in enumerate(skus)}
    description = np.zeros((count, dimensions[0]), dtype=np.float64)
    image = np.zeros((count, dimensions[1]), dtype=np.float64)
    description_mask = np.zeros(count, dtype=np.bool_)
    image_mask = np.zeros(count, dtype=np.bool_)
    description_reasons: list[str | None] = ["vector_absent"] * count
    image_reasons: list[str | None] = ["vector_absent"] * count
    category_values: list[str | None] = [None] * count
    price_bucket = np.zeros(count, dtype=np.int64)
    price_mask = np.zeros(count, dtype=np.bool_)
    seen: set[str] = set()
    for record in metadata.read(skus):
        id_bytes(record.sku)
        if record.sku not in positions or record.sku in seen:
            raise InputValidationError("Метаданные содержат незапрошенный или повторный SKU")
        seen.add(record.sku)
        index = positions[record.sku]
        for raw, dimension, vectors, mask, reasons in (
            (record.description, dimensions[0], description, description_mask, description_reasons),
            (record.image, dimensions[1], image, image_mask, image_reasons),
        ):
            normalized = normalize_external_vector(raw, dimension, epsilon=config.vector_epsilon)
            if not np.all(np.isfinite(normalized.vector)):
                raise NumericError("Неконечный внутренний результат нормировки модальности")
            vectors[index] = normalized.vector
            mask[index] = normalized.mask
            reasons[index] = normalized.reason
        if record.category is not None and record.category != "":
            id_bytes(record.category)
            category_values[index] = record.category
        if record.price_bucket is not None:
            price_bucket[index] = require_int64(record.price_bucket, nonnegative=True)
            price_mask[index] = True

    if categories is None:
        categories = tuple(sorted({value for value in category_values if value}, key=id_bytes))
    category_lookup = {category: index + 1 for index, category in enumerate(categories)}
    category_codes = np.array(
        [category_lookup.get(value, 0) if value is not None else 0 for value in category_values],
        dtype=np.int64,
    )
    return HistoricalContent(
        Modality(description, description_mask, tuple(description_reasons)),
        Modality(image, image_mask, tuple(image_reasons)),
        categories,
        category_codes,
        price_bucket,
        price_mask,
    )
