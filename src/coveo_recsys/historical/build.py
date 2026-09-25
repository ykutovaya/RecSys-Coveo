"""Согласованная сборка принятой истории, без допуска команды."""

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.historical import (
    HistoricalConfig,
    HistoricalInput,
    HistoricalPackage,
    MetadataSource,
)
from coveo_recsys.historical.catalog import build_catalog
from coveo_recsys.historical.content import build_content
from coveo_recsys.historical.graph import build_graph
from coveo_recsys.historical.statistics import build_statistics


def build_historical(
    admitted_A_G: HistoricalInput,
    metadata: MetadataSource,
    config: HistoricalConfig,
    *,
    work_dir: Path,
    check_budget: Callable[[], None] | None = None,
) -> HistoricalPackage:
    if admitted_A_G.snapshot_time_ms != config.snapshot_time_ms:
        raise InputValidationError("Время снимка не соответствует принятому A_G")
    hashes = dict(admitted_A_G.input_hashes)
    for name, digest in metadata.input_hashes.items():
        if name in hashes and hashes[name] != digest:
            raise InputValidationError("Несогласованные хеши входов")
        hashes[name] = digest
    catalog = build_catalog(admitted_A_G.iter_sessions())
    if check_budget:
        check_budget()
    graph = build_graph(admitted_A_G, catalog, config, work_dir=work_dir, check_budget=check_budget)
    statistics = build_statistics(admitted_A_G, catalog, config)
    if check_budget:
        check_budget()
    content = build_content(catalog, metadata, config)
    if check_budget:
        check_budget()
    return HistoricalPackage(
        catalog,
        graph,
        statistics,
        content,
        config.snapshot_time_ms,
        MappingProxyType(hashes),
        MappingProxyType(
            {
                "description_dimension": metadata.description_dimension,
                "image_dimension": metadata.image_dimension,
                "query_description_compatibility": metadata.query_description_compatibility,
                "static_availability": metadata.static_availability,
            }
        ),
    )
