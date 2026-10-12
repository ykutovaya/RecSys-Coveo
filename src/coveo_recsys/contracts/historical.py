"""Контракты исторического пакета, независимые от формата подготовки."""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import InputValidationError, id_bytes, require_int64
from .records import CanonicalBlock

HISTORICAL_VERSION = "historical_package_v1"
RELATIONS = ("detail_detail", "detail_add", "detail_purchase")
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Session = tuple[CanonicalBlock, ...]


@dataclass(frozen=True)
class HistoricalCatalog:
    skus: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.skus != tuple(sorted(set(self.skus), key=id_bytes)):
            raise InputValidationError("Каталог требует уникальные SKU в порядке UTF-8")
        for sku in self.skus:
            id_bytes(sku)


@dataclass(frozen=True)
class HistoricalConfig:
    snapshot_time_ms: int
    graph_max_group_distance: int = 10
    graph_window_ms: int = 1_800_000
    graph_tau_ms: float = 600_000.0
    graph_min_support: int = 3
    graph_max_neighbors: int = 100
    graph_regularization: float = 10.0
    popularity_smoothing: float = 1.0
    vector_epsilon: float = 1e-12
    buffer_records: int = 100_000
    merge_fan_in: int = 16
    max_pairs: int = 100_000_000
    max_group_size: int = 100_000
    max_temp_bytes: int = 2_147_483_648
    max_result_bytes: int = 1_073_741_824
    max_memory_bytes: int = 1_610_612_736
    max_seconds: float = 3600.0
    schema_version: str = HISTORICAL_VERSION
    id_order: str = "utf8"

    def __post_init__(self) -> None:
        require_int64(self.snapshot_time_ms)
        for name in (
            "graph_max_group_distance",
            "graph_window_ms",
            "graph_min_support",
            "graph_max_neighbors",
            "buffer_records",
            "merge_fan_in",
            "max_pairs",
            "max_group_size",
            "max_temp_bytes",
            "max_result_bytes",
            "max_memory_bytes",
        ):
            if require_int64(getattr(self, name), nonnegative=True) == 0:
                raise InputValidationError(f"{name}: требуется положительный предел")
        for name in (
            "graph_tau_ms",
            "graph_regularization",
            "popularity_smoothing",
            "vector_epsilon",
            "max_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise InputValidationError(f"{name}: требуется конечное положительное число")
        if (
            self.merge_fan_in < 2
            or self.schema_version != HISTORICAL_VERSION
            or self.id_order != "utf8"
        ):
            raise InputValidationError("Несовместимый формат исторического пакета")


@dataclass(frozen=True)
class HistoricalInput:
    iter_sessions: Callable[[], Iterator[Session]]
    session_count: int
    snapshot_time_ms: int
    source_namespace: str
    input_hashes: Mapping[str, str]
    scope: str


@dataclass(frozen=True)
class GraphRelation:
    relation: str
    indptr: IntArray
    indices: IntArray
    weights: FloatArray
    probabilities: FloatArray
    support: IntArray
    row_sums: FloatArray
    rho: FloatArray


@dataclass(frozen=True)
class GraphBuildResult:
    relations: tuple[GraphRelation, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class HistoricalStatistics:
    detail_counts: IntArray
    purchase_counts: IntArray
    popularity: FloatArray
    session_count: int


@dataclass(frozen=True)
class MetadataRecord:
    sku: str
    description: Any = None
    image: Any = None
    category: str | None = None
    price_bucket: int | None = None


@dataclass(frozen=True)
class MetadataSource:
    read: Callable[[Sequence[str]], Iterator[MetadataRecord]]
    description_dimension: int
    image_dimension: int
    input_hashes: Mapping[str, str]
    query_description_compatibility: str
    static_availability: str = "unknown"


@dataclass(frozen=True)
class Modality:
    vectors: FloatArray
    mask: NDArray[np.bool_]
    reasons: tuple[str | None, ...]


@dataclass(frozen=True)
class HistoricalContent:
    description: Modality
    image: Modality
    categories: tuple[str, ...]
    category_codes: IntArray
    price_bucket: IntArray
    price_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class HistoricalPackage:
    catalog: HistoricalCatalog
    graphs: GraphBuildResult
    statistics: HistoricalStatistics
    content: HistoricalContent
    snapshot_time_ms: int
    input_hashes: Mapping[str, str]
    metadata_contract: Mapping[str, Any]
    paths: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = HISTORICAL_VERSION
