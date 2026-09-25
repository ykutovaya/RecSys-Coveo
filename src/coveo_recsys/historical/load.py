"""Проверка готового пакета без импорта вычислительного контура."""

import io
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa

from coveo_recsys.artifacts import ArtifactError, LocalArtifactStore, strict_json_loads
from coveo_recsys.contracts.historical import (
    HISTORICAL_VERSION,
    RELATIONS,
    GraphBuildResult,
    GraphRelation,
    HistoricalCatalog,
    HistoricalConfig,
    HistoricalContent,
    HistoricalPackage,
    HistoricalStatistics,
    Modality,
)

PACKAGE_PATH = "historical/package.json"
GRAPH_ARRAYS = ("indptr", "indices", "weights", "probabilities", "support", "row_sums", "rho")


def load_historical(
    store: LocalArtifactStore, *, package_path: str = PACKAGE_PATH
) -> HistoricalPackage:
    if package_path not in {PACKAGE_PATH, "historical/package.candidate.json"}:
        raise ArtifactError("Неизвестный путь состава пакета")
    with store.verification_scope():
        head = store.describe(package_path)
        if (
            head.kind != "json"
            or head.config_id != "historical"
            or head.artifact_version != "1"
            or head.scope not in {"actual", "synthetic"}
        ):
            raise ArtifactError("Неверный тип или область пакета")
        body = strict_json_loads(store.read_bytes(package_path))
        if set(body) != {
            "schema_version",
            "config",
            "input_hashes",
            "artifacts",
            "catalog_size",
            "metadata_contract",
        }:
            raise ArtifactError("Неверный состав пакета")
        if (
            body["schema_version"] != HISTORICAL_VERSION
            or head.schema_version != HISTORICAL_VERSION
        ):
            raise ArtifactError("Неизвестная версия исторического пакета")
        config = HistoricalConfig(**body["config"])
        spaces = body["metadata_contract"]
        if (
            set(spaces)
            != {
                "description_dimension",
                "image_dimension",
                "query_description_compatibility",
                "static_availability",
            }
            or any(
                type(spaces[name]) is not int or spaces[name] <= 0
                for name in ("description_dimension", "image_dimension")
            )
            or spaces["query_description_compatibility"] not in {"publisher_statement", "unknown"}
            or spaces["static_availability"] != "unknown"
        ):
            raise ArtifactError("Неверный контракт пространств метаданных")
        parts = body["artifacts"]
        if (
            head.artifact_dependencies != parts
            or head.inputs != body["input_hashes"]
            or head.configuration != body["config"]
        ):
            raise ArtifactError("Пакет не соответствует закреплённым зависимостям")
        expected = {
            "historical/catalog.parquet",
            "historical/statistics.parquet",
            "historical/statistics_summary.json",
            "historical/graph/summary.json",
            "historical/dictionaries.json",
            *(f"historical/graph/{rel}/{name}.npy" for rel in RELATIONS for name in GRAPH_ARRAYS),
            *(
                f"historical/content/{mod}/{name}"
                for mod in ("description", "image")
                for name in ("vectors.npy", "mask.npy", "reasons.json")
            ),
        }
        if set(parts) != expected:
            raise ArtifactError("Пакет содержит лишние или отсутствующие части")
        for path, digest in parts.items():
            descriptor = store.verify_compatible(
                path,
                expected_inputs=body["input_hashes"],
                expected_code_sha256=head.code_sha256,
                expected_schema_version=HISTORICAL_VERSION,
                expected_configuration=body["config"],
                expected_descriptor_sha256=digest,
            )
            if (
                descriptor.run_id != head.run_id
                or descriptor.scope != head.scope
                or descriptor.config_id != "historical"
            ):
                raise ArtifactError("Часть из чужого запуска")
            if path != "historical/catalog.parquet" and descriptor.artifact_dependencies != {
                "historical/catalog.parquet": parts["historical/catalog.parquet"]
            }:
                raise ArtifactError("Часть не связана с каталогом")
        table = store.read_table("historical/catalog.parquet")
        if not table.schema.equals(pa.schema([("sku", pa.string()), ("row_index", pa.int64())])):
            raise ArtifactError("Неверная схема каталога")
        catalog = HistoricalCatalog(tuple(table["sku"].to_pylist()))
        n = len(catalog.skus)
        if (
            table["row_index"].to_pylist() != list(range(n))
            or type(body["catalog_size"]) is not int
            or body["catalog_size"] != n
        ):
            raise ArtifactError("Неверный порядок строк каталога")

        def array(path: str, dtype: str, shape: tuple[int, ...] | None = None) -> Any:
            descriptor = store.describe(path)
            value = np.load(io.BytesIO(store.read_bytes(path)), allow_pickle=False)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(dtype)
                or not np.isfinite(value).all()
                or (shape is not None and value.shape != shape)
                or descriptor.kind != "npy"
                or descriptor.metadata.get("shape") != list(value.shape)
                or descriptor.metadata.get("dtype") != value.dtype.str
            ):
                raise ArtifactError("Неверная форма, тип или численное содержимое массива")
            value.flags.writeable = False
            return value

        graphs = []
        for relation in RELATIONS:
            prefix = f"historical/graph/{relation}/"
            indptr = array(prefix + "indptr.npy", "int64", (n + 1,))
            if indptr[0] != 0 or np.any(indptr < 0) or np.any(indptr[1:] < indptr[:-1]):
                raise ArtifactError("Неверные смещения графа")
            edges = int(indptr[-1])
            if np.any(indptr > edges):
                raise ArtifactError("Смещения выходят за число рёбер")
            if edges < 0 or edges > n * config.graph_max_neighbors:
                raise ArtifactError("Размер графа превышает предел")
            arrays = {
                name: array(
                    prefix + name + ".npy",
                    "int64" if name in {"indices", "support"} else "float64",
                    (n,) if name in {"row_sums", "rho"} else (edges,),
                )
                for name in GRAPH_ARRAYS
                if name != "indptr"
            }
            graph = GraphRelation(relation, indptr=indptr, **arrays)
            if (
                np.any(graph.indices < 0)
                or np.any(graph.indices >= n)
                or np.any(graph.support < config.graph_min_support)
                or np.any(graph.weights <= 0)
                or np.any(graph.probabilities <= 0)
            ):
                raise ArtifactError("Недопустимое ребро графа")
            for row in range(n):
                start, end = int(indptr[row]), int(indptr[row + 1])
                ids, weights = graph.indices[start:end], graph.weights[start:end]
                if (
                    len(ids) > config.graph_max_neighbors
                    or len(set(ids)) != len(ids)
                    or row in ids
                    or list(range(len(ids)))
                    != sorted(range(len(ids)), key=lambda k: (-weights[k], int(ids[k])))
                ):
                    raise ArtifactError("Неверный порядок или состав соседей")
                total = float(np.sum(weights))
                if not np.isclose(
                    graph.row_sums[row], total, atol=1e-10, rtol=1e-8
                ) or not np.isclose(
                    graph.rho[row],
                    total / (total + config.graph_regularization),
                    atol=1e-10,
                    rtol=1e-8,
                ):
                    raise ArtifactError("Неверная нормировка строки графа")
                if len(ids) and not np.allclose(
                    graph.probabilities[start:end], weights / total, atol=1e-10, rtol=1e-8
                ):
                    raise ArtifactError("Неверные вероятности перехода")
            graphs.append(graph)
        stats = store.read_table("historical/statistics.parquet")
        if (
            not stats.schema.equals(
                pa.schema(
                    [
                        ("sku", pa.string()),
                        ("detail_count", pa.int64()),
                        ("purchase_count", pa.int64()),
                        ("popularity", pa.float64()),
                    ]
                )
            )
            or tuple(stats["sku"].to_pylist()) != catalog.skus
        ):
            raise ArtifactError("Статистики не соответствуют порядку каталога")
        summary = strict_json_loads(store.read_bytes("historical/statistics_summary.json"))
        session_count = summary["session_count"]
        if type(session_count) is not int or session_count < 0:
            raise ArtifactError("Неверное число исторических сессий")
        counts = [
            np.asarray(stats[name].to_numpy(), dtype=np.int64)
            for name in ("detail_count", "purchase_count")
        ]
        popularity = np.asarray(stats["popularity"].to_numpy(), dtype=np.float64)
        if any(np.any(a < 0) or np.any(a > session_count) for a in counts):
            raise ArtifactError("Неверные сессионные счётчики")
        expected_pop = (
            (counts[0] + config.popularity_smoothing)
            / (float(np.sum(counts[0])) + config.popularity_smoothing * n)
            if n
            else np.empty(0)
        )
        if not np.allclose(popularity, expected_pop, atol=1e-10, rtol=1e-8):
            raise ArtifactError("Неверная популярность")
        for values in (*counts, popularity):
            values.flags.writeable = False
        if any(np.any(g.support > session_count) or np.any(g.weights > g.support) for g in graphs):
            raise ArtifactError("Поддержка графа превышает число сессий")
        statistics = HistoricalStatistics(counts[0], counts[1], popularity, session_count)
        modalities = []
        for name in ("description", "image"):
            prefix = f"historical/content/{name}/"
            vectors = array(prefix + "vectors.npy", "float64")
            mask = array(prefix + "mask.npy", "bool", (n,))
            reasons = tuple(strict_json_loads(store.read_bytes(prefix + "reasons.json")))
            if (
                vectors.ndim != 2
                or vectors.shape[0] != n
                or vectors.shape[1] != spaces[name + "_dimension"]
                or len(reasons) != n
            ):
                raise ArtifactError("Неверная форма модальности")
            for row in range(n):
                if mask[row]:
                    if reasons[row] is not None or not np.isclose(
                        np.linalg.norm(vectors[row]), 1, atol=1e-10, rtol=1e-8
                    ):
                        raise ArtifactError("Пригодная модальность не нормирована")
                elif reasons[row] not in {
                    "vector_absent",
                    "vector_invalid",
                    "vector_zero",
                } or np.any(vectors[row] != 0):
                    raise ArtifactError("Неверная маска модальности")
            modalities.append(Modality(vectors, mask, reasons))
        dictionary = strict_json_loads(store.read_bytes("historical/dictionaries.json"))
        categories = tuple(dictionary["categories"])
        if categories != tuple(sorted(set(categories), key=lambda v: v.encode("utf-8"))) or any(
            not v for v in categories
        ):
            raise ArtifactError("Неверный словарь категорий")
        for name in ("category_codes", "price_bucket", "price_mask"):
            values = dictionary[name]
            expected_type = bool if name == "price_mask" else int
            if len(values) != n or any(type(v) is not expected_type for v in values):
                raise ArtifactError("Неверные типы категорий или цены")
        codes = np.asarray(dictionary["category_codes"], dtype=np.int64)
        prices = np.asarray(dictionary["price_bucket"], dtype=np.int64)
        price_mask = np.asarray(dictionary["price_mask"], dtype=np.bool_)
        if (
            np.any(codes < 0)
            or np.any(codes > len(categories))
            or np.any(prices[~price_mask] != 0)
            or np.any(prices < 0)
        ):
            raise ArtifactError("Неверные коды категорий или маски цены")
        for attribute in (codes, prices, price_mask):
            attribute.flags.writeable = False
        content = HistoricalContent(
            modalities[0], modalities[1], categories, codes, prices, price_mask
        )
        metrics = strict_json_loads(store.read_bytes("historical/graph/summary.json"))
        return HistoricalPackage(
            catalog,
            GraphBuildResult(tuple(graphs), metrics),
            statistics,
            content,
            config.snapshot_time_ms,
            MappingProxyType(body["input_hashes"]),
            MappingProxyType(spaces),
            MappingProxyType(parts),
        )
