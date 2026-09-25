"""Публикация и команда сборки исторического пакета."""

import io
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa

from coveo_recsys.artifacts import LocalArtifactStore, sha256_bytes, strict_json_loads
from coveo_recsys.contracts.historical import (
    HISTORICAL_VERSION,
    HistoricalConfig,
    HistoricalPackage,
    Session,
)
from coveo_recsys.historical.load import GRAPH_ARRAYS, PACKAGE_PATH, load_historical


def run_historical(
    project_root: Path,
    settings_path: Path,
    run_dir: Path,
    run_id: str,
    *,
    after_part: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from coveo_recsys.cli.preparation import _peak_memory
    from coveo_recsys.data.historical import open_historical_input
    from coveo_recsys.data.metadata import open_metadata_source
    from coveo_recsys.historical.build import build_historical
    from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

    assert_source_binding(project_root)
    settings_bytes = settings_path.read_bytes()
    settings = strict_json_loads(settings_bytes)
    if set(settings) != {"preparation", "metadata_path", "config"} or not run_id:
        raise ValueError("Неверный состав настроек или ID запуска")
    config = HistoricalConfig(**settings["config"])
    preparation = dict(settings["preparation"])
    source_root = (project_root / preparation.pop("path")).resolve()
    destination = run_dir.resolve()
    if (
        destination == source_root
        or destination in source_root.parents
        or source_root in destination.parents
    ):
        raise ValueError("Выход пересекается с исходной подготовкой")
    store = LocalArtifactStore(run_dir)
    code = source_sha256(project_root)
    settings_hash = sha256_bytes(settings_bytes)
    started = time.monotonic()
    audit: list[dict[str, Any]] = []
    measurements: dict[str, Any] = {"status": "failed", "reused_package": False}
    common: dict[str, Any] = dict(
        code_sha256=code, run_id=run_id, scope="unverified", config_id="historical"
    )
    temp_peak = 0
    last_budget_check = float("-inf")

    def disk(directory: Path) -> tuple[int, int]:
        sizes = [p.stat().st_size for p in directory.rglob("*") if p.is_file()]
        return sum(sizes), len(sizes)

    try:
        admitted = open_historical_input(source_root, audit=audit.append, **preparation)
        common["scope"] = admitted.scope
        input_store = LocalArtifactStore(source_root)
        profile_bytes = input_store.read_bytes("inputs/profile.json")
        if sha256_bytes(profile_bytes) != admitted.input_hashes["profile"]:
            raise ValueError("Профиль поставки изменился")
        metadata = open_metadata_source(
            project_root / settings["metadata_path"],
            strict_json_loads(profile_bytes),
            expected_sha256=admitted.input_hashes["products"],
        )
        admitted = replace(
            admitted,
            input_hashes=MappingProxyType(
                {**admitted.input_hashes, "historical_settings": settings_hash}
            ),
        )
        inputs = {**admitted.input_hashes, **metadata.input_hashes}
        common["inputs"] = inputs
        common["configuration"] = asdict(config)
        store.publish_bytes(
            "historical/settings.json", settings_bytes, kind="json", reuse=True, **common
        )
        with tempfile.TemporaryDirectory(prefix=".historical-", dir=store.root) as temporary:
            work_dir = Path(temporary)

            def check_budget(force: bool = False) -> None:
                nonlocal temp_peak, last_budget_check
                now = time.monotonic()
                if not force and now - last_budget_check < 0.25:
                    return
                last_budget_check = now
                memory = _peak_memory()["peak_process_memory_bytes"]
                temporary_size, _ = disk(work_dir)
                temp_peak = max(temp_peak, temporary_size)
                result_size, _ = disk(store.root)
                if (
                    time.monotonic() - started > config.max_seconds
                    or memory is None
                    or memory > config.max_memory_bytes
                    or temporary_size > config.max_temp_bytes
                    # Резерв для последней публикации, отчёта и служебных копий.
                    or result_size - temporary_size + 1_048_576 > config.max_result_bytes
                ):
                    raise ValueError(
                        "Превышен предел времени, памяти или диска исторической сборки"
                    )

            original_iterator = admitted.iter_sessions

            def sessions() -> Iterator[Session]:
                for index, session in enumerate(original_iterator()):
                    if index % 100 == 0:
                        check_budget()
                    yield session

            admitted = replace(admitted, iter_sessions=sessions)
            check_budget()

            def before_commit() -> None:
                if (
                    source_sha256(project_root) != code
                    or settings_path.read_bytes() != settings_bytes
                ):
                    raise ValueError("Код или настройки изменились во время сборки")
                check_budget(force=True)

            if (store.root / PACKAGE_PATH).exists():
                store.verify_compatible(
                    PACKAGE_PATH,
                    expected_inputs=inputs,
                    expected_code_sha256=code,
                    expected_schema_version=HISTORICAL_VERSION,
                    expected_configuration=asdict(config),
                )
                # Повтор подтверждает вход, а не только сохранённые хеши пакета.
                for _ in admitted.iter_sessions():
                    pass
                package = load_historical(store)
                measurements["reused_package"] = True
                before_commit()
            else:
                package = build_historical(
                    admitted, metadata, config, work_dir=work_dir, check_budget=check_budget
                )
                publish_package(
                    store,
                    package,
                    config,
                    code_sha256=code,
                    run_id=run_id,
                    scope=admitted.scope,
                    after_part=after_part,
                    check_budget=check_budget,
                    before_commit=before_commit,
                )
                package = load_historical(store)
        size, count = disk(store.root)
        coverage = {
            graph.relation: {
                "edges_after_support": package.graphs.metrics["relations"][graph.relation][
                    "edges_after_support"
                ],
                "edges_after_pruning": len(graph.indices),
                "skus_with_outgoing": int(np.count_nonzero(graph.indptr[1:] > graph.indptr[:-1])),
                "fraction_with_outgoing": float(
                    np.count_nonzero(graph.indptr[1:] > graph.indptr[:-1])
                    / len(package.catalog.skus)
                )
                if package.catalog.skus
                else 0.0,
            }
            for graph in package.graphs.relations
        }
        measurements.update(
            status="completed",
            catalog_size=len(package.catalog.skus),
            session_count=package.statistics.session_count,
            graph=package.graphs.metrics,
            coverage=coverage,
            metadata={
                name: {
                    "usable": int(getattr(package.content, name).mask.sum()),
                    "reasons": {
                        reason: getattr(package.content, name).reasons.count(reason)
                        for reason in ("vector_absent", "vector_invalid", "vector_zero")
                    },
                }
                for name in ("description", "image")
            },
            result_bytes_before_measurement=size,
            result_files_before_measurement=count,
            telemetry_complete=None,
            query_description_compatibility=metadata.query_description_compatibility,
            static_metadata_availability=metadata.static_availability,
        )
        return measurements
    except KeyboardInterrupt:
        measurements["status"] = "interrupted"
        raise
    finally:
        measurements.update(
            elapsed_seconds=time.monotonic() - started,
            temporary_peak_bytes=max(
                temp_peak, measurements.get("graph", {}).get("temp_peak_bytes", 0)
            ),
            **_peak_memory(),
            reads=audit,
            recovery="Готовый пакет загружается; при неполной публикации вычисления повторяются, проверенные части сохраняются.",
        )
        dependencies = (
            {PACKAGE_PATH: store.describe(PACKAGE_PATH).descriptor_sha256}
            if measurements["status"] == "completed"
            else {}
        )
        path = f"historical/executions/{uuid4().hex}.json"
        store.publish_json(path, measurements, artifact_dependencies=dependencies, **common)
        if (
            measurements["status"] == "completed"
            and not (store.root / "historical/build_report.json").exists()
        ):
            store.publish_json(
                "historical/build_report.json",
                measurements,
                artifact_dependencies=dependencies,
                **common,
            )


def publish_package(
    store: LocalArtifactStore,
    package: HistoricalPackage,
    config: HistoricalConfig,
    *,
    code_sha256: str,
    run_id: str,
    scope: str,
    after_part: Callable[[str], None] | None = None,
    check_budget: Callable[[], None] | None = None,
    before_commit: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Части неизменяемы; незавершённый состав не является готовым пакетом."""
    common: dict[str, Any] = dict(
        code_sha256=code_sha256,
        run_id=run_id,
        scope=scope,
        config_id="historical",
        schema_version=HISTORICAL_VERSION,
        inputs=dict(package.input_hashes),
        configuration=asdict(config),
        reuse=True,
    )
    parts: dict[str, str] = {}
    dependencies: dict[str, str] = {}

    def remember(descriptor: Any) -> None:
        parts[descriptor.path] = descriptor.descriptor_sha256
        if check_budget:
            check_budget()

    def json_part(path: str, value: Any) -> None:
        remember(store.publish_json(path, value, artifact_dependencies=dependencies, **common))

    def array_part(path: str, value: Any) -> None:
        values = np.asarray(value)
        if values.dtype.hasobject or not np.isfinite(values).all():
            raise ValueError("Неконечный массив или объектный тип")
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        remember(
            store.publish_bytes(
                path,
                stream.getvalue(),
                kind="npy",
                metadata={"shape": list(values.shape), "dtype": values.dtype.str},
                artifact_dependencies=dependencies,
                **common,
            )
        )

    def boundary(name: str) -> None:
        if after_part:
            after_part(name)

    catalog = package.catalog.skus
    path = "historical/catalog.parquet"
    remember(
        store.publish_table(
            path,
            pa.table(
                {
                    "sku": pa.array(catalog, type=pa.string()),
                    "row_index": pa.array(range(len(catalog)), type=pa.int64()),
                }
            ),
            data_role="auxiliary",
            **common,
        )
    )
    dependencies = {path: parts[path]}
    boundary("catalog")
    for graph in package.graphs.relations:
        for name in GRAPH_ARRAYS:
            array_part(f"historical/graph/{graph.relation}/{name}.npy", getattr(graph, name))
    json_part("historical/graph/summary.json", package.graphs.metrics)
    boundary("graph")
    stats = package.statistics
    remember(
        store.publish_table(
            "historical/statistics.parquet",
            pa.table(
                {
                    "sku": pa.array(catalog, type=pa.string()),
                    "detail_count": pa.array(stats.detail_counts, type=pa.int64()),
                    "purchase_count": pa.array(stats.purchase_counts, type=pa.int64()),
                    "popularity": pa.array(stats.popularity, type=pa.float64()),
                }
            ),
            data_role="auxiliary",
            artifact_dependencies=dependencies,
            **common,
        )
    )
    json_part("historical/statistics_summary.json", {"session_count": stats.session_count})
    boundary("statistics")
    for name in ("description", "image"):
        modality = getattr(package.content, name)
        prefix = f"historical/content/{name}/"
        array_part(prefix + "vectors.npy", modality.vectors)
        array_part(prefix + "mask.npy", modality.mask)
        json_part(prefix + "reasons.json", modality.reasons)
    content = package.content
    json_part(
        "historical/dictionaries.json",
        {
            "categories": content.categories,
            "category_codes": content.category_codes.tolist(),
            "price_bucket": content.price_bucket.tolist(),
            "price_mask": content.price_mask.tolist(),
        },
    )
    boundary("content")
    body = {
        "schema_version": HISTORICAL_VERSION,
        "config": asdict(config),
        "input_hashes": dict(package.input_hashes),
        "artifacts": parts,
        "catalog_size": len(catalog),
        "metadata_contract": dict(package.metadata_contract),
    }
    candidate = "historical/package.candidate.json"
    store.publish_json(candidate, body, artifact_dependencies=parts, **common)
    load_historical(store, package_path=candidate)
    if check_budget:
        check_budget()
    if before_commit:
        before_commit()
    store.publish_json(PACKAGE_PATH, body, artifact_dependencies=parts, **common)
    return parts
