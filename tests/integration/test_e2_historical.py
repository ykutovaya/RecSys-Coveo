from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.artifacts import ArtifactError, LocalArtifactStore, sha256_bytes
from coveo_recsys.cli.historical import publish_package
from coveo_recsys.contracts.historical import HistoricalConfig
from coveo_recsys.historical.build import build_historical
from coveo_recsys.historical.load import load_historical
from tests.reference.e2_cases import history, metadata
from tests.unit.test_e2_input import prepared_source  # noqa: F401 — общий малый вход Э1


def sample():
    return history(
        [
            (sid, t, action, sku)
            for sid in ("1", "2", "3")
            for t, action, sku in [(1, "detail", "a"), (2, "detail", "b"), (3, "purchase", "b")]
        ]
    )


def test_build_publish_load_preserves_catalog_and_sparse_arrays(tmp_path):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    store = LocalArtifactStore(tmp_path / "result")
    publish_package(store, package, config, code_sha256="c" * 64, run_id="test", scope="synthetic")
    loaded = load_historical(store)
    assert loaded.catalog.skus == ("a", "b")
    for original, restored in zip(package.graphs.relations, loaded.graphs.relations, strict=True):
        np.testing.assert_array_equal(original.support, restored.support)
        np.testing.assert_allclose(
            original.probabilities, restored.probabilities, atol=1e-10, rtol=1e-8
        )
    assert loaded.statistics.session_count == 3
    assert not loaded.content.description.vectors.flags.writeable


def test_package_does_not_exist_before_all_parts_are_published(tmp_path):
    store = LocalArtifactStore(tmp_path)
    with pytest.raises((ArtifactError, FileNotFoundError)):
        load_historical(store)


def test_snapshot_mismatch_refuses_build(tmp_path):
    with pytest.raises(ValueError, match="сним"):
        build_historical(
            sample(), metadata(), HistoricalConfig(snapshot_time_ms=1), work_dir=tmp_path
        )


def test_publication_rejects_nonintegral_counters(tmp_path):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    package = replace(
        package, statistics=replace(package.statistics, detail_counts=np.array([3.5, 3.0]))
    )
    store = LocalArtifactStore(tmp_path / "result")
    with pytest.raises((ArtifactError, ValueError)):
        publish_package(
            store, package, config, code_sha256="c" * 64, run_id="test", scope="synthetic"
        )
    assert not (store.root / "historical/package.json").exists()


def test_final_guard_prevents_ready_marker(tmp_path):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    store = LocalArtifactStore(tmp_path / "result")

    def reject():
        raise ValueError("changed inputs")

    with pytest.raises(ValueError, match="changed inputs"):
        publish_package(
            store,
            package,
            config,
            code_sha256="c" * 64,
            run_id="test",
            scope="synthetic",
            before_commit=reject,
        )
    assert not (store.root / "historical/package.json").exists()


@pytest.mark.parametrize("boundary", ["catalog", "graph", "statistics", "content"])
def test_interruption_reuses_only_verified_parts(tmp_path, boundary):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    store = LocalArtifactStore(tmp_path / "result")

    def interrupt(name):
        if name == boundary:
            raise KeyboardInterrupt

    kwargs = dict(code_sha256="c" * 64, run_id="test", scope="synthetic")
    with pytest.raises(KeyboardInterrupt):
        publish_package(store, package, config, after_part=interrupt, **kwargs)
    assert not (store.root / "historical/package.json").exists()
    original = store.describe("historical/catalog.parquet").descriptor_sha256
    publish_package(store, package, config, **kwargs)
    assert store.describe("historical/catalog.parquet").descriptor_sha256 == original
    assert load_historical(store).catalog.skus == ("a", "b")


@pytest.mark.parametrize("change", ["code", "config", "payload"])
def test_incompatible_or_corrupt_parts_are_not_reused(tmp_path, change):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    store = LocalArtifactStore(tmp_path / "result")
    kwargs = dict(code_sha256="c" * 64, run_id="test", scope="synthetic")
    publish_package(store, package, config, **kwargs)
    if change == "payload":
        (store.root / "historical/statistics.parquet").write_bytes(b"corrupt")
        with pytest.raises(ArtifactError):
            load_historical(store)
    else:
        if change == "code":
            kwargs["code_sha256"] = "d" * 64
        else:
            config = replace(config, graph_min_support=4)
        with pytest.raises(ArtifactError):
            publish_package(store, package, config, **kwargs)


@pytest.fixture
def historical_cli_args(tmp_path, request):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = request.getfixturevalue("prepared_source")
    settings = {
        "preparation": {
            "path": str(source),
            "report_sha256": sha256_bytes((source / "reports/preparation.json").read_bytes()),
        },
        "metadata_path": str(source.parent / "input/sku_to_content.csv"),
        "config": {"snapshot_time_ms": 200_000_000},
    }
    settings_path = tmp_path / "historical.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    destination = tmp_path / "historical"
    return [
        "--project-root",
        str(root),
        "historical",
        "build",
        "--config",
        str(settings_path),
        "--run-dir",
        str(destination),
        "--run-id",
        "small-e2",
    ]


def test_cli_builds_and_reuses_completed_preparation(tmp_path, capsys, historical_cli_args):
    import json

    from coveo_recsys.cli import main

    for reused in (False, True):
        assert main(historical_cli_args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "completed" and result["session_count"] == 2
        assert result["reused_package"] is reused
    assert load_historical(LocalArtifactStore(tmp_path / "historical")).catalog.skus == (
        "A",
        "B",
        "C",
    )


def test_cli_rejects_interrupted_artifact_and_preserves_its_files(
    tmp_path, capsys, monkeypatch, historical_cli_args
):
    from coveo_recsys.cli import main

    destination = tmp_path / "historical"
    original = LocalArtifactStore._atomic_write

    def interrupt(store, path, data):
        if path == "historical/catalog.parquet.artifact.json":
            raise OSError("simulated interrupted publication")
        original(store, path, data)

    monkeypatch.setattr(LocalArtifactStore, "_atomic_write", interrupt)
    assert main(historical_cli_args) != 0
    assert "simulated interrupted publication" in capsys.readouterr().out
    monkeypatch.setattr(LocalArtifactStore, "_atomic_write", original)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert main(historical_cli_args) != 0
    assert "новый каталог запуска" in capsys.readouterr().out
    # Повтор может добавить запись о неудачной команде, но не менять её остатки.
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items()
    )
    assert not (destination / "historical/catalog.parquet.artifact.json").exists()
    assert not (destination / "historical/package.json").exists()


@pytest.mark.parametrize("defect", ["dimension", "price"])
def test_published_content_must_match_metadata_contract(tmp_path, defect):
    config = HistoricalConfig(snapshot_time_ms=10_000_000)
    package = build_historical(sample(), metadata(), config, work_dir=tmp_path / "work")
    content = package.content
    if defect == "dimension":
        content = replace(
            content,
            description=replace(
                content.description, vectors=np.pad(content.description.vectors, ((0, 0), (0, 1)))
            ),
        )
    else:
        content = replace(content, price_bucket=np.array([-1, 0], dtype=np.int64))
    store = LocalArtifactStore(tmp_path / "result")
    with pytest.raises((ValueError, ArtifactError)):
        publish_package(
            store,
            replace(package, content=content),
            config,
            code_sha256="c" * 64,
            run_id="test",
            scope="synthetic",
        )
    assert not (store.root / "historical/package.json").exists()


def test_csr_offsets_cannot_wrap_int64(tmp_path):
    from coveo_recsys.contracts.historical import GraphBuildResult

    source = history(
        [(str(s), t, "detail", sku) for s in range(3) for t, sku in [(1, "a"), (2, "b"), (3, "c")]]
    )
    config = HistoricalConfig(snapshot_time_ms=source.snapshot_time_ms)
    package = build_historical(source, metadata(), config, work_dir=tmp_path / "work")
    original = package.graphs.relations[0]
    graph = replace(
        original,
        indptr=np.array([0, np.iinfo(np.int64).max, -2, 1], dtype=np.int64),
        indices=np.array([1], dtype=np.int64),
        weights=np.array([1.0]),
        probabilities=np.array([1.0]),
        support=np.array([3], dtype=np.int64),
        row_sums=np.array([1.0, 0.0, 1.0]),
        rho=np.array([1 / 11, 0.0, 1 / 11]),
    )
    package = replace(
        package,
        graphs=GraphBuildResult((graph, *package.graphs.relations[1:]), package.graphs.metrics),
    )
    store = LocalArtifactStore(tmp_path / "result")
    with pytest.raises(ArtifactError):
        publish_package(
            store, package, config, code_sha256="c" * 64, run_id="test", scope="synthetic"
        )
    assert not (store.root / "historical/package.json").exists()
