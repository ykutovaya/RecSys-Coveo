"""Сервис загружает готовые зависимости один раз, не расширяя каталог."""

import csv
import io
import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.features import FeatureEngine
from coveo_recsys.ranking.load import RankerModel
from coveo_recsys.ranking.predict import predict
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import package, request


class Booster:
    def predict(self, matrix, **kwargs):
        return np.sum(matrix, axis=1)


@pytest.fixture
def ready_inputs(tmp_path, monkeypatch):
    from coveo_recsys.explanations.templates import TEMPLATE_SHA256, TEMPLATE_VERSION
    from coveo_recsys.serving import package as module

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
        return sha256_file(path)

    profile = json.loads(Path("configs/synthetic_csv.json").read_bytes())
    profile["dimensions"]["image"] = 2
    profile_hash = write("metadata/profile.json", profile)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["sku", "description_vector", "image_vector", "category", "price_bucket"])
    writer.writerows(
        [
            ["a", "[99,0]", "[0,99]", "foreign", "99"],
            ["outside", "[0,3]", "[4,0]", "root/b", "2"],
            ["bad-vector", "broken", "", "", ""],
        ]
    )
    metadata_path = tmp_path / "metadata/products.csv"
    metadata_path.write_text(stream.getvalue(), encoding="utf-8", newline="")
    metadata_hash = sha256_file(metadata_path)
    history = replace(
        package(),
        input_hashes={
            "metadata_sha256": metadata_hash,
            "metadata_profile_sha256": profile_hash,
        },
    )
    store = LocalArtifactStore(tmp_path / "history")
    common = dict(
        code_sha256="e" * 64,
        run_id="historical-producer",
        scope="synthetic",
        config_id="historical",
    )
    catalog = store.publish_bytes("historical/catalog.parquet", b"unit-catalog", **common)
    head = store.publish_json(
        "historical/package.json",
        {"artifacts": {"historical/catalog.parquet": catalog.descriptor_sha256}},
        **common,
    )
    schema = json.loads(Path("configs/feature_schema.json").read_bytes())
    schema_hash = write("schema.json", schema)
    engine = FeatureEngine(RetrievalIndex(history, catalog.sha256), schema)
    features = dict(
        status="completed",
        feature_schema=schema,
        identity=dict(
            format_version="features_v1",
            period="B",
            scope="synthetic",
            source_sha256="f" * 64,
            preparation_sha256="b" * 64,
            historical_sha256=head.sha256,
            historical_descriptor_sha256=head.descriptor_sha256,
            catalog_sha256=catalog.sha256,
            feature_schema_file_sha256=schema_hash,
            configuration=asdict(RetrievalConfig()),
        ),
    )
    feature_hash = write("features/reports/features.json", features)
    identity = dict(
        source_sha256="c" * 64,
        run_id="ranker-producer",
        period="B",
        scope="synthetic",
        evaluation_protocol="observed_purchases_v1",
        preparation_sha256="b" * 64,
        features_sha256=feature_hash,
        historical_sha256=head.sha256,
        catalog_sha256=catalog.sha256,
        feature_schema_file_sha256=schema_hash,
        feature_schema_sha256=engine.schema_sha256,
    )
    model = RankerModel(
        Booster(),
        dict(
            status="completed",
            format_version="ranker_v1",
            identity=identity,
            model_sha256=sha256_bytes(b"unit-model"),
            feature_schema_sha256=engine.schema_sha256,
            numeric_columns=list(engine.numeric_columns),
            categorical_columns=list(engine.categorical_columns),
            actual_tree_count=1,
        ),
    )
    model_hash = write("model/reports/ranker.json", model.summary)
    (tmp_path / "model/models").mkdir()
    (tmp_path / "model/models/ranker.txt").write_bytes(b"unit-model")
    calls = []

    def load_history(actual):
        calls.append("historical")
        assert actual.root == store.root
        return history

    def load_model(actual, *, expected_identity):
        calls.append("model")
        assert actual == tmp_path / "model"
        assert expected_identity == identity
        return model

    monkeypatch.setattr(module, "load_historical", load_history)
    monkeypatch.setattr(module, "load_ranker", load_model)
    monkeypatch.setattr(module, "peak_memory", lambda: {"peak_process_memory_bytes": 1000})
    manifest = dict(
        format_version="serving_package_v1",
        profile="local_checksum_only",
        published_at_ms=1500,
        historical=dict(path="history", package_sha256=head.sha256),
        model=dict(path="model", report_sha256=model_hash),
        training_features=dict(path="features", report_sha256=feature_hash),
        feature_schema=dict(path="schema.json", sha256=schema_hash),
        retrieval={},
        metadata=dict(
            path="metadata/products.csv",
            profile_path="metadata/profile.json",
            sha256=metadata_hash,
            profile_sha256=profile_hash,
        ),
        templates=dict(version=TEMPLATE_VERSION, sha256=TEMPLATE_SHA256),
        limits=dict(
            max_metadata_rows=10,
            max_metadata_bytes=100000,
            max_memory_bytes=1000000,
            max_seconds=30,
        ),
    )
    return module, tmp_path, manifest, history, identity, calls


def test_startup_preserves_producer_and_keeps_complete_metadata_outside_catalog(
    ready_inputs, monkeypatch
):
    module, root, manifest, original, identity, calls = ready_inputs
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    loaded = module.load_serving_package(root, manifest)
    assert loaded.package_id == sha256_bytes(canonical_json_bytes(manifest))
    assert loaded.producer_identity == identity
    assert loaded.engine.index.package is original
    assert loaded.engine.index.package.catalog.skus == ("a", "b", "c", "я")
    assert loaded.metadata_available is True and loaded.metadata_rows == 3
    assert loaded.engine.index.row("outside").description.tolist() == [0, 1]
    assert loaded.engine.index.row("bad-vector").description_mask is False
    assert loaded.engine.index.row("not-in-supply") is None
    assert loaded.engine.index.row("a").category == "root/a"
    assert "outside" not in loaded.engine.index.positions
    with pytest.raises(FrozenInstanceError):
        loaded.published_at_ms = 0
    monkeypatch.setattr(
        module, "row_batches", lambda *_args, **_kwargs: pytest.fail("CSV per request")
    )
    context = bind_context(request(((2000, "detail", "outside"),)), loaded.engine.index)
    for _ in range(2):
        scores = score_request(context, loaded.engine.index, loaded.retrieval)
        candidates = retrieve(context, loaded.engine.index, loaded.retrieval, scores=scores)
        batch = loaded.engine.build(context, candidates, scores=scores)
        assert predict(loaded.model, batch, catalog_sha256=loaded.catalog_sha256)
    assert calls == ["historical", "model"]
    assert before == {path: (sha256_file(path), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize("time, expected", [(1500, None), (1501, 1500), (2000, 1500), (2001, 2000)])
def test_publication_is_strictly_before_event_time(ready_inputs, time, expected):
    module, root, manifest, *_ = ready_inputs
    first = module.load_serving_package(root, manifest)
    second = replace(first, package_id="2" * 64, published_at_ms=2000)
    if expected is None:
        with pytest.raises(ValueError, match="version_unavailable"):
            module.select_package((second, first), t_q=time)
    else:
        assert module.select_package((second, first), t_q=time).published_at_ms == expected


@pytest.mark.parametrize(
    "section,key",
    [
        ("historical", "package_sha256"),
        ("model", "report_sha256"),
        ("training_features", "report_sha256"),
        ("feature_schema", "sha256"),
        ("metadata", "sha256"),
        ("metadata", "profile_sha256"),
        ("templates", "sha256"),
    ],
)
def test_changed_pin_is_rejected(ready_inputs, section, key):
    module, root, manifest, *_ = ready_inputs
    manifest[section][key] = "0" * 64
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


@pytest.mark.parametrize("value", [0, -1, True])
def test_nonpositive_or_boolean_budget_rejected_before_reading(ready_inputs, monkeypatch, value):
    module, root, manifest, *_ = ready_inputs
    manifest["limits"]["max_metadata_rows"] = value
    monkeypatch.setattr(
        module, "row_batches", lambda *_args, **_kwargs: pytest.fail("read before limits")
    )
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


@pytest.mark.parametrize("limit", ["max_metadata_rows", "max_metadata_bytes", "max_memory_bytes"])
def test_exceeded_budget_never_publishes_partial_lookup(ready_inputs, limit):
    module, root, manifest, *_ = ready_inputs
    manifest["limits"][limit] = 1
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


@pytest.mark.parametrize(
    "relative",
    [
        "schema.json",
        "features/reports/features.json",
        "model/reports/ranker.json",
        "metadata/profile.json",
        "model/models/ranker.txt",
        "history/historical/package.json",
        "history/historical/catalog.parquet",
        "history/historical/package.json.artifact.json",
        "catalog-object",
    ],
)
def test_large_input_is_rejected_before_open_or_loader(ready_inputs, monkeypatch, relative):
    module, root, manifest, *_, calls = ready_inputs
    path = root / relative
    if relative == "catalog-object":
        descriptor = json.loads(
            (root / "history/historical/catalog.parquet.artifact.json").read_bytes()
        )
        path = root / "history/.objects" / descriptor["sha256"]
    with path.open("r+b") as stream:
        stream.truncate(manifest["limits"]["max_memory_bytes"] // 4 + 1)
    original_open = Path.open

    def guarded_open(actual, *args, **kwargs):
        assert actual != path, "Oversized input was opened before checking its size"
        return original_open(actual, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="бюджет"):
        module.load_serving_package(root, manifest)
    assert calls == []


def test_combined_materialized_size_is_checked_before_loaders(ready_inputs):
    module, root, manifest, *_, calls = ready_inputs
    with (root / "model/models/ranker.txt").open("r+b") as stream:
        stream.truncate(manifest["limits"]["max_memory_bytes"] // 4)
    with pytest.raises(ValueError, match="бюджет"):
        module.load_serving_package(root, manifest)
    assert calls == []


def test_json_has_absolute_size_limit_even_with_large_memory_budget(ready_inputs, monkeypatch):
    module, root, manifest, *_ = ready_inputs
    manifest["limits"]["max_memory_bytes"] = 1 << 30
    path = root / "features/reports/features.json"
    with path.open("r+b") as stream:
        stream.truncate((32 << 20) + 1)
    original_open = Path.open

    def guarded_open(actual, *args, **kwargs):
        assert actual != path, "Oversized JSON was read"
        return original_open(actual, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="бюджет"):
        module.load_serving_package(root, manifest)


def test_time_budget_is_checked_between_hash_chunks(ready_inputs, monkeypatch):
    module, root, manifest, *_ = ready_inputs
    path = root / "metadata/products.csv"
    with path.open("r+b") as stream:
        stream.truncate(3 << 20)
    manifest["limits"]["max_metadata_bytes"] = 4 << 20
    manifest["limits"]["max_seconds"] = 1
    clock, reads = [0.0], []
    original_open = Path.open

    class SlowReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            reads.append(size)
            value = self.stream.read(size)
            clock[0] = 2.0
            return value

    def delayed_open(actual, *args, **kwargs):
        stream = original_open(actual, *args, **kwargs)
        return SlowReader(stream) if actual == path else stream

    monkeypatch.setattr(Path, "open", delayed_open)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    with pytest.raises(ValueError, match="бюджет"):
        module.load_serving_package(root, manifest)
    assert len(reads) == 1 and 0 < reads[0] <= 1 << 20


@pytest.mark.parametrize(
    "section,path",
    [("historical", "../outside"), ("model", "/outside"), ("metadata", "metadata/../products.csv")],
)
def test_paths_cannot_escape_pinned_root(ready_inputs, section, path):
    module, root, manifest, *_ = ready_inputs
    manifest[section]["path"] = path
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


def test_retrieval_must_match_features_used_by_producer(ready_inputs):
    module, root, manifest, *_ = ready_inputs
    manifest["retrieval"] = {"history_limit": 19}
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


def test_metadata_read_failure_is_not_record_absence(ready_inputs, monkeypatch):
    module, root, manifest, *_ = ready_inputs

    def broken(*args, **kwargs):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(module, "row_batches", broken)
    with pytest.raises((OSError, ValueError)):
        module.load_serving_package(root, manifest)


def test_duplicate_metadata_id_is_rejected_even_inside_historical_catalog(
    ready_inputs, monkeypatch
):
    module, root, manifest, *_ = ready_inputs
    real = module.row_batches

    def repeated(*args, **kwargs):
        rows = next(real(*args, **kwargs))
        yield (*rows, rows[0])

    monkeypatch.setattr(module, "row_batches", repeated)
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


def test_working_profile_is_not_silently_downgraded(ready_inputs):
    module, root, manifest, *_ = ready_inputs
    manifest["profile"] = "production"
    with pytest.raises(ValueError):
        module.load_serving_package(root, manifest)


@pytest.mark.parametrize(
    "relative", ["schema.json", "model/models/ranker.txt", "metadata/products.csv"]
)
def test_inputs_changed_during_startup_are_not_published(ready_inputs, monkeypatch, relative):
    module, root, manifest, *_ = ready_inputs
    real = module.row_batches

    def changed(*args, **kwargs):
        yield from real(*args, **kwargs)
        path = root / relative
        path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(module, "row_batches", changed)
    with pytest.raises(ValueError, match="изменился"):
        module.load_serving_package(root, manifest)


@pytest.mark.parametrize("offset", [-1, 0])
def test_publication_cannot_precede_or_equal_historical_snapshot(ready_inputs, offset):
    module, root, manifest, history, *_ = ready_inputs
    manifest["published_at_ms"] = history.snapshot_time_ms + offset
    with pytest.raises(ValueError, match="Публикация"):
        module.load_serving_package(root, manifest)


def test_missing_memory_measurement_is_not_unlimited_startup(ready_inputs, monkeypatch):
    module, root, manifest, *_ = ready_inputs
    monkeypatch.setattr(module, "peak_memory", lambda: {"peak_process_memory_bytes": None})
    with pytest.raises(ValueError, match="бюджет"):
        module.load_serving_package(root, manifest)


def test_ambiguous_publication_is_rejected(ready_inputs):
    module, root, manifest, *_ = ready_inputs
    first = module.load_serving_package(root, manifest)
    second = replace(first, package_id="2" * 64)
    with pytest.raises(ValueError, match="Неоднозначная"):
        module.select_package((first, second), t_q=first.published_at_ms + 1)
