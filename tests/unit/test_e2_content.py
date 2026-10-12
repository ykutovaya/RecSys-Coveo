"""Э2-05: независимые модальности, словарь A_G и закреплённая поставка."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from coveo_recsys.artifacts import canonical_json_bytes
from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.historical import (
    HistoricalCatalog,
    HistoricalConfig,
    MetadataRecord,
    MetadataSource,
)
from coveo_recsys.data import metadata as metadata_module
from coveo_recsys.data.io import load_profile, parse_profile
from coveo_recsys.data.metadata import open_metadata_source
from coveo_recsys.historical import content as content_module
from coveo_recsys.historical.content import build_content, read_observed_content
from coveo_recsys.numerics.vectors import NormalizedVector
from tests.reference.e2_content_oracle import vector

ATOL, RTOL = 1e-10, 1e-8
CONFIG = HistoricalConfig(snapshot_time_ms=100)


def source(records, *, description_dimension=2, image_dimension=3, requested=None):
    def read(skus):
        if requested is not None:
            requested.append(tuple(skus))
        return iter(record for record in records if record.sku in skus)

    return MetadataSource(read, description_dimension, image_dimension, {}, "unknown")


def synthetic_csv(tmp_path, rows):
    profile = json.loads(Path("configs/synthetic_csv.json").read_text(encoding="utf-8"))
    path = tmp_path / "sku_to_content.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(profile["columns"]["products"].values())
        writer.writerows(rows)
    return path, profile, hashlib.sha256(path.read_bytes()).hexdigest()


def test_modalities_keep_their_own_dimensions_and_directions():
    actual = build_content(
        HistoricalCatalog(("A",)),
        source([MetadataRecord("A", [3, 4], [0, 0, -9])]),
        CONFIG,
    )
    np.testing.assert_allclose(actual.description.vectors, [[0.6, 0.8]], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.image.vectors, [[0, 0, -1]], atol=ATOL, rtol=RTOL)
    assert actual.description.mask.tolist() == actual.image.mask.tolist() == [True]


@pytest.mark.parametrize(
    "raw,reason",
    [
        (None, "vector_absent"),
        ("bad", "vector_invalid"),
        ([True, 1], "vector_invalid"),
        ([1], "vector_invalid"),
        ([[1, 2]], "vector_invalid"),
        ([float("nan"), 0], "vector_invalid"),
        ([float("inf"), 0], "vector_invalid"),
        ([0, 0], "vector_zero"),
        ([np.nextafter(1e-12, 0), 0], "vector_zero"),
        ([1e-12, 0], "vector_zero"),
        ([np.nextafter(1e-12, np.inf), 0], None),
        ([1.7e308, -1.7e308], None),
    ],
)
def test_external_vector_reason_does_not_disable_other_modality(raw, reason):
    actual = build_content(
        HistoricalCatalog(("A",)), source([MetadataRecord("A", raw, [0, 3, 4])]), CONFIG
    )
    assert actual.description.reasons == (reason,)
    assert actual.description.mask.tolist() == [reason is None]
    if reason is not None:
        assert np.all(actual.description.vectors == 0)
    else:
        expected, _, _ = vector([float(value) for value in raw], 2)
        np.testing.assert_allclose(actual.description.vectors[0], expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual.image.vectors, [[0, 0.6, 0.8]], atol=ATOL, rtol=RTOL)
    assert actual.image.mask.tolist() == [True]


def test_internal_nonfinite_normalization_result_is_an_error(monkeypatch):
    monkeypatch.setattr(
        content_module,
        "normalize_external_vector",
        lambda raw, dimension, **kwargs: NormalizedVector(np.full(dimension, np.nan), True, None),
    )
    with pytest.raises(NumericError):
        build_content(HistoricalCatalog(("A",)), source([MetadataRecord("A", [1, 0])]), CONFIG)


def test_only_catalog_categories_form_utf8_dictionary_and_prices_remain_ordinal():
    requested = []
    records = [
        MetadataRecord("A", category="я", price_bucket=7),
        MetadataRecord("B", category="Z", price_bucket=0),
        MetadataRecord("C", category="é", price_bucket=2),
        MetadataRecord("outside", category="future", price_bucket=99),
    ]
    actual = build_content(
        HistoricalCatalog(("A", "B", "C", "missing")),
        source(records[::-1], requested=requested),
        CONFIG,
    )
    assert requested == [("A", "B", "C", "missing")]
    assert actual.categories == ("Z", "é", "я")
    assert actual.category_codes.tolist() == [3, 1, 2, 0]
    assert actual.price_bucket.dtype == np.int64
    assert actual.price_bucket.tolist() == [7, 0, 2, 0]
    assert actual.price_mask.tolist() == [True, True, True, False]
    assert actual.description.reasons == ("vector_absent",) * 4


def test_observed_external_skus_use_frozen_dictionary_without_changing_catalog():
    records = [
        MetadataRecord("A", [1, 0], category="known", price_bucket=4),
        MetadataRecord("outside", [0, 2], category="future", price_bucket=9),
        MetadataRecord("seen", [0, 0], category="known"),
    ]
    metadata = source(records)
    catalog = HistoricalCatalog(("A",))
    historical = build_content(catalog, metadata, CONFIG)
    actual = read_observed_content(
        ("missing", "outside", "seen"), metadata, CONFIG, categories=historical.categories
    )
    assert catalog.skus == ("A",)
    assert actual.categories == historical.categories == ("known",)
    assert historical.category_codes.tolist() == [1]
    assert actual.category_codes.tolist() == [0, 0, 1]
    assert actual.description.mask.tolist() == [False, True, False]
    assert actual.description.reasons == ("vector_absent", None, "vector_zero")
    np.testing.assert_array_equal(actual.description.vectors, [[0, 0], [0, 1], [0, 0]])


def test_empty_catalog_has_empty_arrays_with_declared_dimensions():
    actual = build_content(HistoricalCatalog(()), source([]), CONFIG)
    assert actual.description.vectors.shape == (0, 2)
    assert actual.image.vectors.shape == (0, 3)
    assert actual.categories == actual.description.reasons == ()
    assert actual.category_codes.shape == actual.price_mask.shape == (0,)


@pytest.mark.parametrize(
    "records", [[MetadataRecord("A"), MetadataRecord("A")], [MetadataRecord("B")]]
)
def test_source_cannot_duplicate_requested_sku_or_return_unrequested_sku(records):
    metadata = MetadataSource(lambda skus: iter(records), 2, 3, {}, "unknown")
    with pytest.raises(InputValidationError):
        build_content(HistoricalCatalog(("A",)), metadata, CONFIG)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", 2**63])
def test_price_requires_nonnegative_int64(value):
    with pytest.raises(InputValidationError):
        build_content(
            HistoricalCatalog(("A",)), source([MetadataRecord("A", price_bucket=value)]), CONFIG
        )


def test_profile_dictionary_uses_same_validation_as_file_profile():
    path = Path("configs/synthetic_csv.json")
    loaded = load_profile(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert parse_profile(data, sha256=loaded.sha256) == loaded
    data["dimensions"]["description"] = True
    with pytest.raises(InputValidationError):
        parse_profile(data, sha256=loaded.sha256)


def test_metadata_reads_only_requested_rows_and_repeats_without_inferred_compatibility(tmp_path):
    path, profile, digest = synthetic_csv(
        tmp_path,
        [["A", "[3,4]", "[0,1,0]", "a", "3.0"], ["outside", "garbage", "", "late", "4"]],
    )
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    for _ in range(2):
        rows = list(metadata.read(("A", "missing")))
        assert len(rows) == 1
        assert rows[0] == MetadataRecord("A", [3, 4], [0, 1, 0], "a", 3)
    assert metadata.query_description_compatibility == "unknown"
    assert metadata.static_availability == "unknown"
    assert digest in metadata.input_hashes.values()
    profile_hash = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    assert profile_hash in metadata.input_hashes.values()


def test_equal_dimensions_do_not_create_space_compatibility(tmp_path):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "[1,0]", "[0,1]", "", ""]])
    profile["dimensions"]["image"] = 2
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    assert metadata.description_dimension == metadata.image_dimension == 2
    assert metadata.query_description_compatibility == "unknown"


def test_actual_profile_preserves_publisher_statement_and_exact_header(tmp_path):
    profile = json.loads(Path("configs/coveo_supply_v1.json").read_text(encoding="utf-8"))
    path = tmp_path / "sku_to_content.csv"
    columns = profile["files"][path.name]["columns"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow(["A", "", "category", "", "7.0"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    profile["files"][path.name]["expected_sha256"] = digest
    profile["files"][path.name]["expected_size_bytes"] = path.stat().st_size
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    assert metadata.description_dimension == metadata.image_dimension == 50
    assert metadata.query_description_compatibility == "publisher_statement"
    assert list(metadata.read(("A",))) == [MetadataRecord("A", category="category", price_bucket=7)]
    profile["files"][path.name]["expected_size_bytes"] += 1
    with pytest.raises(InputValidationError):
        open_metadata_source(path, profile, expected_sha256=digest)


@pytest.mark.parametrize("phase", ["open", "reread"])
def test_metadata_rejects_hash_mismatch_on_open_or_repeated_read(tmp_path, phase):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "", "", "a", "1"]])
    if phase == "open":
        with pytest.raises(InputValidationError):
            open_metadata_source(path, profile, expected_sha256="0" * 64)
    else:
        metadata = open_metadata_source(path, profile, expected_sha256=digest)
        assert len(list(metadata.read(("A",)))) == 1
        path.write_bytes(path.read_bytes().replace(b",a,", b",b,"))
        with pytest.raises(InputValidationError):
            list(metadata.read(("A",)))


def test_metadata_does_not_return_rows_if_input_changes_during_read(tmp_path, monkeypatch):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "", "", "a", "1"]])
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    original_reader = metadata_module.row_batches

    def changed_reader(*args, **kwargs):
        yield from original_reader(*args, **kwargs)
        path.write_bytes(path.read_bytes().replace(b",a,", b",b,"))

    monkeypatch.setattr(metadata_module, "row_batches", changed_reader)
    with pytest.raises(InputValidationError):
        next(metadata.read(("A",)))


def test_metadata_rejects_duplicate_requested_rows(tmp_path):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "", "", "", ""]] * 2)
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    with pytest.raises(InputValidationError):
        list(metadata.read(("A",)))


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "vector_absent"),
        ("null", "vector_absent"),
        ("no", "vector_invalid"),
        ("[NaN,0]", "vector_invalid"),
        ("[true,0]", "vector_invalid"),
        ("[0,0]", "vector_zero"),
    ],
)
def test_csv_vector_defects_preserve_reasons(tmp_path, text, reason):
    path, profile, digest = synthetic_csv(tmp_path, [["A", text, "[0,0,1]", "", ""]])
    actual = build_content(
        HistoricalCatalog(("A",)),
        open_metadata_source(path, profile, expected_sha256=digest),
        CONFIG,
    )
    assert actual.description.reasons == (reason,)
    assert actual.image.mask.tolist() == [True]


@pytest.mark.parametrize("text", ["low", "1.5", "-1", "nan", "9223372036854775808"])
def test_metadata_does_not_turn_invalid_prices_into_categories(tmp_path, text):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "", "", "a", text]])
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    with pytest.raises(InputValidationError):
        list(metadata.read(("A",)))


def test_csv_metadata_is_streamed_with_existing_bounded_reader(tmp_path, monkeypatch):
    path, profile, digest = synthetic_csv(tmp_path, [["A", "", "", "", ""]])
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("CSV read_bytes запрещён"))
    monkeypatch.setattr(
        Path, "read_text", lambda self, **kwargs: pytest.fail("CSV read_text запрещён")
    )
    metadata = open_metadata_source(path, profile, expected_sha256=digest)
    assert list(metadata.read(("A",))) == [MetadataRecord("A")]
