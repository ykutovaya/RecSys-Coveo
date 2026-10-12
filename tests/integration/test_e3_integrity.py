"""Целостность публикации Э3 и отказ при превышении закреплённых пределов."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.cli.features import run_features
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.prepared import TABLE_SCHEMA, table_info
from tests.integration.test_e3_pipeline import ROOT, e3_settings  # noqa: F401
from tests.unit.test_e2_input import prepared_source  # noqa: F401


@pytest.mark.parametrize("changed", ["prefix", "historical", "output"])
def test_changed_table_before_publication_never_produces_completed_result(
    e3_settings,  # noqa: F811
    tmp_path,
    changed,
):
    settings, source, historical = e3_settings
    destination = tmp_path / "features"
    paths = {
        "prefix": source / "datasets/B/prefixes.parquet",
        "historical": historical / "historical/content/description/vectors.npy",
        "output": destination / "features/values.parquet",
    }

    def change():
        path = paths[changed]
        path.write_bytes(path.read_bytes() + b"\x00")

    with pytest.raises(ValueError):
        run_features(ROOT, settings, destination, "features", period="B", before_publish=change)
    assert not (destination / "reports/features.json").exists()


@pytest.mark.parametrize("changed", ["schema", "record_type", "rows", "period"])
def test_ready_report_cannot_substitute_schema_or_table_metadata(e3_settings, tmp_path, changed):  # noqa: F811
    settings, _, _ = e3_settings
    destination = tmp_path / "features"
    run_features(ROOT, settings, destination, "features", period="B")
    path = destination / "reports/features.json"
    report = json.loads(path.read_bytes())
    if changed == "schema":
        report["feature_schema"]["columns"][0]["formula"] = "changed formula"
    else:
        info = report["tables"]["features/values.parquet"]
        info[changed] = {"record_type": "TargetRecord", "rows": info["rows"] + 1, "period": "D"}[
            changed
        ]
    path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError):
        run_features(ROOT, settings, destination, "features", period="B")


def test_repeat_cannot_reuse_features_after_retrieval_configuration_change(e3_settings, tmp_path):  # noqa: F811
    settings, _, _ = e3_settings
    destination = tmp_path / "features"
    run_features(ROOT, settings, destination, "features", period="B")
    before = {path: sha256_file(path) for path in destination.rglob("*") if path.is_file()}
    body = json.loads(settings.read_bytes())
    body["retrieval"]["history_limit"] = 1
    settings.write_bytes(canonical_json_bytes(body))
    with pytest.raises(ValueError, match="несовместим"):
        run_features(ROOT, settings, destination, "features", period="B")
    assert before == {path: sha256_file(path) for path in before}


@pytest.mark.parametrize("limit", ["max_memory_bytes", "max_result_bytes"])
def test_resource_limit_refuses_result_without_publishing(e3_settings, tmp_path, limit):  # noqa: F811
    settings, _, _ = e3_settings
    body = json.loads(settings.read_bytes())
    body["limits"][limit] = 1
    settings.write_bytes(canonical_json_bytes(body))
    destination = tmp_path / "features"
    with pytest.raises(ValueError, match="ресурс"):
        run_features(ROOT, settings, destination, "features", period="B")
    assert not (destination / "reports/features.json").exists()


def test_request_limit_rejects_complete_input_instead_of_truncating(e3_settings, tmp_path):  # noqa: F811
    settings, source, _ = e3_settings
    path = source / "datasets/B/prefixes.parquet"
    original = pq.read_table(path)
    first = from_json(original["record_json"][0].as_py(), PreparedRequest)
    second_blocks = tuple(
        replace(
            block,
            session_id="t",
            events=tuple(replace(event, session_id="t") for event in block.events),
        )
        for block in first.prefix.blocks
    )
    second = replace(
        first,
        request_id="q2",
        session_id="t",
        prefix=replace(first.prefix, session_id="t", blocks=second_blocks),
    )
    table = pa.Table.from_pylist(
        [
            {"session_id": first.session_id, "record_json": to_json(first)},
            {"session_id": second.session_id, "record_json": to_json(second)},
        ],
        schema=TABLE_SCHEMA,
    )
    pq.write_table(table, path)
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"]["datasets/B/prefixes.parquet"] = table_info(
        path, PreparedRequest, period="B", role="prefix"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    body = json.loads(settings.read_bytes())
    body["preparation"]["report_sha256"] = sha256_file(report_path)
    body["limits"]["max_requests"] = 1
    settings.write_bytes(canonical_json_bytes(body))
    destination = tmp_path / "features"
    with pytest.raises(ValueError, match="число запросов"):
        run_features(ROOT, settings, destination, "features", period="B")
    assert not (destination / "reports/features.json").exists()
