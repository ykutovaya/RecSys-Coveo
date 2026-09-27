"""B01/B02 и §7.2: сырой вход → префикс → кандидаты → признаки."""

import csv
import json
from dataclasses import replace
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from coveo_recsys.cli.features import run_features
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest, TaskConfig
from coveo_recsys.contracts.records import CandidateResult, FeatureBatch, RequestContext
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.io import iter_events, load_profile
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from coveo_recsys.evaluation.replay import recommend
from coveo_recsys.features import FeatureEngine
from coveo_recsys.historical.load import load_historical
from coveo_recsys.ranking.load import RankerModel
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from coveo_recsys.state import build_prefix
from tests.integration.test_e3_pipeline import e3_settings  # noqa: F401
from tests.reference.e3_cases import CATALOG_HASH, package, request
from tests.unit.test_e2_input import prepared_source  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("change", ["future_event", "future_availability", "search_lists"])
def test_future_events_and_search_lists_do_not_change_candidates_or_features(tmp_path, change):
    profile = load_profile(ROOT / "configs/synthetic_csv.json")
    historical = package()
    index = RetrievalIndex(historical, CATALOG_HASH)
    engine = FeatureEngine(index, json.loads((ROOT / "configs/feature_schema.json").read_bytes()))
    model = RankerModel(
        SimpleNamespace(predict=lambda matrix, **_: np.sum(matrix, axis=1)),
        {
            "identity": {"catalog_sha256": CATALOG_HASH},
            "feature_schema_sha256": engine.schema_sha256,
            "numeric_columns": list(engine.numeric_columns),
            "categorical_columns": list(engine.categorical_columns),
            "actual_tree_count": 1,
        },
    )
    results = []
    for changed in (False, True):
        browsing = tmp_path / "browsing.csv"
        search = tmp_path / "search.csv"
        mutate_future = changed and change == "future_event"
        mutate_late = changed and change == "future_availability"
        mutate_lists = changed and change == "search_lists"
        with browsing.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(
                [
                    [
                        "session_id",
                        "event_id",
                        "server_timestamp_epoch_ms",
                        "product_action",
                        "sku",
                    ],
                    ["s", "view-a", 2000, "detail", "a"],
                    ["s", "view-b", 3000, "detail", "b"],
                    [
                        "s",
                        "late",
                        1000,
                        "purchase" if mutate_late else "detail",
                        "я" if mutate_late else "c",
                    ],
                    [
                        "s",
                        "future",
                        9000 if mutate_future else 4000,
                        "detail" if mutate_future else "purchase",
                        "a" if mutate_future else "c",
                    ],
                ]
            )
        with search.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(
                [
                    [
                        "session_id",
                        "event_id",
                        "server_timestamp_epoch_ms",
                        "query_vector",
                        "product_skus_hash",
                        "clicked_skus_hash",
                    ],
                    [
                        "s",
                        "search",
                        2500,
                        "[1,0]",
                        '["я","outside"]' if mutate_lists else '["a"]',
                        '["я"]' if mutate_lists else "[]",
                    ],
                    [
                        "s",
                        "future-search",
                        5000,
                        "[-1,0]" if mutate_future else "[0,1]",
                        "[]",
                        "[]",
                    ],
                ]
            )
        raw = chain(
            iter_events(browsing, "browsing", profile), iter_events(search, "search", profile)
        )
        delayed = (
            replace(event, available_time_ms=4000) if event.event_id == "late" else event
            for event in raw
        )
        blocks = tuple(canonicalize(delayed, temp_dir=tmp_path, chunk_size=2))
        prefix = build_prefix(
            blocks,
            source_namespace="synthetic",
            session_id="s",
            event_cutoff_ms=3000,
            availability_cutoff_ms=3000,
            config=TaskConfig(),
        )
        trigger = next(block.block_id for block in prefix.blocks if block.event_time_ms == 3000)
        context = RequestContext("q", "s", trigger, 3000, 3000, 1, prefix, CATALOG_HASH)
        selected = retrieve(context, index)
        features = engine.build(context, selected)
        # У изменённого CSV иной хеш: исключаем только ссылки на исходные байты.
        semantic_prefix = replace(
            prefix,
            blocks=tuple(
                replace(
                    block,
                    events=tuple(
                        replace(event, provenance="fixture", origins=()) for event in block.events
                    ),
                )
                for block in prefix.blocks
            ),
        )
        replies = tuple(
            recommend(context, engine, model, policy=policy, package_id="test", trace_ref="q")
            for policy in ("GCS+Rk", "P", "R")
        )
        results.append((semantic_prefix, selected, features, replies))
    assert results[0] == results[1]
    assert results[0][0].purchased_skus == ()
    assert set(results[0][1].skus) == set(historical.catalog.skus)


def test_saved_batches_match_sequential_path_and_batch_size(e3_settings, tmp_path):  # noqa: F811
    settings_path, source, historical = e3_settings
    prefix_path = source / "datasets/B/prefixes.parquet"
    original = next(iter_sessions(prefix_path, PreparedRequest))[1][0]
    requests = []
    for number in range(3):
        sid = f"s-{number}"
        prepared = request(((original.t_q, "detail", ("A", "B", "C")[number]),))
        blocks = tuple(
            replace(
                block,
                session_id=sid,
                events=tuple(replace(event, session_id=sid) for event in block.events),
            )
            for block in prepared.prefix.blocks
        )
        requests.append(
            replace(
                prepared,
                request_id=f"q-{number}",
                session_id=sid,
                prefix=replace(prepared.prefix, session_id=sid, blocks=blocks),
            )
        )
    prefix_path.unlink()
    with TableWriter(prefix_path, PreparedRequest, batch_rows=1) as writer:
        for item in requests:
            writer.append(item.session_id, (item,))
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = from_json(report["plan_json"].encode(), PreparationPlan)
    plan = replace(
        plan,
        periods=tuple(
            replace(period, sample_size=3) if period.name == "B" else period
            for period in plan.periods
        ),
    )
    report["plan_json"] = to_json(plan).decode()
    report["inputs"]["plan"] = sha256_bytes(report["plan_json"].encode())
    report["tables"]["datasets/B/prefixes.parquet"] = table_info(
        prefix_path, PreparedRequest, period="B", role="prefix"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(settings_path.read_bytes())
    settings["preparation"]["report_sha256"] = sha256_file(report_path)
    history = LocalArtifactStore(historical)
    index = RetrievalIndex(
        load_historical(history), history.describe("historical/catalog.parquet").sha256
    )
    schema = json.loads((ROOT / settings["feature_schema"]).read_bytes())
    config = RetrievalConfig(**settings["retrieval"])
    engine = FeatureEngine(index, schema, config)
    sequential = []
    for _, items in iter_sessions(prefix_path, PreparedRequest):
        for item in items:
            context = bind_context(item, index)
            scores = score_request(context, index, config)
            selected = retrieve(context, index, config, scores=scores)
            sequential.append((selected, engine.build(context, selected, scores=scores)))
    previous = None
    for size in (1, 2):
        settings["limits"]["batch_size"] = size
        settings_path.write_bytes(canonical_json_bytes(settings))
        destination = tmp_path / f"features-{size}"
        assert (
            run_features(ROOT, settings_path, destination, f"features-{size}", period="B")[
                "requests"
            ]
            == 3
        )
        candidates = [
            item
            for _, items in iter_sessions(
                destination / "features/candidates.parquet", CandidateResult
            )
            for item in items
        ]
        batches = [
            item
            for _, items in iter_sessions(destination / "features/values.parquet", FeatureBatch)
            for item in items
        ]
        assert candidates == [item[0] for item in sequential]
        for actual, (_, expected) in zip(batches, sequential, strict=True):
            assert replace(actual, numeric=expected.numeric) == expected
            replay, serve = np.asarray(actual.numeric), np.asarray(expected.numeric)
            relative_error = np.abs(replay - serve) / (1e-8 + np.abs(replay))
            assert np.max(relative_error, initial=0.0) <= 1e-6
        if previous is not None:
            assert batches == previous
        previous = batches
