"""B10.metadata: синтетические A_G/B → собственные признаки → модель → выдача."""

import json
from dataclasses import replace

import numpy as np

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.config.resources import resource_bytes
from coveo_recsys.contracts import OutcomeStatus
from coveo_recsys.contracts.historical import Modality
from coveo_recsys.contracts.retrieval import NO_METADATA_CONFIG_ID, RetrievalConfig
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.features.engine import FeatureEngine
from coveo_recsys.ranking.load import load_ranker
from coveo_recsys.ranking.predict import predict
from coveo_recsys.ranking.table import build_training_table
from coveo_recsys.ranking.train import fit_ranker
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context
from tests.reference.e3_cases import CATALOG_HASH, package, request


def _example(historical, prepared, config_id):
    no_metadata = config_id == NO_METADATA_CONFIG_ID
    config = (
        RetrievalConfig(quotas=(50, 50, 50, 0, 0, 0, 20, 10)) if no_metadata else RetrievalConfig()
    )
    index = RetrievalIndex(historical, CATALOG_HASH, config_id=config_id)
    schema_path = (
        "configs/feature_schema_no_metadata.json" if no_metadata else "configs/feature_schema.json"
    )
    schema = json.loads(resource_bytes(schema_path))
    engine = FeatureEngine(index, schema, config)
    context = bind_context(prepared, index)
    candidates = retrieve(context, index, config)
    batch = engine.build(context, candidates)
    return prepared, candidates, batch


def _train_and_load(root, example, config_id):
    prepared, candidates, batch = example
    examples = []
    # 160 положительных строк позволяют выполнить штатный min_data_in_leaf=100.
    for i in range(160):
        sid, qid = f"s{i:04}", f"q{i:04}"
        blocks = tuple(
            replace(
                block,
                session_id=sid,
                events=tuple(replace(event, session_id=sid) for event in block.events),
            )
            for block in prepared.prefix.blocks
        )
        item = replace(
            prepared,
            request_id=qid,
            session_id=sid,
            prefix=replace(prepared.prefix, session_id=sid, blocks=blocks),
        )
        examples.append(
            (
                item,
                replace(candidates, request_id=qid),
                replace(batch, request_id=qid),
                TargetRecord(qid, ("c",), OutcomeStatus.KNOWN, 1800000),
            )
        )
    table = build_training_table(
        examples,
        feature_schema_sha256=batch.feature_schema_sha256,
        numeric_columns=batch.numeric_columns,
        categorical_columns=batch.categorical_columns,
        max_candidates=300,
        max_rows=48000,
        max_memory_bytes=1_000_000_000,
        max_disk_bytes=1_000_000_000,
    )
    assert table.group_sizes == (4,) * 160
    assert table.statistics["pair_count"] == 480
    assert table.labels.tolist() == [0, 0, 1, 0] * 160
    booster, report = fit_ranker(table, get_ranker_recipe(config_id), config_id=config_id)
    payload = booster.model_to_string(num_iteration=report["actual_tree_count"]).encode("utf-8")
    summary = {
        **report,
        "format_version": "ranker_v1",
        "status": "completed",
        "config_id": config_id,
        "identity": {
            "config_id": config_id,
            "catalog_sha256": CATALOG_HASH,
            "feature_schema_sha256": batch.feature_schema_sha256,
        },
        "model_sha256": sha256_bytes(payload),
        "feature_schema_sha256": batch.feature_schema_sha256,
        "numeric_columns": list(batch.numeric_columns),
        "categorical_columns": list(batch.categorical_columns),
    }
    (root / "models").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "models/ranker.txt").write_bytes(payload)
    (root / "reports/ranker.json").write_bytes(canonical_json_bytes(summary))
    loaded = load_ranker(root, expected_identity=summary["identity"])
    ranked = predict(loaded, batch, catalog_sha256=CATALOG_HASH)
    assert ranked[0][0] == "c" and len(ranked) == 4
    assert ranked[0][1] > ranked[-1][1]
    return loaded, payload, ranked


def _replacements(historical):
    content = historical.content
    description = Modality(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, -1.0]]),
        np.array([False, True, False, True]),
        ("vector_absent", None, "vector_absent", None),
    )
    image = Modality(
        np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        np.array([False, True, True, False]),
        ("vector_absent", None, None, "vector_absent"),
    )
    changes = {
        "description": {"description": description},
        "image": {"image": image},
        "category": {
            "categories": ("other/x", "other/y"),
            "category_codes": np.array([0, 2, 0, 1], dtype=np.int64),
        },
        "price": {
            "price_bucket": np.array([0, 8, 0, 9], dtype=np.int64),
            "price_mask": np.array([False, True, False, True]),
        },
    }
    changes["all"] = {key: value for fields in changes.values() for key, value in fields.items()}
    for name, fields in changes.items():
        yield (
            name,
            replace(
                historical,
                content=replace(content, **fields),
                input_hashes={**historical.input_hashes, "metadata_sha256": "d" * 64},
            ),
        )


def test_no_metadata_replacements_preserve_candidates_features_model_and_delivery(tmp_path):
    historical = package()
    prepared = request(query=(1.0, 0.0), search_time=1500)
    example = _example(historical, prepared, NO_METADATA_CONFIG_ID)
    _, candidates, batch = example
    assert candidates.skus == ("a", "b", "c", "я")
    assert tuple(source.skus for source in candidates.sources) == (
        ("b", "c"),
        ("c",),
        ("b",),
        (),
        (),
        (),
        ("a",),
        ("a", "b", "c", "я"),
    )
    assert tuple(source.quota for source in candidates.sources) == (50, 50, 50, 0, 0, 0, 20, 10)
    assert batch.categorical_columns == ()
    assert all(row == () for row in batch.categorical)
    assert not any(
        forbidden in column
        for column in batch.numeric_columns
        for forbidden in ("description", "image", "search", "category", "price")
    )
    model, payload, ranked = _train_and_load(tmp_path / "original", example, NO_METADATA_CONFIG_ID)
    alternatives = [
        (
            name,
            altered,
            request(query=(0.0, -1.0), search_time=1500) if name == "all" else prepared,
        )
        for name, altered in _replacements(historical)
    ]
    alternatives.extend(
        ("search", historical, request(query=query, search_time=1500))
        for query in ((0.0, -1.0), None)
    )
    for name, altered, altered_request in alternatives:
        changed = _example(altered, altered_request, NO_METADATA_CONFIG_ID)
        assert changed[1] == candidates, name
        assert changed[2] == batch, name
        assert predict(model, changed[2], catalog_sha256=CATALOG_HASH) == ranked, name
        if name == "all":
            _, altered_payload, altered_ranked = _train_and_load(
                tmp_path / "replaced", changed, NO_METADATA_CONFIG_ID
            )
            assert altered_payload == payload
            assert altered_ranked == ranked


def test_metadata_profile_still_trains_loads_and_delivers_with_category(tmp_path):
    example = _example(package(), request(query=(1.0, 0.0), search_time=1500), "gcs_rk_v2_7")
    assert example[2].categorical_columns == ("category",)
    assert "score_description" in example[2].numeric_columns
    model, _, _ = _train_and_load(tmp_path, example, "gcs_rk_v2_7")
    assert model.summary["categorical_columns"] == ["category"]
