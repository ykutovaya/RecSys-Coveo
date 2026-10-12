"""Э3 → обучающие группы → модель; только искусственные данные."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.cli.features import run_features
from coveo_recsys.contracts import OutcomeStatus
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import FeatureBatch
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from tests.integration.test_e3_pipeline import e3_settings  # noqa: F401
from tests.reference.e3_cases import request
from tests.unit.test_e2_input import prepared_source  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def e4_settings(e3_settings, tmp_path):  # noqa: F811
    path, source, historical = e3_settings
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    plan = from_json(report["plan_json"].encode(), PreparationPlan)
    plan = replace(
        plan,
        periods=tuple(replace(p, sample_size=160) if p.name == "B" else p for p in plan.periods),
    )
    report["plan_json"] = to_json(plan).decode()
    report["inputs"]["plan"] = sha256_bytes(report["plan_json"].encode())
    (source / "datasets/B/prefixes.parquet").unlink()
    with (
        TableWriter(source / "datasets/B/prefixes.parquet", PreparedRequest) as pw,
        TableWriter(source / "datasets/B/targets.parquet", TargetRecord) as tw,
    ):
        for i in range(160):
            sid, qid = f"s{i:04}", f"q{i:04}"
            base = request(((400001000, "detail", "A"),))
            blocks = tuple(
                replace(
                    b, session_id=sid, events=tuple(replace(e, session_id=sid) for e in b.events)
                )
                for b in base.prefix.blocks
            )
            prefix = replace(base.prefix, session_id=sid, blocks=blocks)
            row = replace(base, request_id=qid, session_id=sid, prefix=prefix)
            pw.append(sid, (row,))
            tw.append(sid, (TargetRecord(qid, ("C",), OutcomeStatus.KNOWN, 1800000),))
    for role, name, kind in [
        ("prefix", "prefixes", PreparedRequest),
        ("targets", "targets", TargetRecord),
    ]:
        relative = f"datasets/B/{name}.parquet"
        report["tables"][relative] = table_info(source / relative, kind, period="B", role=role)
    report_path.write_bytes(canonical_json_bytes(report))
    features_settings = json.loads(path.read_bytes())
    features_settings["preparation"]["report_sha256"] = sha256_file(report_path)
    features_settings["limits"]["max_requests"] = 160
    path.write_bytes(canonical_json_bytes(features_settings))
    features_root = tmp_path / "features"
    run_features(ROOT, path, features_root, "e4-features", period="B")
    settings = {
        "preparation": features_settings["preparation"],
        "historical": features_settings["historical"],
        "features": {
            "path": str(features_root),
            "report_sha256": sha256_file(features_root / "reports/features.json"),
        },
        "feature_schema": "configs/feature_schema.json",
        "recipe": "configs/ranker_recipe.json",
        "limits": {
            "max_requests": 160,
            "max_rows": 48000,
            "max_seconds": 300,
            "max_memory_bytes": 2147483648,
            "max_result_bytes": 67108864,
        },
    }
    configuration = tmp_path / "training.json"
    configuration.write_bytes(canonical_json_bytes(settings))
    return configuration, source, features_root


def test_ranker_cli_end_to_end_repeat_and_predictions(e4_settings, tmp_path, capsys):
    from coveo_recsys.cli import main
    from coveo_recsys.cli.training import run_training
    from coveo_recsys.ranking.load import load_ranker
    from coveo_recsys.ranking.predict import predict

    config, _, features_root = e4_settings
    destination = tmp_path / "ranker"
    result = run_training(ROOT, config, destination, "e4")
    assert result["status"] == "completed" and result["groups"]["pair_count"] == 320
    assert result["groups"]["rows"] == 480
    model = load_ranker(destination)
    batch = next(iter_sessions(features_root / "features/values.parquet", FeatureBatch))[1][0]
    ranked = predict(model, batch, catalog_sha256=result["identity"]["catalog_sha256"])
    assert ranked[0][0] == "C" and len(ranked) == 3
    assert ranked[0][1] > ranked[-1][1]
    assert all(item["path"].startswith("datasets/B/") for item in result["reads"])
    assert run_training(ROOT, config, destination, "e4") == result
    capsys.readouterr()
    assert (
        main(
            [
                "--project-root",
                str(ROOT),
                "train",
                "ranker",
                "--config",
                str(config),
                "--run-dir",
                str(destination),
                "--run-id",
                "e4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["model_sha256"] == result["model_sha256"]


def test_interrupted_training_resumes_without_changing_model(e4_settings, tmp_path):
    from coveo_recsys.cli.training import run_training

    config, _, _ = e4_settings
    destination = tmp_path / "ranker"

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_training(ROOT, config, destination, "e4", before_publish=interrupt)
    assert not (destination / "reports/ranker.json").exists()
    before = sha256_file(destination / "models/ranker.txt")
    result = run_training(ROOT, config, destination, "e4")
    assert result["model_sha256"] == before
    body = json.loads(config.read_bytes())
    body["limits"]["max_seconds"] += 1
    config.write_bytes(canonical_json_bytes(body))
    with pytest.raises(ValueError, match="совместим"):
        run_training(ROOT, config, destination, "e4")
