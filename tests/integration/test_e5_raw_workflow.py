"""ТЗ15.4: исходные CSV → A_G/B/C_smoke → готовая модель → независимый Hit.

Цели не конструируются контрактами и не вписываются в Parquet вручную:
их получает штатная подготовка из событий CSV. Ожидания заданы до расчёта.
"""

import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_file
from coveo_recsys.cli import main
from coveo_recsys.cli.features import run_features
from coveo_recsys.cli.historical import run_historical
from coveo_recsys.cli.preparation import Preparation
from coveo_recsys.cli.training import run_training
from coveo_recsys.contracts import OutcomeStatus
from coveo_recsys.contracts.evaluation import POLICIES, EvaluationRecord, ReplayRecord
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import iter_sessions

ROOT = Path(__file__).resolve().parents[2]
MEMORY_BYTES = 2_147_483_648
RESULT_BYTES = 67_108_864


def _config(path, value):
    path.write_bytes(canonical_json_bytes(value))
    return path


def _prepare(root, inputs, base, period, sample_size, batch_size, name):
    plan = replace(
        base,
        task=replace(base.task, horizon_ms=1000),
        periods=tuple(
            replace(item, sample_size=sample_size if item.name == period else 0)
            for item in base.periods
        ),
        max_memory_bytes=MEMORY_BYTES,
        max_temp_bytes=MEMORY_BYTES,
    )
    plan_path = root / f"{name}-plan.json"
    plan_path.write_bytes(to_json(plan))
    destination = root / name
    Preparation(LocalArtifactStore(destination), ROOT, name).prepare(
        inputs,
        ROOT / "configs/synthetic_csv.json",
        plan_path,
        batch_size=batch_size,
        max_batch_chars=4096,
    )
    report_path = destination / "reports/preparation.json"
    return destination, plan_path, json.loads(report_path.read_bytes())


def _report_input(path, filename="preparation"):
    return {
        "path": str(path),
        "report_sha256": sha256_file(path / f"reports/{filename}.json"),
    }


def test_raw_csv_to_report_keeps_unknown_targets_and_is_chunk_invariant(tmp_path, capsys):
    inputs = tmp_path / "input"
    shutil.copytree(ROOT / "tests/fixtures/e1", inputs)
    products = inputs / "sku_to_content.csv"
    products.write_text(
        products.read_text(encoding="utf-8").replace(",low", ",1").replace(",high", ",2"),
        encoding="utf-8",
    )
    # В A_G сохраняются A/B/C из готовых строк Э1. B и C добавлены в те же CSV.
    with (inputs / "browsing_train.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for index in range(160):
            sid = f"b{index:04d}"
            writer.writerow((sid, sid + "-view", 400001000, "detail", "A"))
            writer.writerow((sid, sid + "-purchase", 400001500, "purchase", "C"))
        for sid in ("e0", "e1", "e2"):
            writer.writerow((sid, sid + "-view", 600001000, "detail", "A"))
        writer.writerow(("e0", "e0-purchase", 600001500, "purchase", "C"))
        writer.writerow(("e2", "e2-purchase", 600001500, "purchase", "outside"))

    base = from_json((ROOT / "configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    ag, _, ag_report = _prepare(tmp_path, inputs, base, "A_G", 3, 7, "ag")
    training_source, _, training_report = _prepare(tmp_path, inputs, base, "B", 160, 7, "b")
    assert ag_report["N_obs"] == 2
    assert training_report["N_obs"] == training_report["N_plus"] == 160

    historical = tmp_path / "historical"
    run_historical(
        ROOT,
        _config(
            tmp_path / "historical.json",
            {
                "preparation": _report_input(ag),
                "metadata_path": str(products),
                "config": {"snapshot_time_ms": 200000000, "max_memory_bytes": MEMORY_BYTES},
            },
        ),
        historical,
        "raw-historical",
    )
    historical_input = {
        "path": str(historical),
        "package_sha256": sha256_file(historical / "historical/package.json"),
    }
    features = tmp_path / "features"
    run_features(
        ROOT,
        _config(
            tmp_path / "features.json",
            {
                "preparation": _report_input(training_source),
                "historical": historical_input,
                "feature_schema": "configs/feature_schema.json",
                "retrieval": {},
                "limits": {
                    "max_requests": 160,
                    "batch_size": 7,
                    "max_observed_skus": 100,
                    "max_seconds": 300,
                    "max_memory_bytes": MEMORY_BYTES,
                    "max_result_bytes": RESULT_BYTES,
                },
            },
        ),
        features,
        "raw-features",
        period="B",
    )
    model = tmp_path / "model"
    run_training(
        ROOT,
        _config(
            tmp_path / "training.json",
            {
                "preparation": _report_input(training_source),
                "historical": historical_input,
                "features": _report_input(features, "features"),
                "feature_schema": "configs/feature_schema.json",
                "recipe": "configs/ranker_recipe.json",
                "limits": {
                    "max_requests": 160,
                    "max_rows": 48000,
                    "max_seconds": 300,
                    "max_memory_bytes": MEMORY_BYTES,
                    "max_result_bytes": RESULT_BYTES,
                },
            },
        ),
        model,
        "raw-model",
    )
    estimates = (*POLICIES, "GCS+Rk-P", "GCS+Rk-R")
    evaluation_plan = {
        "period": "C_smoke",
        "policies": list(POLICIES),
        "k": 20,
        "diagnostic_depth": 3,
        "intervals": {
            "grid_step_ms": 1981990,
            "block_cells": 2,
            "sensitivity_block_cells": [1, 3],
            "mc_error_method": "binomial_order_statistic_95pct",
            "dependence": {name: {"supported": None, "assumptions": []} for name in estimates},
            "half_width_fraction": dict.fromkeys(estimates),
        },
    }
    prepared_outputs, saved_answers, saved_evaluations = [], [], []
    for batch_size in (1, 7):
        source, plan_path, prepared = _prepare(
            tmp_path, inputs, base, "C_smoke", 3, batch_size, f"c-{batch_size}"
        )
        prepared_outputs.append((source, prepared))
        observed_targets = {
            sid: (records[0].status, records[0].purchases)
            for sid, records in iter_sessions(
                source / "datasets/C_smoke/targets.parquet", TargetRecord
            )
        }
        assert observed_targets == {
            "e0": (OutcomeStatus.KNOWN, ("C",)),
            "e1": (OutcomeStatus.UNKNOWN, ()),
            "e2": (OutcomeStatus.KNOWN, ("outside",)),
        }
        settings = _config(
            tmp_path / f"replay-{batch_size}.json",
            {
                "preparation": _report_input(source),
                "training_preparation": _report_input(training_source),
                "training_features": _report_input(features, "features"),
                "historical": historical_input,
                "model": _report_input(model, "ranker"),
                "preparation_plan": str(plan_path),
                "feature_schema": "configs/feature_schema.json",
                "recipe": "configs/ranker_recipe.json",
                "retrieval": {},
                "evaluation_plan": evaluation_plan,
                "limits": {
                    "max_requests": 3,
                    "max_observed_skus": 100,
                    "batch_size": batch_size,
                    "max_seconds": 300,
                    "max_memory_bytes": MEMORY_BYTES,
                    "max_result_bytes": RESULT_BYTES,
                },
            },
        )
        destination = tmp_path / f"result-{batch_size}"
        capsys.readouterr()
        assert (
            main(
                [
                    "--project-root",
                    str(ROOT),
                    "replay",
                    "--config",
                    str(settings),
                    "--period",
                    "C_smoke",
                    "--run-dir",
                    str(destination),
                    "--run-id",
                    f"raw-{batch_size}",
                ]
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "completed"
        assert {key: report["data"][key] for key in ("N_obs", "N", "N_unknown")} == {
            "N_obs": 3,
            "N": 2,
            "N_unknown": 1,
        }
        assert report["S_offline"] == "not_run"
        assert report["business_effect"] == "not_observed"
        answers = dict(iter_sessions(destination / "replay/answers.parquet", ReplayRecord))
        for policy in POLICIES:
            metrics = report["policies"][policy]["metrics"]
            assert metrics["Hit@20"]["value"] == metrics["Recall@20"]["value"] == 0.5
            assert metrics["Hit@20"]["denominator"] == 2
            shown = next(row.result.delivered_skus for row in answers["e0"] if row.policy == policy)
            assert set(shown) == {"A", "B", "C"}
            # Один положительный SKU в одной из двух целевых сессий; ранг не предполагается.
            expected_ndcg = 1 / (2 * math.log2(shown.index("C") + 2))
            assert metrics["NDCG@20"]["value"] == pytest.approx(expected_ndcg)
        saved_answers.append(
            [
                (
                    sid,
                    tuple(
                        replace(row, result=replace(row.result, elapsed_ms=0, trace_ref=""))
                        for row in rows
                    ),
                )
                for sid, rows in answers.items()
            ]
        )
        evaluations = dict(iter_sessions(destination / "replay/sessions.parquet", EvaluationRecord))
        saved_evaluations.append(
            [
                (sid, tuple((row.policy, row.evaluation) for row in rows))
                for sid, rows in evaluations.items()
            ]
        )
        assert all(row.evaluation.hit_s is None for row in evaluations["e1"])
        assert main(["--project-root", str(ROOT), "report", "--run", str(destination)]) == 0
        assert json.loads(capsys.readouterr().out) == report
        assert (destination / "reports/replay.md").is_file()

    (first, first_report), (second, second_report) = prepared_outputs
    for path, kind in (
        ("canonical/C_smoke.parquet", CanonicalBlock),
        ("datasets/C_smoke/prefixes.parquet", PreparedRequest),
        ("datasets/C_smoke/targets.parquet", TargetRecord),
    ):
        assert list(iter_sessions(first / path, kind)) == list(iter_sessions(second / path, kind))
        assert first_report["tables"][path]["sha256"] == second_report["tables"][path]["sha256"]
    assert saved_answers[0] == saved_answers[1]
    assert saved_evaluations[0] == saved_evaluations[1]
