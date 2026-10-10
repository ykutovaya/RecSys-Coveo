"""Малый D только в tmp_path; допуск отдельно проверяется без подмены валидатора."""

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_file
from coveo_recsys.contracts.evaluation import EvaluationRecord, ReplayRecord
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.retrieval import NO_METADATA_CONFIG_ID
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from tests.integration.test_e5_replay_pipeline import ROOT, make_replay_template
from tests.unit.test_e2_input import (
    prepared_source,  # noqa: F401
    rewrite_records,
)
from tests.unit.test_e7_evaluate_admission import fixed_manifest
from tests.unit.test_e7_retrieval_features import MetadataReadGuard


@pytest.fixture(scope="module")
def final_template(prepared_source, tmp_path_factory):  # noqa: F811
    from coveo_recsys.cli.features import run_features
    from coveo_recsys.cli.training import run_training

    root = tmp_path_factory.mktemp("e7-final")
    settings, roots = make_replay_template(prepared_source, root, "D")
    schema = "configs/feature_schema_no_metadata.json"
    recipe = "configs/ranker_recipe_no_metadata.json"
    retrieval = {"quotas": [50, 50, 50, 0, 0, 0, 20, 10]}
    features, model = root / "metadata-features", root / "metadata-model"
    path = root / "metadata.json"
    path.write_bytes(
        canonical_json_bytes(
            dict(
                preparation=settings["training_preparation"],
                historical=settings["historical"],
                feature_schema=schema,
                retrieval=retrieval,
                limits={**settings["limits"], "max_requests": 160},
            )
        )
    )
    run_features(
        ROOT, path, features, "no-meta", period="B", required_config_id=NO_METADATA_CONFIG_ID
    )
    feature_binding = {
        "path": str(features),
        "report_sha256": sha256_file(features / "reports/features.json"),
    }
    # Тот же B и рецепт; структура настроек берётся из готового малого обучения.
    training_path = root / "training.json"
    training = json.loads(training_path.read_bytes())
    training.update(features=feature_binding, feature_schema=schema, recipe=recipe)
    path.write_bytes(canonical_json_bytes(training))
    run_training(ROOT, path, model, "no-meta", required_config_id=NO_METADATA_CONFIG_ID)
    settings["metadata_sensitivity"] = dict(
        model={"path": str(model), "report_sha256": sha256_file(model / "reports/ranker.json")},
        training_features=feature_binding,
        feature_schema=schema,
        recipe=recipe,
        retrieval=retrieval,
    )
    policies = ["GCS+Rk", "P", "R", NO_METADATA_CONFIG_ID]
    settings["evaluation_plan"]["policies"] = policies
    intervals = settings["evaluation_plan"]["intervals"]
    for name in (NO_METADATA_CONFIG_ID, "GCS+Rk-" + NO_METADATA_CONFIG_ID):
        intervals["dependence"][name] = {"supported": None, "assumptions": []}
        intervals["half_width_fraction"][name] = None
    return settings, roots


@pytest.fixture
def final_settings(final_template, tmp_path, monkeypatch):
    from coveo_recsys.cli import admission

    # Только вычислительный тест на синтетических таблицах. Отдельные тесты
    # проверяют настоящий pre_D и отказ до чтения D.
    monkeypatch.setattr(admission, "require_final_admission", lambda *a, **k: "a" * 64)
    settings, roots = deepcopy(final_template)
    path = tmp_path / "evaluate.json"
    path.write_bytes(canonical_json_bytes(settings))
    return path, roots


def test_final_common_denominator_own_features_and_saved_report(
    final_settings, tmp_path, monkeypatch
):
    from coveo_recsys.cli import replay
    from coveo_recsys.features import FeatureEngine

    path, _ = final_settings
    destination = tmp_path / "result"
    builds = []
    original = FeatureEngine.build
    original_index = replay.RetrievalIndex

    def guarded_index(package, *args, **kwargs):
        if kwargs.get("config_id") == NO_METADATA_CONFIG_ID:
            package = MetadataReadGuard(package, {"content", "metadata_contract", "input_hashes"})
        return original_index(package, *args, **kwargs)

    def capture(self, context, *args, **kwargs):
        batch = original(self, context, *args, **kwargs)
        builds.append((context.t_q, self.index.config_id, batch))
        return batch

    monkeypatch.setattr(FeatureEngine, "build", capture)
    monkeypatch.setattr(replay, "RetrievalIndex", guarded_index)
    result = replay.run_replay(ROOT, path, destination, "final-test", period="D")
    assert result["period"] == "D" and result["requests"] == 3
    assert result["S_offline"] == "not_run" and result["business_effect"] == "not_observed"
    assert {row[1] for row in builds} == {"gcs_rk_v2_7", NO_METADATA_CONFIG_ID}
    assert all(time >= 1_200_000_000 for time, _, _ in builds)
    answers = dict(iter_sessions(destination / "replay/answers.parquet", ReplayRecord))
    evaluations = dict(iter_sessions(destination / "replay/sessions.parquet", EvaluationRecord))
    assert all(len(rows) == 4 for rows in answers.values())
    for policy, report in result["policies"].items():
        # Независимые суммы на двух зарегистрированных целях; e1 неизвестна.
        hits = [
            int(bool(set(row.result.delivered_skus) & ({"C"} if sid == "e0" else {"outside"})))
            for sid, rows in answers.items()
            if sid != "e1"
            for row in rows
            if row.policy == policy
        ]
        assert report["counts"]["N"] == 2
        assert report["metrics"]["Hit@20"]["value"] == sum(hits) / 2
        assert result["intervals"][policy]["metric"]["status"] == "insufficient"
    assert all(row.evaluation.hit_s is None for row in evaluations["e1"])
    assert all(row.evaluation.hit_s == 0 for row in evaluations["e2"])
    assert result["metadata_sensitivity"]["confirmatory"] is False
    assert result["interpretation"]["next_comparison"] == "review_final_result_without_retuning"
    for other in ("P", "R", NO_METADATA_CONFIG_ID):
        expected = (
            result["policies"]["GCS+Rk"]["metrics"]["Hit@20"]["value"]
            - result["policies"][other]["metrics"]["Hit@20"]["value"]
        )
        pair = "GCS+Rk-" + other
        assert result["pairs"][pair]["value"] == expected
        assert result["intervals"][pair]["metric"]["value"] == expected

    def forbidden(*args, **kwargs):
        pytest.fail("Готовая сводка не пересчитывает рекомендации")

    monkeypatch.setattr(replay, "recommend", forbidden)
    assert replay.read_report(destination) == result
    assert replay.run_replay(ROOT, path, destination, "final-test", period="D") == result


@pytest.mark.parametrize(
    "change", ["metadata_schema", "primary_model", "metadata_model", "historical_part"]
)
@pytest.mark.parametrize("consumer", ["evaluate", "prepare"])
def test_substituted_input_fails_before_d_read(
    final_settings, tmp_path, monkeypatch, change, consumer
):
    from coveo_recsys.cli import replay

    path, roots = final_settings
    settings = json.loads(path.read_bytes())
    if change == "metadata_schema":
        settings["metadata_sensitivity"]["feature_schema"] = settings["feature_schema"]
    elif change == "historical_part":
        copied = tmp_path / "historical"
        shutil.copytree(roots["historical"], copied)
        settings["historical"]["path"] = str(copied)
        (copied / "historical/statistics.parquet").write_bytes(b"corrupt")
    else:
        profile = settings if change == "primary_model" else settings["metadata_sensitivity"]
        copied = tmp_path / "model"
        shutil.copytree(ROOT / profile["model"]["path"], copied)
        profile["model"]["path"] = str(copied)
        (copied / "models/ranker.txt").write_bytes(b"corrupt")
    path.write_bytes(canonical_json_bytes(settings))
    original = replay.read_preparation_report

    def read(source, *args, **kwargs):
        assert source != roots["source"], "Нельзя читать D с подменённой схемой"
        return original(source, *args, **kwargs)

    monkeypatch.setattr(replay, "read_preparation_report", read)
    monkeypatch.setattr("coveo_recsys.data.prepared.read_report", read)
    with pytest.raises(ValueError):
        if consumer == "evaluate":
            replay.run_replay(ROOT, path, tmp_path / "result", "bad", period="D")
        else:
            from coveo_recsys.cli.admission import require_evaluation_preparation
            from coveo_recsys.contracts.preparation import PreparationPlan
            from coveo_recsys.contracts.serialization import from_json

            training = json.loads(
                (roots["training_source"] / "reports/preparation.json").read_bytes()
            )
            plan = from_json(roots["plan"].read_bytes(), PreparationPlan)
            require_evaluation_preparation(
                ROOT, path, plan, profile_sha256=training["inputs"]["profile"]
            )


def test_final_gate_binds_runtime_and_both_models_without_touching_d(
    final_template, tmp_path, monkeypatch
):
    from coveo_recsys.cli import admission

    settings, _ = deepcopy(final_template)
    settings["preparation"] = {"path": "D-must-not-be-read", "report_sha256": "0" * 64}
    store = LocalArtifactStore(tmp_path / "evidence")
    manifest, references = fixed_manifest(store)
    intervals = settings["evaluation_plan"]["intervals"]
    intervals["half_width_fraction"]["GCS+Rk"] = 0.01
    intervals["dependence"]["GCS+Rk"] = {"supported": True, "assumptions": ["synthetic fixture"]}
    manifest["inference"].update(
        calendar_cell_seconds=intervals["grid_step_ms"] / 1000,
        block_cells=intervals["block_cells"],
        sensitivity_block_cells=intervals["sensitivity_block_cells"],
    )
    manifest["inference"]["dependence_assumptions"]["single_policy_hit"] = intervals["dependence"][
        "GCS+Rk"
    ]

    def publish(name, report):
        return store.publish_bytes(
            name,
            canonical_json_bytes(report),
            code_sha256="0" * 64,
            run_id=manifest["run_id"],
            scope="synthetic",
            config_id="gcs_rk_v2_7",
        ).sha256

    models = []
    for identifier, profile in (
        ("gcs_rk_v2_7", settings),
        (NO_METADATA_CONFIG_ID, settings["metadata_sensitivity"]),
    ):
        summary = json.loads((ROOT / profile["model"]["path"] / "reports/ranker.json").read_bytes())
        models.append(
            dict(
                config_id=identifier,
                model_sha256=summary["model_sha256"],
                feature_schema_sha256=sha256_file(ROOT / profile["feature_schema"]),
                actual_tree_count=summary["actual_tree_count"],
                training_status="passed",
            )
        )
    manifest["artifacts"]["trained_models_sha256"] = publish(
        "models.json", {"configurations": models}
    )
    references["before_D"] = publish(
        "execution.json", {"execution": admission.final_execution_binding(ROOT, settings)}
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    settings["admission"] = dict(
        manifest=str(path),
        manifest_sha256=sha256_file(path),
        run_dir=str(store.root),
        input_paths={},
        plan_references=references,
    )
    calls = []

    def validator(_manifest, stage, _store, **kwargs):
        calls.append((stage, kwargs["operation"], kwargs["plan_references"]))
        return SimpleNamespace(gate_passed=True)

    # Сам валидатор проверяется test_fixed_single_policy... без подмены.
    # Здесь проверяются дополнительные привязки команды к его результату.
    monkeypatch.setattr(admission, "validate_manifest", validator)
    assert (
        admission.require_final_admission(ROOT, settings)
        == settings["admission"]["manifest_sha256"]
    )
    assert calls == [("pre_D", "evaluate", references)]
    changed = deepcopy(settings)
    changed["limits"]["max_requests"] += 1
    with pytest.raises(ValueError, match="before_D"):
        admission.require_final_admission(ROOT, changed)
    changed = deepcopy(settings)
    changed["metadata_sensitivity"]["feature_schema"] = settings["feature_schema"]
    with pytest.raises(ValueError, match="before_D"):
        admission.require_final_admission(ROOT, changed)
    monkeypatch.setattr(
        admission,
        "validate_manifest",
        lambda *a, **k: SimpleNamespace(gate_passed=False, issues=[]),
    )
    with pytest.raises(ValueError, match="pre_D"):
        admission.require_final_admission(ROOT, settings)


@pytest.mark.parametrize("target_mode", ["changed", "unknown"])
def test_final_future_events_and_targets_do_not_change_candidates_or_features(
    final_settings, tmp_path, monkeypatch, target_mode
):
    from coveo_recsys.cli import replay
    from coveo_recsys.contracts import OutcomeStatus
    from coveo_recsys.features import FeatureEngine

    path, roots = final_settings
    captures = []
    original = FeatureEngine.build

    def capture(self, *args, **kwargs):
        batch = original(self, *args, **kwargs)
        captures.append(batch)
        return batch

    monkeypatch.setattr(FeatureEngine, "build", capture)
    first = tmp_path / "first"
    replay.run_replay(ROOT, path, first, "same", period="D")
    before = list(captures)
    captures.clear()
    source = tmp_path / "changed"
    shutil.copytree(roots["source"], source)

    def future(rows):
        # У каждого SID первый блок сохранён, заменяются только последующие покупки.
        first_time = {}
        output = []
        for block in rows:
            first_time.setdefault(block.session_id, block.event_time_ms)
            output.append(
                replace(
                    block, events=tuple(replace(event, sku="future-only") for event in block.events)
                )
                if block.event_time_ms > first_time[block.session_id]
                else block
            )
        return output

    rewrite_records(source, "canonical/D.parquet", CanonicalBlock, future)
    target_path = source / "datasets/D/targets.parquet"
    replacement = tmp_path / "targets.parquet"
    with TableWriter(replacement, TargetRecord) as writer:
        for sid, rows in iter_sessions(target_path, TargetRecord):
            writer.append(
                sid,
                [
                    replace(
                        row,
                        purchases=(),
                        status=OutcomeStatus.UNKNOWN,
                        reason="no_registered_purchase",
                    )
                    if target_mode == "unknown"
                    else replace(row, purchases=("future-only",))
                    if row.purchases
                    else row
                    for row in rows
                ],
            )
    replacement.replace(target_path)
    report_path = source / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    report["tables"]["datasets/D/targets.parquet"] = table_info(
        target_path, TargetRecord, "D", "targets"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    settings = json.loads(path.read_bytes())
    settings["preparation"] = {
        "path": str(source),
        "report_sha256": sha256_file(source / "reports/preparation.json"),
    }
    path.write_bytes(canonical_json_bytes(settings))
    second = tmp_path / "second"
    result = replay.run_replay(ROOT, path, second, "same", period="D")
    if target_mode == "unknown":
        assert all(
            value["counts"]["N"] == 0 and value["metrics"]["Hit@20"]["value"] is None
            for value in result["policies"].values()
        )
        assert all(
            value["metric"]["status"] == "insufficient" for value in result["intervals"].values()
        )
    assert captures == before

    def deterministic_answers(root):
        return [
            (sid, [(row.policy, replace(row.result, elapsed_ms=0)) for row in rows])
            for sid, rows in iter_sessions(root / "replay/answers.parquet", ReplayRecord)
        ]

    assert deterministic_answers(first) == deterministic_answers(second)
