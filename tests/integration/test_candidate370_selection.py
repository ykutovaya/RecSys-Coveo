"""Малый B → две модели → общий C_select. Реальные периоды не читаются."""

import json
import shutil
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.evaluation import ReplayRecord
from coveo_recsys.contracts.retrieval import METADATA370_CONFIG_ID, METADATA370_QUOTAS
from coveo_recsys.data.prepared import iter_sessions
from tests.integration.test_e5_replay_pipeline import ROOT, make_replay_template
from tests.unit.test_e2_input import prepared_source  # noqa: F401


@pytest.fixture(scope="module")
def selection_template(prepared_source, tmp_path_factory):  # noqa: F811
    from coveo_recsys.cli.features import run_features
    from coveo_recsys.cli.training import run_training

    root = tmp_path_factory.mktemp("candidate370")
    settings, roots = make_replay_template(prepared_source, root, "C_select")
    schema = "configs/feature_schema_metadata370.json"
    retrieval = dict(quotas=list(METADATA370_QUOTAS), max_candidates=370)
    features, model = root / "features370", root / "model370"
    path = root / "variant.json"
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
    run_features(ROOT, path, features, "370", period="B")
    binding = dict(
        path=str(features), report_sha256=sha256_file(features / "reports/features.json")
    )
    train = json.loads((root / "training.json").read_bytes())
    train.update(features=binding, feature_schema=schema)
    train["limits"]["max_rows"] = 160 * 370
    path.write_bytes(canonical_json_bytes(train))
    run_training(ROOT, path, model, "370")
    settings["candidate_variant"] = dict(
        model=dict(path=str(model), report_sha256=sha256_file(model / "reports/ranker.json")),
        training_features=binding,
        feature_schema=schema,
        recipe=settings["recipe"],
        retrieval=retrieval,
    )
    policies = ["GCS+Rk", METADATA370_CONFIG_ID]
    estimates = [*policies, "GCS+Rk-" + METADATA370_CONFIG_ID]
    settings["evaluation_plan"]["policies"] = policies
    settings["evaluation_plan"]["intervals"].update(
        dependence={name: dict(supported=None, assumptions=[]) for name in estimates},
        half_width_fraction=dict.fromkeys(estimates),
    )
    return settings, roots


@pytest.mark.parametrize("diagnostic_delay", [0.0, 11.0])
def test_select_shared_denominator_own_features_and_reuse(
    selection_template, tmp_path, monkeypatch, diagnostic_delay
):
    from coveo_recsys.cli import admission, replay
    from coveo_recsys.data.access import DatasetAccess
    from coveo_recsys.features import FeatureEngine

    settings, roots = deepcopy(selection_template)
    path, output = tmp_path / "settings.json", tmp_path / "result"
    path.write_bytes(canonical_json_bytes(settings))
    # Только вычислительный тест. Реальный валидатор проверяется отдельно.
    monkeypatch.setattr(admission, "require_selection_admission", lambda *a, **k: "a" * 64)
    original, original_read = FeatureEngine.build, DatasetAccess.read
    profiles = set()
    clock = [0.0]
    originals = {}
    original_recommend = replay.recommend
    original_compare = replay.compare_multiplicities
    original_json = replay.canonical_json_bytes
    original_append = replay.TableWriter.append

    def recommend(*args, **kwargs):
        answer = original_recommend(*args, **kwargs)
        clock[0] += 1.0
        originals[answer.trace_ref] = answer
        return answer

    def compare(*args, **kwargs):
        result = original_compare(*args, **kwargs)
        clock[0] += diagnostic_delay
        return result

    def encode(value):
        if isinstance(value, dict) and {"source_lists", "feature_rows"} <= value.keys():
            clock[0] += 5.0
        return original_json(value)

    def append(self, *args, **kwargs):
        result = original_append(self, *args, **kwargs)
        clock[0] += 7.0
        return result

    monkeypatch.setattr(
        replay, "time", SimpleNamespace(perf_counter=lambda: clock[0], monotonic=time.monotonic)
    )
    monkeypatch.setattr(replay, "recommend", recommend)
    monkeypatch.setattr(replay, "compare_multiplicities", compare)
    monkeypatch.setattr(replay, "canonical_json_bytes", encode)
    monkeypatch.setattr(replay.TableWriter, "append", append)

    def build(self, context, *a, **k):
        profiles.add(self.index.config_id)
        assert all(block.available_time_ms <= context.t_q for block in context.prefix.blocks)
        return original(self, context, *a, **k)

    def read(self, path, kind, *, purpose):
        if purpose == "labels":
            assert len(list(iter_sessions(output / "replay/answers.parquet", ReplayRecord))) == 3
        return original_read(self, path, kind, purpose=purpose)

    monkeypatch.setattr(FeatureEngine, "build", build)
    monkeypatch.setattr(DatasetAccess, "read", read)
    result = replay.run_replay(ROOT, path, output, "select-test", period="C_select")
    assert profiles == {"gcs_rk_v2_7", METADATA370_CONFIG_ID}
    assert result["requests"] == 3
    answers = dict(iter_sessions(output / "replay/answers.parquet", ReplayRecord))
    for rows in answers.values():
        for row in rows:
            expected = originals[row.result.trace_ref]
            assert replace(row.result, elapsed_ms=expected.elapsed_ms) == expected
    for policy, data in result["policies"].items():
        hits = sum(
            bool(set(row.result.delivered_skus) & ({"C"} if sid == "e0" else {"outside"}))
            for sid, rows in answers.items()
            if sid != "e1"
            for row in rows
            if row.policy == policy
        )
        assert data["counts"]["N"] == 2
        assert data["metrics"]["Hit@20"]["value"] == hits / 2
        assert result["full_recommendation_timing"][policy]["requests"] == 3
        timing = result["full_recommendation_timing"][policy]
        assert timing["recommend_seconds"] == 3.0
        assert timing["calculation_seconds"] == 3 * (
            1.0 + 5.0 + (diagnostic_delay if policy == "GCS+Rk" else 0.0)
        )
        assert result["intervals"][policy]["metric"]["status"] == "insufficient"
    baseline, variant = [
        result["policies"][p]["metrics"]["Hit@20"]["value"] for p in result["identity"]["policies"]
    ]
    assert result["selection"]["selected"] == (
        METADATA370_CONFIG_ID if variant > baseline else "gcs_rk_v2_7"
    )
    monkeypatch.setattr(replay, "recommend", lambda *a, **k: pytest.fail("Повторный расчёт"))
    assert replay.run_replay(ROOT, path, output, "select-test", period="C_select") == result
    assert replay.read_report(output) == result


def test_select_denied_before_any_prepared_io(tmp_path, monkeypatch):
    from coveo_recsys.cli import replay

    settings = tmp_path / "settings.json"
    settings.write_bytes(canonical_json_bytes({"preparation": {"path": "never-open"}}))
    monkeypatch.setattr(
        replay, "read_preparation_report", lambda *a, **k: pytest.fail("C_select прочитан")
    )
    with pytest.raises(InputValidationError, match="pre_selection|допуск"):
        replay.run_replay(ROOT, settings, tmp_path / "result", "denied", period="C_select")


def test_legacy_comparison_warns_of_missing_recommend_time_without_changing_files(
    selection_template, tmp_path, monkeypatch
):
    from coveo_recsys.cli import admission, replay
    from coveo_recsys.evaluation.reporting import render_report

    settings, _ = selection_template
    path, output = tmp_path / "settings.json", tmp_path / "legacy"
    path.write_bytes(canonical_json_bytes(settings))
    monkeypatch.setattr(admission, "require_selection_admission", lambda *a, **k: "a" * 64)
    summary = replay.run_replay(ROOT, path, output, "legacy", period="C_select")
    for policy in summary["policies"]:
        summary["full_recommendation_timing"][policy].pop("recommend_seconds", None)
    (output / replay.REPORT).write_bytes(canonical_json_bytes(summary))
    (output / replay.MARKDOWN).write_text(render_report(summary), encoding="utf-8", newline="")
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.warns(RuntimeWarning, match="recommend_seconds.*не измерено"):
        assert replay.read_report(output) == summary
    assert before == {path: (sha256_file(path), path.stat().st_mtime_ns) for path in before}


def test_feature_pilot_is_target_independent_and_not_training_input(selection_template, tmp_path):
    from coveo_recsys.cli.features import run_features
    from coveo_recsys.cli.training import run_training
    from coveo_recsys.contracts.preparation import PreparedRequest
    from coveo_recsys.contracts.records import FeatureBatch
    from coveo_recsys.data.periods import session_key

    settings, roots = selection_template
    source = tmp_path / "prepared"
    shutil.copytree(roots["training_source"], source)
    prepared = json.loads((source / "reports/preparation.json").read_bytes())
    prepared["selection"]["B"] = [
        sid for sid, _ in iter_sessions(source / "datasets/B/prefixes.parquet", PreparedRequest)
    ]
    (source / "reports/preparation.json").write_bytes(canonical_json_bytes(prepared))
    profile = settings["candidate_variant"]
    config = dict(
        preparation=dict(
            path=str(source), report_sha256=sha256_file(source / "reports/preparation.json")
        ),
        historical=settings["historical"],
        feature_schema=profile["feature_schema"],
        retrieval=profile["retrieval"],
        limits={**settings["limits"], "max_requests": 160},
        pilot_requests=3,
    )
    path, output = tmp_path / "pilot.json", tmp_path / "pilot"
    path.write_bytes(canonical_json_bytes(config))
    pilot = run_features(ROOT, path, output, "pilot", period="B")
    assert pilot["status"] == "pilot_completed" and pilot["requests"] == 3
    assert 0 < pilot["calculation_seconds_including_write"] <= pilot["elapsed_seconds"]
    chosen = sorted(prepared["selection"]["B"], key=lambda sid: session_key(sid, 17))[:3]
    assert {
        sid for sid, _ in iter_sessions(output / "features/values.parquet", FeatureBatch)
    } == set(chosen)
    assert run_features(ROOT, path, output, "pilot", period="B") == pilot
    train = json.loads((roots["model"].parent / "training.json").read_bytes())
    train.update(
        features=dict(
            path=str(output), report_sha256=sha256_file(output / "reports/features.json")
        ),
        feature_schema=profile["feature_schema"],
    )
    path.write_bytes(canonical_json_bytes(train))
    with pytest.raises(InputValidationError):
        run_training(ROOT, path, tmp_path / "must-not-train", "denied")


@pytest.mark.parametrize("field", ["model", "feature_schema", "training_features", "retrieval"])
def test_variant_cannot_borrow_baseline_bindings(selection_template, tmp_path, monkeypatch, field):
    from coveo_recsys.cli import admission, replay

    settings, roots = deepcopy(selection_template)
    settings["candidate_variant"][field] = settings[field]
    path = tmp_path / "changed.json"
    path.write_bytes(canonical_json_bytes(settings))
    monkeypatch.setattr(admission, "require_selection_admission", lambda *a, **k: "a" * 64)
    read = replay.read_preparation_report

    def protected(source, *a, **k):
        assert source != roots["source"], "C_select открыт до проверки моделей"
        return read(source, *a, **k)

    monkeypatch.setattr(replay, "read_preparation_report", protected)
    with pytest.raises(InputValidationError):
        replay.run_replay(ROOT, path, tmp_path / "result", "changed", period="C_select")
