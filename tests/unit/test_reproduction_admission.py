"""Сборка нового допуска: свидетельства связаны с файлами, не с прежними runs."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts.historical import HistoricalConfig
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.serialization import from_json, to_json

ROOT = Path(__file__).resolve().parents[2]


def preparation_case(tmp_path):
    report = {
        "run_id": "new-b",
        "source_sha256": "a" * 64,
        "scope": "actual",
        "status": "completed",
        "inputs": {"plan": "b" * 64, "profile": "c" * 64},
        "tables": {"canonical/B.parquet": {"sha256": "d" * 64}},
        "selection": {"A_G": [], "B": ["s1", "s2"]},
        "request_count": 2,
        "telemetry_complete": None,
    }
    payload = canonical_json_bytes(report)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/preparation.json").write_bytes(payload)
    checks = {
        "check_report_format_version": "preparation_check_v1",
        "status": "passed",
        "scope": "actual",
        "current_code_sha256": "a" * 64,
        "period": "B",
        "bindings": {
            "run_id": "new-b",
            "tables": report["tables"],
            "inputs": report["inputs"],
            "preparation_code_sha256": "a" * 64,
            "report_sha256": sha256_bytes(payload),
            "plan_sha256": "b" * 64,
        },
        "schema_valid": True,
        "whole_sessions": True,
        "published_rows_complete": True,
        "calendar_admission_checked": True,
        "registered_targets_checked": True,
        "preparation_plan_sha256": "b" * 64,
        "preparation_report_sha256": sha256_bytes(payload),
        "session_count": 2,
        "request_count": 2,
        "overlapping_sessions": 0,
        "ag_intersection_count": 0,
        "telemetry_completeness": "unknown",
    }
    return report, checks


@pytest.mark.parametrize(
    "change",
    [
        None,
        "status",
        "current_code_sha256",
        "published_rows_complete",
        "bindings",
        "preparation_report_sha256",
        "session_count",
        "telemetry_completeness",
    ],
)
def test_data_check_requires_current_complete_bound_evidence(tmp_path, change):
    from coveo_recsys.verification.admission import checked_preparation

    report, checks = preparation_case(tmp_path)
    if change == "bindings":
        checks[change] = {**checks[change], "tables": {}}
    elif change is not None:
        checks[change] = False
    (tmp_path / "reports/data-check.json").write_bytes(canonical_json_bytes(checks))
    if change is None:
        assert checked_preparation(tmp_path, report, "a" * 64, "B") == checks
    else:
        with pytest.raises(ValueError):
            checked_preparation(tmp_path, report, "a" * 64, "B")


def test_published_manifest_is_immutable_and_equal_repeat_is_allowed(tmp_path):
    from coveo_recsys.verification.admission import _write_manifest

    path = tmp_path / "manifest.json"
    original = {"run_id": "current"}
    _write_manifest(path, original)
    _write_manifest(path, deepcopy(original))
    with pytest.raises(ValueError, match="друг"):
        _write_manifest(path, {"run_id": "replacement"})
    assert json.loads(path.read_bytes()) == original


@pytest.mark.parametrize("stage", ["pre_D", "post_D", "unknown"])
def test_builder_does_not_open_other_stages(tmp_path, stage):
    from coveo_recsys.verification.admission import assemble_admission

    with pytest.raises(ValueError, match="стади"):
        assemble_admission(
            tmp_path,
            tmp_path / "result",
            run_id="new",
            input_dir=tmp_path / "raw",
            prepared_dir=tmp_path / "b",
            ag_dir=tmp_path / "ag",
            historical_dir=tmp_path / "history",
            test_run=tmp_path / "tests",
            stage=stage,
        )


@pytest.fixture
def admission_case(tmp_path, monkeypatch):
    """Вымышленные входы лишь для проверки сборщика; настоящий валидатор не подменён."""
    from coveo_recsys.config import load_registry
    from coveo_recsys.data.io import FILES
    from coveo_recsys.verification import admission
    from coveo_recsys.verification.test_evidence import PARTS, TestEvidence

    inventory = {"test_fixture_only": "a" * 64}
    code = sha256_bytes(canonical_json_bytes(inventory))
    for name in (
        "configs/ranker_recipe.json",
        "configs/feature_schema.json",
        "schemas/manifest.schema.json",
        "uv.lock",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(admission, "assert_source_binding", lambda root: None)
    monkeypatch.setattr(admission, "source_inventory", lambda root: inventory)
    from coveo_recsys.cli import admission as consumer

    monkeypatch.setattr(consumer, "assert_source_binding", lambda root: None)
    monkeypatch.setattr(consumer, "source_sha256", lambda root: code)
    registry = load_registry()["implemented_checks"]
    paths = {path for parts in PARTS.values() for part in parts for path in registry[part]["paths"]}
    cases = {(path.removesuffix(".py").replace("/", "."), "fixture"): "passed" for path in paths}
    property_counts = {
        f"{path}::fixture": 1000 for path in paths if path.startswith("tests/properties/")
    }
    junit = b"fixture only, not pytest evidence"
    tests = tmp_path / "test-run"
    tests.mkdir()
    (tests / "pytest.xml").write_bytes(junit)
    (tests / "source-inventory.json").write_bytes(canonical_json_bytes(inventory))
    result = dict(
        status="passed",
        exit_code=0,
        source_sha256=code,
        junit_sha256=sha256_bytes(junit),
        unsuccessful_cases=[],
    )
    (tests / "result.json").write_bytes(canonical_json_bytes(result))
    monkeypatch.setattr(
        admission, "read_test_evidence", lambda *args: TestEvidence(cases, property_counts, result)
    )
    profile = json.loads((ROOT / "configs/coveo_supply_v1.json").read_bytes())
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    inputs, sizes = {}, {}
    for kind, name in FILES.items():
        content = (name + " artificial test data").encode()
        (input_dir / name).write_bytes(content)
        inputs[kind] = sha256_bytes(content)
        sizes[name] = len(content)
        profile["files"][name].update(
            expected_sha256=inputs[kind], expected_size_bytes=len(content)
        )
    profile_bytes = canonical_json_bytes(profile)
    inputs["profile"] = sha256_bytes(profile_bytes)
    base = from_json((ROOT / "configs/coveo_preparation.json").read_bytes(), PreparationPlan)
    reports, checks = {}, {}
    for period in ("A_G", "B"):
        path = tmp_path / period
        path.mkdir()
        plan = replace(
            base,
            periods=tuple(
                replace(p, sample_size=2 if p.name == period else 0) for p in base.periods
            ),
        )
        plan_bytes = to_json(plan)
        report, check = preparation_case(path)
        report.update(
            run_id="fixture-" + period,
            source_sha256=code,
            plan_json=plan_bytes.decode(),
            profile_json=profile_bytes.decode(),
            evaluation_protocol="observed_purchases_v1",
            inputs={**inputs, "plan": sha256_bytes(plan_bytes)},
            selection={
                p.name: [period + "1", period + "2"] if p.name == period else []
                for p in base.periods
            },
            sampled={
                "rule": "lowest_seeded_hash_after_calendar_admission",
                "normative_sample": True,
                "period_counts": {period: {"selected": 2}},
            },
        )
        digest = sha256_bytes(canonical_json_bytes(report))
        check.update(
            current_code_sha256=code,
            period=period,
            preparation_report_sha256=digest,
            preparation_plan_sha256=report["inputs"]["plan"],
            input_size_bytes=sizes,
            bindings=dict(
                run_id=report["run_id"],
                tables=report["tables"],
                inputs=report["inputs"],
                preparation_code_sha256=code,
                report_sha256=digest,
                plan_sha256=report["inputs"]["plan"],
            ),
        )
        (path / "reports/preparation.json").write_bytes(canonical_json_bytes(report))
        reports[path], checks[path] = report, check
    historical_dir = tmp_path / "historical"
    store = LocalArtifactStore(historical_dir)
    head = store.publish_bytes(
        "historical/package.json",
        b"fixture historical package",
        inputs=inputs,
        configuration=asdict(HistoricalConfig(snapshot_time_ms=base.periods[0].end_ms)),
        code_sha256=code,
        run_id="fixture-ag",
        scope="actual",
        config_id="historical",
    )
    ag_digest = checks[tmp_path / "A_G"]["preparation_report_sha256"]
    checks[tmp_path / "B"]["bindings"].update(
        ag_report_sha256=ag_digest, historical_package_sha256=head.sha256
    )
    for path, check in checks.items():
        (path / "reports/data-check.json").write_bytes(canonical_json_bytes(check))
    monkeypatch.setattr(admission, "read_report", lambda path: reports[path])
    monkeypatch.setattr(admission, "verify_tables", lambda *args: None)
    monkeypatch.setattr(
        admission,
        "load_historical",
        lambda store: SimpleNamespace(input_hashes={"preparation_report": ag_digest}),
    )
    kwargs = dict(
        run_id="fixture-admission",
        input_dir=input_dir,
        prepared_dir=tmp_path / "B",
        ag_dir=tmp_path / "A_G",
        historical_dir=historical_dir,
        test_run=tests,
    )
    return tmp_path, kwargs


@pytest.mark.parametrize("stage", ["preflight", "post_training"])
def test_builder_passes_real_validator_without_previous_runs(admission_case, monkeypatch, stage):
    from coveo_recsys.contracts.retrieval import RetrievalConfig
    from coveo_recsys.verification import admission
    from coveo_recsys.verification.admission import assemble_admission

    root, kwargs = admission_case
    if stage == "post_training":
        report = json.loads((kwargs["prepared_dir"] / "reports/preparation.json").read_bytes())
        plan = from_json(report["plan_json"].encode(), PreparationPlan)
        plan = replace(
            plan,
            periods=tuple(
                replace(p, sample_size=2 if p.name == "C_smoke" else 0) for p in plan.periods
            ),
        )
        plan_path = root / "c-smoke-plan.json"
        plan_path.write_bytes(to_json(plan))
        features_dir, model_dir = root / "features", root / "model"
        (features_dir / "reports").mkdir(parents=True)
        (model_dir / "models").mkdir(parents=True)
        schema = (root / "configs/feature_schema.json").read_bytes()
        recipe = (root / "configs/ranker_recipe.json").read_bytes()
        head = LocalArtifactStore(kwargs["historical_dir"]).describe("historical/package.json")
        shared = dict(
            scope="actual",
            period="B",
            source_sha256=report["source_sha256"],
            preparation_sha256=sha256_bytes(canonical_json_bytes(report)),
            historical_sha256=head.sha256,
            feature_schema_file_sha256=sha256_bytes(schema),
        )
        features = dict(
            status="completed",
            identity={**shared, "configuration": asdict(RetrievalConfig())},
            requests=2,
            feature_schema=json.loads(schema),
        )
        feature_bytes = canonical_json_bytes(features)
        (features_dir / "reports/features.json").write_bytes(feature_bytes)
        model_bytes = b"fixture tree only; load_ranker is checked separately"
        (model_dir / "models/ranker.txt").write_bytes(model_bytes)
        model = dict(
            identity={
                **shared,
                "features_sha256": sha256_bytes(feature_bytes),
                "recipe_sha256": sha256_bytes(recipe),
            },
            recipe=json.loads(recipe),
            actual_tree_count=1,
            model_sha256=sha256_bytes(model_bytes),
        )
        monkeypatch.setattr(admission, "load_ranker", lambda root: SimpleNamespace(summary=model))
        kwargs.update(
            features_dir=features_dir, model_dir=model_dir, evaluation_plan_path=plan_path
        )
    kwargs["stage"] = stage
    binding = assemble_admission(root, root / "new-admission", **kwargs)
    manifest = json.loads((root / binding["manifest"]).read_bytes())
    assert manifest["verification"]["data_status"] == "passed"
    assert manifest["dataset"]["session_sampling"]["session_count_by_period"]["period_B"] == 2
    assert (manifest["artifacts"]["trained_models_sha256"] is None) == (stage == "preflight")
    if stage == "post_training":
        store = LocalArtifactStore(root / binding["run_dir"])
        report = json.loads(
            store.read_by_sha256(manifest["verification"]["results_by_stage"][stage])
        )
        assert report["operation"] == "evaluate"
        assert next(row for row in report["tests"] if row["test_id"] == "B12")["status"] == "passed"
    assert set(binding) == {"manifest", "manifest_sha256", "run_dir", "input_paths"}
    from coveo_recsys.cli.admission import require_training_admission

    assert (
        require_training_admission(
            root,
            binding,
            prepared=json.loads((kwargs["prepared_dir"] / "reports/preparation.json").read_bytes()),
            historical=LocalArtifactStore(kwargs["historical_dir"]).describe(
                "historical/package.json"
            ),
            feature_schema_bytes=(root / "configs/feature_schema.json").read_bytes(),
            recipe=json.loads((root / "configs/ranker_recipe.json").read_bytes()),
            retrieval=asdict(RetrievalConfig()),
            stage=stage,
            operation="evaluate" if stage == "post_training" else None,
        )
        == binding["manifest_sha256"]
    )


def test_builder_rejects_raw_change_before_publishing_manifest(admission_case):
    from coveo_recsys.verification.admission import assemble_admission

    root, kwargs = admission_case
    (kwargs["input_dir"] / "search_train.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Допуск preflight"):
        assemble_admission(root, root / "new-admission", **kwargs)
    assert not (root / "new-admission/manifest.json").exists()


@pytest.mark.parametrize(
    "change", [{"status": "failed"}, {"error": "interrupted"}, {"error": None}, {"status": None}]
)
def test_builder_rejects_unsuccessful_test_command_despite_valid_junit(admission_case, change):
    from coveo_recsys.verification.admission import assemble_admission

    root, kwargs = admission_case
    path = kwargs["test_run"] / "result.json"
    result = json.loads(path.read_bytes())
    result.update(change)
    path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="pytest"):
        assemble_admission(root, root / "new-admission", **kwargs)
    assert not (root / "new-admission/manifest.json").exists()
