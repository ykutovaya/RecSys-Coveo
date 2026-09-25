"""B13: отрицательные проверки политик и происхождения свидетельств."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.config import validate_manifest
from tests.reference.manifest_cases import make_manifest_case


def publish(store, manifest, value):
    data = canonical_json_bytes(value)
    return store.publish_bytes(
        "changed/" + sha256_bytes(data) + ".json",
        data,
        code_sha256=manifest["reproducibility"]["code_commit_or_archive_sha256"],
        run_id=manifest["run_id"],
        scope=manifest["scope"],
        config_id=manifest["base_model"]["config_id"],
    ).sha256


@pytest.mark.parametrize(
    "stage,former", [("pre_D", "preflight"), ("pre_D", "post_training"), ("pre_AB", "pre_D")]
)
def test_old_report_cannot_satisfy_repeat(tmp_path: Path, stage, former) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    records = manifest["verification"]["results_by_stage"]
    records[stage] = records[former]
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.code == "report_stage" for issue in result.issues)


@pytest.mark.parametrize(
    "path,value",
    [
        (("calibration", "fit_period"), "period_D"),
        (("inference", "alpha"), 0.99),
        (("inference", "minimum_effective_blocks"), 0),
        (
            ("ranker", "resolved_parameters"),
            {name: {"x": 1} for name in ("model", "dataset", "training", "prediction")},
        ),
        (
            ("calibration", "resolved_parameters"),
            {"optimizer": None, "initial": None, "max_iterations": None},
        ),
        (("online_experiment", "stopping_rule"), "allow_efficacy_peeking"),
    ],
)
def test_malformed_or_weakened_recipe_cannot_pass(tmp_path: Path, path, value) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    manifest[path[0]][path[1]] = value
    assert not validate_manifest(manifest, "preflight", store, plan_references=plans).gate_passed


@pytest.mark.parametrize(
    "stage,status", [("preflight", "data_status"), ("post_training", "math_status")]
)
def test_summary_failure_cannot_be_ignored(tmp_path: Path, stage, status) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, stage)
    manifest["verification"][status] = "failed"
    result = validate_manifest(manifest, stage, store, plan_references=plans)
    assert not result.gate_passed
    assert any(issue.code == "required_status" for issue in result.issues)


def test_paired_preflight_does_not_require_future_winner(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    manifest["inference"].update(primary_estimate_type="paired", half_width_per_1000=10)
    manifest["inference"]["dependence_assumptions"]["paired_por"] = {"justification": "synthetic"}
    assert manifest["selection"]["primary_comparison"] is None
    result = validate_manifest(manifest, "preflight", store, plan_references=plans)
    assert result.gate_passed, result.issues


def test_registered_untrained_config_cannot_enter_selection(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "pre_selection")
    manifest["selection"]["registered_configurations"].append(
        dict(config_id="untrained", configuration_sha256="1" * 64, parameter_count=1)
    )
    plan = json.loads(store.read_by_sha256(manifest["evidence"]["selection_plan"]))
    plan["configurations"] = manifest["selection"]["registered_configurations"]
    manifest["evidence"]["selection_plan"] = publish(store, manifest, plan)
    result = validate_manifest(manifest, "pre_selection", store, plan_references=plans)
    assert any(issue.code == "untrained_config" for issue in result.issues)


def test_loosened_tolerances_cannot_pass(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_training")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["post_training"])
    )
    row = next(row for row in report["tests"] if row["test_id"] == "B05")
    row["derivative_tolerances"]["callback_fp64"] = [1e20, 1e20]
    manifest["verification"]["results_by_stage"]["post_training"] = publish(store, manifest, report)
    result = validate_manifest(manifest, "post_training", store, plan_references=plans)
    assert any(issue.code == "derivative_tolerances" for issue in result.issues)


def test_validator_hash_is_not_arbitrary_blob(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "preflight")
    manifest["reproducibility"]["manifest_validator_sha256"] = publish(
        store, manifest, {"unrelated": True}
    )
    result = validate_manifest(manifest, "preflight", store, plan_references=plans)
    assert any(issue.code == "validator_binding" for issue in result.issues)


def test_external_inputs_are_stream_checked_and_never_silently_replaced(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    manifest, plans = make_manifest_case(store, "preflight")
    paths = {}
    for item in manifest["dataset"]["files"]:
        path = tmp_path / item["name"]
        path.write_bytes(store.read_by_sha256(item["sha256"]))
        paths[item["name"]] = path
    assert validate_manifest(
        manifest, "preflight", store, plan_references=plans, input_paths=paths
    ).gate_passed
    # The valid copy inside store must not hide a missing or corrupted external source.
    missing = dict(paths)
    missing.pop("search_train.csv")
    assert not validate_manifest(manifest, "preflight", store, input_paths=missing).gate_passed
    paths["search_train.csv"].write_bytes(b"changed")
    assert not validate_manifest(manifest, "preflight", store, input_paths=paths).gate_passed


def test_honest_failed_exact_check_can_be_recorded_without_admission(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_load")
    report = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["post_load"])
    )
    row = next(x for x in report["tests"] if x["test_id"] == "B14")
    row.update(
        status="failed",
        reason="synthetic exact comparison failed",
        exact_fields_passed=False,
        counts=dict(collected=1, passed=0, failed=1, skipped=0, errors=0),
    )
    manifest["verification"]["results_by_stage"]["post_load"] = publish(store, manifest, report)
    manifest["verification"]["serve_status"] = "failed"
    result = validate_manifest(manifest, "post_load", store, plan_references=plans)
    assert result.semantic_valid and result.results_complete and not result.gate_passed, (
        result.issues
    )


def test_complete_zero_data_report_is_insufficient_and_recordable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_D")
    report = json.loads(store.read_by_sha256(manifest["evidence"]["final_evaluation"]))
    report["metrics"] = {"N": 0, "POR": None}
    report["intervals"] = {"POR": None}
    report["interval_details"].update(valid=False, reason="no observations")
    report["information"].update(
        nonempty_blocks=0, category_counts={"0": 0, "1": 0}, category_blocks={"0": 0, "1": 0}
    )
    report["statuses"]["offline_status"] = "insufficient"
    report["reason"] = "no observations"
    manifest["verification"]["offline_status"] = "insufficient"
    digest = publish(store, manifest, report)
    manifest["evidence"]["final_evaluation"] = manifest["artifacts"][
        "final_evaluation_report_sha256"
    ] = digest
    result = validate_manifest(manifest, "post_D", store, plan_references=plans)
    assert result.semantic_valid and result.results_complete and not result.gate_passed, (
        result.issues
    )


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("model", "objective", "lambdarank"),
        ("model", "learning_rate", "fast"),
        ("dataset", "weight", [1]),
        ("prediction", "raw_score", False),
    ],
)
def test_new_configuration_name_does_not_disable_ranker_contract(
    tmp_path: Path, group, key, value
) -> None:
    from coveo_recsys.config.recipes import resolved_recipe_issues

    store = LocalArtifactStore(tmp_path)
    manifest, _ = make_manifest_case(store, "preflight")
    manifest["base_model"]["config_id"] = "registered_variant"
    manifest["ranker"]["resolved_parameters"][group][key] = value
    issues = resolved_recipe_issues(manifest)
    assert any(issue.path == f"ranker.resolved_parameters.{group}.{key}" for issue in issues)


def test_neural_zero_attention_correction_and_sid_zero_penalties_are_legal() -> None:
    from coveo_recsys.config import init_manifest
    from coveo_recsys.config.recipes import resolved_recipe_issues

    manifest = init_manifest("neural-bounds")
    manifest["neural_extension"]["enabled"] = True
    params = dict(
        d=64,
        J_I=2,
        H_P=2,
        d_P=50,
        B_P=100,
        R_P=20.5,
        c_G=0.1,
        eta_N=0,
        tau_P=1,
        tau_N=0.1,
        tau_C=0.1,
        tau_SN_seconds=300,
        maximum_search_gate=0.5,
        initial_search_gate=0.1,
        lambda_next=0,
        lambda_C=0,
        lambda_G=0,
        lambda_theta=0,
        p_S=0,
        delta_I=0.05,
        epsilon_N=1e-6,
        branch_dropout=0.2,
        relation_weights=[1 / 3] * 3,
    )
    manifest["neural_extension"]["resolved_configuration"] = {
        "parameters": params,
        "training": manifest["neural_extension"]["training_defaults"],
    }
    manifest["sid_extension"].update(
        enabled=True,
        resolved_configuration={
            "parameters": dict(
                levels=4,
                vocabulary=32,
                code_dimension=16,
                hidden_dimension=64,
                beam_width=32,
                codes_per_item=4,
                quota=50,
                lambda_R=1,
                lambda_X=0,
                lambda_B=0,
                lambda_2=0,
                tau_R=1,
            ),
            "training": manifest["sid_extension"]["training_defaults"],
        },
    )
    assert not any(
        issue.path.startswith(("neural", "sid")) for issue in resolved_recipe_issues(manifest)
    )
    params["delta_I"] = 1
    assert any(issue.path.startswith("neural") for issue in resolved_recipe_issues(manifest))


@pytest.mark.parametrize("status", ["passed", "not_run"])
def test_unfit_E_cannot_be_reported_as_successful_or_unfinished_D(
    tmp_path: Path, status: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_D")
    report = json.loads(store.read_by_sha256(manifest["evidence"]["final_evaluation"]))
    report["statuses"]["calibration_status"] = manifest["verification"]["calibration_status"] = (
        status
    )
    digest = publish(store, manifest, report)
    manifest["evidence"]["final_evaluation"] = manifest["artifacts"][
        "final_evaluation_report_sha256"
    ] = digest
    result = validate_manifest(manifest, "post_D", store, plan_references=plans)
    assert not result.gate_passed
    assert any(
        issue.code == ("calibration_unfitted" if status == "passed" else "final_calibration_status")
        for issue in result.issues
    )


@pytest.mark.parametrize("status", ["passed", "failed", "insufficient"])
def test_successful_E_allows_final_D_calibration_verdict_without_blocking_raw_scores(
    tmp_path: Path, status: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    manifest, plans = make_manifest_case(store, "post_D")
    fit = json.loads(store.read_by_sha256(manifest["evidence"]["calibration"]))
    fit.update(status="passed", reason=None)
    manifest["evidence"]["calibration"] = publish(store, manifest, fit)
    manifest["artifacts"]["calibrator_sha256"] = publish(
        store,
        manifest,
        {
            "kind": "calibrator",
            "run_id": manifest["run_id"],
            "scope": manifest["scope"],
            "config_id": manifest["base_model"]["config_id"],
            "evidence_scope": "run",
            "a": 0,
            "b": 0,
        },
    )
    for key in ("dependence_assumptions", "block_plan", "ece_regularity_justification"):
        manifest["calibration"][key] = {"justification": "synthetic B13"}
    plan = json.loads(store.read_by_sha256(plans["before_D"]))
    plan["inputs"].update(
        calibration=deepcopy(manifest["calibration"]),
        calibration_result_sha256=manifest["evidence"]["calibration"],
        calibrator_sha256=manifest["artifacts"]["calibrator_sha256"],
    )
    digest = publish(store, manifest, plan)
    plans["before_D"] = manifest["freeze"]["plan_references"]["before_D"] = manifest["evidence"][
        "evaluation_plan"
    ] = digest
    for stage in ("pre_D", "post_D"):
        tests = json.loads(
            store.read_by_sha256(manifest["verification"]["results_by_stage"][stage])
        )
        tests["applicable_plan_reference"] = digest
        manifest["verification"]["results_by_stage"][stage] = publish(store, manifest, tests)
    report = json.loads(store.read_by_sha256(manifest["evidence"]["final_evaluation"]))
    report["evaluation_plan_sha256"] = plans["before_D"]
    report["statuses"]["calibration_status"] = manifest["verification"]["calibration_status"] = (
        status
    )
    report["calibration_reason"] = (
        None if status == "passed" else "synthetic D calibration unavailable"
    )
    digest = publish(store, manifest, report)
    manifest["evidence"]["final_evaluation"] = manifest["artifacts"][
        "final_evaluation_report_sha256"
    ] = digest
    result = validate_manifest(manifest, "post_D", store, plan_references=plans)
    assert result.gate_passed, result.issues
    if status == "passed":
        manifest["artifacts"]["calibrator_sha256"] = None
        assert not validate_manifest(manifest, "post_D", store, plan_references=plans).gate_passed
