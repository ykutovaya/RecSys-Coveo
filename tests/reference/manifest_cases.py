"""Искусственные свидетельства B13, НЕ результаты алгоритмов или Coveo.

Эталонный порядок стадий и условия заданы прямо из нормативных таблиц.
Конструктор не вызывает валидатор, required_checks или функции построения планов.
Даже положительные случаи не публикуются как фактические допуски запуска.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.config.resources import resource_bytes, validator_fingerprint_bytes

STAGES = (
    "preflight",
    "post_training",
    "post_load",
    "pre_selection",
    "pre_D",
    "post_D",
    "pre_AB",
    "post_AB",
)


def make_manifest_case(
    store: LocalArtifactStore, stage: str, *, operation: str | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Возвращает паспорт и закреплённые ссылки на четыре плана внутри test tmp_path."""
    manifest = json.loads(
        resource_bytes("documents/math/coveo_session_recommender_run_manifest.template.json")
    )
    manifest.update(
        run_id="b13-fixture",
        scope="synthetic",
        status="draft",
        modules=dict.fromkeys(("sampled", "sequence", "freshness", "optional", "ppr"), False),
        module_configurations=dict.fromkeys(("sampled", "sequence", "freshness", "ppr")),
        evidence={},
    )
    config_id = manifest["base_model"]["config_id"]
    serial = 0
    provenance = dict(
        code_sha256=sha256_bytes(b"synthetic code bytes"),
        run_id="b13-fixture",
        scope="synthetic",
        config_id=config_id,
    )

    def publish(value: Any, *, raw: bool = False, name: str | None = None) -> str:
        nonlocal serial
        serial += 1
        path = name or f"fixtures/{serial}.json"
        data = value if raw else canonical_json_bytes(value)
        return store.publish_bytes(path, data, **provenance).sha256

    def envelope(kind: str, **fields: Any) -> dict[str, Any]:
        return dict(
            run_id=manifest["run_id"],
            scope="synthetic",
            config_id=config_id,
            evidence_scope="run",
            kind=kind,
            **fields,
        )

    def evidence(kind: str, **fields: Any) -> str:
        digest = publish(envelope(kind, **fields))
        manifest["evidence"][kind] = digest
        return digest

    dataset = manifest["dataset"]
    dataset.update(version="SYNTHETIC_B13_NOT_COVEO", telemetry_gaps=[])
    for entry in dataset["files"]:
        entry["sha256"] = publish((entry["name"] + "\nsynthetic\n").encode(), raw=True)
    dataset["schema_sha256"] = publish({"synthetic": True, "type": "fixture"})
    dataset["vector_dimensions"] = {"description": 2, "image": 2, "query": 2}
    start = datetime(2020, 1, 1, tzinfo=UTC)
    dataset["coverage_start_utc"] = start.isoformat().replace("+00:00", "Z")
    dataset["coverage_end_utc"] = (start + timedelta(days=70)).isoformat().replace("+00:00", "Z")
    for index, name in enumerate(("A_G", "A_N", "B", "C_smoke", "C_select", "E", "D")):
        manifest["periods"][f"period_{name}"] = {
            "start_utc": (start + timedelta(days=10 * index)).isoformat().replace("+00:00", "Z"),
            "end_utc": (start + timedelta(days=10 * (index + 1)))
            .isoformat()
            .replace("+00:00", "Z"),
        }
    dataset["session_sampling"]["session_count_by_period"] = {
        f"period_{name}": 1000 for name in ("A_G", "A_N", "B", "C_smoke", "C_select", "E", "D")
    }
    feature = {
        "schema_version": "1",
        "config_id": config_id,
        "event_semantics": "set_v1",
        "columns": [
            {
                "name": "synthetic_count",
                "position": 0,
                "dtype": "float64",
                "formula": "synthetic_fixture_only",
                "mask": None,
                "transformation": "identity",
                "fit_period": "none",
                "provenance": {
                    "entity": "request",
                    "sources": ["synthetic"],
                    "available_at": "a_q",
                    "version": "1",
                    "missing_rule": "zero",
                },
            },
            {
                "name": "category",
                "position": 1,
                "dtype": "category",
                "formula": "synthetic_fixture_category",
                "mask": None,
                "transformation": "identity",
                "fit_period": "A_G",
                "provenance": {
                    "entity": "item",
                    "sources": ["synthetic"],
                    "available_at": "a_q",
                    "version": "1",
                    "missing_rule": "unknown_zero",
                },
            },
        ],
    }
    repro = manifest["reproducibility"]
    for key in (
        "code_commit_or_archive_sha256",
        "dependency_lockfile_sha256",
        "manifest_validator_sha256",
        "ranker_callback_sha256",
        "resolved_library_defaults_sha256",
    ):
        repro[key] = publish(("SYNTHETIC_NOT_EXECUTED:" + key).encode(), raw=True)
    repro["manifest_schema_sha256"] = publish(
        resource_bytes("schemas/manifest.schema.json"), raw=True
    )
    repro["manifest_validator_sha256"] = publish(validator_fingerprint_bytes(), raw=True)
    repro["feature_schema_sha256"] = publish(feature)
    repro["platform_and_compiler"] = {"platform": "synthetic", "compiler": "synthetic"}
    repro["random_generators_and_versions"] = {"algorithm": "PCG64", "version": "2.1.3", "seed": 17}
    manifest["ranker"]["resolved_parameters"] = json.loads(
        resource_bytes("configs/ranker_recipe.json")
    )
    repro["resolved_library_defaults_sha256"] = publish(
        {
            "library": "lightgbm",
            "version": "4.6.0",
            "resolved_parameters": manifest["ranker"]["resolved_parameters"],
        }
    )
    manifest["calibration"]["resolved_parameters"] = {
        "optimizer": "projected_newton",
        "initial": {"a": 0, "b": "b0_from_B"},
        "max_iterations": 200,
        "backtracking_factor": 0.5,
        "max_backtracking_steps": 60,
        "armijo": 1e-4,
        "projected_gradient_tolerance": 1e-8,
        "dtype": "float64",
    }
    manifest["inference"].update(
        single_policy_half_width_per_1000=10,
        planned_outcome_variance=0.1,
        dependence_multiplier=1,
        pilot_provenance={"source": "synthetic"},
    )
    manifest["inference"]["dependence_assumptions"]["single_policy_por"] = {
        "justification": "synthetic"
    }
    evidence(
        "data",
        schema_valid=True,
        whole_sessions=True,
        mature_targets=True,
        coverage_checked=True,
        overlapping_sessions=0,
        input_sha256=[x["sha256"] for x in dataset["files"]],
        input_size_bytes={
            x["name"]: len(store.read_by_sha256(x["sha256"])) for x in dataset["files"]
        },
    )
    definitions = []
    identifiers = (
        [f"B{i:02d}" for i in range(1, 19)]
        + [f"N{i:02d}" for i in range(1, 8)]
        + [f"X{i:02d}" for i in range(1, 6)]
        + ["EXPL01"]
    )
    expanded = []
    for identifier in identifiers:
        expanded.extend(
            {
                "B03": ["B03.data", "B03.service"],
                "B10": ["B10.data", "B10.ranker", "B10.calibration", "B10.metadata"],
                "X02": ["X02.support", "X02.loss"],
            }.get(identifier, [identifier])
        )
    early = {"B01", "B02", "B03.data", "B06", "B09", "B10.data", "B13", "X02.support"}
    load_ids = {"B03.service", "B14", "B15", "B16", "B17", "EXPL01"}
    for identifier in expanded:
        condition = (
            "neural"
            if identifier.startswith("N")
            else {
                "B17": "ann",
                "X01": "freshness_or_ppr",
                "X02.support": "sampled",
                "X02.loss": "sampled",
                "X03": "sequence",
                "X04": "sid",
                "X05": "sid",
            }.get(identifier, "always")
        )
        due = (
            "preflight"
            if identifier in early
            else ("post_load" if identifier in load_ids else "post_training")
        )
        if identifier in {"B10.calibration", "B10.metadata", "B11", "B12"}:
            due = "pre_D"
        if identifier == "B18":
            due = "pre_AB"
        definitions.append(dict(test_id=identifier, enabled_when=condition, required_by_stage=due))
    plans: dict[str, str] = {}

    def fixed_inputs() -> dict[str, Any]:
        return {
            key: deepcopy(manifest[key])
            for key in (
                "dataset",
                "periods",
                "task",
                "base_model",
                "ranker",
                "modules",
                "neural_extension",
                "sid_extension",
                "ann_extension",
                "reproducibility",
                "module_configurations",
            )
        }

    def fixture_plan_inputs(name: str) -> dict[str, Any]:
        inputs = fixed_inputs()
        if name == "before_C_select":
            inputs["selection"] = {
                key: deepcopy(manifest["selection"][key])
                for key in ("registered_configurations", "tie_break", "metadata_sensitivity")
            }
        elif name == "before_E":
            inputs.update(
                calibration_recipe=deepcopy(manifest["calibration"]["resolved_parameters"]),
                selected_policies_sha256=manifest["artifacts"]["selected_policies_sha256"],
                selection_report_sha256=manifest["artifacts"]["selection_report_sha256"],
                selection_plan_reference=plans["before_C_select"],
            )
        elif name == "before_D":
            inputs.update(
                inference=deepcopy(manifest["inference"]),
                selection=deepcopy(manifest["selection"]),
                calibration=deepcopy(manifest["calibration"]),
                calibrator_sha256=manifest["artifacts"]["calibrator_sha256"],
                selected_policies_sha256=manifest["artifacts"]["selected_policies_sha256"],
                calibration_result_sha256=manifest["evidence"]["calibration"],
            )
        elif name == "before_AB":
            online = deepcopy(manifest["online_experiment"])
            online.pop("protocol_sha256")
            inputs.update(
                online_experiment=online,
                serving=deepcopy(manifest["serving"]),
                operations=deepcopy(manifest["operations"]),
            )
        return inputs

    def pin(name: str, kind: str, **fields: Any) -> str:
        digest = evidence(kind, inputs=fixture_plan_inputs(name), **fields)
        manifest["freeze"]["plan_references"][name] = plans[name] = digest
        return digest

    model_entry = dict(
        config_id=config_id,
        configuration_sha256=publish(
            {"config_id": config_id, "ranker": deepcopy(manifest["ranker"])}
        ),
        model_sha256=publish(b"SYNTHETIC_NOT_A_MODEL", raw=True),
        feature_schema_sha256=repro["feature_schema_sha256"],
        training_status="passed",
        actual_tree_count=1,
    )
    stages = [
        item
        for item in STAGES[: STAGES.index(stage) + 1]
        if item != "post_load" or stage in {"post_load", "pre_AB", "post_AB"}
    ]
    for current in stages:
        index = STAGES.index(current)
        manifest["stage"] = current
        rows = []
        for definition in definitions:
            identifier = definition["test_id"]
            due = definition["required_by_stage"] == current
            due |= current == "pre_D" and (
                identifier.startswith("B10.") or identifier in {"B11", "B12"}
            )
            due |= (
                current == stage
                and operation == "calibrate"
                and identifier in {"B10.calibration", "B11"}
            )
            due |= current == stage and operation == "evaluate" and identifier == "B12"
            if not due:
                continue
            disabled = definition["enabled_when"] != "always"
            row = dict(
                test_id=identifier,
                status="not_used" if disabled else "passed",
                reason="module disabled"
                if disabled
                else "SYNTHETIC B13 fixture, no algorithm executed",
            )
            if not disabled:
                row.update(
                    complete_test=True,
                    exact_fields_passed=True,
                    counts=dict(collected=1, passed=1, failed=0, skipped=0, errors=0),
                    covered_parts=["B05.numerics", "B05.derivatives", "B05.ranker"]
                    if identifier == "B05"
                    else [],
                )
                if identifier == "B05":
                    row["derivative_tolerances"] = {
                        "callback_fp64": [1e-12, 1e-10],
                        "callback_fp32": [1e-7, 1e-6],
                        "one_step_tree": [1e-6, 0],
                    }
            rows.append(row)
        manifest["verification"].update(data_status="passed", math_status="passed")
        if index == 1:
            manifest["artifacts"]["trained_models_sha256"] = publish(
                envelope("trained_models", configurations=[model_entry])
            )
        if index == 2:
            manifest["serving"].update(
                cpu_model_and_threads={"model": "synthetic", "threads": 8},
                ram_meter_sampling_interval_ms=100,
                ram_process_scope=["server", "state", "log_writer"],
            )
            evidence(
                "load",
                trials=[
                    dict(
                        p95_ms=20, p99_ms=30, error_fraction=0, fallback_fraction=0, peak_ram_gib=1
                    )
                    for _ in range(3)
                ],
                replay_passed=True,
                recovery_passed=True,
                input_limits_passed=True,
                integrity_passed=True,
                rto_seconds=1,
                rpo=0,
                load_profile=deepcopy(manifest["serving"]),
                replay_mode="ideal_deterministic",
                evaluated_config_id=config_id,
                configuration_sha256=model_entry["configuration_sha256"],
            )
            model_entry["load_report_sha256"] = manifest["evidence"]["load"]
            manifest["artifacts"]["trained_models_sha256"] = publish(
                envelope("trained_models", configurations=[model_entry])
            )
            manifest["verification"]["serve_status"] = "passed"
        if index == 3:
            manifest["selection"]["registered_configurations"] = [
                dict(
                    config_id=config_id,
                    configuration_sha256=model_entry["configuration_sha256"],
                    parameter_count=1,
                )
            ]
            pin(
                "before_C_select",
                "selection_plan",
                configurations=manifest["selection"]["registered_configurations"],
                budget={"variants": 1},
                control=config_id,
                tie_break=manifest["selection"]["tie_break"],
            )
        if index == 4 or (current == stage and operation == "calibrate"):
            manifest["artifacts"]["selected_policies_sha256"] = publish(
                envelope("selected_policies", policies=[config_id])
            )
            manifest["artifacts"]["selection_report_sha256"] = publish(
                envelope(
                    "selection",
                    status="passed",
                    policies=[config_id],
                    selection_plan_sha256=manifest["evidence"]["selection_plan"],
                )
            )
            pin("before_E", "calibration_plan")
        if index == 4:
            manifest["inference"].update(
                calendar_anchor_utc=manifest["periods"]["period_D"]["start_utc"],
                calendar_cell_seconds=60,
                planned_cells=12930,
                block_cells=60,
                sensitivity_block_cells=[30, 120],
            )
            manifest["verification"]["calibration_status"] = "insufficient"
            evidence(
                "calibration",
                status="insufficient",
                reason="synthetic class support missing",
                calibration_plan_sha256=plans["before_E"],
            )
            pin(
                "before_D",
                "evaluation_plan",
                inference=deepcopy(manifest["inference"]),
                primary_comparison=None,
            )
        if index == 5:
            manifest["verification"]["offline_status"] = "passed"
            manifest["artifacts"]["final_evaluation_report_sha256"] = evidence(
                "final_evaluation",
                metrics={"N": 1000, "POR": 100},
                intervals={"POR": [90, 110]},
                maturity={"complete": True},
                metadata_sensitivity={"evaluated": True},
                statuses={"offline_status": "passed", "calibration_status": "insufficient"},
                calibration_reason="synthetic class support missing",
                evaluation_plan_sha256=manifest["evidence"]["evaluation_plan"],
                interval_details={
                    "valid": True,
                    "reason": None,
                    "method": manifest["inference"]["primary_interval"],
                    "confidence_level": 0.95,
                },
                information={
                    "full_blocks": 215,
                    "nonempty_blocks": 20,
                    "category_counts": {"0": 900, "1": 100},
                    "category_blocks": {"0": 20, "1": 20},
                    "undefined_repetitions": 0,
                    "dependence_assumptions_satisfied": True,
                },
                contract_violations=[],
            )
        if index == 6:
            manifest["online_experiment"].update(
                enabled=True,
                assignment_unit="session",
                estimand="session_conversion",
                business_threshold=0,
                sample_size_and_horizon={
                    "n_per_arm": 1000,
                    "start_utc": "2020-04-01T00:00:00Z",
                    "end_utc": "2020-04-10T00:00:00Z",
                    "analysis_not_before_utc": "2020-04-12T00:00:00Z",
                    "alpha": 0.05,
                    "power": 0.8,
                    "delta_plan": 50.0,
                    "baseline_probability": 0.1,
                    "interval_method": "clopper_pearson_bonferroni",
                    "assumptions_justification": "synthetic",
                    "sample_size_justification": "synthetic",
                    "missing_outcome_policy": "require_complete",
                },
                guardrail_margins={
                    key: {
                        "limit": 10 if key == "latency_ms" else 0.01,
                        "comparison": "le",
                        "unit": "ms" if key == "latency_ms" else "fraction",
                        "window_seconds": 60,
                        "denominator": "assigned_units",
                        "minimum_count": 100,
                        "responsible_action": "pause",
                        "schedule_utc": ["2020-04-05T00:00:00Z"],
                    }
                    for key in (
                        "error_fraction",
                        "latency_ms",
                        "empty_response_fraction",
                        "purchase_logging_loss_fraction",
                        "invalid_item_fraction",
                    )
                },
            )
            manifest["online_experiment"]["srm_plan"] = {
                "assignment_mechanism": "independent_bernoulli",
                "cohort_membership_rule": "synthetic immutable cohort",
                "conditional_null_justification": "synthetic",
                "test_method": "exact_two_sided_binomial",
                "familywise_error_control_justification": "Bonferroni",
                "schedule_utc": ["2020-04-05T00:00:00Z"],
                "number_of_checks": 1,
                "alpha": 0.001,
                "expected_treatment_probability": 0.5,
                "strata": [],
            }
            manifest["operations"].update(
                artifact_authentication="signed",
                production_tenant_limits={"events_per_second": 100},
                production_sku_contribution_limits={"sessions_per_day": 100},
                production_event_time_range_utc={
                    "start_utc": "2020-01-01T00:00:00Z",
                    "end_utc": "2021-01-01T00:00:00Z",
                },
                trusted_production_key_ids=["synthetic-key"],
                production_block_aggregator_id="synthetic-owner",
                responsible_role="synthetic-role",
                analysis_close_utc="2021-01-01T00:00:00Z",
                monitoring_resolved_configuration={"threshold": 1, "control_windows": 100},
            )
            manifest["operations"]["production_owner_registry_sha256"] = publish(
                {"synthetic-owner": "synthetic"}
            )
            evidence(
                "production",
                signature_verified=True,
                access_and_retention_verified=True,
                limits_verified=True,
                monitoring_calibrated=True,
                profile="production",
            )
            online_plan = deepcopy(manifest["online_experiment"])
            online_plan.pop("protocol_sha256")
            manifest["online_experiment"]["protocol_sha256"] = pin(
                "before_AB", "online_protocol", plan=online_plan
            )
        if index == 7:
            manifest["verification"]["business_status"] = "passed"
            evidence(
                "online_results",
                mature=True,
                integrity_passed=True,
                itt_complete=True,
                stopping_status="completed",
                assigned_counts={"control": 1000, "treatment": 1000},
                outcomes={"control": 20, "treatment": 40},
                business_status="passed",
                unknown_outcomes={"control": 0, "treatment": 0},
                protocol_sha256=manifest["online_experiment"]["protocol_sha256"],
                assignment_unit="session",
                estimand="session_conversion",
                assumptions_satisfied=True,
                srm_passed=True,
                analysis_mode="confirmatory",
                reason=None,
                interval={
                    "lower": 1.0,
                    "upper": 39.0,
                    "valid": True,
                    "reason": None,
                    "method": "clopper_pearson_bonferroni",
                    "confidence_level": 0.95,
                },
                guardrail_results={
                    name: {
                        "value": 0.0,
                        "count": 1000,
                        "status": "passed",
                        "unit": rule["unit"],
                        "limit": rule["limit"],
                        "comparison": rule["comparison"],
                        "reason": None,
                    }
                    for name, rule in manifest["online_experiment"]["guardrail_margins"].items()
                },
            )
        plan_name = (
            "before_AB"
            if current in {"pre_AB", "post_AB"}
            else "before_D"
            if current in {"pre_D", "post_D"}
            else "before_E"
            if current == stage and operation == "calibrate"
            else "before_C_select"
            if current == "pre_selection"
            else None
        )
        manifest["verification"]["results_by_stage"][current] = publish(
            envelope(
                "verification_results",
                stage=current,
                operation=operation if current == stage else None,
                manifest_reference=manifest["run_id"],
                code_version=repro["code_commit_or_archive_sha256"],
                configuration_or_reference=sha256_bytes(canonical_json_bytes(fixed_inputs())),
                random_seed=17,
                applicable_plan_reference=plans.get(plan_name),
                tests=rows,
            )
        )
    return manifest, plans
