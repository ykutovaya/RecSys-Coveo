"""Проверка свидетельств восьми стадий, без исполнения будущих алгоритмов.

Ссылка на отчёт недостаточна: проверяются байты, область, запуск, модель,
версия кода и содержимое. Неизменяемые ссылки на планы передаёт вызывающий
сценарий; цепочка снимков паспорта не требуется. Синтетические свидетельства допустимы
только для синтетического запуска и не подтверждают результаты на Coveo.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts.common import Stage, ValidationIssue, ValidationResult

from .evidence import validate_final_evaluation, validate_online_plan, validate_online_results
from .recipes import resolved_recipe_issues
from .resources import load_registry, load_template, resource_bytes, validator_fingerprint_bytes
from .schema import structure_issues, validate_feature_schema, validate_stage_report

STAGES = tuple(stage.value for stage in Stage)
PERIODS = ("A_G", "A_N", "B", "C_smoke", "C_select", "E", "D")
SERVICE_STAGES = {"post_load", "pre_AB", "post_AB"}


def _at(value: Mapping[str, Any], path: str) -> Any:
    result: Any = value
    for part in path.split("."):
        if not isinstance(result, dict) or part not in result:
            return None
        result = result[part]
    return result


def _time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Нужно время UTC с суффиксом Z")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != UTC.utcoffset(result):
        raise ValueError("Нужно UTC")
    return result


def _positive(value: Any, *, zero: bool = False) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value) and (value >= 0 if zero else value > 0)
    except OverflowError:
        return False


def _checks() -> list[dict[str, Any]]:
    """Сроки требований не берутся из редактируемого паспорта."""
    definitions: list[dict[str, Any]] = []
    early = {"B01", "B02", "B06", "B09", "B13"}
    registry = load_registry()
    for identifier in registry["normative_ids"] + registry["own_ids"]:
        due = "preflight" if identifier in early else "post_training"
        if identifier in {"B14", "B15", "B16", "B17", "EXPL01"}:
            due = "post_load"
        if identifier == "B12":
            due = "pre_D"
        if identifier == "B18":
            due = "pre_AB"
        enabled = (
            "neural"
            if identifier.startswith("N")
            else {
                "B17": "ann",
                "X01": "freshness_or_ppr",
                "X02": "sampled",
                "X03": "sequence",
                "X04": "sid",
                "X05": "sid",
            }.get(identifier, "always")
        )
        split = {
            "B03": {"B03.data": "preflight", "B03.service": "post_load"},
            "B10": {
                "B10.data": "preflight",
                "B10.ranker": "post_training",
                "B10.metadata": "pre_D",
            },
            "X02": {"X02.support": "preflight", "X02.loss": "post_training"},
        }
        for key, deadline in split.get(identifier, {identifier: due}).items():
            repeat = ["pre_D"] if key.startswith("B10.") or key == "B12" else []
            definitions.append(
                dict(
                    test_id=key,
                    required_by_stage=deadline,
                    enabled_when=enabled,
                    repeat_at=repeat,
                    parts=["B05.numerics", "B05.derivatives", "B05.ranker"] if key == "B05" else [],
                )
            )
    return definitions


def required_checks(
    manifest: Mapping[str, Any],
    stage: Stage | str,
    *,
    operation: str | None = None,
) -> list[dict[str, Any]]:
    """Применимые требования для обычных тестов и сохранённых отчётов."""
    stage_name = Stage(stage).value
    if operation not in {None, "evaluate"}:
        raise ValueError("Неизвестная операция")
    index = STAGES.index(stage_name)
    result = []
    for definition in _checks():
        deadline = definition["required_by_stage"]
        if deadline == "post_load" and stage_name not in SERVICE_STAGES:
            continue
        operation_ids = {"B12"} if operation == "evaluate" else set()
        if definition["test_id"] in operation_ids and index < STAGES.index(deadline):
            definition = dict(definition, required_by_stage=stage_name)
        if STAGES.index(definition["required_by_stage"]) <= index:
            result.append(definition)
    return result


def verification_configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Закреплённые входы и настройки, общие для отчёта этапа."""
    return {
        key: deepcopy(manifest[key])
        for key in (
            "evaluation_protocol",
            "calibration",
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


def plan_inputs(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Конкретные планы связываются с используемыми входами."""
    inputs = verification_configuration(manifest)
    if name == "before_C_select":
        inputs["selection"] = {
            key: deepcopy(manifest["selection"][key])
            for key in ("registered_configurations", "tie_break", "metadata_sensitivity")
        }
    elif name == "before_D":
        inputs.update(
            inference=deepcopy(manifest["inference"]),
            selection=deepcopy(manifest["selection"]),
            selected_policies_sha256=manifest["artifacts"]["selected_policies_sha256"],
            selection_report_sha256=manifest["artifacts"]["selection_report_sha256"],
            selection_plan_reference=manifest["freeze"]["plan_references"]["before_C_select"],
        )
    elif name == "before_AB":
        online = deepcopy(manifest["online_experiment"])
        online.pop("protocol_sha256")
        inputs.update(
            online_experiment=online,
            serving=deepcopy(manifest["serving"]),
            operations=deepcopy(manifest["operations"]),
        )
    else:
        raise ValueError("Неизвестный план")
    return inputs


def _enabled(manifest: Mapping[str, Any], condition: str) -> bool:
    if condition == "always":
        return True
    if condition in {"neural", "sid", "ann"}:
        return bool(_at(manifest, f"{condition}_extension.enabled"))
    if condition == "freshness_or_ppr":
        return bool(_at(manifest, "modules.freshness") or _at(manifest, "modules.ppr"))
    return bool(_at(manifest, f"modules.{condition}"))


class _Validation:
    def __init__(
        self,
        manifest: dict[str, Any],
        stage: str,
        store: LocalArtifactStore,
        plan_references: Mapping[str, str],
        input_paths: Mapping[str, str | Path] | None,
        operation: str | None,
    ) -> None:
        self.m, self.stage, self.store = manifest, stage, store
        self.index = STAGES.index(stage)
        self.plan_references = plan_references
        self.operation = operation
        self.reports: dict[str, dict[str, Any]] = {}
        self.input_paths = input_paths
        self.input_sizes: dict[str, int] = {}
        self.issues: list[ValidationIssue] = []
        self.template = load_template()

    def issue(self, path: str, code: str, message: str, layer: str = "semantic") -> None:
        self.issues.append(ValidationIssue(path, code, message, layer))

    def need(self, path: str) -> Any:
        value = _at(self.m, path)
        if value is None or value == "" or value == {} or value == []:
            self.issue(path, "required", "Обязательное поле не заполнено", "results")
        return value

    def read_hash(
        self, digest: Any, path: str, *, envelope: bool = False, config_id: str | None = None
    ) -> Any:
        if digest is None:
            self.issue(path, "missing_artifact", "Нет обязательного хеша", "results")
            return None
        try:
            content = self.store.read_by_sha256(digest)
            if not envelope:
                return content
            report = strict_json_loads(content)
            if not isinstance(report, dict):
                raise ValueError("Нужен объект отчёта")
            expected = {
                "run_id": self.m["run_id"],
                "scope": self.m["scope"],
                "config_id": config_id or self.m["base_model"]["config_id"],
            }
            if any(report.get(key) != value for key, value in expected.items()):
                raise ValueError("Отчёт относится к другому запуску/области/модели")
            if report.get("evidence_scope") != "run":
                raise ValueError("Проверка кода не является свидетельством конкретного запуска")
            return report
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.issue(path, "artifact_invalid", str(exc))
            return None

    def evidence(self, name: str) -> dict[str, Any] | None:
        result = self.read_hash(self.m["evidence"].get(name), f"evidence.{name}", envelope=True)
        if result is not None and result.get("kind") != name:
            self.issue(f"evidence.{name}", "evidence_kind", "Неверный вид свидетельства")
            return None
        return result

    def policy(self) -> None:
        if self.operation == "evaluate" and self.index < 1:
            self.issue(
                "operation",
                "operation_stage",
                "Операция требует завершённого обучения и применимого выбора",
            )
        if self.m["stage"] != self.stage:
            self.issue("stage", "stage_mismatch", "Запрошенная стадия не совпала с паспортом")
        if self.m["status"] == "template_not_runnable":
            self.issue("status", "template_not_runnable", "Шаблон не является запуском")
        if self.m["stage_validation"] != self.template["stage_validation"]:
            self.issue("stage_validation", "policy_changed", "Нормативные правила неизменяемы")
        if (
            self.m["verification"]["property_case_count"] < 1000
            or self.m["verification"]["seed"] != 17
        ):
            self.issue("verification", "test_protocol", "Требуется минимум 1000 случаев и seed=17")
        for group in ("task",):
            if self.m[group] != self.template[group]:
                self.issue(group, "task_changed", "Изменена нормативная постановка первого профиля")
        mutable = {
            "calibration": set(),
            "inference": {key for key, value in self.template["inference"].items() if value is None}
            | {"primary_estimate_type", "primary_policy_config_id", "dependence_assumptions"},
            "selection": {"registered_configurations", "primary_comparison"},
            "online_experiment": {
                "enabled",
                "protocol_sha256",
                "assignment_unit",
                "estimand",
                "business_threshold",
                "sample_size_and_horizon",
                "srm_plan",
                "guardrail_margins",
            },
            "serving": {
                "cpu_model_and_threads",
                "ram_meter_sampling_interval_ms",
                "ram_process_scope",
            },
            "operations": {
                key for key, value in self.template["operations"].items() if value is None
            }
            | {"artifact_authentication", "replay_mode"},
            "reproducibility": {
                key for key, value in self.template["reproducibility"].items() if value is None
            },
        }
        for group, permitted in mutable.items():
            for key, expected in self.template[group].items():
                if key not in permitted and self.m[group][key] != expected:
                    self.issue(
                        f"{group}.{key}",
                        "normative_policy",
                        "Изменён нормативный метод/период/порог",
                    )
        for path in (
            "evaluation_protocol",
            "dataset.telemetry_completeness",
            "dataset.timezone",
            "dataset.timestamp_unit",
            "dataset.search_payload_policy",
            "dataset.search_canonicalization",
            "dataset.session_sampling.method",
            "dataset.session_sampling.seed",
            "dataset.duplicate_sensitivity",
            "periods.interval_convention",
        ):
            if _at(self.m, path) != _at(self.template, path):
                self.issue(path, "data_contract", "Изменён контракт данных")
        for module in ("neural", "sid", "ann"):
            for key, expected in self.template[f"{module}_extension"].items():
                if key not in {
                    "enabled",
                    "resolved_configuration",
                    "regret_score_formula_references",
                    "maximum_hit_drop_fraction",
                }:
                    if self.m[f"{module}_extension"][key] != expected:
                        self.issue(
                            f"{module}_extension.{key}",
                            "module_recipe_changed",
                            "Изменён нормативный контракт расширения",
                        )
        if self.m["base_model"]["config_id"] == "gcs_rk_v2_7":
            if self.m["base_model"] != self.template["base_model"]:
                self.issue("base_model", "baseline_changed", "Изменён закреплённый рецепт базы")
            for key, expected in self.template["ranker"].items():
                if key != "resolved_parameters" and self.m["ranker"][key] != expected:
                    self.issue(f"ranker.{key}", "baseline_changed", "Изменён рецепт ранжировщика")
        for key in (
            "library",
            "version",
            "api",
            "objective_callable",
            "formula_contract",
            "record_actual_tree_count",
            "pair_weights_applied_in_callback_once",
            "dataset_weight_mode",
            "callback_accumulation_dtype",
            "callback_output_dtype",
            "native_lambdarank_controls_used",
        ):
            if self.m["ranker"][key] != self.template["ranker"][key]:
                self.issue(
                    f"ranker.{key}", "ranker_contract", "Нарушен общий контракт ранжировщика"
                )
        if self.m["sid_extension"]["enabled"] and not self.m["neural_extension"]["enabled"]:
            self.issue("sid_extension", "module_dependency", "SID требует нейросетевой опоры")
        for module in ("sampled", "sequence"):
            if self.m["modules"][module] and not self.m["neural_extension"]["enabled"]:
                self.issue(f"modules.{module}", "module_dependency", "Нужен нейросетевой модуль")
        for module in ("neural", "sid", "ann"):
            if self.m[f"{module}_extension"]["enabled"]:
                config = self.need(f"{module}_extension.resolved_configuration")
                required = {"parameters", "implementation_sha256"}
                if (
                    not isinstance(config, dict)
                    or not required <= config.keys()
                    or not config.get("parameters")
                ):
                    self.issue(
                        f"{module}_extension.resolved_configuration",
                        "module_configuration",
                        "Нужны parameters и implementation_sha256 включённого модуля",
                    )
                else:
                    self.read_hash(
                        config["implementation_sha256"], f"{module}_extension.implementation"
                    )
        if self.m["operations"]["replay_mode"] == "execution_trace":
            self.read_hash(
                self.need("operations.execution_trace_manifest_sha256"), "execution_trace"
            )
        elif self.m["operations"]["replay_mode"] != "ideal_deterministic":
            self.issue(
                "operations.replay_mode", "unknown_replay_mode", "Неизвестный режим воспроизведения"
            )
        if self.m["operations"]["artifact_authentication"] not in ("local_checksum_only", "signed"):
            self.issue(
                "operations.artifact_authentication",
                "unknown_authentication",
                "Неизвестный режим проверки пакетов",
            )
        self.issues.extend(resolved_recipe_issues(self.m))

    def preflight(self) -> None:
        for path in (
            "dataset.version",
            "dataset.schema_sha256",
            "dataset.telemetry_gaps",
            "dataset.session_sampling.session_count_by_period",
            "reproducibility.platform_and_compiler",
            "reproducibility.random_generators_and_versions",
            "ranker.resolved_parameters",
        ):
            if path == "dataset.telemetry_gaps" and _at(self.m, path) == []:
                continue
            self.need(path)
        files = self.m["dataset"]["files"]
        if len(files) != 3 or {x["name"] for x in files} != {
            "browsing_train.csv",
            "search_train.csv",
            "sku_to_content.csv",
        }:
            self.issue("dataset.files", "dataset_files", "Нужны три различные нормативные таблицы")
        for index, item in enumerate(files):
            if self.input_paths is None:
                raw = self.read_hash(item["sha256"], f"dataset.files.{index}.sha256")
                if raw is not None:
                    self.input_sizes[item["name"]] = len(raw)
            else:
                try:
                    if item["name"] not in self.input_paths:
                        raise ValueError(
                            "Нет внешнего пути обязательной таблицы; копия в store не подставляется"
                        )
                    input_path = Path(self.input_paths[item["name"]])
                    before = input_path.stat()
                    digest = sha256_file(input_path)
                    after = input_path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise ValueError("Входной файл изменился во время проверки")
                    if digest != item["sha256"]:
                        raise ValueError("Хеш внешней таблицы не совпал с паспортом")
                    self.input_sizes[item["name"]] = after.st_size
                except (OSError, ValueError, TypeError) as exc:
                    self.issue(f"dataset.files.{index}", "external_input", str(exc))
        for key in ("description", "image", "query"):
            self.need(f"dataset.vector_dimensions.{key}")
        if _at(self.m, "dataset.vector_dimensions.query") != _at(
            self.m, "dataset.vector_dimensions.description"
        ):
            self.issue(
                "dataset.vector_dimensions", "query_dimension", "Поиск и описание несовместимы"
            )
        hash_fields = (
            "code_commit_or_archive_sha256",
            "dependency_lockfile_sha256",
            "feature_schema_sha256",
            "manifest_schema_sha256",
            "manifest_validator_sha256",
            "ranker_callback_sha256",
            "resolved_library_defaults_sha256",
        )
        for key in hash_fields:
            data = self.read_hash(self.need(f"reproducibility.{key}"), f"reproducibility.{key}")
            if data is not None and key == "feature_schema_sha256":
                try:
                    feature = strict_json_loads(data)
                    self.issues.extend(validate_feature_schema(feature))
                    if (
                        feature.get("config_id") != self.m["base_model"]["config_id"]
                        or not feature.get("columns")
                        or feature.get("event_semantics") != self.m["dataset"]["event_semantics"]
                    ):
                        self.issue(
                            "reproducibility.feature_schema_sha256",
                            "feature_binding",
                            "Нужна полная схема варианта",
                        )
                    categories = [
                        column["name"]
                        for column in feature.get("columns", [])
                        if column.get("dtype") == "category"
                    ]
                    if (
                        _at(self.m, "ranker.resolved_parameters.dataset.categorical_feature")
                        != categories
                        or len(categories) != 1
                    ):
                        self.issue(
                            "ranker.resolved_parameters.dataset.categorical_feature",
                            "categorical_binding",
                            "Единственный категориальный столбец не совпал со схемой",
                        )
                except (ValueError, TypeError, AttributeError) as exc:
                    self.issue("reproducibility.feature_schema_sha256", "feature_schema", str(exc))
            if data is not None and key == "manifest_schema_sha256":
                if data != resource_bytes("schemas/manifest.schema.json"):
                    self.issue(
                        "reproducibility.manifest_schema_sha256",
                        "schema_binding",
                        "Использована другая схема",
                    )
            if (
                data is not None
                and key == "manifest_validator_sha256"
                and data != validator_fingerprint_bytes()
            ):
                self.issue(
                    "reproducibility.manifest_validator_sha256",
                    "validator_binding",
                    "Хеш не описывает исполняемый валидатор",
                )
            if data is not None and key == "resolved_library_defaults_sha256":
                try:
                    defaults = strict_json_loads(data)
                    if (
                        not isinstance(defaults, dict)
                        or defaults.get("library") != "lightgbm"
                        or defaults.get("version") != "4.6.0"
                        or defaults.get("resolved_parameters")
                        != self.m["ranker"]["resolved_parameters"]
                    ):
                        raise ValueError(
                            "Артефакт полных настроек не совпал с рецептом ранжировщика"
                        )
                except (ValueError, TypeError) as exc:
                    self.issue(
                        "reproducibility.resolved_library_defaults_sha256",
                        "defaults_binding",
                        str(exc),
                    )
        self.read_hash(self.m["dataset"]["schema_sha256"], "dataset.schema_sha256")
        resolved = self.m["ranker"]["resolved_parameters"]
        if not isinstance(resolved, dict) or not all(
            isinstance(resolved.get(k), dict) and resolved[k]
            for k in ("model", "dataset", "training", "prediction")
        ):
            self.issue(
                "ranker.resolved_parameters",
                "resolved_recipe",
                "Нужны четыре части полного рецепта",
            )
        self.calendar()
        estimate = self.m["inference"]["primary_estimate_type"]
        for path in self.template["stage_validation"]["primary_estimate_required_fields"][estimate]:
            if path == "selection.primary_comparison" and self.index < 4:
                continue
            self.need(path)
        for path in ("inference.dependence_multiplier", "inference.pilot_provenance"):
            self.need(path)
        if estimate == "paired" and self.index >= 4:
            pair = self.m["selection"]["primary_comparison"]
            if pair and pair["control"] == pair["treatment"]:
                self.issue(
                    "selection.primary_comparison",
                    "same_policy",
                    "Основная пара должна различаться",
                )
        data_report = self.evidence("data")
        if data_report:
            for key in (
                "schema_valid",
                "whole_sessions",
                "published_rows_complete",
                "calendar_admission_checked",
                "registered_targets_checked",
            ):
                if data_report.get(key) is not True:
                    self.issue(
                        f"evidence.data.{key}",
                        "data_condition",
                        "Не подтверждено условие данных",
                        "gate",
                    )
            if (
                type(data_report.get("overlapping_sessions")) is not int
                or data_report["overlapping_sessions"] != 0
            ):
                self.issue(
                    "evidence.data.overlapping_sessions",
                    "session_overlap",
                    "Сессии пересекаются",
                    "gate",
                )
            if data_report.get("telemetry_completeness") != "unknown":
                self.issue(
                    "evidence.data.telemetry_completeness",
                    "telemetry_completeness",
                    "Полнота телеметрии должна оставаться неизвестной",
                )
            expected = [x["sha256"] for x in files]
            if data_report.get("input_sha256") != expected:
                self.issue(
                    "evidence.data.input_sha256", "data_binding", "Отчёт проверял другие файлы"
                )
            if data_report.get("input_size_bytes") != self.input_sizes:
                self.issue(
                    "evidence.data.input_size_bytes",
                    "data_size_binding",
                    "Размеры таблиц не совпали с проверенными входами",
                )

    def calendar(self) -> None:
        try:
            start = _time(self.need("dataset.coverage_start_utc"))
            end = _time(self.need("dataset.coverage_end_utc"))
            previous = start
            minimum = self.m["task"]["horizon_seconds"] + self.m["task"]["inactivity_seconds"]
            for name in PERIODS:
                interval = self.m["periods"][f"period_{name}"]
                left, right = _time(interval["start_utc"]), _time(interval["end_utc"])
                if left < previous or right <= left or right > end:
                    raise ValueError("Периоды пересекаются, переставлены или выходят за покрытие")
                if (right - left).total_seconds() <= minimum:
                    raise ValueError("Период не содержит непустого окна допуска H+g")
                previous = right
            for gap in self.m["dataset"]["telemetry_gaps"] or []:
                if not start <= _time(gap["start_utc"]) < _time(gap["end_utc"]) <= end:
                    raise ValueError("Разрыв вне покрытия")
        except (ValueError, TypeError) as exc:
            self.issue("periods", "calendar", str(exc))

    def results(self) -> None:
        for stage, digest in self.m["verification"]["results_by_stage"].items():
            if digest is None or STAGES.index(stage) > self.index:
                continue
            if stage == "post_load" and self.stage not in SERVICE_STAGES:
                continue
            report = self.read_hash(digest, f"verification.results_by_stage.{stage}", envelope=True)
            if report is None:
                continue
            issues = validate_stage_report(report)
            if issues:
                self.issues.extend(issues)
                continue
            self.reports[stage] = report
            if report.get("kind") != "verification_results" or report.get("stage") != stage:
                self.issue(stage, "report_stage", "Неверный вид или этап отчёта")
            if (
                stage == self.stage
                and self.operation is not None
                and report.get("operation") != self.operation
            ):
                self.issue(stage, "report_operation", "Отчёт относится к другой операции")
            expected = sha256_bytes(canonical_json_bytes(verification_configuration(self.m)))
            if (
                report.get("code_version")
                != self.m["reproducibility"]["code_commit_or_archive_sha256"]
                or report.get("configuration_or_reference") != expected
                or report.get("random_seed") != self.m["verification"]["seed"]
                or report.get("manifest_reference") != self.m["run_id"]
            ):
                self.issue(
                    stage,
                    "report_configuration",
                    "Отчёт не связан с кодом, входами и настройками запуска",
                )
            rows = report.get("tests")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                self.issue(stage, "result_rows", "Нет списка фактических результатов")
                continue
            identifiers = [row.get("test_id") for row in rows]
            if any(not isinstance(identifier, str) for identifier in identifiers) or len(
                set(identifiers)
            ) != len(rows):
                self.issue(stage, "duplicate_test", "Некорректные или повторные ID требований")
            plan_name = (
                "before_AB"
                if stage in {"pre_AB", "post_AB"}
                else "before_D"
                if stage in {"pre_D", "post_D"}
                else "before_C_select"
                if stage == "pre_selection"
                else None
            )
            expected_plan = (
                self.m["freeze"]["plan_references"].get(plan_name) if plan_name else None
            )
            if report.get("applicable_plan_reference") != expected_plan:
                self.issue(
                    stage, "repeat_inputs", "Результаты не связаны с применимым неизменяемым планом"
                )
        for definition in required_checks(self.m, self.stage, operation=self.operation):
            self.test_result(definition, disabled=not _enabled(self.m, definition["enabled_when"]))

    def test_result(self, definition: dict[str, Any], *, disabled: bool = False) -> None:
        identifier = definition["test_id"]
        earliest = STAGES.index(definition["required_by_stage"])
        for repeated in definition["repeat_at"]:
            if STAGES.index(repeated) <= self.index:
                earliest = STAGES.index(repeated)
        candidates: list[dict[str, Any]] = []
        for stage in STAGES[earliest : self.index + 1]:
            rows = self.reports.get(stage, {}).get("tests", [])
            if isinstance(rows, list):
                candidates.extend(
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get("test_id") == identifier
                )
        if not candidates:
            self.issue(identifier, "not_run", "Нет обязательного результата", "results")
            return
        row = candidates[-1]
        status, reason = row.get("status"), row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            self.issue(identifier, "result_reason", "Нужна причина результата")
        if disabled:
            if status != "not_used":
                self.issue(
                    identifier,
                    "disabled_reason",
                    "Отключённая проверка требует not_used с причиной",
                )
            return
        if status == "not_run":
            self.issue(identifier, "not_run", "Проверка не выполнена", "results")
            return
        if status not in {"passed", "failed", "insufficient"}:
            self.issue(identifier, "active_test_omitted", "Нет допустимого фактического результата")
            return
        counts = row.get("counts")
        keys = {"collected", "passed", "failed", "skipped", "errors"}
        if (
            not isinstance(counts, dict)
            or set(counts) != keys
            or any(type(counts[key]) is not int or counts[key] < 0 for key in keys)
        ):
            self.issue(
                identifier, "test_counts", "Нет корректных счётчиков обычного тестового запуска"
            )
        elif counts["collected"] != counts["passed"] + counts["failed"] + counts[
            "skipped"
        ] + counts["errors"] or (
            status == "passed"
            and (counts["collected"] == 0 or counts["passed"] != counts["collected"])
        ):
            self.issue(
                identifier,
                "test_counts",
                "Нулевой, пропущенный или неуспешный запуск не даёт passed",
            )
        if row.get("complete_test") is not True:
            self.issue(identifier, "partial_test", "Частичный тест не закрывает ID")
        if definition["parts"] and row.get("covered_parts") != definition["parts"]:
            self.issue(identifier, "missing_parts", "Не подтверждены все части теста")
        if row.get("exact_fields_passed") is not True:
            layer = (
                "gate"
                if row.get("exact_fields_passed") is False and status == "failed"
                else "semantic"
            )
            self.issue(
                identifier,
                "exact_fields",
                "Счётчики, идентификаторы, порядок и маски проверяются точно",
                layer,
            )
        if identifier == "B05":
            expected = {
                "callback_fp64": [1e-12, 1e-10],
                "callback_fp32": [1e-7, 1e-6],
                "one_step_tree": [1e-6, 0],
            }
            if row.get("derivative_tolerances") != expected:
                self.issue(
                    identifier, "derivative_tolerances", "Нужны нормативные допуски производных"
                )
        if status != "passed":
            self.issue(identifier, "test_failed", reason or status, "gate")

    def plans(self) -> None:
        names = []
        if self.index >= 3:
            names.append(("before_C_select", "selection_plan"))
        if self.index >= 4:
            names.append(("before_D", "evaluation_plan"))
        if self.index >= 6:
            names.append(("before_AB", "online_protocol"))
        for name, evidence_name in names:
            pinned = self.plan_references.get(name)
            current = self.m["freeze"]["plan_references"][name]
            if pinned is None:
                self.issue(
                    name,
                    "missing_plan_reference",
                    "Нет внешней закреплённой ссылки на план",
                    "results",
                )
            if pinned != current or current != self.m["evidence"].get(evidence_name):
                self.issue(name, "plan_reference", "План не совпал с закреплённой ссылкой")
            report = self.read_hash(pinned, name, envelope=True) if pinned is not None else None
            if report and report.get("kind") != evidence_name:
                self.issue(name, "evidence_kind", "Закреплена ссылка на другой вид плана")
            if report and report.get("inputs") != plan_inputs(self.m, name):
                code = {
                    "before_C_select": "selection_plan_changed",
                    "before_D": "evaluation_plan_changed",
                    "before_AB": "online_plan_changed",
                }[name]
                self.issue(name, code, "Код, данные или настройки изменились после фиксации плана")

    def model_index(self, report: Any) -> None:
        entries = report.get("configurations") if isinstance(report, dict) else None
        if not isinstance(entries, list) or not entries:
            self.issue(
                "artifacts.trained_models_sha256",
                "model_index",
                "Нужен непустой индекс обученных вариантов",
            )
            return
        ids: set[str] = set()
        for index, item in enumerate(entries):
            path = f"artifacts.trained_models.{index}"
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("config_id"), str)
                or not item["config_id"]
            ):
                self.issue(path, "model_entry", "Нужен идентификатор варианта")
                continue
            if item["config_id"] in ids:
                self.issue(path, "duplicate_config", "Повтор варианта в индексе")
            ids.add(item["config_id"])
            configuration: dict[str, Any] = {}
            for key in ("configuration_sha256", "model_sha256", "feature_schema_sha256"):
                content = self.read_hash(item.get(key), f"{path}.{key}")
                if key == "configuration_sha256" and content is not None:
                    try:
                        value = strict_json_loads(content)
                        if (
                            not isinstance(value, dict)
                            or value.get("config_id") != item["config_id"]
                        ):
                            raise ValueError("Конфигурация не связана с вариантом")
                        configuration = value
                    except (ValueError, TypeError) as exc:
                        self.issue(path, "model_config_binding", str(exc))
                if key == "feature_schema_sha256" and content is not None:
                    try:
                        feature = strict_json_loads(content)
                        self.issues.extend(validate_feature_schema(feature))
                        if feature.get("config_id") != item["config_id"] or not feature.get(
                            "columns"
                        ):
                            self.issue(
                                path,
                                "model_schema_binding",
                                "Схема не относится к данному варианту",
                            )
                    except (ValueError, AttributeError, TypeError) as exc:
                        self.issue(path, "model_schema", str(exc))
            if item.get("training_status") != "passed":
                self.issue(path, "training_failed", "Обучение варианта не завершено", "gate")
            count = item.get("actual_tree_count")
            maximum = _at(configuration, "ranker.resolved_parameters.training.num_boost_round")
            if type(count) is not int or type(maximum) is not int or not 0 < count <= maximum:
                self.issue(
                    path, "tree_count", "Нужно фактическое число деревьев в пределах бюджета"
                )

    def check_variant_load(self, report: dict[str, Any], path: str) -> None:
        if report.get("load_profile") != self.m["serving"]:
            self.issue(
                path, "load_profile_binding", "Измерения выполнены на другом нагрузочном профиле"
            )
        if report.get("replay_mode") != self.m["operations"]["replay_mode"]:
            self.issue(path, "replay_binding", "В отчёте не закреплён режим воспроизведения")
        trials = report.get("trials")
        thresholds = {
            "p95_ms": 50,
            "p99_ms": 100,
            "error_fraction": 0.001,
            "fallback_fraction": 0.01,
            "peak_ram_gib": 28,
        }
        if not isinstance(trials, list) or len(trials) != 3:
            self.issue(path, "load_trials", "Вариант требует три нагрузочных прогона")
            return
        for trial in trials:
            if not isinstance(trial, dict) or not all(key in trial for key in thresholds):
                self.issue(path, "trial_fields", "Неполные измерения варианта")
            elif any(
                not _positive(trial[key], zero=True) or trial[key] > limit
                for key, limit in thresholds.items()
            ):
                self.issue(path, "load_failed", "Вариант нарушил нагрузочные пределы", "gate")
        for key in ("replay_passed", "recovery_passed", "input_limits_passed", "integrity_passed"):
            if report.get(key) is not True:
                self.issue(path, "serve_check", "Вариант не прошёл приёмку обслуживания", "gate")
        if (
            not _positive(report.get("rto_seconds"), zero=True)
            or report["rto_seconds"] > 60
            or report.get("rpo") != 0
        ):
            self.issue(path, "recovery_failed", "Вариант нарушил RTO/RPO", "gate")

    def later_stages(self) -> None:
        for name, due in (("data_status", 0), ("math_status", 1)):
            if self.index >= due and self.m["verification"][name] != "passed":
                self.issue(
                    f"verification.{name}",
                    "required_status",
                    "Стадия не получила необходимый системный статус",
                    "gate",
                )
        if self.index >= 1:
            models = self.read_hash(
                self.need("artifacts.trained_models_sha256"),
                "artifacts.trained_models_sha256",
                envelope=True,
            )
            self.model_index(models)
        if self.stage in SERVICE_STAGES:
            if self.m["verification"]["serve_status"] != "passed":
                self.issue(
                    "verification.serve_status",
                    "required_status",
                    "Обслуживание не прошло приёмку",
                    "gate",
                )
            for key in (
                "cpu_model_and_threads",
                "ram_meter_sampling_interval_ms",
                "ram_process_scope",
            ):
                self.need(f"serving.{key}")
            if not _positive(self.m["serving"]["ram_meter_sampling_interval_ms"]):
                self.issue(
                    "serving.ram_meter_sampling_interval_ms",
                    "ram_meter",
                    "Нужен положительный период замера RAM",
                )
            load = self.evidence("load")
            if load:
                self.check_variant_load(load, "evidence.load")
        if self.index >= 3:
            configs = self.need("selection.registered_configurations")
            plan = self.evidence("selection_plan")
            if configs and len({c["config_id"] for c in configs}) != len(configs):
                self.issue(
                    "selection.registered_configurations", "duplicate_config", "Повтор config_id"
                )
            if plan:
                for key in ("configurations", "budget", "control", "tie_break"):
                    if not plan.get(key):
                        self.issue(
                            f"evidence.selection_plan.{key}", "selection_plan", "Не заполнен план"
                        )
                if (
                    plan.get("configurations") != configs
                    or plan.get("tie_break") != self.m["selection"]["tie_break"]
                ):
                    self.issue(
                        "evidence.selection_plan", "selection_binding", "План не совпал с паспортом"
                    )
                if plan.get("control") != "gcs_rk_v2_7":
                    self.issue(
                        "evidence.selection_plan.control",
                        "control_required",
                        "Контроль GCS+Rk должен входить в план",
                    )
            if isinstance(configs, list):
                models = self.read_hash(
                    self.m["artifacts"]["trained_models_sha256"], "trained_models", envelope=True
                )
                registered = {c["config_id"]: c for c in configs}
                if "gcs_rk_v2_7" not in registered:
                    self.issue(
                        "selection.registered_configurations",
                        "control_required",
                        "Нет контрольного варианта",
                    )
                entries = models.get("configurations") if isinstance(models, dict) else None
                if isinstance(entries, list):
                    trained = {
                        c["config_id"]: c
                        for c in entries
                        if isinstance(c, dict) and isinstance(c.get("config_id"), str)
                    }
                    for identifier, configuration in registered.items():
                        item = trained.get(identifier)
                        if (
                            item is None
                            or item.get("configuration_sha256")
                            != configuration["configuration_sha256"]
                        ):
                            self.issue(
                                f"selection.{identifier}",
                                "untrained_config",
                                "Нет связанной обученной модели варианта",
                            )
                            continue
        if self.index >= 4:
            self.pre_d()
        if self.index < 5:
            if (
                self.m["verification"]["offline_status"] != "not_run"
                or self.m["artifacts"]["final_evaluation_report_sha256"] is not None
                or self.m["evidence"].get("final_evaluation") is not None
            ):
                self.issue("verification.offline_status", "future_D", "До post_D результатов D нет")
        else:
            report = self.evidence("final_evaluation")
            if self.m["artifacts"]["final_evaluation_report_sha256"] != self.m["evidence"].get(
                "final_evaluation"
            ):
                self.issue(
                    "artifacts.final_evaluation_report_sha256",
                    "report_binding",
                    "Разные ссылки на итоговый отчёт",
                )
            if report:
                self.issues.extend(validate_final_evaluation(report, self.m))
                for key in (
                    "metrics",
                    "intervals",
                    "observation",
                    "metadata_sensitivity",
                    "statuses",
                ):
                    if not report.get(key):
                        self.issue(
                            f"evidence.final_evaluation.{key}",
                            "final_report",
                            "Неполный итоговый отчёт",
                        )
                statuses = report.get("statuses", {})
                if not isinstance(statuses, dict):
                    self.issue(
                        "evidence.final_evaluation.statuses",
                        "report_type",
                        "Статусы должны быть объектом",
                    )
                    statuses = {}
                if statuses.get("offline_status") != self.m["verification"]["offline_status"]:
                    self.issue(
                        "verification.offline_status",
                        "status_binding",
                        "Статус расходится с отчётом",
                    )
                if self.m["verification"]["offline_status"] != "passed":
                    self.issue(
                        "verification.offline_status",
                        "offline_unfit",
                        "Оценка не прошла приёмку",
                        "gate",
                    )
        if self.index >= 6:
            self.production()
        if self.index < 7 and self.m["verification"]["business_status"] != "not_observed":
            self.issue(
                "verification.business_status", "future_AB", "До post_AB эффект не наблюдался"
            )
        if self.index < 7 and self.m["evidence"].get("online_results") is not None:
            self.issue(
                "evidence.online_results", "future_AB", "До post_AB исходы A/B не записываются"
            )
        if self.index == 7:
            report = self.evidence("online_results")
            if report:
                self.issues.extend(validate_online_results(report, self.m))
                for key in ("mature", "integrity_passed", "itt_complete"):
                    if report.get(key) is not True:
                        self.issue(
                            f"evidence.online_results.{key}",
                            "online_condition",
                            "Не подтверждено условие анализа",
                            "gate",
                        )
                if (
                    "stopping_status" not in report
                    or "assigned_counts" not in report
                    or "outcomes" not in report
                ):
                    self.issue(
                        "evidence.online_results",
                        "online_fields",
                        "Нет сведений о назначениях/исходах/остановках",
                    )
                if report.get("business_status") != self.m["verification"]["business_status"]:
                    self.issue(
                        "verification.business_status",
                        "status_binding",
                        "Статус расходится с отчётом",
                    )
                if (
                    report.get("stopping_status") != "completed"
                    and report.get("business_status") == "passed"
                ):
                    self.issue(
                        "evidence.online_results",
                        "stopped_confirmatory",
                        "Остановленный опыт не подтверждающий",
                    )
                if report.get("business_status") != "passed":
                    self.issue(
                        "verification.business_status",
                        "business_unfit",
                        "Нет положительного бизнес-решения",
                        "gate",
                    )

    def pre_d(self) -> None:
        selected = self.read_hash(
            self.need("artifacts.selected_policies_sha256"),
            "artifacts.selected_policies_sha256",
            envelope=True,
        )
        selection = self.read_hash(
            self.need("artifacts.selection_report_sha256"),
            "artifacts.selection_report_sha256",
            envelope=True,
        )
        pair = self.m["selection"]["primary_comparison"]
        identifiers = (
            [pair["control"], pair["treatment"]]
            if self.m["inference"]["primary_estimate_type"] == "paired" and pair
            else [self.m["inference"]["primary_policy_config_id"]]
        )
        registered = {
            c["config_id"] for c in self.m["selection"]["registered_configurations"] or []
        }
        if not set(identifiers) <= registered:
            self.issue(
                "selection", "unregistered_policy", "Выбранная политика не входила в принятый план"
            )
        if selected and selected.get("policies") != identifiers:
            self.issue(
                "artifacts.selected_policies_sha256",
                "selected_binding",
                "Выбранные политики не совпали с планом D",
            )
        if selection and (
            selection.get("policies") != identifiers
            or selection.get("selection_plan_sha256") != self.m["evidence"].get("selection_plan")
            or selection.get("status") != "passed"
        ):
            self.issue(
                "artifacts.selection_report_sha256",
                "selection_binding",
                "Нет принятого выбора по зарегистрированному плану",
            )
        plan = self.evidence("evaluation_plan")
        if plan and (
            plan.get("inference") != self.m["inference"]
            or plan.get("primary_comparison") != self.m["selection"]["primary_comparison"]
        ):
            self.issue(
                "evidence.evaluation_plan", "evaluation_binding", "План не совпал с паспортом"
            )
        for key in (
            "calendar_anchor_utc",
            "calendar_cell_seconds",
            "block_cells",
            "planned_cells",
            "sensitivity_block_cells",
        ):
            self.need(f"inference.{key}")
        inf = self.m["inference"]
        try:
            left = _time(self.m["periods"]["period_D"]["start_utc"])
            right = _time(self.m["periods"]["period_D"]["end_utc"])
            support = (
                (right - left).total_seconds()
                - self.m["task"]["horizon_seconds"]
                - self.m["task"]["inactivity_seconds"]
            )
            cell = inf["calendar_cell_seconds"]
            if (
                _time(inf["calendar_anchor_utc"]) != left
                or not cell
                or support % cell != 0
                or inf["planned_cells"] != support // cell
            ):
                raise ValueError("Сетка не соответствует окну допустимых стартов")
            if not inf["block_cells"] or inf["planned_cells"] // inf["block_cells"] < 20:
                raise ValueError("Недостаточно полных блоков")
            if any(x > inf["planned_cells"] for x in inf["sensitivity_block_cells"]):
                raise ValueError("Блок чувствительности длиннее сетки")
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            self.issue("inference", "calendar_grid", str(exc))

    def production(self) -> None:
        self.issues.extend(validate_online_plan(self.m))
        if self.m["online_experiment"]["enabled"] is not True:
            self.issue("online_experiment.enabled", "online_disabled", "A/B не включён")
        for name in ("data_status", "math_status", "offline_status", "serve_status"):
            if self.m["verification"][name] != "passed":
                self.issue(
                    f"verification.{name}", "pilot_required", "Не выполнен Ready_pilot", "gate"
                )
        if self.m["operations"]["artifact_authentication"] != "signed":
            self.issue(
                "operations.artifact_authentication",
                "signature_required",
                "Локальный checksum не разрешает пилот",
                "gate",
            )
        for path in (
            "operations.production_tenant_limits",
            "operations.production_sku_contribution_limits",
            "operations.production_event_time_range_utc",
            "operations.trusted_production_key_ids",
            "operations.production_block_aggregator_id",
            "operations.responsible_role",
            "operations.analysis_close_utc",
            "operations.monitoring_resolved_configuration",
            "online_experiment.protocol_sha256",
            "online_experiment.assignment_unit",
            "online_experiment.estimand",
            "online_experiment.business_threshold",
            "online_experiment.sample_size_and_horizon",
            "online_experiment.srm_plan",
            "online_experiment.guardrail_margins",
        ):
            self.need(path)
        self.read_hash(
            self.need("operations.production_owner_registry_sha256"),
            "operations.production_owner_registry_sha256",
        )
        self.read_hash(
            self.m["online_experiment"]["protocol_sha256"], "online_experiment.protocol_sha256"
        )
        srm = self.m["online_experiment"]["srm_plan"]
        required = self.template["online_experiment"]["srm_defaults"]["required_plan_fields"]
        if not isinstance(srm, dict) or not all(key in srm for key in required):
            self.issue("online_experiment.srm_plan", "srm_plan", "Не заполнен полный план SRM")
        elif srm.get("assignment_mechanism") == "independent_bernoulli" and srm.get(
            "membership_uses_assignment_or_exposure_history", False
        ):
            self.issue(
                "online_experiment.srm_plan",
                "adaptive_srm",
                "Нужен отдельный тест адаптивного состава",
            )
        report = self.evidence("production")
        if report:
            for key in (
                "signature_verified",
                "access_and_retention_verified",
                "limits_verified",
                "monitoring_calibrated",
            ):
                if report.get(key) is not True:
                    self.issue(
                        f"evidence.production.{key}",
                        "production_required",
                        "Не выполнено условие рабочего профиля",
                        "gate",
                    )
            if report.get("profile") != "production":
                self.issue(
                    "evidence.production.profile",
                    "production_profile",
                    "Нужен рабочий профиль",
                    "gate",
                )
        protocol = self.evidence("online_protocol")
        if protocol:
            expected = deepcopy(self.m["online_experiment"])
            expected.pop("protocol_sha256")
            if (
                protocol.get("plan") != expected
                or self.m["evidence"].get("online_protocol")
                != self.m["online_experiment"]["protocol_sha256"]
            ):
                self.issue(
                    "evidence.online_protocol",
                    "protocol_binding",
                    "План не совпал с хешированным протоколом",
                )


def validate_manifest(
    manifest: dict[str, Any],
    stage: Stage | str,
    artifacts: LocalArtifactStore,
    *,
    plan_references: Mapping[str, str] | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
    operation: str | None = None,
) -> ValidationResult:
    structure = structure_issues(manifest)
    try:
        stage_name = Stage(stage).value
    except ValueError:
        structure += (ValidationIssue("stage", "unknown_stage", "Неизвестная стадия", "structure"),)
        stage_name = "preflight"
    if structure:
        return ValidationResult(False, False, False, False, structure)
    if operation not in {None, "evaluate"}:
        return ValidationResult(
            True,
            False,
            False,
            False,
            (ValidationIssue("operation", "unknown_operation", "Неизвестная операция"),),
        )
    validation = _Validation(
        manifest, stage_name, artifacts, plan_references or {}, input_paths, operation
    )
    validation.policy()
    validation.preflight()
    validation.results()
    validation.plans()
    validation.later_stages()
    issues = tuple(validation.issues)
    semantic_valid = not any(x.layer == "semantic" for x in issues)
    results_complete = not any(x.layer == "results" for x in issues)
    return ValidationResult(
        True,
        semantic_valid,
        results_complete,
        semantic_valid and results_complete and not issues,
        issues,
    )
