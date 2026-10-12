"""Структура JSON и формат схемы признаков, без запуска алгоритмов."""

from __future__ import annotations

import math
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from coveo_recsys.artifacts import strict_json_loads
from coveo_recsys.contracts.common import ValidationIssue
from coveo_recsys.contracts.retrieval import METADATA370_CONFIG_ID, NO_METADATA_CONFIG_ID

from .resources import load_schema, load_template, resource_bytes


def init_manifest(run_id: str, scope: str = "synthetic") -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Нужен непустой run_id")
    if scope not in {"synthetic", "actual", "offline_first_run", "production", "research"}:
        raise ValueError("Неизвестная область запуска")
    manifest = load_template()
    manifest.update(
        run_id=run_id,
        scope=scope,
        status="draft",
        modules=dict.fromkeys(("sampled", "sequence", "freshness", "optional", "ppr"), False),
        module_configurations=dict.fromkeys(("sampled", "sequence", "freshness", "ppr")),
        evidence={},
    )
    return manifest


def structure_issues(value: Any, schema_name: str = "manifest") -> tuple[ValidationIssue, ...]:
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FormatChecker())
    issues = [
        ValidationIssue(
            ".".join(map(str, error.absolute_path)), "schema", error.message, "structure"
        )
        for error in sorted(validator.iter_errors(value), key=lambda e: str(e.path))
    ]

    def finite(node: Any, path: str) -> None:
        if isinstance(node, float) and not math.isfinite(node):
            issues.append(
                ValidationIssue(path, "nonfinite", "NaN/Inf запрещён в JSON", "structure")
            )
        elif isinstance(node, dict):
            for key, item in node.items():
                finite(item, f"{path}.{key}".strip("."))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                finite(item, f"{path}.{index}")

    finite(value, "")
    return tuple(issues)


def validate_feature_schema(value: Any) -> tuple[ValidationIssue, ...]:
    issues = list(structure_issues(value, "feature"))
    if issues:
        return tuple(issues)
    columns = value["columns"]
    if value["config_id"] == METADATA370_CONFIG_ID:
        fixed = strict_json_loads(resource_bytes("configs/feature_schema.json"))
        if value["schema_version"] != fixed["schema_version"] or columns != fixed["columns"]:
            issues.append(ValidationIssue("columns", "profile_schema", "Изменена схема 370"))
    if value["config_id"] == NO_METADATA_CONFIG_ID:
        fixed = strict_json_loads(resource_bytes("configs/feature_schema_no_metadata.json"))
        if value["schema_version"] != fixed["schema_version"] or columns != fixed["columns"]:
            issues.append(
                ValidationIssue("columns", "profile_schema", "Изменена схема без метаданных")
            )
    names = [column["name"] for column in columns]
    if len(set(names)) != len(names) or [c["position"] for c in columns] != list(
        range(len(columns))
    ):
        issues.append(
            ValidationIssue("columns", "column_order", "Имена уникальны; позиции идут от 0")
        )
    for index, column in enumerate(columns):
        if column["mask"] is not None and column["mask"] not in names:
            issues.append(
                ValidationIssue(f"columns.{index}.mask", "unknown_mask", "Нет столбца маски")
            )
        if column["name"] in {"item_id", "sku", "target", "label", "session_weight"}:
            issues.append(
                ValidationIssue(
                    f"columns.{index}.name",
                    "forbidden_feature",
                    "ID, будущая метка и вес не являются признаками",
                )
            )
    return tuple(issues)


def validate_stage_report(value: Any) -> tuple[ValidationIssue, ...]:
    schema = load_schema()
    validator = Draft202012Validator({"$ref": "#/$defs/stage_report", "$defs": schema["$defs"]})
    return tuple(
        ValidationIssue(
            "verification.report." + ".".join(map(str, error.absolute_path)),
            "report_schema",
            error.message,
        )
        for error in sorted(validator.iter_errors(value), key=lambda error: str(error.path))
    )
