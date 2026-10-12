"""Фактические результаты pytest для допуска B и C_smoke, без старых запусков."""

import base64
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.config import load_registry, required_checks
from coveo_recsys.config.manifest import _enabled
from coveo_recsys.verification.provenance import source_inventory

# Части реестра, применимые к базовому пути до C_smoke; обслуживание сюда не входит.
PARTS = {
    "B01": ("B01.data", "B01.features"),
    "B02": ("B02.context", "B02.features"),
    "B03.data": ("B03.data",),
    "B04": ("B04.graph",),
    "B05": ("B05.numerics", "B05.ranker", "B05.content", "B05.features"),
    "B06": ("B06",),
    "B07": ("B07.evaluation",),
    "B08": ("B08.evaluation",),
    "B09": ("B09",),
    "B10.data": ("B10.periods", "B10.features", "B10.ranker"),
    "B10.ranker": ("B10.ranker", "B10.features", "B10.evaluation"),
    "B11": ("B11",),
    "B12": ("B12",),
    "B13": ("B13.preparation",),
}
WINDOWS_SYMLINK_REASON = "windows_symlink_unavailable: WinError 1314"
WINDOWS_SYMLINK_TESTS = {
    ("tests.unit.test_e1_access", "test_access_rejects_link_outside_root"),
    ("tests.unit.test_prepared_tables", "test_writer_rejects_symlink_before_creating_files"),
    ("tests.unit.test_e6_state_store", "test_file_rejects_symlink"),
}


@dataclass(frozen=True)
class TestEvidence:
    cases: dict[tuple[str, str], str]
    property_counts: dict[str, int]
    summary: dict[str, Any]


def _counts(cases: Mapping[tuple[str, str], str]) -> dict[str, int]:
    counts = Counter(cases.values())
    return {
        "collected": len(cases),
        **{name: counts[name] for name in ("passed", "failed", "skipped", "errors")},
    }


def read_test_evidence(
    root: Path, junit_path: Path, source_before: Mapping[str, str]
) -> TestEvidence:
    """source_before снимается вызывающей командой непосредственно перед pytest.

    Команда должна дождаться pytest и вызвать эту функцию в том же шаге. Снимок
    проверяет изменения содержимого; время JUnit исключает прежний запуск.
    """
    inventory = source_inventory(root)
    if not inventory or dict(source_before) != inventory:
        raise ValueError("Исходники изменились во время pytest или их снимок отсутствует")
    latest_source = max((root / path).stat().st_mtime for path in inventory)
    payload = junit_path.read_bytes()
    tree = ET.fromstring(payload)
    suites = list(tree.iter("testsuite"))
    if not suites:
        raise ValueError("Нет testsuite в JUnit")
    cases: dict[tuple[str, str], str] = {}
    property_counts: dict[str, int] = {}
    unsuccessful = []
    for suite in suites:
        started = datetime.fromisoformat(suite.attrib["timestamp"])
        if started.tzinfo is None or started.timestamp() < latest_source:
            raise ValueError("Исходники новее JUnit или его время не содержит часового пояса")
        local = {}
        for case in suite.findall("testcase"):
            key = case.attrib.get("classname", ""), case.attrib.get("name", "")
            if not all(key) or key in cases:
                raise ValueError(f"Пустой или повторный testcase JUnit: {key}")
            results = [(tag, case.find(tag)) for tag in ("failure", "error", "skipped")]
            results = [(tag, result) for tag, result in results if result is not None]
            if len(results) > 1:
                raise ValueError(f"Несколько исходов testcase JUnit: {key}")
            status = "passed"
            if results:
                tag, result = results[0]
                assert result is not None
                status = {"failure": "failed", "error": "errors", "skipped": "skipped"}[tag]
                unsuccessful.append(
                    dict(
                        classname=key[0],
                        name=key[1],
                        status=status,
                        message=result.get("message", ""),
                    )
                )
            cases[key] = local[key] = status
        counts = _counts(local)
        for attribute, count_key in (
            ("tests", "collected"),
            ("failures", "failed"),
            ("errors", "errors"),
            ("skipped", "skipped"),
        ):
            if int(suite.attrib[attribute]) != counts[count_key]:
                raise ValueError(f"Счётчики JUnit не совпали: {attribute}")
        for prop in suite.findall("properties/property"):
            name = prop.attrib.get("name", "")
            if not name.startswith("hypothesis-statistics-"):
                continue
            nodeid = name.removeprefix("hypothesis-statistics-")
            if nodeid in property_counts:
                raise ValueError(f"Повтор статистики Hypothesis: {nodeid}")
            text = base64.b64decode(prop.attrib["value"], validate=True).decode("utf-8")
            passing = re.findall(r"(\d+) passing examples", text)
            property_counts[nodeid] = max(map(int, passing), default=0)
    if not cases:
        raise ValueError("JUnit не содержит выполненных тестов")
    return TestEvidence(
        cases,
        property_counts,
        {
            "source_sha256": sha256_bytes(canonical_json_bytes(inventory)),
            "junit_sha256": sha256_bytes(payload),
            "counts": _counts(cases),
            "property_counts": property_counts,
            "unsuccessful_cases": unsuccessful,
        },
    )


def stage_test_rows(
    manifest: Mapping[str, Any],
    stage: str,
    evidence: TestEvidence,
    references: str,
    *,
    operation: str | None = None,
) -> list[dict[str, Any]]:
    """Состав проверок — из реестра, исходы и счётчики — только из JUnit."""
    if stage not in {"preflight", "post_training"}:
        raise ValueError("Не поддерживается стадия вне базового пути B/C_smoke")
    platform_skips = {
        (case["classname"], case["name"])
        for case in evidence.summary["unsuccessful_cases"]
        if sys.platform == "win32"
        and case["status"] == "skipped"
        and case["message"] == WINDOWS_SYMLINK_REASON
        and (case["classname"], case["name"].split("[", 1)[0]) in WINDOWS_SYMLINK_TESTS
        and evidence.cases.get((case["classname"], case["name"])) == "skipped"
    }
    registry = load_registry()["implemented_checks"]
    rows = []
    for definition in required_checks(manifest, stage, operation=operation):
        identifier = definition["test_id"]
        if not _enabled(manifest, definition["enabled_when"]):
            rows.append(
                dict(
                    test_id=identifier,
                    status="not_used",
                    reason=f"Модуль {definition['enabled_when']} выключен",
                )
            )
            continue
        if identifier not in PARTS:
            raise ValueError(f"Нет покрытия проверки: {identifier}")
        parts = PARTS[identifier]
        paths = sorted({path for part in parts for path in registry[part]["paths"]})
        selected = {}
        for path in paths:
            module = path.removesuffix(".py").replace("/", ".")
            matching = {
                key: value
                for key, value in evidence.cases.items()
                if key[0] == module or key[0].startswith(module + ".")
            }
            if not matching:
                raise ValueError(f"Нет исполненных проверок {identifier}: {path}")
            selected.update(matching)
            if path.startswith("tests/properties/"):
                for classname, name in matching:
                    class_suffix = classname.removeprefix(module).replace(".", "::")
                    nodeid = f"{path}{class_suffix}::{name}"
                    if evidence.property_counts.get(nodeid, 0) < 1000:
                        raise ValueError(f"Нет 1000 выполненных случаев Hypothesis: {nodeid}")
        excluded = sorted(selected.keys() & platform_skips)
        counts = _counts(
            {key: status for key, status in selected.items() if key not in platform_skips}
        )
        if counts["passed"] != counts["collected"] or not counts["collected"]:
            raise ValueError(f"Неполное исполнение {identifier}: {counts}")
        row = dict(
            test_id=identifier,
            status="passed",
            complete_test=True,
            exact_fields_passed=True,
            counts=counts,
            covered_parts=definition["parts"] or list(parts),
            reason=f"Части {list(parts)}, JUnit SHA-256 {evidence.summary['junit_sha256']}; {references}",
        )
        if excluded:
            row["reason"] += (
                f"; {WINDOWS_SYMLINK_REASON}: не выполнены и исключены из нормативных счётчиков "
                + ", ".join(f"{module}::{name}" for module, name in excluded)
                + "; исходные skipped сохранены в JUnit и сводке"
            )
        if identifier == "B05":
            row["derivative_tolerances"] = {
                "callback_fp64": [1e-12, 1e-10],
                "callback_fp32": [1e-7, 1e-6],
                "one_step_tree": [1e-6, 0],
            }
        rows.append(row)
    return rows
