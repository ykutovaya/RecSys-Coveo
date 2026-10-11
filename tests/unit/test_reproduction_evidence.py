import base64
import importlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from coveo_recsys.verification.provenance import source_inventory
from coveo_recsys.verification.test_evidence import read_test_evidence, stage_test_rows


@pytest.fixture
def evidence_inputs(tmp_path, monkeypatch):
    paths = ("tests/unit/test_rule.py", "tests/properties/test_rule.py")
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def test_rule(): pass\n", encoding="utf-8")
        created = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(target, (created, created))
    registry = {
        "implemented_checks": {"B01.data": {"paths": list(paths)}, "B01.features": {"paths": []}}
    }
    monkeypatch.setattr("coveo_recsys.verification.test_evidence.load_registry", lambda: registry)
    monkeypatch.setattr(
        "coveo_recsys.verification.test_evidence.required_checks",
        lambda *args, **kwargs: [dict(test_id="B01", enabled_when="always", parts=[])],
    )
    return tmp_path, source_inventory(tmp_path)


def write_junit(root, *, outcome=None, property_count=1000, timestamp=None):
    tree = ET.Element("testsuites")
    suite = ET.SubElement(
        tree,
        "testsuite",
        tests="2",
        failures=str(int(outcome == "failure")),
        errors=str(int(outcome == "error")),
        skipped=str(int(outcome == "skipped")),
        timestamp=(timestamp or datetime.now(UTC)).isoformat(),
    )
    props = ET.SubElement(suite, "properties")
    if property_count is not None:
        text = f"  - {property_count} passing examples, 0 failing examples, 0 invalid examples\n"
        ET.SubElement(
            props,
            "property",
            name="hypothesis-statistics-tests/properties/test_rule.py::test_rule",
            value=base64.b64encode(text.encode()).decode(),
        )
    for module in ("tests.unit.test_rule", "tests.properties.test_rule"):
        case = ET.SubElement(suite, "testcase", classname=module, name="test_rule")
        if outcome is not None and module.startswith("tests.unit"):
            ET.SubElement(case, outcome, message="Фактический отказ")
    path = root / "junit.xml"
    path.write_bytes(ET.tostring(tree))
    return path


def test_stage_rows_use_current_registry_counts_and_property_statistics(evidence_inputs):
    root, before = evidence_inputs
    evidence = read_test_evidence(root, write_junit(root), before)
    (row,) = stage_test_rows({}, "preflight", evidence, "Проверка исходных строк")
    assert row["status"] == "passed"
    assert row["counts"] == dict(collected=2, passed=2, failed=0, skipped=0, errors=0)
    assert evidence.summary["source_sha256"]
    assert evidence.summary["junit_sha256"]
    assert evidence.summary["property_counts"] == {"tests/properties/test_rule.py::test_rule": 1000}
    assert row["covered_parts"] == ["B01.data", "B01.features"]


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_unsuccessful_normative_case_is_retained_and_cannot_pass(evidence_inputs, outcome):
    root, before = evidence_inputs
    evidence = read_test_evidence(root, write_junit(root, outcome=outcome), before)
    assert evidence.summary["unsuccessful_cases"][0]["message"] == "Фактический отказ"
    with pytest.raises(ValueError, match="Неполное исполнение"):
        stage_test_rows({}, "preflight", evidence, "Проверка исходных строк")


@pytest.mark.parametrize("property_count", [None, 0, 999])
def test_missing_or_insufficient_property_statistics_do_not_prove_property(
    evidence_inputs, property_count
):
    root, before = evidence_inputs
    evidence = read_test_evidence(root, write_junit(root, property_count=property_count), before)
    with pytest.raises(ValueError, match="1000"):
        stage_test_rows({}, "preflight", evidence, "Проверка исходных строк")


@pytest.mark.parametrize("changed", ["source", "timestamp"])
def test_junit_cannot_be_bound_to_a_changed_producer(evidence_inputs, changed):
    root, before = evidence_inputs
    stamp = datetime.now(UTC) - timedelta(days=1) if changed == "timestamp" else None
    junit = write_junit(root, timestamp=stamp)
    if changed == "source":
        (root / "tests/unit/test_rule.py").write_text("changed = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Исходники"):
        read_test_evidence(root, junit, before)


@pytest.mark.parametrize("mutation", ["counts", "duplicate", "missing_module", "property_encoding"])
def test_malformed_or_incomplete_junit_is_rejected(evidence_inputs, mutation):
    root, before = evidence_inputs
    junit = write_junit(root)
    tree = ET.fromstring(junit.read_bytes())
    suite = tree.find("testsuite")
    if mutation == "counts":
        suite.set("tests", "3")
    elif mutation == "duplicate":
        suite.append(suite.find("testcase"))
        suite.set("tests", "3")
    elif mutation == "missing_module":
        suite.find("testcase").set("classname", "tests.unit.unrelated")
    else:
        suite.find("properties/property").set("value", "!bad!")
    junit.write_bytes(ET.tostring(tree))
    with pytest.raises(ValueError):
        evidence = read_test_evidence(root, junit, before)
        stage_test_rows({}, "preflight", evidence, "Проверка исходных строк")


def test_disabled_check_is_not_used_and_unsupported_stage_is_rejected(evidence_inputs, monkeypatch):
    root, before = evidence_inputs
    evidence = read_test_evidence(root, write_junit(root), before)
    monkeypatch.setattr(
        "coveo_recsys.verification.test_evidence.required_checks",
        lambda *args, **kwargs: [dict(test_id="X02.support", enabled_when="sampled", parts=[])],
    )
    (row,) = stage_test_rows({"modules": {"sampled": False}}, "preflight", evidence, "Данные")
    assert row["status"] == "not_used"
    with pytest.raises(ValueError, match="стадия"):
        stage_test_rows({}, "pre_D", evidence, "Данные")


def test_post_training_evaluation_includes_b12_and_derivative_tolerances(
    evidence_inputs, monkeypatch
):
    from coveo_recsys.verification.test_evidence import PARTS

    root, before = evidence_inputs
    evidence = read_test_evidence(root, write_junit(root), before)
    normative_parts = ["B05.numerics", "B05.derivatives", "B05.ranker"]

    def definitions(manifest, stage, *, operation):
        assert stage == "post_training" and operation == "evaluate"
        return [
            dict(test_id="B05", enabled_when="always", parts=normative_parts),
            dict(test_id="B12", enabled_when="always", parts=[]),
        ]

    monkeypatch.setattr("coveo_recsys.verification.test_evidence.required_checks", definitions)
    monkeypatch.setattr(
        "coveo_recsys.verification.test_evidence.load_registry",
        lambda: {
            "implemented_checks": {
                part: {"paths": ["tests/unit/test_rule.py"]}
                for identifier in ("B05", "B12")
                for part in PARTS[identifier]
            }
        },
    )
    rows = stage_test_rows({}, "post_training", evidence, "Данные", operation="evaluate")
    assert [row["test_id"] for row in rows] == ["B05", "B12"]
    assert rows[0]["covered_parts"] == normative_parts
    assert rows[0]["derivative_tolerances"] == {
        "callback_fp64": [1e-12, 1e-10],
        "callback_fp32": [1e-7, 1e-6],
        "one_step_tree": [1e-6, 0],
    }


SYMLINK_CASES = (
    ("tests.unit.test_e1_access", "test_access_rejects_link_outside_root"),
    (
        "tests.unit.test_prepared_tables",
        "test_writer_rejects_symlink_before_creating_files[root-table]",
    ),
    ("tests.unit.test_e6_state_store", "test_file_rejects_symlink"),
)
SYMLINK_REASON = "windows_symlink_unavailable: WinError 1314"


@pytest.mark.parametrize(
    "platform,outcome,message,identity,excluded",
    [
        *(("win32", "skipped", SYMLINK_REASON, case, True) for case in SYMLINK_CASES),
        ("linux", "skipped", SYMLINK_REASON, SYMLINK_CASES[0], False),
        ("win32", "skipped", "windows_symlink_unavailable: WinError 5", SYMLINK_CASES[0], False),
        ("win32", "skipped", "another skip", SYMLINK_CASES[0], False),
        ("win32", "skipped", SYMLINK_REASON, (SYMLINK_CASES[0][0], "test_another"), False),
        ("win32", "skipped", SYMLINK_REASON, ("tests.unit.other", SYMLINK_CASES[0][1]), False),
        ("win32", "failure", SYMLINK_REASON, SYMLINK_CASES[0], False),
    ],
)
def test_only_agreed_windows_symlink_skip_is_excluded_from_normative_counts(
    evidence_inputs, monkeypatch, platform, outcome, message, identity, excluded
):
    root, before = evidence_inputs
    junit = write_junit(root)
    tree = ET.fromstring(junit.read_bytes())
    suite = tree.find("testsuite")
    suite.set("tests", "3")
    suite.set("skipped" if outcome == "skipped" else "failures", "1")
    case = ET.SubElement(suite, "testcase", classname=identity[0], name=identity[1])
    ET.SubElement(case, outcome, message=message)
    junit.write_bytes(ET.tostring(tree))
    monkeypatch.setattr(
        "coveo_recsys.verification.test_evidence.sys",
        SimpleNamespace(platform=platform),
        raising=False,
    )
    monkeypatch.setattr(
        "coveo_recsys.verification.test_evidence.load_registry",
        lambda: {
            "implemented_checks": {
                "B01.data": {
                    "paths": ["tests/unit/test_rule.py", identity[0].replace(".", "/") + ".py"]
                },
                "B01.features": {"paths": ["tests/properties/test_rule.py"]},
            }
        },
    )
    evidence = read_test_evidence(root, junit, before)
    if not excluded:
        with pytest.raises(ValueError, match="Неполное исполнение"):
            stage_test_rows({}, "preflight", evidence, "Данные")
        return
    (row,) = stage_test_rows({}, "preflight", evidence, "Данные")
    assert row["counts"] == dict(collected=2, passed=2, failed=0, skipped=0, errors=0)
    assert identity[0] + "::" + identity[1] in row["reason"]
    assert SYMLINK_REASON in row["reason"]
    assert evidence.cases[identity] == "skipped"
    assert evidence.summary["counts"] == dict(collected=3, passed=2, failed=0, skipped=1, errors=0)
    assert evidence.summary["unsuccessful_cases"] == [
        dict(classname=identity[0], name=identity[1], status="skipped", message=message)
    ]


@pytest.mark.parametrize("identity", SYMLINK_CASES)
@pytest.mark.parametrize("platform,winerror", [("win32", 1314), ("win32", 5), ("linux", 1314)])
def test_symlink_checks_skip_only_windows_missing_privilege(
    tmp_path, monkeypatch, identity, platform, winerror
):
    module = importlib.import_module(identity[0])
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform), raising=False)
    error = OSError("symlink denied")
    error.winerror = winerror

    def denied(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "symlink_to", denied)
    kwargs = {"tmp_path": tmp_path}
    if "test_e1_access" in identity[0]:
        kwargs["prepared_access"] = (tmp_path, None, [], [], [])
    elif "test_prepared_tables" in identity[0]:
        kwargs.update(writer="table", position="root")
    function = getattr(module, identity[1].split("[", 1)[0])
    if platform == "win32" and winerror == 1314:
        with pytest.raises(pytest.skip.Exception, match=SYMLINK_REASON):
            function(**kwargs)
    else:
        with pytest.raises(OSError) as actual:
            function(**kwargs)
        assert actual.value is error
