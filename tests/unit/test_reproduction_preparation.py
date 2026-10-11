"""Независимая сверка новых общих таблиц без прежних запусков."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info
from coveo_recsys.verification.preparation import _session, _vector, verify_preparation
from tests.integration import test_actual_preparation as cases

ROOT, actual_inputs, run = cases.ROOT, cases.actual_inputs, cases.run


def verify(inputs, destination, **kwargs):
    source, profile, plan, _ = inputs
    return verify_preparation(
        ROOT,
        destination,
        source,
        profile,
        plan,
        max_seconds=60,
        max_memory_bytes=2 * 1024**3,
        **kwargs,
    )


@pytest.mark.parametrize("actual_inputs", [("A_G", 2), ("B", 3)], indirect=True)
def test_new_preparation_is_verified_without_previous_runs(actual_inputs, tmp_path):
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    result = verify(actual_inputs, destination)
    assert result["status"] == "passed"
    assert result["published_rows_complete"] and result["registered_targets_checked"]
    assert result["telemetry_completeness"] == "unknown"
    report = json.loads((destination / "reports/preparation.json").read_bytes())
    assert result["preparation_report_sha256"] == sha256_bytes(canonical_json_bytes(report))
    assert result["preparation_plan_sha256"] == sha256_file(actual_inputs[2])
    assert result["session_count"] == report["N_obs"]
    assert result["source_rows_verified"] == sum(report["composition"]["source_rows"].values())
    assert len(result["raw_scans"]) == 2


def change_table(destination, relative, kind, mutate):
    path = destination / relative
    rows = list(iter_sessions(path, kind))
    path.unlink()
    with TableWriter(path, kind) as writer:
        changed = False
        for sid, records in rows:
            output = []
            for record in records:
                value = mutate(record) if not changed else record
                changed |= value != record
                output.append(value)
            writer.append(sid, output)
    assert changed
    report_path = destination / "reports/preparation.json"
    report = json.loads(report_path.read_bytes())
    old = report["tables"][relative]
    report["tables"][relative] = table_info(path, kind, old["period"], old["data_role"])
    report_path.write_bytes(canonical_json_bytes(report))


@pytest.mark.parametrize("mutation", ["target", "prefix", "raw_origin", "search"])
def test_semantic_corruption_is_rejected_even_with_updated_table_hash(
    actual_inputs, tmp_path, mutation
):
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    if mutation == "target":
        relative, kind = "datasets/A_G/targets.parquet", TargetRecord

        def mutate(row):
            return replace(row, purchases=("wrong-sku",)) if row.purchases else row

    elif mutation == "prefix":
        relative, kind = "datasets/A_G/prefixes.parquet", PreparedRequest

        def mutate(row):
            return replace(row, prefix=replace(row.prefix, counters=(("detail", "wrong", 1),)))

    else:
        relative, kind = "canonical/A_G.parquet", CanonicalBlock

        def mutate(row):
            if mutation == "raw_origin":
                event = row.events[0]
                origin = event.origins[0].rsplit("=", 1)[0] + "=9999"
                event = replace(event, origins=(origin,), provenance=origin)
                return replace(row, events=(event, *row.events[1:]))
            if row.search_payloads:
                value = replace(row.search_payloads[0], vector=(0.0, 1.0, *([0.0] * 48)))
                return replace(row, search_payloads=(value,))
            return row

    change_table(destination, relative, kind, mutate)
    with pytest.raises(InputValidationError):
        verify(actual_inputs, destination)


@pytest.mark.parametrize("kind", ["raw", "profile", "plan", "table"])
def test_changed_input_is_rejected(actual_inputs, tmp_path, kind):
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    paths = {
        "raw": actual_inputs[0] / "browsing_train.csv",
        "profile": actual_inputs[1],
        "plan": actual_inputs[2],
        "table": destination / "canonical/A_G.parquet",
    }
    with paths[kind].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(InputValidationError):
        verify(actual_inputs, destination)


def test_session_intersection_with_other_preparation_is_rejected(actual_inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run(actual_inputs, first)
    run(actual_inputs, second)
    with pytest.raises(InputValidationError, match="Пересечение"):
        verify(actual_inputs, second, other_preparations=(first,))


@pytest.mark.parametrize("limit", ["max_seconds", "max_memory_bytes"])
def test_verification_does_not_turn_resource_limit_into_success(actual_inputs, tmp_path, limit):
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    limits = {"max_seconds": 60, "max_memory_bytes": 2 * 1024**3}
    limits[limit] = 1e-12 if limit == "max_seconds" else 1
    with pytest.raises(InputValidationError, match="Предел"):
        verify_preparation(ROOT, destination, *actual_inputs[:3], **limits)


@pytest.mark.parametrize("actual_inputs", [("B", 3)], indirect=True)
@pytest.mark.parametrize("boundary", ["horizon", "ag_tail"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_calendar_admission_requires_strict_boundaries(actual_inputs, tmp_path, boundary, delta):
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    plan = from_json(actual_inputs[2].read_bytes(), PreparationPlan)
    report = json.loads((destination / "reports/preparation.json").read_bytes())
    kinds = (CanonicalBlock, PreparedRequest, TargetRecord, SessionRecord)
    paths = (
        "canonical/B.parquet",
        "datasets/B/prefixes.parquet",
        "datasets/B/targets.parquet",
        "preparation/sessions.parquet",
    )
    tables = [dict(iter_sessions(destination / path, kind)) for path, kind in zip(paths, kinds)]
    sid = max(tables[3], key=lambda key: tables[3][key][0].last_event_ms)
    blocks, (request,), (target,), (session,) = (table[sid] for table in tables)
    if boundary == "horizon":
        end = request.t_q + plan.task.horizon_ms + plan.task.gap_ms + delta
        period = SimpleNamespace(name="B", start_ms=request.t_q, end_ms=end)
    else:
        end = session.last_event_ms + plan.task.gap_ms + delta
        period = SimpleNamespace(name="A_G", start_ms=request.t_q, end_ms=end)
        session = replace(session, period="A_G")
    args = (blocks, request, target, session, report["sampled"]["selected_rows"][sid], plan, period)
    if delta > 0:
        _session(*args)
    else:
        with pytest.raises(InputValidationError, match="временной допуск"):
            _session(*args)


@pytest.mark.parametrize("search_kind", ["normal", "zero", "ambiguous"])
def test_search_before_first_view_is_checked_independently(actual_inputs, tmp_path, search_kind):
    source, profile_path, _, selected = actual_inputs
    profile = json.loads(profile_path.read_bytes())
    vector = [1.0] + [0.0] * 49 if search_kind != "zero" else [0.0] * 50
    rows = [[selected[1], json.dumps(vector), "[]", "[]", 90]]
    if search_kind == "ambiguous":
        rows.append([selected[1], json.dumps([0.0, 1.0] + [0.0] * 48), "[]", "[]", 90])
    path = source / "search_train.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(profile["files"][path.name]["columns"])
        writer.writerows(rows)
    profile["files"][path.name]["expected_sha256"] = sha256_file(path)
    profile["files"][path.name]["expected_size_bytes"] = path.stat().st_size
    profile_path.write_bytes(canonical_json_bytes(profile))
    destination = tmp_path / "new"
    run(actual_inputs, destination)
    assert verify(actual_inputs, destination)["status"] == "passed"


@pytest.mark.parametrize("field", ["detail_history", "full_last_detail"])
@pytest.mark.parametrize("mutation", ["duplicate", "reverse"])
def test_detail_history_requires_exact_ordered_records(actual_inputs, tmp_path, field, mutation):
    source, profile_path, _, selected = actual_inputs
    path = source / "browsing_train.csv"
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow([selected[1], "event_product", "detail", "sku-z", 100, "url"])
    profile = json.loads(profile_path.read_bytes())
    profile["files"][path.name]["expected_sha256"] = sha256_file(path)
    profile["files"][path.name]["expected_size_bytes"] = path.stat().st_size
    profile_path.write_bytes(canonical_json_bytes(profile))
    destination = tmp_path / "new"
    run(actual_inputs, destination)

    def mutate(row):
        history = getattr(row.prefix, field)
        if len(history) < 2:
            return row
        changed = (*history, history[0]) if mutation == "duplicate" else history[::-1]
        return replace(row, prefix=replace(row.prefix, **{field: changed}))

    change_table(destination, "datasets/A_G/prefixes.parquet", PreparedRequest, mutate)
    with pytest.raises(InputValidationError, match="История"):
        verify(actual_inputs, destination)


def test_unrepresentable_integer_is_an_invalid_vector():
    assert _vector([10**400], 1) == (None, "vector_invalid")
