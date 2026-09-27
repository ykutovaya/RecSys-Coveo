"""Малые общие таблицы: ответы закрыты до присоединения отдельных целей."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from coveo_recsys.contracts import InputValidationError, OutcomeStatus
from coveo_recsys.contracts.evaluation import POLICIES, EvaluationRecord, ReplayRecord
from coveo_recsys.contracts.preparation import (
    EVALUATION_PROTOCOL,
    PERIOD_NAMES,
    PREPARATION_VERSION,
    Coverage,
    Period,
    PreparationPlan,
    PreparedRequest,
    TaskConfig,
)
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestResult,
    SourceResult,
)
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.prepared import (
    FORMAT_VERSION,
    TableWriter,
    iter_sessions,
    table_info,
    write_report,
)
from coveo_recsys.evaluation import reporting
from coveo_recsys.evaluation.reporting import render_report, summarize_replay
from tests.reference.e1_periods_oracle import key as independent_session_key
from tests.reference.e3_cases import package, request


def evaluation_plan():
    estimates = (*POLICIES, "GCS+Rk-P", "GCS+Rk-R")
    return dict(
        period="C_smoke",
        policies=list(POLICIES),
        k=20,
        diagnostic_depth=10,
        intervals=dict(
            grid_step_ms=10,
            block_cells=2,
            sensitivity_block_cells=[1, 3],
            mc_error_method="binomial_order_statistic_95pct",
            dependence={name: dict(supported=None, assumptions=[]) for name in estimates},
            half_width_fraction=dict.fromkeys(estimates),
        ),
    )


def preparation_plan():
    original = from_json(Path("configs/synthetic_preparation.json").read_bytes(), PreparationPlan)
    return replace(
        original,
        source_namespace="fixture",
        diagnostic_sessions=0,
        task=TaskConfig(horizon_ms=100, gap_ms=10),
        coverage=Coverage("fixture", 0, 7000, (), "hand fixture", None),
        periods=tuple(
            Period(name, i * 1000, (i + 1) * 1000, 3 if name == "C_smoke" else 0)
            for i, name in enumerate(PERIOD_NAMES)
        ),
    )


def saved_inputs(tmp_path, *, unknown=False, mutation=None, fallback=False, answer_ids=None):
    source, destination = tmp_path / "prepared", tmp_path / "evaluation"
    plan = preparation_plan()
    requests, targets, answers = [], [], []
    for number, sid in enumerate(("a", "b", "c")):
        req = request(((3100 + number * 10, "detail", "a"),))
        blocks = tuple(
            replace(
                block,
                session_id=sid,
                events=tuple(replace(event, session_id=sid) for event in block.events),
            )
            for block in req.prefix.blocks
        )
        req = replace(
            req,
            session_id=sid,
            request_id="q-" + sid,
            prefix=replace(req.prefix, session_id=sid, blocks=blocks),
        )
        purchases = () if unknown or sid == "c" else ("b",) if sid == "a" else ("outside",)
        target = TargetRecord(
            req.request_id,
            purchases,
            OutcomeStatus.KNOWN if purchases else OutcomeStatus.UNKNOWN,
            100,
            None if purchases else "no_registered_purchase",
        )
        requests.append(req)
        targets.append(target)
        records = []
        for policy in POLICIES:
            branch = (
                Branch.FALLBACK
                if fallback and policy == "GCS+Rk" and sid == "b"
                else Branch.PRIMARY
            )
            candidates = CandidateResult(
                req.request_id,
                (SourceResult("popularity", ("a", "b"), (2.0, 1.0), True, 20, None),),
                ("a", "b"),
                (("a", ("popularity",)), ("b", ("popularity",))),
                branch,
                (),
            )
            delivered = ("b",) if policy == "GCS+Rk" else ("a",)
            result = RequestResult(
                req.request_id,
                "frozen-package",
                candidates,
                delivered,
                (1.0,),
                (),
                DeliveryStatus.DELIVERED,
                (),
                3,
                "answers:" + sid + ":" + policy,
            )
            diagnostic = dict(
                source_lists={
                    "popularity": dict(skus=["a", "b"], depth=10, truncated=False, reason=None)
                },
                feature_mask_counts={"x": 1}
                if policy == "GCS+Rk" and branch == Branch.PRIMARY
                else None,
                feature_rows=2 if policy == "GCS+Rk" and branch == Branch.PRIMARY else None,
                mask_reason=None
                if policy == "GCS+Rk" and branch == Branch.PRIMARY
                else "not_applicable",
            )
            if policy == "GCS+Rk":
                diagnostic["sensitivity"] = (
                    dict(status="not_measured", reason="fallback")
                    if branch == Branch.FALLBACK
                    else dict(
                        request_id=req.request_id,
                        scope="prefix_sensitivity_fixed_historical",
                        set_schema_sha256="a" * 64,
                        multiset_schema_sha256="b" * 64,
                        affected=True,
                        prefix_changed=True,
                        candidate_changed=True,
                        common_candidates=1,
                        changed_features=[
                            dict(name="x", changed_rows=1, mask_changed_rows=0, max_abs_delta=2.0)
                        ],
                        set_candidate_skus=["a", "b"],
                        multiset_candidate_skus=["a", "c"],
                    )
                )
            records.append(ReplayRecord(sid, policy, result, json.dumps(diagnostic)))
        if mutation == "request_id" and sid == "b":
            wrong = replace(records[0].result.candidates, request_id="wrong")
            records[0] = replace(
                records[0], result=replace(records[0].result, request_id="wrong", candidates=wrong)
            )
        if mutation == "policy_missing" and sid == "b":
            records.pop()
        if mutation == "policy_duplicate" and sid == "b":
            records[-1] = records[0]
        if mutation == "target_id" and sid == "b":
            targets[-1] = replace(target, request_id="foreign")
        if mutation == "late" and sid == "b":
            # Подмена времени производится целым валидным PreparedRequest.
            late = request(((3990, "detail", "a"),))
            late_blocks = tuple(
                replace(
                    block,
                    session_id=sid,
                    events=tuple(replace(event, session_id=sid) for event in block.events),
                )
                for block in late.prefix.blocks
            )
            requests[-1] = replace(
                late,
                session_id=sid,
                request_id=req.request_id,
                prefix=replace(late.prefix, session_id=sid, blocks=late_blocks),
            )
        answers.append((sid, records))
    tables = {}
    for name, kind, values, role in (
        ("prefixes", PreparedRequest, requests, "prefix"),
        ("targets", TargetRecord, targets, "targets"),
    ):
        path = f"datasets/C_smoke/{name}.parquet"
        with TableWriter(source / path, kind) as writer:
            for req, record in zip(requests, values, strict=True):
                writer.append(req.session_id, (record,))
        tables[path] = table_info(source / path, kind, "C_smoke", role)
    plan_json = to_json(plan).decode()
    prepared = dict(
        format_version=FORMAT_VERSION,
        status="completed",
        contract_version=PREPARATION_VERSION,
        evaluation_protocol=EVALUATION_PROTOCOL,
        telemetry_complete=None,
        scope="synthetic",
        real_preflight=False,
        run_id="old-producer",
        source_sha256="a" * 64,
        inputs={
            **dict.fromkeys(("browsing", "search", "products"), "b" * 64),
            "profile": sha256_bytes(b"{}"),
            "plan": sha256_bytes(plan_json.encode()),
        },
        profile_json="{}",
        plan_json=plan_json,
        tables=tables,
        selection={"C_smoke": [req.session_id for req in requests]},
        period_counts={"C_smoke": dict(eligible=5, admitted=3, selected=3)},
        exclusions={"no_admissible_request": 2},
        N_obs=3,
        N_eligible=5,
    )
    write_report(source, prepared)
    answers_path = destination / "replay/answers.parquet"
    with TableWriter(answers_path, ReplayRecord) as writer:
        for sid, records in answers:
            if answer_ids is not None and sid not in answer_ids:
                continue
            if mutation == "session_missing" and sid == "b":
                continue
            writer.append(sid, records)
    return source, prepared, answers_path, package(), plan, evaluation_plan(), destination


def test_closed_answers_join_targets_and_match_independent_three_session_counts(tmp_path):
    inputs = saved_inputs(tmp_path)
    checks = []
    summary = summarize_replay(*inputs, check_budget=lambda: checks.append(True))
    for policy in POLICIES:
        result = summary["policies"][policy]
        assert result["counts"] == dict(N_eligible=5, N_obs=3, N=2, N_unknown=1)
        assert result["metrics"]["Hit@20"]["value"] == (0.5 if policy == "GCS+Rk" else 0.0)
        assert result["cost"]["request_count"] == 3
        assert result["cost"]["elapsed_ms_total"] == 9
    assert summary["pairs"]["GCS+Rk-P"]["value"] == 0.5
    assert summary["pairs"]["GCS+Rk-R"]["value"] == 0.5
    assert summary["S_offline"] == "not_run" and summary["business_effect"] == "not_observed"
    assert summary["telemetry_completeness"] == "unknown"
    assert summary["preparation"]["run_id"] == "old-producer"
    assert summary["intervals"]["GCS+Rk"]["metric"]["status"] == "insufficient"
    sensitivity = summary["multiplicity_sensitivity"]
    assert sensitivity["measured_sessions"] == 3
    assert sensitivity["target_sessions"] == 2
    assert sensitivity["set_candidate_coverage"]["value"] == 0.5
    assert sensitivity["multiset_candidate_coverage"]["value"] == 0
    assert sensitivity["changed_features"]["x"]["changed_rows"] == 3
    rows = list(iter_sessions(inputs[-1] / "replay/sessions.parquet", EvaluationRecord))
    assert len(rows) == 3 and all(len(records) == 3 for _, records in rows)
    assert [
        record.evaluation.hit_s
        for _, records in rows
        for record in records
        if record.policy == "GCS+Rk"
    ] == [1, 0, None]
    assert checks


def test_unknown_only_dataset_does_not_invent_zero_quality(tmp_path):
    summary = summarize_replay(*saved_inputs(tmp_path, unknown=True), check_budget=lambda: None)
    for policy in POLICIES:
        assert summary["policies"][policy]["counts"]["N"] == 0
        assert summary["policies"][policy]["metrics"]["Hit@20"]["value"] is None
        assert summary["intervals"][policy]["metric"]["value"] is None
    assert summary["multiplicity_sensitivity"]["set_candidate_coverage"]["value"] is None


def test_fallback_keeps_target_denominator_and_missing_sensitivity_explicit(tmp_path):
    summary = summarize_replay(*saved_inputs(tmp_path, fallback=True), check_budget=lambda: None)
    assert summary["policies"]["GCS+Rk"]["counts"]["N"] == 2
    assert summary["policies"]["GCS+Rk"]["cost"]["fallback_count"] == 1
    sensitivity = summary["multiplicity_sensitivity"]
    assert sensitivity["measured_target_sessions"] == 1
    assert sensitivity["target_sessions"] == 2
    assert sensitivity["set_candidate_coverage"]["reason"] == "incomplete_sensitivity_measurement"
    assert sensitivity["set_candidate_coverage"]["value"] is None


@pytest.mark.parametrize(
    "mutation",
    ["request_id", "policy_missing", "policy_duplicate", "target_id", "session_missing", "late"],
)
def test_inconsistent_join_is_error_and_has_no_completed_session_table(tmp_path, mutation):
    inputs = saved_inputs(tmp_path, mutation=mutation)
    with pytest.raises(InputValidationError):
        summarize_replay(*inputs, check_budget=lambda: None)
    assert not (inputs[-1] / "replay/sessions.parquet").exists()


def test_repeat_uses_identical_saved_answers_and_does_not_overwrite_table(tmp_path):
    inputs = saved_inputs(tmp_path)
    first = summarize_replay(*inputs, check_budget=lambda: None)
    path = inputs[-1] / "replay/sessions.parquet"
    identity = (sha256_file(path), path.stat().st_mtime_ns)
    second = summarize_replay(*inputs, check_budget=lambda: None)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert identity == (sha256_file(path), path.stat().st_mtime_ns)
    assert render_report(first) == render_report(json.loads(canonical_json_bytes(first)))


def test_budget_interruption_does_not_publish_partial_evaluations(tmp_path):
    inputs = saved_inputs(tmp_path)
    count = 0

    def stop():
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("budget")

    with pytest.raises(RuntimeError, match="budget"):
        summarize_replay(*inputs, check_budget=stop)
    assert not (inputs[-1] / "replay/sessions.parquet").exists()


@pytest.mark.parametrize("size", [1, 2, 3])
def test_pilot_uses_independent_hash_prefix_and_unchanged_calendar(tmp_path, monkeypatch, size):
    chosen = tuple(sorted(("a", "b", "c"), key=independent_session_key)[:size])
    inputs = saved_inputs(tmp_path, answer_ids=chosen)
    read_ids = {"prefix": set(), "labels": set()}
    original_read = DatasetAccess.read

    def observed(self, relative_path, kind, *, purpose):
        for sid, values in original_read(self, relative_path, kind, purpose=purpose):
            read_ids[purpose].add(sid)
            yield sid, values

    monkeypatch.setattr(DatasetAccess, "read", observed)
    summary = summarize_replay(*inputs, check_budget=lambda: None, selected_ids=chosen)
    assert read_ids == {"prefix": {"a", "b", "c"}, "labels": {"a", "b", "c"}}
    assert summary["data"]["evaluation_scope"] == "pilot"
    assert summary["data"]["selected_ids"] == list(chosen)
    assert summary["data"]["selected_size"] == size
    assert summary["data"]["prepared_selected_size"] == 3
    assert summary["data"]["N_eligible"] == 5 and summary["data"]["N_admitted"] == 3
    target_count = sum(sid != "c" for sid in chosen)
    expected_hit = int("a" in chosen) / target_count if target_count else None
    assert summary["policies"]["GCS+Rk"]["metrics"]["Hit@20"]["value"] == expected_hit
    for policy in POLICIES:
        assert summary["policies"][policy]["counts"]["N_obs"] == size
        assert summary["policies"][policy]["counts"]["N"] == target_count
        assert summary["intervals"][policy]["details"]["grid"] == {
            "start_ms": 3000,
            "end_ms": 3890,
            "step_ms": 10,
            "cells": 89,
        }
    saved = list(iter_sessions(inputs[-1] / "replay/sessions.parquet", EvaluationRecord))
    assert [sid for sid, _ in saved] == sorted(chosen)
    assert inputs[1]["N_obs"] == 3 and inputs[1]["period_counts"]["C_smoke"]["selected"] == 3


@pytest.mark.parametrize(
    "kind", ["empty", "duplicate", "foreign", "too_many", "reordered", "not_first", "string"]
)
def test_invalid_pilot_ids_are_rejected_before_target_access(tmp_path, monkeypatch, kind):
    ordered = tuple(sorted(("a", "b", "c"), key=independent_session_key))
    invalid = {
        "empty": (),
        "duplicate": (ordered[0],) * 2,
        "foreign": ("foreign",),
        "too_many": (*ordered, "foreign"),
        "reordered": ordered[:2][::-1],
        "not_first": ordered[1:2],
        "string": ordered[0],
    }[kind]
    inputs = saved_inputs(tmp_path)

    def no_access(*args, **kwargs):
        raise AssertionError("ID пилота должны проверяться до чтения целевых таблиц")

    monkeypatch.setattr(DatasetAccess, "read", no_access)
    with pytest.raises(InputValidationError):
        summarize_replay(*inputs, check_budget=lambda: None, selected_ids=invalid)


@pytest.mark.parametrize("extra_answers", [False, True])
def test_pilot_does_not_hide_extra_or_missing_answers(tmp_path, extra_answers):
    ordered = tuple(sorted(("a", "b", "c"), key=independent_session_key))
    inputs = saved_inputs(tmp_path, answer_ids=ordered if extra_answers else ordered[:1])
    with pytest.raises(InputValidationError):
        summarize_replay(*inputs, check_budget=lambda: None, selected_ids=ordered[:2])
    assert not (inputs[-1] / "replay/sessions.parquet").exists()


def test_full_scope_is_explicit_and_pilot_cannot_change_prepared_report(tmp_path):
    inputs = saved_inputs(tmp_path)
    summary = summarize_replay(*inputs, check_budget=lambda: None)
    assert summary["data"]["evaluation_scope"] == "full"
    assert summary["data"]["prepared_selected_size"] == summary["data"]["selected_size"] == 3
    inputs[1]["N_obs"] = 2
    chosen = tuple(sorted(("a", "b", "c"), key=independent_session_key)[:2])
    with pytest.raises(InputValidationError, match="отчёт"):
        summarize_replay(*inputs, check_budget=lambda: None, selected_ids=chosen)


def test_aggregates_read_closed_evaluation_table_lazily(tmp_path, monkeypatch):
    inputs = saved_inputs(tmp_path)
    read = reporting._evaluation_rows
    seen = []

    def observed(*args, **kwargs):
        path = inputs[-1] / "replay/sessions.parquet"
        assert path.is_file(), "Сначала закрывается полная таблица оценок"
        assert table_info(path, EvaluationRecord)["rows"] == 9
        rows = read(*args, **kwargs)
        assert isinstance(rows, Iterator), "Подробные оценки должны поступать потоком"
        seen.append(True)
        return rows

    monkeypatch.setattr(reporting, "_evaluation_rows", observed)
    summary = summarize_replay(*inputs, check_budget=lambda: None)
    assert len(seen) == 1
    assert set(summary["policies"]) == set(POLICIES)
    assert summary["multiplicity_sensitivity"]["sessions"] == 3
    assert summary["policies"]["GCS+Rk"]["metrics"]["Hit@20"]["value"] == 0.5


def test_paired_comparisons_keep_only_compact_shared_session_fields(tmp_path, monkeypatch):
    inputs = saved_inputs(tmp_path)
    paired = reporting.paired_metrics
    calls = []

    def observed(left, right):
        for group in (left, right):
            assert len(group) == 3
            for evaluation, details in group:
                assert set(details) == {"session_id", "t_q", "k", "purchases"}
                assert details["session_id"] == evaluation.session_id
                assert all(set(purchase) == {"sku"} for purchase in details["purchases"])
        calls.append(True)
        return paired(left, right)

    monkeypatch.setattr(reporting, "paired_metrics", observed)
    summary = summarize_replay(*inputs, check_budget=lambda: None)
    assert len(calls) == 2
    assert summary["pairs"]["GCS+Rk-P"]["value"] == 0.5
