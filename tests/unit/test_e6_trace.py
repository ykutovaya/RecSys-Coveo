import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestContext,
    RequestResult,
    SourceEvidence,
    SourceResult,
)
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.explanations import TEMPLATE_SHA256, explain
from coveo_recsys.retrieval import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from coveo_recsys.serving.trace import Journal, replay_trace, validate_trace
from tests.reference.e3_cases import CATALOG_HASH, package, request


def trace(*, phase="terminal", delivery=DeliveryStatus.DELIVERED, timeout_ms=1000):
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(request(), index)
    scores = score_request(context, index)
    candidates = retrieve(context, index, scores=scores)
    delivered = candidates.skus[:2] if delivery == DeliveryStatus.DELIVERED else ()
    result = RequestResult(
        context.request_id,
        "package",
        candidates,
        delivered,
        tuple(0.0 for _ in delivered),
        explain(candidates, delivered, package_id="package", scores=scores, index=index),
        delivery,
        (),
        4,
        "journal:q",
    )
    stages = {
        name: {"elapsed_ms": 0.1, "status": "ok"}
        for name in ("input", "representation", "sources", "ranking")
    }
    if phase == "terminal":
        stages["return"] = {
            "elapsed_ms": 0.1,
            "status": "ok" if delivery == DeliveryStatus.DELIVERED else "failed",
        }
    return {
        "format_version": "execution_trace_v1",
        "phase": phase,
        "source_namespace": context.prefix.source_namespace,
        "session_id": context.session_id,
        "idempotency_key": "key",
        "trigger_block_id": "transport-block",
        "canonical_block_id": context.trigger_block_id,
        "t_q": context.t_q,
        "a_q": context.a_q,
        "package_id": "package",
        "published_at_ms": 1001,
        "template_sha256": TEMPLATE_SHA256,
        "context": to_json(context).decode(),
        "source_runtime": scores.reasons,
        "timeout_ms": timeout_ms,
        "deadlines_ms": {
            "input": timeout_ms * 0.9,
            "state": timeout_ms * 0.9,
            "representation": timeout_ms * 0.9,
            "sources": timeout_ms * 0.9,
            "ranking": timeout_ms * 0.9,
            "record": timeout_ms,
            "return": timeout_ms,
        },
        "stages": stages,
        "result": to_json(result).decode(),
    }


def reserve(journal):
    return journal.reserve("fixture", "s", "key", "transport-block", 2000, 2000)


def large_trace():
    """300 искусственных кандидатов и малый срез; без исторических данных."""
    body = trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED)
    result = from_json(body["result"].encode(), RequestResult)
    context = from_json(body["context"].encode(), RequestContext)
    skus = tuple(f"{i:064x}" for i in range(300))
    values = tuple((300 - i) / 45150 for i in range(300))
    candidates = CandidateResult(
        result.request_id,
        (SourceResult("popularity", skus, values, True, 300, None),),
        skus,
        tuple((sku, ("popularity",)) for sku in skus),
        Branch.FALLBACK,
        (),
        tuple(
            SourceEvidence(sku, "popularity", None, value, None, None, None)
            for sku, value in zip(skus, values, strict=True)
        ),
    )
    result = replace(result, candidates=candidates)
    body["result"] = to_json(result).decode()
    return body, result, context


@pytest.mark.parametrize("typed", [False, True])
def test_checked_trace_objects_preserve_validation_and_journal_bytes(tmp_path, monkeypatch, typed):
    from coveo_recsys.serving import trace as module

    body, result, context = large_trace()
    expected = validate_trace(body)
    ordinary = Journal(tmp_path / "ordinary.jsonl")
    checked = Journal(tmp_path / "checked.jsonl")
    reserve(ordinary)
    reserve(checked)
    ordinary.record(body)
    if typed:
        body = dict(body, result=result, context=context)
    serialized = []
    serialize = module.to_json

    def serialize_once(value):
        serialized.append(value)
        return serialize(value)

    def parse_forbidden(*args):
        raise AssertionError("Повторный разбор проверенных записей")

    with monkeypatch.context() as patch:
        patch.setattr(module, "from_json", parse_forbidden)
        patch.setattr(module, "to_json", serialize_once)
        actual = validate_trace(body, checked_result=result, checked_context=context)
        assert actual is result and actual == expected
        serialized.clear()
        checked.record(body, checked_result=result, checked_context=context)
        assert len(serialized) == 2
        assert serialized[0] is result and serialized[1] is context
    assert ordinary.path.read_bytes() == checked.path.read_bytes()
    assert Journal(checked.path).lookup("fixture", "s").state == "cancelled"


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("problem", ["package", "namespace", "deadlines", "missing", "phase"])
def test_checked_objects_preserve_semantic_errors(problem, typed):
    body, result, context = large_trace()
    if problem == "package":
        body["package_id"] = "other"
    elif problem == "namespace":
        body["source_namespace"] = "other"
    elif problem == "deadlines":
        body["deadlines_ms"]["sources"] = body["timeout_ms"] + 1.0
    elif problem == "missing":
        del body["canonical_block_id"]
    else:
        body["phase"] = "unknown"
    with pytest.raises(InputValidationError) as ordinary:
        validate_trace(body)
    if typed:
        body = dict(body, result=result, context=context)
    with pytest.raises(InputValidationError) as checked:
        validate_trace(body, checked_result=result, checked_context=context)
    assert str(ordinary.value) == str(checked.value)


@pytest.mark.parametrize("field", ["result", "context"])
def test_typed_trace_requires_same_checked_object(field):
    body, result, context = large_trace()
    body.update(result=result, context=context)
    body[field] = replace(body[field])
    with pytest.raises(InputValidationError):
        validate_trace(body, checked_result=result, checked_context=context)
    with pytest.raises(InputValidationError):
        validate_trace(body)


@pytest.mark.parametrize("field", ["context", "result"])
@pytest.mark.parametrize("change", ["whitespace", "version", "structure"])
def test_checked_objects_require_exact_serialized_bytes(field, change):
    body, result, context = large_trace()
    if change == "whitespace":
        body[field] += " "
        assert validate_trace(body) == result
    else:
        envelope = json.loads(body[field])
        if change == "version":
            envelope["contract_version"] = "unknown"
        else:
            envelope["data"]["unexpected"] = True
        body[field] = json.dumps(envelope)
        with pytest.raises(InputValidationError):
            validate_trace(body)
    with pytest.raises(InputValidationError):
        validate_trace(body, checked_result=result, checked_context=context)


def test_journal_reserves_one_first_request_atomically_per_namespace_session(tmp_path):
    journal = Journal(tmp_path / "trace.jsonl")
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: reserve(journal), range(12))) == 1
    assert journal.reserve("other", "s", "key", "2000", 2000, 2000)
    assert not journal.reserve("fixture", "s", "changed", "3000", 3000, 3000)
    saved = journal.lookup("fixture", "s")
    assert saved.key == "key" and saved.trigger_block_id == "transport-block"
    assert saved.state == "pending" and saved.result is None
    assert journal.ready


@pytest.mark.parametrize("timeout_ms", [120, 1000, 2500])
def test_selected_then_terminal_preserves_delivered_result_across_restart(tmp_path, timeout_ms):
    path = tmp_path / "trace.jsonl"
    journal = Journal(path)
    assert reserve(journal)
    selected = trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED, timeout_ms=timeout_ms)
    terminal = trace(timeout_ms=timeout_ms)
    if timeout_ms == 120:
        legacy_deadlines = {
            "input": 10.0,
            "state": 20.0,
            "representation": 35.0,
            "sources": 70.0,
            "ranking": 95.0,
            "record": 105.0,
            "return": 120.0,
        }
        selected["deadlines_ms"] = terminal["deadlines_ms"] = legacy_deadlines
    journal.record(selected)
    assert journal.lookup("fixture", "s").state == "pending"
    expected = validate_trace(terminal)
    journal.record(terminal)
    assert journal.lookup("fixture", "s").result == expected
    restored = Journal(path)
    assert not reserve(restored)
    assert restored.lookup("fixture", "s").result == expected
    assert restored.lookup("fixture", "s").state == "finished"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.parametrize("selected", [False, True])
def test_interrupted_reservation_remains_cancelled_without_starting_again(tmp_path, selected):
    path = tmp_path / "trace.jsonl"
    journal = Journal(path)
    reserve(journal)
    if selected:
        journal.record(trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED))
    restored = Journal(path)
    saved = restored.lookup("fixture", "s")
    assert saved.state == "cancelled" and saved.result is None
    assert not reserve(restored)
    with pytest.raises(InputValidationError):
        restored.record(trace())


def test_delivered_terminal_requires_the_same_persisted_selection(tmp_path):
    journal = Journal(tmp_path / "trace.jsonl")
    reserve(journal)
    with pytest.raises(InputValidationError):
        journal.record(trace())
    selected = trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED)
    journal.record(selected)
    terminal = trace()
    terminal["published_at_ms"] = 1002
    with pytest.raises(InputValidationError):
        journal.record(terminal)


def test_same_immutable_selection_is_not_serialized_again(tmp_path, monkeypatch):
    from coveo_recsys.serving import trace as module

    body, result, context = large_trace()
    journal = Journal(tmp_path / "trace.jsonl")
    reserve(journal)
    serialize = module.to_json

    def without_candidate_copy(value):
        assert not isinstance(value, CandidateResult)
        return serialize(value)

    monkeypatch.setattr(module, "to_json", without_candidate_copy)
    journal.record(body, checked_result=result, checked_context=context)
    failed = replace(result, delivery=DeliveryStatus.FAILED)
    body.update(phase="terminal", result=to_json(failed).decode())
    journal.record(body, checked_result=failed, checked_context=context)
    assert journal.lookup("fixture", "s").result == failed


@pytest.mark.parametrize("changed_zero", [False, True])
def test_selection_of_distinct_objects_uses_exact_bytes(tmp_path, changed_zero):
    body, result, context = large_trace()
    evidence = replace(result.candidates.evidence[0], score=0.0)
    result = replace(
        result,
        candidates=replace(result.candidates, evidence=(evidence, *result.candidates.evidence[1:])),
    )
    body["result"] = to_json(result).decode()
    journal = Journal(tmp_path / "trace.jsonl")
    reserve(journal)
    journal.record(body, checked_result=result, checked_context=context)
    replacement = replace(evidence, score=-0.0 if changed_zero else 0.0)
    failed = replace(
        result,
        delivery=DeliveryStatus.FAILED,
        candidates=replace(
            result.candidates, evidence=(replacement, *result.candidates.evidence[1:])
        ),
    )
    body.update(phase="terminal", result=to_json(failed).decode())
    if changed_zero:
        with pytest.raises(InputValidationError):
            journal.record(body, checked_result=failed, checked_context=context)
    else:
        journal.record(body, checked_result=failed, checked_context=context)
        assert Journal(journal.path).lookup("fixture", "s").result == failed


def test_write_failure_does_not_deliver_and_keeps_memory_failure(tmp_path, monkeypatch):
    from coveo_recsys.serving import trace as module

    path = tmp_path / "trace.jsonl"
    journal = Journal(path)
    reserve(journal)
    journal.record(trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED))
    initial = path.read_bytes()
    monkeypatch.setattr(module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        journal.record(trace())
    assert path.read_bytes() == initial
    assert not journal.ready
    failed = validate_trace(trace(delivery=DeliveryStatus.FAILED))
    journal.fail("fixture", "s", failed)
    assert journal.lookup("fixture", "s").result == failed
    with pytest.raises(InputValidationError):
        journal.fail("fixture", "s", validate_trace(trace()))


def test_memory_failure_cannot_replace_an_already_selected_candidate_set(tmp_path):
    journal = Journal(tmp_path / "trace.jsonl")
    reserve(journal)
    journal.record(trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED))
    failed = validate_trace(trace(delivery=DeliveryStatus.FAILED))
    changed = replace(
        failed.candidates, skus=(), origins=(), evidence=(), sources=(), branch=Branch.NONE
    )
    with pytest.raises(InputValidationError):
        journal.fail("fixture", "s", replace(failed, candidates=changed))
    journal.fail("fixture", "s", failed)
    assert journal.lookup("fixture", "s").result.candidates == failed.candidates


def test_terminal_fsync_deadline_rolls_back_before_publishing_finished_result(
    tmp_path, monkeypatch
):
    from coveo_recsys.serving import trace as module
    from coveo_recsys.serving.runtime import Deadline, ServiceError

    path = tmp_path / "trace.jsonl"
    journal = Journal(path)
    reserve(journal)
    journal.record(trace(phase="selected", delivery=DeliveryStatus.NOT_ATTEMPTED))
    original = path.read_bytes()
    now = [0.0]
    deadline = Deadline(clock=lambda: now[0])
    fsync = module.os.fsync

    def slow_fsync(fd):
        fsync(fd)
        now[0] = deadline.end

    monkeypatch.setattr(module.os, "fsync", slow_fsync)
    with pytest.raises(ServiceError, match="backend_timeout"):
        journal.record(trace(), check_deadline=deadline.check)
    assert path.read_bytes() == original
    assert journal.lookup("fixture", "s").state == "pending"
    assert journal.lookup("fixture", "s").result is None
    assert journal.ready
    failed_trace = trace(delivery=DeliveryStatus.FAILED)
    failed = validate_trace(failed_trace)
    journal.record(failed_trace)
    assert journal.lookup("fixture", "s").result == failed


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "context",
        "canonical",
        "publication",
        "template",
        "stage",
        "version",
        "delivery",
        "purchased",
        "targets",
    ],
)
def test_trace_detects_missing_or_inconsistent_recorded_conditions(mutation):
    body = trace()
    if mutation == "missing":
        del body["source_runtime"]
    elif mutation == "context":
        body["source_namespace"] = "other"
    elif mutation == "canonical":
        body["canonical_block_id"] = "transport-block"
    elif mutation == "publication":
        body["published_at_ms"] = body["t_q"]
    elif mutation == "template":
        body["template_sha256"] = "0" * 64
    elif mutation == "stage":
        body["stages"]["input"]["elapsed_ms"] = float("nan")
    elif mutation == "version":
        body["format_version"] = "unknown"
    elif mutation == "delivery":
        body["phase"] = "selected"
    elif mutation == "purchased":
        context = json.loads(body["context"])
        context["data"]["prefix"]["purchased_skus"] = ["a"]
        body["context"] = json.dumps(context)
    else:
        body["targets"] = []
    with pytest.raises(InputValidationError):
        validate_trace(body)


def test_trace_replay_returns_recorded_failure_not_a_healthy_recalculation():
    body = trace(delivery=DeliveryStatus.FAILED)
    body["stages"]["return"] = {"elapsed_ms": 101.0, "status": "timeout"}
    recorded = validate_trace(body)
    recorded = replace(recorded, reasons=("return_timeout",), elapsed_ms=body["timeout_ms"] + 5)
    body["result"] = to_json(recorded).decode()
    result = replay_trace(body)
    assert result == recorded and result.delivery == DeliveryStatus.FAILED
    assert result.candidates.skus and result.delivered_skus == ()


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "stage",
        "limit_missing",
        "negative",
        "nan",
        "infinite",
        "negative_infinite",
        "boolean",
        "text",
        "late",
    ],
)
def test_trace_requires_declared_deadlines_for_known_stages(problem):
    body = trace()
    if problem == "missing":
        del body["deadlines_ms"]
    elif problem == "stage":
        body["deadlines_ms"]["unknown"] = 1.0
        body["stages"]["unknown"] = {"elapsed_ms": 0.0, "status": "ok"}
    elif problem == "limit_missing":
        del body["deadlines_ms"]["sources"]
    else:
        body["deadlines_ms"]["sources"] = {
            "negative": -0.1,
            "nan": float("nan"),
            "infinite": float("inf"),
            "negative_infinite": float("-inf"),
            "boolean": True,
            "text": "900",
            "late": body["timeout_ms"] + 0.1,
        }[problem]
    with pytest.raises(InputValidationError):
        validate_trace(body)


@pytest.mark.parametrize("stage", ["representation", "sources", "ranking"])
@pytest.mark.parametrize("delivery", [DeliveryStatus.DELIVERED, DeliveryStatus.FAILED])
def test_primary_trace_retains_successful_compute_stages_even_on_nondelivery(stage, delivery):
    body = trace(delivery=delivery)
    body["stages"][stage]["status"] = "timeout"
    with pytest.raises(InputValidationError):
        validate_trace(body)


@pytest.mark.parametrize("timeout_ms", [1, 120, 1000, 2500])
@pytest.mark.parametrize("late_by_ms", [0, 1])
def test_delivered_trace_requires_timely_successful_return(timeout_ms, late_by_ms):
    body = trace(timeout_ms=timeout_ms)
    recorded = from_json(body["result"].encode(), RequestResult)
    elapsed_ms = timeout_ms + late_by_ms
    body["result"] = to_json(replace(recorded, elapsed_ms=elapsed_ms)).decode()
    if late_by_ms == 0:
        assert validate_trace(body).elapsed_ms == elapsed_ms
        body["stages"]["return"]["status"] = "timeout"
    with pytest.raises(InputValidationError):
        validate_trace(body)


@pytest.mark.parametrize("timeout_ms", [0, -1, True, False, 1000.0, "1000", None])
def test_trace_requires_positive_integer_request_timeout(timeout_ms):
    body = trace()
    body["timeout_ms"] = timeout_ms
    with pytest.raises(InputValidationError):
        validate_trace(body)


def test_trace_stage_measurements_do_not_impose_legacy_stage_budgets():
    body = trace()
    recorded = validate_trace(body)
    body["result"] = to_json(replace(recorded, elapsed_ms=700)).decode()
    body["stages"]["sources"]["elapsed_ms"] = 200.0
    body["stages"]["ranking"]["elapsed_ms"] = 150.0
    assert replay_trace(body).delivery == DeliveryStatus.DELIVERED


def test_delivered_trace_requires_recorded_return_deadline_within_request_timeout():
    body = trace()
    recorded = validate_trace(body)
    body["result"] = to_json(replace(recorded, elapsed_ms=990)).decode()
    body["deadlines_ms"]["return"] = 980
    with pytest.raises(InputValidationError):
        validate_trace(body)


@pytest.mark.parametrize(
    "content", [b'{"type":"reserve"', b"\n", b"{}\n", b'{"type":"x","type":"y"}\n']
)
def test_damaged_journal_is_not_silently_skipped(tmp_path, content):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(content)
    with pytest.raises(InputValidationError):
        Journal(path)


def test_journal_availability_check_does_not_write_records(tmp_path):
    path = tmp_path / "trace.jsonl"
    journal = Journal(path)
    journal.check_available()
    assert not path.exists()
    reserve(journal)
    content = path.read_bytes()
    journal.check_available()
    assert path.read_bytes() == content
