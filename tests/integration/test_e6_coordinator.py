import asyncio
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coveo_recsys.contracts.records import Branch, DeliveryStatus
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.service_state import StateError
from coveo_recsys.explanations.templates import TEMPLATE_SHA256, TEMPLATE_VERSION
from coveo_recsys.features import FeatureEngine
from coveo_recsys.ranking.predict import predict
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from coveo_recsys.serving.coordinator import Coordinator
from coveo_recsys.serving.runtime import BoundedWork, Deadline, ServiceError
from coveo_recsys.serving.trace import Journal
from tests.reference.e3_cases import CATALOG_HASH, package, request


class InlineWork:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    async def run(self, operation, limit):
        self.calls += 1
        if self.calls == self.failure:
            raise ServiceError("backend_timeout")
        return operation()


def setup_service(tmp_path, failure=None):
    prepared = request()
    # Реальная отдельная ось поступления; это не идеальное воспроизведение.
    prepared = replace(
        prepared, a_q=2010, prefix=replace(prepared.prefix, availability_cutoff_ms=2010)
    )
    schema = json.loads(Path("configs/feature_schema.json").read_text(encoding="utf8"))
    index = RetrievalIndex(package(), CATALOG_HASH)
    engine = FeatureEngine(index, schema, RetrievalConfig())
    context = bind_context(prepared, index)
    batch = engine.build(context, retrieve(context, index))
    model = SimpleNamespace(
        booster=SimpleNamespace(predict=lambda matrix, **kw: np.arange(len(matrix), dtype=float)),
        summary={
            "identity": {"catalog_sha256": CATALOG_HASH},
            "numeric_columns": batch.numeric_columns,
            "categorical_columns": batch.categorical_columns,
            "feature_schema_sha256": batch.feature_schema_sha256,
            "actual_tree_count": 1,
        },
    )
    bundle = SimpleNamespace(
        engine=engine,
        model=model,
        package_id="p" * 64,
        published_at_ms=1500,
        template_sha256=TEMPLATE_SHA256,
        template_version=TEMPLATE_VERSION,
    )
    receipt = SimpleNamespace(event_time_ms=2000, canonical_block_id=prepared.trigger_block_id)
    intake = SimpleNamespace(
        prefix=lambda *a, **kw: prepared,
        receipt=lambda *a: receipt,
        first_trigger=lambda *a, **kw: receipt,
    )
    service = Coordinator(
        (bundle,),
        intake,
        Journal(tmp_path / "requests.jsonl"),
        work=InlineWork(failure),
        utc_ms=lambda: 2010,
    )
    return service, prepared, bundle


def call(service, prepared, send=None, *, key="key", deadline=None):
    async def deliver(payload):
        if send:
            await send(payload)

    return asyncio.run(
        service.recommend(
            prepared.source_namespace,
            prepared.session_id,
            "transport",
            key,
            deadline=deadline or Deadline(0.0, clock=lambda: 0.0),
            deliver=deliver,
        )
    )


def test_service_uses_shared_calculation_and_distinct_arrival_cutoff(tmp_path):
    service, prepared, bundle = setup_service(tmp_path)
    result = call(service, prepared)
    context = bind_context(prepared, bundle.engine.index)
    scores = score_request(context, bundle.engine.index, bundle.engine.config)
    candidates = retrieve(context, bundle.engine.index, bundle.engine.config, scores=scores)
    expected = predict(
        bundle.model,
        bundle.engine.build(context, candidates, scores=scores),
        catalog_sha256=CATALOG_HASH,
    )
    assert result.candidates == candidates
    assert tuple(zip(result.delivered_skus, result.scores)) == expected
    assert result.delivery == DeliveryStatus.DELIVERED
    assert len(result.explanations) == len(result.delivered_skus)
    assert call(service, prepared) == result
    with pytest.raises(ServiceError, match="request_conflict"):
        call(service, prepared, key="other")


def test_service_records_already_checked_objects_without_reparsing(tmp_path, monkeypatch):
    service, prepared, _ = setup_service(tmp_path)

    def forbid_parse(*args, **kwargs):
        pytest.fail("Сервис повторно разбирает уже проверенные записи")

    monkeypatch.setattr("coveo_recsys.serving.trace.from_json", forbid_parse)
    result = call(service, prepared)
    assert result.delivery == DeliveryStatus.DELIVERED
    assert service.journal.lookup(prepared.source_namespace, prepared.session_id).result == result


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_compute_deadline_selects_prepared_fallback_not_partial_primary(tmp_path, failure):
    service, prepared, _ = setup_service(tmp_path, failure)
    result = call(service, prepared)
    assert result.candidates.branch == Branch.FALLBACK
    assert result.delivery == DeliveryStatus.DELIVERED
    assert "backend_timeout" in result.reasons


def test_expired_feature_build_does_not_start_model(tmp_path, monkeypatch):
    service, prepared, bundle = setup_service(tmp_path)
    context = bind_context(prepared, bundle.engine.index)
    batch = bundle.engine.build(context, retrieve(context, bundle.engine.index))
    now, predictions = [0.0], []

    def build(*args, **kwargs):
        now[0] = 0.900
        return batch

    def model(*args, **kwargs):
        predictions.append(True)
        return predict(*args, **kwargs)

    monkeypatch.setattr(bundle.engine, "build", build)
    monkeypatch.setattr("coveo_recsys.serving.coordinator.predict", model)
    result = call(service, prepared, deadline=Deadline(0, clock=lambda: now[0]))
    assert result.delivery == DeliveryStatus.DELIVERED
    assert result.candidates.branch == Branch.FALLBACK
    assert predictions == []


def test_delayed_worker_checks_deadline_before_reading_mutable_fallback_state(
    tmp_path, monkeypatch
):
    service, prepared, bundle = setup_service(tmp_path)
    now = [0.0]

    class DelayedWork(InlineWork):
        async def run(self, operation, limit):
            if self.calls == 2:
                self.late = operation
                now[0] = limit
                raise ServiceError("backend_timeout")
            return await super().run(operation, limit)

    work = DelayedWork()
    service.work = work
    result = call(service, prepared, deadline=Deadline(0, clock=lambda: now[0]))
    before = service.journal.path.read_bytes()
    monkeypatch.setattr(bundle.engine, "build", lambda *a, **kw: pytest.fail("Late build"))
    with pytest.raises(ServiceError):
        work.late()
    assert service.journal.path.read_bytes() == before
    assert result.delivery == DeliveryStatus.DELIVERED


def test_running_worker_stops_after_timeout_without_changing_delivered_fallback(
    tmp_path, monkeypatch
):
    service, prepared, bundle = setup_service(tmp_path)
    context = bind_context(prepared, bundle.engine.index)
    batch = bundle.engine.build(context, retrieve(context, bundle.engine.index))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    pool = BoundedWork(1)
    predictions = []
    now = [0.0]

    def build(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(2)
            return batch
        finally:
            finished.set()

    def model(*args, **kwargs):
        predictions.append(True)
        return predict(*args, **kwargs)

    class ThreadedRanking(InlineWork):
        async def run(self, operation, limit):
            if self.calls == 2:
                try:
                    return await pool.run(operation, time.perf_counter() + 0.025)
                finally:
                    now[0] = limit
            return await super().run(operation, limit)

    service.work = ThreadedRanking()
    monkeypatch.setattr(bundle.engine, "build", build)
    monkeypatch.setattr("coveo_recsys.serving.coordinator.predict", model)
    try:
        result = call(service, prepared, deadline=Deadline(0.0, clock=lambda: now[0]))
        assert entered.is_set() and not finished.is_set()
        assert result.delivery == DeliveryStatus.DELIVERED
        assert result.candidates.branch == Branch.FALLBACK
        before = service.journal.path.read_bytes()
    finally:
        release.set()
        pool.close()
    assert finished.is_set() and predictions == []
    assert service.journal.path.read_bytes() == before


def test_delivery_failure_preserves_candidates_and_clears_delivered_list(tmp_path):
    service, prepared, _ = setup_service(tmp_path)

    async def disconnected(payload):
        assert not {"candidates", "scores", "prefix", "support"} & payload.keys()
        raise ConnectionError("disconnected")

    result = call(service, prepared, disconnected)
    assert result.candidates.skus
    assert result.delivered_skus == result.scores == result.explanations == ()
    assert result.delivery == DeliveryStatus.FAILED
    assert call(service, prepared) == result


@pytest.mark.parametrize(
    "send_ms,delivered", [(20, True), (999, True), (1000, False), (1001, False)]
)
def test_return_uses_remaining_total_deadline_without_a_separate_stage_budget(
    tmp_path, send_ms, delivered
):
    service, prepared, _ = setup_service(tmp_path)
    now = [0.0]

    async def send(payload):
        now[0] += send_ms / 1000

    result = call(service, prepared, send, deadline=Deadline(0, clock=lambda: now[0]))
    assert (result.delivery == DeliveryStatus.DELIVERED) == delivered
    if not delivered:
        assert result.candidates.skus and not result.delivered_skus


def test_missing_version_cannot_use_new_package_even_for_fallback(tmp_path):
    service, prepared, bundle = setup_service(tmp_path)
    bundle.published_at_ms = 2000
    result = call(service, prepared)
    assert result.candidates.branch == Branch.NONE
    assert result.delivered_skus == result.candidates.skus == ()
    assert "version_unavailable" in result.reasons


@pytest.mark.parametrize("code", ["state_incomplete", "state_unavailable", "state_replay_limit"])
def test_unknown_eligibility_never_becomes_empty_history(tmp_path, code):
    service, prepared, _ = setup_service(tmp_path)

    def unavailable(*args, **kwargs):
        raise StateError(code)

    service.intake.prefix = unavailable
    result = call(service, prepared)
    assert result.candidates.branch == Branch.NONE
    assert result.candidates.skus == result.delivered_skus == ()
    assert code in result.reasons


@pytest.mark.parametrize("phase", ["selected", "terminal"])
def test_required_record_failure_cannot_confirm_delivery(tmp_path, phase):
    service, prepared, _ = setup_service(tmp_path)
    original = service.journal.record
    sends = []

    def failing(trace, **kwargs):
        if trace["phase"] == phase:
            raise OSError("disk failure")
        return original(trace, **kwargs)

    service.journal.record = failing

    async def capture(payload):
        sends.append(payload)

    result = call(service, prepared, capture)
    assert result.candidates.skus and result.delivered_skus == ()
    assert result.delivery == DeliveryStatus.FAILED
    assert len(sends) == (phase == "terminal")
    assert len(result.reasons) == len(set(result.reasons))
    assert service.journal.lookup(prepared.source_namespace, prepared.session_id).result == result


def test_cancellation_does_not_send_prepared_fallback(tmp_path):
    service, prepared, _ = setup_service(tmp_path)

    async def cancel(operation, limit):
        raise asyncio.CancelledError

    service.work.run = cancel
    sends = []

    async def capture(payload):
        sends.append(payload)

    result = call(service, prepared, capture)
    assert not sends and not result.delivered_skus
    assert "cancelled" in result.reasons


def test_stalled_delivery_is_bounded_and_does_not_erase_candidates(tmp_path):
    service, prepared, _ = setup_service(tmp_path)

    async def stalled(payload):
        await asyncio.sleep(10)

    result = call(service, prepared, stalled, deadline=Deadline())
    assert result.candidates.skus and not result.delivered_skus
    assert result.delivery == DeliveryStatus.FAILED


@pytest.mark.parametrize("serialization_ms,delivered", [(20, True), (1000, False)])
def test_serialization_is_inside_total_deadline(tmp_path, monkeypatch, serialization_ms, delivered):
    from coveo_recsys.serving import trace

    service, prepared, _ = setup_service(tmp_path)
    now = [0.0]
    original = trace.to_json

    def slow(value):
        payload = original(value)
        if value.__class__.__name__ == "RequestContext":
            now[0] += serialization_ms / 1000
        return payload

    monkeypatch.setattr(trace, "to_json", slow)
    result = call(service, prepared, deadline=Deadline(0.0, clock=lambda: now[0]))
    assert result.candidates.skus
    assert (result.delivery == DeliveryStatus.DELIVERED) == delivered


@pytest.mark.parametrize("start,timeout_ms", [(0.0, 1000), (100.0, 1501)])
def test_all_compute_stages_share_one_deadline_and_exceed_old_budgets(tmp_path, start, timeout_ms):
    service, prepared, _ = setup_service(tmp_path)
    now = [start]
    deadline = Deadline(start, timeout_ms=timeout_ms, clock=lambda: now[0])
    original_prefix = service.intake.prefix

    def prefix(*args, **kwargs):
        now[0] += 0.050
        return original_prefix(*args, **kwargs)

    class SlowWork(InlineWork):
        async def run(self, operation, limit):
            assert limit == deadline.compute_end
            now[0] += 0.100
            return await super().run(operation, limit)

    service.intake.prefix = prefix
    service.work = SlowWork()
    result = call(service, prepared, deadline=deadline)
    assert result.candidates.branch == Branch.PRIMARY
    assert result.delivery == DeliveryStatus.DELIVERED
    trace = json.loads(service.journal.path.read_text(encoding="utf-8").splitlines()[-1])["trace"]
    assert trace["timeout_ms"] == timeout_ms
    assert trace["deadlines_ms"]["ranking"] == timeout_ms - 100
    assert trace["deadlines_ms"]["record"] == trace["deadlines_ms"]["return"] == timeout_ms
