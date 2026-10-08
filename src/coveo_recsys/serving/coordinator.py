"""Один закреплённый запрос: срез, расчёт, резерв и обязательная запись."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from threading import Event
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestContext,
    RequestResult,
)
from coveo_recsys.contracts.retrieval import SOURCES
from coveo_recsys.contracts.service_state import StateError
from coveo_recsys.explanations import explain
from coveo_recsys.ranking.predict import predict
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.fallback import fallback
from coveo_recsys.retrieval.scores import bind_context, score_request
from coveo_recsys.state.service import IntakeService

from .package import ServingPackage, select_package
from .runtime import BoundedWork, Deadline, ServiceError, await_until
from .trace import Journal


def public_result(result: RequestResult) -> dict[str, Any]:
    return {
        "query_id": result.request_id,
        "package_id": result.package_id,
        "items": list(result.delivered_skus),
        "branch": result.candidates.branch.value,
        "delivery": result.delivery.value,
        "reasons": list(result.reasons),
        "explanations": [
            {"sku": item.sku, "code": item.template_id, "text": item.public_text}
            for item in result.explanations
        ],
    }


class Coordinator:
    def __init__(
        self,
        packages: Sequence[ServingPackage],
        intake: IntakeService,
        journal: Journal,
        *,
        work: BoundedWork | None = None,
        utc_ms: Callable[[], int] | None = None,
        timeout_ms: int = 1000,
        reserve_ms: int = 100,
    ):
        Deadline(0, timeout_ms=timeout_ms, reserve_ms=reserve_ms)
        self.timeout_ms, self.reserve_ms = timeout_ms, reserve_ms
        self.packages = tuple(packages)
        self.intake = intake
        self.journal = journal
        self.work = work or BoundedWork()
        self.utc_ms = utc_ms or (lambda: time.time_ns() // 1_000_000)

    async def recommend(
        self,
        namespace: str,
        session_id: str,
        trigger_block_id: str,
        key: str,
        *,
        deadline: Deadline,
        deliver: Callable[[dict[str, Any]], Awaitable[None]],
        source_runtime: Mapping[str, str | None] | None = None,
    ) -> RequestResult:
        previous = self.journal.lookup(namespace, session_id)
        if previous is not None:
            if previous.key != key or previous.trigger_block_id != trigger_block_id:
                raise ServiceError("request_conflict")
            if previous.result is None:
                raise ServiceError(
                    "request_in_progress" if previous.state == "pending" else "cancelled"
                )
            # Передача уже записанного результата не создаёт второго запроса или нового срока.
            await await_until(
                deliver(public_result(previous.result)),
                deadline.stage_limit("return"),
                clock=deadline.clock,
            )
            return previous.result
        a_q = self.utc_ms()
        state_limit = deadline.stage_limit("state")
        receipt = self.intake.first_trigger(
            namespace,
            session_id,
            trigger_block_id,
            a_q=a_q,
            check_deadline=lambda: deadline.check(state_limit),
        )
        t_q = receipt.event_time_ms
        if not self.journal.reserve(namespace, session_id, key, trigger_block_id, t_q, a_q):
            raise ServiceError("request_conflict")
        request_id = sha256_bytes(canonical_json_bytes([namespace, session_id, trigger_block_id]))
        package: ServingPackage | None = None
        context: RequestContext | None = None
        chosen: CandidateResult | None = None
        reserve: CandidateResult | None = None
        ranked: tuple[tuple[str, float], ...] = ()
        scores = None
        reasons: tuple[str, ...] = ()
        stages: dict[str, dict[str, Any]] = {
            "input": {"elapsed_ms": float(deadline.elapsed_ms), "status": "ok"}
        }
        deadlines_ms = {
            "input": float(deadline.timeout_ms - deadline.reserve_ms),
            "state": float(deadline.timeout_ms - deadline.reserve_ms),
        }
        runtime = {name: None for name in SOURCES} | dict(source_runtime or {})
        unavailable = {name: reason for name, reason in runtime.items() if reason is not None}
        stage = "state"
        started = deadline.clock()

        async def calculate(name: str, operation: Callable[[Callable[[], None]], Any]) -> Any:
            nonlocal stage, started
            stage, started = name, deadline.clock()
            limit = deadline.stage_limit(name)
            deadlines_ms[name] = float(deadline.timeout_ms - deadline.reserve_ms)
            deadline.check(limit)
            abandoned = Event()

            def check() -> None:
                if abandoned.is_set():
                    raise ServiceError("cancelled")
                deadline.check(limit)

            def execute() -> Any:
                check()
                value = operation(check)
                check()
                return value

            try:
                result = await self.work.run(execute, limit)
            finally:
                abandoned.set()
            deadline.check(limit)
            stages[name] = {"elapsed_ms": (deadline.clock() - started) * 1000, "status": "ok"}
            return result

        try:
            if not any(p.published_at_ms < t_q for p in self.packages):
                raise ServiceError("version_unavailable")
            package = select_package(self.packages, t_q=t_q)
            prepared = self.intake.prefix(
                namespace,
                session_id,
                trigger_block_id,
                a_q=a_q,
                check_deadline=lambda: deadline.check(state_limit),
            )
            context = bind_context(prepared, package.engine.index)
            request_id = context.request_id
            reserve = fallback(context, package.engine.index, package.engine.config)
            deadline.check(state_limit)
            stages[stage] = {"elapsed_ms": (deadline.clock() - started) * 1000, "status": "ok"}
            scores = await calculate(
                "representation",
                lambda _check: score_request(
                    context, package.engine.index, package.engine.config, source_runtime=unavailable
                ),
            )
            primary_scores = scores
            candidates = await calculate(
                "sources",
                lambda _check: retrieve(
                    context, package.engine.index, package.engine.config, scores=primary_scores
                ),
            )

            def rank(check: Callable[[], None]) -> tuple[tuple[str, float], ...]:
                batch = package.engine.build(
                    context, candidates, scores=primary_scores, check_deadline=check
                )
                check()
                return predict(
                    package.model,
                    batch,
                    catalog_sha256=context.catalog_sha256,
                )

            ranked = await calculate("ranking", rank)
            chosen = candidates
        except (ServiceError, StateError) as exc:
            reasons = (exc.code,)
        except asyncio.CancelledError:
            reasons = ("cancelled",)
        except (ValueError, OSError, RuntimeError):
            # Некорректное внутреннее число/форма не попадает в публичный текст ошибки.
            reasons = ("calculation_failure",)
        if reasons:
            stages[stage] = {
                "elapsed_ms": (deadline.clock() - started) * 1000,
                "status": reasons[0],
            }
            if (
                reserve is not None
                and context is not None
                and "cancelled" not in reasons
                and deadline.clock() < deadline.end
            ):
                chosen = reserve
                values = {item.sku: item.score for item in reserve.evidence}
                ranked = tuple((sku, values[sku]) for sku in reserve.skus[:20])
                scores = None
            else:
                ranked = ()
        if chosen is None:
            chosen = CandidateResult(request_id, (), (), (), Branch.NONE, reasons)
        if context is not None and package is not None:
            eligible = set(package.engine.index.positions) - set(context.prefix.purchased_skus)
            if not set(chosen.skus) <= eligible or not {s for s, _ in ranked} <= set(chosen.skus):
                raise InputValidationError("Нарушена допустимость выдачи")
        record_start = deadline.clock()
        limit = deadline.stage_limit("record")
        deadlines_ms["record"] = float(deadline.timeout_ms)
        deadlines_ms["return"] = float(deadline.timeout_ms)
        result = RequestResult(
            request_id,
            package.package_id if package else "unavailable",
            chosen,
            (),
            (),
            (),
            DeliveryStatus.NOT_ATTEMPTED,
            tuple(dict.fromkeys((*reasons, *chosen.reasons))),
            deadline.elapsed_ms,
            f"requests.jsonl#{request_id}",
        )
        trace = {
            "format_version": "execution_trace_v1",
            "phase": "selected",
            "source_namespace": namespace,
            "session_id": session_id,
            "idempotency_key": key,
            "trigger_block_id": trigger_block_id,
            "canonical_block_id": receipt.canonical_block_id,
            "t_q": t_q,
            "a_q": a_q,
            "package_id": result.package_id,
            "published_at_ms": package.published_at_ms if package else None,
            "template_sha256": package.template_sha256 if package else None,
            "context": context,
            "source_runtime": runtime,
            "stages": stages,
            "timeout_ms": deadline.timeout_ms,
            "deadlines_ms": deadlines_ms,
            "result": result,
        }
        sent = False
        delivery_started = False
        try:
            deadline.check(limit)
            explanations = (
                ()
                if package is None
                else explain(
                    chosen,
                    tuple(s for s, _ in ranked),
                    package_id=package.package_id,
                    scores=scores,
                    index=package.engine.index,
                    template_version=package.template_version,
                )
            )
            self.journal.record(
                trace,
                check_deadline=lambda: deadline.check(limit),
                checked_result=result,
                checked_context=context,
            )
            deadline.check(limit)
            stages["record"] = {
                "elapsed_ms": (deadline.clock() - record_start) * 1000,
                "status": "ok",
            }
            if chosen.branch != Branch.NONE:
                result = replace(
                    result,
                    delivered_skus=tuple(s for s, _ in ranked),
                    scores=tuple(v for _, v in ranked),
                    explanations=explanations,
                    delivery=DeliveryStatus.DELIVERED,
                )
            deadline.check()
            if "cancelled" in reasons:
                raise ServiceError("cancelled")
            delivery_started = True
            return_limit = deadline.stage_limit("return")
            await await_until(deliver(public_result(result)), return_limit, clock=deadline.clock)
            sent = True
            deadline.check()
            stages["return"] = {"elapsed_ms": float(deadline.elapsed_ms), "status": "ok"}
        except (OSError, ValueError, RuntimeError, asyncio.CancelledError) as exc:
            reason = (
                exc.code
                if isinstance(exc, ServiceError)
                else (
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else ("delivery_failure" if delivery_started else "logging_failure")
                )
            )
            result = replace(
                result,
                delivered_skus=(),
                scores=(),
                explanations=(),
                delivery=DeliveryStatus.FAILED,
                reasons=tuple(dict.fromkeys((*result.reasons, reason))),
            )
            stages["return"] = {"elapsed_ms": float(deadline.elapsed_ms), "status": reason}
        result = replace(result, elapsed_ms=deadline.elapsed_ms)
        trace.update(phase="terminal", result=result)
        try:
            self.journal.record(
                trace,
                checked_result=result,
                checked_context=context,
                check_deadline=deadline.check
                if result.delivery == DeliveryStatus.DELIVERED
                else None,
            )
        except (OSError, ValueError):
            result = replace(
                result,
                delivered_skus=(),
                scores=(),
                explanations=(),
                delivery=DeliveryStatus.FAILED,
                reasons=tuple(
                    dict.fromkeys(
                        (
                            *result.reasons,
                            "logging_failure_after_send" if sent else "logging_failure",
                        )
                    )
                ),
            )
            self.journal.fail(namespace, session_id, result)
        return result
