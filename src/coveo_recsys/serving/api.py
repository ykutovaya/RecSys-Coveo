"""Локальный HTTP-транспорт; сырые события и секреты не входят в ответы."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from coveo_recsys.artifacts import strict_json_loads
from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.service_state import StateError

from .coordinator import Coordinator, public_result
from .runtime import Deadline, ServiceError, await_until

MAX_BODY = 1_048_576
PUBLIC_FIELDS = ("query_id", "package_id", "items", "branch", "delivery", "reasons", "explanations")
REQUEST_FIELDS = {"source_namespace", "session_id", "trigger_block_id", "idempotency_key"}


class _Arrival:
    """Сохранить первый байт протокола; без него начать срок на входе ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("coveo.request_start", time.perf_counter())
        await self.app(scope, receive, send)


class _Admission:
    """Одна очередь на приложение: 16 выполняемых и не более 16 ожидающих."""

    def __init__(self) -> None:
        self.slots = asyncio.Semaphore(16)
        self.count = 0

    async def acquire(self, limit: float, clock: Callable[[], float]) -> None:
        if self.count >= 32:
            raise ServiceError("overload_rejected")
        self.count += 1
        operation = None
        try:
            if clock() >= limit:
                raise ServiceError("backend_timeout")
            operation = asyncio.create_task(self.slots.acquire())
            await await_until(operation, limit, clock=clock)
        except BaseException:
            self.count -= 1
            if (
                operation is not None
                and operation.done()
                and not operation.cancelled()
                and operation.exception() is None
                and operation.result()
            ):
                self.slots.release()
            raise

    def release(self) -> None:
        self.slots.release()
        self.count -= 1


class _SenderLimiter:
    """До распознавания сессии: всплеск 100 попыток, пополнение 10/с."""

    def __init__(self) -> None:
        self.clock: Callable[[], float] = time.monotonic
        self.buckets: dict[str, tuple[float, float]] = {}

    def charge(self, namespace: str) -> None:
        now = self.clock()
        available, previous = self.buckets.get(namespace, (100.0, now))
        available = min(100.0, available + max(0.0, now - previous) * 10)
        allowed = available >= 1
        self.buckets[namespace] = (available - 1 if allowed else available, max(previous, now))
        if not allowed:
            raise ServiceError("rate_rejected")


async def _body(request: Request, deadline: Deadline, limit: float) -> bytes:
    header = request.headers.get("content-length")
    if header is not None:
        try:
            size = int(header)
        except ValueError as exc:
            raise ServiceError("invalid_input") from exc
        if not 0 <= size <= MAX_BODY:
            raise ServiceError("request_too_large")
    chunks = bytearray()
    while True:
        deadline.check(limit)
        message = await await_until(request.receive(), limit, clock=deadline.clock)
        if message["type"] == "http.disconnect":
            raise ServiceError("cancelled")
        part = message.get("body", b"")
        if len(chunks) + len(part) > MAX_BODY:
            raise ServiceError("request_too_large")
        chunks.extend(part)
        if not message.get("more_body", False):
            deadline.check(limit)
            return bytes(chunks)


def _error(code: str) -> JSONResponse:
    status = {
        "unauthorized": 401,
        "namespace_forbidden": 403,
        "invalid_input": 400,
        "request_too_large": 413,
        "rate_rejected": 429,
        "overload_rejected": 503,
        "backend_timeout": 504,
        "cancelled": 499,
    }.get(code, 503)
    return JSONResponse(
        {"reason": "invalid_input" if code == "request_too_large" else code}, status_code=status
    )


class _ServiceResponse(Response):
    """Координатор видит результат именно ASGI-send, не построения словаря."""

    def __init__(self, request: Request, coordinator: Coordinator, kind: str) -> None:
        super().__init__()
        self.request, self.coordinator, self.kind = request, coordinator, kind
        self.deadline = Deadline(
            start=request.scope["coveo.request_start"],
            timeout_ms=coordinator.timeout_ms,
            reserve_ms=coordinator.reserve_ms,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        coordinator, request, deadline = self.coordinator, self.request, self.deadline
        admission = request.app.state.admission
        acquired, sent = False, False

        async def deliver(payload: dict[str, Any]) -> None:
            nonlocal sent
            if sent:
                raise ServiceError("delivery_already_attempted")
            response = JSONResponse({name: payload[name] for name in PUBLIC_FIELDS})
            deadline.check()
            sent = True
            await response(scope, receive, send)

        try:
            input_limit = deadline.stage_limit("input")
            await admission.acquire(input_limit, deadline.clock)
            acquired = True
            auth = request.headers.getlist("authorization")
            if len(auth) != 1 or not auth[0].startswith("Bearer "):
                raise ServiceError("unauthorized")
            token = auth[0][7:]
            namespace = coordinator.intake.namespace_for(token)
            request.app.state.sender_limiter.charge(namespace)
            body = await _body(request, deadline, input_limit)
            if self.kind == "blocks":
                complete = request.headers.get("x-stream-complete")
                if complete not in (None, "true", "false"):
                    raise ServiceError("invalid_input")
                state_limit = deadline.stage_limit("state")
                accepted = coordinator.intake.accept(
                    body,
                    token=token,
                    now_ms=coordinator.utc_ms(),
                    check_deadline=lambda: deadline.check(state_limit),
                    stream_complete=complete == "true",
                )
                deadline.check()
                sent = True
                await await_until(
                    JSONResponse(asdict(accepted))(scope, receive, send),
                    deadline.end,
                    clock=deadline.clock,
                )
            else:
                value = strict_json_loads(body)
                if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
                    raise ServiceError("invalid_input")
                for field in REQUEST_FIELDS:
                    id_bytes(value[field])
                if value["source_namespace"] != namespace:
                    raise ServiceError("namespace_forbidden")
                deadline.check(input_limit)
                result = await coordinator.recommend(
                    namespace,
                    value["session_id"],
                    value["trigger_block_id"],
                    value["idempotency_key"],
                    deadline=deadline,
                    deliver=deliver,
                )
                if not sent:
                    await JSONResponse(public_result(result), status_code=503)(scope, receive, send)
        except (ServiceError, StateError) as exc:
            if not sent:
                await _error(exc.code)(scope, receive, send)
        except TimeoutError:
            if not sent:
                await _error("backend_timeout")(scope, receive, send)
        except (InputValidationError, ValueError, TypeError, UnicodeError, RecursionError):
            if not sent:
                await _error("invalid_input")(scope, receive, send)
        except Exception:
            if not sent:
                await _error("backend_unavailable")(scope, receive, send)
        finally:
            if acquired:
                admission.release()


def create_app(coordinator: Coordinator) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_Arrival)
    app.state.admission = _Admission()
    app.state.sender_limiter = _SenderLimiter()

    @app.post("/v1/blocks")
    async def blocks(request: Request) -> Response:
        return _ServiceResponse(request, coordinator, "blocks")

    @app.post("/v1/recommendations")
    async def recommendations(request: Request) -> Response:
        return _ServiceResponse(request, coordinator, "recommendations")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request) -> Response:
        deadline = Deadline(
            start=request.scope["coveo.request_start"],
            timeout_ms=coordinator.timeout_ms,
            reserve_ms=coordinator.reserve_ms,
        )

        def inspect() -> None:
            if not coordinator.packages:
                raise ServiceError("not_ready")
            coordinator.intake.store.check_available()
            coordinator.journal.check_available()

        try:
            limit = deadline.stage_limit("state")
            await coordinator.work.run(inspect, limit)
            deadline.check(limit)
        except Exception:
            return _error("not_ready")
        return JSONResponse({"status": "ready"})

    return app
