"""Время первого HTTP-байта; разбор и транспорт остаются у h11/Uvicorn."""

import asyncio
import time
from collections import deque
from typing import Any

import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn import Config
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from .runtime import Deadline, await_until


class _ArrivalConnection(h11.Connection):
    """Снять только границы потреблённых h11 порций, не разбирать HTTP повторно."""

    def __init__(self, arrivals: deque[tuple[int, float]], max_size: int) -> None:
        super().__init__(h11.SERVER, max_incomplete_event_size=max_size)
        self.arrivals = arrivals
        self.request_start: float | None = None

    def next_event(self) -> Any:
        before = len(self.trailing_data[0])
        first = self.arrivals[0][1] if self.arrivals else None
        event = super().next_event()
        consumed = before - len(self.trailing_data[0])
        while consumed:
            size, arrived = self.arrivals.popleft()
            if size > consumed:
                self.arrivals.appendleft((size - consumed, arrived))
                break
            consumed -= size
        if isinstance(event, h11.Request):
            self.request_start = first
        return event


class TimedH11Protocol(H11Protocol):
    """Первый байт каждой заявки, включая уже принятый конвейерный остаток.

    Предназначен для закреплённого Uvicorn и обычного asyncio Python 3.11:
    созданная run_asgi-задача запускается после возврата handle_events.
    """

    def __init__(
        self,
        config: Config,
        server_state: ServerState,
        app_state: dict[str, Any],
        _loop: asyncio.AbstractEventLoop | None = None,
        *,
        timeout_ms: int = 1000,
        reserve_ms: int = 100,
    ) -> None:
        self._compute_budget = Deadline(
            0.0, timeout_ms=timeout_ms, reserve_ms=reserve_ms
        ).compute_end
        super().__init__(config, server_state, app_state, _loop)
        self._arrivals: deque[tuple[int, float]] = deque()
        self._timed_connection = _ArrivalConnection(
            self._arrivals,
            config.h11_max_incomplete_event_size
            if config.h11_max_incomplete_event_size is not None
            else DEFAULT_MAX_INCOMPLETE_EVENT_SIZE,
        )
        self.conn = self._timed_connection
        self._header_generation = 0
        self._header_start: float | None = None
        self._header_timer: asyncio.Task[None] | None = None
        self._headers_finished: asyncio.Future[None] | None = None

    def data_received(self, data: bytes) -> None:
        if data:
            self._arrivals.append((len(data), time.perf_counter()))
        super().data_received(data)

    def handle_events(self) -> None:
        previous = self.cycle
        super().handle_events()
        if self.cycle is not previous:
            first = self._timed_connection.request_start
            if first is None:
                self.transport.close()
                return
            self.scope["coveo.request_start"] = first  # type: ignore[typeddict-unknown-key]
        self._guard_incomplete_headers()

    def _cancel_header_guard(self) -> None:
        self._header_generation += 1
        if self._header_timer is not None:
            self._header_timer.cancel()
            self._header_timer = None
        if self._headers_finished is not None:
            self._headers_finished.cancel()
            self._headers_finished = None
        self._header_start = None

    def _guard_incomplete_headers(self) -> None:
        if (
            self.transport.is_closing()
            or self.conn.their_state is not h11.IDLE
            or not self._arrivals
            or (self.cycle is not None and not self.cycle.response_complete)
        ):
            self._cancel_header_guard()
            return
        first = self._arrivals[0][1]
        if self._header_timer is not None and self._header_start == first:
            return
        self._cancel_header_guard()
        self._header_start = first
        self._headers_finished = self.loop.create_future()
        self._header_timer = self.loop.create_task(
            self._wait_headers(first, self._header_generation, self._headers_finished)
        )

    async def _wait_headers(
        self, first: float, generation: int, completed: asyncio.Future[None]
    ) -> None:
        limit = first + self._compute_budget
        try:
            await await_until(completed, limit, clock=time.perf_counter)
        except TimeoutError:
            if (
                generation == self._header_generation
                and time.perf_counter() >= limit
                and self.conn.their_state is h11.IDLE
                and not self.transport.is_closing()
                and (self.cycle is None or self.cycle.response_complete)
            ):
                # До полных заголовков нет безопасного ASGI-ответа. Следующий
                # буферизованный запрос не обрывает ещё активный первый ответ.
                self.transport.close()
        except asyncio.CancelledError:
            pass
        finally:
            if generation == self._header_generation:
                self._header_timer = None
                self._headers_finished = None

    def connection_lost(self, exc: Exception | None) -> None:
        self._cancel_header_guard()
        self._arrivals.clear()
        super().connection_lost(exc)
