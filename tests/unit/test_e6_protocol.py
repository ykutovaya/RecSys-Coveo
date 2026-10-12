"""Первый байт HTTP проверяется без сокетов и собственного разбора запросов."""

import asyncio
from functools import partial

import pytest
from uvicorn import Config
from uvicorn.server import ServerState


class Transport(asyncio.Transport):
    def __init__(self):
        self.closed = False
        self.writes = []
        self.paused = False

    def get_extra_info(self, name, default=None):
        return {"sockname": ("127.0.0.1", 8000), "peername": ("127.0.0.1", 4321)}.get(name, default)

    def is_closing(self):
        return self.closed

    def close(self):
        self.closed = True

    def write(self, value):
        self.writes.append(value)

    def pause_reading(self):
        self.paused = True

    def resume_reading(self):
        self.paused = False


class Application:
    def __init__(self, *, hold=(), early=()):
        self.scopes = []
        self.gates = {path: asyncio.Event() for path in hold}
        self.early = early

    async def __call__(self, scope, receive, send):
        self.scopes.append(dict(scope))
        if scope["path"] not in self.early:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if not message.get("more_body", False):
                    break
        if scope["path"] in self.gates:
            await self.gates[scope["path"]].wait()
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"0")]}
        )
        await send({"type": "http.response.body", "body": b""})


def request(path):
    return f"GET {path} HTTP/1.1\r\nHost: local\r\n\r\n".encode()


def protocol(monkeypatch, clock, app, **timing):
    from coveo_recsys.serving import protocol as module

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    state = ServerState()
    config = Config(
        app,
        interface="asgi3",
        lifespan="off",
        ws="none",
        log_config=None,
        access_log=False,
        http=partial(module.TimedH11Protocol, **timing),
    )
    config.load()
    instance = config.http_protocol_class(config, state, {}, _loop=asyncio.get_running_loop())
    transport = Transport()
    instance.connection_made(transport)
    return instance, transport, state


async def turn():
    for _ in range(6):
        await asyncio.sleep(0)


async def finish(instance, state):
    instance.connection_lost(None)
    await asyncio.gather(*tuple(state.tasks), return_exceptions=True)
    await turn()


def test_fragmented_headers_and_body_keep_first_byte_time(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application()
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(b"POST /one HTT")
        await turn()
        assert app.scopes == []
        clock[0] = 1.040
        instance.data_received(b"P/1.1\r\nHost: local\r\nContent-Length: 3\r\n\r\na")
        await turn()
        assert app.scopes[0]["coveo.request_start"] == 1.0
        assert transport.writes == []
        clock[0] = 1.080
        instance.data_received(b"bc")
        await turn()
        assert not transport.closed and len(app.scopes) == 1
        await finish(instance, state)

    asyncio.run(scenario())


def test_keep_alive_idle_does_not_start_next_request_clock(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application()
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(request("/one"))
        await turn()
        clock[0] = 101.0
        await turn()
        assert not transport.closed
        instance.data_received(request("/two"))
        await turn()
        assert [scope["coveo.request_start"] for scope in app.scopes] == [1.0, 101.0]
        assert not instance._arrivals
        await finish(instance, state)

    asyncio.run(scenario())


@pytest.mark.parametrize("same_chunk", [True, False])
def test_pipeline_preserves_each_buffered_first_byte_without_closing_active_response(
    monkeypatch, same_chunk
):
    async def scenario():
        clock, app = [1.0], Application(hold=("/one", "/two"))
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(request("/one") + (request("/two") if same_chunk else b""))
        await turn()
        clock[0] = 1.003
        if not same_chunk:
            instance.data_received(request("/two"))
        clock[0] = 1.006
        instance.data_received(request("/three"))
        clock[0] = 1.030
        await turn()
        assert not transport.closed and len(app.scopes) == 1
        app.gates["/one"].set()
        await turn()
        assert app.scopes[1]["coveo.request_start"] == (1.0 if same_chunk else 1.003)
        assert not transport.closed
        app.gates["/two"].set()
        await turn()
        assert app.scopes[2]["coveo.request_start"] == 1.006
        await finish(instance, state)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "timing,elapsed,closed",
    [
        ({}, 0.020, False),
        ({}, 0.899, False),
        ({}, 0.900, True),
        ({}, 0.901, True),
        ({"timeout_ms": 175, "reserve_ms": 25}, 0.149, False),
        ({"timeout_ms": 175, "reserve_ms": 25}, 0.150, True),
        ({"timeout_ms": 175, "reserve_ms": 25}, 0.151, True),
    ],
)
def test_expired_incomplete_headers_close_without_starting_application(
    monkeypatch, timing, elapsed, closed
):
    async def scenario():
        clock, app = [1.0], Application()
        instance, transport, state = protocol(monkeypatch, clock, app, **timing)
        instance.data_received(b"GET /one HTTP/1.1\r\nHo")
        clock[0] = 1.0 + elapsed
        await turn()
        assert transport.closed is closed and app.scopes == [] and transport.writes == []
        await finish(instance, state)

    asyncio.run(scenario())


def test_pending_partial_headers_do_not_close_active_response(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application(hold=("/one",))
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(request("/one"))
        await turn()
        clock[0] = 1.002
        instance.data_received(b"GET /two HTTP/1.1\r\nHo")
        clock[0] = 2.030
        await turn()
        assert not transport.closed and len(app.scopes) == 1
        app.gates["/one"].set()
        await turn()
        assert transport.closed and len(app.scopes) == 1
        await finish(instance, state)

    asyncio.run(scenario())


def test_old_header_timeout_cannot_close_new_complete_request(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application(hold=("/two",))
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(b"GET /one HTTP/1.1\r\n")
        await turn()
        clock[0] = 1.005
        instance.data_received(b"Host: local\r\n\r\n")
        await turn()
        clock[0] = 2.030
        instance.data_received(request("/two"))
        await turn()
        assert not transport.closed
        assert [scope["coveo.request_start"] for scope in app.scopes] == [1.0, 2.030]
        app.gates["/two"].set()
        await turn()
        await finish(instance, state)

    asyncio.run(scenario())


def test_early_response_does_not_attribute_old_body_time_to_next_request(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application(early=("/early",))
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(b"POST /early HTTP/1.1\r\nHost: local\r\nContent-Length: 3\r\n\r\n")
        await turn()
        clock[0] = 1.002
        instance.data_received(b"a")
        clock[0] = 1.004
        instance.data_received(b"bc" + request("/next"))
        await turn()
        assert not transport.closed
        assert [scope["coveo.request_start"] for scope in app.scopes] == [1.0, 1.004]
        await finish(instance, state)

    asyncio.run(scenario())


def test_connection_lost_cancels_incomplete_header_guard(monkeypatch):
    async def scenario():
        clock, app = [1.0], Application()
        instance, transport, state = protocol(monkeypatch, clock, app)
        instance.data_received(b"GET /one HTTP/1.1\r\n")
        await turn()
        await finish(instance, state)
        assert transport.closed and app.scopes == []
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario())
