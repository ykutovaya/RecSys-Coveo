"""Локальный HTTP: транспортные границы без CSV и готовых моделей."""

import asyncio
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from coveo_recsys.contracts.service_state import StateError


@dataclass
class ReceiptResult:
    status: str = "accepted"
    receipt: str = "receipt"
    decision_time_ms: int = 2000


class Dependency:
    def __init__(self):
        self.checks = 0
        self.fail = False

    def check_available(self):
        self.checks += 1
        if self.fail:
            raise OSError("private path and raw secret")


class Intake:
    def __init__(self):
        self.store = Dependency()
        self.calls = []

    def namespace_for(self, token):
        if token != "private-token":
            raise StateError("unauthorized")
        return "shop"

    def accept(self, payload, **kwargs):
        kwargs["check_deadline"]()
        self.calls.append((payload, kwargs))
        return ReceiptResult()


class Work:
    async def run(self, operation, limit):
        return operation()


class Coordinator:
    def __init__(self):
        self.timeout_ms = 1000
        self.reserve_ms = 100
        self.intake = Intake()
        self.journal = Dependency()
        self.packages = (SimpleNamespace(package_id="package", published_at_ms=1000),)
        self.work = Work()
        self.calls = []
        self.send_failed = False
        self.pause = None

    def utc_ms(self):
        return 2000

    async def recommend(self, *args, deadline, deliver):
        self.calls.append((args, deadline))
        if self.pause is not None:
            await self.pause.wait()
        try:
            await deliver(
                dict(
                    query_id="q",
                    package_id="package",
                    items=["sku"],
                    branch="primary",
                    delivery="delivered",
                    reasons=[],
                    explanations=[],
                    private_features=[99],
                )
            )
        except OSError:
            self.send_failed = True


@pytest.fixture
def service():
    from coveo_recsys.serving import api

    coordinator = Coordinator()
    return api.create_app(coordinator), coordinator


async def request(
    app,
    path="/v1/recommendations",
    *,
    method="POST",
    payload=None,
    token="private-token",
    extra_headers=(),
    delay=0,
    fail_send=False,
    send_delay=0,
    receive_forbidden=False,
):
    body = (
        json.dumps(
            dict(source_namespace="shop", session_id="s", trigger_block_id="b", idempotency_key="k")
        ).encode()
        if payload is None
        else payload
    )
    headers = list(extra_headers)
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = dict(
        type="http",
        asgi={"version": "3.0"},
        method=method,
        path=path,
        raw_path=path.encode(),
        root_path="",
        query_string=b"",
        headers=headers,
        http_version="1.1",
        scheme="http",
        server=("127.0.0.1", 8000),
        client=("127.0.0.1", 50000),
    )
    sent = []

    async def receive():
        assert not receive_forbidden, "Тело прочитано до транспортного допуска"
        if delay:
            await asyncio.sleep(delay)
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)
        if fail_send and message["type"] == "http.response.body":
            raise OSError("client disconnected")
        if send_delay and message["type"] == "http.response.body":
            await asyncio.sleep(send_delay)

    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    encoded = b"".join(message.get("body", b"") for message in sent)
    return status, json.loads(encoded), sent


def test_recommendation_uses_actual_asgi_send_and_only_public_fields(service):
    app, coordinator = service
    status, body, sent = asyncio.run(request(app))
    assert status == 200
    assert set(body) == {
        "query_id",
        "package_id",
        "items",
        "branch",
        "delivery",
        "reasons",
        "explanations",
    }
    assert body["items"] == ["sku"]
    assert coordinator.calls[0][0] == ("shop", "s", "b", "k")
    assert sum(row["type"] == "http.response.start" for row in sent) == 1


def test_send_failure_reaches_coordinator_without_second_response(service):
    app, coordinator = service
    _, _, sent = asyncio.run(request(app, fail_send=True))
    assert coordinator.send_failed
    assert sum(row["type"] == "http.response.start" for row in sent) == 1


@pytest.mark.parametrize("token", [None, "unknown"])
def test_unauthorized_request_is_rejected_before_body_or_state(service, token):
    app, coordinator = service
    status, body, _ = asyncio.run(request(app, token=token, receive_forbidden=True))
    assert status == 401 and body == {"reason": "unauthorized"}
    assert not coordinator.calls and not coordinator.intake.calls


@pytest.mark.parametrize(
    "payload,reason,status",
    [
        (b'{"source_namespace":"shop","source_namespace":"other"}', "invalid_input", 400),
        (
            b'{"source_namespace":"other","session_id":"s","trigger_block_id":"b",'
            b'"idempotency_key":"k"}',
            "namespace_forbidden",
            403,
        ),
        (
            b'{"source_namespace":"shop","session_id":"s","trigger_block_id":"b",'
            b'"idempotency_key":"k","extra":1}',
            "invalid_input",
            400,
        ),
        (
            b'{"source_namespace":"shop","session_id":1,"trigger_block_id":"b",'
            b'"idempotency_key":"k"}',
            "invalid_input",
            400,
        ),
        (b"[]", "invalid_input", 400),
        (b"x" * (1_048_576 + 1), "invalid_input", 413),
    ],
    ids=["duplicate-key", "foreign-namespace", "extra-key", "wrong-type", "array", "oversize"],
)
def test_strict_recommendation_input_before_expensive_state(service, payload, reason, status):
    app, coordinator = service
    actual_status, body, _ = asyncio.run(request(app, payload=payload))
    assert actual_status == status and body == {"reason": reason}
    assert not coordinator.calls


def test_stalled_first_body_is_bounded_by_original_input_deadline(service):
    app, coordinator = service
    coordinator.timeout_ms, coordinator.reserve_ms = 100, 50
    start = time.monotonic()
    status, body, _ = asyncio.run(request(app, delay=0.2))
    assert status == 504 and body == {"reason": "backend_timeout"}
    assert time.monotonic() - start < 0.15
    assert not coordinator.calls


def test_delay_before_routing_consumes_original_application_input_budget(service):
    app, coordinator = service
    coordinator.timeout_ms, coordinator.reserve_ms = 20, 10
    router = app.router

    async def delayed_route(scope, receive, send):
        await asyncio.sleep(0.020)
        await router(scope, receive, send)

    app.router = delayed_route
    status, body, _ = asyncio.run(request(app, receive_forbidden=True))
    assert status == 504 and body == {"reason": "backend_timeout"}
    assert not coordinator.calls


@pytest.mark.parametrize("protocol_start", [None, 1.25], ids=["asgi-only", "protocol-first-byte"])
def test_arrival_preserves_protocol_clock_or_sets_missing_clock(monkeypatch, protocol_start):
    from coveo_recsys.serving import api

    scope = {"type": "http"}
    if protocol_start is not None:
        scope["coveo.request_start"] = protocol_start
    observed = []

    async def downstream(actual, receive, send):
        observed.append(actual["coveo.request_start"])

    monkeypatch.setattr(api.time, "perf_counter", lambda: 2.5)
    asyncio.run(api._Arrival(downstream)(scope, None, None))
    assert observed == [2.5 if protocol_start is None else protocol_start]


@pytest.mark.parametrize("complete", [False, True])
def test_block_receipt_preserves_raw_bytes_and_explicit_stream_signal(service, complete):
    app, coordinator = service
    headers = [(b"x-stream-complete", b"true")] if complete else []
    raw = b'{ "opaque_exact_input" : 1 }'
    status, body, _ = asyncio.run(request(app, "/v1/blocks", payload=raw, extra_headers=headers))
    assert status == 200 and body["receipt"] == "receipt"
    payload, parameters = coordinator.intake.calls[0]
    assert payload == raw and parameters["stream_complete"] is complete
    assert parameters["now_ms"] == 2000


def test_missing_events_is_invalid_input_through_real_intake(service):
    from coveo_recsys.contracts.preparation import TaskConfig
    from coveo_recsys.state.service import IntakeService
    from coveo_recsys.state.store import MemoryStateStore
    from tests.unit.test_e6_state import packet

    app, coordinator = service
    coordinator.intake = IntakeService(
        MemoryStateStore(), TaskConfig(), {"private-token": "shop"}, 0, 10_000
    )
    header = json.loads(packet(namespace="shop"))
    del header["events"]
    status, body, _ = asyncio.run(request(app, "/v1/blocks", payload=json.dumps(header).encode()))
    assert status == 200
    assert body["status"] == "rejected" and body["reason"] == "invalid_input"
    state = coordinator.intake.store.read("shop", "s")
    assert not state.blocks and not state.stream_complete
    assert state.gap_reasons == ("invalid_input",)


def test_slow_receipt_send_cannot_hold_admission_past_deadline(service):
    app, coordinator = service
    coordinator.timeout_ms, coordinator.reserve_ms = 100, 50
    start = time.perf_counter()
    _, _, sent = asyncio.run(request(app, "/v1/blocks", payload=b"{}", send_delay=0.3))
    assert time.perf_counter() - start < 0.22
    assert len(coordinator.intake.calls) == 1
    assert app.state.admission.count == 0
    assert sum(row["type"] == "http.response.start" for row in sent) == 1


def test_sender_limit_charges_invalid_json_before_session_parsing(service):
    app, coordinator = service
    app.state.sender_limiter.clock = lambda: 1.0

    async def attempts():
        for _ in range(100):
            status, _, _ = await request(app, payload=b"invalid-json")
            assert status == 400
        return await request(app, payload=b"invalid-json", receive_forbidden=True)

    status, body, _ = asyncio.run(attempts())
    assert status == 429 and body == {"reason": "rate_rejected"}
    assert not coordinator.calls


def test_live_does_not_probe_or_mutate_user_state(service):
    app, coordinator = service
    coordinator.packages = ()
    coordinator.journal.fail = True
    status, body, _ = asyncio.run(request(app, "/health/live", method="GET", token=None))
    assert status == 200 and body == {"status": "live"}
    assert coordinator.intake.store.checks == 0 and coordinator.journal.checks == 0


@pytest.mark.parametrize("failed", ["packages", "store", "journal"])
def test_ready_requires_all_dependencies_without_exposing_internal_error(service, failed):
    app, coordinator = service
    if failed == "packages":
        coordinator.packages = ()
    else:
        dependency = coordinator.intake.store if failed == "store" else coordinator.journal
        dependency.fail = True
    status, body, _ = asyncio.run(request(app, "/health/ready", method="GET", token=None))
    assert status == 503 and body == {"reason": "not_ready"}


def test_ready_success_probes_store_and_mandatory_journal(service):
    app, coordinator = service
    status, body, _ = asyncio.run(request(app, "/health/ready", method="GET", token=None))
    assert status == 200 and body == {"status": "ready"}
    assert coordinator.intake.store.checks == coordinator.journal.checks == 1


@pytest.mark.parametrize("warm", [False, True], ids=["cold-pool", "warm-pool"])
def test_ready_real_local_dependencies_with_bounded_work(tmp_path, warm):
    from fastapi.testclient import TestClient

    from coveo_recsys.serving.api import create_app
    from coveo_recsys.serving.runtime import BoundedWork
    from coveo_recsys.serving.trace import Journal
    from coveo_recsys.state.store import MemoryStateStore
    from tests.integration.test_e6_actual_service import ControlledWork

    pool = BoundedWork()
    if warm:
        asyncio.run(pool.run(lambda: None, time.perf_counter() + 1))
    coordinator = SimpleNamespace(
        timeout_ms=1000,
        reserve_ms=100,
        packages=(object(),),
        intake=SimpleNamespace(store=MemoryStateStore()),
        journal=Journal(tmp_path / "requests.jsonl"),
        work=ControlledWork(pool),
    )
    try:
        with TestClient(create_app(coordinator)) as client:
            response = client.get("/health/ready")
    finally:
        pool.close()
    assert response.status_code == 200


def test_admission_releases_slot_acquired_at_expired_boundary(monkeypatch):
    from coveo_recsys.serving import api

    async def expired_after_acquire(operation, limit, *, clock):
        await operation
        raise TimeoutError

    monkeypatch.setattr(api, "await_until", expired_after_acquire)

    async def scenario():
        gate = api._Admission()
        with pytest.raises(TimeoutError):
            await gate.acquire(2.0, lambda: 1.0)
        assert gate.count == 0 and gate.slots._value == 16

    asyncio.run(scenario())


def test_ready_slow_dependency_does_not_hold_connection_indefinitely(service):
    from coveo_recsys.serving.runtime import BoundedWork

    app, coordinator = service
    coordinator.timeout_ms, coordinator.reserve_ms = 100, 50
    pool = BoundedWork(1)
    coordinator.work = pool
    coordinator.journal.check_available = lambda: time.sleep(0.2)
    try:
        start = time.perf_counter()
        status, body, _ = asyncio.run(request(app, "/health/ready", method="GET", token=None))
        assert status == 503 and body == {"reason": "not_ready"}
        assert time.perf_counter() - start < 0.15
    finally:
        pool.close()


def test_node_admission_is_16_active_plus_16_waiting_before_body(service):
    app, coordinator = service
    coordinator.timeout_ms = 10000

    async def concurrent():
        coordinator.pause = asyncio.Event()
        tasks = [asyncio.create_task(request(app)) for _ in range(32)]
        try:
            for _ in range(20):
                await asyncio.sleep(0)
                if len(coordinator.calls) == 16 and app.state.admission.count == 32:
                    break
            assert len(coordinator.calls) == 16 and app.state.admission.count == 32
            status, body, _ = await request(app, receive_forbidden=True)
            assert status == 503 and body == {"reason": "overload_rejected"}
        finally:
            coordinator.pause.set()
            await asyncio.gather(*tasks)
        assert app.state.admission.count == 0

    asyncio.run(concurrent())


@pytest.fixture
def serving_configuration(tmp_path):
    from dataclasses import asdict

    from coveo_recsys.contracts.preparation import TaskConfig

    settings = dict(
        package=dict(
            historical={"path": "inputs/history"},
            model={"path": "inputs/model"},
            training_features={"path": "inputs/features"},
            feature_schema={"path": "inputs/schema.json"},
            metadata={"path": "inputs/content.csv", "profile_path": "inputs/profile.json"},
        ),
        task=asdict(TaskConfig()),
        owners={"COVEO_TEST_TOKEN": "shop"},
        calendar_start_ms=0,
        calendar_end_ms=10_000,
        state_adapter="memory",
        workers=4,
        timeout_ms=1500,
        reserve_ms=200,
    )
    path = tmp_path / "serve.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return tmp_path, path, settings


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost", "127.0.0.2", "192.168.1.1"])
def test_cli_refuses_nonpermitted_host_before_loading(serving_configuration, monkeypatch, host):
    from coveo_recsys.cli import serving

    root, config, _ = serving_configuration
    monkeypatch.setattr(serving, "load_serving_package", lambda *_: pytest.fail("package read"))
    with pytest.raises(ValueError):
        serving.run_serve(root, config, root / "run", host=host)


@pytest.mark.parametrize("adapter", ["memory", "file"])
@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_cli_builds_one_local_process_with_env_auth_only(
    serving_configuration, monkeypatch, adapter, host
):
    from coveo_recsys.cli import serving
    from coveo_recsys.serving.protocol import TimedH11Protocol
    from coveo_recsys.state.store import FileStateStore, MemoryStateStore

    root, config, settings = serving_configuration
    settings["state_adapter"] = adapter
    config.write_text(json.dumps(settings), encoding="utf-8")
    before = config.read_bytes()
    monkeypatch.setenv("COVEO_TEST_TOKEN", "not-in-config-or-log")
    package = SimpleNamespace(
        schema={"event_semantics": "set_v1"},
        engine=SimpleNamespace(
            index=SimpleNamespace(
                package=SimpleNamespace(metadata_contract={"description_dimension": 2})
            )
        ),
    )
    loads, controllers, launches = [], [], []

    def loaded(actual_root, manifest):
        loads.append((actual_root, manifest))
        return package

    def controller(packages, intake, journal, *, work, timeout_ms, reserve_ms):
        assert (timeout_ms, reserve_ms) == (1500, 200)
        result = SimpleNamespace(packages=packages, intake=intake, journal=journal, work=work)
        controllers.append(result)
        return result

    monkeypatch.setattr(serving, "load_serving_package", loaded)
    monkeypatch.setattr(serving, "Coordinator", controller)
    monkeypatch.setattr(serving, "create_app", lambda coordinator: coordinator)
    monkeypatch.setattr(
        serving.uvicorn, "run", lambda app, **kwargs: launches.append((app, kwargs))
    )
    serving.run_serve(root, config, root / "run", host=host, port=8123)
    assert len(loads) == len(controllers) == len(launches) == 1
    controller = controllers[0]
    assert controller.intake.namespace_for("not-in-config-or-log") == "shop"
    assert type(controller.intake.store) is (
        MemoryStateStore if adapter == "memory" else FileStateStore
    )
    assert controller.journal.path == root / "run/requests.jsonl"
    options = launches[0][1]
    assert options["host"] == host and options["port"] == 8123 and options["workers"] == 1
    assert options["access_log"] is False and options["proxy_headers"] is False
    assert options["http"].func is TimedH11Protocol and options["ws"] == "none"
    assert options["http"].keywords == {"timeout_ms": 1500, "reserve_ms": 200}
    assert config.read_bytes() == before
    assert not any(
        b"not-in-config-or-log" in path.read_bytes()
        for path in (root / "run").rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize("change", ["missing-token", "extra-setting", "unknown-adapter", "workers"])
def test_cli_rejects_unsafe_config_before_package_read(serving_configuration, monkeypatch, change):
    from coveo_recsys.cli import serving

    root, config, settings = serving_configuration
    monkeypatch.setenv("COVEO_TEST_TOKEN", "secret")
    if change == "missing-token":
        monkeypatch.delenv("COVEO_TEST_TOKEN")
    elif change == "extra-setting":
        settings["unexpected"] = True
    elif change == "unknown-adapter":
        settings["state_adapter"] = "production"
    else:
        settings["workers"] = True
    config.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(serving, "load_serving_package", lambda *_: pytest.fail("package read"))
    with pytest.raises(ValueError):
        serving.run_serve(root, config, root / "run")


def test_cli_output_cannot_overlap_producer(serving_configuration, monkeypatch):
    from coveo_recsys.cli import serving

    root, config, _ = serving_configuration
    monkeypatch.setenv("COVEO_TEST_TOKEN", "secret")
    monkeypatch.setattr(serving, "load_serving_package", lambda *_: pytest.fail("package read"))
    with pytest.raises(ValueError):
        serving.run_serve(root, config, root / "inputs/history/service")
