"""Разрешаемая отдельно управляемая проверка; это не качество и не нагрузка.

Без COVEO_SERVICE_SMOKE_RUN_DIR тест пропускается. Каталог должен содержать
только заранее закреплённый plan.json с config_sha256. Исходные сессии/цели
не открываются; SKU берутся из готового каталога, события создаются на текущем UTC.
"""

import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import id_bytes
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.process_memory import peak_memory
from coveo_recsys.serving.api import create_app
from coveo_recsys.serving.coordinator import Coordinator, public_result
from coveo_recsys.serving.package import load_serving_package
from coveo_recsys.serving.runtime import BoundedWork, ServiceError
from coveo_recsys.serving.trace import Journal
from coveo_recsys.state.service import IntakeService
from coveo_recsys.state.store import MemoryStateStore
from coveo_recsys.verification.provenance import source_sha256
from tests.unit.test_e6_state import packet

MAX_SECONDS = 60
MAX_MEMORY = 2_147_483_648
MAX_OUTPUT = 33_554_432
CASES = ("normal", "repeat", "compute_timeout", "incomplete_state")


class ControlledWork:
    """Одна реальная операция задерживается в пуле до вычислительного срока."""

    def __init__(self, work):
        self.work = work
        self.fail_next = False
        self.calls = 0
        self.forced_started = Event()
        self.forced_computed = Event()
        self.forced_finished = Event()
        self.forced_timeout = False

    async def run(self, operation, limit):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False

            def delayed():
                self.forced_started.set()
                try:
                    result = operation()
                    self.forced_computed.set()
                    Event().wait(max(0.0, limit - time.perf_counter()) + 0.010)
                    return result
                finally:
                    self.forced_finished.set()

            try:
                return await self.work.run(delayed, limit)
            except ServiceError as exc:
                self.forced_timeout = exc.code == "backend_timeout"
                raise
        return await self.work.run(operation, limit)


def disk_bytes(directory):
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


@pytest.mark.coveo
def test_controlled_actual_local_service():
    requested = os.environ.get("COVEO_SERVICE_SMOKE_RUN_DIR")
    if not requested:
        pytest.skip("Требуется отдельный допуск и COVEO_SERVICE_SMOKE_RUN_DIR")
    root = Path(__file__).resolve().parents[2]
    relative = Path(requested)
    if relative.is_absolute():
        relative = relative.relative_to(root)
    destination = LocalArtifactStore(root)._path(relative.as_posix())
    assert destination.is_dir(), "Сначала закрепите план в новом каталоге"
    assert {path.name for path in destination.iterdir()} == {"plan.json"}, (
        "Готовый запуск неизменяем"
    )
    plan = strict_json_loads((destination / "plan.json").read_bytes())
    summary = dict(
        format_version="controlled_local_service_check_v1",
        status="started",
        interpretation="controlled_events_on_actual_package_not_quality_or_load",
        business_effect="not_observed",
        consumer_source_sha256=source_sha256(root),
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            logical_cpu_count=os.cpu_count(),
        ),
        plan_sha256=sha256_file(destination / "plan.json"),
        plan=plan,
        limits=dict(
            startup_seconds=180,
            check_seconds=MAX_SECONDS,
            max_process_memory_bytes=MAX_MEMORY,
            max_result_bytes=MAX_OUTPUT,
        ),
        case_order=list(CASES),
        request_counts=dict(health=0, blocks=0, recommendations=0),
        cases=[],
    )
    started = time.perf_counter()
    checks_started = None
    pool = None
    failed = False

    def check_budget():
        memory = peak_memory()["peak_process_memory_bytes"]
        assert memory is not None and memory <= MAX_MEMORY, "Превышен бюджет памяти проверки"
        assert disk_bytes(destination) <= MAX_OUTPUT, "Превышен бюджет результата проверки"
        if checks_started is not None:
            assert time.perf_counter() - checks_started <= MAX_SECONDS, (
                "Превышен бюджет времени проверки"
            )

    try:
        config_path = root / "configs/coveo_service.json"
        encoded_config = config_path.read_bytes()
        config = strict_json_loads(encoded_config)
        assert plan["config_sha256"] == sha256_bytes(encoded_config), (
            "Конфигурация не соответствует плану"
        )
        assert plan["sessions"] == 4 and plan["cases"] == [
            "ordinary",
            "block_and_request_repeat",
            "forced_compute_timeout",
            "incomplete_stream",
        ], "Сценарии не соответствуют закреплённому плану"
        assert plan["limits"] == dict(
            startup_seconds=180,
            check_seconds_after_startup=MAX_SECONDS,
            peak_process_memory_bytes=MAX_MEMORY,
            output_bytes=MAX_OUTPUT,
        ), "Бюджеты не соответствуют закреплённому плану"
        assert plan["quality_evaluation"] is False and plan["normative_load"] is False
        summary["config_sha256"] = sha256_bytes(encoded_config)
        assert config["package"]["limits"] == dict(
            max_metadata_rows=100000,
            max_metadata_bytes=104857600,
            max_memory_bytes=MAX_MEMORY,
            max_seconds=180,
        ), "Startup-бюджеты изменены"
        assert config["state_adapter"] == "memory" and config["workers"] == 4
        assert config["owners"] == {"COVEO_LOCAL_TOKEN": "local-demo"}
        timeout_ms = config["timeout_ms"]
        reserve_ms = config["reserve_ms"]
        summary["runtime"] = dict(
            timeout_ms=timeout_ms, reserve_ms=reserve_ms, workers=config["workers"]
        )
        token = os.environ["COVEO_LOCAL_TOKEN"]
        assert token, "Не задан секрет локального отправителя"
        task = TaskConfig(**config["task"])
        namespace = "local-demo"
        startup = time.perf_counter()
        package = load_serving_package(root, config["package"])
        summary["startup"] = dict(
            elapsed_seconds=time.perf_counter() - startup,
            **peak_memory(),
            metadata_rows=package.metadata_rows,
        )
        assert summary["startup"]["elapsed_seconds"] <= 180, "Превышен бюджет запуска"
        check_budget()
        checks_started = time.perf_counter()
        skus = sorted(package.engine.index.positions, key=id_bytes)[:4]
        assert len(skus) == 4, "Нужны четыре SKU готового каталога"
        summary["selected_skus"] = skus
        summary["selection_rule"] = "first_4_catalog_skus_utf8_no_targets"
        summary["package_id"] = package.package_id
        summary["package_hashes"] = dict(package.hashes)
        summary["published_at_ms"] = package.published_at_ms
        store = MemoryStateStore()
        intake = IntakeService(
            store, task, {token: namespace}, config["calendar_start_ms"], config["calendar_end_ms"]
        )
        journal = Journal(destination / "requests.jsonl")
        pool = BoundedWork(workers=4)
        work = ControlledWork(pool)
        coordinator = Coordinator(
            (package,), intake, journal, work=work, timeout_ms=timeout_ms, reserve_ms=reserve_ms
        )
        authorization = {"Authorization": f"Bearer {token}"}

        with TestClient(create_app(coordinator)) as client:
            summary["request_counts"]["health"] += 1
            ready = client.get("/health/ready")
            summary["ready"] = dict(http_status=ready.status_code, body=ready.json())
            assert ready.status_code == 200, "Проверка готовности не прошла"
            check_budget()

            def check_case(name, sku):
                row = dict(case=name, session_id=f"controlled-{name}", sku=sku)
                summary["cases"].append(row)
                case_start = time.perf_counter()
                sid = row["session_id"]
                event_time = coordinator.utc_ms()
                assert package.published_at_ms < event_time, "Публикация не предшествует событию"
                assert config["calendar_start_ms"] <= event_time < config["calendar_end_ms"]
                wire = packet(
                    event_time,
                    block=name,
                    namespace=namespace,
                    session=sid,
                    events=[(f"event-{name}", "detail", sku)],
                )
                headers = {
                    **authorization,
                    "X-Stream-Complete": str(name != "incomplete_state").lower(),
                }
                begin = time.perf_counter()
                summary["request_counts"]["blocks"] += 1
                receipt = client.post("/v1/blocks", content=wire, headers=headers)
                row["block"] = dict(
                    http_status=receipt.status_code,
                    body=receipt.json(),
                    elapsed_ms=(time.perf_counter() - begin) * 1000,
                )
                assert receipt.status_code == 200 and receipt.json()["status"] == "accepted"
                assert row["block"]["elapsed_ms"] <= timeout_ms, "Приём блока превысил общий срок"
                check_budget()
                query = dict(
                    source_namespace=namespace,
                    session_id=sid,
                    trigger_block_id=name,
                    idempotency_key=f"key-{name}",
                )
                work.fail_next = name == "compute_timeout"
                calls_before = work.calls
                begin = time.perf_counter()
                summary["request_counts"]["recommendations"] += 1
                response = client.post("/v1/recommendations", json=query, headers=authorization)
                row["http"] = dict(
                    status=response.status_code,
                    body=response.json(),
                    elapsed_ms=(time.perf_counter() - begin) * 1000,
                )
                row["computation_calls"] = work.calls - calls_before
                if name == "compute_timeout":
                    row["forced_compute_timeout"] = dict(
                        started=work.forced_started.is_set(),
                        computed=work.forced_computed.is_set(),
                        timed_out=work.forced_timeout,
                    )
                row["state_sha256_after_response"] = sha256_bytes(
                    to_json(store.read(namespace, sid))
                )
                reservation = journal.lookup(namespace, sid)
                assert reservation is not None and reservation.result is not None, (
                    "Нет записанного исхода"
                )
                result = reservation.result
                row["journal"] = dict(
                    key=reservation.key,
                    result=public_result(result),
                    result_sha256=sha256_bytes(to_json(result)),
                    candidate_count=len(result.candidates.skus),
                    elapsed_ms=result.elapsed_ms,
                )
                row["http_matches_journal"] = response.json() == public_result(result)
                assert row["http_matches_journal"], "Ответ не соответствует итоговой записи"
                assert (
                    set(result.delivered_skus)
                    <= set(result.candidates.skus)
                    <= set(package.engine.index.positions)
                )
                if name == "repeat":
                    state_before = to_json(store.read(namespace, sid))
                    record_before = sha256_file(journal.path)
                    calls_before = work.calls
                    repeat_start = time.perf_counter()
                    summary["request_counts"]["blocks"] += 1
                    block_again = client.post("/v1/blocks", content=wire, headers=headers)
                    block_elapsed_ms = (time.perf_counter() - repeat_start) * 1000
                    repeat_start = time.perf_counter()
                    summary["request_counts"]["recommendations"] += 1
                    again = client.post("/v1/recommendations", json=query, headers=authorization)
                    request_elapsed_ms = (time.perf_counter() - repeat_start) * 1000
                    repeated = journal.lookup(namespace, sid)
                    row["repeat"] = dict(
                        block_status=block_again.json()["status"],
                        block_elapsed_ms=block_elapsed_ms,
                        request_elapsed_ms=request_elapsed_ms,
                        http_status=again.status_code,
                        http=again.json(),
                        same_result=repeated == reservation,
                        same_state=to_json(store.read(namespace, sid)) == state_before,
                        same_journal=sha256_file(journal.path) == record_before,
                        recomputations=work.calls - calls_before,
                    )
                    assert block_again.status_code == 200 and again.status_code == 200
                    assert block_again.json()["status"] == "duplicate"
                    assert repeated == reservation and repeated.key == query["idempotency_key"]
                    assert row["repeat"]["same_state"] and row["repeat"]["same_journal"]
                    assert work.calls == calls_before and again.json() == public_result(result)
                    assert block_elapsed_ms <= timeout_ms, "Повтор блока превысил общий срок"
                    assert request_elapsed_ms <= timeout_ms, "Полный повтор превысил срок ответа"
                assert row["http"]["elapsed_ms"] <= timeout_ms, (
                    "Полный HTTP-вызов с итоговой записью превысил срок ответа"
                )
                if name in {"normal", "repeat"}:
                    assert response.status_code == 200, "HTTP не подтвердил доставку"
                    assert result.candidates.branch.value == "primary", "Не выбрана основная ветвь"
                    assert result.delivery.value == "delivered", "Нет успешной локальной доставки"
                    assert result.delivered_skus, "Основная ветвь не доставила рекомендации"
                elif name == "compute_timeout":
                    assert all(row["forced_compute_timeout"].values()), (
                        "Тайм-аут не достиг реального вычисления в рабочем пуле"
                    )
                    assert row["http"]["elapsed_ms"] >= timeout_ms - reserve_ms, (
                        "Вычисление остановилось раньше общего вычислительного срока"
                    )
                    assert result.candidates.branch.value == "fallback"
                    assert response.status_code == 200, "HTTP не подтвердил резерв"
                    assert (
                        result.delivery.value == "delivered" and "backend_timeout" in result.reasons
                    )
                    assert result.delivered_skus, "Резерв не доставил рекомендации"
                else:
                    assert (
                        result.candidates.branch.value == "none"
                        and "state_incomplete" in result.reasons
                    )
                    assert not result.candidates.skus and not result.delivered_skus
                    assert result.delivery.value == "not_attempted" and not result.explanations
                    assert row["computation_calls"] == 0, "Неполное состояние дошло до вычислений"
                row["elapsed_seconds"] = time.perf_counter() - case_start
                row["resources"] = dict(**peak_memory(), result_bytes=disk_bytes(destination))
                check_budget()

            failures = []
            for name, sku in zip(CASES, skus, strict=True):
                try:
                    check_case(name, sku)
                except AssertionError as exc:
                    summary["cases"][-1].update(status="failed", error=str(exc))
                    failures.append(name)
                else:
                    summary["cases"][-1]["status"] = "passed"
                check_budget()
            if work.forced_started.is_set():
                summary["late_computation_finished"] = work.forced_finished.wait(timeout=1)
                assert summary["late_computation_finished"], "Позднее вычисление не завершилось"
            reopened = Journal(journal.path)
            for row in summary["cases"]:
                if "journal" not in row:
                    continue
                saved = reopened.lookup(namespace, row["session_id"])
                row["http_matches_reopened_journal"] = (
                    saved is not None
                    and saved.result is not None
                    and row["http"]["body"] == public_result(saved.result)
                )
                row["same_terminal_result_after_late_computation"] = (
                    saved is not None
                    and saved.result is not None
                    and sha256_bytes(to_json(saved.result)) == row["journal"]["result_sha256"]
                )
                row["same_state_after_late_computation"] = (
                    sha256_bytes(to_json(store.read(namespace, row["session_id"])))
                    == row["state_sha256_after_response"]
                )
                if not (
                    row["http_matches_reopened_journal"]
                    and row["same_terminal_result_after_late_computation"]
                    and row["same_state_after_late_computation"]
                ):
                    row.update(status="failed", error="Изменился итоговый журнал или состояние")
                    if row["case"] not in failures:
                        failures.append(row["case"])
            check_budget()
            assert not failures, f"Не пройдены сценарии: {', '.join(failures)}"
        summary["task"] = asdict(task)
        summary["status"] = "completed"
    except BaseException as exc:
        failed = True
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        raise
    finally:
        if pool is not None:
            pool.close()
        summary["consumer_source_sha256_after"] = source_sha256(root)
        summary["consumer_source_stable"] = (
            summary["consumer_source_sha256_after"] == summary["consumer_source_sha256"]
        )
        if not summary["consumer_source_stable"]:
            summary["status"] = "failed"
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["check_elapsed_seconds"] = (
            None if checks_started is None else time.perf_counter() - checks_started
        )
        summary["resources"] = dict(
            **peak_memory(), result_bytes_before_summary=disk_bytes(destination)
        )
        memory = summary["resources"]["peak_process_memory_bytes"]
        exceeded = (
            memory is None
            or memory > MAX_MEMORY
            or (checks_started is not None and summary["check_elapsed_seconds"] > MAX_SECONDS)
            or disk_bytes(destination) + len(canonical_json_bytes(summary)) > MAX_OUTPUT
        )
        if exceeded:
            summary["status"] = "failed"
            summary["budget_exceeded"] = True
        with (destination / "summary.json").open("xb") as output:
            output.write(canonical_json_bytes(summary))
        if exceeded and not failed:
            raise AssertionError("Превышен закреплённый бюджет; повтор и усечение запрещены")
        if not summary["consumer_source_stable"] and not failed:
            raise AssertionError("Исходники потребителя изменились во время проверки")
