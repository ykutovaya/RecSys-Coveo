"""Один локальный процесс API на готовом пакете, без подготовки и обучения."""

import ipaddress
import os
import re
from functools import partial
from pathlib import Path

import uvicorn

from coveo_recsys.artifacts import LocalArtifactStore, strict_json_loads
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.serving.api import create_app
from coveo_recsys.serving.coordinator import Coordinator
from coveo_recsys.serving.package import load_serving_package
from coveo_recsys.serving.protocol import TimedH11Protocol
from coveo_recsys.serving.runtime import BoundedWork, Deadline
from coveo_recsys.serving.trace import Journal
from coveo_recsys.state.service import IntakeService
from coveo_recsys.state.store import FileStateStore, MemoryStateStore


def run_serve(
    project_root: Path,
    settings_path: Path,
    run_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    if ipaddress.ip_address(host) not in {
        ipaddress.ip_address("127.0.0.1"),
        ipaddress.ip_address("::1"),
    }:
        raise InputValidationError("Локальный сервис доступен только на 127.0.0.1 или ::1")
    if type(port) is not int or not 1 <= port <= 65535:
        raise InputValidationError("Неверный локальный порт")
    root = project_root.resolve()
    if not root.is_dir():
        raise InputValidationError("Нет корня проекта")
    root_store = LocalArtifactStore(root)

    def local(path: Path) -> Path:
        relative = path.relative_to(root) if path.is_absolute() else path
        return root_store._path(relative.as_posix())

    config_path, destination = local(settings_path), local(run_dir)
    settings = strict_json_loads(config_path.read_bytes())
    if not isinstance(settings, dict) or set(settings) != {
        "package",
        "task",
        "owners",
        "calendar_start_ms",
        "calendar_end_ms",
        "state_adapter",
        "workers",
        "timeout_ms",
        "reserve_ms",
    }:
        raise InputValidationError("Неверный состав конфигурации сервиса")
    timing = {name: settings[name] for name in ("timeout_ms", "reserve_ms")}
    Deadline(0.0, **timing)
    if (
        settings["state_adapter"] not in {"memory", "file"}
        or type(settings["workers"]) is not int
        or settings["workers"] != 4
        or not isinstance(settings["package"], dict)
        or not isinstance(settings["task"], dict)
    ):
        raise InputValidationError("Нужен тестовый адаптер и фиксированные четыре потока")
    try:
        task = TaskConfig(**settings["task"])
    except TypeError as exc:
        raise InputValidationError("Неверная конфигурация задачи") from exc
    start = require_int64(settings["calendar_start_ms"])
    end = require_int64(settings["calendar_end_ms"])
    if start >= end or (task.detail_history_limit, task.block_history_limit) != (20, 20):
        raise InputValidationError("Неверный календарь или пределы префикса")
    names = settings["owners"]
    if not isinstance(names, dict) or not names:
        raise InputValidationError("Нужны зарегистрированные владельцы пространств")
    owners: dict[str, str] = {}
    for name, namespace in names.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise InputValidationError("Нужно имя переменной окружения с секретом")
        id_bytes(namespace)
        token = os.environ.get(name)
        if not token or token in owners:
            raise InputValidationError("Не задан уникальный секрет владельца пространства")
        id_bytes(token)
        owners[token] = namespace
    manifest = settings["package"]
    try:
        inputs = [
            root_store._path(manifest[name]["path"])
            for name in ("historical", "model", "training_features", "feature_schema", "metadata")
        ]
        inputs.extend((config_path, root_store._path(manifest["metadata"]["profile_path"])))
    except (KeyError, TypeError) as exc:
        raise InputValidationError("Не заданы пути готового пакета") from exc
    if any(
        destination == path or destination in path.parents or path in destination.parents
        for path in inputs
    ):
        raise InputValidationError("Каталог сервиса пересекается с входами")
    package = load_serving_package(root, manifest)
    if (
        task.event_semantics != package.schema["event_semantics"]
        or task.query_dimension
        != package.engine.index.package.metadata_contract["description_dimension"]
    ):
        raise InputValidationError("Задача не соответствует готовому пакету")
    destination.mkdir(parents=True, exist_ok=True)
    output = LocalArtifactStore(destination)
    store = (
        MemoryStateStore()
        if settings["state_adapter"] == "memory"
        else FileStateStore(output._path("state.json"))
    )
    intake = IntakeService(store, task, owners, start, end)
    journal = Journal(output._path("requests.jsonl"))
    work = BoundedWork(workers=settings["workers"])
    try:
        app = create_app(Coordinator((package,), intake, journal, work=work, **timing))
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,
            access_log=False,
            proxy_headers=False,
            server_header=False,
            # Uvicorn вызывает фабрику напрямую, хотя аннотация допускает только класс.
            http=partial(TimedH11Protocol, **timing),  # type: ignore[arg-type]
            ws="none",
        )
    finally:
        work.close()
