"""Проверяемое чтение всех опубликованных событий закреплённого A_G.

Вход остаётся неизменным: журнал передаёт вызывающий сценарий новой сборки.
Хеш производителя подготовки независим от кода сборщика Э2.
"""

from __future__ import annotations

import io
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from coveo_recsys.artifacts import LocalArtifactStore, sha256_bytes, strict_json_loads
from coveo_recsys.contracts import CONTRACT_VERSION, InputValidationError, id_bytes
from coveo_recsys.contracts.historical import HistoricalInput, Session
from coveo_recsys.contracts.preparation import (
    EVALUATION_PROTOCOL,
    PERIOD_NAMES,
    PREPARATION_VERSION,
    PreparationPlan,
)
from coveo_recsys.contracts.records import CanonicalBlock, EventType
from coveo_recsys.contracts.serialization import from_json, record_schema, record_version, to_json
from coveo_recsys.contracts.training import SessionRecord
from coveo_recsys.data.io import FILES, load_profile
from coveo_recsys.data.periods import admit, session_key
from coveo_recsys.numerics import checked_add

_REPORT = "reports/preparation.json"
_TABLE_SCHEMA = pa.schema([pa.field("record_json", pa.binary(), nullable=False)])
_R = TypeVar("_R", CanonicalBlock, SessionRecord)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise InputValidationError(reason)


def _records(content: bytes, kind: type[_R], rows: int) -> tuple[_R, ...]:
    table = pq.read_table(io.BytesIO(content))
    _require(table.schema.equals(_TABLE_SCHEMA), "Несовместимая схема таблицы записей")
    _require(table.num_rows == rows, "Число строк таблицы не совпало")
    return tuple(from_json(value, kind) for value in table["record_json"].to_pylist())


def _validate_session(
    blocks: Session,
    session: SessionRecord,
    plan: PreparationPlan,
    selected: dict[str, Any],
    index: dict[str, Any],
    inputs: dict[str, str],
    seen_origins: set[str],
) -> None:
    _require(bool(blocks), "Пустая историческая сессия")
    previous: int | None = None
    block_ids: set[str] = set()
    first_request: int | None = None
    purchased = False
    counts: Counter[str] = Counter()
    for block in blocks:
        _require(
            (block.source_namespace, block.session_id)
            == (plan.source_namespace, session.session_id),
            "Смешаны сессии или пространства источника",
        )
        _require(
            (previous is None or block.event_time_ms > previous)
            and block.block_id not in block_ids,
            "Повтор блока либо неверный порядок полной сессии",
        )
        _require(
            block.available_time_ms == block.event_time_ms
            and all(e.available_time_ms == e.event_time_ms for e in block.events),
            "Исторический вход требует a=t",
        )
        previous = block.event_time_ms
        block_ids.add(block.block_id)
        actions = {event.event_type for event in block.events}
        purchased |= EventType.PURCHASE in actions
        if first_request is None and not purchased and EventType.DETAIL in actions:
            first_request = block.event_time_ms
        for event in block.events:
            _require(
                bool(event.origins)
                and event.origins == tuple(sorted(set(event.origins), key=id_bytes))
                and event.provenance == event.origins[0],
                "Неверное происхождение канонического события",
            )
            for origin in event.origins:
                match = re.fullmatch(
                    r"(browsing_train\.csv|search_train\.csv):([0-9a-f]{64}):line=([1-9][0-9]*)",
                    origin,
                )
                _require(match is not None, "Неизвестный формат происхождения строки")
                assert match is not None
                filename, digest, line = match.groups()
                table = "search" if filename == FILES["search"] else "browsing"
                _require(
                    digest == inputs[table]
                    and int(line) >= 2
                    and origin not in seen_origins
                    and (table == "search") == (event.event_type == EventType.SEARCH),
                    "Чужая либо повторная исходная строка",
                )
                seen_origins.add(origin)
                counts[filename] += 1
    admission = admit(
        first_request, plan.periods, plan.task, last_event_ms=blocks[-1].event_time_ms
    )
    _require(admission.admitted and admission.period == "A_G", "Сессия не допущена в A_G")
    _require(
        session.admitted
        and session.exclusion_reason is None
        and session.period == "A_G"
        and session.source_namespace == plan.source_namespace
        and session.admission_ms == first_request == selected["first_request_ms"]
        and session.last_event_ms == blocks[-1].event_time_ms == selected["last_event_ms"]
        and session.request_count == 1
        and session.telemetry_complete is None,
        "Реестр допуска не соответствует полной сессии",
    )
    assert first_request is not None
    _require(
        session.target_end_ms
        == min(checked_add(first_request, plan.task.horizon_ms), blocks[-1].event_time_ms),
        "Реестр содержит неверную границу наблюдаемого окна",
    )
    _require(
        len(blocks) == index["blocks"]
        and sum(len(to_json(block)) for block in blocks) == index["encoded_bytes"],
        "Потеряны опубликованные блоки сессии",
    )
    for count_map in (selected["source_rows"], index["source_rows"]):
        _require(
            set(count_map) <= {FILES["browsing"], FILES["search"]}
            and all(type(n) is int and n >= 0 for n in count_map.values())
            and all(
                counts[name] == count_map.get(name, 0)
                for name in (FILES["browsing"], FILES["search"])
            ),
            "Потеряны исходные строки выбранной сессии",
        )


def open_historical_input(
    root: Path,
    *,
    report_sha256: str,
    report_descriptor_sha256: str,
    execution_path: str,
    execution_descriptor_sha256: str,
    audit: Callable[[dict[str, Any]], None],
) -> HistoricalInput:
    """Открыть только готовую подготовку, привязанную к доверенному отчёту.

    Читаемые пути закреплены дескриптором отчёта. Каждый проход повторно
    проверяет использованные файлы и объектные копии перед успешным возвратом.
    Исходы datasets не открываются; отчёт закрепляет их хеши без чтения целей.
    """
    opened: dict[str, Any] = {
        "purpose": "historical_input",
        "allowed": False,
        "boundary": "local_routing_only",
        "reason": "incomplete_open",
    }
    try:
        _require(root.is_dir() and not root.is_symlink(), "Нет корня готовой подготовки")
        store = LocalArtifactStore(root)
        report_descriptor = store.describe(_REPORT)
        _require(
            report_descriptor.sha256 == report_sha256
            and report_descriptor.descriptor_sha256 == report_descriptor_sha256,
            "Отчёт не совпал с закреплённой идентичностью",
        )
        report = strict_json_loads(store.read_by_sha256(report_sha256))
        _require(
            report["status"] == "materialized"
            and report["contract_version"] == PREPARATION_VERSION
            and report["evaluation_protocol"] == EVALUATION_PROTOCOL
            and report["scope"] in {"actual", "synthetic"}
            and report["real_preflight"] is False
            and report["telemetry_complete"] is None
            and report["source_sha256"] == report_descriptor.code_sha256
            and report["artifacts"] == report_descriptor.artifact_dependencies,
            "Отчёт не является совместимой подготовкой observed_purchases_v1",
        )
        inputs: dict[str, str] = report["inputs"]
        _require(set(inputs) == {*FILES, "profile", "plan"}, "Неполный состав входов подготовки")
        pins: dict[str, str] = {
            **report["artifacts"],
            _REPORT: report_descriptor_sha256,
            execution_path: execution_descriptor_sha256,
        }
        used: dict[str, str] = {}

        def payload(
            path: str,
            *,
            version: str,
            kind: str = "json",
            expected_inputs: dict[str, str] | None = None,
            configuration: dict[str, Any] | None = None,
        ) -> bytes:
            descriptor = store.verify_compatible(
                path,
                expected_inputs=inputs if expected_inputs is None else expected_inputs,
                expected_code_sha256=report["source_sha256"],
                expected_schema_version=version,
                expected_configuration={} if configuration is None else configuration,
                expected_descriptor_sha256=pins[path],
            )
            _require(
                descriptor.run_id == report["run_id"]
                and descriptor.scope == report["scope"]
                and descriptor.config_id == "e1_code"
                and descriptor.artifact_version == "1"
                and descriptor.kind == kind
                and all(
                    pins.get(p) == digest for p, digest in descriptor.artifact_dependencies.items()
                ),
                "Чужой производитель, область или зависимость артефакта",
            )
            used[path] = descriptor.descriptor_sha256
            return store.read_by_sha256(descriptor.sha256)

        def recheck() -> None:
            for path, expected in used.items():
                descriptor = store.describe(path)
                _require(descriptor.descriptor_sha256 == expected, "Вход изменился при чтении")
                store.read_by_sha256(descriptor.sha256)

        payload(_REPORT, version=PREPARATION_VERSION)
        _require(
            re.fullmatch(r"preparation/executions/[^/]+\.json", execution_path) is not None,
            "Некорректный путь завершённого выполнения",
        )
        execution = strict_json_loads(payload(execution_path, version="1"))
        _require(
            execution["status"] == "completed"
            and store.describe(execution_path).artifact_dependencies
            == {_REPORT: report_descriptor_sha256},
            "Подготовка не имеет завершённого выполнения",
        )
        plan_bytes = payload("inputs/plan.json", version=PREPARATION_VERSION)
        _require(sha256_bytes(plan_bytes) == inputs["plan"], "Подменён план подготовки")
        plan = from_json(plan_bytes, PreparationPlan)
        profile_bytes = payload(
            "inputs/profile.json", version="1", expected_inputs={"profile": inputs["profile"]}
        )
        _require(sha256_bytes(profile_bytes) == inputs["profile"], "Подменён профиль поставки")
        profile = load_profile(root / "inputs/profile.json")
        _require(
            profile.sha256 == inputs["profile"]
            and profile.namespace == plan.source_namespace
            and profile.scope == plan.scope == report["scope"]
            and profile.dimensions["query"] == plan.task.query_dimension
            and plan.diagnostic_sessions == 0
            and all(p.sample_size == 0 for p in plan.periods if p.name != "A_G"),
            "Профиль и план не разрешают исторический вход A_G",
        )
        catalog = strict_json_loads(
            payload(
                "inputs/catalog.json",
                version="1",
                expected_inputs={k: v for k, v in inputs.items() if k != "plan"},
            )
        )
        _require(
            catalog["scope"] == plan.scope
            and catalog["profile_sha256"] == inputs["profile"]
            and all(catalog["files"][key]["sha256"] == inputs[key] for key in FILES),
            "Происхождение исходных таблиц не совпало",
        )
        for record_type in (CanonicalBlock, SessionRecord):
            path = f"preparation/schemas/{record_type.__name__}.json"
            schema = strict_json_loads(
                payload(path, version=record_version(record_type), expected_inputs={})
            )
            _require(schema == record_schema(record_type), "Несовместимая схема записей подготовки")
        selection_bytes = payload("preparation/selection.json", version=PREPARATION_VERSION)
        sampled = strict_json_loads(selection_bytes)
        selection = sampled["selection"]
        _require(
            sampled["evaluation_protocol"] == EVALUATION_PROTOCOL
            and sampled["normative_sample"] is True
            and sampled["rule"] == "lowest_seeded_hash_after_calendar_admission"
            and selection == report["selection"]
            and set(selection) == set(PERIOD_NAMES)
            and all(not selection[name] for name in PERIOD_NAMES if name != "A_G"),
            "Выборка не является закреплённым A_G",
        )
        selected_ids = selection["A_G"]
        selected = set(selected_ids)
        _require(
            bool(selected)
            and len(selected) == len(selected_ids)
            and selected_ids == sorted(selected, key=lambda sid: session_key(sid, plan.seed))
            and selected == set(sampled["selected_rows"])
            and len(selected) == report["canonical_sessions"] == report["N_obs"]
            and len(selected) == sampled["period_counts"]["A_G"]["selected"],
            "Состав выбранных сессий не совпал",
        )
        index = strict_json_loads(payload("canonical/index.json", version=CONTRACT_VERSION))
        _require(
            len(index) == len(selected)
            and {item["session_id"] for item in index} == selected
            and len({item["path"] for item in index}) == len(index),
            "Индекс не соответствует выбранным сессиям",
        )
        for item in index:
            key = sha256_bytes(
                json.dumps([plan.source_namespace, item["session_id"]], ensure_ascii=False).encode(
                    "utf-8"
                )
            )
            _require(
                item["key"] == key and item["path"] == f"canonical/{key}.parquet",
                "Подменён путь канонической сессии",
            )
        datasets = report["datasets"]
        _require(
            len(datasets) == len(selected)
            and {d["session_id"] for d in datasets} == selected
            and all(d["period"] == "A_G" and d["requests"] == 1 for d in datasets)
            and {d["session_id"]: d["canonical_path"] for d in datasets}
            == {item["session_id"]: item["path"] for item in index},
            "Отчёт наборов не соответствует индексу A_G",
        )
        sessions = _records(
            payload(
                "preparation/sessions.parquet",
                version=record_version(SessionRecord),
                kind="parquet",
                configuration={"record_type": "SessionRecord"},
            ),
            SessionRecord,
            len(selected),
        )
        registry = {session.session_id: session for session in sessions}
        _require(
            set(registry) == selected
            and all(
                s.source_namespace == plan.source_namespace and s.admitted and s.period == "A_G"
                for s in sessions
            ),
            "Реестр допуска не соответствует выбранным сессиям A_G",
        )
        snapshot = next(p.end_ms for p in plan.periods if p.name == "A_G")
        hashes = {
            **inputs,
            "preparation_code": report["source_sha256"],
            "preparation_report": report_sha256,
            "preparation_report_descriptor": report_descriptor_sha256,
            "preparation_execution_descriptor": execution_descriptor_sha256,
            "preparation_selection": sha256_bytes(selection_bytes),
            "preparation_selection_descriptor": pins["preparation/selection.json"],
            "preparation_index_descriptor": pins["canonical/index.json"],
            "preparation_sessions_descriptor": pins["preparation/sessions.parquet"],
            "canonical_schema": pins["preparation/schemas/CanonicalBlock.json"],
            "session_schema": pins["preparation/schemas/SessionRecord.json"],
        }
        recheck()
        opened.update(allowed=True, reason=None, scope=plan.scope)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        opened["reason"] = str(exc)
        raise InputValidationError(str(exc)) from exc
    finally:
        audit(opened)

    def iterate() -> Iterator[Session]:
        reading: dict[str, Any] = dict(
            purpose="historical_sessions",
            allowed=False,
            complete=False,
            sessions=0,
            blocks=0,
            source_rows=0,
            scope=plan.scope,
            reason="incomplete_read",
            boundary="local_routing_only",
        )
        try:
            recheck()
            seen_origins: set[str] = set()
            for item in index:
                blocks = _records(
                    payload(
                        item["path"],
                        version=record_version(CanonicalBlock),
                        kind="parquet",
                        configuration={
                            "record_type": "CanonicalBlock",
                            "source_namespace": plan.source_namespace,
                            "session_id": item["session_id"],
                        },
                    ),
                    CanonicalBlock,
                    item["blocks"],
                )
                _validate_session(
                    blocks,
                    registry[item["session_id"]],
                    plan,
                    sampled["selected_rows"][item["session_id"]],
                    item,
                    inputs,
                    seen_origins,
                )
                reading["sessions"] += 1
                reading["blocks"] += len(blocks)
                reading["source_rows"] = len(seen_origins)
                yield blocks
            _require(reading["blocks"] == report["canonical_blocks"], "Потеряны блоки подготовки")
            recheck()
            reading.update(allowed=True, complete=True, reason=None)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            reading["reason"] = str(exc)
            raise InputValidationError(str(exc)) from exc
        finally:
            audit(reading)

    return HistoricalInput(
        iterate,
        len(selected),
        snapshot,
        plan.source_namespace,
        MappingProxyType(hashes),
        plan.scope,
    )
