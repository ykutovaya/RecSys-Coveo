"""Порционное чтение закреплённого A_G без открытия запросов и целей."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any

from coveo_recsys.artifacts import sha256_bytes, strict_json_loads
from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.historical import HistoricalInput, Session
from coveo_recsys.contracts.preparation import EVALUATION_PROTOCOL, PERIOD_NAMES, PreparationPlan
from coveo_recsys.contracts.records import CanonicalBlock, EventType
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.contracts.training import SessionRecord
from coveo_recsys.data.io import FILES, parse_profile
from coveo_recsys.data.periods import admit, session_key
from coveo_recsys.data.prepared import iter_sessions, read_report, verify_tables
from coveo_recsys.numerics import checked_add

_CANONICAL = "canonical/A_G.parquet"
_SESSIONS = "preparation/sessions.parquet"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise InputValidationError(reason)


def _validate_session(
    blocks: Session,
    session: SessionRecord,
    plan: PreparationPlan,
    selected: dict[str, Any],
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
    count_map = selected["source_rows"]
    _require(
        set(count_map) <= {FILES["browsing"], FILES["search"]}
        and all(type(n) is int and n >= 0 for n in count_map.values())
        and all(
            counts[name] == count_map.get(name, 0) for name in (FILES["browsing"], FILES["search"])
        ),
        "Потеряны исходные строки выбранной сессии",
    )


def open_historical_input(
    root: Path,
    *,
    report_sha256: str,
    audit: Callable[[dict[str, Any]], None],
) -> HistoricalInput:
    """Открыть завершённый набор; каждый проход проверяет только используемые таблицы."""
    opened: dict[str, Any] = dict(
        purpose="historical_input",
        allowed=False,
        boundary="local_routing_only",
        reason="incomplete_open",
    )
    try:
        report = read_report(root, report_sha256=report_sha256, verify_tables=False)
        inputs: dict[str, str] = report["inputs"]
        _require(set(inputs) == {*FILES, "profile", "plan"}, "Неполный состав входов подготовки")
        plan_bytes = report["plan_json"].encode("utf-8")
        profile_bytes = report["profile_json"].encode("utf-8")
        _require(
            sha256_bytes(plan_bytes) == inputs["plan"]
            and sha256_bytes(profile_bytes) == inputs["profile"],
            "Подменён план подготовки либо профиль поставки",
        )
        plan = from_json(plan_bytes, PreparationPlan)
        profile = parse_profile(strict_json_loads(profile_bytes), sha256=inputs["profile"])
        _require(
            profile.namespace == plan.source_namespace
            and profile.scope == plan.scope == report["scope"]
            and profile.dimensions["query"] == plan.task.query_dimension
            and plan.diagnostic_sessions == 0
            and all(p.sample_size == 0 for p in plan.periods if p.name != "A_G"),
            "Профиль и план не разрешают исторический вход A_G",
        )
        sampled = report["sampled"]
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
        for path, kind, period, count in (
            (_CANONICAL, CanonicalBlock, "A_G", report["canonical_blocks"]),
            (_SESSIONS, SessionRecord, None, len(selected)),
        ):
            table = report["tables"][path]
            _require(
                table["record_type"] == kind.__name__
                and table["period"] == period
                and table["data_role"] is None
                and table["rows"] == count,
                "Несовместимая таблица исторического входа",
            )

        def recheck() -> None:
            read_report(root, report_sha256=report_sha256, verify_tables=False)
            verify_tables(root, report, paths=(_CANONICAL, _SESSIONS))

        hashes = {
            **inputs,
            "preparation_code": report["source_sha256"],
            "preparation_report": report_sha256,
            "preparation_canonical": report["tables"][_CANONICAL]["sha256"],
            "preparation_sessions": report["tables"][_SESSIONS]["sha256"],
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
            registry = iter_sessions(root / _SESSIONS, SessionRecord, max_session_records=1)
            seen_sessions: set[str] = set()
            for sid, blocks in iter_sessions(
                root / _CANONICAL,
                CanonicalBlock,
                max_session_records=plan.max_session_blocks,
                max_session_bytes=plan.max_session_bytes,
            ):
                entry = next(registry, None)
                _require(
                    sid in selected and entry is not None and entry[0] == sid,
                    "Реестр и события не соответствуют выбранным сессиям",
                )
                assert entry is not None
                _validate_session(
                    blocks, entry[1][0], plan, sampled["selected_rows"][sid], inputs, seen_origins
                )
                seen_sessions.add(sid)
                reading["sessions"] += 1
                reading["blocks"] += len(blocks)
                reading["source_rows"] = len(seen_origins)
                yield blocks
            _require(
                next(registry, None) is None
                and seen_sessions == selected
                and reading["blocks"] == report["canonical_blocks"],
                "Потеряны сессии или блоки подготовки",
            )
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
        next(p.end_ms for p in plan.periods if p.name == "A_G"),
        plan.source_namespace,
        MappingProxyType(hashes),
        plan.scope,
    )
