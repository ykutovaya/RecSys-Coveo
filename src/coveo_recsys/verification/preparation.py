"""Прямая сверка новых общих таблиц с опубликованными строками Coveo.

Рабочие канонизатор, расчёт состояния и целей не используются как эталоны.
Глобальный отбор проверяется по закреплённому плану и отчёту; CSV сверяются
для всех выбранных ID, без зависимости от прежнего размера выборки.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import time
from collections import Counter
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.io import parse_profile
from coveo_recsys.data.prepared import REPORT, iter_sessions, read_report, verify_tables
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.process_memory import peak_memory
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

FILES = {
    "browsing": "browsing_train.csv",
    "search": "search_train.csv",
    "products": "sku_to_content.csv",
}
DATA_RULES = (
    "schema_valid",
    "whole_sessions",
    "published_rows_complete",
    "calendar_admission_checked",
    "registered_targets_checked",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputValidationError(message)


def _key(sid: str, seed: int) -> tuple[bytes, bytes]:
    raw = sid.encode("utf-8")
    return hashlib.sha256(struct.pack(">QQ", seed, len(raw)) + raw).digest(), raw


def _vector(value: Any, dimension: int) -> tuple[tuple[float, ...] | None, str | None]:
    if value is None:
        return None, "vector_absent"
    if (
        not isinstance(value, list)
        or len(value) != dimension
        or any(type(x) not in (int, float) for x in value)
    ):
        return None, "vector_invalid"
    try:
        vector = tuple(float(x) for x in value)
    except OverflowError:
        return None, "vector_invalid"
    if not all(math.isfinite(x) for x in vector):
        return None, "vector_invalid"
    return (None, "vector_zero") if math.hypot(*vector) <= 1e-12 else (vector, None)


def _prefix(
    blocks: tuple[CanonicalBlock, ...], request: PreparedRequest, plan: PreparationPlan
) -> None:
    visible = [block for block in blocks if block.event_time_ms <= request.t_q]
    prefix = request.prefix
    _require(
        prefix.source_namespace == plan.source_namespace
        and prefix.event_semantics == plan.task.event_semantics
        and prefix.version == len(visible)
        and prefix.blocks == tuple(visible[-plan.task.block_history_limit :])
        and prefix.first_event_time_ms == visible[0].event_time_ms,
        "Состав префикса не соответствует первому запросу",
    )
    counts: Counter[tuple[str, str | None]] = Counter()
    cart: dict[str, tuple[str, int, int | None]] = {}
    details: list[tuple[str, int]] = []
    views = {}
    latest_search = None
    for block in visible:
        actions: dict[str, set[str]] = {}
        for event in block.events:
            kind, sku = event.event_type.value, event.sku
            count = (
                dict(block.raw_multiplicities)[event.event_id]
                if plan.task.event_semantics == "multiset_v1"
                else 1
            )
            counts[kind, sku] += count
            if kind == "detail":
                assert sku is not None
                details.extend([(sku, block.event_time_ms)] * count)
                views[sku] = block.event_time_ms
            elif kind in {"add", "remove"}:
                assert sku is not None
                actions.setdefault(sku, set()).add(kind)
        for sku, kinds in actions.items():
            cart[sku] = (
                (next(iter(kinds)), block.event_time_ms, block.event_time_ms)
                if len(kinds) == 1
                else ("unknown", block.event_time_ms, cart.get(sku, ("", 0, None))[2])
            )
        if block.search_payloads:
            latest_search = block
    ordered = sorted(details, key=lambda x: (-x[1], x[0].encode()))
    history = Counter(ordered[: plan.task.detail_history_limit])
    full = Counter(x for x in details if x[1] == request.t_q)
    expected_history, expected_full = (
        tuple(
            (sku, stamp, count)
            for (sku, stamp), count in sorted(
                values.items(), key=lambda item: (item[0][1], item[0][0].encode())
            )
        )
        for values in (history, full)
    )
    _require(
        tuple((x.sku, x.event_time_ms, x.count) for x in prefix.detail_history) == expected_history
        and tuple((x.sku, x.event_time_ms, x.count) for x in prefix.full_last_detail)
        == expected_full
        and {(kind, sku): n for kind, sku, n in prefix.counters} == counts
        and dict(prefix.last_views) == views
        and not prefix.purchased_skus
        and {x.sku: (x.state.value, x.updated_at_ms, x.last_unambiguous_ms) for x in prefix.cart}
        == cart,
        "История, счётчики или корзина префикса не совпали",
    )
    if latest_search is None:
        _require(prefix.search is None, "Лишний поисковый контекст")
        return
    search = prefix.search
    _require(search is not None, "Потерян поисковый контекст")
    assert search is not None
    reasons = {p.reason for p in latest_search.search_payloads if p.reason is not None}
    values = {p.vector for p in latest_search.search_payloads if p.reason is None}
    reason = (
        min(reasons, key=lambda x: x.encode())
        if reasons
        else ("search_ambiguous" if len(values) != 1 else None)
    )
    _require(
        (search.block_id, search.event_time_ms, search.available_time_ms)
        == (latest_search.block_id, latest_search.event_time_ms, latest_search.event_time_ms)
        and search.payload.reason == reason,
        "Неверный поисковый контекст",
    )
    if reason is not None:
        _require(search.payload.vector is None, "Недопустимый вектор поиска")
    else:
        vector = next(iter(values))
        assert vector is not None
        scale = max(abs(x) for x in vector)
        norm = math.hypot(*(x / scale for x in vector))
        _require(
            search.payload.vector is not None
            and all(
                math.isclose(x, (y / scale) / norm, rel_tol=1e-12, abs_tol=1e-12)
                for x, y in zip(search.payload.vector, vector, strict=True)
            ),
            "Направление поиска не совпало",
        )


def _session(
    blocks: tuple[CanonicalBlock, ...],
    request: PreparedRequest,
    target: TargetRecord,
    session: SessionRecord,
    selected: dict[str, Any],
    plan: PreparationPlan,
    period: Any,
) -> None:
    times = [block.event_time_ms for block in blocks]
    _require(times == sorted(set(times)), "Повторные или неупорядоченные блоки")
    _require(len({block.block_id for block in blocks}) == len(blocks), "Повторный ID блока")
    first = next(
        (
            block
            for block in blocks
            if any(e.event_type in {"detail", "purchase"} for e in block.events)
        ),
        None,
    )
    _require(
        first is not None and all(e.event_type != "purchase" for e in first.events),
        "Нет первого просмотра до покупки",
    )
    assert first is not None
    t0, last = first.event_time_ms, times[-1]
    _require(
        (request.t_q, request.a_q, request.trigger_block_id, request.ordinal)
        == (t0, t0, first.block_id, 1)
        and period.start_ms <= t0 < period.end_ms
        and t0 + plan.task.horizon_ms + plan.task.gap_ms < period.end_ms
        and (period.name != "A_G" or last + plan.task.gap_ms < period.end_ms),
        "Первый запрос или временной допуск не совпал",
    )
    tau = min(t0 + plan.task.horizon_ms, last)
    _require(
        session.session_id == request.session_id == blocks[0].session_id
        and session.source_namespace == plan.source_namespace
        and session.admitted
        and session.exclusion_reason is None
        and session.period == period.name
        and session.admission_ms == selected["first_request_ms"] == t0
        and session.last_event_ms == selected["last_event_ms"] == last
        and session.target_end_ms == tau
        and session.request_count == 1
        and session.telemetry_complete is None,
        "Реестр допуска сессий не совпал",
    )
    purchases = tuple(
        sorted(
            {
                e.sku
                for b in blocks
                if t0 < b.event_time_ms <= tau
                for e in b.events
                if e.event_type == "purchase" and e.sku is not None
            },
            key=lambda x: x.encode(),
        )
    )
    _require(
        target.request_id == request.request_id
        and target.purchases == purchases
        and target.horizon_ms == plan.task.horizon_ms
        and target.status == ("known" if purchases else "unknown")
        and target.reason == (None if purchases else "no_registered_purchase"),
        "Зарегистрированная цель или неизвестный статус не совпали",
    )
    _prefix(blocks, request, plan)


def verify_preparation(
    root: Path,
    prepared_dir: Path,
    input_dir: Path,
    profile_path: Path,
    plan_path: Path,
    *,
    max_seconds: float,
    max_memory_bytes: int,
    ag_dir: Path | None = None,
    historical_dir: Path | None = None,
    other_preparations: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Проверить один активный A_G/B/C_smoke; вернуть отчёт, не записывая файлы."""
    started = time.monotonic()
    _require(max_seconds > 0 and max_memory_bytes > 0, "Пределы должны быть положительными")

    def budget() -> None:
        memory = peak_memory()["peak_process_memory_bytes"]
        _require(memory is not None, "Недоступно измерение памяти")
        _require(memory + 128 * 1024**2 <= max_memory_bytes, "Предел памяти проверки")
        _require(time.monotonic() - started <= max_seconds, "Предел времени проверки")

    budget()
    assert_source_binding(root)
    code_hash = source_sha256(root)
    report = read_report(prepared_dir)
    report_file_hash = sha256_file(prepared_dir / REPORT)
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
    profile = strict_json_loads(profile_path.read_bytes())
    parse_profile(profile, sha256=sha256_file(profile_path))
    active = [period for period in plan.periods if period.sample_size]
    _require(
        len(active) == 1 and active[0].name in {"A_G", "B", "C_smoke"},
        "Нужен один период A_G/B/C_smoke",
    )
    period = active[0]
    _require(
        report["scope"] == plan.scope == profile["scope"] == "actual"
        and report["source_sha256"] == code_hash
        and plan.diagnostic_sessions == 0
        and report["inputs"]["plan"] == sha256_file(plan_path)
        and report["inputs"]["profile"] == sha256_file(profile_path),
        "Подготовка не связана с текущими кодом, планом или профилем",
    )
    selected = report["selection"][period.name]
    selected_ids = set(selected)
    _require(
        len(selected) == report["N_obs"] == min(period.sample_size, report["N_admitted"])
        and selected == sorted(selected_ids, key=lambda sid: _key(sid, plan.seed))
        and bool(selected)
        and all(not ids for name, ids in report["selection"].items() if name != period.name),
        "Объём или хеш-порядок выбранных сессий не совпал",
    )
    sampled = report["sampled"]
    _require(
        sampled["selection"] == report["selection"]
        and sampled["period_counts"] == report["period_counts"]
        and set(sampled["selected_rows"]) == selected_ids
        and report["N_admitted"] == report["period_counts"][period.name]["admitted"]
        and report["N_eligible"] == report["period_counts"][period.name]["eligible"],
        "Состав и счётчики отбора не совпали",
    )
    bindings = dict(
        run_id=report["run_id"],
        tables=report["tables"],
        inputs=report["inputs"],
        preparation_code_sha256=code_hash,
        report_sha256=report_file_hash,
        plan_sha256=report["inputs"]["plan"],
    )
    related_hashes = {}
    for other in dict.fromkeys((*other_preparations, *((ag_dir,) if ag_dir else ()))):
        old = read_report(other, verify_tables=False)
        old_plan = from_json(old["plan_json"].encode(), PreparationPlan)
        _require(
            not selected_ids.intersection(sid for ids in old["selection"].values() for sid in ids),
            "Пересечение выбранных сессий разных периодов",
        )
        _require(
            old["scope"] == "actual"
            and old_plan.task == plan.task
            and old_plan.seed == plan.seed
            and old_plan.coverage == plan.coverage
            and [(p.name, p.start_ms, p.end_ms) for p in old_plan.periods]
            == [(p.name, p.start_ms, p.end_ms) for p in plan.periods]
            and all(old["inputs"][key] == report["inputs"][key] for key in (*FILES, "profile")),
            "Периоды или входы связанных подготовок различаются",
        )
        related_hashes[other / REPORT] = sha256_file(other / REPORT)
        if other == ag_dir:
            bindings["ag_report_sha256"] = related_hashes[other / REPORT]
            bindings["ag_run_id"] = old["run_id"]
    if historical_dir is not None:
        _require(ag_dir is not None, "Для исторического пакета нужен проверяемый A_G")
        historical = load_historical(LocalArtifactStore(historical_dir))
        _require(
            historical.input_hashes["preparation_report"] == bindings["ag_report_sha256"],
            "Исторический пакет не связан с A_G",
        )
        bindings["historical_package_sha256"] = sha256_file(historical_dir / PACKAGE_PATH)
        related_hashes[historical_dir / PACKAGE_PATH] = bindings["historical_package_sha256"]
        del historical
    table_types = {
        f"canonical/{period.name}.parquet": CanonicalBlock,
        f"datasets/{period.name}/prefixes.parquet": PreparedRequest,
        f"datasets/{period.name}/targets.parquet": TargetRecord,
        "preparation/sessions.parquet": SessionRecord,
    }
    _require(set(report["tables"]) == set(table_types), "Нужны ровно четыре общие таблицы")
    readers: list[Iterator[Any]] = [
        iter_sessions(
            prepared_dir / path,
            kind,
            max_session_records=plan.max_session_blocks,
            max_session_bytes=plan.max_session_bytes,
        )
        for path, kind in table_types.items()
    ]
    origins: dict[str, tuple[str, int, str, str | None]] = {}
    searches: dict[tuple[str, int], Counter[Any]] = {}
    source_rows: Counter[str] = Counter()
    checked = set()
    block_count = positives = 0
    for records in zip_longest(*readers):
        budget()
        _require(all(row is not None for row in records), "Неравные наборы сессий таблиц")
        sid = records[0][0]
        _require(
            all(row[0] == sid for row in records) and sid in selected_ids and sid not in checked,
            "Чужая или повторная сессия таблицы",
        )
        blocks = records[0][1]
        _require(
            all(len(row[1]) == 1 for row in records[1:]),
            "Нужны один запрос, цель и допуск на сессию",
        )
        request, target, session = (row[1][0] for row in records[1:])
        _session(blocks, request, target, session, sampled["selected_rows"][sid], plan, period)
        counts: Counter[str] = Counter()
        for block in blocks:
            _require(
                block.session_id == sid
                and block.source_namespace == plan.source_namespace
                and block.available_time_ms == block.event_time_ms,
                "Чужой блок или a != t",
            )
            for event in block.events:
                _require(
                    event.available_time_ms == block.event_time_ms
                    and event.transport_id is None
                    and event.origins == tuple(sorted(set(event.origins), key=lambda x: x.encode()))
                    and bool(event.origins)
                    and event.provenance == event.origins[0]
                    and dict(block.raw_multiplicities)[event.event_id] == len(event.origins),
                    "Происхождение или кратность события не совпали",
                )
                kind = "search" if event.event_type == "search" else "browsing"
                prefix = f"{FILES[kind]}:{report['inputs'][kind]}:line="
                for origin in event.origins:
                    _require(
                        origin.startswith(prefix)
                        and origin[len(prefix) :].isdigit()
                        and int(origin[len(prefix) :]) >= 2
                        and origin not in origins,
                        "Повторная или чужая исходная строка",
                    )
                    origins[origin] = (sid, block.event_time_ms, event.event_type.value, event.sku)
                    counts[FILES[kind]] += 1
            if block.search_payloads:
                searches[sid, block.event_time_ms] = Counter(
                    {
                        (p.vector, p.reason): n
                        for p, n in zip(
                            block.search_payloads, block.search_multiplicities, strict=True
                        )
                    }
                )
        _require(
            {name: counts[name] for name in (FILES["browsing"], FILES["search"])}
            == {
                name: sampled["selected_rows"][sid]["source_rows"].get(name, 0)
                for name in (FILES["browsing"], FILES["search"])
            },
            "Число исходных строк сессии не совпало",
        )
        source_rows.update(counts)
        checked.add(sid)
        block_count += len(blocks)
        positives += bool(target.purchases)
    _require(
        checked == selected_ids
        and len(checked) == report["canonical_sessions"] == report["request_count"]
        and block_count == report["canonical_blocks"]
        and positives == report["N_plus"]
        and {name: source_rows[name] for name in (FILES["browsing"], FILES["search"])}
        == report["composition"]["source_rows"],
        "Итоговые счётчики подготовки не совпали",
    )
    origin_count, scans, sizes, identities = len(origins), [], {}, {}
    csv_limit = csv.field_size_limit(1_048_576)
    try:
        for kind, name in FILES.items():
            path = input_dir / name
            before = path.stat()
            identities[path] = (before.st_size, before.st_mtime_ns)
            sizes[name] = before.st_size
            expected = profile["files"][name]
            _require(
                before.st_size == expected["expected_size_bytes"]
                and report["inputs"][kind] == expected["expected_sha256"],
                "Вход не соответствует закреплённому профилю",
            )
            digest = hashlib.sha256()
            rows = selected_rows = 0
            with path.open("rb") as stream:

                def lines() -> Iterator[str]:
                    for line in stream:
                        digest.update(line)
                        yield line.decode(profile["encoding"])

                if kind == "products":
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        budget()
                else:
                    reader = csv.DictReader(
                        lines(),
                        delimiter=profile["delimiter"],
                        quotechar=profile["quotechar"],
                        strict=True,
                    )
                    _require(reader.fieldnames == expected["columns"], "Столбцы CSV не совпали")
                    for row in reader:
                        rows += 1
                        if rows % 8192 == 0:
                            budget()
                        _require(
                            None not in row and None not in row.values(),
                            "Неверное число столбцов CSV",
                        )
                        sid = row["session_id_hash"]
                        if sid not in checked:
                            continue
                        origin = f"{name}:{report['inputs'][kind]}:line={reader.line_num}"
                        stamp = int(row["server_timestamp_epoch_ms"])
                        action = "search" if kind == "search" else row["product_action"] or "page"
                        sku = None if kind == "search" else row["product_sku_hash"] or None
                        _require(
                            origins.pop(origin, None) == (sid, stamp, action, sku),
                            "Исходная строка потеряна или отнесена к чужому событию",
                        )
                        selected_rows += 1
                        if kind == "search":
                            try:
                                value = (
                                    json.loads(row["query_vector"]) if row["query_vector"] else None
                                )
                            except (ValueError, RecursionError):
                                value = row["query_vector"]
                            payload = _vector(value, plan.task.query_dimension)
                            _require(
                                (sid, stamp) in searches and searches[sid, stamp][payload] > 0,
                                "Поисковая нагрузка исходной строки не совпала",
                            )
                            searches[sid, stamp][payload] -= 1
            _require(digest.hexdigest() == report["inputs"][kind], "Хеш исходного файла изменился")
            if kind != "products":
                scans.append(
                    dict(
                        file=name,
                        rows_scanned=rows,
                        selected_rows=selected_rows,
                        bytes_read=before.st_size,
                        sha256=digest.hexdigest(),
                        passes=1,
                    )
                )
    except (csv.Error, UnicodeError, ValueError) as exc:
        if isinstance(exc, InputValidationError):
            raise
        raise InputValidationError(f"Повреждён исходный CSV: {type(exc).__name__}") from exc
    finally:
        csv.field_size_limit(csv_limit)
    _require(
        not origins and all(not any(counts.values()) for counts in searches.values()),
        "Остались канонические строки без исходной строки",
    )
    verify_tables(prepared_dir, report)
    _require(
        sha256_file(prepared_dir / REPORT) == report_file_hash
        and sha256_file(plan_path) == report["inputs"]["plan"]
        and sha256_file(profile_path) == report["inputs"]["profile"]
        and all(sha256_file(path) == digest for path, digest in related_hashes.items())
        and all(
            (path.stat().st_size, path.stat().st_mtime_ns) == identity
            for path, identity in identities.items()
        )
        and source_sha256(root) == code_hash,
        "Входы или код изменились во время проверки",
    )
    budget()
    return dict(
        check_report_format_version="preparation_check_v1",
        status="passed",
        scope="actual",
        real_preflight=False,
        current_code_sha256=code_hash,
        bindings=bindings,
        **dict.fromkeys(DATA_RULES, True),
        preparation_plan_sha256=report["inputs"]["plan"],
        preparation_report_sha256=sha256_bytes(canonical_json_bytes(report)),
        period=period.name,
        session_count=len(checked),
        request_count=len(checked),
        N_plus=positives,
        unknown_empty_targets=len(checked) - positives,
        overlapping_sessions=0,
        ag_intersection_count=0,
        telemetry_completeness="unknown",
        input_size_bytes=sizes,
        source_rows_verified=origin_count,
        raw_scans=scans,
        selection_check="fixed_plan_report_counts_and_seeded_hash_order",
        elapsed_seconds=time.monotonic() - started,
        **peak_memory(),
    )
