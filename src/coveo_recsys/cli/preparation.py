"""Подготовка целых сессий и зарегистрированных целей observed_purchases_v1."""

import bisect
import hashlib
import itertools
import json
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import (
    ArtifactDescriptor,
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.preparation import (
    EVALUATION_PROTOCOL,
    PREPARATION_VERSION,
    PreparationPlan,
    PreparedRequest,
)
from coveo_recsys.contracts.records import CanonicalBlock
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.io import (
    FILES,
    inspect_inputs,
    iter_events,
    load_profile,
    row_batches,
    timestamp,
)
from coveo_recsys.data.periods import admit, session_key
from coveo_recsys.data.prepared import (
    FORMAT_VERSION,
    REPORT,
    TableWriter,
    read_report,
    table_info,
    verify_tables,
    write_report,
)
from coveo_recsys.data.schedule import make_schedule
from coveo_recsys.data.supply import SupplyLimits, inspect_supply
from coveo_recsys.process_memory import peak_memory as _peak_memory
from coveo_recsys.state import empty_prefix, reduce_state
from coveo_recsys.targets import prepare_targets
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

IDENTITY = "preparation/identity.json"


class Preparation:
    def __init__(self, store: LocalArtifactStore, root: Path, run_id: str) -> None:
        assert_source_binding(root)
        self.store, self.root, self.run_id = store, root, run_id
        self.code = source_sha256(root)
        self.common: dict[str, Any] = dict(
            code_sha256=self.code, run_id=run_id, scope="synthetic", config_id="e1_code", reuse=True
        )
        self.started = time.monotonic()
        self.artifacts: dict[str, str] = {}
        self.inputs: dict[str, str] = {}
        self._verified_trees: dict[str, str] = {}
        self.plan: PreparationPlan | None = None
        self.reads: list[dict[str, Any]] = []
        self.reused_stages: list[str] = []
        self.temp_peak = 0
        self.disk_peak = 0
        self._last_budget_check = 0.0
        self._last_scan_progress = 0.0
        self._last_emitted_progress = -float("inf")
        self.last_progress: dict[str, Any] = {}
        self._preparing = False
        self._temporary_tables: set[Path] = set()

    def _check_budget(self, *, force: bool = False) -> None:
        if self.plan is None:
            return
        now = time.monotonic()
        if not force and now - self._last_budget_check < 0.5:
            return
        self._last_budget_check = now
        memory = _peak_memory()["peak_process_memory_bytes"]
        temporary = sum(
            p.stat().st_size
            for directory in self.store.root.glob(".prepare-*")
            for p in directory.rglob("*")
            if p.is_file()
        )
        temporary += sum(p.stat().st_size for p in self._temporary_tables if p.is_file())
        temporary += sum(
            p.stat().st_size
            for p in (self.store.root / "reports").glob(".preparation.json.*.tmp")
            if p.is_file()
        )
        self.temp_peak = max(self.temp_peak, temporary)
        if memory is None:
            raise InputValidationError("Нельзя проверить бюджет памяти")
        if (
            now - self.started > self.plan.max_seconds
            or memory > self.plan.max_memory_bytes
            or temporary > self.plan.max_temp_bytes
        ):
            raise InputValidationError("Превышен бюджет времени, памяти или диска подготовки")

    def _scan_progress(self, table: str, rows: int) -> None:
        if self.reads and not self.reads[-1].get("complete", True):
            self.reads[-1]["rows"] = rows
        self._check_budget()
        now = time.monotonic()
        if now - self._last_scan_progress >= 10:
            self.progress("reading", table=table, rows=rows)
            self._last_scan_progress = now

    def _input_hashes(self, input_dir: Path) -> dict[str, str]:
        hashes = {}
        for table, filename in FILES.items():
            started = time.monotonic()
            digest, size = hashlib.sha256(), 0
            reading: dict[str, Any] = dict(
                purpose="integrity", table=table, bytes=0, complete=False
            )
            self.reads.append(reading)
            with (input_dir / filename).open("rb") as stream:
                while block := stream.read(8_388_608):
                    digest.update(block)
                    size += len(block)
                    reading["bytes"] = size
                    self.progress("hashing", table=table, bytes=size)
            hashes[table] = digest.hexdigest()
            reading.update(complete=True, seconds=time.monotonic() - started)
        return hashes

    @contextmanager
    def _operation(
        self, input_dir: Path, profile_path: Path, plan_path: Path | None = None
    ) -> Iterator[None]:
        if self._preparing:
            raise InputValidationError("Нельзя запускать проверку внутри активной подготовки")
        with self.store.verification_scope():
            self._verified_trees.clear()
            self.artifacts.clear()
            try:
                yield
            finally:
                self._verified_trees.clear()
        # Внешние входы и код проверяются после заключительного обхода артефактов.
        self._verify_current(input_dir, profile_path, plan_path)

    def progress(self, step: str, **fields: Any) -> None:
        """Прогресс в stderr; частые сообщения ограничены одним в десять секунд."""
        if step.endswith("_reused"):
            self.reused_stages.append(step)
        if step != "failed":
            self._check_budget(force=True)
        now = time.monotonic()
        self.last_progress = {
            "step": step,
            "elapsed_seconds": now - self.started,
            **_peak_memory(),
            **fields,
        }
        if (
            step not in {"canonical_session", "prepared_session", "reading", "hashing"}
            or now - self._last_emitted_progress >= 10
        ):
            print(json.dumps(self.last_progress, ensure_ascii=True), file=sys.stderr, flush=True)
            self._last_emitted_progress = now

    def _remember(self, descriptor: ArtifactDescriptor) -> ArtifactDescriptor:
        self.artifacts[descriptor.path] = descriptor.descriptor_sha256
        return descriptor

    def _verify_current(
        self, input_dir: Path, profile_path: Path, plan_path: Path | None = None
    ) -> None:
        if (
            source_sha256(self.root) != self.code
            or any(
                digest != self.inputs[key] for key, digest in self._input_hashes(input_dir).items()
            )
            or sha256_file(profile_path) != self.inputs["profile"]
            or (plan_path is not None and sha256_file(plan_path) != self.inputs["plan"])
        ):
            raise InputValidationError("Код, входы или план изменились во время подготовки")

    def inspect(
        self, input_dir: Path, profile_path: Path, *, batch_size: int, max_batch_chars: int
    ) -> dict[str, Any]:
        with self._operation(input_dir, profile_path):
            return self._inspect(
                input_dir, profile_path, batch_size=batch_size, max_batch_chars=max_batch_chars
            )

    def _inspect(
        self, input_dir: Path, profile_path: Path, *, batch_size: int, max_batch_chars: int
    ) -> dict[str, Any]:
        profile = load_profile(profile_path)
        self.common["scope"] = profile.scope
        paths = {key: input_dir / filename for key, filename in FILES.items()}
        if profile.scope == "actual":
            catalog_path = "inputs/catalog.json"
            if (self.store.root / catalog_path).exists():
                self.inputs = {**self._input_hashes(input_dir), "profile": profile.sha256}
                descriptor = self.store.verify_compatible(
                    catalog_path,
                    expected_inputs=self.inputs,
                    expected_code_sha256=self.code,
                    expected_schema_version="1",
                    expected_configuration={},
                )
                self._verify_tree(catalog_path, descriptor.descriptor_sha256)
                catalog = strict_json_loads(self.store.read_bytes(catalog_path))
                self.progress("inspection_reused")
                return {"status": "inspected", "real_preflight": False, **catalog}
            started = time.monotonic()
            reading: dict[str, Any] = dict(
                purpose="format_and_integrity", bytes=0, per_file_bytes={}, complete=False
            )
            self.reads.append(reading)

            def supply_progress(name: str, rows: int, size: int) -> None:
                reading["per_file_bytes"][name] = size
                reading["bytes"] = sum(reading["per_file_bytes"].values())
                self._scan_progress(name, rows)

            if self.plan is not None:
                hashes = self._input_hashes(input_dir)
                checked_files = {}
                for table, filename in FILES.items():
                    expected = profile.expected_files[table]
                    size = paths[table].stat().st_size
                    if (
                        hashes[table] != expected["expected_sha256"]
                        or size != expected["expected_size_bytes"]
                    ):
                        raise InputValidationError(
                            f"Вход не соответствует проверенной поставке: {filename}"
                        )
                    with closing(
                        row_batches(
                            paths[table],
                            table,
                            profile,
                            batch_size=1,
                            max_batch_chars=max_batch_chars,
                        )
                    ) as rows:
                        next(rows, None)  # Заголовок; полный проход поведения/поиска — при отборе.
                    checked_files[filename] = dict(sha256=hashes[table], size_bytes=size, rows=None)
                raw = {"files": checked_files}
            else:
                raw = inspect_supply(
                    input_dir,
                    strict_json_loads(profile_path.read_bytes()),
                    limits=SupplyLimits(batch_size=batch_size, max_batch_chars=max_batch_chars),
                    progress=supply_progress,
                )
            reading.update(complete=True, seconds=time.monotonic() - started)
            catalog = {
                "schema_version": "1",
                "scope": "actual",
                "profile_sha256": profile.sha256,
                "files": {
                    key: {**raw["files"][name], "name": name, "coverage_confirmed": False}
                    for key, name in FILES.items()
                },
                "disk_plan": {"input_bytes": sum(p.stat().st_size for p in paths.values())},
                "coverage_status": "unknown",
                "real_preflight": False,
            }
        else:
            catalog = inspect_inputs(
                paths, profile, batch_size=batch_size, max_batch_chars=max_batch_chars
            )
        self.inputs = {key: item["sha256"] for key, item in catalog["files"].items()}
        self.inputs["profile"] = profile.sha256
        content = profile_path.read_bytes()
        if sha256_bytes(content) != profile.sha256:
            raise InputValidationError("Профиль изменился во время чтения")
        descriptor = self._remember(
            self.store.publish_bytes(
                "inputs/profile.json",
                content,
                kind="json",
                inputs={"profile": profile.sha256},
                **self.common,
            )
        )
        saved = self._remember(
            self.store.publish_json(
                "inputs/catalog.json",
                catalog,
                inputs=self.inputs,
                artifact_dependencies={descriptor.path: descriptor.descriptor_sha256},
                **self.common,
            )
        )
        self.progress(
            "inspected",
            input_rows=sum(item["rows"] or 0 for item in catalog["files"].values()),
            input_bytes=catalog["disk_plan"]["input_bytes"],
        )
        self._verify_current(input_dir, profile_path)
        return {
            "status": "inspected",
            "catalog_path": saved.path,
            "catalog_sha256": saved.sha256,
            "real_preflight": False,
            **catalog,
        }

    def _verify_tree(
        self, path: str, expected: str, ancestry: frozenset[str] = frozenset()
    ) -> None:
        if path in ancestry:
            raise InputValidationError("Цикл зависимостей контрольной точки")
        if path in self._verified_trees:
            if self._verified_trees[path] != expected:
                raise InputValidationError("Изменилась идентичность контрольной точки")
            return
        descriptor = self.store.describe(path)
        if (
            descriptor.descriptor_sha256 != expected
            or descriptor.code_sha256 != self.code
            or descriptor.run_id != self.run_id
            or descriptor.scope != self.common["scope"]
            or descriptor.config_id != "e1_code"
        ):
            raise InputValidationError("Несовместимая контрольная точка подготовки")
        self.store.read_by_sha256(descriptor.sha256)
        for parent, digest in descriptor.artifact_dependencies.items():
            self._verify_tree(parent, digest, ancestry | {path})
        self._remember(descriptor)
        self._verified_trees[path] = expected

    def _materialize(
        self,
        input_dir: Path,
        profile_path: Path,
        plan: PreparationPlan,
        sampled: dict[str, Any],
        *,
        batch_size: int,
        max_batch_chars: int,
    ) -> dict[str, Any]:
        """Один поток целых сессий в общие таблицы; математика не меняется."""
        profile = load_profile(profile_path)
        periods = {sid: period for period, ids in sampled["selection"].items() for sid in ids}
        selected = frozenset(periods)
        if len(selected) > plan.max_sessions:
            raise InputValidationError("Превышение числа выбранных сессий")

        def stream(table: str) -> Any:
            started = time.monotonic()
            reading: dict[str, Any] = dict(
                purpose="selected_session_rows", table=table, bytes=None, complete=False
            )
            self.reads.append(reading)
            yield from iter_events(
                input_dir / FILES[table],
                table,
                profile,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
                file_sha256=self.inputs[table],
                session_ids=selected,
                progress=self._scan_progress,
            )
            reading.update(
                complete=True,
                bytes=(input_dir / FILES[table]).stat().st_size,
                seconds=time.monotonic() - started,
            )

        raw_rows: Counter[str] = Counter()
        product_skus: set[str] = set()
        event_counts: Counter[str] = Counter()
        session_counts: Counter[str] = Counter()
        sku_sessions: dict[str, Counter[str]] = {"detail": Counter(), "purchase": Counter()}
        target_skus: set[str] = set()
        target_sizes: Counter[int] = Counter()
        seen: set[str] = set()
        positives = block_count = request_count = repeated_detail_pairs = target_purchase_events = 0
        tables: dict[str, dict[str, Any]] = {}
        kinds = (
            ("canonical", CanonicalBlock, None),
            ("prefixes", PreparedRequest, "prefix"),
            ("targets", TargetRecord, "targets"),
        )
        with ExitStack() as stack:
            session_path = "preparation/sessions.parquet"
            session_writer = stack.enter_context(
                TableWriter(self.store.root / session_path, SessionRecord)
            )
            self._temporary_tables.add(session_writer.temporary)
            writers: dict[str, dict[str, TableWriter]] = {}
            for period, ids in sampled["selection"].items():
                if not ids:
                    continue
                writers[period] = {}
                for name, kind, role in kinds:
                    path = (
                        f"canonical/{period}.parquet"
                        if name == "canonical"
                        else f"datasets/{period}/{name}.parquet"
                    )
                    writers[period][name] = stack.enter_context(
                        TableWriter(self.store.root / path, kind)
                    )
                    self._temporary_tables.add(writers[period][name].temporary)
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix=".prepare-", dir=self.store.root)
            )
            blocks = stack.enter_context(
                closing(
                    canonicalize(
                        itertools.chain(stream("browsing"), stream("search")),
                        temp_dir=Path(temporary),
                        chunk_size=plan.chunk_size,
                        chunk_bytes=plan.chunk_bytes,
                        fan_in=plan.fan_in,
                        max_group_events=plan.max_group_events,
                        max_group_bytes=plan.max_group_bytes,
                        check_resources=lambda: self._check_budget(force=True),
                    )
                )
            )
            for (namespace, sid), group in itertools.groupby(
                blocks, key=lambda block: (block.source_namespace, block.session_id)
            ):
                if namespace != plan.source_namespace or sid not in selected or sid in seen:
                    raise InputValidationError("Чужая или повторная выбранная сессия")
                rows: list[CanonicalBlock] = []
                size = 0
                for block in group:
                    size += len(to_json(block))
                    if len(rows) >= plan.max_session_blocks or size > plan.max_session_bytes:
                        raise InputValidationError("Полная сессия превышает объявленный бюджет")
                    rows.append(block)
                row_counts = Counter(
                    origin.split(":", 1)[0]
                    for block in rows
                    for event in block.events
                    for origin in event.origins
                )
                requests = self._requests(rows, plan)
                session, targets = prepare_targets(
                    rows, requests, plan.coverage, plan.periods, plan.task
                )
                expected = sampled["selected_rows"][sid]
                if (
                    not session.admitted
                    or session.period != periods[sid]
                    or session.admission_ms != expected["first_request_ms"]
                    or session.last_event_ms != expected["last_event_ms"]
                    or any(row_counts[name] != n for name, n in expected["source_rows"].items())
                ):
                    raise InputValidationError(
                        "Отбор и полный журнал сессии не совпадают; потеря исходных строк"
                    )
                output = writers[periods[sid]]
                output["canonical"].append(sid, rows)
                self.progress("canonical_session", sessions=len(seen) + 1, blocks=len(rows))
                output["prefixes"].append(sid, requests)
                output["targets"].append(sid, targets)
                session_writer.append(sid, (session,))
                seen.add(sid)
                block_count += len(rows)
                request_count += len(requests)
                positives += int(any(target.purchases for target in targets))
                raw_rows.update(row_counts)
                actions: dict[str, Counter[str]] = {"detail": Counter(), "purchase": Counter()}
                for block in rows:
                    for event in block.events:
                        if event.sku is None:
                            continue
                        if event.event_type in {"detail", "add", "remove", "purchase"}:
                            product_skus.add(event.sku)
                        if event.event_type in actions:
                            actions[event.event_type][event.sku] += 1
                        if (
                            event.event_type == "purchase"
                            and session.admission_ms is not None
                            and session.target_end_ms is not None
                            and session.admission_ms < block.event_time_ms <= session.target_end_ms
                        ):
                            target_purchase_events += 1
                for action, occurrences in actions.items():
                    event_counts[action] += occurrences.total()
                    session_counts[action] += bool(occurrences)
                    sku_sessions[action].update(occurrences.keys())
                repeated_detail_pairs += sum(n >= 2 for n in actions["detail"].values())
                for target in targets:
                    target_sizes[len(target.purchases)] += 1
                    target_skus.update(target.purchases)
                self.progress("prepared_session", sessions=len(seen), requests=len(requests))
            if seen != selected:
                raise InputValidationError("Потеряны выбранные сессии")
            self._check_budget(force=True)
        tables[session_path] = table_info(self.store.root / session_path, SessionRecord)
        for period in writers:
            for name, kind, role in kinds:
                path = (
                    f"canonical/{period}.parquet"
                    if name == "canonical"
                    else f"datasets/{period}/{name}.parquet"
                )
                tables[path] = table_info(self.store.root / path, kind, period=period, role=role)
        return dict(
            canonical_sessions=len(seen),
            canonical_blocks=block_count,
            request_count=request_count,
            N_obs=len(seen),
            N_plus=positives,
            tables=tables,
            composition={
                "event_semantics": plan.task.event_semantics,
                "source_rows": {
                    FILES[kind]: raw_rows[FILES[kind]] for kind in ("browsing", "search")
                },
                "product_skus": len(product_skus),
                **{
                    kind: {
                        "events": event_counts[kind],
                        "sessions": session_counts[kind],
                        "skus": len(frequencies),
                        "skus_in_at_least_2_sessions": sum(n >= 2 for n in frequencies.values()),
                        "skus_in_at_least_3_sessions": sum(n >= 3 for n in frequencies.values()),
                    }
                    for kind, frequencies in sku_sessions.items()
                },
                "repeated_detail_session_sku_pairs": repeated_detail_pairs,
                "target_window": {
                    "purchase_events": target_purchase_events,
                    "skus": len(target_skus),
                    "sessions_by_target_size": {
                        str(n): count for n, count in sorted(target_sizes.items())
                    },
                },
            },
        )

    def _select_sessions(
        self,
        input_dir: Path,
        profile_path: Path,
        plan: PreparationPlan,
        *,
        batch_size: int,
        max_batch_chars: int,
    ) -> dict[str, Any]:
        """Полный проход кратких сведений; SQLite ограничивает память, не совокупность."""
        profile = load_profile(profile_path)
        chosen: dict[str, list[tuple[tuple[bytes, bytes], str]]] = {
            p.name: [] for p in plan.periods
        }
        selected_rows: dict[str, dict[str, Any]] = {}
        counts = {p.name: {"eligible": 0, "admitted": 0, "selected": 0} for p in plan.periods}
        exclusions: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        considered = 0
        # Временная БД не является контрольной точкой: после прерывания проход повторяется.
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=self.store.root) as temporary:
            with closing(sqlite3.connect(Path(temporary) / "sessions.sqlite")) as db:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA cache_size=-16384")
                db.execute(
                    "CREATE TABLE sessions (sid TEXT PRIMARY KEY, detail INTEGER, purchase INTEGER, last INTEGER NOT NULL, browsing INTEGER NOT NULL, search INTEGER NOT NULL) WITHOUT ROWID"
                )
                insert = """INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sid) DO UPDATE SET
                    detail=CASE WHEN detail IS NULL THEN excluded.detail WHEN excluded.detail IS NULL THEN detail ELSE min(detail,excluded.detail) END,
                    purchase=CASE WHEN purchase IS NULL THEN excluded.purchase WHEN excluded.purchase IS NULL THEN purchase ELSE min(purchase,excluded.purchase) END,
                    last=max(last,excluded.last), browsing=browsing+excluded.browsing,
                    search=search+excluded.search"""
                for table in ("browsing", "search"):
                    started = time.monotonic()
                    reading: dict[str, Any] = dict(
                        purpose="session_admission", table=table, bytes=None, complete=False
                    )
                    self.reads.append(reading)
                    rows = 0
                    for batch in row_batches(
                        input_dir / FILES[table],
                        table,
                        profile,
                        batch_size=batch_size,
                        max_batch_chars=max_batch_chars,
                        progress=self._scan_progress,
                    ):
                        values = []
                        for _, row in batch:
                            sid, time_ms = row["session_id"], timestamp(row["timestamp"])
                            id_bytes(sid)
                            action = row.get("action", "")
                            if table == "browsing":
                                if action not in {
                                    "",
                                    "page",
                                    "detail",
                                    "add",
                                    "remove",
                                    "purchase",
                                }:
                                    raise InputValidationError("Неизвестное товарное действие")
                                if action in {"detail", "add", "remove", "purchase"}:
                                    id_bytes(row["sku"])
                                if profile.scope == "actual" and (
                                    row["event_type"] not in profile.event_types
                                    or action not in profile.actions
                                ):
                                    raise InputValidationError(
                                        "Несовместимый тип или действие поведения"
                                    )
                            values.append(
                                (
                                    sid,
                                    time_ms if action == "detail" else None,
                                    time_ms if action == "purchase" else None,
                                    time_ms,
                                    int(table == "browsing"),
                                    int(table == "search"),
                                )
                            )
                        db.executemany(insert, values)
                        rows += len(batch)
                        self._check_budget()
                    db.commit()
                    reading.update(
                        complete=True,
                        rows=rows,
                        bytes=(input_dir / FILES[table]).stat().st_size,
                        seconds=time.monotonic() - started,
                    )
                for sid, detail, purchase, last, browsing, search in db.execute(
                    "SELECT * FROM sessions"
                ):
                    considered += 1
                    first = (
                        detail
                        if detail is not None and (purchase is None or detail < purchase)
                        else None
                    )
                    admission = admit(first, plan.periods, plan.task, last_event_ms=last)
                    if admission.period:
                        counts[admission.period]["eligible"] += int(admission.eligible)
                        counts[admission.period]["admitted"] += int(admission.admitted)
                    if admission.reason:
                        exclusions[admission.reason] += 1
                        if len(examples) < 16:
                            examples.append(
                                dict(
                                    session_id=sid,
                                    reason=admission.reason,
                                    first_request_ms=first,
                                    last_event_ms=last,
                                )
                            )
                    if admission.admitted and admission.period is not None:
                        period = next(p for p in plan.periods if p.name == admission.period)
                        candidates = chosen[period.name]
                        if period.sample_size:
                            key = session_key(sid, plan.seed)
                            if len(candidates) < period.sample_size or key < candidates[-1][0]:
                                bisect.insort(candidates, (key, sid))
                                selected_rows[sid] = dict(
                                    first_request_ms=first,
                                    last_event_ms=last,
                                    source_rows={
                                        FILES["browsing"]: browsing,
                                        FILES["search"]: search,
                                    },
                                )
                                if len(candidates) > period.sample_size:
                                    selected_rows.pop(candidates.pop()[1])
                    if considered % 4096 == 0:
                        self._check_budget()
                self._check_budget(force=True)
        selection = {name: [sid for _, sid in candidates] for name, candidates in chosen.items()}
        for name, count in counts.items():
            count["selected"] = len(selection[name])
        result = dict(
            selection=selection,
            selected_rows=selected_rows,
            considered_sessions=considered,
            exclusions=dict(exclusions),
            exclusion_examples=examples,
            period_counts=counts,
            evaluation_protocol=EVALUATION_PROTOCOL,
            rule="lowest_seeded_hash_after_calendar_admission",
            normative_sample=True,
        )
        self.progress(
            "selection_complete", sessions=len(selected_rows), considered_sessions=considered
        )
        return result

    @staticmethod
    def _requests(
        blocks: Sequence[CanonicalBlock], plan: PreparationPlan
    ) -> tuple[PreparedRequest, ...]:
        def snapshots() -> Any:
            state = empty_prefix(plan.source_namespace, blocks[0].session_id, plan.task)
            for block in blocks:
                state = reduce_state(state, block, plan.task).next_snapshot
                yield state

        return make_schedule(snapshots(), plan.task)

    def prepare(
        self,
        input_dir: Path,
        profile_path: Path,
        plan_path: Path,
        *,
        batch_size: int,
        max_batch_chars: int,
        evaluation_config: Path | None = None,
    ) -> dict[str, Any]:
        if self._preparing:
            raise InputValidationError("Подготовка уже выполняется")
        self._preparing = True
        try:
            report = self._prepare(
                input_dir,
                profile_path,
                plan_path,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
                evaluation_config=evaluation_config,
            )
            return {
                **report,
                "status": "prepared",
                "report_path": REPORT,
                "report_sha256": sha256_file(self.store.root / REPORT),
                "execution": self._measurements(),
            }
        finally:
            self._preparing = False

    def _measurements(self) -> dict[str, Any]:
        result_bytes = result_files = 0
        for path in self.store.root.rglob("*"):
            if path.is_file():
                result_bytes += path.stat().st_size
                result_files += 1
        self.disk_peak = max(self.disk_peak, result_bytes)
        return {
            "status": "completed",
            "elapsed_seconds": time.monotonic() - self.started,
            **_peak_memory(),
            "reads": self.reads,
            "logical_read_bytes": sum(item["bytes"] or 0 for item in self.reads),
            "read_bytes_complete": all(item["bytes"] is not None for item in self.reads),
            "reused_stages": self.reused_stages,
            "temporary_peak_bytes": self.temp_peak,
            "run_disk_peak_bytes": self.disk_peak,
            "result_bytes_before_measurement": result_bytes,
            "result_files_before_measurement": result_files,
            "disk_measurement": "temporary: bounded-stage callbacks; result: at completion; no per-session logs or copies",
        }

    def _prepare(
        self,
        input_dir: Path,
        profile_path: Path,
        plan_path: Path,
        *,
        batch_size: int,
        max_batch_chars: int,
        evaluation_config: Path | None = None,
    ) -> dict[str, Any]:
        content, profile_content = plan_path.read_bytes(), profile_path.read_bytes()
        plan = from_json(content, PreparationPlan)
        self.plan = plan
        self.common["scope"] = plan.scope
        self._check_budget(force=True)
        profile = load_profile(profile_path)
        if plan.diagnostic_sessions:
            raise InputValidationError(
                "Технический режим не является подготовкой observed_purchases_v1"
            )
        active = [p.name for p in plan.periods if p.sample_size]
        evaluation_admission = None
        evaluation_bytes = None
        if (
            "D" in active
            or plan.scope == "actual"
            and any(name not in {"A_G", "B"} for name in active)
        ):
            if active not in (["C_smoke"], ["D"]) or evaluation_config is None:
                raise InputValidationError(
                    "C_smoke требует post_training, D — pre_D; остальные периоды закрыты"
                )
            from coveo_recsys.cli.admission import require_evaluation_preparation

            evaluation_bytes = evaluation_config.read_bytes()
            evaluation_admission = {
                "settings_sha256": sha256_bytes(evaluation_bytes),
                "manifest_sha256": require_evaluation_preparation(
                    self.root, evaluation_config, plan, profile_sha256=sha256_bytes(profile_content)
                ),
            }
        elif evaluation_config is not None:
            raise InputValidationError("Допуск оценки используется только для реального C_smoke")

        def check_evaluation_admission() -> None:
            if evaluation_config is None or evaluation_admission is None:
                return
            if evaluation_config.read_bytes() != evaluation_bytes or (
                require_evaluation_preparation(
                    self.root, evaluation_config, plan, profile_sha256=sha256_bytes(profile_content)
                )
                != evaluation_admission["manifest_sha256"]
            ):
                raise InputValidationError("Допуск C_smoke изменился во время подготовки")

        if (
            plan.source_namespace != profile.namespace
            or plan.task.query_dimension != profile.dimensions["query"]
            or plan.scope != profile.scope
        ):
            raise InputValidationError("План и профиль несовместимы")
        if plan.scope == "actual" and plan.coverage.confirmed_at_ms is not None:
            raise InputValidationError("Профиль поставки не подтверждает покрытие реальных событий")
        self.inputs = {
            **self._input_hashes(input_dir),
            "profile": sha256_bytes(profile_content),
            "plan": sha256_bytes(content),
        }
        if self.inputs["profile"] != profile.sha256:
            raise InputValidationError("Профиль изменился во время чтения")
        if profile.scope == "synthetic":
            inspect_inputs(
                {table: input_dir / filename for table, filename in FILES.items()},
                profile,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
                progress=self._scan_progress,
            )
        else:
            for table, filename in FILES.items():
                path = input_dir / filename
                expected = profile.expected_files[table]
                if (
                    self.inputs[table] != expected["expected_sha256"]
                    or path.stat().st_size != expected["expected_size_bytes"]
                ):
                    raise InputValidationError("Вход не соответствует проверенной поставке")
                with closing(
                    row_batches(path, table, profile, batch_size=1, max_batch_chars=max_batch_chars)
                ) as rows:
                    next(rows, None)
        identity = {
            "format_version": FORMAT_VERSION,
            "source_sha256": self.code,
            "run_id": self.run_id,
            "scope": plan.scope,
            "inputs": self.inputs,
        }
        if evaluation_admission is not None:
            identity["evaluation_admission"] = evaluation_admission
        if (self.store.root / REPORT).exists():
            digest = sha256_file(self.store.root / REPORT)
            report = read_report(self.store.root, report_sha256=digest, verify_tables=False)
            if any(report.get(key) != value for key, value in identity.items()):
                raise InputValidationError("Подготовка несовместима с кодом, входами или планом")
            self.progress("prepared_reused", sessions=report["canonical_sessions"])
            self._verify_current(input_dir, profile_path, plan_path)
            read_report(self.store.root, report_sha256=digest)
            check_evaluation_admission()
            self._check_budget(force=True)
            return dict(report)
        if (self.store.root / "canonical/index.json").exists() or (
            self.store.root / "preparation/selection.json"
        ).exists():
            raise InputValidationError("Для компактной подготовки нужен новый каталог запуска")
        identity_path = self.store.root / IDENTITY
        if identity_path.exists():
            if strict_json_loads(identity_path.read_bytes()) != identity:
                raise InputValidationError("Незавершённая подготовка имеет другие код или входы")
        else:
            self.store._atomic_write(IDENTITY, canonical_json_bytes(identity))
        sampled = self._select_sessions(
            input_dir, profile_path, plan, batch_size=batch_size, max_batch_chars=max_batch_chars
        )
        result = self._materialize(
            input_dir,
            profile_path,
            plan,
            sampled,
            batch_size=batch_size,
            max_batch_chars=max_batch_chars,
        )
        counts, selection = sampled["period_counts"], sampled["selection"]
        report = {
            **identity,
            "status": "completed",
            "contract_version": PREPARATION_VERSION,
            "evidence_scope": "actual_preparation" if plan.scope == "actual" else "synthetic_code",
            "real_preflight": False,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "telemetry_complete": None,
            "profile_json": profile_content.decode("utf-8"),
            "plan_json": content.decode("utf-8"),
            "sampled": sampled,
            "selection": selection,
            "period_counts": counts,
            "exclusions": sampled["exclusions"],
            "exclusion_examples": sampled["exclusion_examples"],
            "considered_sessions": sampled["considered_sessions"],
            "N_eligible": sum(counts[p.name]["eligible"] for p in plan.periods if p.sample_size),
            "N_admitted": sum(counts[p.name]["admitted"] for p in plan.periods if p.sample_size),
            "selection_shortfalls": {
                p.name: {
                    "requested": p.sample_size,
                    "admitted": counts[p.name]["admitted"],
                    "selected": len(selection[p.name]),
                    "reason": "fewer_admitted_sessions",
                }
                for p in plan.periods
                if counts[p.name]["admitted"] < p.sample_size
            },
            **result,
        }
        self.progress(
            "prepared", sessions=report["canonical_sessions"], requests=report["request_count"]
        )
        self._verify_current(input_dir, profile_path, plan_path)
        verify_tables(self.store.root, report)
        check_evaluation_admission()
        self._check_budget(force=True)
        report["measurements"] = self._measurements()
        if strict_json_loads(identity_path.read_bytes()) != identity:
            raise InputValidationError("Изменилась идентичность незавершённой подготовки")
        write_report(self.store.root, report, before_publish=lambda: self._check_budget(force=True))
        identity_path.unlink()
        return report
