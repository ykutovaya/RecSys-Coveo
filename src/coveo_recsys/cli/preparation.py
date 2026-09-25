"""Подготовка Э1 и отдельная ограниченная проверка реальных сессий."""

import bisect
import hashlib
import itertools
import json
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import pyarrow as pa

from coveo_recsys.artifacts import (
    ArtifactDescriptor,
    LocalArtifactStore,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
)
from coveo_recsys.contracts import CONTRACT_VERSION, InputValidationError
from coveo_recsys.contracts.preparation import PERIOD_NAMES, PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import CanonicalBlock, Record
from coveo_recsys.contracts.serialization import from_json, record_schema, to_json
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
from coveo_recsys.data.periods import sample_sessions, session_key
from coveo_recsys.data.schedule import make_schedule
from coveo_recsys.data.supply import SupplyLimits, inspect_supply
from coveo_recsys.state import empty_prefix, reduce_state
from coveo_recsys.targets import historical_cohort, prepare_targets
from coveo_recsys.targets.cohort import CohortRecord
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

R = TypeVar("R", bound=Record)
TABLE_SCHEMA = pa.schema([pa.field("record_json", pa.binary(), nullable=False)])


def _peak_memory() -> dict[str, Any]:
    """Пик RSS/working set процесса, не оценка памяти рекомендательной системы."""
    import sys

    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("faults", wintypes.DWORD),
                    *[
                        (name, ctypes.c_size_t)
                        for name in (
                            "peak",
                            "current",
                            "peak_paged",
                            "paged",
                            "peak_nonpaged",
                            "nonpaged",
                            "pagefile",
                            "peak_pagefile",
                        )
                    ],
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            measure = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not measure(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error())
            value, method = int(counters.peak), "GetProcessMemoryInfo.PeakWorkingSetSize"
        else:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            value *= 1 if sys.platform == "darwin" else 1024
            method = "getrusage.ru_maxrss"
        return {"peak_process_memory_bytes": value, "memory_method": method}
    except (OSError, AttributeError, ImportError) as exc:
        return {"peak_process_memory_bytes": None, "memory_reason": type(exc).__name__}


class Preparation:
    def __init__(self, store: LocalArtifactStore, root: Path, run_id: str) -> None:
        assert_source_binding(root)
        self.store, self.root, self.run_id = store, root, run_id
        self.code = source_sha256(root)
        self.common: dict[str, Any] = dict(
            code_sha256=self.code, run_id=run_id, scope="synthetic", config_id="e1_code", reuse=True
        )
        self.execution_id = uuid4().hex
        self.started = time.monotonic()
        self.sequence = 0
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

    def _check_budget(self, *, force: bool = False) -> None:
        if self.plan is None or not self.plan.diagnostic_sessions:
            return
        now = time.monotonic()
        if not force and now - self._last_budget_check < 0.5:
            return
        self._last_budget_check = now
        memory = _peak_memory()["peak_process_memory_bytes"]
        files = [(p, p.stat().st_size) for p in self.store.root.rglob("*") if p.is_file()]
        disk = sum(size for _, size in files)
        temporary = sum(
            size for p, size in files if any(part.startswith(".prepare-") for part in p.parts)
        )
        self.temp_peak = max(self.temp_peak, temporary)
        self.disk_peak = max(self.disk_peak, disk)
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
                    self._check_budget()
            hashes[table] = digest.hexdigest()
            reading.update(complete=True, seconds=time.monotonic() - started)
        return hashes

    @contextmanager
    def _operation(
        self, input_dir: Path, profile_path: Path, plan_path: Path | None = None
    ) -> Iterator[None]:
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
        if step.endswith("_reused"):
            self.reused_stages.append(step)
        if step != "failed":
            self._check_budget(force=True)
        self.sequence += 1
        self.store.publish_json(
            f"preparation/progress/{self.execution_id}/{self.sequence:06}.json",
            {
                "step": step,
                "elapsed_seconds": time.monotonic() - self.started,
                **_peak_memory(),
                **fields,
            },
            **self.common,
        )

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

            raw = inspect_supply(
                input_dir,
                strict_json_loads(profile_path.read_bytes()),
                limits=SupplyLimits(
                    batch_size=batch_size,
                    max_batch_chars=max_batch_chars,
                    max_seconds=self.plan.max_seconds if self.plan else 1800,
                ),
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
            input_rows=sum(item["rows"] for item in catalog["files"].values()),
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

    def _table(
        self,
        path: str,
        records: Sequence[R],
        record_type: type[R],
        *,
        configuration: dict[str, Any],
        dependencies: dict[str, str],
        role: str = "auxiliary",
    ) -> None:
        schema_path = f"preparation/schemas/{record_type.__name__}.json"
        schema = self._remember(
            self.store.publish_json(
                schema_path,
                record_schema(record_type),
                schema_version=CONTRACT_VERSION,
                **self.common,
            )
        )
        table = pa.Table.from_arrays(
            [pa.array([to_json(row) for row in records], type=pa.binary())], schema=TABLE_SCHEMA
        )
        descriptor = self.store.publish_table(
            path,
            table,
            data_role=role,
            inputs=self.inputs,
            configuration={**configuration, "record_type": record_type.__name__},
            schema_version=CONTRACT_VERSION,
            artifact_dependencies={**dependencies, schema.path: schema.descriptor_sha256},
            **self.common,
        )
        self._remember(descriptor)

    def _records(self, path: str, record_type: type[R]) -> tuple[R, ...]:
        descriptor = self.store.describe(path)
        if (
            descriptor.inputs != self.inputs
            or descriptor.schema_version != CONTRACT_VERSION
            or descriptor.configuration.get("record_type") != record_type.__name__
        ):
            raise InputValidationError("Набор не соответствует входам или типу записи")
        self._verify_tree(path, self.artifacts[path])
        table = self.store.read_table(path)
        if not table.schema.equals(TABLE_SCHEMA):
            raise InputValidationError("Несовместимая схема таблицы записей")
        return tuple(from_json(value, record_type) for value in table["record_json"].to_pylist())

    def _canonical(
        self,
        input_dir: Path,
        profile_path: Path,
        plan: PreparationPlan,
        *,
        batch_size: int,
        max_batch_chars: int,
        session_ids: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        path = "canonical/index.json"
        if (self.store.root / path).exists():
            descriptor = self.store.verify_compatible(
                path,
                expected_inputs=self.inputs,
                expected_code_sha256=self.code,
                expected_schema_version=CONTRACT_VERSION,
                expected_configuration={},
            )
            self._verify_tree(path, descriptor.descriptor_sha256)
            result: list[dict[str, Any]] = strict_json_loads(self.store.read_bytes(path))
            self.progress("canonical_reused", sessions=len(result))
            return result
        profile = load_profile(profile_path)

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
                session_ids=session_ids,
                progress=self._scan_progress,
            )
            reading.update(
                complete=True,
                bytes=(input_dir / FILES[table]).stat().st_size,
                seconds=time.monotonic() - started,
            )

        streams = [stream(table) for table in ("browsing", "search")]
        index: list[dict[str, Any]] = []
        dependencies = {
            key: self.artifacts[key] for key in ("inputs/catalog.json", "inputs/plan.json")
        }
        if session_ids is not None:
            dependencies["preparation/technical_selection.json"] = self.artifacts[
                "preparation/technical_selection.json"
            ]
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=self.store.root) as temporary:
            with closing(
                canonicalize(
                    itertools.chain(*streams),
                    temp_dir=Path(temporary),
                    chunk_size=plan.chunk_size,
                    chunk_bytes=plan.chunk_bytes,
                    fan_in=plan.fan_in,
                    max_group_events=plan.max_group_events,
                    max_group_bytes=plan.max_group_bytes,
                    check_resources=lambda: self._check_budget(force=True),
                )
            ) as blocks:
                for (namespace, session_id), group in itertools.groupby(
                    blocks,
                    key=lambda b: (b.source_namespace, b.session_id),
                ):
                    if namespace != plan.source_namespace or len(index) >= plan.max_sessions:
                        raise InputValidationError("Чужое пространство или превышение числа сессий")
                    rows: list[CanonicalBlock] = []
                    size = 0
                    for block in group:
                        size += len(to_json(block))
                        if len(rows) >= plan.max_session_blocks or size > plan.max_session_bytes:
                            raise InputValidationError("Полная сессия превышает объявленный бюджет")
                        rows.append(block)
                    key = sha256_bytes(
                        json.dumps([namespace, session_id], ensure_ascii=False).encode("utf-8")
                    )
                    table_path = f"canonical/{key}.parquet"
                    self._table(
                        table_path,
                        rows,
                        CanonicalBlock,
                        configuration={"source_namespace": namespace, "session_id": session_id},
                        dependencies=dependencies,
                    )
                    index.append(
                        {
                            "session_id": session_id,
                            "key": key,
                            "path": table_path,
                            "blocks": len(rows),
                            "encoded_bytes": size,
                        }
                    )
                    self.progress(
                        "canonical_session",
                        sessions=len(index),
                        blocks=len(rows),
                        encoded_bytes=size,
                    )
        if any(digest != self.inputs[key] for key, digest in self._input_hashes(input_dir).items()):
            raise InputValidationError("Вход изменился во время подготовки")
        self._remember(
            self.store.publish_json(
                path,
                index,
                inputs=self.inputs,
                schema_version=CONTRACT_VERSION,
                artifact_dependencies={
                    **dependencies,
                    **{item["path"]: self.artifacts[item["path"]] for item in index},
                },
                **self.common,
            )
        )
        return index

    def _technical_selection(
        self,
        input_dir: Path,
        profile_path: Path,
        plan: PreparationPlan,
        *,
        batch_size: int,
        max_batch_chars: int,
    ) -> tuple[str, ...]:
        """Минимальные хеши наблюдаемых ID, не нормативная выборка допущенных A_G."""
        path = "preparation/technical_selection.json"
        if (self.store.root / path).exists():
            descriptor = self.store.verify_compatible(
                path,
                expected_inputs=self.inputs,
                expected_code_sha256=self.code,
                expected_schema_version=CONTRACT_VERSION,
                expected_configuration={},
            )
            self._verify_tree(path, descriptor.descriptor_sha256)
            self.progress("technical_selection_reused")
            return tuple(strict_json_loads(self.store.read_bytes(path))["session_ids"])
        profile, period = load_profile(profile_path), plan.periods[0]
        chosen: list[tuple[tuple[bytes, bytes], str]] = []
        seen: set[str] = set()
        for table in ("browsing", "search"):
            started = time.monotonic()
            reading: dict[str, Any] = dict(
                purpose="technical_id_selection", table=table, bytes=None, complete=False
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
                for _, row in batch:
                    rows += 1
                    if not period.start_ms <= timestamp(row["timestamp"]) < period.end_ms:
                        continue
                    sid = row["session_id"]
                    if sid in seen:
                        continue
                    key = session_key(sid, plan.seed)
                    if len(chosen) == plan.diagnostic_sessions and key >= chosen[-1][0]:
                        continue
                    bisect.insort(chosen, (key, sid))
                    seen.add(sid)
                    if len(chosen) > plan.diagnostic_sessions:
                        seen.remove(chosen.pop()[1])
            reading.update(
                complete=True,
                rows=rows,
                bytes=(input_dir / FILES[table]).stat().st_size,
                seconds=time.monotonic() - started,
            )
        result = tuple(sid for _, sid in chosen)
        self._remember(
            self.store.publish_json(
                path,
                {
                    "session_ids": result,
                    "rule": "lowest_seeded_hash_of_ids_with_any_row_in_A_G",
                    "normative_sample": False,
                },
                inputs=self.inputs,
                schema_version=CONTRACT_VERSION,
                artifact_dependencies={
                    key: self.artifacts[key] for key in ("inputs/catalog.json", "inputs/plan.json")
                },
                **self.common,
            )
        )
        self.progress("technical_selection_complete", sessions=len(result))
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
    ) -> dict[str, Any]:
        try:
            with self._operation(input_dir, profile_path, plan_path):
                report = self._prepare(
                    input_dir,
                    profile_path,
                    plan_path,
                    batch_size=batch_size,
                    max_batch_chars=max_batch_chars,
                )
            self._check_budget(force=True)
        except (Exception, KeyboardInterrupt) as exc:
            if self.plan is not None and self.plan.diagnostic_sessions:
                self._execution("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise
        if self.plan is not None and self.plan.diagnostic_sessions:
            report = {**report, "status": "prepared_diagnostic", **self._execution("completed")}
        return report

    def _execution(self, status: str) -> dict[str, Any]:
        """Успех отмечается только после заключительной проверки входов и бюджета."""
        measurements = {
            "status": status,
            "elapsed_seconds": time.monotonic() - self.started,
            **_peak_memory(),
            "reads": self.reads,
            "logical_read_bytes": sum(item["bytes"] or 0 for item in self.reads),
            "read_bytes_complete": all(item["bytes"] is not None for item in self.reads),
            "reused_stages": self.reused_stages,
            "temporary_peak_bytes": self.temp_peak,
            "run_disk_peak_bytes": self.disk_peak,
            "result_bytes_before_measurement": sum(
                p.stat().st_size for p in self.store.root.rglob("*") if p.is_file()
            ),
            "disk_measurement": "at CSV batches, sorting writes/merges and stages; result includes copies/logs",
        }
        path = f"preparation/executions/{self.execution_id}.json"
        dependencies = (
            {
                "reports/preparation.json": self.store.describe(
                    "reports/preparation.json"
                ).descriptor_sha256
            }
            if status == "completed"
            else {}
        )
        self.store.publish_json(
            path,
            measurements,
            inputs=self.inputs,
            artifact_dependencies=dependencies,
            **self.common,
        )
        return {"execution_path": path, "execution": measurements}

    def _prepare(
        self,
        input_dir: Path,
        profile_path: Path,
        plan_path: Path,
        *,
        batch_size: int,
        max_batch_chars: int,
    ) -> dict[str, Any]:
        content = plan_path.read_bytes()
        plan = from_json(content, PreparationPlan)
        self.plan = plan
        self.common["scope"] = plan.scope
        self._check_budget(force=True)
        profile = load_profile(profile_path)
        if (
            plan.source_namespace != profile.namespace
            or plan.task.query_dimension != profile.dimensions["query"]
            or plan.scope != profile.scope
        ):
            raise InputValidationError("План и профиль несовместимы")
        if plan.scope == "actual" and plan.coverage.confirmed_at_ms is not None:
            raise InputValidationError("Профиль поставки не подтверждает покрытие реальных событий")
        self._inspect(
            input_dir, profile_path, batch_size=batch_size, max_batch_chars=max_batch_chars
        )
        self.inputs["plan"] = sha256_bytes(content)
        self._remember(
            self.store.publish_bytes(
                "inputs/plan.json",
                content,
                kind="json",
                inputs=self.inputs,
                schema_version=CONTRACT_VERSION,
                **self.common,
            )
        )
        report_path = "reports/preparation.json"
        if (self.store.root / report_path).exists():
            descriptor = self.store.verify_compatible(
                report_path,
                expected_inputs=self.inputs,
                expected_code_sha256=self.code,
                expected_schema_version=CONTRACT_VERSION,
                expected_configuration={},
            )
            self._verify_tree(report_path, descriptor.descriptor_sha256)
            report: dict[str, Any] = strict_json_loads(self.store.read_bytes(report_path))
            self.progress("prepared_reused", sessions=report["canonical_sessions"])
            self._verify_current(input_dir, profile_path, plan_path)
            return {**report, "report_path": report_path, "report_sha256": descriptor.sha256}

        technical_selection = (
            self._technical_selection(
                input_dir,
                profile_path,
                plan,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
            )
            if plan.diagnostic_sessions
            else None
        )
        index = self._canonical(
            input_dir,
            profile_path,
            plan,
            batch_size=batch_size,
            max_batch_chars=max_batch_chars,
            session_ids=frozenset(technical_selection) if technical_selection is not None else None,
        )
        sessions: list[SessionRecord] = []
        for item in index:
            blocks = self._records(item["path"], CanonicalBlock)
            session, _ = prepare_targets(
                blocks, self._requests(blocks, plan), plan.coverage, plan.periods, plan.task
            )
            sessions.append(session)
        parent = {"canonical/index.json": self.artifacts["canonical/index.json"]}
        self._table(
            "preparation/sessions.parquet",
            sessions,
            SessionRecord,
            configuration={},
            dependencies=parent,
        )
        selection = {
            p.name: sample_sessions(
                (
                    s.session_id
                    for s in sessions
                    if s.period == p.name
                    and s.exclusion_reason is None
                    and not plan.diagnostic_sessions
                ),
                p.sample_size,
                seed=plan.seed,
                max_sessions=plan.max_sessions,
            )
            for p in plan.periods
        }
        self._remember(
            self.store.publish_json(
                "preparation/selection.json",
                selection,
                inputs=self.inputs,
                schema_version=CONTRACT_VERSION,
                artifact_dependencies={
                    "preparation/sessions.parquet": self.artifacts["preparation/sessions.parquet"]
                },
                **self.common,
            )
        )
        selected = {sid for values in selection.values() for sid in values}
        datasets: list[dict[str, Any]] = []
        diagnostic_datasets: list[dict[str, Any]] = []
        for item, session in zip(index, sessions, strict=True):
            if session.session_id not in selected and not plan.diagnostic_sessions:
                continue
            blocks = self._records(item["path"], CanonicalBlock)
            requests = self._requests(blocks, plan)
            _, targets = prepare_targets(blocks, requests, plan.coverage, plan.periods, plan.task)
            base = "diagnostic/A_G" if plan.diagnostic_sessions else f"datasets/{session.period}"
            prefix_path, target_path = (
                f"{base}/{kind}/{item['key']}.parquet" for kind in ("prefixes", "targets")
            )
            dependencies = {
                item["path"]: self.artifacts[item["path"]],
                "preparation/selection.json": self.artifacts["preparation/selection.json"],
            }
            self._table(
                prefix_path,
                requests,
                PreparedRequest,
                role="prefix",
                dependencies=dependencies,
                configuration={
                    "period": session.period,
                    "dataset_kind": "prefixes",
                    "diagnostic": bool(plan.diagnostic_sessions),
                },
            )
            self._table(
                target_path,
                targets,
                TargetRecord,
                role="targets",
                dependencies={**dependencies, prefix_path: self.artifacts[prefix_path]},
                configuration={
                    "period": session.period,
                    "dataset_kind": "targets",
                    "diagnostic": bool(plan.diagnostic_sessions),
                },
            )
            cohort_path = None
            if session.period == "A_G" and not plan.diagnostic_sessions:
                cohort_path = f"cohort/A_G/{item['key']}.parquet"
                self._table(
                    cohort_path,
                    historical_cohort(blocks, session, plan.coverage, plan.periods, plan.task),
                    CohortRecord,
                    dependencies=dependencies,
                    configuration={"period": "A_G"},
                )
            destination = diagnostic_datasets if plan.diagnostic_sessions else datasets
            destination.append(
                dict(
                    period=session.period,
                    session_id=session.session_id,
                    canonical_path=item["path"],
                    prefixes=prefix_path,
                    targets=target_path,
                    cohort=cohort_path,
                    requests=len(requests),
                )
            )
            self.progress("prepared_session", sessions=len(destination), requests=len(requests))
        counts = {}
        for period in PERIOD_NAMES:
            eligible = sum(s.period == period for s in sessions)
            admitted = sum(s.period == period and s.exclusion_reason is None for s in sessions)
            counts[period] = dict(
                eligible=eligible,
                admitted=admitted,
                selected=len(selection[period]),
                retention=admitted / eligible if eligible else None,
                retention_reason=None if eligible else "empty_denominator",
            )
        self._verify_current(input_dir, profile_path, plan_path)
        report = dict(
            status="diagnostic_materialized" if plan.diagnostic_sessions else "prepared",
            scope=plan.scope,
            evidence_scope="actual_diagnostic" if plan.scope == "actual" else "synthetic_code",
            run_id=self.run_id,
            real_preflight=False,
            contract_version=CONTRACT_VERSION,
            source_sha256=self.code,
            inputs=self.inputs,
            canonical_sessions=len(index),
            canonical_blocks=sum(item["blocks"] for item in index),
            request_count=sum(item["requests"] for item in datasets + diagnostic_datasets),
            selection=selection,
            exclusions=dict(Counter(s.exclusion_reason for s in sessions if s.exclusion_reason)),
            period_counts=counts,
            datasets=datasets,
            diagnostic_datasets=diagnostic_datasets,
            technical_selection=technical_selection,
            artifacts=dict(self.artifacts),
            limitations=[
                "Техническая выборка не заменяет нормативный отбор и подтверждение покрытия",
                "Каталог I_G и кандидаты ещё не построены",
                "Локальное разделение наборов не заменяет рабочие права Э6",
                "Малый прогон не подтверждает масштабируемость полной подготовки",
            ],
        )
        descriptor = self.store.publish_json(
            report_path,
            report,
            inputs=self.inputs,
            schema_version=CONTRACT_VERSION,
            artifact_dependencies=self.artifacts,
            **self.common,
        )
        self.progress("prepared", sessions=len(sessions), requests=report["request_count"])
        self._verify_current(input_dir, profile_path, plan_path)
        return {**report, "report_path": report_path, "report_sha256": descriptor.sha256}
