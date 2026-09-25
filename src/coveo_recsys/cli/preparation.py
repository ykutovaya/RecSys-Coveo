"""Сборка сценария Э1 из принятых компонентов; только искусственные входы."""

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
from coveo_recsys.data.io import FILES, inspect_inputs, iter_events, load_profile
from coveo_recsys.data.periods import sample_sessions
from coveo_recsys.data.schedule import make_schedule
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
            or any(sha256_file(input_dir / FILES[key]) != self.inputs[key] for key in FILES)
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
        paths = {key: input_dir / filename for key, filename in FILES.items()}
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
            or descriptor.scope != "synthetic"
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
        streams = [
            iter_events(
                input_dir / FILES[table],
                table,
                profile,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
                file_sha256=self.inputs[table],
            )
            for table in ("browsing", "search")
        ]
        index: list[dict[str, Any]] = []
        dependencies = {
            key: self.artifacts[key] for key in ("inputs/catalog.json", "inputs/plan.json")
        }
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
        if any(sha256_file(input_dir / FILES[key]) != self.inputs[key] for key in FILES):
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
        with self._operation(input_dir, profile_path, plan_path):
            return self._prepare(
                input_dir,
                profile_path,
                plan_path,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
            )

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
        profile = load_profile(profile_path)
        if (
            plan.source_namespace != profile.namespace
            or plan.task.query_dimension != profile.dimensions["query"]
        ):
            raise InputValidationError("План и профиль несовместимы")
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

        index = self._canonical(
            input_dir, profile_path, plan, batch_size=batch_size, max_batch_chars=max_batch_chars
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
                    if s.period == p.name and s.exclusion_reason is None
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
        for item, session in zip(index, sessions, strict=True):
            if session.session_id not in selected:
                continue
            blocks = self._records(item["path"], CanonicalBlock)
            requests = self._requests(blocks, plan)
            _, targets = prepare_targets(blocks, requests, plan.coverage, plan.periods, plan.task)
            base = f"datasets/{session.period}"
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
                configuration={"period": session.period, "dataset_kind": "prefixes"},
            )
            self._table(
                target_path,
                targets,
                TargetRecord,
                role="targets",
                dependencies={**dependencies, prefix_path: self.artifacts[prefix_path]},
                configuration={"period": session.period, "dataset_kind": "targets"},
            )
            cohort_path = None
            if session.period == "A_G":
                cohort_path = f"cohort/A_G/{item['key']}.parquet"
                self._table(
                    cohort_path,
                    historical_cohort(blocks, session, plan.coverage, plan.periods, plan.task),
                    CohortRecord,
                    dependencies=dependencies,
                    configuration={"period": "A_G"},
                )
            datasets.append(
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
            self.progress("prepared_session", sessions=len(datasets), requests=len(requests))
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
            status="prepared",
            scope="synthetic",
            evidence_scope="synthetic_code",
            run_id=self.run_id,
            real_preflight=False,
            contract_version=CONTRACT_VERSION,
            source_sha256=self.code,
            inputs=self.inputs,
            canonical_sessions=len(index),
            canonical_blocks=sum(item["blocks"] for item in index),
            request_count=sum(item["requests"] for item in datasets),
            selection=selection,
            exclusions=dict(Counter(s.exclusion_reason for s in sessions if s.exclusion_reason)),
            period_counts=counts,
            datasets=datasets,
            artifacts=dict(self.artifacts),
            limitations=[
                "Фактическая схема Coveo не подтверждена",
                "Каталог I_G и кандидаты ещё не построены",
                "Локальное разделение наборов не заменяет рабочие права Э6",
                "Время и память на объёме Coveo ещё не измерены",
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
