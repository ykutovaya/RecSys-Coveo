"""B04: ограниченная внешняя сортировка пар и разрежённые отношения.

Промежуточная запись содержит (отношение, источник, сосед, сессия, задержка).
Минимум задержки вычисляется при слиянии, поэтому даже одна большая сессия
не требует словаря всех пар. Уровни слияния ограничивают число файлов.
"""

import heapq
import math
import struct
from array import array
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeAlias, cast

import numpy as np

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.historical import (
    RELATIONS,
    GraphBuildResult,
    GraphRelation,
    HistoricalCatalog,
    HistoricalConfig,
    HistoricalInput,
    Session,
)
from coveo_recsys.contracts.records import EventType
from coveo_recsys.numerics import checked_subtract, compensated_sum, counter_add

Pair: TypeAlias = tuple[int, int, int, int, int]
ProductGroup: TypeAlias = tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]
PAIR = struct.Struct("<QQQQQ")
PRODUCT_TYPES = {EventType.DETAIL, EventType.ADD, EventType.REMOVE, EventType.PURCHASE}
TARGET_RELATION = {EventType.DETAIL: 0, EventType.ADD: 1, EventType.PURCHASE: 2}


def _limit(value: int, maximum: int, name: str) -> None:
    if value > maximum:
        raise InputValidationError(f"Превышен ресурсный предел {name}: {value} > {maximum}")


@dataclass(frozen=True)
class _Run:
    path: Path
    count: int


class _PairSort:
    def __init__(
        self,
        directory: Path,
        config: HistoricalConfig,
        metrics: dict[str, Any],
        check_budget: Callable[[], None],
    ) -> None:
        self.directory = directory
        self.config = config
        self.metrics = metrics
        self.check_budget = check_budget
        self.buffer: list[Pair] = []
        self.levels: list[list[_Run]] = []
        self.temp_bytes = 0
        self.next_file = 0

    def _reserve(self, count: int) -> Path:
        self.check_budget()
        self.temp_bytes += count * PAIR.size
        _limit(self.temp_bytes, self.config.max_temp_bytes, "max_temp_bytes")
        self.metrics["temp_peak_bytes"] = max(self.metrics["temp_peak_bytes"], self.temp_bytes)
        path = self.directory / f"pairs-{self.next_file:08d}.bin"
        self.next_file += 1
        return path

    def _write(self, records: Iterable[Pair], count: int) -> _Run:
        path = self._reserve(count)
        with path.open("wb") as stream:
            for index, record in enumerate(records):
                stream.write(PAIR.pack(*record))
                if index % 4096 == 0:
                    self.check_budget()
        self.metrics["temp_written_bytes"] += count * PAIR.size
        return _Run(path, count)

    def _read(self, run: _Run) -> Generator[Pair, None, None]:
        with run.path.open("rb") as stream:
            for index in range(run.count):
                if index % 4096 == 0:
                    self.check_budget()
                raw = stream.read(PAIR.size)
                if len(raw) != PAIR.size:
                    raise InputValidationError("Неполная временная порция графа")
                self.metrics["temp_read_bytes"] += PAIR.size
                yield cast(Pair, PAIR.unpack(raw))

    def _merge(self, runs: list[_Run]) -> _Run:
        self.metrics["merge_peak_readers"] = max(self.metrics["merge_peak_readers"], len(runs))
        readers = [self._read(run) for run in runs]
        try:
            merged = self._write(heapq.merge(*readers), sum(run.count for run in runs))
        finally:
            for reader in readers:
                reader.close()
        for run in runs:
            run.path.unlink()
            self.temp_bytes -= run.count * PAIR.size
        self.metrics["merge_count"] += 1
        return merged

    def add(self, record: Pair) -> None:
        self.metrics["candidate_pairs"] = counter_add(self.metrics["candidate_pairs"])
        _limit(self.metrics["candidate_pairs"], self.config.max_pairs, "max_pairs")
        self.buffer.append(record)
        self.metrics["buffer_peak_records"] = max(
            self.metrics["buffer_peak_records"], len(self.buffer)
        )
        if len(self.buffer) == self.config.buffer_records:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort()
        run = self._write(self.buffer, len(self.buffer))
        self.metrics["sorted_chunks"] += 1
        self.buffer.clear()
        level = 0
        while True:
            if level == len(self.levels):
                self.levels.append([])
            self.levels[level].append(run)
            if len(self.levels[level]) < self.config.merge_fan_in:
                return
            run = self._merge(self.levels[level])
            self.levels[level].clear()
            level += 1

    def finish(self) -> Generator[Pair, None, None]:
        self._flush()
        runs = [run for level in self.levels for run in level]
        while len(runs) > 1:
            next_runs = []
            for start in range(0, len(runs), self.config.merge_fan_in):
                portion = runs[start : start + self.config.merge_fan_in]
                next_runs.append(portion[0] if len(portion) == 1 else self._merge(portion))
            runs = next_runs
        if runs:
            yield from self._read(runs[0])


def _product_groups(
    session: Session,
    sku_indices: dict[str, int],
    config: HistoricalConfig,
    metrics: dict[str, Any],
    check_budget: Callable[[], None],
) -> Iterator[ProductGroup]:
    for time_ms, blocks in groupby(
        sorted(session, key=lambda block: block.event_time_ms),
        key=lambda block: block.event_time_ms,
    ):
        check_budget()
        if time_ms >= config.snapshot_time_ms:
            break
        details: set[int] = set()
        targets: set[tuple[int, int]] = set()
        count = 0
        for block in blocks:
            for event in block.events:
                if event.event_type not in PRODUCT_TYPES:
                    continue
                count = counter_add(count)
                _limit(count, config.max_group_size, "max_group_size")
                if event.sku not in sku_indices:
                    raise InputValidationError("Исторический SKU отсутствует в общем каталоге")
                item = sku_indices[event.sku]
                if event.event_type == EventType.DETAIL:
                    details.add(item)
                if event.event_type in TARGET_RELATION:
                    targets.add((TARGET_RELATION[event.event_type], item))
        if count:
            metrics["max_group_size"] = max(metrics["max_group_size"], count)
            yield time_ms, tuple(sorted(details)), tuple(sorted(targets))


def _generate_pairs(
    history: HistoricalInput,
    sku_indices: dict[str, int],
    config: HistoricalConfig,
    sorter: _PairSort,
) -> None:
    session_count = 0
    for session_index, session in enumerate(history.iter_sessions()):
        session_count = counter_add(session_count)
        previous: deque[tuple[int, tuple[int, ...]]] = deque(maxlen=config.graph_max_group_distance)
        for time_ms, details, targets in _product_groups(
            session, sku_indices, config, sorter.metrics, sorter.check_budget
        ):
            for earlier_time, earlier_details in previous:
                delay = checked_subtract(time_ms, earlier_time)
                if delay > config.graph_window_ms:
                    continue
                for source in earlier_details:
                    for relation, target in targets:
                        if source != target:
                            sorter.add((relation, source, target, session_index, delay))
            previous.append((time_ms, details))
    if session_count != history.session_count:
        raise InputValidationError("Число прочитанных сессий графа не совпало с проверенным входом")
    sorter.metrics["sessions"] = session_count


def _edge_weights(
    records: Iterator[Pair],
    config: HistoricalConfig,
    metrics: dict[str, Any],
) -> Iterator[tuple[int, int, int, float, int]]:
    for (relation, source, target), edge in groupby(records, key=lambda item: item[:3]):
        support = 0

        def contributions() -> Iterator[float]:
            nonlocal support
            for _, session_records in groupby(edge, key=lambda item: item[3]):
                delay = next(session_records)[4]  # Первый — минимум после внешней сортировки.
                weight = math.exp(-delay / config.graph_tau_ms)
                if weight == 0 or not math.isfinite(weight):
                    raise NumericError("Положительное затухание ребра не представимо в FP64")
                support = counter_add(support)
                yield weight

        # compensated_sum материализует вход; fsum здесь сохраняет потоковую память.
        try:
            weight = math.fsum(contributions())
        except OverflowError as exc:
            raise NumericError("Вес ребра не представим в FP64") from exc
        metrics["session_edge_pairs"] = counter_add(metrics["session_edge_pairs"], support)
        relation_metrics = metrics["relations"][RELATIONS[relation]]
        relation_metrics["edges_before_support"] += 1
        if support >= config.graph_min_support:
            relation_metrics["edges_after_support"] += 1
            yield relation, source, target, weight, support


def _relations(
    records: Iterator[Pair],
    catalog: HistoricalCatalog,
    config: HistoricalConfig,
    metrics: dict[str, Any],
    check_budget: Callable[[], None],
) -> tuple[GraphRelation, ...]:
    size = len(catalog.skus)
    all_graphs = []
    grouped = groupby(_edge_weights(records, config, metrics), key=lambda item: item[0])
    current = next(grouped, None)
    for relation_index, name in enumerate(RELATIONS):
        indptr = np.zeros(size + 1, dtype=np.int64)
        row_sums = np.zeros(size, dtype=np.float64)
        rho = np.zeros(size, dtype=np.float64)
        indices, supports = array("q"), array("q")
        weights, probabilities = array("d"), array("d")
        if current is not None and current[0] == relation_index:
            for source, row in groupby(current[1], key=lambda item: item[1]):
                check_budget()
                kept: list[tuple[float, int, int]] = []
                for _, _, target, weight, support in row:
                    entry = (weight, -target, support)
                    if len(kept) < config.graph_max_neighbors:
                        heapq.heappush(kept, entry)
                    elif entry > kept[0]:
                        heapq.heapreplace(kept, entry)
                kept.sort(key=lambda entry: (-entry[0], -entry[1]))
                total = compensated_sum(weight for weight, _, _ in kept)
                row_sums[source] = total
                rho[source] = total / (total + config.graph_regularization)
                if rho[source] == 0 or not math.isfinite(rho[source]):
                    raise NumericError("Регуляризация строки не представима в FP64")
                metrics["result_bytes"] += len(kept) * 32
                _limit(metrics["result_bytes"], config.max_result_bytes, "max_result_bytes")
                indptr[source + 1] = len(kept)
                for weight, negative_target, support in kept:
                    probability = weight / total
                    if probability == 0 or not math.isfinite(probability):
                        raise NumericError("Нормировка ребра не представима в FP64")
                    indices.append(-negative_target)
                    supports.append(support)
                    weights.append(weight)
                    probabilities.append(probability)
            current = next(grouped, None)
        np.cumsum(indptr, out=indptr)
        graph = GraphRelation(
            name,
            indptr,
            np.frombuffer(indices, dtype=np.int64),
            np.frombuffer(weights, dtype=np.float64),
            np.frombuffer(probabilities, dtype=np.float64),
            np.frombuffer(supports, dtype=np.int64),
            row_sums,
            rho,
        )
        for values in (
            graph.indptr,
            graph.indices,
            graph.weights,
            graph.probabilities,
            graph.support,
            graph.row_sums,
            graph.rho,
        ):
            values.setflags(write=False)
        metrics["relations"][name]["edges_after_pruning"] = len(indices)
        all_graphs.append(graph)
    return tuple(all_graphs)


def build_graph(
    history: HistoricalInput,
    catalog: HistoricalCatalog,
    config: HistoricalConfig,
    *,
    work_dir: Path,
    check_budget: Callable[[], None] | None = None,
) -> GraphBuildResult:
    """Построить три отношения только из проверенного исторического входа.

    check_budget контролирует внешние пределы времени и RSS; пределы пар,
    временного диска, группы, буфера и готовых массивов соблюдаются здесь.
    Временный подкаталог удаляется также при численном/ресурсном отказе.
    """
    budget = check_budget if check_budget is not None else lambda: None
    budget()
    if config.snapshot_time_ms != history.snapshot_time_ms:
        raise InputValidationError("Граница снимка не совпала с проверенным историческим входом")
    base_bytes = len(RELATIONS) * (24 * len(catalog.skus) + 8)
    _limit(base_bytes, config.max_result_bytes, "max_result_bytes")
    metrics: dict[str, Any] = {
        "candidate_pairs": 0,
        "session_edge_pairs": 0,
        "max_group_size": 0,
        "temp_peak_bytes": 0,
        "temp_written_bytes": 0,
        "temp_read_bytes": 0,
        "buffer_peak_records": 0,
        "merge_peak_readers": 0,
        "sorted_chunks": 0,
        "merge_count": 0,
        "result_bytes": base_bytes,
        "relations": {
            name: {"edges_before_support": 0, "edges_after_support": 0, "edges_after_pruning": 0}
            for name in RELATIONS
        },
    }
    sku_indices = {sku: index for index, sku in enumerate(catalog.skus)}
    work_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="graph-", dir=work_dir) as directory:
        sorter = _PairSort(Path(directory), config, metrics, budget)
        _generate_pairs(history, sku_indices, config, sorter)
        records = sorter.finish()
        try:
            relations = _relations(records, catalog, config, metrics, budget)
        finally:
            records.close()
    budget()
    return GraphBuildResult(relations, metrics)
