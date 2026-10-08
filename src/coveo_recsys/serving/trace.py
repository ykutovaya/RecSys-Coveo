"""Один локальный журнал запросов; воспроизведение только записанного исхода."""

import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, strict_json_loads
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestContext,
    RequestResult,
)
from coveo_recsys.contracts.retrieval import SOURCES
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.explanations import TEMPLATE_SHA256, TEMPLATE_VERSION

from .runtime import STAGES

_TRACE_FIELDS = {
    "format_version",
    "phase",
    "source_namespace",
    "session_id",
    "idempotency_key",
    "trigger_block_id",
    "canonical_block_id",
    "t_q",
    "a_q",
    "package_id",
    "published_at_ms",
    "template_sha256",
    "context",
    "source_runtime",
    "stages",
    "timeout_ms",
    "deadlines_ms",
    "result",
}
_MAX_RECORD_BYTES = 8_388_608


def validate_trace(
    trace: dict[str, Any],
    *,
    checked_result: RequestResult | None = None,
    checked_context: RequestContext | None = None,
) -> RequestResult:
    """Проверить связи и привязку к тем же объектам либо к их точным байтам."""
    if (
        not isinstance(trace, dict)
        or set(trace) != _TRACE_FIELDS
        or trace["format_version"] != "execution_trace_v1"
        or trace["phase"] not in {"selected", "terminal"}
    ):
        raise InputValidationError("Неполная или неизвестная трасса исполнения")
    for key in (
        "source_namespace",
        "session_id",
        "idempotency_key",
        "trigger_block_id",
        "package_id",
    ):
        id_bytes(trace[key])
    if trace["canonical_block_id"] is not None:
        id_bytes(trace["canonical_block_id"])
    tq, aq = require_int64(trace["t_q"]), require_int64(trace["a_q"])
    if aq < tq:
        raise InputValidationError("Некорректное время или результат трассы")
    if type(checked_result) is RequestResult and trace["result"] is checked_result:
        result = checked_result
    elif isinstance(trace["result"], str):
        payload = trace["result"].encode("utf-8")
        if checked_result is None:
            result = from_json(payload, RequestResult)
        elif type(checked_result) is not RequestResult or to_json(checked_result) != payload:
            raise InputValidationError("Проверенный результат не соответствует байтам трассы")
        else:
            result = checked_result
    else:
        raise InputValidationError("Некорректный результат трассы")
    if result.package_id != trace["package_id"] or not result.trace_ref:
        raise InputValidationError(
            "Результат относится к другому пакету или не имеет ссылки на трассу"
        )
    if trace["phase"] == "selected" and result.delivery != DeliveryStatus.NOT_ATTEMPTED:
        raise InputValidationError("До возврата ответа доставка не подтверждена")
    published = trace["published_at_ms"]
    if published is None:
        if (
            trace["template_sha256"] is not None
            or result.candidates.branch != Branch.NONE
            or result.candidates.skus
            or result.delivery == DeliveryStatus.DELIVERED
        ):
            raise InputValidationError("Отсутствующая версия не даёт кандидатов или доставки")
    elif require_int64(published) >= tq or trace["template_sha256"] != TEMPLATE_SHA256:
        raise InputValidationError("Публикация или шаблоны не соответствуют запросу")
    context = trace["context"]
    if context is None:
        if (
            checked_context is not None
            or result.candidates.skus
            or result.delivery == DeliveryStatus.DELIVERED
        ):
            raise InputValidationError("Нельзя проверить кандидатов без записанного среза")
    else:
        if type(checked_context) is RequestContext and context is checked_context:
            prepared = checked_context
        elif isinstance(context, str):
            payload = context.encode("utf-8")
            if checked_context is None:
                prepared = from_json(payload, RequestContext)
            elif type(checked_context) is not RequestContext or to_json(checked_context) != payload:
                raise InputValidationError("Проверенный срез не соответствует байтам трассы")
            else:
                prepared = checked_context
        else:
            raise InputValidationError("Некорректный срез трассы")
        if (
            prepared.prefix.source_namespace != trace["source_namespace"]
            or prepared.session_id != trace["session_id"]
            or prepared.trigger_block_id != trace["canonical_block_id"]
            or prepared.request_id != result.request_id
            or prepared.t_q != tq
            or prepared.a_q != aq
            or prepared.ordinal != 1
            or set(result.candidates.skus) & set(prepared.prefix.purchased_skus)
        ):
            raise InputValidationError("Результат расходится с зафиксированным срезом")
    runtime, stages = trace["source_runtime"], trace["stages"]
    deadlines = trace["deadlines_ms"]
    if (
        type(trace["timeout_ms"]) is not int
        or trace["timeout_ms"] <= 0
        or not isinstance(deadlines, dict)
        or not set(deadlines) <= set(STAGES)
        or any(
            type(value) not in {int, float} or not 0 <= value <= trace["timeout_ms"]
            for value in deadlines.values()
        )
    ):
        raise InputValidationError("Не заданы допустимые сроки этапов запроса")
    if (
        not isinstance(runtime, dict)
        or not set(runtime) <= set(SOURCES)
        or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in runtime.values()
        )
    ):
        raise InputValidationError("Не заданы состояния источников")
    if not isinstance(stages, dict) or not stages:
        raise InputValidationError("Нет измерений этапов запроса")
    for name, stage in stages.items():
        if (
            not isinstance(name, str)
            or name not in STAGES
            or name not in deadlines
            or not isinstance(stage, dict)
            or set(stage) != {"elapsed_ms", "status"}
        ):
            raise InputValidationError("Неполное измерение этапа")
        elapsed = stage["elapsed_ms"]
        if (
            type(elapsed) not in {int, float}
            or not math.isfinite(elapsed)
            or elapsed < 0
            or not isinstance(stage["status"], str)
            or not stage["status"]
        ):
            raise InputValidationError("Некорректное измерение этапа")
    if result.candidates.branch == Branch.PRIMARY and any(
        stages.get(name, {}).get("status") != "ok"
        for name in ("representation", "sources", "ranking")
    ):
        raise InputValidationError("Основная ветвь не подтверждена этапами вычислений")
    if result.delivery == DeliveryStatus.DELIVERED and (
        stages.get("return", {}).get("status") != "ok"
        or result.elapsed_ms > trace["timeout_ms"]
        or result.elapsed_ms > deadlines.get("return", 0)
    ):
        raise InputValidationError("Не подтверждён своевременный возврат результата")
    origins = dict(result.candidates.origins)
    if len(origins) != len(result.candidates.origins) or set(origins) != set(
        result.candidates.skus
    ):
        raise InputValidationError("Неполные происхождения кандидатов")
    expected = {(sku, source) for sku, names in origins.items() for source in names}
    evidence = {(item.sku, item.source) for item in result.candidates.evidence}
    if (
        expected != evidence
        or len(evidence) != len(result.candidates.evidence)
        or any(
            not names or len(set(names)) != len(names) or not set(names) <= {*SOURCES, "fill"}
            for names in origins.values()
        )
    ):
        raise InputValidationError("Неполные опоры выбранных кандидатов")
    if result.candidates.branch == Branch.FALLBACK and any(
        not set(names) <= {"recent", "popularity"} for names in origins.values()
    ):
        raise InputValidationError("Резерв содержит источники основной ветви")
    if tuple(item.sku for item in result.explanations) != result.delivered_skus:
        raise InputValidationError("Объяснения не соответствуют доставленному списку")
    if any(
        item.branch != result.candidates.branch
        or item.package_id != result.package_id
        or set(item.sources) != set(origins[item.sku])
        or item.template_id != TEMPLATE_VERSION + ":" + "+".join(item.sources)
        for item in result.explanations
    ):
        raise InputValidationError("Объяснение относится к другой ветви или версии")
    return result


def replay_trace(trace: dict[str, Any]) -> RequestResult:
    """Вернуть сохранённый исход, не подменяя отказ исправным повтором вычислений."""
    if trace.get("phase") != "terminal":
        raise InputValidationError("В трассе нет окончательного исхода")
    return validate_trace(trace)


@dataclass(frozen=True)
class Reservation:
    key: str
    trigger_block_id: str
    t_q: int
    a_q: int
    result: RequestResult | None = None
    state: str = "pending"


def _selection_key(trace: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        trace[key]
        for key in (
            "package_id",
            "published_at_ms",
            "template_sha256",
            "canonical_block_id",
            "context",
        )
    ) + (tuple(sorted(trace["source_runtime"].items())),)


def _same_candidates(left: CandidateResult, right: CandidateResult) -> bool:
    # Записи неизменяемы. Разные объекты, включая прочитанные из журнала,
    # сравниваются по точным байтам: обычное равенство теряет знак нуля.
    return left is right or to_json(left) == to_json(right)


class Journal:
    """Журнал одного процесса сервиса; вызовы потоков сериализованы одним замком."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._ready = True
        self._reservations: dict[tuple[str, str], Reservation] = {}
        self._selected: dict[tuple[str, str], tuple[tuple[Any, ...], CandidateResult]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("rb") as stream:
                while line := stream.readline(_MAX_RECORD_BYTES + 1):
                    if len(line) > _MAX_RECORD_BYTES or not line.endswith(b"\n"):
                        raise InputValidationError(
                            "Повреждённая или слишком большая запись журнала"
                        )
                    try:
                        entry = strict_json_loads(line)
                    except (ValueError, UnicodeError, RecursionError) as exc:
                        raise InputValidationError("Повреждённая запись журнала") from exc
                    if not isinstance(entry, dict):
                        raise InputValidationError("Некорректная запись журнала")
                    if entry.get("type") == "reserve" and set(entry) == {
                        "type",
                        "source_namespace",
                        "session_id",
                        "key",
                        "trigger_block_id",
                        "t_q",
                        "a_q",
                    }:
                        identity, reservation = self._reservation(
                            entry["source_namespace"],
                            entry["session_id"],
                            entry["key"],
                            entry["trigger_block_id"],
                            entry["t_q"],
                            entry["a_q"],
                        )
                        if identity in self._reservations:
                            raise InputValidationError("Повторное назначение первого запроса")
                        self._reservations[identity] = reservation
                    elif entry.get("type") == "trace" and set(entry) == {"type", "trace"}:
                        body = entry["trace"]
                        result = validate_trace(body)
                        identity = self._check_trace(body, result)
                        self._finish_record(body, result, identity)
                    else:
                        raise InputValidationError("Неизвестная запись журнала")
        self._reservations = {
            key: replace(value, state="cancelled") if value.state == "pending" else value
            for key, value in self._reservations.items()
        }
        self._selected.clear()

    @property
    def ready(self) -> bool:
        return self._ready

    def check_available(self) -> None:
        """Проверка доступа без записи; последующая запись всё равно может завершиться отказом."""
        with self._lock:
            if not self._ready:
                raise OSError("Журнал недоступен для записи")
            if self.path.exists():
                with self.path.open("r+b"):
                    pass
            elif not self.path.parent.is_dir() or not os.access(self.path.parent, os.W_OK):
                raise OSError("Каталог журнала недоступен")

    @staticmethod
    def _reservation(
        namespace: str, session_id: str, key: str, trigger_block_id: str, t_q: int, a_q: int
    ) -> tuple[tuple[str, str], Reservation]:
        for value in (namespace, session_id, key, trigger_block_id):
            id_bytes(value)
        if require_int64(a_q) < require_int64(t_q):
            raise InputValidationError("Время принятия запроса предшествует событию")
        return (namespace, session_id), Reservation(key, trigger_block_id, t_q, a_q)

    def _append(
        self, entry: dict[str, Any], check_deadline: Callable[[], None] | None = None
    ) -> None:
        if not self._ready:
            raise OSError("Журнал недоступен для записи")
        payload = canonical_json_bytes(entry)
        if len(payload) > _MAX_RECORD_BYTES:
            raise InputValidationError("Превышен размер записи журнала")
        try:
            if check_deadline is not None:
                check_deadline()
            with self.path.open("a+b", buffering=0) as stream:
                start = stream.tell()
                try:
                    if stream.write(payload) != len(payload):
                        raise OSError("Неполная запись журнала")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if check_deadline is not None:
                        check_deadline()
                except (OSError, ValueError):
                    stream.truncate(start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    raise
        except OSError:
            self._ready = False
            raise

    def reserve(
        self, namespace: str, session_id: str, key: str, trigger_block_id: str, t_q: int, a_q: int
    ) -> bool:
        with self._lock:
            if (namespace, session_id) in self._reservations:
                return False
            identity, reservation = self._reservation(
                namespace, session_id, key, trigger_block_id, t_q, a_q
            )
            self._append(
                {
                    "type": "reserve",
                    "source_namespace": namespace,
                    "session_id": session_id,
                    "key": key,
                    "trigger_block_id": trigger_block_id,
                    "t_q": t_q,
                    "a_q": a_q,
                }
            )
            self._reservations[identity] = reservation
            return True

    def lookup(self, namespace: str, session_id: str) -> Reservation | None:
        with self._lock:
            return self._reservations.get((namespace, session_id))

    def _check_trace(self, trace: dict[str, Any], result: RequestResult) -> tuple[str, str]:
        identity = trace["source_namespace"], trace["session_id"]
        reservation = self._reservations.get(identity)
        if (
            reservation is None
            or reservation.state != "pending"
            or (reservation.key, reservation.trigger_block_id, reservation.t_q, reservation.a_q)
            != (trace["idempotency_key"], trace["trigger_block_id"], trace["t_q"], trace["a_q"])
        ):
            raise InputValidationError("Трасса не соответствует активному назначению запроса")
        selected = self._selected.get(identity)
        if trace["phase"] == "selected":
            if selected is not None:
                raise InputValidationError("Ветвь запроса уже записана")
        elif selected is not None:
            if selected[0] != _selection_key(trace) or not _same_candidates(
                selected[1], result.candidates
            ):
                raise InputValidationError(
                    "Окончательный исход изменяет выбранную ветвь или версии"
                )
        elif result.delivery == DeliveryStatus.DELIVERED:
            raise InputValidationError("Нет обязательной записи выбранной ветви до доставки")
        return identity

    def _finish_record(
        self, trace: dict[str, Any], result: RequestResult, identity: tuple[str, str]
    ) -> None:
        if trace["phase"] == "selected":
            self._selected[identity] = _selection_key(trace), result.candidates
        else:
            self._reservations[identity] = replace(
                self._reservations[identity], result=result, state="finished"
            )
            self._selected.pop(identity, None)

    def record(
        self,
        trace: dict[str, Any],
        *,
        checked_result: RequestResult | None = None,
        checked_context: RequestContext | None = None,
        check_deadline: Callable[[], None] | None = None,
    ) -> None:
        result = validate_trace(
            trace, checked_result=checked_result, checked_context=checked_context
        )
        # Координатор передаёт сами записи: сериализуем один раз перед записью.
        # Прочитанные трассы сохраняют строгую проверку исходных байтов.
        trace = dict(trace)
        for name in ("result", "context"):
            if trace[name] is not None and not isinstance(trace[name], str):
                trace[name] = to_json(trace[name]).decode("utf-8")
        with self._lock:
            identity = self._check_trace(trace, result)
            self._append({"type": "trace", "trace": trace}, check_deadline)
            self._finish_record(trace, result, identity)

    def fail(self, namespace: str, session_id: str, result: RequestResult) -> None:
        if result.delivery == DeliveryStatus.DELIVERED or result.explanations:
            raise InputValidationError("Отказ записи не подтверждает доставку")
        with self._lock:
            identity = namespace, session_id
            reservation = self._reservations.get(identity)
            if reservation is None or reservation.state == "finished":
                raise InputValidationError("Нет незавершённого запроса для фиксации отказа")
            selected = self._selected.get(identity)
            if selected is not None and not _same_candidates(selected[1], result.candidates):
                raise InputValidationError("Недоставка не меняет выбранных кандидатов")
            self._reservations[identity] = replace(reservation, result=result, state="finished")
            self._selected.pop(identity, None)
