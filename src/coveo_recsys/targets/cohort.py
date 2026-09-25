"""Пары первого просмотра A_G; без сглаживания и сборки исторического пакета."""

from collections.abc import Sequence
from dataclasses import dataclass

from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.preparation import Coverage, Period, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, Record
from coveo_recsys.contracts.training import SessionRecord
from coveo_recsys.data.periods import admit
from coveo_recsys.numerics import checked_add

from .maturity import coverage_reason, observe_session


@dataclass(frozen=True, slots=True)
class CohortRecord(Record):
    source_namespace: str
    session_id: str
    sku: str
    anchor_ms: int
    target_end_ms: int
    mature: bool
    purchased: int | None
    reason: str | None

    def __post_init__(self) -> None:
        Record.__post_init__(self)
        id_bytes(self.source_namespace)
        if self.target_end_ms <= self.anchor_ms:
            raise InputValidationError("Окно пары должно следовать за первым просмотром")
        if self.mature:
            if self.purchased not in (0, 1):
                raise InputValidationError("Зрелая пара требует бинарный исход")
        elif self.purchased is not None or not self.reason:
            raise InputValidationError("Исключённая пара требует неизвестный исход и причину")


def historical_cohort(
    blocks: Sequence[CanonicalBlock],
    session: SessionRecord,
    coverage: Coverage,
    periods: Sequence[Period],
    config: TaskConfig,
) -> tuple[CohortRecord, ...]:
    """Одна запись на (пространство, сессия, первый detail товара).

    V_i и B_i впоследствии используют только mature=True. Исключённые пары
    сохраняются для отчёта с purchased=None. Время окна — anchor+H, не tau
    первого запроса. Полный журнал читается и после остановки рекомендаций.
    Счётчик всех сессий с покупкой товара этой когортой не заменяется.
    """
    observation = observe_session(blocks, session.admission_ms, coverage, config)
    if (session.source_namespace, session.session_id, session.last_event_ms) != (
        blocks[0].source_namespace,
        blocks[0].session_id,
        blocks[-1].event_time_ms,
    ):
        raise InputValidationError("Запись сессии не соответствует истории")
    admission = admit(
        session.admission_ms,
        periods,
        config,
        coverage_complete=observation.coverage_complete,
        session_complete=observation.session_complete,
        close_ms=observation.close_ms,
    )
    if (
        session.period,
        session.mature,
        session.coverage_complete,
        session.close_ms,
        session.target_end_ms,
    ) != (
        admission.period,
        observation.mature,
        observation.coverage_complete,
        observation.close_ms,
        observation.target_end_ms,
    ):
        raise InputValidationError("Запись зрелости несовместима с покрытием или планом")
    session_reason = admission.reason
    if session_reason == "coverage_incomplete":
        session_reason = observation.reason
    if session.exclusion_reason != session_reason:
        raise InputValidationError("Причина исключения несовместима с планом допуска")
    if admission.period != "A_G":
        session_reason = "outside_A_G"

    anchors: dict[str, int] = {}
    purchases: dict[str, list[int]] = {}
    for block in blocks:
        for event in block.events:
            if event.sku is None:
                continue
            if event.event_type == EventType.DETAIL:
                anchors.setdefault(event.sku, block.event_time_ms)
            elif event.event_type == EventType.PURCHASE:
                purchases.setdefault(event.sku, []).append(block.event_time_ms)

    records = []
    for sku in sorted(anchors, key=id_bytes):
        anchor = anchors[sku]
        end = checked_add(anchor, config.horizon_ms)
        maturity_boundary = checked_add(end, config.gap_ms)
        reason = session_reason
        if reason is None and maturity_boundary >= periods[0].end_ms:
            reason = "anchor_maturity_boundary"
        if reason is None:
            reason = coverage_reason(coverage, blocks[0].event_time_ms, maturity_boundary)
        mature = reason is None
        purchased = (
            int(any(anchor < time <= end for time in purchases.get(sku, ()))) if mature else None
        )
        if mature and purchased == 0:
            reason = "no_purchase_in_anchor_window"
        records.append(
            CohortRecord(
                session.source_namespace,
                session.session_id,
                sku,
                anchor,
                end,
                mature,
                purchased,
                reason,
            )
        )
    return tuple(records)
