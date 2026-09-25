"""Зарегистрированные будущие покупки отдельно от входного префикса."""

from collections.abc import Sequence

from coveo_recsys.contracts import InputValidationError, OutcomeStatus, id_bytes
from coveo_recsys.contracts.preparation import Coverage, Period, PreparedRequest, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.periods import admit

from .maturity import observe_session


def _validate_requests(
    blocks: Sequence[CanonicalBlock], requests: Sequence[PreparedRequest], config: TaskConfig
) -> None:
    """Проверить первый допустимый запрос без повторного построения состояния."""
    first_block = None
    for block in blocks:
        actions = {event.event_type for event in block.events}
        if EventType.PURCHASE in actions:
            break
        if EventType.DETAIL in actions:
            first_block = block
            break
    if len(requests) != int(first_block is not None):
        raise InputValidationError("Требуется единственный первый допустимый запрос")
    if first_block is None:
        return
    request = requests[0]
    if (request.source_namespace, request.session_id) != (
        first_block.source_namespace,
        first_block.session_id,
    ):
        raise InputValidationError("Запрос другой сессии или источника")
    if request.ordinal != 1 or request.t_q != first_block.event_time_ms:
        raise InputValidationError("Запрос не соответствует первому допустимому detail")
    if request.a_q != request.t_q:
        raise InputValidationError("Поддерживается только историческое расписание a=t")
    if request.prefix.event_semantics != config.event_semantics:
        raise InputValidationError("Несовместимая семантика префикса")
    trigger = next(
        (item for item in request.prefix.blocks if item.block_id == request.trigger_block_id),
        None,
    )
    if request.trigger_block_id != first_block.block_id or trigger != first_block:
        raise InputValidationError("Запускающий блок не соответствует полной истории")


def prepare_targets(
    blocks: Sequence[CanonicalBlock],
    requests: Sequence[PreparedRequest],
    coverage: Coverage,
    periods: Sequence[Period],
    config: TaskConfig,
) -> tuple[SessionRecord, tuple[TargetRecord, ...]]:
    """Календарный допуск и все различные покупки в (t0, min(t0+H, t_last_obs)].

    Пустая цель остаётся неизвестной. Наличие покупки не подтверждает полноту
    телеметрии; метаданные покрытия не меняют допуск или зарегистрированную цель.
    """
    first_request = requests[0].t_q if requests else None
    observation = observe_session(blocks, first_request, coverage, config)
    _validate_requests(blocks, requests, config)
    admission = admit(first_request, periods, config, last_event_ms=observation.last_event_ms)
    session = SessionRecord(
        blocks[0].session_id,
        first_request,
        admission.period,
        admission.admitted,
        admission.reason,
        source_namespace=blocks[0].source_namespace,
        last_event_ms=observation.last_event_ms,
        target_end_ms=observation.target_end_ms,
        request_count=len(requests),
        telemetry_complete=None,
    )
    if not requests:
        return session, ()
    purchases: tuple[str, ...] = ()
    if admission.admitted:
        tau = observation.target_end_ms
        if tau is None or first_request is None:
            raise InputValidationError("Допущенная сессия должна иметь целевое окно")
        purchases = tuple(
            sorted(
                {
                    event.sku
                    for block in blocks
                    if first_request < block.event_time_ms <= tau
                    for event in block.events
                    if event.event_type == EventType.PURCHASE and event.sku is not None
                },
                key=id_bytes,
            )
        )
    target = TargetRecord(
        requests[0].request_id,
        purchases,
        OutcomeStatus.KNOWN if purchases else OutcomeStatus.UNKNOWN,
        config.horizon_ms,
        None if purchases else admission.reason or "no_registered_purchase",
    )
    return session, (target,)
