"""Будущие покупки отдельны от префикса; неизвестный исход не становится нулём."""

from collections.abc import Sequence

from coveo_recsys.contracts import InputValidationError, OutcomeStatus, id_bytes
from coveo_recsys.contracts.preparation import Coverage, Period, PreparedRequest, TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType
from coveo_recsys.contracts.training import SessionRecord, TargetRecord
from coveo_recsys.data.periods import admit
from coveo_recsys.numerics import checked_add

from .maturity import observe_session


def _validate_requests(
    blocks: Sequence[CanonicalBlock], requests: Sequence[PreparedRequest], config: TaskConfig
) -> None:
    """Проверить совместимость принятого расписания Э1-04, не строить его повторно."""
    identity = (blocks[0].source_namespace, blocks[0].session_id)
    by_id = {block.block_id: block for block in blocks}
    if len(requests) > config.max_requests:
        raise InputValidationError("Число запросов превышает Q_max")
    request_ids: set[str] = set()
    previous: int | None = None
    for ordinal, request in enumerate(requests, 1):
        if (request.source_namespace, request.session_id) != identity:
            raise InputValidationError("Запрос другой сессии или пространства источника")
        if request.ordinal != ordinal or request.request_id in request_ids:
            raise InputValidationError("Повтор ID или неполная нумерация запросов")
        if request.a_q != request.t_q:
            raise InputValidationError("Поддерживается только историческое расписание a=t")
        if request.prefix.event_semantics != config.event_semantics:
            raise InputValidationError("Несовместимая семантика префикса")
        if previous is not None and request.t_q <= previous:
            raise InputValidationError("Запросы должны иметь строго возрастающие времена")
        if request.t_q > checked_add(requests[0].t_q, config.horizon_ms):
            raise InputValidationError("Запрос вне горизонта расписания")
        block = by_id.get(request.trigger_block_id)
        trigger = next(
            (item for item in request.prefix.blocks if item.block_id == request.trigger_block_id),
            None,
        )
        if block is None or trigger != block or block.event_time_ms != request.t_q:
            raise InputValidationError("Запускающий блок не соответствует полной истории")
        request_ids.add(request.request_id)
        previous = request.t_q


def prepare_targets(
    blocks: Sequence[CanonicalBlock],
    requests: Sequence[PreparedRequest],
    coverage: Coverage,
    periods: Sequence[Period],
    config: TaskConfig,
) -> tuple[SessionRecord, tuple[TargetRecord, ...]]:
    """Обработать одну непустую сессию и её полное принятое расписание.

    Нет подвыборки, каталога кандидатов или построения префиксов. Даже для
    исключённой сессии сохраняются исходное число запросов и неизвестные цели.
    SessionRecord.mature описывает окно, exclusion_reason — отдельный допуск.
    TargetRecord.horizon_ms — длительность H; абсолютная tau хранится в сессии.
    """
    first_request = requests[0].t_q if requests else None
    observation = observe_session(blocks, first_request, coverage, config)
    _validate_requests(blocks, requests, config)
    admission = admit(
        first_request,
        periods,
        config,
        coverage_complete=observation.coverage_complete,
        session_complete=observation.session_complete,
        close_ms=observation.close_ms,
    )
    reason = admission.reason
    if reason == "coverage_incomplete":
        reason = observation.reason
    session = SessionRecord(
        blocks[0].session_id,
        first_request,
        observation.close_ms,
        observation.coverage_complete,
        admission.period,
        observation.mature,
        reason,
        source_namespace=blocks[0].source_namespace,
        last_event_ms=blocks[-1].event_time_ms,
        target_end_ms=observation.target_end_ms,
        request_count=len(requests),
    )
    if not admission.admitted:
        return session, tuple(
            TargetRecord(
                request.request_id,
                None,
                observation.mature,
                OutcomeStatus.UNKNOWN,
                config.horizon_ms,
                None,
                reason or "not_admitted",
            )
            for request in requests
        )

    tau = observation.target_end_ms
    if tau is None or first_request is None:
        raise InputValidationError("Допущенный исход должен иметь подтверждённое окно")
    purchases = tuple(
        (block.event_time_ms, event.sku)
        for block in blocks
        if first_request < block.event_time_ms <= tau
        for event in block.events
        if event.event_type == EventType.PURCHASE and event.sku is not None
    )
    b_s = int(bool(purchases))
    targets = tuple(
        TargetRecord(
            request.request_id,
            tuple(sorted({sku for time, sku in purchases if request.t_q < time}, key=id_bytes)),
            True,
            OutcomeStatus.KNOWN,
            config.horizon_ms,
            b_s,
            None if any(request.t_q < time for time, _ in purchases) else "no_future_purchase",
        )
        for request in requests
    )
    return session, targets
