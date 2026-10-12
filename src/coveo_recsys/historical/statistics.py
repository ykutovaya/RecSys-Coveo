"""Сессионные счётчики A_G и сглаженная популярность просмотров."""

import numpy as np

from coveo_recsys.contracts import InputValidationError, NumericError, require_int64
from coveo_recsys.contracts.historical import (
    HistoricalCatalog,
    HistoricalConfig,
    HistoricalInput,
    HistoricalStatistics,
)
from coveo_recsys.contracts.records import EventType
from coveo_recsys.numerics import compensated_sum, counter_add


def build_statistics(
    admitted_A_G: HistoricalInput,
    catalog: HistoricalCatalog,
    config: HistoricalConfig,
) -> HistoricalStatistics:
    """Считать каждую пару «сессия — SKU» один раз независимо от целей.

    Нормировка использует только detail. Покупки сохраняют собственный счётчик
    зарегистрированных сессий и не задают вероятности покупки.
    """
    require_int64(admitted_A_G.session_count, nonnegative=True)
    if config.snapshot_time_ms != admitted_A_G.snapshot_time_ms:
        raise InputValidationError("Исторический срез статистик не совпал с входом")
    positions = {sku: index for index, sku in enumerate(catalog.skus)}
    details = np.zeros(len(catalog.skus), dtype=np.int64)
    purchases = np.zeros(len(catalog.skus), dtype=np.int64)
    seen: set[str] = set()
    session_count = 0
    product_actions = {EventType.DETAIL, EventType.ADD, EventType.REMOVE, EventType.PURCHASE}
    for session in admitted_A_G.iter_sessions():
        if not session or session[0].session_id in seen:
            raise InputValidationError("Пустая либо повторная историческая сессия")
        session_id = session[0].session_id
        seen.add(session_id)
        session_count = counter_add(session_count)
        if session_count > admitted_A_G.session_count:
            raise InputValidationError("Число сессий превысило объявленный вход")
        detail_indices: set[int] = set()
        purchase_indices: set[int] = set()
        for block in session:
            if (block.source_namespace, block.session_id) != (
                admitted_A_G.source_namespace,
                session_id,
            ):
                raise InputValidationError("Смешаны сессии либо пространства ID")
            for event in block.events:
                if event.event_type not in product_actions:
                    continue
                if event.sku not in positions:
                    raise InputValidationError("Товар исторической сессии отсутствует в каталоге")
                assert event.sku is not None
                index = positions[event.sku]
                if event.event_type == EventType.DETAIL:
                    detail_indices.add(index)
                elif event.event_type == EventType.PURCHASE:
                    purchase_indices.add(index)
        for index in detail_indices:
            details[index] = counter_add(int(details[index]))
        for index in purchase_indices:
            purchases[index] = counter_add(int(purchases[index]))
    if session_count != admitted_A_G.session_count:
        raise InputValidationError("Прочитаны не все объявленные исторические сессии")
    popularity = np.empty(len(catalog.skus), dtype=np.float64)
    if catalog.skus:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                mass = details.astype(np.float64) + config.popularity_smoothing
                total = compensated_sum(mass)
                popularity = mass / total
        except FloatingPointError as exc:
            raise NumericError("Популярность не представима в FP64") from exc
        if not np.isfinite(popularity).all():
            raise NumericError("Неконечная популярность")
    return HistoricalStatistics(details, purchases, popularity, session_count)
