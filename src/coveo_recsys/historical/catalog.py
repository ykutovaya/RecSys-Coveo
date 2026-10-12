"""Каталог всех товарных событий принятой истории."""

from collections.abc import Iterable

from coveo_recsys.contracts import id_bytes
from coveo_recsys.contracts.historical import HistoricalCatalog, Session
from coveo_recsys.contracts.records import EventType


def build_catalog(sessions: Iterable[Session]) -> HistoricalCatalog:
    skus = {
        event.sku
        for session in sessions
        for block in session
        for event in block.events
        if event.sku is not None
        and event.event_type
        in {EventType.DETAIL, EventType.ADD, EventType.REMOVE, EventType.PURCHASE}
    }
    return HistoricalCatalog(tuple(sorted(skus, key=id_bytes)))
