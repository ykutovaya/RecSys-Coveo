from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.historical import HistoricalCatalog, HistoricalConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent
from coveo_recsys.historical.catalog import build_catalog


@pytest.mark.parametrize("skus", [("b", "a"), ("a", "a"), ("",)])
def test_catalog_requires_unique_utf8_order(skus):
    with pytest.raises(InputValidationError):
        HistoricalCatalog(skus)


@pytest.mark.parametrize("field", ["graph_tau_ms", "graph_min_support", "max_temp_bytes"])
def test_historical_limits_must_be_positive(field):
    with pytest.raises(InputValidationError):
        replace(HistoricalConfig(snapshot_time_ms=100), **{field: 0})


def test_empty_catalog_is_valid():
    assert build_catalog(iter(())).skus == ()


def test_catalog_uses_all_product_actions_not_only_details():
    events = tuple(
        RawEvent("n", "s", str(i), 1, 1, kind, sku, None, "fixture")
        for i, (kind, sku) in enumerate(
            [(EventType.ADD, "я"), (EventType.REMOVE, "b"), (EventType.PURCHASE, "a")]
        )
    )
    block = CanonicalBlock("block", "s", 1, 1, events, tuple((e.event_id, 1) for e in events), ())
    assert build_catalog(iter([(block,)])).skus == ("a", "b", "я")
