"""Статистики всех сессий A_G; отсутствие покупки не служит отрицательной меткой."""

from dataclasses import fields, replace

import numpy as np
import pytest

from coveo_recsys.contracts import INT64_MAX, InputValidationError, NumericError
from coveo_recsys.contracts.historical import HistoricalCatalog, HistoricalConfig
from coveo_recsys.historical.statistics import build_statistics
from tests.reference.e2_cases import history
from tests.reference.e2_statistics_oracle import statistics

CONFIG = HistoricalConfig(snapshot_time_ms=200_000_000)
ATOL, RTOL = 1e-10, 1e-8


def test_session_counts_and_smoothed_popularity_have_hand_computed_values():
    rows = [
        ("s1", 0, "detail", "a"),
        ("s1", 1, "detail", "a"),
        ("s1", 2, "detail", "b"),
        ("s1", 3, "purchase", "b"),
        ("s1", 4, "purchase", "b"),
        ("s2", 0, "detail", "a"),  # Сессия без зарегистрированной покупки.
        ("s3", 100_000_000, "purchase", "c"),  # Нет detail; покупка за пределами H.
        ("s4", 0, "add", "d"),
    ]
    actual = build_statistics(
        history(rows, snapshot=CONFIG.snapshot_time_ms),
        HistoricalCatalog(("a", "b", "c", "d")),
        CONFIG,
    )
    assert actual.detail_counts.tolist() == [2, 1, 0, 0]
    assert actual.purchase_counts.tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(
        actual.popularity, [3 / 7, 2 / 7, 1 / 7, 1 / 7], atol=ATOL, rtol=RTOL
    )
    assert actual.session_count == 4
    assert actual.detail_counts.dtype == actual.purchase_counts.dtype == np.dtype("int64")
    assert actual.popularity.dtype == np.dtype("float64")
    assert {item.name for item in fields(actual)} == {
        "detail_counts",
        "purchase_counts",
        "popularity",
        "session_count",
    }


@pytest.mark.parametrize(
    "actions", [[], ["detail"], ["detail", "detail"], ["purchase"], ["purchase", "purchase"]]
)
def test_zero_one_and_repeated_actions_match_independent_pairs(actions):
    rows = [("s", i, action, "a") for i, action in enumerate(actions)]
    catalog = HistoricalCatalog(("a", "b"))
    actual = build_statistics(history(rows, snapshot=CONFIG.snapshot_time_ms), catalog, CONFIG)
    expected = statistics(rows, catalog.skus)
    assert actual.detail_counts.tolist() == expected[0]
    assert actual.purchase_counts.tolist() == expected[1]
    np.testing.assert_allclose(actual.popularity, expected[2], atol=ATOL, rtol=RTOL)
    assert actual.session_count == expected[3]


def test_empty_catalog_preserves_all_sessions_and_returns_empty_arrays():
    source = history([("page-session", 0, "page", None)], snapshot=CONFIG.snapshot_time_ms)
    actual = build_statistics(source, HistoricalCatalog(()), CONFIG)
    assert actual.session_count == 1
    assert (
        actual.detail_counts.shape
        == actual.purchase_counts.shape
        == actual.popularity.shape
        == (0,)
    )


@pytest.mark.parametrize("action", ["detail", "purchase", "add", "remove"])
def test_product_event_outside_catalog_is_rejected(action):
    source = history([("s", 0, action, "outside")], snapshot=CONFIG.snapshot_time_ms)
    with pytest.raises(InputValidationError):
        build_statistics(source, HistoricalCatalog(()), CONFIG)


@pytest.mark.parametrize("declared", [-1, True, INT64_MAX + 1, 2])
def test_invalid_or_incomplete_declared_session_count_is_rejected(declared):
    source = history([("s", 0, "detail", "a")], snapshot=CONFIG.snapshot_time_ms)
    source = replace(source, session_count=declared)
    with pytest.raises(InputValidationError):
        build_statistics(source, HistoricalCatalog(("a",)), CONFIG)


def test_duplicate_session_cannot_increase_session_counts():
    source = history([("s", 0, "detail", "a")], snapshot=CONFIG.snapshot_time_ms)
    session = next(source.iter_sessions())
    source = replace(source, iter_sessions=lambda: iter((session, session)), session_count=2)
    with pytest.raises(InputValidationError):
        build_statistics(source, HistoricalCatalog(("a",)), CONFIG)


def test_nonfinite_popularity_denominator_is_a_numeric_error():
    config = replace(CONFIG, popularity_smoothing=1e308)
    with pytest.raises(NumericError):
        build_statistics(
            history([], snapshot=config.snapshot_time_ms), HistoricalCatalog(("a", "b")), config
        )


def test_declared_snapshot_must_match_prepared_history():
    with pytest.raises(InputValidationError):
        build_statistics(history([], snapshot=1), HistoricalCatalog(()), CONFIG)
