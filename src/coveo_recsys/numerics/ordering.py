"""Точный порядок ID и баллов; без локали, нормализации и допуска на равенство."""

from collections.abc import Iterable

from coveo_recsys.contracts import id_bytes

from ._validation import finite_scalar


def sort_ids(ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(ids, key=id_bytes))


def sort_scored_ids(items: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    """Баллы по убыванию, точные равенства — по исходным UTF-8 байтам ID.

    Повторы сохраняются: сортировка не выполняет канонизацию или отбор.
    """
    checked = [(sku, finite_scalar(score)) for sku, score in items]
    return tuple(sorted(checked, key=lambda item: (-item[1], id_bytes(item[0]))))
