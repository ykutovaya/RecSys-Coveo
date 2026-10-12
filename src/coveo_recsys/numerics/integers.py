"""Точная арифметика времени в миллисекундах и счётчиков int64."""

from coveo_recsys.contracts import INT64_MAX, INT64_MIN, NumericError, require_int64


def _result(value: int) -> int:
    if not INT64_MIN <= value <= INT64_MAX:
        raise NumericError("Переполнение int64")
    return value


def checked_add(left: int, right: int) -> int:
    return _result(require_int64(left) + require_int64(right))


def checked_subtract(left: int, right: int) -> int:
    return _result(require_int64(left) - require_int64(right))


def checked_multiply(left: int, right: int) -> int:
    return _result(require_int64(left) * require_int64(right))


def seconds_to_milliseconds(seconds: int) -> int:
    """Явный перевод целых секунд, без округления дробей или float."""
    return checked_multiply(seconds, 1000)


def counter_add(counter: int, delta: int = 1) -> int:
    """Изменение неотрицательного счётчика; отрицательный итог не зажимается."""
    require_int64(counter, nonnegative=True)
    result = checked_add(counter, delta)
    if result < 0:
        raise NumericError("Отрицательный результат счётчика")
    return result
