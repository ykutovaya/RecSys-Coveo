"""Чистое состояние префикса; без хранилища, квитанций и конкурирующих записей."""

from .reducer import build_prefix, empty_prefix, reduce_state

__all__ = ["build_prefix", "empty_prefix", "reduce_state"]
