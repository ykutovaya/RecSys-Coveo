"""Точный отбор основной и резервной ветвей."""

from .candidates import retrieve
from .fallback import fallback as retrieve_fallback

__all__ = ["retrieve", "retrieve_fallback"]
