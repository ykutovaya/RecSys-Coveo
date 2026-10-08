"""Измерение пика памяти текущего процесса без зависимости от команд запуска."""

import sys
from functools import cache
from typing import Any


@cache
def _windows_memory_api() -> tuple[Any, Any, Any]:
    """Однократная настройка WinAPI; сами показания не кешируются."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("faults", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "current",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ],
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    process = kernel.GetCurrentProcess
    process.argtypes = []
    process.restype = wintypes.HANDLE
    measure = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    measure.restype = wintypes.BOOL
    return Counters, process, measure


def peak_memory() -> dict[str, Any]:
    """Пик RSS/working set процесса, не оценка памяти рекомендательной системы."""
    try:
        if sys.platform == "win32":
            import ctypes

            Counters, process, measure = _windows_memory_api()
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not measure(process(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error())
            value, method = int(counters.peak), "GetProcessMemoryInfo.PeakWorkingSetSize"
        else:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            value *= 1 if sys.platform == "darwin" else 1024
            method = "getrusage.ru_maxrss"
        return {"peak_process_memory_bytes": value, "memory_method": method}
    except (OSError, AttributeError, ImportError) as exc:
        return {"peak_process_memory_bytes": None, "memory_reason": type(exc).__name__}
