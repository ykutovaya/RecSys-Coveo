"""Настройка WinAPI повторно используется, показания и ошибки — нет."""

import ctypes
import sys
from types import SimpleNamespace

import pytest

from coveo_recsys import process_memory


@pytest.fixture(autouse=True)
def clear_windows_api_cache():
    process_memory._windows_memory_api.cache_clear()
    yield
    process_memory._windows_memory_api.cache_clear()


class Function:
    def __init__(self, call):
        self.call = call
        self.settings = []

    def __setattr__(self, name, value):
        if name in {"argtypes", "restype"}:
            self.settings.append(name)
        object.__setattr__(self, name, value)

    def __call__(self, *args):
        return self.call(*args)


def windows_api(monkeypatch, readings):
    values = iter(readings)
    counters, loads, handles = [], [], []

    def get_handle():
        handles.append(71)
        return 71

    def measure(handle, pointer, size):
        assert handle == 71
        current = pointer._obj
        assert size == current.cb == ctypes.sizeof(current)
        counters.append(current)
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        if value == 0:
            return 0
        current.peak = value
        return 1

    kernel = SimpleNamespace(GetCurrentProcess=Function(get_handle))
    psapi = SimpleNamespace(GetProcessMemoryInfo=Function(measure))

    def load(name, *, use_last_error):
        assert use_last_error is True
        loads.append(name)
        return {"kernel32": kernel, "psapi": psapi}[name]

    monkeypatch.setattr(process_memory, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    return SimpleNamespace(
        counters=counters, loads=loads, handles=handles, kernel=kernel, psapi=psapi, load=load
    )


def test_windows_reuses_api_setup_but_measures_each_call(monkeypatch):
    api = windows_api(monkeypatch, [100, 200, 300])
    assert [process_memory.peak_memory() for _ in range(3)] == [
        {
            "peak_process_memory_bytes": value,
            "memory_method": "GetProcessMemoryInfo.PeakWorkingSetSize",
        }
        for value in (100, 200, 300)
    ]
    assert api.loads == ["kernel32", "psapi"]
    assert api.handles == [71, 71, 71]
    assert len({type(counters) for counters in api.counters}) == 1
    assert len({id(counters) for counters in api.counters}) == 3
    assert api.kernel.GetCurrentProcess.settings.count("restype") == 1
    assert api.psapi.GetProcessMemoryInfo.settings.count("argtypes") == 1


@pytest.mark.parametrize(
    "failure", [0, OSError("denied"), AttributeError("missing"), ImportError("missing")]
)
def test_windows_measurement_error_never_returns_cached_value(monkeypatch, failure):
    api = windows_api(monkeypatch, [100, failure, 300])
    assert process_memory.peak_memory()["peak_process_memory_bytes"] == 100
    reason = "OSError" if failure == 0 else type(failure).__name__
    assert process_memory.peak_memory() == {
        "peak_process_memory_bytes": None,
        "memory_reason": reason,
    }
    assert process_memory.peak_memory()["peak_process_memory_bytes"] == 300
    assert api.loads == ["kernel32", "psapi"]


@pytest.mark.parametrize("error", [OSError, AttributeError, ImportError])
def test_windows_setup_error_is_reported_and_retried(monkeypatch, error):
    api = windows_api(monkeypatch, [300])
    attempts = []

    def load(name, *, use_last_error):
        attempts.append(name)
        if len(attempts) == 1:
            raise error("not ready")
        return api.load(name, use_last_error=use_last_error)

    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    assert process_memory.peak_memory() == {
        "peak_process_memory_bytes": None,
        "memory_reason": error.__name__,
    }
    assert process_memory.peak_memory()["peak_process_memory_bytes"] == 300
    assert attempts == ["kernel32", "kernel32", "psapi"]


@pytest.mark.parametrize(("platform", "scale"), [("linux", 1024), ("darwin", 1)])
def test_nonwindows_keeps_current_getrusage_units(monkeypatch, platform, scale):
    readings = iter((7, 19))
    calls = []

    def getrusage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=next(readings))

    monkeypatch.setattr(process_memory, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(
        sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=getrusage)
    )
    assert [process_memory.peak_memory() for _ in range(2)] == [
        {"peak_process_memory_bytes": value * scale, "memory_method": "getrusage.ru_maxrss"}
        for value in (7, 19)
    ]
    assert calls == [0, 0]
