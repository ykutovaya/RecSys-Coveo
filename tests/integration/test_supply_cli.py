"""Отдельный маршрут проверки поставки не открывает подготовку данных."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coveo_recsys.cli import main
from tests.reference.supply_oracle import ROWS, SEARCH, synthetic_supply

ROOT = Path(__file__).resolve().parents[2]


def arguments(tmp_path, input_dir, profile_path):
    return [
        "--project-root",
        str(ROOT),
        "data",
        "inspect-supply",
        "--input-dir",
        str(input_dir),
        "--profile",
        str(profile_path),
        "--run-dir",
        str(tmp_path / "run"),
        "--run-id",
        "supply_test",
    ]


def profile_fixture(tmp_path, *, rows=None):
    input_dir, profile = synthetic_supply(tmp_path, rows=rows)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    return input_dir, profile_path


def test_inspect_supply_command_reports_structural_status(tmp_path, capsys):
    input_dir, profile_path = profile_fixture(tmp_path)
    code = main(arguments(tmp_path, input_dir, profile_path))
    output = capsys.readouterr()
    assert code == 0
    result = json.loads(output.out)
    assert result["scan_complete"] is True
    assert result["format_status"] == "confirmed"
    assert result["coverage_confirmed"] is False
    assert result["preflight_ready"] is False
    assert result["real_prepare_ready"] is False


@pytest.mark.parametrize("changed", ["profile", "code"])
def test_input_or_code_change_before_publication_has_no_success_report(
    tmp_path,
    capsys,
    monkeypatch,
    changed,
):
    from coveo_recsys.cli import supply as cli_supply

    input_dir, profile_path = profile_fixture(tmp_path)
    original_inspect = cli_supply.inspect_supply
    original_source_sha = cli_supply.source_sha256
    code_changed = False

    def inspect_then_change(*args, **kwargs):
        nonlocal code_changed
        result = original_inspect(*args, **kwargs)
        if changed == "profile":
            profile_path.write_bytes(profile_path.read_bytes() + b" ")
        else:
            code_changed = True
        return result

    def source_after_change(path):
        return "f" * 64 if code_changed else original_source_sha(path)

    monkeypatch.setattr(cli_supply, "inspect_supply", inspect_then_change)
    monkeypatch.setattr(cli_supply, "source_sha256", source_after_change)
    code = main(arguments(tmp_path, input_dir, profile_path))
    output = capsys.readouterr()
    assert code != 0
    assert json.loads(output.out)["status"] == "failed"
    assert not (tmp_path / "run/supply/report.json").exists()


def test_compatible_repeat_keeps_report_hash_and_corruption_blocks_reuse(tmp_path, capsys):
    input_dir, profile_path = profile_fixture(tmp_path)
    values = arguments(tmp_path, input_dir, profile_path)
    assert main(values) == 0
    first = json.loads(capsys.readouterr().out)
    report_path = tmp_path / "run/supply/report.json"
    original = report_path.read_bytes()
    assert main(values) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["report_sha256"] == first["report_sha256"]
    assert report_path.read_bytes() == original

    report_path.write_bytes(b"{damaged")
    assert main(values) != 0
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert report_path.read_bytes() == b"{damaged"


def test_missing_profile_error_does_not_disclose_local_path(tmp_path, capsys):
    input_dir, profile_path = profile_fixture(tmp_path)
    missing = profile_path.with_name("unavailable_profile.json")
    code = main(arguments(tmp_path, input_dir, missing))
    output = capsys.readouterr()
    assert code != 0
    assert json.loads(output.out)["status"] == "failed"
    assert str(tmp_path) not in output.out + output.err


def test_invalid_timestamp_reports_safe_location(tmp_path, capsys):
    rows = copy.deepcopy(ROWS)
    rows[SEARCH][0][4] = "SECRET_INVALID_TIME"
    input_dir, profile_path = profile_fixture(tmp_path, rows=rows)
    code = main(arguments(tmp_path, input_dir, profile_path))
    output = capsys.readouterr()
    assert code != 0
    report = json.loads(output.out)
    assert report["status"] == "failed"
    assert report["error"] == {
        "code": "invalid_timestamp",
        "file": SEARCH,
        "column": "server_timestamp_epoch_ms",
        "line": 2,
    }
    assert "SECRET_INVALID_TIME" not in output.out + output.err
    assert str(tmp_path) not in output.out + output.err


def test_final_hash_stops_inside_read_when_time_budget_expires(tmp_path, capsys, monkeypatch):
    from coveo_recsys.cli import supply as cli_supply

    input_dir, profile_path = profile_fixture(tmp_path)
    target = input_dir / "browsing_train.csv"
    original_open = Path.open
    clock = SimpleNamespace(now=100.0)
    opened = 0

    class SlowHashStream:
        def __init__(self, stream):
            self.stream = stream
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            if self.reads and clock.now > 101.0:
                raise AssertionError("Hasher read another chunk after the time budget")
            chunk = self.stream.read(min(size, 1))
            if chunk:
                self.reads += 1
                clock.now = 102.0
            return chunk

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def controlled_open(path, *args, **kwargs):
        nonlocal opened
        stream = original_open(path, *args, **kwargs)
        if path == target and args and args[0] == "rb":
            opened += 1
            if opened == 2:  # Первое открытие — проверяющее чтение, второе — итоговый SHA.
                return SlowHashStream(stream)
        return stream

    monkeypatch.setattr(cli_supply, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(Path, "open", controlled_open)
    values = arguments(tmp_path, input_dir, profile_path) + ["--max-seconds", "1"]
    code = main(values)
    output = capsys.readouterr()
    assert opened >= 2
    assert code != 0
    assert json.loads(output.out)["status"] == "failed"
    assert not (tmp_path / "run/supply/report.json").exists()


def test_progress_writes_respect_cumulative_disk_budget(tmp_path, capsys):
    input_dir, profile_path = profile_fixture(tmp_path)
    code = main(arguments(tmp_path, input_dir, profile_path) + ["--max-temp-bytes", "4096"])
    capsys.readouterr()
    assert code != 0
    assert not (tmp_path / "run/supply/report.json").exists()


def test_profile_limit_bounds_read_before_allocation(tmp_path, capsys, monkeypatch):
    input_dir, profile_path = profile_fixture(tmp_path)
    profile_path.write_bytes(b" " * 65_537)
    original_open = Path.open

    class BoundedRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            assert 0 < size <= 65_537, "Profile read exceeded its allocation bound"
            return self.stream.read(size)

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BoundedRead(stream) if path == profile_path else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    assert main(arguments(tmp_path, input_dir, profile_path)) != 0
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "run/supply/report.json").exists()


def test_change_after_first_final_hash_prevents_success(tmp_path, capsys, monkeypatch):
    from coveo_recsys.cli import supply as cli_supply

    input_dir, profile_path = profile_fixture(tmp_path)
    original_hash = cli_supply._sha256_with_deadline

    def hash_then_change(path, deadline):
        digest = original_hash(path, deadline)
        if path.name == "sku_to_content.csv":
            first = input_dir / "browsing_train.csv"
            first.write_bytes(first.read_bytes() + b"\n")
        return digest

    monkeypatch.setattr(cli_supply, "_sha256_with_deadline", hash_then_change)
    assert main(arguments(tmp_path, input_dir, profile_path)) != 0
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "run/supply/report.json").exists()
