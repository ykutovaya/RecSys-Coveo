"""Проверка поставки на вымышленных CSV, без чтения Coveo."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import event, given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.data.supply import SupplyLimits, _parse_epoch_ms, inspect_supply
from tests.reference.supply_oracle import (
    BROWSING,
    HEADERS,
    PRODUCTS,
    ROWS,
    SEARCH,
    csv_bytes,
    synthetic_supply,
)


def limits(**changes):
    values = {
        "batch_size": 2,
        "max_batch_chars": 4096,
        "max_header_bytes": 1024,
        "max_record_bytes": 4096,
        "max_field_chars": 1024,
        "max_seconds": 30,
        "max_temp_bytes": 1_048_576,
    }
    values.update(changes)
    return SupplyLimits(**values)


def scan(tmp_path: Path, *, rows=None, raw=None, scan_limits=None):
    input_dir, profile = synthetic_supply(tmp_path, rows=rows, raw=raw)
    return inspect_supply(input_dir, profile, limits=scan_limits or limits())


def test_full_scan_reports_only_structure_and_unknown_coverage(tmp_path):
    result = scan(tmp_path)
    assert result["scan_complete"] is True
    assert result["format_status"] == "confirmed"
    assert result["coverage_confirmed"] is False
    assert result["coverage_status"] == "unknown"
    assert result["coverage_reason"] == "no_independent_telemetry_evidence"
    assert result["unknown_intervals"] == "not_determined"
    assert result["preflight_ready"] is False
    assert result["real_prepare_ready"] is False
    assert set(result["files"]) == {BROWSING, SEARCH, PRODUCTS}
    assert {name: result["files"][name]["rows"] for name in result["files"]} == {
        BROWSING: 3,
        SEARCH: 2,
        PRODUCTS: 2,
    }
    assert (
        result["files"][BROWSING]["min_timestamp_ms"],
        result["files"][BROWSING]["max_timestamp_ms"],
    ) == (0, 2)
    assert result["files"][PRODUCTS]["min_timestamp_ms"] is None
    assert result["files"][PRODUCTS]["max_timestamp_ms"] is None
    assert result["files"][PRODUCTS]["timestamp_status"] == "not_applicable"
    assert result["files"][SEARCH]["vector_dimensions"]["query_vector"] == 2
    assert result["files"][PRODUCTS]["vector_dimensions"] == {
        "description_vector": 2,
        "image_vector": 3,
    }
    assert result["files"][SEARCH]["search_lists"]["clicked_skus_hash"]["missing"] == 1
    assert result["files"][SEARCH]["search_lists"]["product_skus_hash"]["explicit_empty"] == 1
    for name, details in result["files"].items():
        source = tmp_path / "input" / name
        content = source.read_bytes()
        assert details["size_bytes"] == len(content)
        assert details["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "empty",
        "unfinished_quote",
        "wrong_hash",
        "missing_table",
        "unknown_profile_version",
        "unknown_time_unit",
    ],
)
def test_incomplete_or_unconfirmed_input_cannot_succeed(tmp_path, case):
    input_dir, profile = synthetic_supply(tmp_path)
    if case == "missing":
        (input_dir / BROWSING).unlink()
    elif case == "empty":
        (input_dir / BROWSING).write_bytes(b"")
    elif case == "unfinished_quote":
        content = csv_bytes(HEADERS[BROWSING], ROWS[BROWSING]) + b'"unfinished\n'
        (input_dir / BROWSING).write_bytes(content)
        profile["files"][BROWSING]["expected_size_bytes"] = len(content)
        profile["files"][BROWSING]["expected_sha256"] = hashlib.sha256(content).hexdigest()
    elif case == "wrong_hash":
        profile["files"][BROWSING]["expected_sha256"] = "0" * 64
    elif case == "missing_table":
        del profile["files"][PRODUCTS]
    elif case == "unknown_profile_version":
        profile["schema_version"] = "unknown"
    else:
        profile["timestamp_unit"] = "seconds"
    with pytest.raises((InputValidationError, OSError)):
        inspect_supply(input_dir, profile, limits=limits())


@pytest.mark.parametrize(
    "case", ["extra_column", "missing_column", "reordered_columns", "bad_utf8"]
)
def test_csv_schema_and_encoding_are_strict(tmp_path, case):
    header = list(HEADERS[BROWSING])
    rows = copy.deepcopy(ROWS[BROWSING])
    if case == "extra_column":
        header.append("unexpected")
        rows = [row + ["secret"] for row in rows]
    elif case == "missing_column":
        header.pop()
        rows = [row[:-1] for row in rows]
    elif case == "reordered_columns":
        header[0], header[1] = header[1], header[0]
    content = b"\xff" if case == "bad_utf8" else csv_bytes(header, rows)
    input_dir, profile = synthetic_supply(tmp_path, raw={BROWSING: content})
    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits())


def test_quoted_newline_unicode_and_pageview_action_are_accepted(tmp_path):
    rows = copy.deepcopy(ROWS)
    rows[BROWSING].append(["0001", "pageview", "detail", "sku-C", "3", "строка\n2"])
    rows[BROWSING].append(["е\u0301", "pageview", "", "", "4", ""])
    rows[BROWSING].append(["é", "pageview", "", "", "5", ""])
    result = scan(tmp_path, rows=rows)
    assert result["files"][BROWSING]["rows"] == len(rows[BROWSING])
    assert result["files"][BROWSING]["max_timestamp_ms"] == 5


@pytest.mark.parametrize(
    "value", ["", "1.5", "1e3", " 1", "9223372036854775808", "-9223372036854775809", "−1"]
)
def test_time_rejects_non_int64_milliseconds(value):
    with pytest.raises(InputValidationError):
        _parse_epoch_ms(value)


@pytest.mark.parametrize(
    "value,expected", [("-9223372036854775808", -(2**63)), ("9223372036854775807", 2**63 - 1)]
)
def test_time_accepts_int64_boundaries(value, expected):
    assert _parse_epoch_ms(value) == expected


@seed(17)
@settings(max_examples=1000, deadline=None)
@given(st.integers(min_value=-(2**63), max_value=2**63 - 1))
def test_time_property_matches_independent_integer_oracle(value):
    event("negative" if value < 0 else "nonnegative")
    assert _parse_epoch_ms(str(value)) == value


@pytest.mark.parametrize(
    "value,reason",
    [
        ("", "vector_absent"),
        ("[0,0]", "vector_zero"),
        ("[1e-13,0]", "vector_zero"),
        ("[NaN,0]", "vector_invalid"),
        ("[1]", "vector_invalid"),
        ("[1e308,1e308]", "valid"),
    ],
)
def test_vector_reasons_keep_absence_and_numerical_failure_separate(tmp_path, value, reason):
    rows = copy.deepcopy(ROWS)
    rows[SEARCH][0][1] = value
    result = scan(tmp_path, rows=rows)
    expected_count = 2 if reason == "valid" else 1
    assert result["files"][SEARCH]["vector_reasons"]["query_vector"][reason] == expected_count
    assert result["format_status"] == "confirmed"  # Вторая поисковая строка пригодна.


def test_no_valid_query_vector_leaves_format_unknown(tmp_path):
    rows = copy.deepcopy(ROWS)
    for row in rows[SEARCH]:
        row[1] = "[0,0]"
    result = scan(tmp_path, rows=rows)
    assert result["scan_complete"] is True
    assert result["format_status"] == "unknown"
    assert result["files"][SEARCH]["vector_dimensions"]["query_vector"] is None


@pytest.mark.parametrize("value", ["3.5", "NaN", "Inf", "-Infinity"])
def test_price_bucket_rejects_non_integral_or_nonfinite_value(tmp_path, value):
    rows = copy.deepcopy(ROWS)
    rows[PRODUCTS][0][4] = value
    input_dir, profile = synthetic_supply(tmp_path, rows=rows)
    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits())


@pytest.mark.parametrize("value", ["['unclosed'", "[1]", "__import__('os').system('echo bad')"])
def test_search_list_uses_safe_string_list_grammar(tmp_path, value):
    rows = copy.deepcopy(ROWS)
    rows[SEARCH][0][2] = value
    input_dir, profile = synthetic_supply(tmp_path, rows=rows)
    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits())


def test_report_and_error_exclude_input_values(tmp_path):
    marker = "SECRET_TARGET_SKU_DO_NOT_LOG"
    session_marker = "SECRET_SESSION_ID_DO_NOT_LOG"
    rows = copy.deepcopy(ROWS)
    rows[BROWSING][0][0] = session_marker
    rows[SEARCH][0][0] = session_marker
    rows[BROWSING][2][3] = marker
    rows[SEARCH][0][2] = f"['{marker}']"
    rows[PRODUCTS][0][0] = marker
    result = scan(tmp_path, rows=rows)
    serialized = json.dumps(result, ensure_ascii=False)
    assert marker not in serialized
    assert session_marker not in serialized
    assert "purchase" not in serialized
    assert "action_counts" not in serialized
    rows[BROWSING][0][1] = marker  # Недопустимое значение не должно попасть в сообщение ошибки.
    input_dir, profile = synthetic_supply(tmp_path / "bad", rows=rows)
    with pytest.raises(InputValidationError) as error:
        inspect_supply(input_dir, profile, limits=limits())
    assert marker not in str(error.value)


def test_change_to_first_file_during_third_file_fails(tmp_path):
    input_dir, profile = synthetic_supply(tmp_path)
    changed = False

    def progress(filename, rows_read, bytes_read):
        nonlocal changed
        if filename == PRODUCTS and rows_read == 0 and bytes_read == 0 and not changed:
            with (input_dir / BROWSING).open("ab") as stream:
                stream.write(b"\n")
            changed = True

    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits(), progress=progress)
    assert changed


def test_batch_size_does_not_change_structural_result(tmp_path):
    input_dir, profile = synthetic_supply(tmp_path)
    small = inspect_supply(input_dir, profile, limits=limits(batch_size=1, max_batch_chars=128))
    large = inspect_supply(input_dir, profile, limits=limits(batch_size=100))
    assert small["files"] == large["files"]
    assert small["format_status"] == large["format_status"]


@pytest.mark.parametrize(
    "kind", ["max_header_bytes", "max_record_bytes", "max_field_chars", "max_batch_chars"]
)
def test_size_limits_accept_boundary_and_reject_one_less(tmp_path, kind):
    input_dir, profile = synthetic_supply(tmp_path)
    header_bytes = max(len(csv_bytes(HEADERS[name], [])) for name in HEADERS)
    record_bytes = max(
        len(csv_bytes(HEADERS[name], [row])) - len(csv_bytes(HEADERS[name], []))
        for name in HEADERS
        for row in ROWS[name]
    )
    field_chars = max(
        len(cell) for name in HEADERS for row in [HEADERS[name], *ROWS[name]] for cell in row
    )
    batch_chars = max(sum(map(len, row)) for rows in ROWS.values() for row in rows)
    exact = dict(
        max_header_bytes=header_bytes,
        max_record_bytes=record_bytes,
        max_field_chars=field_chars,
        max_batch_chars=batch_chars,
    )
    assert inspect_supply(input_dir, profile, limits=limits(batch_size=1, **exact))["scan_complete"]
    exact[kind] -= 1
    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits(batch_size=1, **exact))


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 0},
        {"max_batch_chars": 1},
        {"max_header_bytes": 1},
        {"max_record_bytes": 1},
        {"max_field_chars": 1},
        {"max_seconds": 0},
        {"max_temp_bytes": -1},
    ],
)
def test_resource_limits_fail_cleanly(tmp_path, change):
    input_dir, profile = synthetic_supply(tmp_path)
    with pytest.raises(InputValidationError):
        inspect_supply(input_dir, profile, limits=limits(**change))
