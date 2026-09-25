import csv
import hashlib
import json
from dataclasses import replace
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import EventType
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.io import FILES, inspect_inputs, iter_events, load_profile, row_batches

ROOT = Path(__file__).resolve().parents[2]


def actual_files(tmp_path, *, browsing=None, search=None, products=None):
    profile = json.loads((ROOT / "configs/coveo_supply_v1.json").read_text(encoding="utf-8"))
    vector = json.dumps([2.0] + [0.0] * 49)
    rows = {
        "browsing": browsing
        if browsing is not None
        else [["001", "event_product", "detail", "sku", "1000", ""]],
        "search": search
        if search is not None
        else [["001", vector, "['sku']", "['other', 'sku']", "1001"]],
        "products": products
        if products is not None
        else [["sku", vector, "parent/child", "", "2.0"]],
    }
    paths = {table: tmp_path / filename for table, filename in FILES.items()}
    for table, path in paths.items():
        specification = profile["files"][path.name]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(specification["columns"])
            writer.writerows(rows[table])
        specification["expected_size_bytes"] = path.stat().st_size
        specification["expected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return paths, profile_path


@pytest.mark.parametrize("batch_size", [1, 3])
def test_actual_profile_maps_events_without_inventing_delivery_ids(tmp_path, batch_size):
    actions = ["detail", "detail", "add", "remove", "purchase"]
    browsing = [["001", "event_product", action, "sku", "1000", ""] for action in actions]
    browsing[0][1] = "pageview"  # Товарное действие задаёт product_action, не event_type.
    browsing += [["001", "pageview", "", "sku", "1000", "page"]]
    paths, profile_path = actual_files(tmp_path, browsing=browsing)
    profile = load_profile(profile_path)
    assert (profile.scope, profile.namespace) == ("actual", "coveo-sigir-1.0.0")
    assert dict(profile.dimensions) == {"query": 50, "description": 50, "image": 50}
    with pytest.raises(TypeError):
        profile.expected_files["browsing"]["expected_sha256"] = "changed"
    events = list(iter_events(paths["browsing"], "browsing", profile, batch_size=batch_size))
    assert [event.event_type for event in events] == [EventType(a) for a in actions] + [
        EventType.PAGE
    ]
    assert all(event.session_id == "001" and event.transport_id is None for event in events)
    assert all(event.event_time_ms == event.available_time_ms == 1000 for event in events)
    assert len({event.event_id for event in events}) == len(events)
    digest = hashlib.sha256(paths["browsing"].read_bytes()).hexdigest()
    assert events[0].origins == (f"browsing_train.csv:{digest}:line=2",)
    searches = list(iter_events(paths["search"], "search", profile, batch_size=batch_size))
    assert len(searches) == 1
    assert searches[0].event_type == EventType.SEARCH and searches[0].sku is None
    assert searches[0].search.vector == (2.0,) + (0.0,) * 49


def test_actual_session_filter_keeps_all_selected_rows_before_vector_parsing(tmp_path):
    paths, profile_path = actual_files(
        tmp_path,
        search=[
            ["001", "", "", "[]", "0"],
            ["other", "bad", "not a list", "[]", "1"],
            ["001", "[0]", "[]", "['sku']", "99999999"],
        ],
    )
    events = list(
        iter_events(
            paths["search"], "search", load_profile(profile_path), session_ids=frozenset({"001"})
        )
    )
    assert [event.event_time_ms for event in events] == [0, 99999999]
    assert [event.search.reason for event in events] == ["vector_absent", "vector_invalid"]
    assert events[1].provenance.endswith("line=4")
    with paths["search"].open("a", encoding="utf-8") as stream:
        stream.write("other,too,few\n")
    with pytest.raises(InputValidationError, match="Неверная строка"):
        list(row_batches(paths["search"], "search", load_profile(profile_path)))


@pytest.mark.parametrize(
    "field,value",
    [("event_type", "unknown"), ("action", "unknown"), ("sku", ""), ("timestamp", "1.5")],
)
def test_actual_invalid_behavior_fails_without_disclosing_values(tmp_path, field, value):
    columns = ["session_id", "event_type", "action", "sku", "timestamp", "url"]
    row = ["private-session", "event_product", "detail", "private-sku", "1000", ""]
    row[columns.index(field)] = value
    paths, profile_path = actual_files(tmp_path, browsing=[row])
    with pytest.raises(InputValidationError) as error:
        list(iter_events(paths["browsing"], "browsing", load_profile(profile_path)))
    assert "private-session" not in str(error.value) and "private-sku" not in str(error.value)


@pytest.mark.parametrize("value", ["[__import__('os')]", "['sku', 1]", '["sku"]'])
def test_actual_search_lists_only_accept_confirmed_format(tmp_path, value):
    paths, profile_path = actual_files(tmp_path, search=[["s", "", value, "[]", "0"]])
    with pytest.raises(InputValidationError):
        list(iter_events(paths["search"], "search", load_profile(profile_path)))


@pytest.mark.parametrize(
    "price,valid", [("", True), ("2.0", True), ("2", True), ("2.5", False), ("NaN", False)]
)
def test_actual_products_keep_category_and_validate_integral_price(tmp_path, price, valid):
    paths, profile_path = actual_files(
        tmp_path, products=[["sku", "", "raw/category", "[0]", price]]
    )
    profile = load_profile(profile_path)
    row = next(row_batches(paths["products"], "products", profile))[0][1]
    assert row["category"] == "raw/category" and row["price_bucket"] == price
    if not valid:
        with pytest.raises(InputValidationError):
            inspect_inputs(paths, profile)
        return
    report = inspect_inputs(paths, profile)
    assert report["scope"] == "actual"
    assert report["files"]["products"]["vector_reasons"] == {
        "description:vector_absent": 1,
        "image:vector_invalid": 1,
    }


def test_actual_reader_enforces_profile_identity_and_record_budget(tmp_path):
    paths, profile_path = actual_files(tmp_path)
    profile = load_profile(profile_path)
    with pytest.raises(InputValidationError):
        list(iter_events(paths["browsing"], "browsing", profile, file_sha256="0" * 64))
    with pytest.raises(InputValidationError):
        list(row_batches(paths["search"], "search", replace(profile, max_row_chars=100)))


def test_actual_filter_reports_progress_for_unselected_rows(tmp_path):
    paths, profile_path = actual_files(
        tmp_path, browsing=[["other", "pageview", "", "", "0", ""]] * 3
    )
    progress = []
    events = list(
        iter_events(
            paths["browsing"],
            "browsing",
            load_profile(profile_path),
            batch_size=2,
            session_ids=frozenset({"001"}),
            progress=lambda table, rows: progress.append((table, rows)),
        )
    )
    assert events == [] and progress == [("browsing", 2), ("browsing", 3)]


def test_actual_header_order_is_not_inferred(tmp_path):
    paths, profile_path = actual_files(tmp_path)
    path = paths["browsing"]
    path.write_text(
        "event_type,session_id_hash,product_action,product_sku_hash,server_timestamp_epoch_ms,hashed_url\n",
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError):
        list(row_batches(path, "browsing", load_profile(profile_path)))


@seed(17)
@given(
    behavior_order=st.permutations((0, 1, 2)),
    search_order=st.permutations((0, 1, 2)),
    batch_size=st.integers(1, 7),
    magnitude=st.integers(1, 10),
)
def test_actual_tied_group_semantics_do_not_depend_on_order_or_batches(
    behavior_order, search_order, batch_size, magnitude
):
    browsing = [
        ["001", "event_product", action, "sku", "1000", ""]
        for action in ("detail", "detail", "add")
    ]
    vectors = [(float(value),) + (0.0,) * 49 for value in (magnitude, 2 * magnitude)]
    search = [
        ["001", json.dumps(vector), "[]", "['sku']", "1000"]
        for vector in (vectors[0], vectors[0], vectors[1])
    ]
    with TemporaryDirectory() as directory:
        paths, profile_path = actual_files(
            Path(directory),
            browsing=[browsing[i] for i in behavior_order],
            search=[search[i] for i in search_order],
        )
        profile = load_profile(profile_path)
        streams = (
            iter_events(paths[table], table, profile, batch_size=batch_size)
            for table in ("browsing", "search")
        )
        blocks = list(canonicalize(chain.from_iterable(streams), chunk_size=3, fan_in=2))
    assert len(blocks) == 1
    block = blocks[0]
    assert (
        block.source_namespace,
        block.session_id,
        block.event_time_ms,
        block.available_time_ms,
    ) == ("coveo-sigir-1.0.0", "001", 1000, 1000)
    assert {
        (event.event_type, event.sku): count
        for event, (_, count) in zip(block.events, block.raw_multiplicities, strict=True)
    } == {(EventType.DETAIL, "sku"): 2, (EventType.ADD, "sku"): 1, (EventType.SEARCH, None): 3}
    assert {
        (payload.vector, payload.reason): count
        for payload, count in zip(block.search_payloads, block.search_multiplicities, strict=True)
    } == {(vectors[0], None): 2, (vectors[1], None): 1}
