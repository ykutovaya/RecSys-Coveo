from dataclasses import replace
from itertools import chain
from pathlib import Path

import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import EventType, RawEvent, SearchPayload
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.io import FILES, iter_events, load_profile

ROOT = Path(__file__).resolve().parents[2]


def event(identifier="e", *, sku="A", kind=EventType.DETAIL, search=None, time=0, transport=None):
    return RawEvent(
        "n", "s", identifier, time, time, kind, sku, search, "fixture:" + identifier, transport
    )


@pytest.mark.parametrize("chunk", [1, 2, 100])
def test_full_groups_across_sources_and_chunks(tmp_path, chunk):
    profile = load_profile(ROOT / "configs/synthetic_csv.json")
    streams = [
        iter_events(ROOT / "tests/fixtures/e1" / FILES[k], k, profile, batch_size=1)
        for k in ("browsing", "search")
    ]
    blocks = list(canonicalize(chain(*streams), temp_dir=tmp_path, chunk_size=chunk, fan_in=2))
    assert len(blocks) == 9
    assert blocks[0].events[0].sku == "A"
    assert blocks[0].raw_multiplicities[0][1] == 2
    assert len(blocks[0].events[0].origins) == 2
    assert not list(tmp_path.iterdir())


def test_transport_duplicates_and_conflicts(tmp_path):
    a = event("a", transport="transport")
    (block,) = canonicalize([a, replace(a, provenance="retry")], temp_dir=tmp_path)
    assert block.raw_multiplicities[0][1] == 1
    assert block.events[0].origins == ("fixture:a", "retry")
    with pytest.raises(InputValidationError):
        list(canonicalize([a, replace(a, sku="B")], temp_dir=tmp_path))
    (block,) = canonicalize([event("a"), event("b")], temp_dir=tmp_path)
    assert block.raw_multiplicities[0][1] == 2


def test_same_session_other_namespace_is_separate(tmp_path):
    blocks = list(
        canonicalize([event(), replace(event(), source_namespace="other")], temp_dir=tmp_path)
    )
    assert len(blocks) == 2
    assert blocks[0].block_id != blocks[1].block_id


def test_search_identity_before_normalization(tmp_path):
    a = event("a", sku=None, kind=EventType.SEARCH, search=SearchPayload((1.0, -0.0), None))
    b = replace(a, event_id="b", search=SearchPayload((2.0, 0.0), None))
    c = replace(a, event_id="c", search=SearchPayload((1.0, 0.0), None))
    (first,) = canonicalize([a, b, c], temp_dir=tmp_path, chunk_size=1)
    (second,) = canonicalize([c, b, a], temp_dir=tmp_path, chunk_size=2)
    assert first.block_id == second.block_id
    assert first.search_multiplicities == (2, 1)
    assert len(first.search_payloads) == 2


def test_explicit_resource_refusal_and_cleanup(tmp_path):
    with pytest.raises(InputValidationError):
        list(canonicalize([event("a"), event("b")], temp_dir=tmp_path, max_group_events=1))
    assert not list(tmp_path.iterdir())
    with pytest.raises(InputValidationError):
        list(canonicalize([event()], temp_dir=tmp_path, chunk_bytes=10))


@pytest.mark.parametrize("chunk_size", [3, 4])
def test_single_batch_matches_external_sort_without_chunk_files(tmp_path, monkeypatch, chunk_size):
    rows = [event("b", sku="B"), event("a"), event("retry", transport="t")]
    expected = list(canonicalize(rows, temp_dir=tmp_path, chunk_size=1, fan_in=2))
    checked = []

    def no_chunk_file(*args, **kwargs):
        raise AssertionError("Одна порция не требует файлов сортировки")

    def check_resources():
        checked.append(True)
        assert not any(path.name in {"identity", "time"} for path in tmp_path.rglob("*"))

    monkeypatch.setattr(Path, "open", no_chunk_file)
    actual = list(
        canonicalize(
            rows, temp_dir=tmp_path, chunk_size=chunk_size, check_resources=check_resources
        )
    )
    assert actual == expected
    assert len(checked) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("chunk_size", [1, 10])
def test_sort_keeps_nested_record_validation(tmp_path, chunk_size):
    payload = SearchPayload((1.0,), None)
    row = event("search", sku=None, kind=EventType.SEARCH, search=payload)
    object.__setattr__(payload, "reason", 17)
    with pytest.raises(InputValidationError):
        list(canonicalize([row, event()], temp_dir=tmp_path, chunk_size=chunk_size))
    assert not list(tmp_path.iterdir())


def test_absent_search_has_one_canonical_marker_without_order_dependence():
    absent = event("technical", sku=None, kind=EventType.SEARCH, search=None)
    explicit = replace(absent, search=SearchPayload(None, "vector_absent"), provenance="another")
    (a,) = canonicalize([absent, explicit], chunk_size=1)
    (b,) = canonicalize([explicit, absent], chunk_size=2)
    assert a == b
    assert a.events[0].search == SearchPayload(None, "vector_absent")
    assert a.search_multiplicities == (2,)


def test_resource_check_observes_partial_merge_and_input_output_peak(tmp_path):
    measured = []

    def check_resources():
        for output in tmp_path.rglob("merge-*.jsonl"):
            sources = list(output.parent.glob("chunk-*.jsonl"))
            if sources:
                measured.append(
                    (output.stat().st_size, sum(path.stat().st_size for path in sources))
                )

    blocks = list(
        canonicalize(
            (event(str(i)) for i in range(1025)),
            temp_dir=tmp_path,
            chunk_size=512,
            fan_in=3,
            check_resources=check_resources,
        )
    )
    assert blocks[0].raw_multiplicities[0][1] == 1025
    assert any(0 < output < inputs for output, inputs in measured)
    assert any(output == inputs for output, inputs in measured)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["memory", "chunk", "merge"])
def test_resource_callback_refusal_cleans_temporary_sort_files(tmp_path, stage):
    def check_resources():
        if stage == "memory" or any(
            path.stat().st_size for path in tmp_path.rglob(f"{stage}-*.jsonl")
        ):
            raise InputValidationError("test_resource_limit")

    with pytest.raises(InputValidationError, match="test_resource_limit"):
        list(
            canonicalize(
                (event(str(i)) for i in range(5)),
                temp_dir=tmp_path,
                chunk_size=10 if stage == "memory" else 1,
                fan_in=2,
                check_resources=check_resources,
            )
        )
    assert not list(tmp_path.iterdir())
