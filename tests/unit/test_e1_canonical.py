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


def test_absent_search_has_one_canonical_marker_without_order_dependence():
    absent = event("technical", sku=None, kind=EventType.SEARCH, search=None)
    explicit = replace(absent, search=SearchPayload(None, "vector_absent"), provenance="another")
    (a,) = canonicalize([absent, explicit], chunk_size=1)
    (b,) = canonicalize([explicit, absent], chunk_size=2)
    assert a == b
    assert a.events[0].search == SearchPayload(None, "vector_absent")
    assert a.search_multiplicities == (2,)
