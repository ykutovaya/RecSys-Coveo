from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import TaskConfig
from coveo_recsys.contracts.records import CanonicalBlock, EventType, RawEvent, SearchPayload
from coveo_recsys.data.canonical import canonicalize
from coveo_recsys.data.search import resolve_search_context
from coveo_recsys.numerics import last_group_max, last_group_mean
from tests.reference.e1_state_oracle import search as search_oracle


def block_with_search(payloads, counts=None, time=10):
    event = RawEvent(
        "source",
        "session",
        "search",
        time,
        time + 2,
        EventType.SEARCH,
        None,
        payloads[0],
        "fixture",
    )
    return CanonicalBlock(
        "block",
        "session",
        time,
        time + 2,
        (event,),
        (("search", sum(counts) if counts else len(payloads)),),
        tuple(payloads),
        tuple(counts or [1] * len(payloads)),
    )


@pytest.mark.parametrize(
    "payloads,reason",
    [
        ([SearchPayload((1.0, 0.0), None), SearchPayload((0.0, 1.0), None)], "search_ambiguous"),
        ([SearchPayload((1.0, 0.0), None), SearchPayload((2.0, 0.0), None)], "search_ambiguous"),
        ([SearchPayload((1.0, 0.0), None), SearchPayload(None, "vector_absent")], "vector_absent"),
        (
            [SearchPayload((1.0, 0.0), None), SearchPayload(None, "vector_invalid")],
            "vector_invalid",
        ),
        ([SearchPayload((0.0, 0.0), None)], "vector_zero"),
        ([SearchPayload((1.0,), None)], "vector_invalid"),
    ],
)
def test_ambiguous_or_unavailable_context_is_explicit(payloads, reason):
    result = resolve_search_context(block_with_search(payloads), TaskConfig())
    assert result is not None
    assert result.payload == SearchPayload(None, reason)
    assert result.available_time_ms == 12


@pytest.mark.parametrize("vector", [(1.0, -0.0), (1e200, -1e200), (1e-12, 1e-12)])
def test_repetitions_and_large_coordinates_are_safe(vector):
    payloads = [SearchPayload(vector, None)]
    first = resolve_search_context(block_with_search(payloads, [1]), TaskConfig())
    repeated = resolve_search_context(block_with_search(payloads, [100]), TaskConfig())
    expected, reason = search_oracle([(vector, None)], 2)
    assert first == repeated
    assert first is not None and first.payload.reason == reason
    np.testing.assert_allclose(first.payload.vector, expected, atol=1e-10, rtol=1e-8)


def test_signed_zero_and_permutation_after_canonicalization(tmp_path):
    first = RawEvent(
        "n", "s", "a", 1, 1, EventType.SEARCH, None, SearchPayload((1.0, -0.0), None), "row:a"
    )
    second = replace(
        first, event_id="b", provenance="row:b", search=SearchPayload((1.0, 0.0), None)
    )
    results = []
    for rows in ([first, second], [second, first]):
        (block,) = canonicalize(rows, temp_dir=tmp_path, chunk_size=1)
        assert block.search_multiplicities == (2,)
        results.append(resolve_search_context(block, TaskConfig()))
    assert results[0] == results[1]
    assert results[0].payload == SearchPayload((1.0, 0.0), None)


def test_no_search_differs_from_invalid_search():
    search_block = block_with_search([SearchPayload(None, "vector_absent")])
    detail = replace(search_block.events[0], event_type=EventType.DETAIL, sku="sku", search=None)
    block = replace(search_block, events=(detail,), search_payloads=(), search_multiplicities=())
    assert resolve_search_context(block, TaskConfig()) is None
    assert resolve_search_context(search_block, TaskConfig()) is not None
    with pytest.raises(InputValidationError):
        resolve_search_context(
            replace(search_block, search_payloads=(), search_multiplicities=()), TaskConfig()
        )


def test_last_detail_uses_raw_counts_but_masks_before_mean(tmp_path):
    rows = [
        RawEvent(
            "n",
            "s",
            str(index),
            0,
            0,
            EventType.DETAIL,
            "A" if index < 9 else "B",
            None,
            f"row:{index}",
        )
        for index in range(10)
    ]
    (block,) = canonicalize(rows, temp_dir=tmp_path)
    multiplicities = dict(block.raw_multiplicities)
    ordered = sorted(block.events, key=lambda event: event.sku.encode("utf-8"))
    counts = [multiplicities[event.event_id] for event in ordered]
    for semantics, expected in (("set_v1", 0.5), ("multiset_v1", 0.9)):
        assert (
            last_group_mean([1.0, 0.0], [True, True], counts, semantics=semantics).value == expected
        )
        assert last_group_mean([1.0, 0.0], [True, False], counts, semantics=semantics).value == 1
        absent = last_group_mean([1.0, 0.0], [False, False], counts, semantics=semantics)
        assert absent.value == 0 and not absent.mask
        assert last_group_max([1.0, 0.0], [True, True]).value == 1
