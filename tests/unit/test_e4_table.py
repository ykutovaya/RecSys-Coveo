"""Соединение кандидатов с зарегистрированными целями без изменения состава."""

from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts.common import InputValidationError, OutcomeStatus
from coveo_recsys.contracts.records import Branch, CandidateResult, FeatureBatch
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.ranking import table as table_module
from coveo_recsys.ranking.table import build_training_table
from tests.reference.e3_cases import request

SCHEMA = "d" * 64
NUMERIC = ("value", "present")
CATEGORICAL = ("category",)
LIMITS = {
    "feature_schema_sha256": SCHEMA,
    "numeric_columns": NUMERIC,
    "categorical_columns": CATEGORICAL,
    "max_candidates": 8,
    "max_rows": 1000,
    "max_memory_bytes": 10_000_000,
    "max_disk_bytes": 10_000_000,
}


def example(session="s", skus=("a", "b"), purchases=("a",), *, time=2000):
    original = request(rows=((time, "detail", "anchor"),))
    blocks = tuple(
        replace(
            block,
            session_id=session,
            events=tuple(replace(event, session_id=session) for event in block.events),
        )
        for block in original.prefix.blocks
    )
    prepared = replace(
        original,
        request_id=f"q-{session}",
        session_id=session,
        prefix=replace(original.prefix, session_id=session, blocks=blocks),
    )
    candidates = CandidateResult(
        prepared.request_id,
        (),
        skus,
        tuple((sku, ("popularity",)) for sku in skus),
        Branch.PRIMARY,
        (),
    )
    features = FeatureBatch(
        prepared.request_id,
        skus,
        NUMERIC,
        CATEGORICAL,
        tuple((float(i), float(i % 2)) for i in range(len(skus))),
        tuple((i % 3,) for i in range(len(skus))),
        tuple((bool(i % 2), True, True) for i in range(len(skus))),
        SCHEMA,
    )
    target = TargetRecord(
        prepared.request_id,
        purchases,
        OutcomeStatus.KNOWN if purchases else OutcomeStatus.UNKNOWN,
        1000,
        None if purchases else "no_registered_purchase",
    )
    return prepared, candidates, features, target


def build(examples, **overrides):
    return build_training_table(examples, **(LIMITS | overrides))


def test_table_orders_whole_groups_without_adding_targets_or_changing_feature_masks():
    examples = [
        example("я", ("a", "b", "я"), ("b", "я", "outside")),
        example("z", ("a", "b"), ("a",)),
    ]
    actual = build(iter(examples))
    assert actual.group_keys == (("z", 2000, "q-z"), ("я", 2000, "q-я"))
    assert actual.group_sizes == (2, 3)
    assert actual.skus == ("a", "b", "a", "b", "я")
    assert actual.labels.tolist() == [1, 0, 0, 1, 1]
    np.testing.assert_array_equal(
        actual.data, [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1], [2, 0, 2]]
    )
    assert actual.data.dtype == np.float64 and actual.data.flags.c_contiguous
    assert actual.feature_schema_sha256 == SCHEMA
    assert actual.numeric_columns == NUMERIC and actual.categorical_columns == CATEGORICAL
    assert [record.labels for record in actual.supervision] == [(1, 0), (0, 1, 1)]
    assert [record.pair_weights for record in actual.supervision] == [(1.0,), (0.5, 0.5)]
    assert all(
        record.group_id == key[0]
        and record.request_id == key[2]
        and record.session_request_count == 1
        and record.session_weight == 1.0
        for record, key in zip(actual.supervision, actual.group_keys, strict=True)
    )
    assert actual.statistics["target_skus_outside_candidates"] == 1


def test_unknown_targets_and_empty_candidates_do_not_discard_nonempty_zero_pair_groups():
    actual = build(
        [
            example("unknown", purchases=()),
            example("empty", skus=()),
            example("all-negative", purchases=("outside",)),
            example("all-positive", purchases=("a", "b")),
        ]
    )
    assert actual.group_sizes == (2, 2)
    assert actual.labels.tolist() == [0, 0, 1, 1]
    expected = {
        "requests": 4,
        "target_sessions": 3,
        "unknown_sessions": 1,
        "empty_candidate_groups": 1,
        "zero_pair_groups": 2,
        "groups": 2,
        "rows": 4,
        "positive_rows": 2,
        "negative_rows": 2,
        "pair_count": 0,
        "target_skus_outside_candidates": 2,
        "row_upper_bound": 24,
    }
    assert {key: actual.statistics[key] for key in expected} == expected
    assert all(record.pair_weights == () for record in actual.supervision)


@pytest.mark.parametrize("examples", [[], [example(purchases=())], [example(skus=())]])
def test_no_training_rows_returns_typed_empty_table(examples):
    actual = build(examples)
    assert actual.data.shape == (0, 3)
    assert actual.labels.shape == (0,)
    assert actual.group_sizes == actual.group_keys == actual.skus == actual.supervision == ()
    assert actual.statistics["pair_count"] == 0


@pytest.mark.parametrize("component", [1, 2, 3])
def test_mismatched_request_keys_are_rejected_even_for_unknown_target(component):
    parts = list(example(purchases=()))
    parts[component] = replace(parts[component], request_id="another")
    with pytest.raises(InputValidationError, match="запрос"):
        build([tuple(parts)])


def test_absent_target_is_not_interpreted_as_unknown():
    prepared, candidates, features, _ = example()
    with pytest.raises(InputValidationError, match="цел|запис"):
        build([(prepared, candidates, features, None)])


@pytest.mark.parametrize("duplicate", ["request", "session"])
def test_repeated_requests_or_sessions_are_rejected(duplicate):
    first = example("a")
    second = list(example("b" if duplicate == "request" else "a"))
    request_id = first[0].request_id if duplicate == "request" else "second-request"
    second = [replace(item, request_id=request_id) for item in second]
    with pytest.raises(InputValidationError, match="Повтор"):
        build([first, tuple(second)])


@pytest.mark.parametrize("damage", ["fallback", "order", "sku", "schema", "columns", "category"])
def test_incompatible_candidate_feature_records_are_rejected(damage):
    prepared, candidates, features, target = example()
    if damage == "fallback":
        candidates = replace(candidates, branch=Branch.FALLBACK)
    elif damage == "order":
        candidates = replace(candidates, skus=tuple(reversed(candidates.skus)))
        features = replace(features, skus=candidates.skus)
    elif damage == "sku":
        features = replace(features, skus=("a", "c"))
    elif damage == "schema":
        features = replace(features, feature_schema_sha256="e" * 64)
    elif damage == "columns":
        features = replace(features, numeric_columns=tuple(reversed(NUMERIC)))
    else:
        features = replace(features, categorical=((-1,), (2,)))
    with pytest.raises(InputValidationError):
        build([(prepared, candidates, features, target)])


@pytest.mark.parametrize("limit", ["max_rows", "max_memory_bytes", "max_disk_bytes"])
def test_resource_estimates_are_checked_before_feature_materialization(monkeypatch, limit):
    data = [example(), example("other")]
    accepted = build(data)
    stat = {
        "max_rows": "row_upper_bound",
        "max_memory_bytes": "estimated_memory_bytes",
        "max_disk_bytes": "estimated_disk_bytes",
    }[limit]
    boundary = accepted.statistics[stat]
    assert build(data, **{limit: boundary}).statistics == accepted.statistics
    monkeypatch.setattr(table_module, "feature_matrix", lambda *args: pytest.fail("allocation"))
    with pytest.raises(InputValidationError, match="ресурс|строк|бюджет"):
        build(data, **{limit: boundary - 1})


def test_candidate_limit_does_not_truncate_input():
    with pytest.raises(InputValidationError, match="кандидат"):
        build([example()], max_candidates=1)


@pytest.mark.parametrize("value", [0, -1, True, 2**63])
def test_resource_limits_require_positive_int64(value):
    with pytest.raises(InputValidationError):
        build([], max_rows=value)
