from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError, records
from coveo_recsys.contracts.records import Branch, CandidateResult, SourceEvidence
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, record_version, to_json


@pytest.mark.parametrize(
    "changes",
    [
        {"quotas": (1,)},
        {"max_candidates": 2},
        {"history_limit": 0},
        {"block_size": 0},
        {"tau_history_ms": float("nan")},
    ],
)
def test_retrieval_config_rejects_invalid_limits(changes):
    with pytest.raises(InputValidationError):
        replace(RetrievalConfig(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"skus": ("b", "b")},
        {"skus": ("b", 1)},
        {"skus": ("b", "")},
        {"skus": ("b", "\ud800")},
        {"skus": ("outside",)},
    ],
)
def test_candidate_evidence_is_separate_from_source_quota(changes):
    support = SourceEvidence("b", "detail_detail", "a", 0.25, 3, None, None)
    result = CandidateResult(
        "q", (), ("z", "b"), (("b", ("fill",)),), Branch.PRIMARY, (), (support,)
    )
    assert from_json(to_json(result), CandidateResult) == result
    assert result.skus == ("z", "b")
    assert record_version(CandidateResult) == "3"
    with pytest.raises(InputValidationError):
        replace(result, **changes)


def test_candidate_evidence_membership_reuses_one_sku_set(monkeypatch):
    built = []
    checked = []

    class TrackedSet(set):
        def __init__(self, values):
            super().__init__(values)
            built.append(values)

        def __contains__(self, value):
            checked.append(value)
            return super().__contains__(value)

    skus = ("z", "b", "a")
    evidence = tuple(SourceEvidence(sku, "popularity", None, 1.0, None, None, None) for sku in skus)
    monkeypatch.setattr(records, "set", TrackedSet, raising=False)
    result = CandidateResult("q", (), skus, (), Branch.FALLBACK, (), evidence)
    assert result.skus is skus
    assert result.evidence is evidence
    assert built == [skus]
    assert checked == list(skus)


@pytest.mark.parametrize("changes", [{"support": -1}, {"score": float("inf")}, {"source": ""}])
def test_evidence_rejects_invalid_values(changes):
    with pytest.raises(InputValidationError):
        replace(SourceEvidence("b", "detail_detail", "a", 0.25, 3, None, None), **changes)
