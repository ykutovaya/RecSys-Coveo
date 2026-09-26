from dataclasses import replace

import pytest

from coveo_recsys.contracts import InputValidationError
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


def test_candidate_evidence_is_separate_from_source_quota():
    support = SourceEvidence("b", "detail_detail", "a", 0.25, 3, None, None)
    result = CandidateResult("q", (), ("b",), (("b", ("fill",)),), Branch.PRIMARY, (), (support,))
    assert from_json(to_json(result), CandidateResult) == result
    assert record_version(CandidateResult) == "3"
    with pytest.raises(InputValidationError):
        replace(result, evidence=(replace(support, sku="outside"),))


@pytest.mark.parametrize("changes", [{"support": -1}, {"score": float("inf")}, {"source": ""}])
def test_evidence_rejects_invalid_values(changes):
    with pytest.raises(InputValidationError):
        replace(SourceEvidence("b", "detail_detail", "a", 0.25, 3, None, None), **changes)
