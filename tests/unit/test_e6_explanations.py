from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import Branch
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.explanations.templates import TEMPLATE_SHA256, TEMPLATE_VERSION, explain
from coveo_recsys.retrieval import retrieve, retrieve_fallback
from coveo_recsys.retrieval.scores import RetrievalIndex, bind_context, score_request
from tests.reference.e3_cases import CATALOG_HASH, package, request


def inputs(prepared=None, *, maximum=3, **quotas):
    index = RetrievalIndex(package(), CATALOG_HASH)
    context = bind_context(request() if prepared is None else prepared, index)
    config = RetrievalConfig(
        quotas=tuple(quotas.get(source, 0) for source in SOURCES), max_candidates=maximum
    )
    scores = score_request(context, index, config)
    return index, scores, retrieve(context, index, config, scores=scores)


def graph_tie():
    historical = package()
    graph = historical.graphs.relations[0]
    graph = replace(graph, support=np.array([4, 7, 3], dtype=np.int64))
    historical = replace(
        historical,
        graphs=replace(historical.graphs, relations=(graph, *historical.graphs.relations[1:])),
    )
    index = RetrievalIndex(historical, CATALOG_HASH)
    context = bind_context(request(((2000, "detail", "a"), (2000, "detail", "b"))), index)
    config = RetrievalConfig(quotas=(1, 0, 0, 0, 0, 0, 0, 0), max_candidates=1)
    scores = score_request(context, index, config)
    target = index.positions["c"]
    scores.values["detail_detail"][:] = 0.0
    scores.values["detail_detail"][target] = 0.2
    scores.contributions["detail_detail"] = {"b": {target: 0.1}, "a": {target: 0.1}}
    return index, scores, retrieve(context, index, config, scores=scores)


def test_graph_explanation_uses_maximum_anchor_contribution_and_only_its_edge_support():
    index, scores, candidates = graph_tie()
    (result,) = explain(candidates, ("c",), package_id="package", scores=scores, index=index)
    assert candidates.evidence[0].anchor_sku == "a"
    assert result.sku == "c" and result.branch == Branch.PRIMARY
    assert result.package_id == "package" and result.template_id == "1:detail_detail"
    assert dict(result.support) == {
        "detail_detail.score": 0.2,
        "detail_detail.contribution": 0.1,
        "detail_detail.support": 7.0,
    }
    reordered = replace(candidates, evidence=tuple(reversed(candidates.evidence)))
    scores.contributions["detail_detail"] = dict(
        reversed(tuple(scores.contributions["detail_detail"].items()))
    )
    assert explain(reordered, ("c",), package_id="package", scores=scores, index=index) == (result,)


@pytest.mark.parametrize("change", [{"anchor_sku": "b"}, {"support": 11}, {"score": 0.1}])
def test_graph_explanation_rejects_inconsistent_trace_evidence(change):
    index, scores, candidates = graph_tie()
    candidates = replace(candidates, evidence=(replace(candidates.evidence[0], **change),))
    with pytest.raises(InputValidationError):
        explain(candidates, ("c",), package_id="package", scores=scores, index=index)


def test_content_and_search_explanations_preserve_measured_modalities_and_private_context():
    index, scores, candidates = inputs(
        request(query=(0.0, 1.0), search_time=2000), description=1, image=1, search=1
    )
    results = explain(candidates, ("b", "a"), package_id="package", scores=scores, index=index)
    assert [item.sku for item in results] == ["b", "a"]
    assert results[0].sources == ("search",)
    assert dict(results[0].support) == {"search.score": 1.0}
    assert results[1].sources == ("description", "image")
    assert dict(results[1].support) == {"description.score": 1.0, "image.score": 1.0}
    search = next(item for item in candidates.evidence if item.source == "search")
    assert search.context_id == "2000"
    assert search.space_version == "description:" + "b" * 64 + ":2"
    image = next(item for item in candidates.evidence if item.source == "image")
    assert image.space_version == "image:" + "b" * 64 + ":2"
    public = " ".join(item.public_text for item in results)
    assert "2000" not in public and "b" * 64 not in public and "1.0" not in public
    assert "совместим" not in public and "купите" not in public


@pytest.mark.parametrize("query", [None, (0.0, 0.0)])
def test_unusable_search_is_not_explained_as_an_origin(query):
    index, scores, candidates = inputs(request(query=query, search_time=2000), search=1)
    results = explain(candidates, candidates.skus, package_id="package", scores=scores, index=index)
    assert all(item.sources == ("fill",) for item in results)
    assert all("поиск" not in item.public_text for item in results)


def test_fill_and_fallback_explain_only_their_actual_sources_without_primary_scores():
    index, scores, primary = inputs(maximum=2)
    results = explain(primary, primary.skus, package_id="package", scores=None, index=index)
    assert all(item.sources == ("fill",) for item in results)
    assert dict(results[0].support) == {"fill.score": 9 / 21}
    reserve = retrieve_fallback(scores.context, index, scores.config)
    results = explain(reserve, reserve.skus, package_id="package", scores=None, index=index)
    assert [item.sources for item in results] == [("recent",), ("popularity",)]
    assert all(item.branch == Branch.FALLBACK for item in results)
    assert "граф" not in " ".join(item.public_text.lower() for item in results)


@pytest.mark.parametrize(
    "problem", ["missing", "unused", "unavailable", "context", "space", "branch", "duplicate"]
)
def test_explanation_refuses_missing_or_unrelated_evidence(problem):
    index, scores, candidates = inputs(request(query=(0.0, 1.0), search_time=2000), search=1)
    target = next(item.sku for item in candidates.evidence if item.source == "search")
    evidence = next(item for item in candidates.evidence if item.source == "search")
    if problem == "missing":
        candidates = replace(candidates, evidence=())
    elif problem == "unused":
        candidates = replace(
            candidates,
            origins=tuple(
                (sku, ("image",)) if sku == target else (sku, names)
                for sku, names in candidates.origins
            ),
        )
    elif problem == "unavailable":
        scores.masks["search"][index.positions[target]] = False
    elif problem in {"context", "space"}:
        field = "context_id" if problem == "context" else "space_version"
        candidates = replace(
            candidates,
            evidence=tuple(
                replace(item, **{field: "wrong"}) if item == evidence else item
                for item in candidates.evidence
            ),
        )
    elif problem == "branch":
        candidates = replace(candidates, branch=Branch.FALLBACK)
    else:
        candidates = replace(candidates, evidence=(*candidates.evidence, evidence))
    with pytest.raises(InputValidationError):
        explain(candidates, (target,), package_id="package", scores=scores, index=index)


@pytest.mark.parametrize(
    "delivered,version", [(("outside",), "1"), (("a", "a"), "1"), (("a",), "2")]
)
def test_explanations_bind_to_delivery_and_known_template_version(delivered, version):
    index, scores, candidates = inputs()
    with pytest.raises(InputValidationError):
        explain(
            candidates,
            delivered,
            package_id="package",
            scores=scores,
            index=index,
            template_version=version,
        )


def test_no_delivery_has_no_explanations_and_templates_have_stable_content_hash():
    from coveo_recsys.explanations import templates

    index, _, candidates = inputs()
    assert explain(candidates, (), package_id="package", scores=None, index=index) == ()
    assert TEMPLATE_VERSION == "1"
    assert TEMPLATE_SHA256 == sha256_bytes(
        canonical_json_bytes({"version": TEMPLATE_VERSION, "templates": dict(templates._TEXT)})
    )
