"""Э5-03: знаменатели, разложение, разрезы и сохранённые источники."""

import tracemalloc
import weakref
from copy import deepcopy
from dataclasses import replace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts.common import InputValidationError, OutcomeStatus
from coveo_recsys.contracts.records import (
    Branch,
    CandidateResult,
    DeliveryStatus,
    RequestResult,
    SourceResult,
)
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.evaluation import metrics as metrics_module
from coveo_recsys.evaluation.metrics import (
    aggregate_metrics,
    build_sku_slices,
    evaluate_session,
    paired_metrics,
)
from tests.reference.e3_cases import package, request


def result(req, skus=(), delivered=(), *, origins=None, sources=(), branch=Branch.PRIMARY):
    candidate = CandidateResult(
        req.request_id,
        sources,
        tuple(skus),
        tuple(
            (sku, tuple(names)) for sku, names in (origins or dict.fromkeys(skus, ("a",))).items()
        ),
        branch,
        (),
    )
    return RequestResult(
        req.request_id,
        "package",
        candidate,
        tuple(delivered),
        (1.0,) * len(delivered),
        (),
        DeliveryStatus.DELIVERED,
        (),
        3,
        "",
    )


def target(req, purchases=("b",)):
    return TargetRecord(
        req.request_id,
        tuple(purchases),
        OutcomeStatus.KNOWN if purchases else OutcomeStatus.UNKNOWN,
        10_000,
        None if purchases else "no_registered_purchase",
    )


def evaluate(req, response, purchases=("b",), **kwargs):
    return evaluate_session(
        req, response, target(req, purchases), catalog_skus=("a", "b", "c", "я"), **kwargs
    )


def rename(req, name):
    blocks = tuple(
        replace(
            block,
            session_id=name,
            events=tuple(replace(event, session_id=name) for event in block.events),
        )
        for block in req.prefix.blocks
    )
    return replace(
        req,
        request_id="q-" + name,
        session_id=name,
        prefix=replace(req.prefix, session_id=name, blocks=blocks),
    )


def test_b07_three_losses_and_hit_keep_same_denominator():
    rows = []
    for i, (purchases, candidates, delivered) in enumerate(
        [
            (("outside",), (), ()),
            (("b",), (), ()),
            (("b",), ("b",), ()),
            (("b", "c"), ("b", "c"), ("b", "c")),
        ]
    ):
        req = rename(request(), str(i))
        rows.append(evaluate(req, result(req, candidates, delivered), purchases))
    report = aggregate_metrics(rows, n_eligible=8, selected_size=5, exclusions={"late": 2})
    assert report["counts"] == {"N_eligible": 8, "N_obs": 4, "N": 4, "N_unknown": 0}
    assert report["coverage_chain"] == {"V": 3, "U": 2, "Hit": 1}
    assert report["metrics"]["Hit@20"]["value"] == 0.25
    assert report["metrics"]["candidate_coverage_available"]["value"] == 2 / 3
    assert {name: row["value"] for name, row in report["losses"].items()} == {
        "unavailable": 0.25,
        "retrieval": 0.25,
        "ranking_delivery": 0.25,
    }


@pytest.mark.parametrize(
    "branch,delivery",
    [
        (Branch.PRIMARY, DeliveryStatus.FAILED),
        (Branch.NONE, DeliveryStatus.NOT_ATTEMPTED),
        (Branch.FALLBACK, DeliveryStatus.DELIVERED),
    ],
)
def test_refusal_or_fallback_does_not_remove_positive_session(branch, delivery):
    req = request()
    candidates = ("b",) if branch == Branch.PRIMARY else ()
    response = replace(result(req, candidates, branch=branch), delivery=delivery)
    report = aggregate_metrics([evaluate(req, response)], n_eligible=1)
    assert report["metrics"]["Hit@20"]["denominator"] == 1
    assert report["metrics"]["Hit@20"]["value"] == 0.0
    assert report["coverage_chain"]["U"] == int(bool(candidates))


def test_unknown_target_is_not_a_miss_but_cost_is_retained():
    req = request()
    row = evaluate(req, result(req), ())
    evaluation, details = row
    assert (evaluation.b_s, evaluation.v_s, evaluation.u_s, evaluation.hit_s) == (None,) * 4
    assert details["recall"] is None and details["ndcg"] is None
    report = aggregate_metrics([row], n_eligible=1)
    assert report["counts"]["N_unknown"] == 1
    assert report["cost"]["request_count"] == 1
    assert report["cost"]["elapsed_ms_total"] == 3
    assert report["metrics"]["Hit@20"]["value"] is None
    assert report["metrics"]["Hit@20"]["status"] == "insufficient"
    assert report["metrics"]["Hit@20"]["reason"] == "no_registered_targets"


def test_catalog_absence_and_request_ineligibility_are_separate_purchase_rows():
    req = request(((1000, "purchase", "b"), (2000, "detail", "a")))
    evaluation, details = evaluate(req, result(req), ("b", "outside"))
    assert evaluation.v_s == 0
    assert {row["sku"]: row["availability"] for row in details["purchases"]} == {
        "b": "ineligible_at_request",
        "outside": "outside_historical_catalog",
    }
    report = aggregate_metrics([(evaluation, details)], n_eligible=1)
    conditional = report["metrics"]["candidate_coverage_available"]
    assert conditional["value"] is None and conditional["reason"] == "no_available_target"


def test_recorded_failures_and_feature_masks_include_unknown_targets():
    req = request()
    source = SourceResult("image", (), (), False, 2, "history_vector_unavailable")
    response = replace(
        result(req, sources=(source,), branch=Branch.FALLBACK),
        reasons=("model_unavailable", "deadline_exceeded"),
    )
    row = evaluate(
        req, response, (), feature_mask_counts={"image": 2, "description": 1}, feature_rows=3
    )
    report = aggregate_metrics([row], n_eligible=1)
    cost = report["cost"]
    assert cost["fallback_count"] == 1
    assert cost["source_unavailable_count"] == 1
    assert cost["recorded_deadline_exceeded_count"] == 1
    assert cost["feature_masks"]["measured_requests"] == 1
    assert cost["feature_masks"]["columns"]["image"] == {"masked": 2, "rows": 3, "fraction": 2 / 3}


@pytest.mark.parametrize("reason", ["not_measured", "not_applicable"])
def test_absent_feature_masks_preserve_the_reason(reason):
    req = request()
    report = aggregate_metrics([evaluate(req, result(req), mask_reason=reason)], n_eligible=1)
    masks = report["cost"]["feature_masks"]
    assert masks["masked_values"] is None
    assert masks["missing_reasons"] == {reason: 1}


def test_target_cart_groups_restrict_hits_to_their_own_purchases():
    req = request(((1000, "add", "b"), (2000, "detail", "a")), query=(1.0, 0.0), search_time=1500)
    evaluation, details = evaluate(
        req, result(req, ("b", "c"), ("c",)), ("b", "c"), sku_slices=build_sku_slices(package())
    )
    assert {"cart=present", "search=present", "history=1"} <= set(evaluation.slices)
    assert details["slice_metrics"]["target_cart=inside"]["hit"] == 0
    assert details["slice_metrics"]["target_cart=outside"]["hit"] == 1
    assert "target_known=known" in evaluation.slices
    assert "target_description=present" in evaluation.slices


def test_history_slices_count_available_history_not_only_last_block():
    req = request(((1000, "detail", "a"), (2000, "detail", "b")))
    evaluation, _ = evaluate(req, result(req))
    assert "history=2-5" in evaluation.slices


@pytest.mark.parametrize("views", [20, 21, 25])
@pytest.mark.parametrize("semantics", ["set_v1", "multiset_v1"])
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_history_slice_uses_all_detail_counts_not_bounded_tail(views, semantics, multiplicity):
    rows = [(1000 + i, "detail", "a") for i in range(views) for _ in range(multiplicity)]
    # Корзина, покупки и поиск не увеличивают число просмотров.
    rows.extend((2000 + i, kind, "c") for i in range(30) for kind in ("add", "remove", "purchase"))
    req = request(rows, semantics=semantics, search_time=2100, query=(1.0, 0.0))
    size = views * (multiplicity if semantics == "multiset_v1" else 1)
    assert sum(item.count for item in req.prefix.detail_history) == 20
    evaluation, _ = evaluate(req, result(req, ("b",), ("b",)))
    assert f"history={'21+' if size > 20 else '6-20'}" in evaluation.slices
    assert (evaluation.v_s, evaluation.u_s, evaluation.hit_s) == (1, 1, 1)


def test_popularity_terciles_and_modality_slices_use_historical_snapshot():
    labels = build_sku_slices(package())
    assert "target_popularity=high" in labels["a"]
    assert "target_popularity=middle" in labels["b"]
    assert "target_popularity=low" in labels["c"]
    assert {"target_history=unseen", "target_description=absent", "target_image=present"} <= set(
        labels["я"]
    )


def test_source_removal_preserves_multiple_origins_and_other_target_without_refill():
    req = request()
    response = result(req, ("b", "c"), ("b",), origins={"b": ("a", "b"), "c": ("a",)})
    _, details = evaluate(req, response, ("b", "c"))
    assert details["sources"]["a"]["delta_fixed"] == 0
    assert details["sources"]["b"]["delta_fixed"] == 0
    assert details["sources"]["a"]["unique_candidate_skus"] == ["c"]
    only = result(req, ("c",), (), origins={"c": ("a",)})
    assert evaluate(req, only, ("b", "c"))[1]["sources"]["a"]["delta_fixed"] == 1


def test_absent_source_has_zero_fixed_contribution_with_complete_origins():
    rows = []
    for name in ("a", "b"):
        req = rename(request(), name)
        response = result(req, ("b",), ("b",), origins={"b": (name,)})
        rows.append(evaluate(req, response))
    report = aggregate_metrics(rows, n_eligible=2)
    assert report["sources"]["a"]["delta_fixed"]["value"] == 0.5
    assert report["sources"]["b"]["unique_candidate_coverage"]["value"] == 0.5


def test_unmeasured_prequota_and_empty_union_have_distinct_reasons():
    req = request()
    sources = tuple(SourceResult(name, (), (), True, 2, None) for name in ("a", "b"))
    response = result(req, sources=sources)
    unknown = evaluate(req, response)
    measured = evaluate(
        req,
        response,
        source_lists={
            name: {"skus": (), "depth": 4, "truncated": False, "reason": None}
            for name in ("a", "b")
        },
    )
    assert unknown[1]["overlaps"]["a|b"]["reason"] == "not_measured"
    assert measured[1]["overlaps"]["a|b"]["reason"] == "empty_union"
    report = aggregate_metrics([unknown], n_eligible=1)
    before = report["sources"]["a"]["coverage_before_quota"]
    assert before["value"] is None and before["reason"] == "not_measured"
    assert report["sources"]["a"]["coverage_after_quota"]["value"] == 0.0


def test_bounded_prequota_coverage_and_overlap_are_not_sum_of_sources():
    req = request()
    sources = tuple(SourceResult(name, ("b",), (1.0,), True, 1, None) for name in ("a", "b"))
    response = result(req, ("b",), ("b",), origins={"b": ("a", "b")}, sources=sources)
    row = evaluate(
        req,
        response,
        source_lists={
            "a": {"skus": ("b", "c"), "depth": 2, "truncated": True, "reason": None},
            "b": {"skus": ("b",), "depth": 2, "truncated": False, "reason": None},
        },
    )
    report = aggregate_metrics([row], n_eligible=1)
    assert report["overlaps"]["a|b"]["value"] == 0.5
    assert report["overlaps"]["a|b"]["missing_fraction"] == 0.0
    assert report["sources"]["a"]["full_coverage_before_quota"]["reason"] == "truncated"
    assert report["sources"]["a"]["coverage_before_quota"]["value"] == 1.0
    assert report["metrics"]["candidate_coverage"]["value"] == 1.0


def test_overlap_uses_equal_session_weights_and_reports_missing_separately():
    rows = []
    for index, before in enumerate(
        [{"a": ("b", "c"), "b": ("b",)}, {"a": ("b",), "b": ("b",)}, None]
    ):
        req = rename(request(), str(index))
        sources = tuple(SourceResult(name, ("b",), (1.0,), True, 1, None) for name in ("a", "b"))
        response = result(req, ("b",), ("b",), origins={"b": ("a", "b")}, sources=sources)
        measured = (
            None
            if before is None
            else {
                name: {"skus": skus, "depth": 2, "truncated": False, "reason": None}
                for name, skus in before.items()
            }
        )
        rows.append(evaluate(req, response, source_lists=measured))
    report = aggregate_metrics(rows, n_eligible=3)
    assert report["overlaps"]["a|b"]["value"] == 0.75
    assert report["overlaps"]["a|b"]["missing_fraction"] == 1 / 3
    assert report["sources"]["a"]["coverage_before_quota"]["value"] is None


def test_pair_requires_identical_sessions_and_positive_status():
    req = request()
    left = evaluate(req, result(req, ("b",), ("b",)))
    right = evaluate(req, result(req))
    assert paired_metrics([left], [right])["value"] == 1.0
    with pytest.raises(InputValidationError):
        paired_metrics([left], [evaluate(req, result(req), ())])
    with pytest.raises(InputValidationError):
        paired_metrics([left], [])


@pytest.mark.parametrize(
    "invalid",
    [
        "foreign_target",
        "ineligible_candidate",
        "duplicate_session",
        "ineligible_source",
        "none_branch",
        "inconsistent_origins",
    ],
)
def test_inconsistent_saved_inputs_fail_instead_of_repair(invalid):
    req = request()
    response = result(req)
    with pytest.raises(InputValidationError):
        if invalid == "foreign_target":
            evaluate_session(
                req, response, replace(target(req), request_id="other"), catalog_skus=()
            )
        elif invalid == "ineligible_candidate":
            evaluate(req, result(req, ("outside",), ()))
        elif invalid == "ineligible_source":
            evaluate(
                req, result(req, sources=(SourceResult("a", ("outside",), (1.0,), True, 1, None),))
            )
        elif invalid == "none_branch":
            evaluate(req, result(req, ("b",), branch=Branch.NONE))
        elif invalid == "inconsistent_origins":
            evaluate(
                req,
                result(
                    req,
                    ("b",),
                    origins={"b": ("other",)},
                    sources=(SourceResult("a", ("b",), (1.0,), True, 1, None),),
                ),
            )
        else:
            row = evaluate(req, response)
            aggregate_metrics([row, row], n_eligible=2)


def aggregate_example(size=24):
    """Разнородная сохранённая диагностика для побайтового сравнения старого отчёта."""
    rows = []
    for index in range(size):
        req = rename(request(), f"saved-{index}")
        names = ((), ("a",), ("a", "b"), ("b", "c"), ("a", "b", "c"))[index % 5]
        sources = tuple(
            SourceResult(
                name,
                ("b", "c")[: (index + offset) % 3],
                (1.0, 0.5)[: (index + offset) % 3],
                True,
                2,
                None,
            )
            for offset, name in enumerate(names)
        )
        origins = {
            sku: tuple(source.source for source in sources if sku in source.skus)
            for sku in ("b", "c")
            if any(sku in source.skus for source in sources)
        }
        response = replace(
            result(
                req,
                tuple(origins),
                tuple(reversed(origins))[: index % 3],
                origins=origins,
                sources=sources,
            ),
            elapsed_ms=(0, 1, 31, 3)[index % 4],
            reasons=("deadline_exceeded", "source:timeout") if index % 7 == 0 else (),
        )
        source_lists = {
            name: {
                "skus": () if index % 6 == 0 else ("b", "c", "я")[: (index + offset) % 4],
                "depth": 3 + index % 2,
                "truncated": index % 7 == 0,
                "reason": None,
            }
            for offset, name in enumerate(names)
            if (index + offset) % 4
        }
        rows.append(
            evaluate(
                req,
                response,
                ((), ("b",), ("b", "c"), ("outside",), ("c",))[index % 5],
                sku_slices=build_sku_slices(package()),
                source_lists=source_lists,
                feature_mask_counts={"image": index % 3, **({"text": 1} if index % 2 else {})}
                if index % 3
                else None,
                feature_rows=3 if index % 3 else None,
                mask_reason="not_applicable" if index % 2 else "not_measured",
            )
        )
    return rows


@pytest.mark.parametrize(
    "size,digest",
    [
        (0, "3a81849a688bb5fcdd1baf748e29b7a8a9440e31067139c8e28fdbf49bf2f557"),
        (1, "75a23d056159f974e0ddcc9efd42ee861e0c902051fbfa466a881820b7a6ed68"),
        (24, "1718b0323d12ff2742eb7a623f42947b2ae7428578a73dc5a003b9b1325282fc"),
    ],
)
def test_aggregate_consumes_once_and_preserves_complete_legacy_report(size, digest):
    class Once:
        used = False

        def __iter__(self):
            assert not self.used, "Повторное чтение входного потока"
            self.used = True
            yield from aggregate_example(size)

    report = aggregate_metrics(Once(), n_eligible=30, selected_size=25, exclusions={"late": 2})
    assert sha256_bytes(canonical_json_bytes(report)) == digest


def test_streaming_aggregate_does_not_retain_large_source_lists():
    evaluation, template = aggregate_example(3)[2]
    size = 128

    def rows():
        for index in range(size):
            details = deepcopy(template)
            details["session_id"] = f"memory-{index}"
            before = details["sources"]["a"]["before"]
            before["skus"] = ["b", *(f"candidate-{index}-{sku}" for sku in range(4096))]
            before["size"] = before["depth"] = len(before["skus"])
            yield replace(evaluation, session_id=details["session_id"]), details

    tracemalloc.start()
    try:
        report = aggregate_metrics(rows(), n_eligible=size)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert report["counts"]["N_obs"] == size
    assert report["sources"]["a"]["size_before_mean"] == 4097
    assert peak < 8 * 1024**2, "Удерживаются подробные списки прошлых сессий"


@pytest.mark.parametrize("invalid", ["duplicate", "k", "n_eligible", "selected_size"])
def test_streaming_aggregate_preserves_input_rejections(invalid):
    row = aggregate_example(2)[1]
    values = (row, row) if invalid == "duplicate" else (row,)
    with pytest.raises(InputValidationError):
        aggregate_metrics(
            iter(values),
            n_eligible=0 if invalid == "n_eligible" else 2,
            selected_size=0 if invalid == "selected_size" else None,
            k=10 if invalid == "k" else 20,
        )


@pytest.mark.parametrize("set_type", [set, frozenset])
@pytest.mark.parametrize("purchased", [False, True])
def test_evaluate_reuses_catalog_sets_without_mutating_them(set_type, purchased, monkeypatch):
    req = request(
        ((1000, "purchase", "a"), (2000, "detail", "c")) if purchased else ((2000, "detail", "c"),)
    )
    source = SourceResult("a", ("b",), (1.0,), True, 2, None)
    response = result(req, ("b",), ("b",), sources=(source,))
    source_lists = {"a": {"skus": ("b", "c"), "depth": 2, "truncated": False, "reason": None}}
    catalog = set_type(("a", "b", "c", *(f"extra-{index}" for index in range(2048))))
    expected = evaluate_session(
        req,
        response,
        target(req),
        catalog_skus=tuple(catalog),
        source_lists=source_lists,
    )
    saved_catalog = set(catalog)
    saved_lists = deepcopy(source_lists)

    def without_full_set_copy(values=()):
        assert not isinstance(values, (set, frozenset)) or len(values) < 2048, (
            "Копия готового каталога/available"
        )
        return set(values)

    original_sources = metrics_module._sources

    def reuse_available(response, goals, available, before):
        if not purchased:
            assert available is catalog, "Каталог без запрещённых SKU скопирован"
        return original_sources(response, goals, available, before)

    monkeypatch.setattr(metrics_module, "set", without_full_set_copy, raising=False)
    monkeypatch.setattr(metrics_module, "_sources", reuse_available)
    observed = evaluate_session(
        req,
        response,
        target(req),
        catalog_skus=catalog,
        source_lists=source_lists,
    )
    assert observed == expected
    assert catalog == saved_catalog and source_lists == saved_lists
    assert req.prefix.purchased_skus == (("a",) if purchased else ())


def test_accumulators_interleave_without_retaining_details_and_finish_repeatably():
    class Details(dict):
        pass

    rows = aggregate_example()
    expected = aggregate_metrics(rows, n_eligible=30, selected_size=25, exclusions={"late": 2})
    accumulators = [
        metrics_module.MetricsAccumulator(n_eligible=30, selected_size=25, exclusions={"late": 2})
        for _ in range(3)
    ]
    for evaluation, details in rows:
        transient = Details(details)
        reference = weakref.ref(transient)
        for accumulator in accumulators:
            accumulator.add(evaluation, transient)
        del transient
        assert reference() is None, "Накопитель удерживает подробную запись"
    for accumulator in accumulators:
        assert accumulator.finish() == expected
        assert accumulator.finish() == expected
