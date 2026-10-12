"""Чистая оценка сохранённой выдачи с отдельно присоединённой целью.

Неизвестный исход не становится промахом. Диагностика источников использует
только сохранённые списки; удаление происхождения не выполняет новый отбор.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict
from itertools import combinations
from typing import Any

from coveo_recsys.contracts.common import InputValidationError, OutcomeStatus, id_bytes
from coveo_recsys.contracts.historical import HistoricalPackage
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.records import Branch, CartState, DeliveryStatus, RequestResult
from coveo_recsys.contracts.training import MetricResult, SessionEvaluation, TargetRecord

EvaluationRow = tuple[SessionEvaluation, dict[str, Any]]
LOSS_NAMES = ("unavailable", "retrieval", "ranking_delivery")


def build_sku_slices(package: HistoricalPackage) -> dict[str, tuple[str, ...]]:
    """Разрезы снимка A_G: терцили числа просмотров, история и маски векторов.

    Пороги — nearest-rank квантили 1/3 и 2/3 по SKU A_G; совпадающие
    частоты остаются в одной группе. Цели периода оценки здесь недоступны.
    """
    counts = sorted(int(value) for value in package.statistics.detail_counts)
    if not counts:
        return {}
    low = counts[math.ceil(len(counts) / 3) - 1]
    high = counts[math.ceil(2 * len(counts) / 3) - 1]
    slices: dict[str, tuple[str, ...]] = {}
    for index, sku in enumerate(package.catalog.skus):
        count = int(package.statistics.detail_counts[index])
        group = "low" if count <= low else "middle" if count <= high else "high"
        slices[sku] = (
            f"target_popularity={group}",
            f"target_history={'seen' if count else 'unseen'}",
            "target_description="
            + ("present" if package.content.description.mask[index] else "absent"),
            "target_image=" + ("present" if package.content.image.mask[index] else "absent"),
        )
    return slices


def _quality(
    purchases: Collection[str],
    available: Collection[str],
    candidates: Collection[str],
    delivered: Sequence[str],
    k: int,
) -> dict[str, Any]:
    goal = set(purchases)
    if not goal:
        return dict.fromkeys(("v", "u", "hit", "recall", "ndcg"))
    shown = delivered[:k]
    dcg = math.fsum(1 / math.log2(i + 2) for i, sku in enumerate(shown) if sku in goal)
    ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(k, len(goal))))
    return {
        "v": int(bool(goal.intersection(available))),
        "u": int(bool(goal.intersection(candidates))),
        "hit": int(bool(goal.intersection(shown))),
        "recall": len(goal.intersection(shown)) / len(goal),
        "ndcg": dcg / ideal,
    }


def _request_slices(request: PreparedRequest, catalog: Collection[str]) -> list[str]:
    prefix = request.prefix
    size = sum(count for kind, _, count in prefix.counters if kind == "detail")
    history = (
        "0"
        if size == 0
        else "1"
        if size == 1
        else "2-5"
        if size <= 5
        else "6-20"
        if size <= 20
        else "21+"
    )
    prefix_skus = {sku for _, sku, _ in prefix.counters if sku is not None}
    known = prefix_skus.intersection(catalog)
    return [
        f"history={history}",
        f"search={'present' if prefix.search is not None else 'absent'}",
        f"cart={'present' if any(item.state == CartState.ADD for item in prefix.cart) else 'absent'}",
        "cart_state="
        + (
            "ambiguous"
            if any(item.state == CartState.UNKNOWN for item in prefix.cart)
            else "unambiguous"
        ),
        "prefix_known=" + ("all" if known == prefix_skus else "mixed" if known else "none"),
    ]


def _before_list(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None or value.get("skus") is None:
        return {
            "skus": None,
            "size": None,
            "depth": None,
            "truncated": None,
            "reason": "not_measured" if value is None else value.get("reason") or "not_measured",
        }
    skus = tuple(value["skus"])
    depth, truncated = value.get("depth"), value.get("truncated")
    if (
        len(skus) != len(set(skus))
        or type(depth) is not int
        or depth < len(skus)
        or type(truncated) is not bool
    ):
        raise InputValidationError("Измеренный список требует глубину, усечение и уникальные SKU")
    for sku in skus:
        id_bytes(sku)
    return {
        "skus": list(skus),
        "size": len(skus),
        "depth": depth,
        "truncated": truncated,
        "reason": None,
    }


def _sources(
    result: RequestResult,
    purchases: Collection[str],
    available: Collection[str],
    source_lists: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = available if isinstance(available, AbstractSet) else set(available)
    candidates = result.candidates
    origins = {sku: set(names) for sku, names in candidates.origins}
    if (
        len(origins) != len(candidates.origins)
        or set(origins) != set(candidates.skus)
        or any(not names for names in origins.values())
    ):
        raise InputValidationError("Происхождения должны покрывать кандидатов ровно по одному разу")
    after = {source.source: source for source in candidates.sources}
    if len(after) != len(candidates.sources):
        raise InputValidationError("Повтор источника в сохранённом отборе")
    before = source_lists or {}
    names = sorted(
        set(after) | set(before) | {name for values in origins.values() for name in values}
    )
    goal = set(purchases)
    covered = bool(goal.intersection(candidates.skus))
    rows: dict[str, Any] = {}
    for name in names:
        raw = _before_list(before.get(name))
        if raw["skus"] is not None and not set(raw["skus"]) <= allowed:
            raise InputValidationError("Диагностический список содержит недопустимый товар")
        source = after.get(name)
        selected = (
            list(source.skus)
            if source is not None
            else [sku for sku, values in origins.items() if name in values]
        )
        if not set(selected) <= allowed:
            raise InputValidationError("Список источника содержит недопустимый товар")
        if source is not None and any(name not in origins.get(sku, set()) for sku in selected):
            raise InputValidationError("Список после квоты не соответствует происхождениям")
        # Происхождение fill является измеренным результатом заполнения.
        after_known = source is not None or any(name in values for values in origins.values())
        unique = sorted((sku for sku, values in origins.items() if values == {name}), key=id_bytes)
        kept = {sku for sku, values in origins.items() if values - {name}}
        before_hit = (
            int(bool(goal.intersection(raw["skus"]))) if goal and raw["skus"] is not None else None
        )
        after_hit = int(bool(goal.intersection(selected))) if goal and after_known else None
        rows[name] = {
            "before": raw,
            "after": {
                "skus": selected if after_known else None,
                "size": len(selected) if after_known else None,
                "quota": source.quota if source is not None else None,
                "available": source.available
                if source is not None
                else True
                if after_known
                else None,
                "reason": source.reason
                if source is not None
                else None
                if after_known
                else "not_measured",
            },
            "hit_before": before_hit,
            "hit_after": after_hit,
            "quota_loss": int(bool(before_hit) and not after_hit)
            if before_hit is not None and after_hit is not None
            else None,
            "unique_candidate_skus": unique,
            "unique_candidate_hit": int(bool(goal.intersection(unique))) if goal else None,
            "delta_fixed": int(covered) - int(bool(goal.intersection(kept))) if goal else None,
        }
    complete = all(row["before"]["skus"] is not None for row in rows.values())
    for name, row in rows.items():
        others = {
            sku
            for other, other_row in rows.items()
            if other != name
            for sku in (other_row["before"]["skus"] or ())
        }
        unique_before = (
            sorted(set(row["before"]["skus"] or ()) - others, key=id_bytes) if complete else None
        )
        row["unique_before_skus"] = unique_before
        row["unique_before_hit"] = (
            int(bool(goal.intersection(unique_before)))
            if goal and unique_before is not None
            else None
        )
    overlaps: dict[str, Any] = {}
    for first, second in combinations(names, 2):
        left, right = rows[first]["before"]["skus"], rows[second]["before"]["skus"]
        if left is None or right is None:
            overlaps[first + "|" + second] = {"value": None, "reason": "not_measured"}
        else:
            union = set(left) | set(right)
            overlaps[first + "|" + second] = {
                "value": len(set(left) & set(right)) / len(union) if union else None,
                "reason": None if union else "empty_union",
            }
    return rows, overlaps


def evaluate_session(
    request: PreparedRequest,
    result: RequestResult,
    target: TargetRecord,
    *,
    catalog_skus: Collection[str],
    calendar_cell: int = 0,
    k: int = 20,
    sku_slices: Mapping[str, Sequence[str]] | None = None,
    source_lists: Mapping[str, Mapping[str, Any]] | None = None,
    feature_mask_counts: Mapping[str, int] | None = None,
    feature_rows: int | None = None,
    mask_reason: str | None = "not_measured",
) -> EvaluationRow:
    """Оценить единственный запрос; детализация не передаётся рекомендательному пути."""
    if (
        request.request_id != result.request_id
        or request.request_id != target.request_id
        or type(k) is not int
        or k <= 0
        or type(calendar_cell) is not int
        or calendar_cell < 0
    ):
        raise InputValidationError("Несогласованные ID или параметры оценки")
    catalog = catalog_skus if isinstance(catalog_skus, AbstractSet) else set(catalog_skus)
    available = (
        catalog - set(request.prefix.purchased_skus) if request.prefix.purchased_skus else catalog
    )
    if not set(result.candidates.skus) <= available or len(result.delivered_skus) > k:
        raise InputValidationError("Сохранённая выдача нарушает допустимость или K")
    if result.candidates.branch == Branch.NONE and result.candidates.skus:
        raise InputValidationError("До выбора ветви кандидаты должны быть пусты")
    if feature_mask_counts is not None:
        if (
            type(feature_rows) is not int
            or feature_rows < 0
            or any(
                type(count) is not int or not 0 <= count <= feature_rows
                for count in feature_mask_counts.values()
            )
        ):
            raise InputValidationError(
                "Маски требуют число строк и счётчики отсутствующих значений"
            )
    elif feature_rows is not None or not mask_reason:
        raise InputValidationError("Неизмеренные маски требуют причину и неизвестное число строк")
    known = target.status == OutcomeStatus.KNOWN
    quality = _quality(
        target.purchases, available, result.candidates.skus, result.delivered_skus, k
    )
    v, u, hit = quality["v"], quality["u"], quality["hit"]
    observed_cart = {item.sku for item in request.prefix.cart if item.state == CartState.ADD}
    request_slices = _request_slices(request, catalog)
    slice_metrics = {name: dict(quality) for name in request_slices}
    purchase_rows = []
    grouped_targets: dict[str, set[str]] = {}
    for sku in target.purchases:
        availability = (
            "outside_historical_catalog"
            if sku not in catalog
            else "available"
            if sku in available
            else "ineligible_at_request"
        )
        labels = [
            f"target_known={'known' if sku in catalog else 'unknown'}",
            f"target_cart={'inside' if sku in observed_cart else 'outside'}",
        ]
        labels.extend((sku_slices or {}).get(sku, ()))
        if sku not in catalog:
            labels.extend(
                (
                    "target_popularity=unknown",
                    "target_history=unknown",
                    "target_description=unknown",
                    "target_image=unknown",
                )
            )
        for label in labels:
            grouped_targets.setdefault(label, set()).add(sku)
        purchase_rows.append(
            {
                "sku": sku,
                "availability": availability,
                "in_observed_cart": sku in observed_cart,
                "candidate": sku in result.candidates.skus,
                "delivered": sku in result.delivered_skus,
            }
        )
    for label, goal in grouped_targets.items():
        slice_metrics[label] = _quality(
            goal, available, result.candidates.skus, result.delivered_skus, k
        )
    sources, overlaps = _sources(result, target.purchases, available, source_lists)
    evaluation = SessionEvaluation(
        request.session_id,
        1 if known else None,
        v,
        u,
        hit,
        request.t_q if hit else None,
        True,
        calendar_cell,
        tuple(sorted(slice_metrics)),
        None if known else target.reason,
    )
    details = {
        "request_id": request.request_id,
        "session_id": request.session_id,
        "t_q": request.t_q,
        "k": k,
        "recall": quality["recall"],
        "ndcg": quality["ndcg"],
        "losses": dict(zip(LOSS_NAMES, (1 - v, v - u, u - hit) if known else (None,) * 3)),
        "purchases": purchase_rows,
        "slice_metrics": slice_metrics,
        "sources": sources,
        "overlaps": overlaps,
        "elapsed_ms": result.elapsed_ms,
        "branch": result.candidates.branch.value,
        "delivery": result.delivery.value,
        "reasons": sorted(set(result.reasons) | set(result.candidates.reasons)),
        "trace_missing": not bool(result.trace_ref),
        "feature_masks": {
            "counts": dict(feature_mask_counts) if feature_mask_counts is not None else None,
            "rows": feature_rows,
            "reason": mask_reason if feature_mask_counts is None else None,
        },
    }
    return evaluation, details


def _metric(
    total: float,
    denominator: int,
    *,
    reason: str = "no_registered_targets",
    missing_reason: str | None = None,
) -> dict[str, Any]:
    value = float(total / denominator) if denominator and missing_reason is None else None
    return asdict(
        MetricResult(
            value,
            "passed" if value is not None else "insufficient",
            None if value is not None else missing_reason or reason,
            "fraction",
            denominator,
            (),
            None,
            False,
            ("conditional_on_registered_purchase",),
        )
    )


def _check_rows(rows: Sequence[EvaluationRow]) -> None:
    if len({evaluation.session_id for evaluation, _ in rows}) != len(rows):
        raise InputValidationError("Повтор сессии в оценке")
    for evaluation, details in rows:
        _check_row(evaluation, details)


def _check_row(evaluation: SessionEvaluation, details: Mapping[str, Any]) -> None:
    if evaluation.session_id != details["session_id"]:
        raise InputValidationError("Диагностика другой сессии")
    if evaluation.b_s is not None and (
        evaluation.b_s != 1
        or evaluation.v_s is None
        or evaluation.u_s is None
        or evaluation.hit_s is None
    ):
        raise InputValidationError("Основной знаменатель требует полную зарегистрированную цель")


def _add_sum(partials: list[float], value: float) -> None:
    """Хранить неперекрывающиеся частичные суммы для итогового math.fsum."""
    index = 0
    for other in partials:
        if abs(value) < abs(other):
            value, other = other, value
        high = value + other
        low = other - (high - value)
        if low:
            partials[index] = low
            index += 1
        value = high
    partials[index:] = [value]


SOURCE_FIELDS = (
    "hit_before",
    "hit_after",
    "quota_loss",
    "unique_before_hit",
    "unique_candidate_hit",
    "delta_fixed",
)


def _add_source(state: dict[str, Any], row: Mapping[str, Any], positive: bool) -> None:
    counts = state["counts"]
    for side in ("before", "after"):
        if row[side]["skus"] is not None:
            counts[side + "_measured"] += 1
            counts[side + "_size"] += row[side]["size"]
            counts[side + "_empty"] += not row[side]["size"]
    if row["before"]["skus"] is not None:
        state["depths"].add(row["before"]["depth"])
        counts["truncated"] += bool(row["before"]["truncated"])
    if row["after"]["skus"] is not None:
        counts["available"] += bool(row["after"]["available"])
    counts["unique_size"] += len(row["unique_candidate_skus"])
    if positive:
        counts["positive"] += 1
        counts["positive_truncated"] += bool(row["before"]["truncated"])
        for field in SOURCE_FIELDS:
            if row[field] is None:
                state["missing"].add(field)
            else:
                counts[field] += row[field]


def _source_metrics(states: Mapping[str, Any], n_obs: int, n: int) -> dict[str, Any]:
    sources = {}
    for name, state in sorted(states.items()):
        counts = state["counts"]

        def coverage(field: str, *, full: bool = False) -> dict[str, Any]:
            missing = field in state["missing"] or (
                field not in {"delta_fixed", "unique_candidate_hit"} and counts["positive"] < n
            )
            truncated = full and counts["positive_truncated"]
            return _metric(
                counts[field],
                n,
                missing_reason="not_measured" if missing else "truncated" if truncated else None,
            )

        before_count, after_count = counts["before_measured"], counts["after_measured"]
        sources[name] = {
            "coverage_before_quota": coverage("hit_before"),
            "full_coverage_before_quota": coverage("hit_before", full=True),
            "coverage_after_quota": coverage("hit_after"),
            "quota_loss": coverage("quota_loss"),
            "unique_coverage_before_quota": coverage("unique_before_hit"),
            "unique_candidate_coverage": coverage("unique_candidate_hit"),
            "delta_fixed": coverage("delta_fixed"),
            "before_measured_count": before_count,
            "after_measured_count": after_count,
            "before_missing_fraction": (n_obs - before_count) / n_obs if n_obs else None,
            "after_missing_fraction": (n_obs - after_count) / n_obs if n_obs else None,
            "size_before_mean": counts["before_size"] / before_count if before_count else None,
            "size_after_mean": counts["after_size"] / after_count if after_count else None,
            "empty_fraction_after": counts["after_empty"] / after_count if after_count else None,
            "empty_fraction_before": counts["before_empty"] / before_count
            if before_count
            else None,
            "available_fraction": counts["available"] / after_count if after_count else None,
            "depths": sorted(state["depths"]),
            "truncated_count": counts["truncated"],
            "unique_candidate_size_mean": counts["unique_size"] / n_obs if n_obs else None,
        }
    return sources


def _overlap_metrics(states: Mapping[str, Any], names: Sequence[str], n_obs: int) -> dict[str, Any]:
    overlaps = {}
    for first, second in combinations(names, 2):
        key = first + "|" + second
        state = states.get(key, {"seen": 0, "count": 0, "sum": [], "reasons": Counter()})
        count = state["count"]
        reasons = state["reasons"].copy()
        if state["seen"] < n_obs:
            reasons["not_measured"] += n_obs - state["seen"]
        overlaps[key] = {
            "value": math.fsum(state["sum"]) / count if count else None,
            "defined_count": count,
            "missing_fraction": (n_obs - count) / n_obs if n_obs else None,
            "missing_reasons": dict(sorted(reasons.items())),
            "reason": None
            if count
            else "not_measured"
            if "not_measured" in reasons
            else "empty_union",
        }
    return overlaps


class MetricsAccumulator:
    """Компактная сводка: ID, задержки и счётчики; подробные записи не удерживаются."""

    def __init__(
        self,
        *,
        n_eligible: int,
        exclusions: Mapping[str, int] | None = None,
        selected_size: int | None = None,
        k: int = 20,
    ) -> None:
        if (
            type(n_eligible) is not int
            or n_eligible < 0
            or selected_size is not None
            and (selected_size < 0 or selected_size > n_eligible)
            or any(type(count) is not int or count < 0 for count in (exclusions or {}).values())
        ):
            raise InputValidationError("Несогласованные размеры выборки или K")
        self._seen: set[str] = set()
        self._latencies: list[int] = []
        self._totals: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._unknown_reasons: Counter[str | None] = Counter()
        self._purchases: Counter[str] = Counter()
        self._sums: dict[str, list[float]] = {"recall": [], "ndcg": []}
        self._source_states: dict[str, Any] = {}
        self._overlap_states: dict[str, Any] = {}
        self._slice_states: dict[str, Any] = {}
        self._mask_missing: Counter[str] = Counter()
        self._mask_columns: dict[str, Counter[str]] = {}
        self._quality_fields = (
            (f"Hit@{k}", "hit"),
            ("candidate_coverage", "u"),
            ("available_target_fraction", "v"),
            (f"Recall@{k}", "recall"),
            (f"NDCG@{k}", "ndcg"),
        )
        self._n_eligible = n_eligible
        self._selected_size = selected_size
        self._exclusions = dict(exclusions or {})
        self._k = k

    def add(self, evaluation: SessionEvaluation, details: dict[str, Any]) -> None:
        """Добавить одну проверенную сессию, сохранив прежний порядок суммирования."""
        k = self._k
        _check_row(evaluation, details)
        if evaluation.session_id in self._seen:
            raise InputValidationError("Повтор сессии в оценке")
        self._seen.add(evaluation.session_id)
        if (
            len(self._seen) > self._n_eligible
            or self._selected_size is not None
            and len(self._seen) > self._selected_size
            or details["k"] != k
        ):
            raise InputValidationError("Несогласованные размеры выборки или K")
        positive = evaluation.b_s == 1
        if positive:
            self._totals["n"] += 1
            self._totals["v"] += evaluation.v_s or 0
            self._totals["u"] += evaluation.u_s or 0
            self._totals["hit"] += evaluation.hit_s or 0
            for field, partials in self._sums.items():
                _add_sum(partials, details[field])
        else:
            self._unknown_reasons[evaluation.unknown_reason] += 1
        for label, quality in details["slice_metrics"].items():
            state = self._slice_states.get(label)
            if state is None:
                state = {
                    "observed": 0,
                    "positive": 0,
                    "sums": {field: [] for _, field in self._quality_fields},
                }
                self._slice_states[label] = state
            state["observed"] += 1
            if positive:
                state["positive"] += 1
                for field, partials in state["sums"].items():
                    _add_sum(partials, quality[field])
        for name, row in details["sources"].items():
            state = self._source_states.get(name)
            if state is None:
                state = {"counts": Counter(), "depths": set(), "missing": set()}
                self._source_states[name] = state
            _add_source(state, row, positive)
            self._totals["source_unavailable"] += row["after"]["available"] is False
        for name, row in details["overlaps"].items():
            state = self._overlap_states.get(name)
            if state is None:
                state = {"seen": 0, "count": 0, "sum": [], "reasons": Counter()}
                self._overlap_states[name] = state
            state["seen"] += 1
            if row["value"] is None:
                state["reasons"][row["reason"]] += 1
            else:
                state["count"] += 1
                _add_sum(state["sum"], row["value"])
        self._latencies.append(details["elapsed_ms"])
        self._reasons.update(details["reasons"])
        self._purchases.update(item["availability"] for item in details["purchases"])
        self._totals["fallback"] += details["branch"] == Branch.FALLBACK.value
        self._totals["failed"] += details["delivery"] == DeliveryStatus.FAILED.value
        self._totals["not_attempted"] += details["delivery"] == DeliveryStatus.NOT_ATTEMPTED.value
        self._totals["trace_missing"] += details["trace_missing"]
        self._totals["deadline"] += any(
            reason.rsplit(":", 1)[-1] in {"timeout", "deadline_exceeded", "deadline_missed"}
            for reason in details["reasons"]
        )
        masks = details["feature_masks"]
        if masks["counts"] is None:
            self._mask_missing[masks["reason"]] += 1
        else:
            self._totals["mask_measured"] += 1
            self._totals["mask_requests"] += any(masks["counts"].values())
            self._totals["mask_rows"] += masks["rows"]
            self._totals["mask_values"] += sum(masks["counts"].values())
            for name, value in masks["counts"].items():
                column = self._mask_columns.get(name)
                if column is None:
                    column = Counter()
                    self._mask_columns[name] = column
                column["masked"] += value
                column["rows"] += masks["rows"]

    def finish(self) -> dict[str, Any]:
        """Вернуть сводку; повторный вызов без add даёт тот же результат."""
        k = self._k
        n_obs = len(self._seen)
        n, v, u, hits = (self._totals[field] for field in ("n", "v", "u", "hit"))
        metrics = {
            f"Hit@{k}": _metric(hits, n),
            "available_target_fraction": _metric(v, n),
            "candidate_coverage": _metric(u, n),
            "candidate_coverage_available": _metric(u, v, reason="no_available_target"),
            f"Recall@{k}": _metric(math.fsum(self._sums["recall"]), n),
            f"NDCG@{k}": _metric(math.fsum(self._sums["ndcg"]), n),
        }
        losses = {
            name: _metric(value, n) for name, value in zip(LOSS_NAMES, (n - v, v - u, u - hits))
        }
        sources = _source_metrics(self._source_states, n_obs, n)
        overlaps = _overlap_metrics(self._overlap_states, sorted(self._source_states), n_obs)
        slices = {}
        for label, state in sorted(self._slice_states.items()):
            count = state["positive"]
            slices[label] = {
                "N_obs": state["observed"],
                "N": count,
                "metrics": {
                    name: _metric(math.fsum(state["sums"][field]), count)
                    for name, field in self._quality_fields
                },
            }
        self._latencies.sort()
        return {
            "counts": {
                "N_eligible": self._n_eligible,
                "N_obs": n_obs,
                "N": n,
                "N_unknown": n_obs - n,
            },
            "selected_size": self._selected_size,
            "exclusions": dict(self._exclusions or {}),
            "unknown_reasons": dict(self._unknown_reasons),
            "coverage_chain": {"V": v, "U": u, "Hit": hits},
            "metrics": metrics,
            "losses": losses,
            "sources": sources,
            "overlaps": overlaps,
            "slices": slices,
            "purchases": dict(self._purchases),
            "cost": {
                "request_count": n_obs,
                "elapsed_ms_total": sum(self._latencies),
                "local_elapsed_ms_p95": self._latencies[math.ceil(0.95 * len(self._latencies)) - 1]
                if self._latencies
                else None,
                "local_elapsed_ms_p99": self._latencies[math.ceil(0.99 * len(self._latencies)) - 1]
                if self._latencies
                else None,
                "fallback_count": self._totals["fallback"],
                "delivery_failed_count": self._totals["failed"],
                "not_attempted_count": self._totals["not_attempted"],
                "missing_trace_count": self._totals["trace_missing"],
                "source_unavailable_count": self._totals["source_unavailable"],
                "recorded_deadline_exceeded_count": self._totals["deadline"],
                "feature_masks": {
                    "measured_requests": self._totals["mask_measured"],
                    "masked_requests": self._totals["mask_requests"],
                    "rows": self._totals["mask_rows"] if self._totals["mask_measured"] else None,
                    "masked_values": self._totals["mask_values"]
                    if self._totals["mask_measured"]
                    else None,
                    "missing_reasons": dict(self._mask_missing),
                    "columns": {
                        name: {
                            **column,
                            "fraction": column["masked"] / column["rows"]
                            if column["rows"]
                            else None,
                        }
                        for name, column in sorted(self._mask_columns.items())
                    },
                },
                "reason_counts": dict(sorted(self._reasons.items())),
                "service_load_status": "not_measured",
            },
        }


def aggregate_metrics(
    rows: Iterable[EvaluationRow],
    *,
    n_eligible: int,
    exclusions: Mapping[str, int] | None = None,
    selected_size: int | None = None,
    k: int = 20,
) -> dict[str, Any]:
    """Один проход без сохранения подробных строк; неизвестные цели входят в стоимость."""
    accumulator = MetricsAccumulator(
        n_eligible=n_eligible, exclusions=exclusions, selected_size=selected_size, k=k
    )
    for evaluation, details in rows:
        accumulator.add(evaluation, details)
    return accumulator.finish()


def paired_metrics(left: Sequence[EvaluationRow], right: Sequence[EvaluationRow]) -> dict[str, Any]:
    """Парная разность Hit по тем же сессиям и тем же зарегистрированным целям."""
    _check_rows(left)
    _check_rows(right)
    right_by_id = {evaluation.session_id: (evaluation, details) for evaluation, details in right}
    if {evaluation.session_id for evaluation, _ in left} != set(right_by_id):
        raise InputValidationError("Для пары нужны одинаковые сессии")
    difference, count = 0, 0
    for evaluation, details in left:
        other, other_details = right_by_id[evaluation.session_id]
        if (
            evaluation.b_s != other.b_s
            or evaluation.calendar_cell != other.calendar_cell
            or details["t_q"] != other_details["t_q"]
            or details["k"] != other_details["k"]
            or {row["sku"] for row in details["purchases"]}
            != {row["sku"] for row in other_details["purchases"]}
        ):
            raise InputValidationError("Пара должна разделять цель, запрос и календарную ячейку")
        if evaluation.b_s == 1:
            difference += (evaluation.hit_s or 0) - (other.hit_s or 0)
            count += 1
    return _metric(difference, count)
