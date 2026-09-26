"""Фиксированные признаки Э3; без чтения файлов, целей или обучения."""

import hashlib
import json
import math
from typing import Any, cast

import numpy as np

from coveo_recsys.contracts import InputValidationError, NumericError, id_bytes
from coveo_recsys.contracts.historical import RELATIONS
from coveo_recsys.contracts.records import Branch, CandidateResult, FeatureBatch, RequestContext
from coveo_recsys.contracts.retrieval import SOURCES, RetrievalConfig
from coveo_recsys.numerics import (
    MultiplicitySemantics,
    checked_subtract,
    compensated_sum,
    counter_add,
    last_group_mean,
)
from coveo_recsys.retrieval.scores import QueryScores, RetrievalIndex, score_request


def _columns() -> list[tuple[str, str, str | None, str]]:
    """Исполняемый набор столбцов, не интерпретатор формул из JSON."""
    result: list[tuple[str, str, str | None, str]] = []

    def add(name: str, dtype: str = "float64", *, masked: bool = False, log: bool = False) -> None:
        result.append(
            (name, dtype, name + "_mask" if masked else None, "log1p" if log else "identity")
        )
        if masked:
            result.append((name + "_mask", "bool", None, "identity"))

    for source in SOURCES:
        add("origin_" + source, "bool")
        add("score_" + source, masked=True)
        add("rank_" + source, masked=True)
    add("origin_fill", "bool")
    for relation in RELATIONS:
        add("last_" + relation, masked=True)
        add("support_" + relation, masked=True, log=True)
    for modality in ("description", "image"):
        add("last_" + modality, masked=True)
        add("history_" + modality + "_available", "bool")
        add("item_" + modality + "_available", "bool")
    add("search_age", masked=True, log=True)
    add("detail_count", log=True)
    add("detail_age", masked=True, log=True)
    add("cart_add", "bool", masked=True)
    add("cart_remove", "bool", masked=True)
    add("cart_age", masked=True, log=True)
    for name in ("session_event_count", "session_distinct_items", "session_add_count"):
        add(name, log=True)
    add("session_duration", masked=True, log=True)
    add("session_has_search", "bool")
    add("category_similarity", masked=True)
    add("category_match", "bool", masked=True)
    add("price_bucket", masked=True)
    for name in ("equal", "sign", "absolute"):
        add("last_price_" + name, masked=True)
    add("historical_purchase_count", log=True)
    add("cart_unknown", "bool")
    add("category", "category")
    return result


class FeatureEngine:
    def __init__(
        self,
        index: RetrievalIndex,
        schema: dict[str, Any],
        config: RetrievalConfig = RetrievalConfig(),
    ) -> None:
        self.index, self.config = index, config
        self._columns = _columns()
        self._semantics = schema.get("event_semantics")
        columns = schema.get("columns")
        if (
            schema.get("schema_version") != "1"
            or schema.get("config_id") != "gcs_rk_v2_7"
            or self._semantics not in {"set_v1", "multiset_v1"}
            or not isinstance(columns, list)
            or len(columns) != len(self._columns)
        ):
            raise InputValidationError("Несовместимая схема признаков")
        for position, (actual, expected) in enumerate(zip(columns, self._columns, strict=True)):
            if (
                not isinstance(actual, dict)
                or actual.get("position") != position
                or tuple(actual.get(key) for key in ("name", "dtype", "mask", "transformation"))
                != expected
            ):
                raise InputValidationError(
                    "Состав, порядок или преобразование признаков не поддерживаются"
                )
        serialized = (
            json.dumps(
                schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )
        self.schema_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.numeric_columns = tuple(
            name for name, dtype, _, _ in self._columns if dtype != "category"
        )
        self.categorical_columns = ("category",)

    def build(
        self,
        context: RequestContext,
        candidates: CandidateResult,
        *,
        scores: QueryScores | None = None,
    ) -> FeatureBatch:
        scores = score_request(context, self.index, self.config) if scores is None else scores
        self._check(context, candidates, scores)
        prefix = context.prefix
        semantics = cast(MultiplicitySemantics, prefix.event_semantics)
        counts = {(kind, sku): count for kind, sku, count in prefix.counters}
        views = dict(prefix.last_views)
        cart = {item.sku: item for item in prefix.cart}
        origins = dict(candidates.origins)
        ranks = {
            source.source: {sku: 1 / (rank + 2) for rank, sku in enumerate(source.skus)}
            for source in candidates.sources
        }
        last = sorted(prefix.full_last_detail, key=lambda item: id_bytes(item.sku))
        last_rows = [(item, self.index.row(item.sku)) for item in last]
        event_count = add_count = 0
        for (kind, _), count in counts.items():
            event_count = counter_add(event_count, count)
            if kind == "add":
                add_count = counter_add(add_count, count)
        numeric, categorical, masks = [], [], []
        for sku in candidates.skus:
            position = self.index.positions[sku]
            candidate = self.index.row(sku)
            assert candidate is not None
            values: dict[str, tuple[float | int, bool]] = {}

            def put(name: str, value: float | int, mask: bool = True) -> None:
                if not math.isfinite(value):
                    raise NumericError("Неконечный признак")
                values[name] = (value if mask else 0.0, bool(mask))

            for source in SOURCES:
                member = sku in ranks[source]
                put("origin_" + source, float(member))
                put(
                    "score_" + source,
                    float(scores.values[source][position]),
                    bool(scores.masks[source][position]),
                )
                put("rank_" + source, ranks[source].get(sku, 0.0), member)
            put("origin_fill", float("fill" in origins[sku]))
            for relation in RELATIONS:
                options = [
                    (edge, item.sku)
                    for item in last
                    if (edge := self.index.edge(relation, item.sku, position)) is not None
                ]
                edge = (
                    min(options, key=lambda option: (-option[0][0], id_bytes(option[1])))[0]
                    if options
                    else (0.0, 0)
                )
                put("last_" + relation, edge[0], bool(options))
                put("support_" + relation, edge[1], bool(options))
            for modality in ("description", "image"):
                similarities, available, multiplicities = [], [], []
                item_mask = getattr(candidate, modality + "_mask")
                for item, anchor in last_rows:
                    valid = item_mask and anchor is not None and getattr(anchor, modality + "_mask")
                    similarity = (
                        compensated_sum(getattr(candidate, modality) * getattr(anchor, modality))
                        if valid
                        else 0.0
                    )
                    similarities.append(similarity)
                    available.append(bool(valid))
                    multiplicities.append(item.count)
                mean = last_group_mean(similarities, available, multiplicities, semantics=semantics)
                put("last_" + modality, mean.value, mean.mask)
                put("history_" + modality + "_available", float(scores.history_available[modality]))
                put("item_" + modality + "_available", float(item_mask))
            put("search_age", scores.search_age_ms / 1000, scores.reasons["search"] is None)
            put("detail_count", counts.get(("detail", sku), 0))
            put(
                "detail_age",
                checked_subtract(context.t_q, views[sku]) / 1000 if sku in views else 0.0,
                sku in views,
            )
            cart_item = cart.get(sku)
            state = cart_item.state if cart_item is not None else "unknown"
            put("cart_add", float(state == "add"), state != "unknown")
            put("cart_remove", float(state == "remove"), state != "unknown")
            put("cart_unknown", float(state == "unknown"))
            cart_time = cart_item.last_unambiguous_ms if cart_item else None
            put(
                "cart_age",
                checked_subtract(context.t_q, cart_time) / 1000 if cart_time is not None else 0.0,
                cart_time is not None,
            )
            put("session_event_count", event_count)
            put(
                "session_distinct_items",
                len(
                    {item for (_, item), count in counts.items() if item is not None and count > 0}
                ),
            )
            put("session_add_count", add_count)
            first_time = prefix.first_event_time_ms
            put(
                "session_duration",
                checked_subtract(context.t_q, first_time) / 1000 if first_time is not None else 0.0,
                first_time is not None,
            )
            put("session_has_search", float(prefix.search is not None))
            category_values = []
            if candidate.category:
                candidate_path = candidate.category.split("/")
                for _, anchor in last_rows:
                    if anchor is not None and anchor.category:
                        anchor_path = anchor.category.split("/")
                        common = 0
                        for left, right in zip(candidate_path, anchor_path):
                            if left != right:
                                break
                            common += 1
                        category_values.append(
                            (
                                common / max(1, len(candidate_path), len(anchor_path)),
                                float(candidate_path == anchor_path),
                            )
                        )
            put(
                "category_similarity",
                max((value[0] for value in category_values), default=0.0),
                bool(category_values),
            )
            put(
                "category_match",
                max((value[1] for value in category_values), default=0.0),
                bool(category_values),
            )
            put("price_bucket", candidate.price_bucket or 0, candidate.price_bucket is not None)
            deltas, price_masks, multiplicities = [], [], []
            for item, anchor in last_rows:
                if (
                    candidate.price_bucket is not None
                    and anchor is not None
                    and anchor.price_bucket is not None
                ):
                    delta, valid = (
                        checked_subtract(candidate.price_bucket, anchor.price_bucket),
                        True,
                    )
                else:
                    delta, valid = 0, False
                deltas.append(delta)
                price_masks.append(valid)
                multiplicities.append(item.count)
            for name, terms in (
                ("equal", [float(delta == 0) for delta in deltas]),
                ("sign", [float((delta > 0) - (delta < 0)) for delta in deltas]),
                ("absolute", [abs(delta) for delta in deltas]),
            ):
                mean = last_group_mean(terms, price_masks, multiplicities, semantics=semantics)
                put("last_price_" + name, mean.value, mean.mask)
            put(
                "historical_purchase_count",
                int(self.index.package.statistics.purchase_counts[position]),
            )
            put("category", int(self.index.package.content.category_codes[position]))
            numeric_row, categorical_row, mask_row = [], [], []
            for name, dtype, linked_mask, transformation in self._columns:
                if name.endswith("_mask"):
                    value, mask = float(values[name[:-5]][1]), True
                else:
                    value, mask = values[name]
                if transformation == "log1p":
                    if value < 0:
                        raise NumericError("Отрицательная давность или счётчик")
                    value = math.log1p(value)
                if not math.isfinite(value):
                    raise NumericError("Неконечный преобразованный признак")
                if dtype == "category":
                    categorical_row.append(int(value))
                else:
                    numeric_row.append(float(value))
                mask_row.append(mask)
            numeric.append(tuple(numeric_row))
            categorical.append(tuple(categorical_row))
            masks.append(tuple(mask_row))
        return FeatureBatch(
            context.request_id,
            candidates.skus,
            self.numeric_columns,
            self.categorical_columns,
            tuple(numeric),
            tuple(categorical),
            tuple(masks),
            self.schema_sha256,
        )

    def _check(
        self, context: RequestContext, candidates: CandidateResult, scores: QueryScores
    ) -> None:
        if (
            scores.context != context
            or scores.index is not self.index
            or scores.config != self.config
            or context.catalog_sha256 != self.index.catalog_sha256
            or self.index.package.snapshot_time_ms >= context.t_q
            or context.prefix.event_semantics != self._semantics
            or candidates.request_id != context.request_id
            or candidates.branch != Branch.PRIMARY
        ):
            raise InputValidationError("Запрос, баллы, кандидаты или схема несовместимы")
        count = len(self.index.positions)
        if (
            set(scores.values) != set(SOURCES)
            or set(scores.masks) != set(SOURCES)
            or scores.eligible.shape != (count,)
        ):
            raise InputValidationError("Состав общих баллов не соответствует каталогу")
        for name in SOURCES:
            if scores.values[name].shape != (count,) or scores.masks[name].shape != (count,):
                raise InputValidationError("Форма общих баллов не соответствует каталогу")
            if not np.isfinite(scores.values[name]).all():
                raise NumericError("Неконечный общий балл")
        if (
            candidates.skus != tuple(sorted(candidates.skus, key=id_bytes))
            or len(candidates.skus) > self.config.max_candidates
            or any(
                sku not in self.index.positions
                or not scores.eligible[self.index.positions[sku]]
                or sku in context.prefix.purchased_skus
                for sku in candidates.skus
            )
            or tuple(source.source for source in candidates.sources) != SOURCES
            or tuple(sku for sku, _ in candidates.origins) != candidates.skus
        ):
            raise InputValidationError("Несогласованный порядок или состав кандидатов")
        selected = set(candidates.skus)
        for source, quota in zip(candidates.sources, self.config.quotas, strict=True):
            if (
                source.quota != quota
                or not set(source.skus) <= selected
                or source.reason != scores.reasons[source.source]
                or source.available != (source.reason is None)
            ):
                raise InputValidationError("Список источника не соответствует кандидатам или квоте")
            for sku, value in zip(source.skus, source.scores, strict=True):
                position = self.index.positions[sku]
                if (
                    not scores.masks[source.source][position]
                    or value != scores.values[source.source][position]
                ):
                    raise InputValidationError("Список источника не соответствует общим баллам")
        for sku, actual in candidates.origins:
            expected = tuple(source.source for source in candidates.sources if sku in source.skus)
            if actual != (expected or ("fill",)):
                raise InputValidationError("Происхождение не соответствует спискам источников")
