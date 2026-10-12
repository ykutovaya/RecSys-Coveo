"""Чувствительность префикса к кратностям при одном неизменном A_G."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import PreparedRequest, TaskConfig
from coveo_recsys.contracts.records import Branch, CandidateResult, CanonicalBlock, FeatureBatch
from coveo_recsys.features import FeatureEngine
from coveo_recsys.retrieval.candidates import retrieve
from coveo_recsys.retrieval.scores import bind_context, score_request
from coveo_recsys.state import build_prefix


def compare_multiplicities(
    prepared: PreparedRequest,
    blocks: Sequence[CanonicalBlock],
    task: TaskConfig,
    set_engine: FeatureEngine,
    multiset_engine: FeatureEngine,
    *,
    base_candidates: CandidateResult,
    base_features: FeatureBatch,
) -> dict[str, Any]:
    """Сопоставить вычисления на тех же ID без модели и без присоединения целей.

    Нужен полный журнал канонических блоков сессии, поскольку хвост снимка
    не восстанавливает ранние счётчики. Будущее отсекается штатным build_prefix.
    Базовые признаки уже рассчитаны выдачей; матрицы в результат не включаются.
    """
    if (
        task.event_semantics != "set_v1"
        or prepared.prefix.event_semantics != "set_v1"
        or prepared.t_q != prepared.a_q
        or set_engine._semantics != "set_v1"
        or multiset_engine._semantics != "multiset_v1"
        or set_engine.index is not multiset_engine.index
        or set_engine.config != multiset_engine.config
        or base_candidates.request_id != prepared.request_id
        or base_candidates.branch != Branch.PRIMARY
        or base_features.request_id != prepared.request_id
        or base_features.skus != base_candidates.skus
        or base_features.feature_schema_sha256 != set_engine.schema_sha256
        or base_features.numeric_columns != set_engine.numeric_columns
        or base_features.categorical_columns != set_engine.categorical_columns
    ):
        raise InputValidationError("Несовместимые базовые и диагностические входы кратностей")
    base_prefix = build_prefix(
        blocks,
        source_namespace=prepared.source_namespace,
        session_id=prepared.session_id,
        event_cutoff_ms=prepared.t_q,
        availability_cutoff_ms=prepared.a_q,
        config=task,
    )
    if base_prefix != prepared.prefix:
        raise InputValidationError("Канонические блоки не восстанавливают базовый префикс")
    multiset_prefix = build_prefix(
        blocks,
        source_namespace=prepared.source_namespace,
        session_id=prepared.session_id,
        event_cutoff_ms=prepared.t_q,
        availability_cutoff_ms=prepared.a_q,
        config=replace(task, event_semantics="multiset_v1"),
    )
    if multiset_prefix.purchased_skus != prepared.prefix.purchased_skus:
        raise InputValidationError("Семантика кратностей изменила множество покупок")
    prefix_changed = replace(multiset_prefix, event_semantics="set_v1") != base_prefix
    result = {
        "request_id": prepared.request_id,
        "scope": "prefix_sensitivity_fixed_historical",
        "set_schema_sha256": base_features.feature_schema_sha256,
        "multiset_schema_sha256": multiset_engine.schema_sha256,
        "affected": prefix_changed,
        "prefix_changed": prefix_changed,
        "candidate_changed": False,
        "common_candidates": len(base_candidates.skus),
        "changed_features": [],
        "set_candidate_skus": base_candidates.skus,
        "multiset_candidate_skus": base_candidates.skus,
    }
    # При совпадении префиксов все кратности последней группы единичные:
    # обе семантики дают те же веса, баллы, кандидатов, признаки и маски.
    # Базовая матрица уже вычислена выдачей; схема диагностики остаётся отдельной.
    if not prefix_changed:
        return result
    context = bind_context(replace(prepared, prefix=multiset_prefix), multiset_engine.index)
    scores = score_request(context, multiset_engine.index, multiset_engine.config)
    candidates = retrieve(context, multiset_engine.index, multiset_engine.config, scores=scores)
    features = multiset_engine.build(context, candidates, scores=scores)
    positions = {sku: position for position, sku in enumerate(features.skus)}
    common = [
        (position, positions[sku])
        for position, sku in enumerate(base_features.skus)
        if sku in positions
    ]
    before = [
        left + right
        for left, right in zip(base_features.numeric, base_features.categorical, strict=True)
    ]
    after = [
        left + right for left, right in zip(features.numeric, features.categorical, strict=True)
    ]
    changes = []
    for column, name in enumerate(
        base_features.numeric_columns + base_features.categorical_columns
    ):
        changed = mask_changed = 0
        max_delta = 0.0
        for left, right in common:
            delta = abs(float(after[right][column]) - float(before[left][column]))
            mask = base_features.masks[left][column] != features.masks[right][column]
            changed += int(delta != 0 or mask)
            mask_changed += int(mask)
            max_delta = max(max_delta, delta)
        if changed:
            changes.append(
                {
                    "name": name,
                    "changed_rows": changed,
                    "mask_changed_rows": mask_changed,
                    "max_abs_delta": max_delta,
                }
            )
    candidate_changed = candidates.skus != base_candidates.skus
    result.update(
        candidate_changed=candidate_changed,
        common_candidates=len(common),
        changed_features=changes,
        multiset_candidate_skus=candidates.skus,
    )
    return result
