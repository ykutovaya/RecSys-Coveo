"""Соединение закрытой таблицы ответов с целями и отчёт C_smoke/D."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.evaluation import (
    FINAL_POLICIES,
    POLICIES,
    SELECT_POLICIES,
    EvaluationRecord,
    ReplayRecord,
)
from coveo_recsys.contracts.historical import HistoricalPackage
from coveo_recsys.contracts.preparation import EVALUATION_PROTOCOL, PreparationPlan, PreparedRequest
from coveo_recsys.contracts.records import Branch, RequestResult
from coveo_recsys.contracts.retrieval import METADATA370_CONFIG_ID, NO_METADATA_CONFIG_ID
from coveo_recsys.contracts.serialization import from_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.access import DatasetAccess
from coveo_recsys.data.periods import session_key
from coveo_recsys.data.prepared import TableWriter, iter_sessions, read_report, table_info
from coveo_recsys.evaluation.intervals import MC_METHOD, calendar_intervals
from coveo_recsys.evaluation.metrics import (
    EvaluationRow,
    MetricsAccumulator,
    build_sku_slices,
    evaluate_session,
    paired_metrics,
)

ESTIMATES = (*POLICIES, "GCS+Rk-P", "GCS+Rk-R")
FINAL_ESTIMATES = (*ESTIMATES, NO_METADATA_CONFIG_ID, "GCS+Rk-" + NO_METADATA_CONFIG_ID)


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise InputValidationError(f"Неверный состав {name}")


def validate_evaluation_plan(evaluation_plan: dict[str, Any], plan: PreparationPlan) -> None:
    """Проверка фиксированных настроек до чтения исходов; null точности допустим."""
    _keys(
        evaluation_plan,
        {"period", "policies", "k", "diagnostic_depth", "intervals"},
        "плана оценки",
    )
    final = evaluation_plan["period"] == "D"
    selection = evaluation_plan["period"] == "C_select"
    policies = SELECT_POLICIES if selection else FINAL_POLICIES if final else POLICIES
    estimates = (
        (*SELECT_POLICIES, "GCS+Rk-" + METADATA370_CONFIG_ID)
        if selection
        else FINAL_ESTIMATES
        if final
        else ESTIMATES
    )
    if (
        evaluation_plan["period"] not in {"C_smoke", "C_select", "D"}
        or evaluation_plan["policies"] != list(policies)
        or type(evaluation_plan["k"]) is not int
        or evaluation_plan["k"] != 20
        or type(evaluation_plan["diagnostic_depth"]) is not int
        or evaluation_plan["diagnostic_depth"] <= 0
    ):
        raise InputValidationError(
            "Нужны допустимый период, фиксированные политики, K=20 и положительная глубина"
        )
    inference = evaluation_plan["intervals"]
    _keys(
        inference,
        {
            "grid_step_ms",
            "block_cells",
            "sensitivity_block_cells",
            "mc_error_method",
            "dependence",
            "half_width_fraction",
        },
        "плана интервалов",
    )
    period = next(period for period in plan.periods if period.name == evaluation_plan["period"])
    width = period.end_ms - period.start_ms - plan.task.horizon_ms - plan.task.gap_ms
    step = inference["grid_step_ms"]
    if type(step) is not int or step <= 0 or width <= 0 or width % step:
        raise InputValidationError("Сетка не покрывает целыми ячейками допустимое окно стартов")
    sensitivities = inference["sensitivity_block_cells"]
    if not isinstance(sensitivities, list) or len(sensitivities) != 2:
        raise InputValidationError("Нужны две длины чувствительности")
    lengths = [inference["block_cells"], *sensitivities]
    if (
        any(type(value) is not int or not 1 <= value <= width // step for value in lengths)
        or len(set(lengths)) != 3
        or inference["mc_error_method"] != MC_METHOD
    ):
        raise InputValidationError("Неверные длины блоков либо метод MC-погрешности")
    _keys(inference["dependence"], set(estimates), "предпосылок оценок")
    _keys(inference["half_width_fraction"], set(estimates), "полуширин оценок")
    for name in estimates:
        assumptions = inference["dependence"][name]
        _keys(assumptions, {"supported", "assumptions"}, "описания зависимости")
        if (
            assumptions["supported"] is not None
            and type(assumptions["supported"]) is not bool
            or not isinstance(assumptions["assumptions"], list)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in assumptions["assumptions"]
            )
        ):
            raise InputValidationError("Неверные предпосылки зависимости")
        half_width = inference["half_width_fraction"][name]
        if half_width is not None and (
            type(half_width) not in (int, float)
            or not math.isfinite(half_width)
            or not 0 < half_width <= 1
        ):
            raise InputValidationError("Плановая полуширина должна быть положительной долей")


def _diagnostics(record: ReplayRecord, depth: int) -> dict[str, Any]:
    values = record.parsed_json
    names = {"source_lists", "feature_mask_counts", "feature_rows", "mask_reason"}
    if record.policy == "GCS+Rk":
        names.add("sensitivity")
    _keys(values, names, "диагностики ответа")
    if not isinstance(values["source_lists"], dict):
        raise InputValidationError("Нет сохранённых списков источников")
    for source in values["source_lists"].values():
        _keys(source, {"skus", "depth", "truncated", "reason"}, "списка источника")
        if type(source["depth"]) is not int or source["depth"] != depth:
            raise InputValidationError("Глубина источников отличается от фиксированного плана")
    if record.policy in {"P", "R"} and (
        values["feature_mask_counts"] is not None
        or values["feature_rows"] is not None
        or values["mask_reason"] != "not_applicable"
    ):
        raise InputValidationError("Контроли P/R не имеют матрицы признаков модели")
    return values


def _sensitivity_row(
    value: Any,
    request: PreparedRequest,
    result: RequestResult,
    target: TargetRecord,
    catalog: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputValidationError("Нет диагностики кратностей GCS+Rk")
    if value.get("status") == "not_measured":
        _keys(value, {"status", "reason"}, "неизмеренной чувствительности")
        if not isinstance(value["reason"], str) or not value["reason"]:
            raise InputValidationError("Неизмеренная чувствительность требует причину")
        return dict(
            value,
            target_known=bool(target.purchases),
            set_candidate_hit=None,
            multiset_candidate_hit=None,
        )
    _keys(
        value,
        {
            "request_id",
            "scope",
            "set_schema_sha256",
            "multiset_schema_sha256",
            "affected",
            "prefix_changed",
            "candidate_changed",
            "common_candidates",
            "changed_features",
            "set_candidate_skus",
            "multiset_candidate_skus",
        },
        "чувствительности префикса",
    )
    before, after = value["set_candidate_skus"], value["multiset_candidate_skus"]
    if (
        value["request_id"] != request.request_id
        or value["scope"] != "prefix_sensitivity_fixed_historical"
        or result.candidates.branch != Branch.PRIMARY
        or not isinstance(before, list)
        or not isinstance(after, list)
        or tuple(before) != result.candidates.skus
        or any(not isinstance(sku, str) or not sku for sku in after)
        or len(set(after)) != len(after)
        or not set(after) <= catalog - set(request.prefix.purchased_skus)
        or type(value["common_candidates"]) is not int
        or value["common_candidates"] != len(set(before) & set(after))
        or any(
            type(value[name]) is not bool
            for name in ("affected", "prefix_changed", "candidate_changed")
        )
        or value["candidate_changed"] != (before != after)
        or not isinstance(value["changed_features"], list)
    ):
        raise InputValidationError("Чувствительность относится к чужой выдаче или набору")
    seen = set()
    for feature in value["changed_features"]:
        _keys(
            feature,
            {"name", "changed_rows", "mask_changed_rows", "max_abs_delta"},
            "изменения признака",
        )
        if (
            not isinstance(feature["name"], str)
            or not feature["name"]
            or feature["name"] in seen
            or type(feature["changed_rows"]) is not int
            or not 0 < feature["changed_rows"] <= value["common_candidates"]
            or type(feature["mask_changed_rows"]) is not int
            or not 0 <= feature["mask_changed_rows"] <= feature["changed_rows"]
            or type(feature["max_abs_delta"]) not in (float, int)
            or not math.isfinite(feature["max_abs_delta"])
            or feature["max_abs_delta"] < 0
        ):
            raise InputValidationError("Неверные счётчики чувствительности признаков")
        seen.add(feature["name"])
    if value["affected"] != (value["prefix_changed"] or value["candidate_changed"] or bool(seen)):
        raise InputValidationError("Неверный признак влияния кратностей")
    return dict(
        value,
        status="measured",
        target_known=bool(target.purchases),
        set_candidate_hit=int(bool(set(before) & set(target.purchases)))
        if target.purchases
        else None,
        multiset_candidate_hit=int(bool(set(after) & set(target.purchases)))
        if target.purchases
        else None,
    )


def _aggregate_sensitivity(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    schemas: set[tuple[str, str]] = set()
    features: dict[str, dict[str, Any]] = {}
    for row in rows:
        counts["sessions"] += 1
        counts["target_sessions"] += bool(row["target_known"])
        if row["status"] != "measured":
            missing[row["reason"]] += 1
            continue
        counts["measured_sessions"] += 1
        counts["measured_target_sessions"] += bool(row["target_known"])
        for name in ("affected", "prefix_changed", "candidate_changed"):
            counts[name + "_sessions"] += row[name]
        counts["common_candidate_rows"] += row["common_candidates"]
        if row["target_known"]:
            for name in ("set", "multiset"):
                hits[name] += row[name + "_candidate_hit"]
        schemas.add((row["set_schema_sha256"], row["multiset_schema_sha256"]))
        if len(schemas) > 1:
            raise InputValidationError("Менялась схема диагностики кратностей")
        for value in row["changed_features"]:
            item = features.setdefault(
                value["name"], dict(changed_rows=0, mask_changed_rows=0, max_abs_delta=0.0)
            )
            item["changed_rows"] += value["changed_rows"]
            item["mask_changed_rows"] += value["mask_changed_rows"]
            item["max_abs_delta"] = max(item["max_abs_delta"], value["max_abs_delta"])
    reason = (
        "no_registered_targets"
        if not counts["target_sessions"]
        else "incomplete_sensitivity_measurement"
        if counts["measured_target_sessions"] != counts["target_sessions"]
        else None
    )
    coverages = {
        name + "_candidate_coverage": dict(
            value=hits[name] / counts["target_sessions"] if reason is None else None,
            denominator=counts["target_sessions"],
            reason=reason,
        )
        for name in ("set", "multiset")
    }
    return dict(
        scope="prefix_sensitivity_fixed_historical",
        historical_catalog_changed=False,
        model_retrained=False,
        multiset_model_applied=False,
        sessions=counts["sessions"],
        measured_sessions=counts["measured_sessions"],
        target_sessions=counts["target_sessions"],
        measured_target_sessions=counts["measured_target_sessions"],
        missing_reasons=dict(missing),
        affected_sessions=counts["affected_sessions"],
        prefix_changed_sessions=counts["prefix_changed_sessions"],
        candidate_changed_sessions=counts["candidate_changed_sessions"],
        common_candidate_rows=counts["common_candidate_rows"],
        schema_pairs=[list(pair) for pair in sorted(schemas)],
        changed_features=dict(sorted(features.items())),
        **coverages,
    )


def _evaluation_rows(
    path: Path,
    plan: PreparationPlan,
    check_budget: Callable[[], None],
    policies: Sequence[str] = POLICIES,
) -> Iterator[EvaluationRecord]:
    """Один проход политик; подробности удерживаются лишь для одной сессии."""
    for _, records in iter_sessions(
        path,
        EvaluationRecord,
        batch_size=plan.chunk_size,
        max_session_records=len(policies),
        max_session_bytes=plan.max_session_bytes,
    ):
        check_budget()
        if len(records) != len(policies) or {record.policy for record in records} != set(policies):
            raise InputValidationError("Изменилась полная таблица запланированных оценок")
        yield from records


def summarize_replay(
    source: Path,
    prepared: dict[str, Any],
    answers_path: Path,
    history: HistoricalPackage,
    plan: PreparationPlan,
    evaluation_plan: dict[str, Any],
    destination: Path,
    check_budget: Callable[[], None],
    *,
    selected_ids: Sequence[str] | None = None,
    d_admission: Callable[[], str] | None = None,
    selection_admission: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Все выбранные SID и политики; цели присоединяются после закрытых ответов.

    Идентичность нового потребителя и общие ресурсы добавляет вызывающая команда.
    Производитель подготовки закреплён отдельно; обучение здесь не вызывается.
    """
    validate_evaluation_plan(evaluation_plan, plan)
    period_name = evaluation_plan["period"]
    policy_names = tuple(evaluation_plan["policies"])
    if period_name == "D" and (d_admission is None or not d_admission()):
        raise InputValidationError("Нет допуска pre_D")
    if period_name == "C_select" and (selection_admission is None or not selection_admission()):
        raise InputValidationError("Нет допуска pre_selection")
    check_budget()
    if source.resolve() == destination.resolve():
        raise InputValidationError("Оценка должна храниться отдельно от принятой подготовки")
    report_hash = sha256_file(source / "reports/preparation.json")
    if (
        read_report(source, report_hash, verify_tables=False) != prepared
        or from_json(prepared["plan_json"].encode("utf-8"), PreparationPlan) != plan
    ):
        raise InputValidationError("Изменён переданный отчёт или план подготовки")
    selected = prepared["selection"].get(period_name, [])
    if not isinstance(selected, list) or len(set(selected)) != len(selected):
        raise InputValidationError("Некорректная выборка оценочного периода")
    period_counts = prepared["period_counts"][period_name]
    if period_counts["selected"] != len(selected) or prepared["N_obs"] != len(selected):
        raise InputValidationError("Выборка и счётчики подготовки не совпадают")
    prepared_selected_size = len(selected)
    prepared_ids = set(selected)
    if selected_ids is not None:
        if (
            not isinstance(selected_ids, Sequence)
            or isinstance(selected_ids, (str, bytes))
            or any(not isinstance(sid, str) or not sid for sid in selected_ids)
        ):
            raise InputValidationError("Пилот требует последовательность ID сессий")
        chosen = list(selected_ids)
        if (
            not 0 < len(chosen) <= prepared_selected_size
            or len(set(chosen)) != len(chosen)
            or not set(chosen) <= prepared_ids
            or chosen != sorted(selected, key=lambda sid: session_key(sid, 17))[: len(chosen)]
        ):
            raise InputValidationError("ID пилота должны быть первыми N по хешу seed 17")
        selected = chosen
    chosen_ids = set(selected)
    answers_info = table_info(answers_path, ReplayRecord)
    access = DatasetAccess(
        source,
        code_sha256=prepared["source_sha256"],
        run_id=prepared["run_id"],
        scope=prepared["scope"],
        d_admission=d_admission,
        selection_admission=selection_admission,
    )
    prefixes = access.read(
        f"datasets/{period_name}/prefixes.parquet", PreparedRequest, purpose="prefix"
    )
    targets = access.read(f"datasets/{period_name}/targets.parquet", TargetRecord, purpose="labels")
    scanned_prefixes: set[str] = set()
    scanned_targets: set[str] = set()

    def included(group: tuple[str, tuple[Any, ...]], scanned: set[str]) -> bool:
        check_budget()
        sid, values = group
        if sid not in prepared_ids or sid in scanned or len(values) != 1:
            raise InputValidationError("Общая таблица не соответствует полной выборке подготовки")
        scanned.add(sid)
        return sid in chosen_ids

    # Дочитываются обе таблицы целиком, включая штатную конечную проверку хешей.
    prefixes = (group for group in prefixes if included(group, scanned_prefixes))
    targets = (group for group in targets if included(group, scanned_targets))
    answers = iter_sessions(
        answers_path,
        ReplayRecord,
        max_session_records=len(policy_names),
        max_session_bytes=plan.max_session_bytes,
    )
    period = next(period for period in plan.periods if period.name == period_name)
    inference = evaluation_plan["intervals"]
    sku_slices = build_sku_slices(history)
    catalog = set(history.catalog.skus)
    paired_rows: dict[str, list[EvaluationRow]] = {name: [] for name in policy_names}
    times: list[int] = []
    hits: dict[str, list[int]] = {name: [] for name in policy_names}
    seen: set[str] = set()
    output_path = destination / "replay/sessions.parquet"
    with TableWriter(
        output_path, EvaluationRecord, batch_rows=plan.chunk_size, batch_bytes=plan.chunk_bytes
    ) as writer:
        for prefix_group, target_group, answer_group in zip_longest(prefixes, targets, answers):
            check_budget()
            if prefix_group is None or target_group is None or answer_group is None:
                raise InputValidationError("Разные наборы сессий в префиксах, целях и ответах")
            sid = prefix_group[0]
            if (
                sid != target_group[0]
                or sid != answer_group[0]
                or sid not in chosen_ids
                or sid in seen
                or len(prefix_group[1]) != 1
                or len(target_group[1]) != 1
            ):
                raise InputValidationError(
                    "Нет единственной согласованной сессии/цели/первого запроса"
                )
            request, target = prefix_group[1][0], target_group[1][0]
            if (
                request.session_id != sid
                or request.source_namespace != plan.source_namespace
                or request.t_q != request.a_q
                or request.ordinal != 1
                or not period.start_ms
                <= request.t_q
                < period.end_ms - plan.task.horizon_ms - plan.task.gap_ms
                or target.horizon_ms != plan.task.horizon_ms
            ):
                raise InputValidationError("Запрос/цель не соответствует плану и календарю оценки")
            saved = {record.policy: record for record in answer_group[1]}
            if len(answer_group[1]) != len(policy_names) or set(saved) != set(policy_names):
                raise InputValidationError(
                    "Для каждой сессии нужны все запланированные политики без повторов"
                )
            pair_details = {
                "session_id": sid,
                "t_q": request.t_q,
                "k": evaluation_plan["k"],
                "purchases": [{"sku": sku} for sku in target.purchases],
            }
            evaluated = []
            for policy in policy_names:
                record = saved[policy]
                diagnostic = _diagnostics(record, evaluation_plan["diagnostic_depth"])
                evaluation, details = evaluate_session(
                    request,
                    record.result,
                    target,
                    catalog_skus=catalog,
                    calendar_cell=(request.t_q - period.start_ms) // inference["grid_step_ms"],
                    k=evaluation_plan["k"],
                    sku_slices=sku_slices,
                    source_lists=diagnostic["source_lists"],
                    feature_mask_counts=diagnostic["feature_mask_counts"],
                    feature_rows=diagnostic["feature_rows"],
                    mask_reason=diagnostic["mask_reason"],
                )
                if policy == "GCS+Rk":
                    sensitivity = _sensitivity_row(
                        diagnostic["sensitivity"], request, record.result, target, catalog
                    )
                    details["multiplicity_sensitivity"] = sensitivity
                if (
                    details["session_id"] != sid
                    or details["t_q"] != pair_details["t_q"]
                    or details["k"] != pair_details["k"]
                    or {value["sku"] for value in details["purchases"]} != set(target.purchases)
                ):
                    raise InputValidationError("Диагностика пары не соответствует общим целям")
                paired_rows[policy].append((evaluation, pair_details))
                if evaluation.b_s == 1:
                    if evaluation.hit_s is None:
                        raise InputValidationError(
                            "Неизвестное попадание сессии с зарегистрированной целью"
                        )
                    hits[policy].append(evaluation.hit_s)
                    if policy == "GCS+Rk":
                        times.append(request.t_q)
                evaluated.append(
                    EvaluationRecord(
                        sid, policy, evaluation, canonical_json_bytes(details).decode("utf-8")
                    )
                )
            writer.append(sid, evaluated)
            seen.add(sid)
        if seen != chosen_ids:
            raise InputValidationError("Ответы не покрывают всю закреплённую выборку")
        if scanned_prefixes != prepared_ids or scanned_targets != prepared_ids:
            raise InputValidationError("Неполная общая таблица подготовки")
        if (
            sha256_file(source / "reports/preparation.json") != report_hash
            or table_info(answers_path, ReplayRecord) != answers_info
        ):
            raise InputValidationError("Входы изменились во время оценки")
        check_budget()
    # Подробные словари больше не нужны: сводки читают уже закрытую таблицу.
    output_info = table_info(output_path, EvaluationRecord, period_name, "evaluation")
    accumulators = {
        policy: MetricsAccumulator(
            n_eligible=period_counts["eligible"],
            selected_size=len(selected),
            exclusions=prepared["exclusions"],
            k=evaluation_plan["k"],
        )
        for policy in policy_names
    }

    def sensitivity_rows() -> Iterator[dict[str, Any]]:
        # Тот же проход обновляет три сводки и отдаёт диагностику кратностей.
        for record in _evaluation_rows(output_path, plan, check_budget, policy_names):
            details = record.parsed_json
            accumulators[record.policy].add(record.evaluation, details)
            if record.policy == "GCS+Rk":
                yield details["multiplicity_sensitivity"]

    sensitivity_summary = _aggregate_sensitivity(sensitivity_rows())
    policies = {policy: accumulator.finish() for policy, accumulator in accumulators.items()}
    if len({policies[name]["counts"]["N"] for name in policy_names}) != 1:
        raise InputValidationError("Политики имеют разные целевые знаменатели")
    pairs = {
        "GCS+Rk-" + name: paired_metrics(paired_rows["GCS+Rk"], paired_rows[name])
        for name in policy_names[1:]
    }
    intervals = {}
    estimates = {name: (name, None) for name in policy_names}
    estimates.update({"GCS+Rk-" + name: ("GCS+Rk", name) for name in policy_names[1:]})
    for name, (left, right) in estimates.items():
        check_budget()
        assumptions = inference["dependence"][name]
        metric, details = calendar_intervals(
            times,
            hits[left],
            other_hit=hits[right] if right is not None else None,
            start_ms=period.start_ms,
            end_ms=period.end_ms,
            horizon_ms=plan.task.horizon_ms,
            gap_ms=plan.task.gap_ms,
            grid_step_ms=inference["grid_step_ms"],
            block_cells=inference["block_cells"],
            sensitivity_block_cells=tuple(inference["sensitivity_block_cells"]),
            dependence_assumptions=tuple(assumptions["assumptions"]),
            dependence_supported=assumptions["supported"],
            half_width_fraction=inference["half_width_fraction"][name],
        )
        intervals[name] = dict(metric=asdict(metric), details=details)
        point = policies[name]["metrics"]["Hit@20"] if name in policy_names else pairs[name]
        if metric.value != point["value"] or metric.denominator != point["denominator"]:
            raise InputValidationError("Точечная и интервальная оценки не совпали")
    if (
        sha256_file(source / "reports/preparation.json") != report_hash
        or table_info(answers_path, ReplayRecord) != answers_info
        or table_info(output_path, EvaluationRecord, period_name, "evaluation") != output_info
    ):
        raise InputValidationError("Входы или таблица оценок изменились во время агрегации")
    check_budget()
    frequencies = sorted(int(value) for value in history.statistics.detail_counts)
    return dict(
        status="completed",
        period=period_name,
        evaluation_protocol=EVALUATION_PROTOCOL,
        replay_mode="ideal_deterministic",
        telemetry_completeness="unknown",
        S_offline="not_run",
        business_effect="not_observed",
        preparation=dict(
            run_id=prepared["run_id"],
            scope=prepared["scope"],
            source_sha256=prepared["source_sha256"],
            report_sha256=report_hash,
            plan_sha256=prepared["inputs"]["plan"],
        ),
        evaluation_plan=evaluation_plan,
        data=dict(
            **policies["GCS+Rk"]["counts"],
            selected_size=len(selected),
            prepared_selected_size=prepared_selected_size,
            evaluation_scope="pilot" if selected_ids is not None else "full",
            selected_ids=list(selected),
            N_admitted=period_counts["admitted"],
            exclusions=prepared["exclusions"],
        ),
        slice_plan=dict(
            request=["history", "search", "cart", "cart_state", "prefix_known"],
            target=[
                "target_known",
                "target_cart",
                "target_popularity",
                "target_history",
                "target_description",
                "target_image",
            ],
            historical_popularity_rule="nearest_rank_1/3_and_2/3_of_A_G_detail_counts_ties_kept_together",
            historical_popularity_thresholds=[
                frequencies[math.ceil(len(frequencies) / 3) - 1],
                frequencies[math.ceil(2 * len(frequencies) / 3) - 1],
            ]
            if frequencies
            else None,
        ),
        policies=policies,
        pairs=pairs,
        intervals=intervals,
        multiplicity_sensitivity=sensitivity_summary,
        tables={"replay/sessions.parquet": output_info},
        input_answers=answers_info,
        **(
            {
                "metadata_sensitivity": {
                    "policy": NO_METADATA_CONFIG_ID,
                    "confirmatory": False,
                    "comparison": "GCS+Rk-" + NO_METADATA_CONFIG_ID,
                    "selection": "not_used",
                }
            }
            if period_name == "D"
            else {}
        ),
        interpretation=dict(
            quality="conditional_on_registered_purchase",
            service_load="not_measured",
            training_cost="not_supplied_to_reporting",
            full_process_resources="reported_by_command",
            next_comparison="confirmation_requires_new_unused_period"
            if period_name == "C_select"
            else "review_final_result_without_retuning"
            if period_name == "D"
            else "not_selected_pilot_report_requires_review",
            incomplete_telemetry="not_corrected_by_bootstrap",
        ),
    )


def render_report(summary: dict[str, Any]) -> str:
    """Markdown содержит только значения сохранённого JSON, без повторной оценки."""
    lines = [
        f"# Историческая оценка: {summary.get('period')}",
        "",
        f"Протокол: `{summary.get('evaluation_protocol')}`. "
        f"Режим: `{summary.get('replay_mode')}`. Запуск: `{summary.get('run_id')}`.",
        "",
    ]
    if summary.get("policies"):
        lines.extend(
            [
                "| Политика | N_obs | N | Hit@20 | Интервал | Статус интервала | Причина |",
                "|---|---:|---:|---:|---|---|---|",
            ]
        )
        for name, policy in summary["policies"].items():
            counts = policy.get("counts", {})
            metric = policy.get("metrics", {}).get("Hit@20", {})
            interval = summary.get("intervals", {}).get(name, {}).get("metric", {})
            values = [
                name,
                counts.get("N_obs"),
                counts.get("N"),
                metric.get("value"),
                interval.get("interval"),
                interval.get("status"),
                interval.get("reason"),
            ]
            lines.append(
                "| "
                + " | ".join(
                    json.dumps(value, ensure_ascii=False, allow_nan=False) for value in values
                )
                + " |"
            )
        lines.append("")
    labels = {
        "data": "Данные и исключения",
        "policies": "Качество, источники, разрезы и стоимость",
        "pairs": "Парные сравнения",
        "intervals": "Интервалы и достаточность",
        "multiplicity_sensitivity": "Чувствительность к кратностям",
        "metadata_sensitivity": "Чувствительность к метаданным (не подтверждающая гипотеза)",
        "slice_plan": "Зафиксированные разрезы",
        "evaluation_plan": "План оценки",
        "interpretation": "Границы интерпретации и следующее действие",
    }
    identity = {key: value for key, value in summary.items() if key not in labels}
    for title, value in [
        ("Версии, привязки и статусы", identity),
        *((label, summary[key]) for key, label in labels.items() if key in summary),
    ]:
        lines.extend(
            [
                "## " + title,
                "",
                "```json",
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
