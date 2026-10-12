"""Согласованность отчётов и планов; без расчёта интервалов и обучения.

Эти проверки не удостоверяют правдивость измерений. Исполнитель отдельно
проверяет происхождение, хеши, замороженные планы и результаты B10–B18.
Полный отрицательный результат сохраняется: он создаёт только запрет допуска.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, TypeGuard

from coveo_recsys.contracts.common import INT64_MAX, ValidationIssue

GUARDRAILS = frozenset(
    {
        "error_fraction",
        "latency_ms",
        "empty_response_fraction",
        "purchase_logging_loss_fraction",
        "invalid_item_fraction",
    }
)
_STATUSES = {"passed", "failed", "insufficient"}
_ARMS = {"control", "treatment"}


def _choice(value: Any, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


def _finite(value: Any) -> TypeGuard[int | float]:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class _Check:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, path: str, code: str, message: str, layer: str = "semantic") -> None:
        self.issues.append(ValidationIssue(path, code, message, layer))

    def field(self, obj: dict[str, Any], key: str, path: str) -> Any:
        if key not in obj or obj[key] is None:
            self.add(path, "missing_field", "Нет обязательного значения", "results")
            return None
        return obj[key]

    def obj(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.add(
                path, "object_required", "Нужен объект", "results" if value is None else "semantic"
            )
            return {}
        return value

    def number(
        self, value: Any, path: str, *, low: float | None = None, high: float | None = None
    ) -> float | None:
        if (
            not _finite(value)
            or (low is not None and value < low)
            or (high is not None and value > high)
        ):
            self.add(
                path,
                "number",
                "Некорректное конечное число",
                "results" if value is None else "semantic",
            )
            return None
        return float(value)

    def count(self, value: Any, path: str, *, positive: bool = False) -> int | None:
        if type(value) is not int or not (int(positive) <= value <= INT64_MAX):
            self.add(
                path,
                "count",
                "Нужен допустимый целый счётчик",
                "results" if value is None else "semantic",
            )
            return None
        return value

    def boolean(self, value: Any, path: str) -> bool | None:
        if type(value) is not bool:
            self.add(
                path,
                "boolean",
                "Нужен логический признак",
                "results" if value is None else "semantic",
            )
            return None
        return value

    def text(self, value: Any, path: str) -> str | None:
        if not isinstance(value, str) or not value.strip():
            self.add(
                path, "text", "Нужна непустая строка", "results" if value is None else "semantic"
            )
            return None
        return value

    def timestamp(self, value: Any, path: str) -> datetime | None:
        try:
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError
            result = datetime.fromisoformat(value[:-1] + "+00:00")
            if result.utcoffset() != timedelta(0):
                raise ValueError
            return result
        except (ValueError, OverflowError):
            self.add(
                path,
                "utc_time",
                "Нужно корректное время UTC с Z",
                "results" if value is None else "semantic",
            )
            return None

    def schedule(self, value: Any, path: str) -> list[datetime]:
        if not isinstance(value, list) or not value:
            self.add(
                path,
                "schedule",
                "Нужно непустое расписание",
                "results" if value is None else "semantic",
            )
            return []
        dates = [self.timestamp(v, f"{path}.{i}") for i, v in enumerate(value)]
        known = [v for v in dates if v is not None]
        if len(known) == len(dates) and any(a >= b for a, b in zip(known, known[1:])):
            self.add(path, "schedule_order", "Моменты проверок должны строго возрастать")
        return known

    def binding(self, actual: Any, expected: Any, path: str) -> None:
        if expected is None or actual != expected:
            self.add(path, "plan_binding", "Значение не совпало с зарегистрированным планом")

    def verdict(self, actual: Any, declared: Any, expected: str, path: str) -> None:
        if not _choice(actual, _STATUSES):
            self.add(
                path,
                "status",
                "Нужен завершённый статус результата",
                "results" if actual is None else "semantic",
            )
        self.binding(actual, declared, path)
        # Неполный или противоречивый отчёт не получает вычисленного допуска.
        if not any(issue.layer in {"semantic", "results"} for issue in self.issues):
            if actual != expected:
                self.add(path, "status_contradiction", "Статус противоречит измерениям и правилам")
        if expected != "passed":
            self.add(path, expected, "Результат не разрешает приёмку", "gate")


def _interval(
    check: _Check, interval: Any, details: dict[str, Any], path: str, method: Any, confidence: Any
) -> tuple[bool, float | None, float | None]:
    valid = check.boolean(details.get("valid"), path + ".valid")
    check.binding(details.get("method"), method, path + ".method")
    level = check.number(details.get("confidence_level"), path + ".confidence_level", low=0, high=1)
    if level is not None and (
        not _finite(confidence) or not math.isclose(level, confidence, rel_tol=0, abs_tol=1e-12)
    ):
        check.add(path, "confidence_binding", "Уровень интервала отличается от плана")
    if valid is False:
        check.text(details.get("reason"), path + ".reason")
    if interval is None:
        if valid is True:
            check.add(path, "missing_interval", "Пригодный интервал не может отсутствовать")
        return False, None, None
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        check.add(path, "interval_shape", "Нужны две границы или null")
        return False, None, None
    lower = check.number(interval[0], path + ".lower")
    upper = check.number(interval[1], path + ".upper")
    if lower is not None and upper is not None and lower > upper:
        check.add(path, "interval_order", "Нижняя граница превышает верхнюю")
    return valid is True, lower, upper


def validate_final_evaluation(
    report: dict[str, Any], manifest: dict[str, Any]
) -> tuple[ValidationIssue, ...]:
    """Проверяет post_D, в том числе достаточность и согласованность статуса."""
    c = _Check()
    inf = c.obj(manifest.get("inference"), "inference")
    c.binding(manifest.get("evaluation_protocol"), "observed_purchases_v1", "evaluation_protocol")
    c.binding(
        report.get("evaluation_protocol"),
        manifest.get("evaluation_protocol"),
        "evaluation_protocol",
    )
    c.binding(inf.get("precision_unit"), "fraction", "inference.precision_unit")
    verification = c.obj(manifest.get("verification"), "verification")
    evidence = c.obj(manifest.get("evidence"), "evidence")
    c.binding(
        report.get("evaluation_plan_sha256"),
        evidence.get("evaluation_plan"),
        "evaluation_plan_sha256",
    )
    metrics = c.obj(report.get("metrics"), "metrics")
    intervals = c.obj(report.get("intervals"), "intervals")
    info = c.obj(report.get("information"), "information")
    details = c.obj(report.get("interval_details"), "interval_details")
    statuses = c.obj(report.get("statuses"), "statuses")
    observation = c.obj(report.get("observation"), "observation")
    sensitivity = c.obj(report.get("metadata_sensitivity"), "metadata_sensitivity")
    published = c.boolean(
        observation.get("published_rows_complete"), "observation.published_rows_complete"
    )
    calendar = c.boolean(
        observation.get("calendar_admission_checked"), "observation.calendar_admission_checked"
    )
    c.binding(
        observation.get("telemetry_completeness"), "unknown", "observation.telemetry_completeness"
    )
    if "maturity" in report or {"POR", "DeltaPOR"} & (metrics.keys() | intervals.keys()):
        c.add("report", "old_protocol", "Отчёт содержит поля прежней постановки")
    evaluated = c.boolean(sensitivity.get("evaluated"), "metadata_sensitivity.evaluated")
    violations = c.field(report, "contract_violations", "contract_violations")
    if not isinstance(violations, list) or not all(
        isinstance(x, str) and x.strip() for x in violations
    ):
        c.add("contract_violations", "violations", "Нужен список описаний нарушений")
        violations = []
    n = c.count(metrics.get("N"), "metrics.N")
    n_obs = c.count(metrics.get("N_obs"), "metrics.N_obs")
    if n is not None and n_obs is not None and n > n_obs:
        c.add("metrics.N", "sample_count", "Целевых сессий больше выбранных наблюдаемых сессий")
    dataset = c.obj(manifest.get("dataset"), "dataset")
    c.binding(dataset.get("telemetry_completeness"), "unknown", "dataset.telemetry_completeness")
    sampling = dataset.get("session_sampling", {})
    totals = sampling.get("session_count_by_period", {}) if isinstance(sampling, dict) else {}
    if isinstance(totals, dict) and "period_D" in totals:
        selected = c.count(
            totals["period_D"], "dataset.session_sampling.session_count_by_period.period_D"
        )
        if n_obs is not None and selected is not None and n_obs > selected:
            c.add(
                "metrics.N_obs", "sample_count", "Наблюдаемых сессий больше выбранной подвыборки D"
            )
    if not _choice(inf.get("primary_estimate_type"), {"single_policy", "paired"}):
        c.add("inference.primary_estimate_type", "estimate_type", "Неизвестный вид основной оценки")
    single = inf.get("primary_estimate_type") == "single_policy"
    key, categories = ("Hit@20", {"0", "1"}) if single else ("DeltaHit@20", {"-1", "0", "1"})
    counts = c.obj(info.get("category_counts"), "information.category_counts")
    blocks = c.obj(info.get("category_blocks"), "information.category_blocks")
    if set(counts) != categories or set(blocks) != categories:
        c.add("information", "categories", "Нужны все категории зарегистрированного исхода")
    full = c.count(info.get("full_blocks"), "information.full_blocks")
    nonempty = c.count(info.get("nonempty_blocks"), "information.nonempty_blocks")
    undefined = c.count(info.get("undefined_repetitions"), "information.undefined_repetitions")
    dependence = c.boolean(
        info.get("dependence_assumptions_satisfied"), "information.dependence_assumptions_satisfied"
    )
    known_counts: dict[str, int] = {}
    known_blocks: dict[str, int] = {}
    for category in categories:
        count = c.count(counts.get(category), f"information.category_counts.{category}")
        block_count = c.count(blocks.get(category), f"information.category_blocks.{category}")
        if count is not None:
            known_counts[category] = count
        if block_count is not None:
            known_blocks[category] = block_count
        if (
            count is not None
            and block_count is not None
            and (block_count > count or (nonempty is not None and block_count > nonempty))
        ):
            c.add(
                "information.category_blocks",
                "block_count",
                "Число блоков несовместимо с числом сессий",
            )
    if n is not None and len(known_counts) == len(categories) and sum(known_counts.values()) != n:
        c.add("metrics.N", "denominator", "Категории не образуют полный знаменатель")
    if (
        full is not None
        and nonempty is not None
        and (nonempty > full or (n is not None and nonempty > n))
    ):
        c.add("information.nonempty_blocks", "block_count", "Неверное число непустых блоков")
    planned, length = inf.get("planned_cells"), inf.get("block_cells")
    if type(planned) is int and type(length) is int and length > 0 and full != planned // length:
        c.add("information.full_blocks", "grid_binding", "Число полных блоков не совпало с сеткой")
    repetitions = inf.get("bootstrap_repetitions")
    if undefined is not None and type(repetitions) is int and undefined > repetitions:
        c.add(
            "information.undefined_repetitions",
            "count",
            "Неопределённых повторений больше общего числа",
        )
    estimate = metrics.get(key)
    if n == 0:
        if key not in metrics or estimate is not None:
            c.add(f"metrics.{key}", "unknown_estimate", "При N=0 оценка должна быть null")
    elif n is not None:
        value = c.number(estimate, f"metrics.{key}", low=0 if single else -1, high=1)
        if value is not None and set(known_counts) == categories:
            expected = (known_counts["1"] - (0 if single else known_counts["-1"])) / n
            if not math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-10):
                c.add(f"metrics.{key}", "estimate_counts", "Оценка не совпала со счётчиками")
    if key not in intervals:
        c.add(f"intervals.{key}", "missing_field", "Нет результата основного интервала", "results")
    alpha = inf.get("alpha")
    confidence = 1 - alpha if _finite(alpha) else None
    valid, lower, upper = _interval(
        c, intervals.get(key), details, f"intervals.{key}", inf.get("primary_interval"), confidence
    )
    if valid and lower is not None and upper is not None and lower == upper:
        c.add(
            "interval_details.valid",
            "degenerate_interval",
            "Вырожденный бутстрэп-интервал непригоден",
        )
    if lower is not None and upper is not None and (lower < (0 if single else -1) or upper > 1):
        c.add("intervals", "interval_range", "Бутстрэп-границы вне области исхода")
    threshold = inf.get("single_policy_half_width_fraction" if single else "half_width_fraction")
    h = c.number(threshold, "inference.half_width", low=0, high=1)
    if h == 0:
        c.add("inference.half_width", "half_width", "Порог полуширины должен быть положительным")
    supported = sum(
        known_counts.get(k, 0) >= 100 and known_blocks.get(k, 0) >= 20 for k in categories
    )
    insufficient = (
        not n
        or full is None
        or full < 20
        or nonempty is None
        or nonempty < 20
        or supported < 2
        or undefined != 0
        or dependence is not True
        or not valid
        or lower is None
        or upper is None
        or lower == upper
        or h is None
        or (
            lower is not None
            and upper is not None
            and h is not None
            and (upper - lower) / 2 > h
            and not math.isclose((upper - lower) / 2, h, rel_tol=1e-12, abs_tol=0)
        )
    )
    expected_status = (
        "failed"
        if violations or published is False or calendar is False or evaluated is False
        else ("insufficient" if insufficient else "passed")
    )
    if expected_status != "passed":
        c.text(report.get("reason"), "reason")
    c.verdict(
        statuses.get("offline_status"),
        verification.get("offline_status"),
        expected_status,
        "statuses.offline_status",
    )
    return tuple(c.issues)


def validate_online_plan(manifest: dict[str, Any]) -> tuple[ValidationIssue, ...]:
    """Проверяет численные и временные условия плана до назначения вариантов."""
    c = _Check()
    online = c.obj(manifest.get("online_experiment"), "online_experiment")
    threshold = c.number(online.get("business_threshold"), "business_threshold", low=0)
    unit = online.get("assignment_unit")
    if not _choice(unit, {"session", "stable_key"}):
        c.add("assignment_unit", "assignment_unit", "Неизвестная единица назначения")
    c.binding(
        online.get("estimand"),
        "session_conversion" if unit == "session" else "entry_session_conversion",
        "estimand",
    )
    size = c.obj(online.get("sample_size_and_horizon"), "sample_size_and_horizon")
    c.count(size.get("n_per_arm"), "sample_size_and_horizon.n_per_arm", positive=True)
    alpha = c.number(size.get("alpha"), "sample_size_and_horizon.alpha", low=0, high=1)
    power = c.number(size.get("power"), "sample_size_and_horizon.power", low=0, high=1)
    p0 = c.number(
        size.get("baseline_probability"),
        "sample_size_and_horizon.baseline_probability",
        low=0,
        high=1,
    )
    delta = c.number(size.get("delta_plan"), "sample_size_and_horizon.delta_plan", low=0)
    if (
        alpha in (0, 1)
        or power in (0, 1)
        or p0 in (0, 1)
        or (delta is not None and threshold is not None and delta <= threshold)
        or (p0 is not None and delta is not None and p0 + delta / 1000 >= 1)
    ):
        c.add(
            "sample_size_and_horizon", "power_plan", "Неверные вероятности или планируемый эффект"
        )
    for key in ("interval_method", "assumptions_justification", "sample_size_justification"):
        c.text(size.get(key), "sample_size_and_horizon." + key)
    missing_policy = size.get("missing_outcome_policy")
    if not _choice(missing_policy, {"require_complete", "registered_bounds"}):
        c.add(
            "sample_size_and_horizon.missing_outcome_policy",
            "missing_policy",
            "Нужна политика неизвестных исходов",
        )
    if missing_policy == "registered_bounds":
        c.text(
            size.get("missing_outcome_justification"),
            "sample_size_and_horizon.missing_outcome_justification",
        )
    start = c.timestamp(size.get("start_utc"), "sample_size_and_horizon.start_utc")
    end = c.timestamp(size.get("end_utc"), "sample_size_and_horizon.end_utc")
    analysis = c.timestamp(
        size.get("analysis_not_before_utc"), "sample_size_and_horizon.analysis_not_before_utc"
    )
    task = c.obj(manifest.get("task"), "task")
    horizon = c.count(task.get("horizon_seconds"), "task.horizon_seconds")
    inactivity = c.count(task.get("inactivity_seconds"), "task.inactivity_seconds")
    if start is not None and end is not None and start >= end:
        c.add("sample_size_and_horizon", "horizon", "Набор должен иметь положительную длительность")
    if end is not None and analysis is not None and horizon is not None and inactivity is not None:
        try:
            if analysis < end + timedelta(seconds=horizon + inactivity):
                c.add(
                    "sample_size_and_horizon",
                    "maturity_horizon",
                    "Анализ назначен до созревания последнего исхода",
                )
        except OverflowError:
            c.add("sample_size_and_horizon", "horizon", "Горизонт не представим")
    srm = c.obj(online.get("srm_plan"), "srm_plan")
    for key in (
        "assignment_mechanism",
        "cohort_membership_rule",
        "conditional_null_justification",
        "test_method",
        "familywise_error_control_justification",
    ):
        c.text(srm.get(key), "srm_plan." + key)
    schedule = c.schedule(srm.get("schedule_utc"), "srm_plan.schedule_utc")
    count = c.count(srm.get("number_of_checks"), "srm_plan.number_of_checks", positive=True)
    if (
        count is not None
        and isinstance(srm.get("schedule_utc"), list)
        and count != len(srm["schedule_utc"])
    ):
        c.add("srm_plan", "schedule_count", "L не совпало с числом календарных проверок")
    if srm.get("alpha") != 0.001:
        c.add("srm_plan.alpha", "srm_alpha", "Нормативный общий уровень SRM равен 0.001")
    probability = c.number(
        srm.get("expected_treatment_probability"),
        "srm_plan.expected_treatment_probability",
        low=0,
        high=1,
    )
    adaptive = srm.get("membership_uses_assignment_or_exposure_history", False)
    c.boolean(adaptive, "srm_plan.membership_uses_assignment_or_exposure_history")
    standard = srm.get("assignment_mechanism") == "independent_bernoulli" and adaptive is False
    if standard:
        c.binding(srm.get("test_method"), "exact_two_sided_binomial", "srm_plan.test_method")
        c.binding(probability, 0.5, "srm_plan.expected_treatment_probability")
    else:
        c.text(
            srm.get("design_specific_test_justification"),
            "srm_plan.design_specific_test_justification",
        )
        if srm.get("test_method") == "exact_two_sided_binomial":
            c.add(
                "srm_plan.test_method",
                "adaptive_srm",
                "Биномиальный тест не обоснован фактическим механизмом",
            )
    if probability in (0, 1):
        c.add(
            "srm_plan.expected_treatment_probability",
            "allocation",
            "Обе группы должны иметь положительную вероятность",
        )
    strata = c.field(srm, "strata", "srm_plan.strata")
    if not isinstance(strata, list):
        c.add("srm_plan.strata", "strata", "Нужен список страт, возможно пустой")
        strata = []
    identifiers: set[str] = set()
    for index, raw in enumerate(strata):
        path = f"srm_plan.strata.{index}"
        stratum = c.obj(raw, path)
        identifier = c.text(stratum.get("stratum_id"), path + ".stratum_id")
        if identifier is not None:
            if identifier in identifiers:
                c.add(path, "duplicate_stratum", "Повтор идентификатора страты")
            identifiers.add(identifier)
        for key in (
            "membership_rule_version",
            "conditional_null_justification",
            "missing_value_policy",
        ):
            c.text(stratum.get(key), path + "." + key)
        digest = stratum.get("membership_rule_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(x not in "0123456789abcdef" for x in digest)
        ):
            c.add(path, "rule_hash", "Нужен SHA-256 правила принадлежности")
        fields = stratum.get("pre_assignment_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(x, str) and x.strip() for x in fields)
        ):
            c.add(path, "stratum_fields", "Нужен список полей, доступных до назначения")
        p = c.number(
            stratum.get("expected_treatment_probability"),
            path + ".expected_treatment_probability",
            low=0,
            high=1,
        )
        if p in (0, 1) or (standard and p != 0.5):
            c.add(path, "stratum_allocation", "Неверная доля назначения в страте")
    guards = c.obj(online.get("guardrail_margins"), "guardrail_margins")
    if not GUARDRAILS <= set(guards):
        c.add(
            "guardrail_margins",
            "guardrail_set",
            "Не заданы все обязательные защитные показатели",
            "results",
        )
    for name, raw in guards.items():
        path = f"guardrail_margins.{name}"
        guard = c.obj(raw, path)
        c.number(guard.get("limit"), path + ".limit")
        if not _choice(guard.get("comparison"), {"le", "ge"}):
            c.add(path, "comparison", "Нужно правило le либо ge")
        for key in ("unit", "denominator", "responsible_action"):
            c.text(guard.get(key), path + "." + key)
        c.count(guard.get("window_seconds"), path + ".window_seconds", positive=True)
        c.count(guard.get("minimum_count"), path + ".minimum_count", positive=True)
        schedule += c.schedule(guard.get("schedule_utc"), path + ".schedule_utc")
    if (
        start is not None
        and analysis is not None
        and any(t < start or t > analysis for t in schedule)
    ):
        c.add("schedule_utc", "schedule_horizon", "Проверка вне зарегистрированного периода опыта")
    return tuple(c.issues)


def validate_online_results(
    report: dict[str, Any], manifest: dict[str, Any]
) -> tuple[ValidationIssue, ...]:
    """Проверяет post_AB и выводит статус из заранее заданного правила."""
    c = _Check()
    online = c.obj(manifest.get("online_experiment"), "online_experiment")
    size = c.obj(online.get("sample_size_and_horizon"), "sample_size_and_horizon")
    verification = c.obj(manifest.get("verification"), "verification")
    for key in ("protocol_sha256", "assignment_unit", "estimand"):
        c.binding(report.get(key), online.get(key), key)
    assigned = c.obj(report.get("assigned_counts"), "assigned_counts")
    outcomes = c.obj(report.get("outcomes"), "outcomes")
    unknown = c.obj(report.get("unknown_outcomes"), "unknown_outcomes")
    if any(set(x) != _ARMS for x in (assigned, outcomes, unknown)):
        c.add("assigned_counts", "arms", "Нужны обе назначенные группы без дополнительных групп")
    numbers: dict[str, tuple[int, int, int]] = {}
    for arm in sorted(_ARMS):
        n = c.count(assigned.get(arm), f"assigned_counts.{arm}")
        k = c.count(outcomes.get(arm), f"outcomes.{arm}")
        missing = c.count(unknown.get(arm), f"unknown_outcomes.{arm}")
        if n is not None and k is not None and missing is not None:
            numbers[arm] = (n, k, missing)
            if k + missing > n:
                c.add(
                    f"outcomes.{arm}",
                    "denominator",
                    "Покупки и неизвестные исходы превышают число назначенных",
                )
    flags = {
        name: c.boolean(report.get(name), name)
        for name in (
            "mature",
            "integrity_passed",
            "itt_complete",
            "assumptions_satisfied",
            "srm_passed",
        )
    }
    stopping = report.get("stopping_status")
    if not _choice(stopping, {"completed", "safety_stop", "srm_stop", "unplanned_stop"}):
        c.add("stopping_status", "stopping_status", "Неизвестное состояние остановки")
    mode = report.get("analysis_mode")
    if not _choice(mode, {"confirmatory", "descriptive"}):
        c.add("analysis_mode", "analysis_mode", "Неизвестный режим анализа")
    if stopping != "completed" and mode == "confirmatory":
        c.add(
            "analysis_mode",
            "stopped_confirmatory",
            "Прерванный основной опыт имеет только описательный анализ",
        )
    alpha = size.get("alpha")
    if "interval" in report and report["interval"] is None:
        # Нет рассчитанного интервала: его метод не объявляется выполненным.
        interval = {
            "valid": False,
            "reason": report.get("reason"),
            "method": size.get("interval_method"),
            "confidence_level": 1 - alpha if _finite(alpha) else None,
            "lower": None,
            "upper": None,
        }
    else:
        interval = c.obj(report.get("interval"), "interval")
        if "lower" not in interval or "upper" not in interval:
            c.add("interval", "missing_field", "Нужны границы либо явные null", "results")
    bounds = (
        None
        if interval.get("lower") is None and interval.get("upper") is None
        else [interval.get("lower"), interval.get("upper")]
    )
    valid, lower, upper = _interval(
        c,
        bounds,
        interval,
        "interval",
        size.get("interval_method"),
        1 - alpha if _finite(alpha) else None,
    )
    if any(n == 0 for n, _, _ in numbers.values()) and valid:
        c.add("interval.valid", "empty_arm", "Пустая группа не допускает пригодного интервала")
    has_unknown = any(missing > 0 for _, _, missing in numbers.values())
    if (
        has_unknown
        and size.get("missing_outcome_policy") == "registered_bounds"
        and len(numbers) == 2
    ):
        n0, k0, m0 = numbers["control"]
        n1, k1, m1 = numbers["treatment"]
        if n0 > 0 and n1 > 0:
            missing_bounds = c.obj(report.get("missing_outcome_bounds"), "missing_outcome_bounds")
            expected_bounds = {
                "lower": 1000 * (k1 / n1 - (k0 + m0) / n0),
                "upper": 1000 * ((k1 + m1) / n1 - k0 / n0),
            }
            for key, expected in expected_bounds.items():
                value = c.number(missing_bounds.get(key), "missing_outcome_bounds." + key)
                if value is not None and not math.isclose(
                    value, expected, rel_tol=1e-8, abs_tol=1e-10
                ):
                    c.add(
                        "missing_outcome_bounds",
                        "missing_bounds",
                        "Диапазон не совпал с полными знаменателями ITT",
                    )
    guards = c.obj(online.get("guardrail_margins"), "guardrail_margins")
    results = c.obj(report.get("guardrail_results"), "guardrail_results")
    if set(results) != set(guards):
        c.add(
            "guardrail_results",
            "guardrail_set",
            "Нужны результаты всех и только зарегистрированных показателей",
            "results",
        )
    guard_statuses: list[str] = []
    for name, rule_value in guards.items():
        path = f"guardrail_results.{name}"
        rule = c.obj(rule_value, "guardrail_margins." + name)
        result = c.obj(results.get(name), path)
        for key in ("unit", "limit", "comparison"):
            c.binding(result.get(key), rule.get(key), path + "." + key)
        count = c.count(result.get("count"), path + ".count")
        value = result.get("value")
        if "value" not in result:
            c.add(path + ".value", "missing_field", "Нет измерения или явного null", "results")
        if value is None:
            c.text(result.get("reason"), path + ".reason")
        else:
            c.number(value, path + ".value")
        minimum, limit = rule.get("minimum_count"), rule.get("limit")
        enough = count is not None and type(minimum) is int and count >= minimum
        if not enough or value is None or not _finite(value) or not _finite(limit):
            status = "insufficient"
        else:
            passed = value <= limit if rule.get("comparison") == "le" else value >= limit
            status = "passed" if passed else "failed"
        if result.get("status") != status:
            c.add(
                path + ".status",
                "guardrail_status",
                "Статус показателя противоречит измерению/объёму",
            )
        guard_statuses.append(status)
    violation = (
        flags["integrity_passed"] is False
        or flags["itt_complete"] is False
        or flags["srm_passed"] is False
        or "failed" in guard_statuses
    )
    insufficient = (
        flags["mature"] is not True
        or flags["assumptions_satisfied"] is not True
        or mode != "confirmatory"
        or stopping != "completed"
        or not valid
        or lower is None
        or upper is None
        or "insufficient" in guard_statuses
        or len(numbers) != 2
        or any(n == 0 for n, _, _ in numbers.values())
        or (has_unknown and size.get("missing_outcome_policy") != "registered_bounds")
    )
    threshold = c.number(online.get("business_threshold"), "business_threshold", low=0)
    if violation:
        expected_status = "failed"
    elif insufficient or threshold is None:
        expected_status = "insufficient"
    else:
        assert lower is not None and upper is not None
        expected_status = (
            "passed" if lower > threshold else ("failed" if upper <= threshold else "insufficient")
        )
    if expected_status != "passed":
        c.text(report.get("reason"), "reason")
    c.verdict(
        report.get("business_status"),
        verification.get("business_status"),
        expected_status,
        "business_status",
    )
    return tuple(c.issues)
