"""Календарные интервалы Hit и парной разности на зарегистрированных целях.

Сетка и три длины блока задаются до исходов. Численный интервал не означает
подтверждения предпосылок зависимости или выполнения ещё не заданной точности.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.common import require_int64
from coveo_recsys.contracts.training import MetricResult

REPETITIONS = 2000
SEED = 17
ALPHA = 0.05
MC_METHOD = "binomial_order_statistic_95pct"


def nearest_rank(values: Sequence[float] | NDArray[np.float64], probability: float) -> float:
    """Элемент ceil(p*n) без интерполяции; неопределённые значения не отбрасываются."""
    if not math.isfinite(probability) or not 0 < probability <= 1 or len(values) == 0:
        raise InputValidationError("Квантиль требует непустую выборку и 0 < p <= 1")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(ordered)):
        raise InputValidationError("Квантиль неопределённой выборки")
    return float(ordered[math.ceil(probability * len(ordered)) - 1])


def _calendar_sums(
    times: Sequence[int],
    outcomes: Sequence[int],
    start: int,
    step: int,
    cells: int,
    category_values: tuple[int, ...],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    counts = np.zeros(cells, dtype=np.int64)
    numerators = np.zeros(cells, dtype=np.int64)
    categories = np.zeros((len(category_values), cells), dtype=np.int64)
    for time, outcome in zip(times, outcomes, strict=True):
        cell = (time - start) // step
        counts[cell] += 1
        numerators[cell] += outcome
        categories[category_values.index(outcome), cell] += 1
    return counts, numerators, categories


def _bootstrap_repetitions(
    numerators: Sequence[int] | NDArray[np.int64],
    counts: Sequence[int] | NDArray[np.int64],
    block_cells: int,
    *,
    repetitions: int = REPETITIONS,
) -> NDArray[np.float64]:
    """Суммы окон эквивалентны склейке индексов, без матрицы B×J в памяти."""
    cells = len(counts)
    blocks = (cells + block_cells - 1) // block_cells
    last_length = cells - (blocks - 1) * block_cells
    numerator_prefix = np.concatenate(([0], np.cumsum(numerators, dtype=np.int64)))
    count_prefix = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    rng = np.random.Generator(np.random.PCG64(SEED))
    values = np.full(repetitions, np.nan, dtype=np.float64)
    for repetition in range(repetitions):
        starts = rng.integers(0, cells - block_cells + 1, size=blocks)
        ends = starts + block_cells
        ends[-1] = starts[-1] + last_length
        denominator = int((count_prefix[ends] - count_prefix[starts]).sum())
        if denominator:
            values[repetition] = (
                int((numerator_prefix[ends] - numerator_prefix[starts]).sum()) / denominator
            )
    return values


def _monte_carlo_error(values: NDArray[np.float64]) -> dict[str, Any]:
    """95%-диапазон каждого квантиля по биномиальным порядковым статистикам.

    При L=Q_.025(Bin(B,p)), U=Q_.975(Bin(B,p))+1 диапазон [X_(L),X_(U)]
    ограничивает погрешность конечного случайного моделирования, не исходных данных.
    Для закреплённых B=2000 и p=.025/.975 ранги строго внутри выборки.
    """
    ordered = np.sort(values)
    size = len(ordered)
    if size != REPETITIONS or not np.all(np.isfinite(ordered)):
        raise InputValidationError("Оценка MC-погрешности требует 2000 определённых повторений")
    result: dict[str, Any] = dict(method=MC_METHOD, confidence_level=0.95)
    for name, probability in (("lower", ALPHA / 2), ("upper", 1 - ALPHA / 2)):
        log_factorial = math.lgamma(size + 1)
        masses = np.array(
            [
                math.exp(
                    log_factorial
                    - math.lgamma(k + 1)
                    - math.lgamma(size - k + 1)
                    + k * math.log(probability)
                    + (size - k) * math.log1p(-probability)
                )
                for k in range(size + 1)
            ]
        )
        cumulative = np.cumsum(masses / masses.sum())
        lower = int(np.searchsorted(cumulative, 0.025))
        upper = int(np.searchsorted(cumulative, 0.975)) + 1
        lo, hi = float(ordered[lower - 1]), float(ordered[upper - 1])
        bound = nearest_rank(ordered, probability)
        result[name] = dict(
            probability=probability,
            ranks=[lower, upper],
            interval=[lo, hi],
            max_absolute_error=max(bound - lo, hi - bound),
        )
    return result


def _length_result(
    counts: NDArray[np.int64],
    numerators: NDArray[np.int64],
    categories: NDArray[np.int64],
    category_values: tuple[int, ...],
    block_cells: int,
    dependence_supported: bool | None,
    dependence_assumptions: tuple[str, ...],
    half_width_fraction: float | None,
) -> dict[str, Any]:
    cells, sessions = len(counts), int(counts.sum())
    full_blocks = cells // block_cells
    complete_cells = full_blocks * block_cells
    nonempty_blocks = int(np.count_nonzero(counts[:complete_cells].reshape(-1, block_cells).sum(1)))
    category_counts = categories.sum(axis=1)
    category_blocks = np.count_nonzero(
        categories[:, :complete_cells]
        .reshape(len(category_values), full_blocks, block_cells)
        .sum(2),
        axis=1,
    )
    supported = int(np.count_nonzero((category_counts >= 100) & (category_blocks >= 20)))
    repetitions = _bootstrap_repetitions(numerators, counts, block_cells)
    undefined = int(np.count_nonzero(np.isnan(repetitions)))
    interval = (
        None
        if undefined
        else (nearest_rank(repetitions, ALPHA / 2), nearest_rank(repetitions, 1 - ALPHA / 2))
    )
    estimate = int(numerators.sum()) / sessions if sessions else None
    variance = (
        sum(
            int(n) * (value - estimate) ** 2
            for n, value in zip(category_counts, category_values, strict=True)
        )
        / (sessions - 1)
        if sessions > 1 and estimate is not None
        else None
    )
    dependence = (
        None
        if dependence_supported is None
        else dependence_supported and bool(dependence_assumptions)
    )
    reasons = []
    if not sessions:
        reasons.append("no_target_sessions")
    if full_blocks < 20:
        reasons.append("insufficient_full_blocks")
    if nonempty_blocks < 20:
        reasons.append("insufficient_nonempty_blocks")
    if supported < 2:
        reasons.append("insufficient_outcome_categories")
    if variance is None or variance == 0:
        reasons.append("zero_empirical_variance")
    if undefined:
        reasons.append("undefined_bootstrap_repetitions")
    if interval is not None and interval[0] == interval[1]:
        reasons.append("degenerate_interval")
    if dependence is not True:
        reasons.append("dependence_assumptions_unconfirmed")
    interval_valid = not reasons
    half_width = (interval[1] - interval[0]) / 2 if interval is not None else None
    if half_width_fraction is None:
        precision_status = "not_planned"
        reasons.append("precision_not_planned")
    elif not interval_valid:
        precision_status = "not_evaluated"
    elif (
        half_width is not None
        and half_width > half_width_fraction
        and not math.isclose(
            half_width,
            half_width_fraction,
            rel_tol=1e-12,
            abs_tol=0,
        )
    ):
        precision_status = "insufficient"
        reasons.append("half_width_exceeds_plan")
    else:
        precision_status = "passed"
    return dict(
        block_cells=block_cells,
        value=estimate,
        interval=interval,
        interval_valid=interval_valid,
        half_width=half_width,
        precision_status=precision_status,
        status="insufficient" if reasons else "passed",
        reason=";".join(reasons) if reasons else None,
        reasons=reasons,
        empirical_variance=variance,
        information=dict(
            full_blocks=full_blocks,
            nonempty_blocks=nonempty_blocks,
            category_counts={
                str(value): int(n)
                for value, n in zip(category_values, category_counts, strict=True)
            },
            category_blocks={
                str(value): int(n)
                for value, n in zip(category_values, category_blocks, strict=True)
            },
            supported_categories=supported,
            undefined_repetitions=undefined,
            dependence_assumptions_satisfied=dependence,
        ),
        mc_error=None if undefined else _monte_carlo_error(repetitions),
    )


def calendar_intervals(
    times_ms: Sequence[int],
    hit: Sequence[int],
    *,
    start_ms: int,
    end_ms: int,
    horizon_ms: int,
    gap_ms: int,
    grid_step_ms: int,
    block_cells: int,
    sensitivity_block_cells: tuple[int, int],
    other_hit: Sequence[int] | None = None,
    dependence_assumptions: tuple[str, ...] = (),
    dependence_supported: bool | None = None,
    half_width_fraction: float | None = None,
) -> tuple[MetricResult, dict[str, Any]]:
    """Оценить только сессии с целью; для пары hit-other_hit в одном порядке.

    Неизвестные цели исключает вызывающий код до передачи данных; исход 0 означает
    наблюдаемый промах, а не отсутствие цели. Ошибочный план/исход вызывает ошибку;
    недостаточная информация сохраняет точку и получает статус insufficient.
    Предпосылки подтверждаются отдельно для данного функционала и разреза.
    """
    for value in (start_ms, end_ms, horizon_ms, gap_ms, grid_step_ms, block_cells):
        require_int64(value)
    width = end_ms - horizon_ms - gap_ms - start_ms
    if horizon_ms <= 0 or gap_ms < 0 or grid_step_ms <= 0 or width <= 0 or width % grid_step_ms:
        raise InputValidationError("Сетка должна целиком покрывать допустимое окно стартов")
    cells = width // grid_step_ms
    if not isinstance(sensitivity_block_cells, (tuple, list)) or len(sensitivity_block_cells) != 2:
        raise InputValidationError("Нужны две заранее выбранные длины чувствительности")
    lengths = (block_cells, *sensitivity_block_cells)
    for length in lengths:
        require_int64(length)
    if len(set(lengths)) != 3 or any(not 1 <= length <= cells for length in lengths):
        raise InputValidationError("Три разные длины блока должны принадлежать 1..J")
    if (
        dependence_supported is not None
        and type(dependence_supported) is not bool
        or not isinstance(dependence_assumptions, (tuple, list))
        or any(not isinstance(value, str) or not value.strip() for value in dependence_assumptions)
    ):
        raise InputValidationError("Неверное описание предпосылок зависимости")
    if half_width_fraction is not None and (
        type(half_width_fraction) not in (int, float)
        or not math.isfinite(half_width_fraction)
        or not 0 < half_width_fraction <= 1
    ):
        raise InputValidationError("Полуширина должна быть положительной долей не больше единицы")
    if len(times_ms) != len(hit) or other_hit is not None and len(other_hit) != len(hit):
        raise InputValidationError("Несогласованные пары времён и сессионных исходов")
    for time in times_ms:
        require_int64(time)
        if not start_ms <= time < start_ms + width:
            raise InputValidationError("Начало сессии вне допустимого календарного окна")
    for outcomes in (hit,) if other_hit is None else (hit, other_hit):
        if any(type(value) is not int or value not in (0, 1) for value in outcomes):
            raise InputValidationError("Нужны только определённые бинарные Hit целевых сессий")
    values = (
        list(hit)
        if other_hit is None
        else [first - second for first, second in zip(hit, other_hit, strict=True)]
    )
    category_values = (0, 1) if other_hit is None else (-1, 0, 1)
    counts, numerators, categories = _calendar_sums(
        times_ms,
        values,
        start_ms,
        grid_step_ms,
        cells,
        category_values,
    )
    results = [
        _length_result(
            counts,
            numerators,
            categories,
            category_values,
            length,
            dependence_supported,
            tuple(dependence_assumptions),
            half_width_fraction,
        )
        for length in lengths
    ]
    primary = results[0]
    info = primary["information"]
    metric = MetricResult(
        value=primary["value"],
        status=primary["status"],
        reason=primary["reason"],
        unit="fraction",
        denominator=len(hit),
        counters=tuple(
            (key, info[key])
            for key in (
                "full_blocks",
                "nonempty_blocks",
                "supported_categories",
                "undefined_repetitions",
            )
        ),
        interval=primary["interval"],
        interval_valid=primary["interval_valid"],
        assumptions=tuple(dependence_assumptions),
    )
    return metric, dict(
        method="calendar_moving_block_bootstrap",
        confidence_level=1 - ALPHA,
        estimate_type="single_policy" if other_hit is None else "paired",
        plan=dict(
            start_ms=start_ms,
            end_ms=end_ms,
            horizon_ms=horizon_ms,
            gap_ms=gap_ms,
            grid_step_ms=grid_step_ms,
            block_cells=block_cells,
            sensitivity_block_cells=list(sensitivity_block_cells),
            bootstrap_repetitions=REPETITIONS,
            bootstrap_seed=SEED,
            alpha=ALPHA,
            generator="numpy.random.Generator(PCG64)",
            numpy_version=np.__version__,
            random_stream="restart_seed_for_each_block_length",
            quantile_method="nearest_rank_x_ceil_p_n",
            mc_error_method=MC_METHOD,
            dependence_assumptions=list(dependence_assumptions),
            dependence_supported=dependence_supported,
            half_width_fraction=half_width_fraction,
            minimum_full_blocks=20,
            minimum_nonempty_blocks=20,
            minimum_category_sessions=100,
            minimum_category_blocks=20,
            minimum_supported_categories=2,
        ),
        grid=dict(start_ms=start_ms, end_ms=start_ms + width, step_ms=grid_step_ms, cells=cells),
        cells=[
            dict(
                index=index,
                start_ms=start_ms + index * grid_step_ms,
                end_ms=start_ms + (index + 1) * grid_step_ms,
                n=int(count),
                numerator=int(numerator),
                value=int(numerator) / int(count) if count else None,
            )
            for index, (count, numerator) in enumerate(zip(counts, numerators, strict=True))
        ],
        primary=primary,
        sensitivity=results[1:],
    )
