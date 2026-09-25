"""Независимые неравенства и множества целей, без проверяемой реализации."""


def reference_targets(
    rows,
    periods,
    horizon,
    gap,
    coverage_start,
    coverage_end,
    gaps=(),
    confirmed=True,
    max_requests=10,
):
    by_time = {}
    for time, kind, sku in rows:
        by_time.setdefault(time, []).append((kind, sku))
    times = sorted(by_time)
    requests = []
    start = None
    bought = False
    for time in times:
        actions = {kind for kind, _ in by_time[time]}
        bought |= "purchase" in actions
        if bought:
            continue
        if start is None and "detail" in actions:
            start = time
        if (
            start is not None
            and time <= start + horizon
            and len(requests) < max_requests
            and actions.intersection({"detail", "search"})
        ):
            requests.append(time)

    def covers(left, right):
        return (
            confirmed
            and coverage_start <= left
            and right <= coverage_end
            and all(b <= left or a >= right for a, b in gaps)
        )

    first, last = times[0], times[-1]
    close = last + gap if covers(last, last + gap) else None
    whole = covers(first, last + gap)
    mature = start is not None and covers(first, start + horizon + gap)
    tau = min(start + horizon, last) if mature else None
    period = next((p for p in periods if start is not None and p[1] <= start < p[2]), None)
    admitted = bool(
        period
        and mature
        and start + horizon + gap < period[2]
        and (period[0] != "A_G" or whole and close is not None and close < period[2])
    )
    if admitted:
        purchases = [
            (time, sku) for time, kind, sku in rows if kind == "purchase" and start < time <= tau
        ]
        expected = [
            tuple(
                sorted(
                    {sku for time, sku in purchases if query < time},
                    key=lambda v: v.encode("utf-8"),
                )
            )
            for query in requests
        ]
        b_s = int(bool(purchases))
    else:
        expected, b_s = [None] * len(requests), None
    return {
        "requests": requests,
        "close_ms": close,
        "mature": mature,
        "tau": tau,
        "admitted": admitted,
        "targets": expected,
        "b_s": b_s,
        "period": period[0] if period else None,
    }


def reference_cohort(
    rows,
    periods,
    horizon,
    gap,
    coverage_start,
    coverage_end,
    gaps=(),
    confirmed=True,
    max_requests=10,
):
    session = reference_targets(
        rows, periods, horizon, gap, coverage_start, coverage_end, gaps, confirmed, max_requests
    )
    first = min(time for time, _, _ in rows)
    anchors = {
        sku: min(time for time, kind, candidate in rows if kind == "detail" and candidate == sku)
        for _, kind, sku in rows
        if kind == "detail"
    }
    result = {}
    for sku, anchor in anchors.items():
        right = anchor + horizon
        complete = (
            confirmed
            and coverage_start <= first
            and right + gap <= coverage_end
            and all(b <= first or a >= right + gap for a, b in gaps)
        )
        mature = bool(
            session["admitted"]
            and session["period"] == "A_G"
            and right + gap < periods[0][2]
            and complete
        )
        bought = (
            int(
                any(
                    kind == "purchase" and candidate == sku and anchor < time <= right
                    for time, kind, candidate in rows
                )
            )
            if mature
            else None
        )
        result[sku] = (anchor, right, mature, bought)
    return result
