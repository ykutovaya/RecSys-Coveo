"""Независимые неравенства и множества зарегистрированных целей."""


def reference_targets(rows, periods, horizon, gap):
    by_time = {}
    for time, kind, sku in rows:
        by_time.setdefault(time, []).append((kind, sku))
    start = None
    for time in sorted(by_time):
        actions = {kind for kind, _ in by_time[time]}
        if "purchase" in actions:
            break
        if "detail" in actions:
            start = time
            break
    last = max(by_time)
    tau = min(start + horizon, last) if start is not None else None
    period = next((p for p in periods if start is not None and p[1] <= start < p[2]), None)
    admitted = bool(
        period
        and start + horizon + gap < period[2]
        and (period[0] != "A_G" or last + gap < period[2])
    )
    purchases = tuple(
        sorted(
            {
                sku
                for time, kind, sku in rows
                if admitted and kind == "purchase" and start < time <= tau
            },
            key=lambda value: value.encode("utf-8"),
        )
    )
    return {
        "requests": [] if start is None else [start],
        "tau": tau,
        "admitted": admitted,
        "targets": [] if start is None else [purchases],
        "period": period[0] if period else None,
    }
