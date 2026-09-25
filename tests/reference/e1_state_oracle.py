"""Малые эталоны Э1: прямые определения, без вызова проверяемых функций."""

from collections import Counter
from decimal import Decimal, localcontext


def search(payloads, dimension):
    """Исходные пары (вектор, причина) -> направление либо причина."""
    reasons = set()
    valid = set()
    with localcontext() as context:
        context.prec = 90
        for vector, reason in payloads:
            if reason is not None:
                reasons.add(reason)
            elif vector is None:
                reasons.add("vector_absent")
            elif len(vector) != dimension:
                reasons.add("vector_invalid")
            else:
                squared = sum((Decimal.from_float(x) ** 2 for x in vector), Decimal(0))
                norm = squared.sqrt()
                if norm <= Decimal.from_float(1e-12):
                    reasons.add("vector_zero")
                else:
                    valid.add(tuple(vector))
        if reasons:
            return None, min(reasons, key=lambda value: value.encode("utf-8"))
        if len(valid) != 1:
            return None, "search_ambiguous"
        vector = next(iter(valid))
        norm = sum((Decimal.from_float(x) ** 2 for x in vector), Decimal(0)).sqrt()
        return tuple(float(Decimal.from_float(x) / norm) for x in vector), None


def prefix(blocks, event_cutoff, availability_cutoff, config):
    """Полный перебор видимого журнала; история развёрнута по кратностям."""
    visible = sorted(
        (
            block
            for block in blocks
            if block.event_time_ms <= event_cutoff
            and block.available_time_ms <= availability_cutoff
        ),
        key=lambda block: block.event_time_ms,
    )
    counts = Counter()
    details = []
    last_detail = []
    last_views = {}
    purchased = set()
    cart = {}
    latest_search = None
    for block in visible:
        multiplicities = dict(block.raw_multiplicities)
        new_details = []
        actions = {}
        for event in block.events:
            n = multiplicities[event.event_id] if config.event_semantics == "multiset_v1" else 1
            counts[(event.event_type.value, event.sku)] += n
            if event.event_type.value == "detail":
                new_details.extend([(event.sku, block.event_time_ms)] * n)
                last_views[event.sku] = block.event_time_ms
            elif event.event_type.value == "purchase":
                purchased.add(event.sku)
            elif event.event_type.value in ("add", "remove"):
                actions.setdefault(event.sku, set()).add(event.event_type.value)
        details.extend(new_details)
        if new_details:
            last_detail = new_details
        for sku, kinds in actions.items():
            if len(kinds) == 1:
                cart[sku] = (next(iter(kinds)), block.event_time_ms, block.event_time_ms)
            else:
                cart[sku] = ("unknown", block.event_time_ms, cart.get(sku, (None, None, None))[2])
        if block.search_payloads:
            direction, reason = search(
                [(x.vector, x.reason) for x in block.search_payloads], config.query_dimension
            )
            latest_search = (
                block.block_id,
                block.event_time_ms,
                block.available_time_ms,
                direction,
                reason,
            )
    selected = sorted(details, key=lambda item: (-item[1], item[0].encode("utf-8")))

    def compress(values):
        frequencies = Counter(values)
        return tuple(
            (sku, time, count)
            for (sku, time), count in sorted(
                frequencies.items(), key=lambda item: (item[0][1], item[0][0].encode("utf-8"))
            )
        )

    return dict(
        version=len(visible),
        blocks=tuple(visible[-config.block_history_limit :]),
        first_event_time_ms=visible[0].event_time_ms if visible else None,
        detail_history=compress(selected[: config.detail_history_limit]),
        full_last_detail=compress(last_detail),
        counts=dict(counts),
        last_views=last_views,
        purchased=purchased,
        cart=cart,
        search=latest_search,
    )


def schedule(blocks, config):
    """Первый detail до зарегистрированной покупки, историческое a=t."""
    for block in sorted(blocks, key=lambda item: item.event_time_ms):
        actions = {event.event_type.value for event in block.events}
        if "purchase" in actions:
            return []
        if "detail" in actions:
            return [(block.event_time_ms, block.block_id, 1)]
    return []
