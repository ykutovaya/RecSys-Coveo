"""Независимые скалярные признаки базовой схемы: формулы, а не выход FeatureEngine."""

import math

from tests.reference.e3_retrieval_oracle import (
    SOURCES,
    graph_rows,
    item_rows,
    retrieval_oracle,
)


def feature_oracle(context, package, config, *, observed=None, source_runtime=None):
    selection = retrieval_oracle(
        context, package, config, observed=observed, source_runtime=source_runtime
    )
    items = item_rows(package, observed, config.vector_epsilon)
    edges = graph_rows(package)
    prefix = context.prefix
    last = prefix.full_last_detail
    views = dict(prefix.last_views)
    counts = {(kind, sku): count for kind, sku, count in prefix.counters}
    cart = {entry.sku: entry for entry in prefix.cart}
    result = {}
    for sku in selection["main"]:
        row = {}

        def put(name, value, mask=True, *, masked=False, logarithm=False):
            value = value if mask else 0.0
            row[name] = (math.log1p(value) if logarithm else value, bool(mask))
            if masked:
                row[name + "_mask"] = (float(mask), True)

        for source in SOURCES:
            source_list = selection["sources"][source]
            member = sku in source_list
            put("origin_" + source, float(member))
            put(
                "score_" + source,
                selection["values"][source][sku],
                selection["masks"][source][sku],
                masked=True,
            )
            rank_value = 1 / (2 + source_list.index(sku)) if member else 0.0
            put("rank_" + source, rank_value, member, masked=True)
        put("origin_fill", float(sku in selection["fill"]))
        for relation, pairs in edges.items():
            options = [
                (pairs[entry.sku, sku], entry.sku) for entry in last if (entry.sku, sku) in pairs
            ]
            options.sort(key=lambda option: (-option[0][0], option[1].encode()))
            strength, support = options[0][0] if options else (0.0, 0)
            put("last_" + relation, strength, bool(options), masked=True)
            put("support_" + relation, support, bool(options), masked=True, logarithm=True)
        for name in ("description", "image"):
            terms, denominator = [], 0
            for entry in last:
                if items[sku][name][1] and entry.sku in items and items[entry.sku][name][1]:
                    similarity = math.fsum(
                        a * b
                        for a, b in zip(items[sku][name][0], items[entry.sku][name][0], strict=True)
                    )
                    terms.append(entry.count * similarity)
                    denominator += entry.count
            put(
                "last_" + name,
                math.fsum(terms) / denominator if denominator else 0.0,
                denominator > 0,
                masked=True,
            )
            put("history_" + name + "_available", float(selection["history_available"][name]))
            put("item_" + name + "_available", float(items[sku][name][1]))
        put(
            "search_age",
            selection["search_age_ms"] / 1000,
            selection["available"]["search"],
            masked=True,
            logarithm=True,
        )
        put("detail_count", counts.get(("detail", sku), 0), logarithm=True)
        put(
            "detail_age",
            (context.t_q - views[sku]) / 1000 if sku in views else 0.0,
            sku in views,
            masked=True,
            logarithm=True,
        )
        state = cart[sku].state if sku in cart else "unknown"
        known_cart = state != "unknown"
        put("cart_add", float(state == "add"), known_cart, masked=True)
        put("cart_remove", float(state == "remove"), known_cart, masked=True)
        put("cart_unknown", float(state == "unknown"))
        last_action = cart[sku].last_unambiguous_ms if sku in cart else None
        put(
            "cart_age",
            (context.t_q - last_action) / 1000 if last_action is not None else 0.0,
            last_action is not None,
            masked=True,
            logarithm=True,
        )
        put("session_event_count", sum(counts.values()), logarithm=True)
        put(
            "session_distinct_items",
            len({key[1] for key, count in counts.items() if key[1] is not None and count > 0}),
            logarithm=True,
        )
        put(
            "session_add_count",
            sum(count for (kind, _), count in counts.items() if kind == "add"),
            logarithm=True,
        )
        first_time = prefix.first_event_time_ms
        put(
            "session_duration",
            (context.t_q - first_time) / 1000 if first_time is not None else 0.0,
            first_time is not None,
            masked=True,
            logarithm=True,
        )
        put("session_has_search", float(prefix.search is not None))
        paths = []
        if items[sku]["category"]:
            candidate_path = items[sku]["category"].split("/")
            for entry in last:
                if entry.sku in items and items[entry.sku]["category"]:
                    path = items[entry.sku]["category"].split("/")
                    length = 0
                    for left, right in zip(candidate_path, path):
                        if left != right:
                            break
                        length += 1
                    paths.append(
                        (length / max(1, len(candidate_path), len(path)), candidate_path == path)
                    )
        put(
            "category_similarity",
            max((value[0] for value in paths), default=0.0),
            bool(paths),
            masked=True,
        )
        put("category_match", float(any(value[1] for value in paths)), bool(paths), masked=True)
        price = items[sku]["price"]
        put("price_bucket", price or 0, price is not None, masked=True)
        price_differences = []
        if price is not None:
            for entry in last:
                if entry.sku in items and items[entry.sku]["price"] is not None:
                    price_differences.append((price - items[entry.sku]["price"], entry.count))
        denominator = sum(count for _, count in price_differences)
        for name, transform in (
            ("equal", lambda value: float(value == 0)),
            ("sign", lambda value: float((value > 0) - (value < 0))),
            ("absolute", abs),
        ):
            value = math.fsum(transform(delta) * count for delta, count in price_differences)
            put(
                "last_price_" + name,
                value / denominator if denominator else 0.0,
                denominator > 0,
                masked=True,
            )
        index = package.catalog.skus.index(sku)
        put(
            "historical_purchase_count",
            int(package.statistics.purchase_counts[index]),
            logarithm=True,
        )
        put("category", int(package.content.category_codes[index]))
        result[sku] = row
    return result
