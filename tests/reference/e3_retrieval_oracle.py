"""Прямой эталон Э3: списки и скалярные суммы без вычислительного ядра."""

import math

from tests.reference.e2_content_oracle import vector

SOURCES = (
    "detail_detail",
    "detail_add",
    "detail_purchase",
    "description",
    "image",
    "search",
    "recent",
    "popularity",
)


def item_rows(package, observed=None, epsilon=1e-12):
    result = {}
    for i, sku in enumerate(package.catalog.skus):
        code = int(package.content.category_codes[i])
        result[sku] = {
            name: (
                list(getattr(package.content, name).vectors[i]),
                bool(getattr(package.content, name).mask[i]),
            )
            for name in ("description", "image")
        }
        result[sku]["category"] = package.content.categories[code - 1] if code else None
        result[sku]["price"] = (
            int(package.content.price_bucket[i]) if package.content.price_mask[i] else None
        )
    for sku, record in (observed or {}).items():
        if sku in result:
            continue
        result[sku] = {}
        for name in ("description", "image"):
            dimension = package.metadata_contract[f"{name}_dimension"]
            values, mask, _ = vector(getattr(record, name), dimension, epsilon)
            result[sku][name] = values, mask
        result[sku]["category"] = record.category
        result[sku]["price"] = record.price_bucket
    return result


def graph_rows(package):
    rows = {}
    for graph in package.graphs.relations:
        rows[graph.relation] = {}
        for i, sku in enumerate(package.catalog.skus):
            for offset in range(int(graph.indptr[i]), int(graph.indptr[i + 1])):
                target = package.catalog.skus[int(graph.indices[offset])]
                rows[graph.relation][sku, target] = (
                    float(graph.rho[i]) * float(graph.probabilities[offset]),
                    int(graph.support[offset]),
                )
    return rows


def retrieval_oracle(context, package, config, *, observed=None, source_runtime=None):
    """Ожидаемые баллы, источники, штатный набор и отдельный резерв."""
    prefix = context.prefix
    skus = package.catalog.skus
    items = item_rows(package, observed, config.vector_epsilon)
    edges = graph_rows(package)
    disabled = source_runtime or {}
    history = []
    for entry in sorted(prefix.detail_history, key=lambda e: (-e.event_time_ms, e.sku.encode())):
        history.extend([(entry.sku, entry.event_time_ms)] * entry.count)
    history = history[: config.history_limit]
    weights = []
    if history:
        latest = max(time for _, time in history)
        raw = [math.exp((time - latest) / config.tau_history_ms) for _, time in history]
        denominator = math.fsum(raw)
        weights = [value / denominator for value in raw]
    values = {source: {sku: 0.0 for sku in skus} for source in SOURCES}
    masks = {source: {sku: False for sku in skus} for source in SOURCES}
    contributions = {relation: {} for relation in edges}
    available = {source: False for source in SOURCES}
    for relation, pairs in edges.items():
        available[relation] = bool(history) and relation not in disabled
        if not available[relation]:
            continue
        for sku in skus:
            masks[relation][sku] = True
            pieces = []
            for (anchor, _), alpha in zip(history, weights, strict=True):
                if (anchor, sku) in pairs:
                    piece = alpha * pairs[anchor, sku][0]
                    pieces.append(piece)
                    contributions[relation].setdefault(sku, {}).setdefault(anchor, []).append(piece)
            values[relation][sku] = math.fsum(pieces)
    for relation, targets in contributions.items():
        contributions[relation] = {
            sku: {anchor: math.fsum(pieces) for anchor, pieces in anchors.items()}
            for sku, anchors in targets.items()
        }
    history_available = {}
    for name in ("description", "image"):
        dimension = package.metadata_contract[f"{name}_dimension"]
        total = [0.0] * dimension
        for axis in range(dimension):
            total[axis] = math.fsum(
                alpha * items[anchor][name][0][axis]
                for (anchor, _), alpha in zip(history, weights, strict=True)
                if anchor in items and items[anchor][name][1]
            )
        direction, valid, _ = vector(total, dimension, config.vector_epsilon)
        history_available[name] = valid
        available[name] = valid and name not in disabled
        if available[name]:
            for sku in skus:
                masks[name][sku] = items[sku][name][1]
                if masks[name][sku]:
                    values[name][sku] = math.fsum(
                        a * b for a, b in zip(direction, items[sku][name][0], strict=True)
                    )
    search_age_ms = 0
    search = prefix.search
    search_valid = False
    if search is not None and context.t_q - search.event_time_ms <= config.search_max_age_ms:
        query, search_valid, _ = vector(
            search.payload.vector,
            package.metadata_contract["description_dimension"],
            config.vector_epsilon,
        )
        search_valid = (
            search_valid
            and search.payload.reason is None
            and package.metadata_contract["query_description_compatibility"]
            == "publisher_statement"
        )
        if search_valid:
            search_age_ms = context.t_q - search.event_time_ms
    available["search"] = search_valid and "search" not in disabled
    if available["search"]:
        for sku in skus:
            masks["search"][sku] = items[sku]["description"][1]
            if masks["search"][sku]:
                values["search"][sku] = math.fsum(
                    a * b for a, b in zip(query, items[sku]["description"][0], strict=True)
                )
    last_views = dict(prefix.last_views)
    available["recent"] = bool(last_views) and "recent" not in disabled
    available["popularity"] = "popularity" not in disabled
    for i, sku in enumerate(skus):
        if available["recent"] and sku in last_views:
            values["recent"][sku] = -(context.t_q - last_views[sku]) / 1000
            masks["recent"][sku] = True
        if available["popularity"]:
            values["popularity"][sku] = float(package.statistics.popularity[i])
            masks["popularity"][sku] = True
    eligible = tuple(sku for sku in skus if sku not in prefix.purchased_skus)
    sources = {}
    for source, quota in zip(SOURCES, config.quotas, strict=True):
        valid = [
            sku
            for sku in eligible
            if masks[source][sku] and (source not in edges or values[source][sku] > 0)
        ]
        sources[source] = tuple(
            sorted(valid, key=lambda sku: (-values[source][sku], sku.encode()))[:quota]
        )
    origins = {}
    for source in SOURCES:
        for sku in sources[source]:
            origins.setdefault(sku, []).append(source)
    popularity = sorted(
        eligible,
        key=lambda sku: (-float(package.statistics.popularity[skus.index(sku)]), sku.encode()),
    )
    fill = tuple(sku for sku in popularity if sku not in origins)[
        : config.max_candidates - len(origins)
    ]
    for sku in fill:
        origins[sku] = ["fill"]
    main = tuple(sorted(origins, key=lambda sku: sku.encode()))
    recent = sorted(
        (sku for sku in eligible if sku in last_views),
        key=lambda sku: (-last_views[sku], sku.encode()),
    )
    fallback = tuple(
        (recent + [sku for sku in popularity if sku not in recent])[: config.max_candidates]
    )
    return {
        "values": values,
        "masks": masks,
        "available": available,
        "sources": sources,
        "eligible": eligible,
        "main": main,
        "fill": fill,
        "fallback": fallback,
        "origins": {sku: tuple(sources) for sku, sources in origins.items()},
        "contributions": contributions,
        "history_available": history_available,
        "search_age_ms": search_age_ms,
    }
