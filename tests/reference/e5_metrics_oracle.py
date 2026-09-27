"""Независимый полный перебор метрик, без вызова рабочего расчёта."""

import math


def session_oracle(catalog, forbidden, candidates, delivered, purchases, k=20):
    available = [item for item in purchases if item in catalog and item not in forbidden]
    found = [item for item in purchases if item in candidates]
    shown = [item for item in delivered[:k] if item in purchases]
    v, u, hit = int(bool(available)), int(bool(found)), int(bool(shown))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(purchases))))
    dcg = sum(
        1 / math.log2(position + 2)
        for position, item in enumerate(delivered[:k])
        if item in purchases
    )
    return {
        "v": v,
        "u": u,
        "hit": hit,
        "losses": (1 - v, v - u, u - hit),
        "recall": len(shown) / len(purchases),
        "ndcg": dcg / ideal,
    }


def source_oracle(origins, purchases, source):
    original = any(item in purchases for item in origins)
    retained = any(
        item in purchases and any(name != source for name in names)
        for item, names in origins.items()
    )
    return int(original) - int(retained)
