"""Малый эталон обучающих групп по первым запросам с зарегистрированной целью."""


def table_oracle(examples):
    ordered = sorted(
        examples,
        key=lambda example: (
            example[0].session_id.encode("utf-8"),
            example[0].t_q,
            example[0].request_id.encode("utf-8"),
        ),
    )
    result = {"data": [], "labels": [], "group_sizes": [], "skus": [], "group_keys": []}
    stats = {
        "requests": len(ordered),
        "target_sessions": 0,
        "unknown_sessions": 0,
        "empty_candidate_groups": 0,
        "zero_pair_groups": 0,
        "groups": 0,
        "rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "pair_count": 0,
        "target_skus_outside_candidates": 0,
    }
    for request, candidates, features, target in ordered:
        if not target.purchases:
            stats["unknown_sessions"] += 1
            continue
        stats["target_sessions"] += 1
        purchases = set(target.purchases)
        stats["target_skus_outside_candidates"] += len(purchases.difference(candidates.skus))
        if not candidates.skus:
            stats["empty_candidate_groups"] += 1
            continue
        labels = [int(sku in purchases) for sku in candidates.skus]
        positive = sum(labels)
        negative = len(labels) - positive
        stats["positive_rows"] += positive
        stats["negative_rows"] += negative
        stats["pair_count"] += positive * negative
        stats["zero_pair_groups"] += int(positive * negative == 0)
        stats["groups"] += 1
        stats["rows"] += len(labels)
        result["group_keys"].append((request.session_id, request.t_q, request.request_id))
        result["group_sizes"].append(len(labels))
        result["skus"].extend(candidates.skus)
        result["labels"].extend(labels)
        result["data"].extend(
            list(numeric) + list(categorical)
            for numeric, categorical in zip(features.numeric, features.categorical, strict=True)
        )
    result["stats"] = stats
    return result
