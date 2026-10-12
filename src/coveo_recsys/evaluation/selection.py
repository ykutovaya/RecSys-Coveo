"""Одно заранее заданное сравнение, без поиска параметров."""

from typing import Any

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import BASE_CONFIG_ID, METADATA370_CONFIG_ID


def select_candidate370(summary: dict[str, Any]) -> dict[str, Any]:
    base, variant = (summary["policies"][p] for p in ("GCS+Rk", METADATA370_CONFIG_ID))
    if base["counts"]["N"] != variant["counts"]["N"]:
        raise InputValidationError("Разные целевые знаменатели выбора")
    failed = any(
        p["cost"][key]
        for p in (base, variant)
        for key in (
            "delivery_failed_count",
            "not_attempted_count",
            "missing_trace_count",
        )
    )
    status = "failed" if failed else "passed" if base["counts"]["N"] else "insufficient"
    winner = (
        METADATA370_CONFIG_ID
        if variant["coverage_chain"]["Hit"] > base["coverage_chain"]["Hit"]
        else BASE_CONFIG_ID
    )
    return dict(
        status=status,
        selected=winner if status == "passed" else None,
        rule="max_session_hit_count_then_keep_control",
        confirmatory=False,
    )
