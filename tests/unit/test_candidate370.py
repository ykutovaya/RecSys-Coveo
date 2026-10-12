"""370 — отдельный профиль, не замена исходного рецепта."""

from dataclasses import replace

import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.config import init_manifest, validate_manifest
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import (
    METADATA370_CONFIG_ID,
    METADATA370_QUOTAS,
    RetrievalConfig,
    validate_retrieval_profile,
)


@pytest.mark.parametrize("change", [None, "quotas", "max_candidates", "history_limit"])
def test_fixed_candidate370_profile(change):
    config = RetrievalConfig(quotas=METADATA370_QUOTAS, max_candidates=370)
    if change:
        config = replace(
            config, **{change: (49, *config.quotas[1:]) if change == "quotas" else 371}
        )
        with pytest.raises(InputValidationError):
            validate_retrieval_profile(METADATA370_CONFIG_ID, config)
    else:
        validate_retrieval_profile(METADATA370_CONFIG_ID, config)
        assert get_ranker_recipe(METADATA370_CONFIG_ID) == get_ranker_recipe()


@pytest.mark.parametrize("change", [None, "quotas", "graph", "ranker", "limit"])
def test_candidate370_manifest_preserves_other_rules(tmp_path, change):
    manifest = init_manifest("370")
    manifest["base_model"]["config_id"] = METADATA370_CONFIG_ID
    manifest["base_model"]["quotas"].update(description=80, image=60)
    manifest["task"]["candidate_limit"] = 370
    manifest["ranker"]["resolved_parameters"] = get_ranker_recipe(METADATA370_CONFIG_ID)
    if change == "quotas":
        manifest["base_model"]["quotas"]["search"] += 1
    elif change == "graph":
        manifest["base_model"]["graph_min_sessions"] += 1
    elif change == "ranker":
        manifest["ranker"]["num_boost_round"] += 1
    elif change == "limit":
        manifest["task"]["candidate_limit"] = 400
    result = validate_manifest(manifest, "preflight", LocalArtifactStore(tmp_path))
    assert bool(
        [i for i in result.issues if i.code in {"baseline_changed", "task_changed"}]
    ) == bool(change)


@pytest.mark.parametrize(
    "base_hits,new_hits,n,failed,expected",
    [
        (1, 2, 3, False, METADATA370_CONFIG_ID),
        (2, 2, 3, False, "gcs_rk_v2_7"),
        (2, 1, 3, False, "gcs_rk_v2_7"),
        (0, 0, 0, False, None),
        (1, 2, 3, True, None),
    ],
)
def test_choice_uses_integer_hits_without_reducing_denominator(
    base_hits, new_hits, n, failed, expected
):
    from coveo_recsys.evaluation.selection import select_candidate370

    def policy(hits, failure):
        return dict(
            counts={"N": n},
            coverage_chain={"Hit": hits},
            cost=dict(
                delivery_failed_count=int(failure), not_attempted_count=0, missing_trace_count=0
            ),
        )

    result = select_candidate370(
        {
            "policies": {
                "GCS+Rk": policy(base_hits, False),
                METADATA370_CONFIG_ID: policy(new_hits, failed),
            }
        }
    )
    assert result["selected"] == expected
    assert not result["confirmatory"]


def test_pre_selection_freezes_settings_and_needs_external_reference(tmp_path):
    from copy import deepcopy

    from tests.reference.manifest_cases import make_manifest_case

    store = LocalArtifactStore(tmp_path)
    manifest, references = make_manifest_case(store, "pre_selection", operation="evaluate")
    result = validate_manifest(
        manifest, "pre_selection", store, plan_references=references, operation="evaluate"
    )
    assert result.gate_passed, result.issues
    assert not validate_manifest(manifest, "pre_selection", store, operation="evaluate").gate_passed
    changed = deepcopy(manifest)
    changed["selection"]["tie_break"] = ["max_session_hit_count", "min_median_p99"]
    result = validate_manifest(
        changed, "pre_selection", store, plan_references=references, operation="evaluate"
    )
    assert not result.gate_passed and any(i.code == "selection_plan_changed" for i in result.issues)
