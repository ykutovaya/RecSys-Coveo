import copy

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from tests.reference.e4_cases import informative_table, recipe


def test_baseline_learns_pairs_and_reports_actual_tree_and_leaf_sizes():
    from coveo_recsys.ranking.train import fit_ranker

    table, settings = informative_table(), recipe()
    frozen = copy.deepcopy(settings)
    booster, report = fit_ranker(table, settings)
    scores = booster.predict(table.data, raw_score=True, num_threads=1)
    assert np.all(scores[2::3] > scores[0::3])
    assert 1 <= report["actual_tree_count"] <= 200
    assert report["actual_tree_count"] == booster.num_trees()
    assert len(report["leaf_sizes"]) == booster.num_trees()
    assert all(sum(tree) == len(table.labels) for tree in report["leaf_sizes"])
    assert report["resolved_library_parameters"]["num_threads"] == "1"
    assert report["resolved_library_parameters"]["cat_smooth"] == "1"
    assert report["objective_calls"] >= report["actual_tree_count"]
    assert report["stop_reason"] in {"iteration_budget", "no_admissible_split"}
    assert settings == frozen
    second, _ = fit_ranker(table, settings)
    np.testing.assert_allclose(
        second.predict(table.data, raw_score=True, num_threads=1), scores, atol=1e-10, rtol=1e-8
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("dataset", "weight", [1]),
        ("model", "objective", "lambdarank"),
        ("model", "num_threads", 2),
        ("training", "num_boost_round", 1),
    ],
)
def test_training_refuses_changed_baseline_recipe(section, key, value):
    from coveo_recsys.ranking.train import fit_ranker

    settings = recipe()
    settings[section][key] = value
    with pytest.raises(InputValidationError):
        fit_ranker(informative_table(), settings)
