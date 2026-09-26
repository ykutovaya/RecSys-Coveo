"""Искусственные информативные группы; не измерение качества Coveo."""

import json

import numpy as np

from coveo_recsys.config.resources import resource_bytes


def recipe():
    return json.loads(resource_bytes("configs/ranker_recipe.json"))


def informative_table(groups=160):
    from coveo_recsys.contracts.training import SupervisionRecord
    from coveo_recsys.ranking.table import TrainingTable

    return TrainingTable(
        data=np.tile([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], (groups, 1)),
        labels=np.tile([0, 0, 1], groups),
        group_sizes=(3,) * groups,
        skus=("a", "b", "c") * groups,
        group_keys=tuple((f"s{i:04}", 10, f"q{i:04}") for i in range(groups)),
        feature_schema_sha256="a" * 64,
        numeric_columns=("x",),
        categorical_columns=("category",),
        statistics={"groups": groups, "rows": groups * 3, "pair_count": groups * 2},
        supervision=tuple(
            SupervisionRecord(f"q{i:04}", f"s{i:04}", (0, 0, 1), 1, 1.0, (0.5, 0.5))
            for i in range(groups)
        ),
    )
