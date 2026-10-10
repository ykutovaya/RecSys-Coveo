"""Обучение закреплённого рецепта; загрузка и применение находятся отдельно."""

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np

from coveo_recsys.artifacts import canonical_json_bytes
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import BASE_CONFIG_ID, NO_METADATA_CONFIG_ID

from .matrix import validate_no_metadata_schema
from .objective import make_objective
from .table import TrainingTable


def validate_recipe(recipe: dict[str, Any], config_id: str = BASE_CONFIG_ID) -> None:
    if lgb.__version__ != "4.6.0":
        raise InputValidationError("Нужен принятый LightGBM 4.6.0")
    expected = get_ranker_recipe(config_id)
    if canonical_json_bytes(recipe) != canonical_json_bytes(expected):
        raise InputValidationError("Изменён закреплённый рецепт ранжировщика")


def fit_ranker(
    table: TrainingTable,
    recipe: dict[str, Any],
    *,
    config_id: str = BASE_CONFIG_ID,
    check_budget: Callable[[], None] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    validate_recipe(recipe, config_id)
    if table.categorical_columns != tuple(recipe["dataset"]["categorical_feature"]):
        raise InputValidationError("Категориальные признаки не соответствуют профилю модели")
    if config_id == NO_METADATA_CONFIG_ID:
        validate_no_metadata_schema(table.numeric_columns, table.feature_schema_sha256)
    if not table.statistics["pair_count"]:
        raise InputValidationError("insufficient: нет обучающих пар")
    objective = make_objective(table.labels, table.group_sizes, table.skus)
    calls = 0

    def checked_objective(scores: Any, dataset: lgb.Dataset) -> tuple[Any, Any]:
        nonlocal calls
        if check_budget is not None:
            check_budget()
        calls += 1
        return objective(scores, dataset)

    parameters = dict(recipe["model"])
    parameters["objective"] = checked_objective
    dataset_parameters = {
        k: v
        for k, v in recipe["dataset"].items()
        if k not in {"weight", "init_score", "categorical_feature"}
    }
    dataset = lgb.Dataset(
        table.data,
        label=table.labels,
        group=list(table.group_sizes),
        feature_name=list(table.numeric_columns + table.categorical_columns),
        categorical_feature=list(table.categorical_columns),
        weight=None,
        init_score=None,
        params=dataset_parameters,
    )
    if check_budget is not None:
        check_budget()
    booster = lgb.train(parameters, dataset, **recipe["training"])
    tree_count = booster.num_trees()
    if not 1 <= tree_count <= recipe["training"]["num_boost_round"]:
        raise InputValidationError("Неверное фактическое число деревьев")
    if check_budget is not None:
        check_budget()
    leaf_sizes = [
        np.zeros(item["num_leaves"], dtype=np.int64) for item in booster.dump_model()["tree_info"]
    ]
    for start in range(0, len(table.labels), 4096):
        rows = table.data[start : start + 4096]
        leaves = np.asarray(
            booster.predict(rows, pred_leaf=True, num_iteration=tree_count, num_threads=1)
        ).reshape(len(rows), tree_count)
        predictions = booster.predict(rows, raw_score=True, num_iteration=tree_count, num_threads=1)
        if not np.isfinite(predictions).all():
            raise InputValidationError("Неконечные баллы обученной модели")
        for i in range(tree_count):
            leaf_sizes[i] += np.bincount(leaves[:, i], minlength=len(leaf_sizes[i]))
        if check_budget is not None:
            check_budget()
    # Штатная сериализация включает фактически разрешённые параметры C++,
    # в том числе значения по умолчанию закреплённой версии.
    text = booster.model_to_string(num_iteration=tree_count)
    section = text.split("parameters:\n", 1)[1].split("end of parameters", 1)[0]
    resolved = {}
    for line in section.splitlines():
        if line.startswith("[") and line.endswith("]"):
            key, _, value = line[1:-1].partition(": ")
            resolved[key] = value
    return booster, {
        "config_id": config_id,
        "library": "lightgbm",
        "library_version": lgb.__version__,
        "recipe": recipe,
        "resolved_library_parameters": resolved,
        "actual_tree_count": tree_count,
        "leaf_sizes": [sizes.tolist() for sizes in leaf_sizes],
        "objective_calls": calls,
        "stop_reason": (
            "iteration_budget"
            if tree_count == recipe["training"]["num_boost_round"]
            else "no_admissible_split"
        ),
        "groups": table.statistics,
    }
