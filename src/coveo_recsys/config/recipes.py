"""Проверка заполненных рецептов; обучение здесь не выполняется."""

from __future__ import annotations

import math
from typing import Any

from coveo_recsys.artifacts import strict_json_loads
from coveo_recsys.contracts.common import ValidationIssue

from .resources import resource_bytes


def resolved_recipe_issues(manifest: dict[str, Any]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def invalid(path: str, text: str) -> None:
        issues.append(ValidationIssue(path, "resolved_recipe", text))

    def number(value: Any, *, zero: bool = False, integer: bool = False) -> bool:
        if not (type(value) is int if integer else type(value) in {int, float}):
            return False
        try:
            return math.isfinite(value) and (value >= 0 if zero else value > 0)
        except OverflowError:
            return False

    expected = strict_json_loads(resource_bytes("configs/ranker_recipe.json"))
    actual = manifest["ranker"]["resolved_parameters"]
    if not isinstance(actual, dict):
        invalid("ranker.resolved_parameters", "Нужны четыре полные части рецепта")
    else:
        for group, fields in expected.items():
            supplied = actual.get(group)
            if not isinstance(supplied, dict) or not fields.keys() <= supplied.keys():
                invalid(f"ranker.resolved_parameters.{group}", "Отсутствуют обязательные параметры")
                continue
            for key, value in fields.items():
                supplied_value = supplied[key]
                if type(value) is bool and type(supplied_value) is not bool:
                    invalid(
                        f"ranker.resolved_parameters.{group}.{key}",
                        "Нужен логический параметр, не число или строка",
                    )
                elif type(value) in {int, float} and not number(
                    supplied_value, zero=True, integer=type(value) is int
                ):
                    invalid(
                        f"ranker.resolved_parameters.{group}.{key}",
                        "Нужен конечный неотрицательный параметр правильного типа",
                    )
                elif isinstance(value, str) and not isinstance(supplied_value, str):
                    invalid(f"ranker.resolved_parameters.{group}.{key}", "Нужен строковый параметр")
            if manifest["base_model"]["config_id"] == "gcs_rk_v2_7":
                for key, value in fields.items():
                    if group == "dataset" and key == "categorical_feature":
                        continue  # Actual column name is checked against the feature schema.
                    if supplied[key] != value or type(supplied[key]) is not type(value):
                        # JSON permits 1 and 1.0 interchangeably, but never bool as a number.
                        if not (
                            type(value) in {int, float}
                            and type(supplied[key]) in {int, float}
                            and supplied[key] == value
                        ):
                            invalid(
                                f"ranker.resolved_parameters.{group}.{key}",
                                "Изменён рецепт первого запуска",
                            )
        invariants = {
            "model": {
                "objective": "session_lambdas_v1",
                "boosting": "gbdt",
                "lambda_l1": 0,
                "boost_from_average": False,
                "linear_tree": False,
                "path_smooth": 0,
                "max_delta_step": 0,
                "cat_l2": 0,
            },
            "dataset": {"weight": None, "init_score": None},
            "training": {"valid_sets": None, "feval": None, "init_model": None, "callbacks": None},
            "prediction": {"raw_score": True, "num_iteration": "actual_tree_count"},
        }
        for group, fields in invariants.items():
            supplied = actual.get(group)
            if isinstance(supplied, dict):
                for key, value in fields.items():
                    if (
                        key not in supplied
                        or supplied[key] != value
                        or (type(value) is bool and type(supplied[key]) is not bool)
                    ):
                        invalid(
                            f"ranker.resolved_parameters.{group}.{key}",
                            "Нарушен общий контракт собственных производных",
                        )
        model, training = actual.get("model"), actual.get("training")
        if isinstance(model, dict) and isinstance(training, dict):
            ranker = manifest["ranker"]
            for key in (
                "learning_rate",
                "num_leaves",
                "max_depth",
                "lambda_l2",
                "min_gain_to_split",
                "min_data_in_leaf",
                "min_sum_hessian_in_leaf",
                "cat_smooth",
                "cat_l2",
            ):
                if model.get(key) != ranker[key]:
                    invalid(f"ranker.{key}", "Сводный параметр не совпал с полным рецептом")
            if training.get("num_boost_round") != ranker["num_boost_round"]:
                invalid("ranker.num_boost_round", "Бюджет не совпал с полным рецептом")
            for key in (
                "learning_rate",
                "num_leaves",
                "max_depth",
                "min_data_in_leaf",
                "cat_smooth",
            ):
                if not number(
                    model.get(key), integer=key in {"num_leaves", "max_depth", "min_data_in_leaf"}
                ):
                    invalid(
                        f"ranker.resolved_parameters.model.{key}",
                        "Параметр должен быть положительным",
                    )
            if not number(training.get("num_boost_round"), integer=True):
                invalid(
                    "ranker.resolved_parameters.training.num_boost_round",
                    "Нужен положительный бюджет обучения",
                )
    fields = {
        "neural": {
            "d": (True, False),
            "J_I": (True, False),
            "H_P": (True, False),
            "d_P": (True, False),
            "B_P": (True, False),
            "R_P": (False, False),
            "c_G": (False, False),
            "eta_N": (False, True),
            "tau_P": (False, False),
            "tau_N": (False, False),
            "tau_C": (False, False),
            "tau_SN_seconds": (False, False),
            "maximum_search_gate": (False, False),
            "initial_search_gate": (False, False),
            "lambda_next": (False, True),
            "lambda_C": (False, True),
            "lambda_G": (False, True),
            "lambda_theta": (False, True),
            "p_S": (False, True),
            "delta_I": (False, False),
            "epsilon_N": (False, False),
            "branch_dropout": (False, True),
        },
        "sid": {
            key: (
                key
                in {
                    "levels",
                    "vocabulary",
                    "code_dimension",
                    "hidden_dimension",
                    "beam_width",
                    "codes_per_item",
                    "quota",
                },
                key in {"lambda_X", "lambda_B", "lambda_2"},
            )
            for key in (
                "levels",
                "vocabulary",
                "code_dimension",
                "hidden_dimension",
                "beam_width",
                "codes_per_item",
                "quota",
                "lambda_R",
                "lambda_X",
                "lambda_B",
                "lambda_2",
                "tau_R",
            )
        },
    }
    for module, requirements in fields.items():
        extension = manifest[f"{module}_extension"]
        if not extension["enabled"]:
            continue
        configuration = extension["resolved_configuration"]
        params = configuration.get("parameters") if isinstance(configuration, dict) else None
        if not isinstance(params, dict) or any(
            not number(params.get(key), integer=integer, zero=zero)
            for key, (integer, zero) in requirements.items()
        ):
            invalid(
                f"{module}_extension.resolved_configuration",
                "Не заполнены численные параметры активного модуля",
            )
            continue
        if configuration.get("training") != extension["training_defaults"]:
            invalid(
                f"{module}_extension.resolved_configuration.training",
                "Нужен полный закреплённый рецепт обучения",
            )
        if module == "neural":
            weights = params.get("relation_weights")
            if (
                not isinstance(weights, list)
                or len(weights) != 3
                or not all(number(x, zero=True) for x in weights)
                or abs(sum(weights) - 1) > 1e-12
                or not 0 < params["initial_search_gate"] < params["maximum_search_gate"] <= 1
                or params["p_S"] > 1
                or params["branch_dropout"] > 1
                or params["delta_I"] >= 1
            ):
                invalid(
                    "neural_extension.resolved_configuration",
                    "Некорректные вероятности/веса/границы поиска",
                )
    ann = manifest["ann_extension"]
    if ann["enabled"]:
        drop = ann["maximum_hit_drop_fraction"]
        if not number(drop, zero=True) or drop > 1:
            invalid("ann_extension.maximum_hit_drop_fraction", "Нужен допуск в долях Hit@20")
        configuration = ann["resolved_configuration"]
        params = configuration.get("parameters") if isinstance(configuration, dict) else None
        if (
            not isinstance(params, dict)
            or any(
                not isinstance(params.get(key), str) or not params[key]
                for key in ("library", "library_version", "index_type", "metric")
            )
            or not number(params.get("dimension"), integer=True)
            or not number(params.get("seed"), zero=True, integer=True)
            or not isinstance(params.get("build"), dict)
            or not params["build"]
            or not isinstance(params.get("search"), dict)
            or not params["search"]
        ):
            invalid(
                "ann_extension.resolved_configuration",
                "Нужны версия, размерность, метрика, seed и полные параметры индекса/поиска",
            )
    if manifest["modules"]["optional"]:
        invalid(
            "modules.optional",
            "Общее имя optional не задаёт алгоритм: требуется отдельная версия схемы исследуемого модуля",
        )
    for module, configuration in manifest["module_configurations"].items():
        if not manifest["modules"][module]:
            continue
        path = f"module_configurations.{module}"
        if not isinstance(configuration, dict):
            invalid(path, "Нужна полная конфигурация активного модуля")
            continue
        if module == "sampled" and (
            not number(configuration.get("negative_count"), integer=True)
            or not number(configuration.get("uniform_mixture"))
            or configuration["uniform_mixture"] > 1
            or configuration.get("support_check") != "cdf_fp64_grid53_v1"
        ):
            invalid(path, "Нужны m, доля равномерной смеси и проверка реальной CDF")
        if module == "sequence" and (
            configuration.get("layer_norm_epsilon") != 1e-5
            or configuration.get("attention_blocks") != 1
            or configuration.get("input") != "detail_timestamp_groups"
            or configuration.get("initialization") != "from_scratch"
            or configuration.get("training") != manifest["neural_extension"]["training_defaults"]
        ):
            invalid(
                path,
                "Нужен полный рецепт последовательного кодировщика без переноса обученных весов",
            )
        if module == "freshness" and (
            not number(configuration.get("window_seconds"))
            or not number(configuration.get("age_decay_seconds"))
            or configuration.get("training_mode") not in ("frozen", "retrained")
            or not isinstance(configuration.get("update_schedule_utc"), list)
            or not configuration["update_schedule_utc"]
            or any(
                not isinstance(configuration.get(key), str) or not configuration[key]
                for key in ("publication_delay_rule", "release_data_plan", "fallback_package_rule")
            )
        ):
            invalid(path, "Нужны окно, старение, календарь и полный протокол выпуска/резерва")
        if module == "ppr":
            weights = configuration.get("relation_weights")
            if (
                not number(configuration.get("restart_probability"))
                or configuration["restart_probability"] > 1
                or not number(configuration.get("iterations"), integer=True)
                or not isinstance(weights, list)
                or len(weights) != 3
                or not all(number(x, zero=True) for x in weights)
                or abs(sum(weights) - 1) > 1e-12
            ):
                invalid(
                    path,
                    "Нужны число итераций, вероятность возврата и нормированные веса отношений",
                )
    return tuple(issues)
