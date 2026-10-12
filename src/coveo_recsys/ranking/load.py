"""Загрузка завершённой модели без обучающих таблиц и целей."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from coveo_recsys.artifacts import (
    ArtifactError,
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    strict_json_loads,
)
from coveo_recsys.artifacts.store import _digest
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.contracts.retrieval import (
    BASE_CONFIG_ID,
    NO_METADATA_CONFIG_ID,
    SUPPORTED_CONFIG_IDS,
)

from .matrix import validate_no_metadata_schema


@dataclass(frozen=True, slots=True)
class RankerModel:
    booster: lgb.Booster
    summary: dict[str, Any]


def load_ranker(root: Path, *, expected_identity: dict[str, Any] | None = None) -> RankerModel:
    if not root.is_dir():
        raise ArtifactError("Каталог модели отсутствует")
    store = LocalArtifactStore(root)
    try:
        summary = strict_json_loads(store._path("reports/ranker.json").read_bytes())
        if not isinstance(summary, dict) or (
            summary.get("format_version") != "ranker_v1"
            or summary.get("status") != "completed"
            or summary.get("library_version") != "4.6.0"
            or lgb.__version__ != "4.6.0"
        ):
            raise ArtifactError("Несовместимая или незавершённая модель")
        identity = summary["identity"]
        if not isinstance(identity, dict):
            raise ArtifactError("Отсутствуют привязки модели")
        config_id = summary.get("config_id", BASE_CONFIG_ID)
        if (
            config_id not in SUPPORTED_CONFIG_IDS
            or identity.get("config_id", config_id) != config_id
        ):
            raise ArtifactError("Неизвестный или несогласованный профиль модели")
        if config_id == NO_METADATA_CONFIG_ID or "recipe" in summary:
            if canonical_json_bytes(summary.get("recipe")) != canonical_json_bytes(
                get_ranker_recipe(config_id)
            ):
                raise ArtifactError("Рецепт модели не соответствует закреплённому профилю")
        _digest(identity["catalog_sha256"])
        _digest(summary["feature_schema_sha256"])
        _digest(summary["model_sha256"])
        if identity["feature_schema_sha256"] != summary["feature_schema_sha256"]:
            raise ArtifactError("Схема модели не соответствует закреплённым входам")
        if expected_identity is not None and (
            canonical_json_bytes(identity) != canonical_json_bytes(expected_identity)
        ):
            raise ArtifactError("Модель относится к другому запуску или входам")
        numeric, categorical = summary["numeric_columns"], summary["categorical_columns"]
        if (
            not isinstance(numeric, list)
            or not numeric
            or categorical != ([] if config_id == NO_METADATA_CONFIG_ID else ["category"])
            or any(not isinstance(name, str) or not name for name in numeric)
            or len(set(numeric + categorical)) != len(numeric + categorical)
        ):
            raise ArtifactError("Неверные имена или порядок признаков модели")
        if config_id == NO_METADATA_CONFIG_ID:
            validate_no_metadata_schema(tuple(numeric), summary["feature_schema_sha256"])
        trees = summary["actual_tree_count"]
        if type(trees) is not int or not 1 <= trees <= 200:
            raise ArtifactError("Неверное число деревьев модели")
        payload = store._path("models/ranker.txt").read_bytes()
        if sha256_bytes(payload) != summary["model_sha256"]:
            raise ArtifactError("Контрольная сумма модели не совпала")
        booster = lgb.Booster(model_str=payload.decode("utf-8"))
        if (
            booster.feature_name() != numeric + categorical
            or booster.num_trees() != trees
            or booster.num_model_per_iteration() != 1
        ):
            raise ArtifactError("Сводка не соответствует содержимому модели")
    except (OSError, ValueError, KeyError, TypeError, lgb.basic.LightGBMError) as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"Не удалось прочитать модель: {exc}") from exc
    return RankerModel(booster, summary)
