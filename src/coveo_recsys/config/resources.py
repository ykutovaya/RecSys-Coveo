"""Чтение поставляемых данных без зависимости от запуска проверок."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, strict_json_loads


def resource_bytes(relative_path: str) -> bytes:
    root = Path(__file__).resolve().parents[3]
    local = root / relative_path
    if local.is_file():
        return local.read_bytes()
    target = (
        "manifest.template.json" if relative_path.startswith("documents/math/") else relative_path
    )
    return files("coveo_recsys").joinpath("resources", target).read_bytes()


def load_template() -> dict[str, Any]:
    content = resource_bytes("documents/math/coveo_session_recommender_run_manifest.template.json")
    result: dict[str, Any] = strict_json_loads(content)
    if result.get("manifest_version") != "2.10":
        raise ValueError("Требуется нормативный шаблон 2.10")
    return result


def load_schema(name: str = "manifest") -> dict[str, Any]:
    if name not in {"manifest", "feature"}:
        raise ValueError("Неизвестная схема")
    result: dict[str, Any] = strict_json_loads(resource_bytes(f"schemas/{name}.schema.json"))
    return result


def load_registry() -> dict[str, Any]:
    """Описания, не результаты исполнения; новое дерево при каждом вызове."""
    result: dict[str, Any] = strict_json_loads(resource_bytes("configs/test_registry.json"))
    return result


def validator_fingerprint_bytes() -> bytes:
    """Точный состав исполняемого валидатора и правил, без результатов запусков."""
    package = files("coveo_recsys")
    inventory = {
        f"config/{name}": sha256_bytes(package.joinpath("config", name).read_bytes())
        for name in (
            "__init__.py",
            "manifest.py",
            "resources.py",
            "schema.py",
            "recipes.py",
            "evidence.py",
        )
    }
    for name in ("contracts/common.py", "artifacts/__init__.py", "artifacts/store.py"):
        inventory[name] = sha256_bytes(package.joinpath(name).read_bytes())
    for name in (
        "schemas/manifest.schema.json",
        "schemas/feature.schema.json",
        "configs/test_registry.json",
        "configs/ranker_recipe.json",
    ):
        inventory[name] = sha256_bytes(resource_bytes(name))
    return canonical_json_bytes(inventory)
