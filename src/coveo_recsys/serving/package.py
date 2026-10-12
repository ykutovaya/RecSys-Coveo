"""Проверенная локальная публикация; все файлы читаются только при загрузке."""

import hashlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    strict_json_loads,
)
from coveo_recsys.artifacts.store import _digest
from coveo_recsys.contracts import InputValidationError, id_bytes, require_int64
from coveo_recsys.contracts.historical import MetadataRecord
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.data.io import FILES, _check_expected_input, parse_profile, row_batches
from coveo_recsys.data.metadata import _vector
from coveo_recsys.data.supply import _price_bucket
from coveo_recsys.explanations.templates import TEMPLATE_SHA256, TEMPLATE_VERSION
from coveo_recsys.features import FeatureEngine
from coveo_recsys.historical.load import PACKAGE_PATH, load_historical
from coveo_recsys.process_memory import peak_memory
from coveo_recsys.ranking.load import RankerModel, load_ranker
from coveo_recsys.retrieval.scores import RetrievalIndex


@dataclass(frozen=True, slots=True)
class ServingPackage:
    package_id: str
    published_at_ms: int
    profile: str
    engine: FeatureEngine
    model: RankerModel
    schema: dict[str, Any]
    retrieval: RetrievalConfig
    catalog_sha256: str
    template_version: str
    template_sha256: str
    hashes: Mapping[str, str]
    producer_identity: Mapping[str, Any]
    metadata_available: bool
    metadata_rows: int


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise InputValidationError(f"Неверный состав {name}")


def _bounded_size(path: Path, limit: int) -> int:
    size = path.stat().st_size
    if not path.is_file() or size > limit:
        raise InputValidationError("Размер файла превышает бюджет загрузки")
    return size


def _chunks(path: Path, limit: int, check_budget: Callable[[], None]) -> Iterator[bytes]:
    check_budget()
    size = _bounded_size(path, limit)
    count = 0
    with path.open("rb") as stream:
        while True:
            check_budget()
            chunk = stream.read(min(1 << 20, limit - count + 1))
            check_budget()
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise InputValidationError("Размер файла превышает бюджет загрузки")
            yield chunk
    if count != size or path.stat().st_size != size:
        raise InputValidationError("Вход изменился во время загрузки")


def _file_hash(path: Path, limit: int, check_budget: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    for chunk in _chunks(path, limit, check_budget):
        digest.update(chunk)
    return digest.hexdigest()


def load_serving_package(root: Path, manifest: Mapping[str, Any]) -> ServingPackage:
    """Загрузить полный lookup поставки; ошибка не публикует частичную готовность.

    Дата публикации задаётся новым манифестом явно и не выводится из времён
    файлов производителя. Исходные привязки модели не заменяются кодом сервиса.
    """
    started = time.monotonic()
    manifest_bytes = canonical_json_bytes(dict(manifest))
    settings = strict_json_loads(manifest_bytes)
    _keys(
        settings,
        {
            "format_version",
            "profile",
            "published_at_ms",
            "historical",
            "model",
            "training_features",
            "feature_schema",
            "retrieval",
            "metadata",
            "templates",
            "limits",
        },
        "манифеста сервиса",
    )
    if (
        settings["format_version"] != "serving_package_v1"
        or settings["profile"] != "local_checksum_only"
    ):
        raise InputValidationError("Поддерживается только локальный исследовательский профиль")
    published_at = require_int64(settings["published_at_ms"], nonnegative=True)
    _keys(
        settings["limits"],
        {"max_metadata_rows", "max_metadata_bytes", "max_memory_bytes", "max_seconds"},
        "бюджетов загрузки",
    )
    limits = settings["limits"]
    if any(require_int64(value, nonnegative=True) == 0 for value in limits.values()):
        raise InputValidationError("Нужны положительные бюджеты загрузки")
    _keys(settings["templates"], {"version", "sha256"}, "шаблонов")
    if settings["templates"] != {"version": TEMPLATE_VERSION, "sha256": TEMPLATE_SHA256}:
        raise InputValidationError("Шаблоны не соответствуют встроенной проверенной версии")
    for name in ("historical", "model", "training_features", "feature_schema"):
        digest = (
            "package_sha256"
            if name == "historical"
            else "sha256"
            if name == "feature_schema"
            else "report_sha256"
        )
        _keys(settings[name], {"path", digest}, name)
    _keys(settings["metadata"], {"path", "profile_path", "sha256", "profile_sha256"}, "метаданных")
    if not isinstance(settings["retrieval"], dict):
        raise InputValidationError("Нужна конфигурация отбора")
    configuration = dict(settings["retrieval"])
    if "quotas" in configuration:
        configuration["quotas"] = tuple(configuration["quotas"])
    retrieval = RetrievalConfig(**configuration)
    if not root.is_dir():
        raise InputValidationError("Нет корня проекта")
    store = LocalArtifactStore(root)
    paths = {
        name: store._path(settings[name]["path"])
        for name in ("historical", "model", "training_features", "feature_schema", "metadata")
    }
    paths["metadata_profile"] = store._path(settings["metadata"]["profile_path"])
    if any(not paths[name].is_dir() for name in ("historical", "model", "training_features")):
        raise InputValidationError("Отсутствует готовый входной пакет")
    metadata_size = paths["metadata"].stat().st_size
    if metadata_size > limits["max_metadata_bytes"]:
        raise InputValidationError("Размер поставки превышает бюджет загрузки")

    def check_budget() -> None:
        memory = peak_memory()["peak_process_memory_bytes"]
        if (
            memory is None
            or memory > limits["max_memory_bytes"]
            or time.monotonic() - started > limits["max_seconds"]
        ):
            raise InputValidationError("Превышен бюджет времени или памяти загрузки")

    check_budget()
    file_limit = limits["max_memory_bytes"] // 4
    json_limit = min(32 << 20, file_limit)
    pinned: dict[Path, tuple[str, int]] = {}

    def read(path: Path, digest: str) -> bytes:
        payload = b"".join(_chunks(path, json_limit, check_budget))
        if sha256_bytes(payload) != digest:
            raise InputValidationError("Изменён закреплённый файл пакета")
        pinned[path] = digest, json_limit
        return payload

    model_store = LocalArtifactStore(paths["model"])
    feature_store = LocalArtifactStore(paths["training_features"])
    historical_store = LocalArtifactStore(paths["historical"])
    model_path = model_store._path("models/ranker.txt")
    sizes = {
        path: _bounded_size(path, json_limit)
        for path in (
            paths["feature_schema"],
            paths["metadata_profile"],
            model_store._path("reports/ranker.json"),
            feature_store._path("reports/features.json"),
        )
    }
    sizes[model_path] = _bounded_size(model_path, file_limit)
    body = strict_json_loads(
        read(historical_store._path(PACKAGE_PATH), settings["historical"]["package_sha256"])
    )
    if not isinstance(body, dict) or not isinstance(body.get("artifacts"), dict):
        raise InputValidationError("Не задан состав исторического пакета")
    declared = {PACKAGE_PATH, *body["artifacts"]}
    # Размеры проверяются до существующих загрузчиков, которые читают целые
    # файлы. Их проверки схем, содержимого и зависимостей не заменяются здесь.
    for relative in sorted(declared):
        check_budget()
        path = historical_store._path(relative)
        limit = json_limit if relative.endswith(".json") else file_limit
        sizes[path] = _bounded_size(path, limit)
        descriptor_path = historical_store._path(relative + ".artifact.json")
        _bounded_size(historical_store._path(relative + ".artifact.sha256"), 65)
        raw = strict_json_loads(b"".join(_chunks(descriptor_path, json_limit, check_budget)))
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("artifact_dependencies"), dict)
            or not isinstance(raw.get("sha256"), str)
        ):
            raise InputValidationError("Неверный дескриптор исторического пакета")
        if set(raw["artifact_dependencies"]) - declared:
            raise InputValidationError("Необъявленная зависимость исторического пакета")
        object_path = historical_store._path(f".objects/{_digest(raw['sha256'])}")
        _bounded_size(object_path, limit)
    if sum(sizes.values()) > file_limit:
        raise InputValidationError("Состав пакета превышает бюджет загрузки")
    check_budget()
    schema_bytes = read(paths["feature_schema"], settings["feature_schema"]["sha256"])
    schema = strict_json_loads(schema_bytes)
    if not isinstance(schema, dict) or schema.get("event_semantics") != "set_v1":
        raise InputValidationError("Нужна неизменная базовая схема set_v1")
    model_bytes = read(model_store._path("reports/ranker.json"), settings["model"]["report_sha256"])
    feature_bytes = read(
        feature_store._path("reports/features.json"), settings["training_features"]["report_sha256"]
    )
    summary, features = strict_json_loads(model_bytes), strict_json_loads(feature_bytes)
    mid, fid = summary["identity"], features["identity"]
    head = historical_store.describe(PACKAGE_PATH)
    catalog = historical_store.describe("historical/catalog.parquet")
    if (
        head.sha256 != settings["historical"]["package_sha256"]
        or summary.get("status") != "completed"
        or mid.get("period") != "B"
        or mid.get("scope") != head.scope
        or mid.get("historical_sha256") != head.sha256
        or mid.get("catalog_sha256") != catalog.sha256
        or mid.get("features_sha256") != sha256_bytes(feature_bytes)
        or mid.get("feature_schema_sha256") != sha256_bytes(canonical_json_bytes(schema))
        or mid.get("feature_schema_file_sha256") != sha256_bytes(schema_bytes)
        or mid.get("evaluation_protocol") != "observed_purchases_v1"
        or features.get("status") != "completed"
        or features.get("feature_schema") != schema
        or fid.get("format_version") != "features_v1"
        or fid.get("period") != "B"
        or fid.get("scope") != head.scope
        or fid.get("preparation_sha256") != mid.get("preparation_sha256")
        or fid.get("historical_sha256") != head.sha256
        or fid.get("historical_descriptor_sha256") != head.descriptor_sha256
        or fid.get("catalog_sha256") != catalog.sha256
        or fid.get("feature_schema_file_sha256") != sha256_bytes(schema_bytes)
        or canonical_json_bytes(fid.get("configuration")) != canonical_json_bytes(asdict(retrieval))
    ):
        raise InputValidationError("Несовместимые зависимости готового производителя")
    check_budget()
    history = load_historical(historical_store)
    check_budget()
    if published_at <= history.snapshot_time_ms:
        raise InputValidationError("Публикация должна следовать историческому снимку")
    check_budget()
    model = load_ranker(paths["model"], expected_identity=mid)
    check_budget()
    if _file_hash(model_path, file_limit, check_budget) != summary["model_sha256"]:
        raise InputValidationError("Изменена готовая модель")
    pinned[model_path] = summary["model_sha256"], file_limit
    profile_bytes = read(paths["metadata_profile"], settings["metadata"]["profile_sha256"])
    raw_profile = strict_json_loads(profile_bytes)
    canonical_profile_hash = sha256_bytes(canonical_json_bytes(raw_profile))
    profile = parse_profile(raw_profile, sha256=canonical_profile_hash)
    metadata_hash = settings["metadata"]["sha256"]
    if (
        metadata_hash != history.input_hashes["metadata_sha256"]
        or canonical_profile_hash != history.input_hashes["metadata_profile_sha256"]
        or profile.scope != head.scope
        or profile.dimensions["description"] != history.metadata_contract["description_dimension"]
        or profile.dimensions["image"] != history.metadata_contract["image_dimension"]
        or profile.dimensions["query"] != history.metadata_contract["description_dimension"]
    ):
        raise InputValidationError("Поставка метаданных несовместима с историческим пакетом")
    if _file_hash(paths["metadata"], limits["max_metadata_bytes"], check_budget) != metadata_hash:
        raise InputValidationError("Изменена поставка метаданных")
    _check_expected_input(paths["metadata"], "products", profile, metadata_hash)
    pinned[paths["metadata"]] = metadata_hash, limits["max_metadata_bytes"]
    observed: dict[str, MetadataRecord] = {}
    seen: set[str] = set()
    historical_skus = set(history.catalog.skus)
    for batch in row_batches(paths["metadata"], "products", profile):
        for line, row in batch:
            sku = row["sku"]
            id_bytes(sku)
            if sku in seen:
                raise InputValidationError("Повтор SKU в поставке метаданных")
            seen.add(sku)
            if len(seen) > limits["max_metadata_rows"]:
                raise InputValidationError("Число метаданных превышает бюджет; усечение запрещено")
            _price_bucket(row["price_bucket"], FILES["products"], line)
            price = (
                require_int64(int(Decimal(row["price_bucket"])), nonnegative=True)
                if row["price_bucket"]
                else None
            )
            category = row["category"] or None
            if category is not None:
                id_bytes(category)
            if sku not in historical_skus:
                observed[sku] = MetadataRecord(
                    sku, _vector(row["description"]), _vector(row["image"]), category, price
                )
        check_budget()
    index = RetrievalIndex(history, catalog.sha256, observed)
    check_budget()
    engine = FeatureEngine(index, schema, retrieval)
    check_budget()
    if (
        engine.schema_sha256 != summary["feature_schema_sha256"]
        or list(engine.numeric_columns) != summary["numeric_columns"]
        or list(engine.categorical_columns) != summary["categorical_columns"]
    ):
        raise InputValidationError("Порядок признаков не соответствует готовому ранжировщику")
    for path, (digest, limit) in pinned.items():
        if _file_hash(path, limit, check_budget) != digest:
            raise InputValidationError("Вход изменился во время загрузки")
    if (
        paths["metadata"].stat().st_size != metadata_size
        or historical_store.describe(PACKAGE_PATH).descriptor_sha256 != head.descriptor_sha256
    ):
        raise InputValidationError("Пакет изменился во время загрузки")
    check_budget()
    hashes = {
        "historical_sha256": head.sha256,
        "historical_descriptor_sha256": head.descriptor_sha256,
        "historical_code_sha256": head.code_sha256,
        "catalog_sha256": catalog.sha256,
        "model_report_sha256": sha256_bytes(model_bytes),
        "model_sha256": summary["model_sha256"],
        "training_features_sha256": sha256_bytes(feature_bytes),
        "feature_schema_sha256": engine.schema_sha256,
        "feature_schema_file_sha256": sha256_bytes(schema_bytes),
        "metadata_sha256": metadata_hash,
        "metadata_profile_sha256": canonical_profile_hash,
        "template_sha256": TEMPLATE_SHA256,
    }
    return ServingPackage(
        sha256_bytes(manifest_bytes),
        published_at,
        settings["profile"],
        engine,
        model,
        schema,
        retrieval,
        catalog.sha256,
        TEMPLATE_VERSION,
        TEMPLATE_SHA256,
        MappingProxyType(hashes),
        MappingProxyType(dict(mid)),
        True,
        len(seen),
    )


def select_package(packages: Sequence[ServingPackage], *, t_q: int) -> ServingPackage:
    """Последняя совместимая публикация строго до времени события запроса."""
    require_int64(t_q)
    eligible = [package for package in packages if package.published_at_ms < t_q]
    if not eligible:
        raise InputValidationError("version_unavailable")
    selected = max(eligible, key=lambda package: package.published_at_ms)
    if any(
        package.published_at_ms == selected.published_at_ms
        and package.package_id != selected.package_id
        for package in eligible
    ):
        raise InputValidationError("Неоднозначная публикация пакета")
    return selected
