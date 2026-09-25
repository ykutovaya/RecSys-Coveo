"""Файловый адаптер. Готовность означает наличие проверенного дескриптора.

Файл публикуется исключающей перезапись ссылкой после записи и fsync во
временный файл того же тома. Незавершённая публикация не читается как артефакт.
Это локальный исследовательский адаптер, не реализация доверия production.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq


class ArtifactError(ValueError):
    """Небезопасный путь, повреждение либо несовместимость артефакта."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def strict_json_loads(data: bytes | str) -> Any:
    """UTF-8 JSON без повторных ключей и неконечных чисел, включая 1e999."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Повтор JSON-ключа: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"Неконечная константа JSON: {value}")

    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Неконечное число JSON")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        if isinstance(value, list):
            for child in value:
                finite(child)

    text = data.decode("utf-8") if isinstance(data, bytes) else data
    result = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    finite(result)
    return result


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactError("Ожидался SHA-256 из 64 строчных шестнадцатеричных цифр")
    return value


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: str
    sha256: str
    size_bytes: int
    schema_version: str
    artifact_version: str
    inputs: dict[str, str]
    configuration: dict[str, Any]
    code_sha256: str
    run_id: str
    scope: str
    config_id: str
    kind: str
    created_at_utc: str
    metadata: dict[str, Any]
    artifact_dependencies: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def descriptor_sha256(self) -> str:
        """Идентичность содержимого вместе со схемой, ID строк и происхождением."""
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass
class _VerificationState:
    pinned: dict[str, str] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    objects: set[str] = field(default_factory=set)

    def pin(self, path: str, digest: str) -> None:
        if self.pinned.setdefault(path, digest) != digest:
            raise ArtifactError("Дескриптор не совпал с закреплённой идентичностью")


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        supplied = Path(root).absolute()
        if supplied.is_symlink():
            raise ArtifactError("Корень хранилища не может быть символьной ссылкой")
        supplied.mkdir(parents=True, exist_ok=True)
        self.root = supplied.resolve()
        self._verification: _VerificationState | None = None

    @contextmanager
    def verification_scope(self) -> Iterator[None]:
        """Одна последовательная операция: запоминание SHA и свежая проверка на выходе.

        Содержимое файлов не кешируется. Это не атомарный снимок файловой системы:
        одновременная запись посторонним процессом не поддерживается.
        """
        if self._verification is not None:
            raise ArtifactError("Операции проверки одного хранилища не могут пересекаться")
        state = _VerificationState()
        self._verification = state
        try:
            yield
            # Не используем ни одного результата проверки из тела операции.
            self._verification = None
            final = _VerificationState(pinned=dict(state.pinned))
            for path, digest in state.pinned.items():
                self._verify_dependency(path, digest, final, set(), verify_objects=True)
            for digest in state.objects - final.objects:
                self.read_by_sha256(digest)
        finally:
            self._verification = None

    def _verify_dependency(
        self,
        path: str,
        expected: str,
        state: _VerificationState,
        ancestry: set[str],
        *,
        verify_objects: bool = False,
    ) -> None:
        if path in ancestry:
            raise ArtifactError("Цикл локальных зависимостей артефакта")
        state.pin(path, _digest(expected))
        if path in state.completed:
            return
        descriptor = self.describe(path)
        state.pin(path, descriptor.descriptor_sha256)
        if verify_objects and descriptor.sha256 not in state.objects:
            self.read_by_sha256(descriptor.sha256)
            state.objects.add(descriptor.sha256)
        for parent, digest in descriptor.artifact_dependencies.items():
            self._verify_dependency(
                parent, digest, state, ancestry | {path}, verify_objects=verify_objects
            )
        state.completed.add(path)

    def _published(self, descriptor: ArtifactDescriptor) -> ArtifactDescriptor:
        if self._verification is not None:
            self._verification.pin(descriptor.path, descriptor.descriptor_sha256)
            self._verification.completed.add(descriptor.path)
        return descriptor

    def _path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
            raise ArtifactError("Нужен относительный путь с разделителем /")
        parts = relative_path.split("/")
        forbidden = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if PurePosixPath(relative_path).is_absolute() or any(
            p in {"", ".", ".."}
            or p.endswith((" ", "."))
            or any(c in p for c in ':\x00<>"|?*')
            or p.split(".")[0].upper() in forbidden
            for p in parts
        ):
            raise ArtifactError("Недопустимый путь артефакта")
        current = self.root
        for part in parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError("Символьные ссылки в хранилище запрещены")
        if not current.resolve().is_relative_to(self.root):
            raise ArtifactError("Путь вышел за границы хранилища")
        return current

    def _atomic_write(self, relative_path: str, data: bytes) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path(relative_path)  # повторная проверка созданных родителей
        handle, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if Path(temporary).read_bytes() != data:
                raise ArtifactError("Промежуточный файл не прошёл проверку")
            os.link(temporary, path)  # атомарно и без перезаписи существующего пути
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _read_verified(self, relative_path: str) -> tuple[ArtifactDescriptor, bytes]:
        try:
            encoded = self._path(relative_path + ".artifact.json").read_bytes()
            digest = (
                self._path(relative_path + ".artifact.sha256").read_text(encoding="ascii").strip()
            )
            if sha256_bytes(encoded) != digest:
                raise ArtifactError("Повреждён дескриптор артефакта")
            raw = strict_json_loads(encoded)
            descriptor = ArtifactDescriptor(**raw)
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactError("Нет корректного дескриптора завершённой публикации") from exc
        if descriptor.path != relative_path:
            raise ArtifactError("Путь не совпал с дескриптором")
        data = self._path(relative_path).read_bytes()
        if len(data) != descriptor.size_bytes or sha256_bytes(data) != descriptor.sha256:
            raise ArtifactError("Содержимое не совпало с дескриптором")
        if self._verification is not None:
            self._verification.pin(relative_path, descriptor.descriptor_sha256)
        return descriptor, data

    def describe(self, relative_path: str) -> ArtifactDescriptor:
        return self._read_verified(relative_path)[0]

    def read_bytes(self, relative_path: str) -> bytes:
        return self._read_verified(relative_path)[1]

    def verify_compatible(
        self,
        relative_path: str,
        *,
        expected_inputs: Mapping[str, str],
        expected_code_sha256: str,
        expected_schema_version: str,
        expected_configuration: Mapping[str, Any],
        expected_descriptor_sha256: str | None = None,
    ) -> ArtifactDescriptor:
        descriptor = self.describe(relative_path)
        if (
            descriptor.inputs != dict(expected_inputs)
            or descriptor.code_sha256 != expected_code_sha256
            or descriptor.schema_version != expected_schema_version
            or descriptor.configuration != dict(expected_configuration)
        ):
            raise ArtifactError("Артефакт несовместим с ожидаемыми зависимостями")
        if (
            expected_descriptor_sha256 is not None
            and descriptor.descriptor_sha256 != expected_descriptor_sha256
        ):
            raise ArtifactError("Дескриптор не совпал с закреплённой идентичностью")
        return descriptor

    def read_by_sha256(self, digest: str) -> bytes:
        path = self._path(f".objects/{_digest(digest)}")
        data = path.read_bytes()
        if sha256_bytes(data) != digest:
            raise ArtifactError("Повреждён объект SHA-256")
        if self._verification is not None:
            self._verification.objects.add(digest)
        return data

    def publish_bytes(
        self,
        relative_path: str,
        data: bytes,
        *,
        code_sha256: str,
        run_id: str,
        scope: str,
        config_id: str,
        schema_version: str = "1",
        artifact_version: str = "1",
        inputs: Mapping[str, str] | None = None,
        configuration: Mapping[str, Any] | None = None,
        kind: str = "bytes",
        metadata: Mapping[str, Any] | None = None,
        reuse: bool = False,
        artifact_dependencies: Mapping[str, str] | None = None,
    ) -> ArtifactDescriptor:
        self._path(relative_path)
        if relative_path.startswith(".objects/") or relative_path.endswith(
            (".artifact.json", ".artifact.sha256")
        ):
            raise ArtifactError("Зарезервированный путь")
        if not isinstance(data, bytes) or not all(
            isinstance(x, str) and bool(x)
            for x in (run_id, scope, config_id, schema_version, artifact_version, kind)
        ):
            raise ArtifactError("Не заполнены обязательные сведения артефакта")
        dependencies = dict(inputs or {})
        for value in dependencies.values():
            _digest(value)
        # Raw external inputs use payload hashes; local dependencies identify entire
        # immutable descriptors, including row IDs and configuration. Only completed
        # predecessors are allowed, so no forward references or publication cycles.
        local_dependencies = dict(artifact_dependencies or {})
        state = self._verification if self._verification is not None else _VerificationState()
        for path, digest in local_dependencies.items():
            self._verify_dependency(path, digest, state, {relative_path})
        descriptor = ArtifactDescriptor(
            relative_path,
            sha256_bytes(data),
            len(data),
            schema_version,
            artifact_version,
            dependencies,
            dict(configuration or {}),
            _digest(code_sha256),
            run_id,
            scope,
            config_id,
            kind,
            datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            dict(metadata or {}),
            local_dependencies,
        )
        encoded = canonical_json_bytes(descriptor.to_dict())
        if self._path(relative_path).exists():
            if not reuse:
                raise FileExistsError(relative_path)
            existing = self.describe(relative_path)
            old, new = existing.to_dict(), descriptor.to_dict()
            old.pop("created_at_utc")
            new.pop("created_at_utc")
            if old != new:
                raise ArtifactError("Зависимости или содержимое повторного запуска изменились")
            self.read_by_sha256(existing.sha256)
            return self._published(existing)
        self._atomic_write(relative_path, data)
        object_path = f".objects/{descriptor.sha256}"
        try:
            self._atomic_write(object_path, data)
        except FileExistsError:
            self.read_by_sha256(descriptor.sha256)
        self._atomic_write(
            relative_path + ".artifact.sha256", (sha256_bytes(encoded) + "\n").encode("ascii")
        )
        self._atomic_write(relative_path + ".artifact.json", encoded)
        return self._published(descriptor)

    def publish_json(self, relative_path: str, value: Any, **kwargs: Any) -> ArtifactDescriptor:
        return self.publish_bytes(relative_path, canonical_json_bytes(value), kind="json", **kwargs)

    @staticmethod
    def _data_role(relative_path: str, data_role: str) -> None:
        if data_role not in {"prefix", "targets", "auxiliary"}:
            raise ArtifactError("Неизвестная роль данных")
        component = {"prefix": "prefixes", "targets": "targets"}.get(data_role)
        if component is not None and component not in relative_path.split("/"):
            raise ArtifactError("Префиксы и цели должны храниться в отдельных каталогах")

    def publish_table(
        self,
        relative_path: str,
        table: pa.Table,
        *,
        data_role: str,
        **kwargs: Any,
    ) -> ArtifactDescriptor:
        self._data_role(relative_path, data_role)
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        restored = pq.read_table(io.BytesIO(buffer.getvalue()))
        if not table.equals(restored):
            raise ArtifactError("Parquet не сохранил таблицу")
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.update(
            data_role=data_role,
            rows=table.num_rows,
            columns=table.column_names,
            arrow_schema=str(table.schema),
        )
        return self.publish_bytes(
            relative_path, buffer.getvalue(), kind="parquet", metadata=metadata, **kwargs
        )

    def read_table(self, relative_path: str) -> pa.Table:
        descriptor, data = self._read_verified(relative_path)
        if descriptor.kind != "parquet":
            raise ArtifactError("Артефакт не является таблицей Parquet")
        return pq.read_table(io.BytesIO(data))

    def publish_matrix(
        self,
        relative_path: str,
        matrix: npt.NDArray[Any],
        *,
        row_ids: Sequence[str],
        data_role: str = "auxiliary",
        **kwargs: Any,
    ) -> ArtifactDescriptor:
        self._data_role(relative_path, data_role)
        if (
            matrix.ndim != 2
            or matrix.dtype.hasobject
            or matrix.dtype.kind not in "fiu"
            or not np.isfinite(matrix).all()
        ):
            raise ArtifactError("Нужна двумерная числовая матрица без объектов")
        if (
            len(row_ids) != matrix.shape[0]
            or len(set(row_ids)) != len(row_ids)
            or any(not isinstance(x, str) or not x for x in row_ids)
        ):
            raise ArtifactError("Некорректный порядок ID матрицы")
        for identifier in row_ids:
            identifier.encode("utf-8", errors="strict")
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.update(
            data_role=data_role,
            shape=list(matrix.shape),
            dtype=matrix.dtype.str,
            row_ids=list(row_ids),
        )
        return self.publish_bytes(
            relative_path, buffer.getvalue(), kind="npy", metadata=metadata, **kwargs
        )

    def read_matrix(self, relative_path: str) -> tuple[npt.NDArray[Any], tuple[str, ...]]:
        descriptor, data = self._read_verified(relative_path)
        if descriptor.kind != "npy":
            raise ArtifactError("Артефакт не является матрицей")
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
        meta = descriptor.metadata
        if (
            matrix.ndim != 2
            or matrix.dtype.hasobject
            or matrix.dtype.kind not in "fiu"
            or not np.isfinite(matrix).all()
        ):
            raise ArtifactError("Матрица нарушает численный контракт")
        if list(matrix.shape) != meta["shape"] or matrix.dtype.str != meta["dtype"]:
            raise ArtifactError("Форма или тип матрицы не совпали")
        if matrix.shape[0] != len(meta["row_ids"]):
            raise ArtifactError("Число ID матрицы не совпало")
        ids = meta["row_ids"]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ArtifactError("Некорректные ID матрицы")
        for identifier in ids:
            identifier.encode("utf-8", errors="strict")
        return matrix, tuple(meta["row_ids"])
