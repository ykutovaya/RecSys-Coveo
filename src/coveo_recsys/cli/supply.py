"""Проверка фактической поставки без запуска подготовки и открытия периодов."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from coveo_recsys.artifacts import (
    LocalArtifactStore,
    sha256_bytes,
    strict_json_loads,
)
from coveo_recsys.cli.preparation import _peak_memory
from coveo_recsys.data.supply import SupplyLimits, inspect_supply
from coveo_recsys.verification.provenance import source_sha256


class SupplyCommandError(ValueError):
    """Ошибка команды без значения поля, строки CSV или локального пути."""


_FILES = ("browsing_train.csv", "search_train.csv", "sku_to_content.csv")


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _read_profile(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(65_537)
    if not raw or len(raw) > 65_536:
        raise SupplyCommandError("profile_size")
    return raw


class _MeteredStore(LocalArtifactStore):
    """Учитывает точные байты временных файлов атомарной публикации."""

    def __init__(self, root: Path, budget_bytes: int) -> None:
        super().__init__(root)
        self.budget_bytes = budget_bytes
        self.temporary_peak_bytes = 0
        self.temporary_written_bytes = 0

    def _atomic_write(self, relative_path: str, data: bytes) -> None:
        written = self.temporary_written_bytes + len(data)
        if len(data) > self.budget_bytes or written > self.budget_bytes:
            raise SupplyCommandError("temp_budget")
        self.temporary_written_bytes = written
        self.temporary_peak_bytes = max(self.temporary_peak_bytes, len(data))
        super()._atomic_write(relative_path, data)


def _sha256_with_deadline(path: Path, deadline: float) -> str:
    """Повторное хеширование с проверкой срока перед каждым чтением блока."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if time.monotonic() > deadline:
                raise SupplyCommandError("time_budget")
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def inspect_supply_command(
    store: LocalArtifactStore,
    project_root: Path,
    run_id: str,
    input_dir: Path,
    profile_path: Path,
    *,
    batch_size: int,
    max_batch_chars: int,
    max_header_bytes: int,
    max_record_bytes: int,
    max_field_chars: int,
    max_seconds: int,
    max_temp_bytes: int,
) -> dict[str, Any]:
    """Сканирует один раз, затем заново хеширует все внешние зависимости."""
    store = _MeteredStore(store.root, max_temp_bytes)
    started = time.monotonic()
    deadline = started + max_seconds
    limits = SupplyLimits(
        batch_size=batch_size,
        max_batch_chars=max_batch_chars,
        max_header_bytes=max_header_bytes,
        max_record_bytes=max_record_bytes,
        max_field_chars=max_field_chars,
        max_seconds=max_seconds,
        max_temp_bytes=max_temp_bytes,
    )
    paths = [input_dir / name for name in _FILES] + [profile_path]
    identities = {path: _file_identity(path) for path in paths}
    raw_profile = _read_profile(profile_path)
    try:
        profile = strict_json_loads(raw_profile)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SupplyCommandError("profile_invalid") from exc
    if not isinstance(profile, dict):
        raise SupplyCommandError("profile_invalid")

    profile_sha = sha256_bytes(raw_profile)
    code_sha = source_sha256(project_root)
    execution_id = uuid4().hex
    sequence = 0
    common: dict[str, Any] = {
        "code_sha256": code_sha,
        "run_id": run_id,
        "scope": "actual",
        "config_id": "coveo_supply_v1",
    }

    def progress(filename: str, rows_read: int, bytes_read: int) -> None:
        nonlocal sequence
        if filename not in _FILES or rows_read < 0 or bytes_read < 0:
            raise SupplyCommandError("progress_invalid")
        sequence += 1
        entry = {
            "file": filename,
            "raw_rows": rows_read,
            "bytes_read": bytes_read,
            "elapsed_seconds": time.monotonic() - started,
            **_peak_memory(),
        }
        store.publish_json(
            f"supply/progress/{execution_id}/{sequence:06}.json",
            entry,
            **common,
        )

    scan = inspect_supply(input_dir, profile, limits=limits, progress=progress)
    scan_temporary_peak = store.temporary_peak_bytes
    scan_temporary_written = store.temporary_written_bytes
    if not scan.get("scan_complete") or scan.get("format_status") != "confirmed":
        return {
            "status": "incomplete",
            "reason": "format_not_confirmed",
            "scan_complete": bool(scan.get("scan_complete")),
            "format_status": scan.get("format_status", "unknown"),
            "coverage_confirmed": False,
            "preflight_ready": False,
            "real_prepare_ready": False,
        }

    files = scan.get("files")
    if not isinstance(files, dict) or set(files) != set(_FILES):
        raise SupplyCommandError("scan_incomplete")
    for name in _FILES:
        record = files[name]
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise SupplyCommandError("scan_incomplete")
        if _sha256_with_deadline(input_dir / name, deadline) != record["sha256"]:
            raise SupplyCommandError("input_changed")
        if time.monotonic() - started > max_seconds:
            raise SupplyCommandError("time_budget")
    if sha256_bytes(_read_profile(profile_path)) != profile_sha:
        raise SupplyCommandError("profile_changed")
    if source_sha256(project_root) != code_sha:
        raise SupplyCommandError("code_changed")
    if any(_file_identity(path) != identities[path] for path in paths):
        raise SupplyCommandError("input_changed")
    if time.monotonic() > deadline:
        raise SupplyCommandError("time_budget")

    inputs = {name: files[name]["sha256"] for name in _FILES}
    inputs["profile"] = profile_sha
    configuration = asdict(limits)
    profile_artifact = store.publish_bytes(
        "supply/profile.json",
        raw_profile,
        kind="json",
        inputs={"profile": profile_sha},
        configuration=configuration,
        reuse=True,
        **common,
    )
    resources = {
        "elapsed_seconds": time.monotonic() - started,
        **_peak_memory(),
        "scan_passes_per_file": 1,
        "final_hash_passes_per_file": 1,
        "total_input_bytes_read": 2 * sum(files[name]["size_bytes"] for name in _FILES),
        "scan_temporary_disk_peak_bytes": scan_temporary_peak,
        "scan_temporary_disk_written_bytes": scan_temporary_written,
        "temporary_disk_budget_bytes": max_temp_bytes,
    }
    resource_artifact = store.publish_json(
        f"supply/resources/{execution_id}.json",
        resources,
        inputs=inputs,
        configuration=configuration,
        **common,
    )
    report = {
        **scan,
        "schema_version": "coveo_supply_report_v1",
        "scope": "actual",
        "release_claimed": profile.get("release"),
        "source_url": "https://github.com/coveooss/SIGIR-ecom-data-challenge",
        "profile_sha256": profile_sha,
        "code_sha256": code_sha,
        "raw_rows_are_not_canonical_events": True,
        "local_hashes_do_not_prove_publisher_authenticity": True,
        "query_description_compatibility": "publisher_statement",
        "image_space_separate": True,
        "coverage_confirmed": False,
        "preflight_ready": False,
        "real_prepare_ready": False,
        "full_scan_passes_per_file": 1,
        "fresh_hash_passes_per_file": 1,
    }
    descriptor = store.publish_json(
        "supply/report.json",
        report,
        inputs=inputs,
        configuration=configuration,
        schema_version="coveo_supply_report_v1",
        artifact_dependencies={profile_artifact.path: profile_artifact.descriptor_sha256},
        reuse=True,
        **common,
    )
    return {
        "status": "confirmed",
        "scan_complete": True,
        "format_status": "confirmed",
        "coverage_confirmed": False,
        "preflight_ready": False,
        "real_prepare_ready": False,
        "report_path": descriptor.path,
        "report_sha256": descriptor.sha256,
        "resources_path": resource_artifact.path,
    }
