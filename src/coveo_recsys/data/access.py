"""Локальная маршрутизация наборов; не замена правам рабочей СУБД."""

from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import (
    EVALUATION_PROTOCOL,
    PREPARATION_VERSION,
    PreparationPlan,
)
from coveo_recsys.contracts.records import Record
from coveo_recsys.contracts.serialization import from_json, record_version
from coveo_recsys.data.prepared import FORMAT_VERSION, iter_sessions, read_report, verify_tables

R = TypeVar("R", bound=Record)


class DatasetAccess:
    """Читает целые сессии A_G/B/C_smoke, разделяя префиксы и цели."""

    def __init__(
        self,
        root: Path,
        *,
        code_sha256: str,
        run_id: str,
        scope: str = "synthetic",
        audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if scope not in {"synthetic", "actual"}:
            raise InputValidationError("Неизвестная область доступа")
        self.root = root.resolve()
        self.code_sha256 = code_sha256
        self.run_id = run_id
        self.scope = scope
        self.audit = audit

    def read(
        self, relative_path: str, kind: type[R], *, purpose: str
    ) -> Iterator[tuple[str, tuple[R, ...]]]:
        path = PurePosixPath(relative_path)
        parts = path.parts
        allowed = (
            str(path) == relative_path
            and not path.is_absolute()
            and "\\" not in relative_path
            and len(parts) == 3
            and parts[0] == "datasets"
            and parts[1] in {"A_G", "B", "C_smoke"}
            and (purpose, parts[2], kind.__name__)
            in {
                ("prefix", "prefixes.parquet", "PreparedRequest"),
                ("labels", "targets.parquet", "TargetRecord"),
            }
        )
        attempt: dict[str, Any] = {
            "path": relative_path,
            "purpose": purpose,
            "policy_allowed": allowed,
            "allowed": False,
            "boundary": "local_routing_only",
            "reason": None,
            "scope": self.scope,
        }
        try:
            if not allowed:
                raise InputValidationError("Маршрут закрыт для указанной цели чтения")
            destination = self.root / relative_path
            if destination.resolve() != destination or any(
                (self.root.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)
            ):
                raise InputValidationError("Ссылка вместо таблицы набора")
            report = read_report(self.root, verify_tables=False)
            info = report["tables"].get(relative_path)
            plan_json = report.get("plan_json")
            if not isinstance(plan_json, str):
                raise InputValidationError("Нет закреплённого плана подготовки")
            plan = from_json(plan_json.encode("utf-8"), PreparationPlan)
            if (
                report.get("format_version") != FORMAT_VERSION
                or report.get("status") != "completed"
                or report.get("source_sha256") != self.code_sha256
                or report.get("run_id") != self.run_id
                or report.get("scope") != self.scope
                or report.get("evaluation_protocol") != EVALUATION_PROTOCOL
                or report.get("telemetry_complete", False) is not None
                or plan.scope != self.scope
                or plan.diagnostic_sessions != 0
                or record_version(kind) != PREPARATION_VERSION
                or not isinstance(info, dict)
                or info.get("period") != parts[1]
                or info.get("record_type") != kind.__name__
                or info.get("data_role") != {"prefix": "prefix", "labels": "targets"}[purpose]
                or info.get("admitted") is not True
            ):
                raise InputValidationError("Набор не соответствует маршруту, схеме или версии кода")
            verify_tables(self.root, report, paths=[relative_path])
        except (ValueError, OSError) as exc:
            attempt["reason"] = type(exc).__name__
            if self.audit:
                self.audit(attempt)
            raise

        def records() -> Iterator[tuple[str, tuple[R, ...]]]:
            try:
                yield from iter_sessions(
                    destination,
                    kind,
                    batch_size=plan.chunk_size,
                    max_session_records=plan.max_session_blocks,
                    max_session_bytes=plan.max_session_bytes,
                )
                verify_tables(self.root, report, paths=[relative_path])
                attempt["allowed"] = True
            except (ValueError, OSError) as exc:
                attempt["reason"] = type(exc).__name__
                raise
            finally:
                if self.audit:
                    self.audit(attempt)

        return records()
