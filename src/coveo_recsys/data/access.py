"""Локальная маршрутизация наборов; не замена правам рабочей СУБД."""

from pathlib import PurePosixPath
from uuid import uuid4

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.preparation import EVALUATION_PROTOCOL, PREPARATION_VERSION


class DatasetAccess:
    """База читает только A_G/B/C_smoke; закрытые исходы не открываются аргументом пути."""

    def __init__(
        self, store: LocalArtifactStore, *, code_sha256: str, run_id: str, scope: str = "synthetic"
    ) -> None:
        if scope not in {"synthetic", "actual"}:
            raise InputValidationError("Неизвестная область доступа")
        self.store = store
        self.code_sha256 = code_sha256
        self.run_id = run_id
        self.scope = scope

    def read(self, relative_path: str, *, purpose: str) -> bytes:
        path = PurePosixPath(relative_path)
        parts = path.parts
        allowed = (
            str(path) == relative_path
            and not path.is_absolute()
            and "\\" not in relative_path
            and len(parts) == 4
            and parts[0] == "datasets"
            and ".." not in parts
            and parts[1] in {"A_G", "B", "C_smoke"}
            and (purpose, parts[2]) in {("prefix", "prefixes"), ("labels", "targets")}
            and path.suffix == ".parquet"
        )
        audit = {
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
            descriptor = self.store.describe(relative_path)
            if (
                descriptor.configuration.get("period") != parts[1]
                or descriptor.configuration.get("dataset_kind") != parts[2]
                or descriptor.run_id != self.run_id
                or descriptor.code_sha256 != self.code_sha256
                or descriptor.kind != "parquet"
                or descriptor.schema_version != PREPARATION_VERSION
                or descriptor.config_id != "e1_code"
                or descriptor.scope != self.scope
                or descriptor.configuration.get("diagnostic", False)
                or descriptor.configuration.get("admitted") is not True
                or descriptor.configuration.get("evaluation_protocol") != EVALUATION_PROTOCOL
                or descriptor.configuration.get("record_type")
                != {"prefix": "PreparedRequest", "labels": "TargetRecord"}[purpose]
                or descriptor.metadata.get("data_role")
                != {"prefix": "prefix", "labels": "targets"}[purpose]
            ):
                raise InputValidationError("Набор не соответствует маршруту, схеме или версии кода")
            result = self.store.read_bytes(relative_path)
            audit["allowed"] = True
            return result
        except (ValueError, OSError) as exc:
            audit["reason"] = type(exc).__name__
            raise
        finally:
            self.store.publish_json(
                f"access/{uuid4().hex}.json",
                audit,
                code_sha256=self.code_sha256,
                run_id=self.run_id,
                scope=self.scope,
                config_id="e1_code",
            )
