import json

import pyarrow as pa
import pytest

from coveo_recsys.artifacts import LocalArtifactStore
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.data.access import DatasetAccess


@pytest.mark.parametrize("scope", ["synthetic", "actual"])
def test_access_checks_period_kind_path_and_records_attempts(tmp_path, scope):
    store = LocalArtifactStore(tmp_path / "run")
    access = DatasetAccess(store, code_sha256="a" * 64, run_id="synthetic", scope=scope)
    allowed = "datasets/B/prefixes/s.parquet"
    store.publish_table(
        allowed,
        pa.table({"record_json": [b"test"]}),
        data_role="prefix",
        code_sha256="a" * 64,
        run_id="synthetic",
        schema_version="2",
        scope=scope,
        config_id="e1_code",
        configuration={"period": "B", "dataset_kind": "prefixes"},
    )
    assert access.read(allowed, purpose="prefix") == store.read_bytes(allowed)
    for path, purpose in [
        (allowed, "labels"),
        ("datasets/D/targets/s.parquet", "labels"),
        ("datasets/A_N/prefixes/s.parquet", "prefix"),
        ("datasets/B/prefixes/../../D/targets/s.parquet", "labels"),
        ("datasets//B/prefixes/s.parquet", "prefix"),
        (str(tmp_path / "run" / allowed), "prefix"),
    ]:
        with pytest.raises(InputValidationError):
            access.read(path, purpose=purpose)
    logs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (store.root / "access").glob("*.json")
        if not p.name.endswith(".meta.json")
    ]
    # Хранилище имеет отдельные дескрипторы; считаем лишь записи попыток.
    logs = [v for v in logs if "allowed" in v]
    assert sum(v["allowed"] for v in logs) == 1
    assert len(logs) == 7
    assert all(v["scope"] == scope for v in logs)


def test_failed_compatibility_is_not_logged_as_granted_access(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = "datasets/B/prefixes/wrong.parquet"
    store.publish_bytes(
        path,
        b"not a table",
        code_sha256="a" * 64,
        run_id="s",
        scope="synthetic",
        config_id="e1_code",
        configuration={"period": "B", "dataset_kind": "prefixes"},
    )
    with pytest.raises(InputValidationError):
        DatasetAccess(store, code_sha256="a" * 64, run_id="s").read(path, purpose="prefix")
    logs = [json.loads(p.read_bytes()) for p in (store.root / "access").glob("*.json")]
    (attempt,) = [v for v in logs if "allowed" in v]
    assert attempt["policy_allowed"] is True
    assert attempt["allowed"] is False
    assert attempt["reason"] == "InputValidationError"
