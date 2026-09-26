"""Допуск будущего B на придуманных артефактах; эти тесты не допускают Coveo."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.cli.admission import require_training_admission
from coveo_recsys.config import verification_configuration
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.historical import HistoricalConfig
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.io import FILES
from coveo_recsys.ranking import objective
from coveo_recsys.verification.provenance import source_inventory
from tests.reference.manifest_cases import make_manifest_case

ROOT = Path(__file__).resolve().parents[2]


def milliseconds(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


@pytest.fixture
def admitted(tmp_path):
    store = LocalArtifactStore(tmp_path / "manifest-run")
    manifest, _ = make_manifest_case(store, "preflight")
    manifest["scope"] = "actual"
    profile = json.loads((ROOT / "configs/coveo_supply_v1.json").read_bytes())
    input_paths, inputs = {}, {}
    for key, name in FILES.items():
        item = next(item for item in manifest["dataset"]["files"] if item["name"] == name)
        content = store.read_by_sha256(item["sha256"])
        path = tmp_path / name
        path.write_bytes(content)
        input_paths[name] = str(path)
        inputs[key] = item["sha256"]
        profile["files"][name].update(
            expected_sha256=item["sha256"], expected_size_bytes=len(content)
        )

    def publish(content):
        digest = sha256_bytes(content)
        return store.publish_bytes(
            f"admission/{digest}",
            content,
            code_sha256="1" * 64,
            run_id=manifest["run_id"],
            scope="actual",
            config_id=manifest["base_model"]["config_id"],
        ).sha256

    base = from_json((ROOT / "configs/coveo_preparation.json").read_bytes(), PreparationPlan)
    periods = tuple(
        replace(
            period,
            start_ms=milliseconds(manifest["periods"][f"period_{period.name}"]["start_utc"]),
            end_ms=milliseconds(manifest["periods"][f"period_{period.name}"]["end_utc"]),
            sample_size=3 if period.name == "B" else 0,
        )
        for period in base.periods
    )
    plan = replace(
        base,
        periods=periods,
        coverage=replace(
            base.coverage,
            start_ms=milliseconds(manifest["dataset"]["coverage_start_utc"]),
            end_ms=milliseconds(manifest["dataset"]["coverage_end_utc"]),
        ),
    )
    plan_bytes, profile_bytes = to_json(plan), canonical_json_bytes(profile)
    inputs.update(plan=sha256_bytes(plan_bytes), profile=sha256_bytes(profile_bytes))
    prepared = dict(
        scope="actual",
        status="completed",
        evaluation_protocol="observed_purchases_v1",
        telemetry_complete=None,
        inputs=inputs,
        plan_json=plan_bytes.decode(),
        profile_json=profile_bytes.decode(),
        selection={period.name: ["a", "b"] if period.name == "B" else [] for period in periods},
        sampled={
            "rule": "lowest_seeded_hash_after_calendar_admission",
            "normative_sample": True,
            "period_counts": {"B": {"selected": 2}},
        },
    )
    # Законный недобор: запрошено 3, допущено и выбрано 2, без выбора по покупке.
    manifest["dataset"]["session_sampling"]["session_count_by_period"]["period_B"] = 2
    manifest["dataset"]["vector_dimensions"] = dict(description=50, image=50, query=50)
    manifest["dataset"]["schema_sha256"] = publish(profile_bytes)
    schema_bytes = (ROOT / "configs/feature_schema.json").read_bytes()
    repro = manifest["reproducibility"]
    for key, content in {
        "code_commit_or_archive_sha256": canonical_json_bytes(source_inventory(ROOT)),
        "dependency_lockfile_sha256": (ROOT / "uv.lock").read_bytes(),
        "feature_schema_sha256": schema_bytes,
        "ranker_callback_sha256": Path(objective.__file__).read_bytes(),
    }.items():
        repro[key] = publish(content)
    data = json.loads(store.read_by_sha256(manifest["evidence"]["data"]))
    data.update(
        scope="actual",
        preparation_plan_sha256=inputs["plan"],
        preparation_report_sha256=sha256_bytes(canonical_json_bytes(prepared)),
    )
    manifest["evidence"]["data"] = publish(canonical_json_bytes(data))
    check = json.loads(
        store.read_by_sha256(manifest["verification"]["results_by_stage"]["preflight"])
    )
    check.update(
        scope="actual",
        code_version=repro["code_commit_or_archive_sha256"],
        configuration_or_reference=sha256_bytes(
            canonical_json_bytes(verification_configuration(manifest))
        ),
    )
    manifest["verification"]["results_by_stage"]["preflight"] = publish(canonical_json_bytes(check))
    historical = store.publish_bytes(
        "historical-fixture",
        b"artificial descriptor; package loader is checked separately",
        inputs={**inputs, "plan": "2" * 64},
        configuration=asdict(HistoricalConfig(snapshot_time_ms=periods[0].end_ms)),
        code_sha256="3" * 64,
        run_id="older-A_G-fixture",
        scope="actual",
        config_id="historical",
    )
    path = tmp_path / "manifest.json"
    admission = dict(manifest=str(path), run_dir=str(store.root), input_paths=input_paths)

    def save():
        content = canonical_json_bytes(manifest)
        path.write_bytes(content)
        admission["manifest_sha256"] = sha256_bytes(content)

    save()
    kwargs = dict(
        prepared=prepared,
        historical=historical,
        feature_schema_bytes=schema_bytes,
        recipe=manifest["ranker"]["resolved_parameters"],
        retrieval=asdict(RetrievalConfig()),
    )
    return admission, kwargs, manifest, save, publish, data


def test_actual_gate_uses_real_validator_and_keeps_shortfalls_and_old_ag_code(admitted):
    admission, kwargs, *_ = admitted
    assert require_training_admission(ROOT, admission, **kwargs) == admission["manifest_sha256"]


def test_synthetic_pair_needs_no_actual_admission(admitted):
    _, kwargs, *_ = admitted
    kwargs["prepared"] = {"scope": "synthetic"}
    kwargs["historical"] = replace(kwargs["historical"], scope="synthetic")
    assert require_training_admission(ROOT, None, **kwargs) is None


@pytest.mark.parametrize("bad", ["missing", "mixed", "manifest_hash", "raw", "scope"])
def test_missing_or_substituted_admission_is_rejected(admitted, bad):
    admission, kwargs, manifest, save, *_ = admitted
    if bad == "missing":
        admission = None
    elif bad == "mixed":
        kwargs["historical"] = replace(kwargs["historical"], scope="synthetic")
    elif bad == "manifest_hash":
        admission["manifest_sha256"] = "0" * 64
    elif bad == "raw":
        Path(admission["input_paths"]["browsing_train.csv"]).write_bytes(b"other input")
    else:
        manifest["scope"] = "synthetic"
        save()
    with pytest.raises(InputValidationError):
        require_training_admission(ROOT, admission, **kwargs)


@pytest.mark.parametrize(
    "bad",
    ["code", "callback", "lock", "schema", "recipe", "retrieval", "historical", "plan", "report"],
)
def test_current_inputs_and_configuration_must_match_the_validated_run(admitted, bad):
    admission, kwargs, manifest, save, *_ = admitted
    fields = {
        "code": "code_commit_or_archive_sha256",
        "callback": "ranker_callback_sha256",
        "lock": "dependency_lockfile_sha256",
    }
    if bad in fields:
        manifest["reproducibility"][fields[bad]] = "0" * 64
    elif bad == "schema":
        kwargs["feature_schema_bytes"] += b" "
    elif bad == "recipe":
        kwargs["recipe"] = {"model": {}}
    elif bad == "retrieval":
        kwargs["retrieval"]["max_candidates"] = 299
    elif bad == "historical":
        kwargs["historical"] = replace(
            kwargs["historical"], inputs={**kwargs["historical"].inputs, "browsing": "0" * 64}
        )
    elif bad == "plan":
        value = json.loads(kwargs["prepared"]["plan_json"])
        value["data"]["periods"][2]["sample_size"] = 9
        content = canonical_json_bytes(value)
        kwargs["prepared"]["plan_json"] = content.decode()
        kwargs["prepared"]["inputs"]["plan"] = sha256_bytes(content)
    else:
        kwargs["prepared"]["selection"]["B"] = ["a"]
    save()
    with pytest.raises(InputValidationError):
        require_training_admission(ROOT, admission, **kwargs)


def test_a_failed_preflight_evidence_cannot_be_replaced_by_a_success_flag(admitted):
    admission, kwargs, manifest, save, publish, data = admitted
    data["calendar_admission_checked"] = False
    manifest["evidence"]["data"] = publish(canonical_json_bytes(data))
    manifest["verification"]["data_status"] = "passed"
    save()
    with pytest.raises(InputValidationError, match="preflight"):
        require_training_admission(ROOT, admission, **kwargs)


@pytest.mark.parametrize("changed", ["horizon", "period", "coverage", "confirmed"])
def test_even_hash_bound_preparation_must_match_manifest_calendar_and_task(admitted, changed):
    admission, kwargs, manifest, save, publish, data = admitted
    plan = json.loads(kwargs["prepared"]["plan_json"])
    if changed == "horizon":
        plan["data"]["task"]["horizon_ms"] += 1000
    elif changed == "period":
        plan["data"]["periods"][2]["start_ms"] += 1000
    elif changed == "coverage":
        plan["data"]["coverage"]["start_ms"] -= 1000
    else:
        plan["data"]["coverage"]["confirmed_at_ms"] = plan["data"]["coverage"]["end_ms"]
    content = canonical_json_bytes(plan)
    kwargs["prepared"]["plan_json"] = content.decode()
    kwargs["prepared"]["inputs"]["plan"] = sha256_bytes(content)
    data["preparation_plan_sha256"] = sha256_bytes(content)
    data["preparation_report_sha256"] = sha256_bytes(canonical_json_bytes(kwargs["prepared"]))
    manifest["evidence"]["data"] = publish(canonical_json_bytes(data))
    save()
    with pytest.raises(InputValidationError):
        require_training_admission(ROOT, admission, **kwargs)
