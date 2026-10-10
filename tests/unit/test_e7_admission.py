"""Допуск отдельного профиля на искусственных свидетельствах, без расчётов Coveo."""

import copy
import json
from dataclasses import asdict

import pytest

from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes, sha256_bytes
from coveo_recsys.cli.admission import require_training_admission
from coveo_recsys.config import verification_configuration
from coveo_recsys.config.recipes import get_ranker_recipe
from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.retrieval import (
    NO_METADATA_CONFIG_ID,
    NO_METADATA_QUOTAS,
    RetrievalConfig,
)
from tests.unit.test_e4_admission import ROOT
from tests.unit.test_e4_admission import admitted as admitted  # noqa: F401


@pytest.fixture
def no_metadata_admitted(admitted, tmp_path):  # noqa: F811
    admission, kwargs, manifest, save, _, data = admitted
    store = LocalArtifactStore(admission["run_dir"])
    original_manifest = copy.deepcopy(manifest)
    original_preflight = manifest["verification"]["results_by_stage"]["preflight"]
    serial = 0

    def publish(content):
        nonlocal serial
        serial += 1
        return store.publish_bytes(
            f"e7-admission/{serial}",
            content,
            code_sha256="1" * 64,
            run_id=manifest["run_id"],
            scope="actual",
            config_id=NO_METADATA_CONFIG_ID,
        ).sha256

    manifest["base_model"]["config_id"] = NO_METADATA_CONFIG_ID
    for source in ("description", "image", "search"):
        manifest["base_model"]["quotas"][source] = 0
    kwargs["feature_schema_bytes"] = (ROOT / "configs/feature_schema_no_metadata.json").read_bytes()
    kwargs["recipe"] = get_ranker_recipe(NO_METADATA_CONFIG_ID)
    kwargs["retrieval"] = asdict(RetrievalConfig(quotas=NO_METADATA_QUOTAS))
    manifest["ranker"]["resolved_parameters"] = kwargs["recipe"]
    manifest["reproducibility"]["feature_schema_sha256"] = publish(kwargs["feature_schema_bytes"])
    data["config_id"] = NO_METADATA_CONFIG_ID
    manifest["evidence"]["data"] = publish(canonical_json_bytes(data))

    def bind_check():
        manifest["reproducibility"]["resolved_library_defaults_sha256"] = publish(
            canonical_json_bytes(
                {
                    "library": "lightgbm",
                    "version": "4.6.0",
                    "resolved_parameters": kwargs["recipe"],
                }
            )
        )
        check = json.loads(store.read_by_sha256(original_preflight))
        check.update(
            config_id=NO_METADATA_CONFIG_ID,
            configuration_or_reference=sha256_bytes(
                canonical_json_bytes(verification_configuration(manifest))
            ),
        )
        manifest["verification"]["results_by_stage"]["preflight"] = publish(
            canonical_json_bytes(check)
        )
        save()

    bind_check()
    original_path = tmp_path / "baseline-manifest.json"
    original_bytes = canonical_json_bytes(original_manifest)
    original_path.write_bytes(original_bytes)
    original_admission = {
        **admission,
        "manifest": str(original_path),
        "manifest_sha256": sha256_bytes(original_bytes),
    }
    return (
        admission,
        kwargs,
        manifest,
        save,
        publish,
        bind_check,
        original_admission,
        original_preflight,
    )


def test_separate_profile_passes_real_training_admission_with_its_own_evidence(
    no_metadata_admitted,
):
    admission, kwargs, *_ = no_metadata_admitted
    assert require_training_admission(ROOT, admission, **kwargs) == admission["manifest_sha256"]


@pytest.mark.parametrize("foreign", ["schema", "recipe", "quotas"])
def test_hash_bound_baseline_inputs_cannot_be_declared_as_no_metadata(
    no_metadata_admitted, foreign
):
    admission, kwargs, manifest, _, publish, bind_check, *_ = no_metadata_admitted
    if foreign == "schema":
        kwargs["feature_schema_bytes"] = (ROOT / "configs/feature_schema.json").read_bytes()
        manifest["reproducibility"]["feature_schema_sha256"] = publish(
            kwargs["feature_schema_bytes"]
        )
    elif foreign == "recipe":
        kwargs["recipe"] = get_ranker_recipe()
        manifest["ranker"]["resolved_parameters"] = kwargs["recipe"]
    else:
        kwargs["retrieval"] = asdict(RetrievalConfig())
        manifest["base_model"]["quotas"].update(description=40, image=30, search=50)
    bind_check()
    with pytest.raises(InputValidationError, match="preflight"):
        require_training_admission(ROOT, admission, **kwargs)


@pytest.mark.parametrize("foreign", ["admission", "preflight"])
def test_prior_baseline_admission_or_check_cannot_authorize_new_profile(
    no_metadata_admitted, foreign
):
    admission, kwargs, manifest, save, _, _, original_admission, original_preflight = (
        no_metadata_admitted
    )
    if foreign == "admission":
        admission = original_admission
    else:
        manifest["verification"]["results_by_stage"]["preflight"] = original_preflight
        save()
    with pytest.raises(InputValidationError):
        require_training_admission(ROOT, admission, **kwargs)
