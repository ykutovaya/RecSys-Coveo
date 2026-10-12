# Schemas

English | [Русский](/schemas/README.ru.md)

This directory contains versioned JSON Schemas for the run manifest and feature schema.

| File | Purpose |
|---|---|
| [manifest.schema.json](/schemas/manifest.schema.json) | Manifest format 2.10 for `observed_purchases_v1`: required sections, types, allowed values, hashes, stages, and result references. Unknown fields are rejected. |
| [feature.schema.json](/schemas/feature.schema.json) | Format of the ordered feature definitions: name, type, formula, mask, transformation, provenance, and fitting period. |

The schemas check JSON structure. The [validator](/src/coveo_recsys/config/)
checks cross-field constraints, file hashes, values allowed at the current stage,
feature order, and protection against target leakage. The JSON Schema reference
inside each file does not require network access during normal validation.

`stage_validation.precision_required_by_stage` is fixed at `pre_D`.
The interval half-width and justification for the variance and dependence
assumptions may remain `null` for the first training run, but are required
before opening D. All other data and model admission checks still apply.

`contracts.serialization.record_schema` generates schemas for internal records
from the same types used by the strict JSON codec. Semantic constraints are
checked when a record is created.

The main model's 93 columns are defined in [configs/feature_schema.json](/configs/feature_schema.json);
the 54 columns for the variant without metadata are in [configs/feature_schema_no_metadata.json](/configs/feature_schema_no_metadata.json).
`FeatureEngine` also checks which columns are present and their order. The second
profile has no categorical columns; the validator checks that its schema, quotas,
and recipe match the manifest.

Source CSV formats are defined by profiles in `configs/`, not here. See the
[configuration guide](/configs/README.md). Requirements:
[implementation specification](/documents/en/implementation_spec.md#4-run-manifest-and-stage-control).
