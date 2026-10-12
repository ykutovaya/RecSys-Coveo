# Configurations

English | [Русский](/configs/README.ru.md)

This directory contains versioned settings for data processing, models, and checks.

The main profile uses 300 candidates. D and `post_D` are complete, with statistical
status `insufficient`. The trained 370-candidate variant was tested on C_select but
not selected. Configurations pointing into `runs/` identify local results; they do
not include those results. The [reproduction guide](/documents/en/reproduction.md)
covers the existing demonstration and a new small run from three CSV files through
C_smoke. For a new run, the [helper](/src/coveo_recsys/verification/reproduce.py)
creates plans and configurations with the actual hashes; paths from earlier runs
are not reused. The small Coveo run was checked from a separate, non-editable
package installation.

| File | Purpose |
|---|---|
| [coveo_supply_v1.json](/configs/coveo_supply_v1.json) | Format, dimensions, sizes, and SHA-256 hashes of the actual CSV files. Used by the supply check and preparation adapter; does not establish observation completeness. |
| [coveo_preparation.json](/configs/coveo_preparation.json) | `observed_purchases_v1`: one request per session; up to 50,000 complete A_G sessions after calendar admission; seed 17 and resource limits. Telemetry completeness is unknown. |
| [coveo_historical.json](/configs/coveo_historical.json) | Fixed prepared A_G, metadata, graph parameters, and E2 resource limits. A new build does not change the input preparation and does not require `preflight`. |
| [synthetic_csv.json](/configs/synthetic_csv.json) | Maps synthetic CSV columns to internal fields. |
| [synthetic_preparation.json](/configs/synthetic_preparation.json) | Synthetic preparation plan: target, seven periods, sample sizes, seed, and resource limits. D is disabled; preparing it requires separate admission. Not used for Coveo. |
| [feature_schema.json](/configs/feature_schema.json) | The 93 E3 columns: order, types, formulas, masks, transformations, and provenance. `FeatureEngine` checks the schema; `FeatureBatch` stores its hash. This is a specific feature set, not a JSON Schema for its format. |
| [ranker_recipe.json](/configs/ranker_recipe.json) | Fixed LightGBM parameters for the first run. The validator checks them against the manifest; `train ranker` uses them without tuning. |
| [feature_schema_no_metadata.json](/configs/feature_schema_no_metadata.json) | Separate E7 schema: 54 numeric and Boolean columns, without category or metadata features. |
| [feature_schema_metadata370.json](/configs/feature_schema_metadata370.json) | Schema for the completed 370-candidate experiment; does not replace the main profile. |
| [ranker_recipe_no_metadata.json](/configs/ranker_recipe_no_metadata.json) | Same recipe and limits as the main model; the only difference is `dataset.categorical_feature=[]`. |
| [coveo_service.json](/configs/coveo_service.json) | Local deployment of the prepared package and model, checked hashes, calendar, load limits, and state adapter. `timeout_ms=1000` bounds the whole request from the first HTTP byte; `reserve_ms=100` leaves time for the fallback, write, and response. Both fields are integers with `0 < reserve_ms < timeout_ms` within int64. The owner secret comes from `COVEO_LOCAL_TOKEN`, not JSON. |
| [test_registry.json](/configs/test_registry.json) | Maps requirements to tests and independent reference implementations. It neither runs tests nor stores results. |

Admission rules are defined by the [validator](/src/coveo_recsys/config/manifest.py)
under the [implementation specification, §15](/documents/en/implementation_spec.md#15-testing-and-traceability); the registry does not duplicate them.

`features build` takes settings for a particular run: fixed preparation and
historical package, feature schema path, `RetrievalConfig`, and resource limits.
Real B requires separate E1-10 admission; a configuration alone does not open it.

Raw CSV files, run results, and secrets are not stored here. The CSV files go in
`data/01_raw/`, results in `runs/<run_id>/`; both directories are ignored by Git.

`sample_size` sets the sample size after admission. `max_sessions` limits the
number of saved canonical sessions, not the number of examined IDs. The technical
mode (`diagnostic_sessions > 0`) is not used. Selection reads both full CSV files;
a bounded-cache temporary SQLite table collects brief session records.
`max_temp_bytes` limits that table, sorting, and incomplete shared Parquet files.
For 50,000 sessions, the limits are 4 GiB of temporary disk, 8 GiB of memory,
and 14,400 seconds. These are limits, not expected usage. Free space is checked
before the run; output size is measured separately. Changing the plan requires
a new run.

The actual sample size is `min(50000, N_admitted)`. A shortfall is allowed only
when fewer sessions pass admission, not when rows are lost or resources run out.
An empty sample cannot be used for E2. Purchase targets do not affect selection.

The historical build has separate limits: 7,200 seconds, 4 GiB of memory,
2 GiB of temporary files, and 1 GiB of output. The graph limits intermediate
pairs and group size, not only the number of stored edges. The output report's
path and hash identify the migrated 50,000-session A_G. Separate preparation
descriptors are not needed.

Coveo periods follow calendar weeks on the published shifted UTC timeline,
chosen after inspecting timestamps only: A_G runs from 21 January to
4 February 2019; A_N ends 18 February; B ends 4 March; C_smoke ends 11 March;
C_select ends 25 March; E ends 1 April; D ends 15 April. Left boundaries are
inclusive, right boundaries exclusive. Other periods have `sample_size=0` and
are not prepared in this run. `coverage.gaps=[]` with `confirmed_at_ms=null`
means gaps are unknown, not proven absent.

Preparation permits real A_G and B. B uses a separate plan in `runs/`: a
nonzero B sample, zero samples for all other periods, and the same boundaries
and seed. Preparing B does not require E1-10; building its features and training do.

## E3 feature settings

The run-specific file is stored in `runs/`, for example
`runs/e3/features.json`. Paths below are relative to the repository root;
replace hashes with the actual SHA-256 values of `reports/preparation.json`
and `historical/package.json`. Both inputs must have the same scope: `synthetic`
or `actual`, with the admission described below.

```json
{
  "preparation": {"path": "runs/test-prepared", "report_sha256": "<SHA-256>"},
  "historical": {"path": "runs/test-historical", "package_sha256": "<SHA-256>"},
  "feature_schema": "configs/feature_schema.json",
  "retrieval": {},
  "limits": {
    "max_requests": 100,
    "batch_size": 16,
    "max_observed_skus": 1000,
    "max_seconds": 300,
    "max_memory_bytes": 2147483648,
    "max_result_bytes": 67108864
  }
}
```

An empty `retrieval` uses the [RetrievalConfig](/src/coveo_recsys/contracts/retrieval.py)
defaults. If a prefix contains an SKU outside the historical catalog, add
`metadata_path` pointing to the same fixed metadata supply. It is read once
for the selected SKUs, not for every request. Exceeding a limit fails the run;
it does not truncate the data.

## E4 training settings

The run-specific JSON file is stored in `runs/`. It specifies:

- `preparation`, `historical`, `feature_schema`: the same inputs used to build features;
- `features`: the path to the E3 result and the `report_sha256` of `reports/features.json`;
- `recipe`: `configs/ranker_recipe.json`;
- `limits`: positive `max_requests`, `max_rows`, `max_seconds`, `max_memory_bytes`, and `max_result_bytes`.

`max_rows` limits the advance estimate `N_B⁺ × M`, not just the candidates
actually found. The matrix and Dataset must fit in memory in full. The sample
is not silently reduced, and training is not split into partial runs.

Actual `features build` and `train ranker` runs require `admission` with
`manifest`, `manifest_sha256`, an evidence `run_dir`, and `input_paths`
(three CSV names mapped to paths). Existing `preflight` checks code, schema,
recipe, periods, and selection before reading B. The evidence referenced by
`evidence.data` binds this check to `preparation_plan_sha256` and
`preparation_report_sha256`. The synthetic workflow needs no such admission
and cannot establish it. Precision parameters are required by `pre_D`, not for
the first model; they remain unset until justified by a pilot.

## E7 variant without metadata

The settings retain the E3/E4 format. Both files set
`feature_schema: "configs/feature_schema_no_metadata.json"`. Feature settings
use `retrieval.quotas: [50, 50, 50, 0, 0, 0, 20, 10]` and
`max_candidates: 300`; `metadata_path` is forbidden, including for an unknown
SKU in the history. Training settings use
`recipe: "configs/ranker_recipe_no_metadata.json"` and a hash-checked link
to this variant's feature result. All other recipe parameters stay the same.

`sensitivity metadata --step features` and `sensitivity metadata --step train`
take `--config`, `--run-dir`, and `--run-id` and reuse the existing functions.
Features cover B only. Separate directories and current admission are needed,
with `base_model.config_id=graph_recent_popularity_no_provided_metadata`
and the matching schema, quotas, and recipe. The main model's manifest does
not apply. Resource limits are agreed before a real run. Accepted E7-04
settings are `runs/coveo-b50000-no-metadata-v1/features-config.json` and
`training-config.json` in the same directory; results are in
[README](/README.md#measured-results).

## E5 evaluation settings

The run-specific JSON file is stored in `runs/`. `replay` accepts only C_smoke:

- `preparation`: the C_smoke preparation path and `report_sha256`; `null` before it exists.
- `preparation_plan`: the agreed C_smoke preparation plan path.
- `training_preparation`, `training_features`, `model`: the accepted B preparation, B features, and model, with their report hashes. The B feature table is neither read nor rebuilt.
- `historical`, `feature_schema`, `recipe`, `retrieval`, `metadata_path`: the accepted inputs and main-model settings. Metadata covers observed SKUs outside A_G; it does not expand the catalog.
- `admission`: the same fields as for training, but the new consumer's manifest is at `post_training`. `evidence.data` also records `evaluation_preparation_plan_sha256`, the SHA-256 of the canonical C_smoke plan serialization. The trained-model index points to the existing model; there is no retraining.
- `limits`: positive `max_requests`, `max_observed_skus`, `batch_size`, `max_seconds`, `max_memory_bytes`, and `max_result_bytes`.

Optional `pilot_sessions` selects the first N IDs from prepared C_smoke under
the fixed hash rule. This is a separate run with an explicit size and ID list;
without it, the full sample is evaluated. The pilot retains the original periods
and calendar grid. `max_requests` still fails when exceeded rather than
truncating the sample. C_smoke preparation is not repeated.

`evaluation_plan` fixes `period: "C_smoke"`,
`policies: ["GCS+Rk", "P", "R"]`, `k: 20`, `diagnostic_depth`, and `intervals`.
The latter sets `grid_step_ms`, the primary `block_cells`, two distinct
`sensitivity_block_cells`, and
`mc_error_method: "binomial_order_statistic_95pct"`. The grid start and last
admissible start come from the preparation plan; the step must divide that
window exactly.

Each of the five estimates (`GCS+Rk`, `P`, `R`, `GCS+Rk-P`, `GCS+Rk-R`)
has its own `dependence: {supported, assumptions}` and `half_width_fraction`.
Unjustified values remain `null`. This does not block the pilot, but no
interval can then be reported as adequate. Diagnostic slices are fixed in
code; popularity thresholds use A_G only. See the
[interval requirements](/documents/en/implementation_spec.md#94-intervals-and-sufficiency).

Before preparation, `data prepare --evaluation-config <path>` checks the
existing model, agreed plan, source CSV profile, and `post_training`; it does
not require a C_smoke preparation that does not yet exist. After preparation,
its path and report hash are added to the `replay` settings. B12 is checked
with `operation=evaluate` before reading tables. Changing evaluation code
does not change the producer hashes for B, A_G, or the model.

## The 370-candidate variant and selection on C_select

`gcs_rk_metadata370` uses its [own schema](/configs/feature_schema_metadata370.json)
with the existing columns and the [main recipe](/configs/ranker_recipe.json).
Quotas are `[50, 50, 50, 80, 60, 50, 20, 10]`, with `max_candidates: 370`;
other parameters are unchanged. `features build` and `train ranker` infer the
profile from the schema. Main-model artifacts are not replaced.

Optional `pilot_requests` in feature settings selects the first N B sessions
by the fixed hash, without reading targets. `pilot_completed` status prevents
training on an incomplete set. Timing includes writes; resource checks apply
to both the pilot and full run.

`select --config ... --run-dir ... --run-id ...` compares the main model with
the 370-candidate variant. Settings match the D evaluation except for
`candidate_variant` in place of `metadata_sensitivity`, with fields `model`,
`training_features`, `feature_schema`, `recipe`, and `retrieval`. Admission is
`pre_selection` with an external `before_C_select` reference; only C_select
is active during preparation. The plan fixes
`["GCS+Rk", "gcs_rk_metadata370"]`, their intervals, and the difference
`GCS+Rk-gcs_rk_metadata370`. A positive difference favors the main model.
Preparation uses `data prepare --evaluation-config ...`; afterward, the
prepared set's path and hash are added without changing the fixed plan.

`reports/replay.json` and `.md` record the selection, quality, and cost.
The main model stays when the integer hit totals are equal; errors or no
target-bearing sessions leave no winner. This is selection on C_select, not
independent confirmation of an improvement.

## Final evaluation on D

Future research uses the separate C_select workflow described above. The
completed D results cannot independently confirm an improvement selected later.

`evaluate` uses the same fields and limits as `replay`, with these differences:

- Only D is active in the preparation plan; `evaluation_plan.period` is `D`.
  `pilot_sessions` is forbidden: the entire fixed sample is evaluated.
- `metadata_sensitivity` supplies its own `model`, `training_features` (paths
  and `report_sha256`), `feature_schema`, `recipe`, and `retrieval` for the
  variant without metadata. Existing B and model artifacts are reused; D
  features are built from its prefixes.
- The policies also include `graph_recent_popularity_no_provided_metadata`.
  `intervals.dependence` and `half_width_fraction` need two additional
  estimates: this variant and
  `GCS+Rk-graph_recent_popularity_no_provided_metadata`.
- `admission` requires a `pre_D` manifest and
  `plan_references: {"before_D": "<SHA-256>"}`. `operation=evaluate` is
  checked before preparation. The model index contains both variants.

The completed D evaluation used one untuned GCS+Rk policy:
`primary_estimate_type=single_policy`,
`primary_policy_config_id=gcs_rk_v2_7`. The main model was the only registered
policy. `primary_comparison`, the `before_C_select` reference,
`evidence.selection_plan`, and selection-result hashes remain `null`.
The variant without metadata was not part of selection.

The fixed `evaluation_plan` evidence contains `execution` produced by
[final_execution_binding](/src/coveo_recsys/cli/admission.py): settings without
`admission` and the `preparation` result, plus hashes of the preparation plan,
schemas, and recipes. This fixes the plan before D exists; its summary path
and hash can then be added. Actual `session_count_by_period.period_D` comes
from completed preparation: it is an outcome, not part of the fixed
configuration. The limit and selection rule remain in `before_D`; final
validation matches the evaluated-session count to the selected sessions.
The grid, blocks, and primary half-width match the manifest. Primary
`dependence` matches `inference.dependence_assumptions.single_policy_hit`
(an object with `{supported, assumptions}`). Unknown precision parameters
are not filled in automatically.

Admission checks do not authorize a run. The real D plan and resource limits
are agreed separately; step dependencies are in the
[reproduction guide](/documents/en/reproduction.md#full-computation-workflow).
