# Development plan for the Coveo session recommender

English | [Русский](/documents/development_plan.md)

Version: 0.34. Date: October 11, 2026. Task status is in [§7](/documents/en/development_plan.md#7-current-status); results are in the [README](/README.md).

## 1. Basis and workflow

Basis: [implementation specification 1.8](/documents/en/implementation_spec.md), [mathematical specification 2.10](/documents/en/math/coveo_session_recommender_math.tex), [run manifest template 2.10](/documents/math/coveo_session_recommender_run_manifest.template.json), and [architecture](/documents/en/architecture.md). Formulas, parameters, and test examples come from these sources; `sec:...`, `eq:...`, and `app:...` are labels in the mathematical specification.

Э0–Э10 are development phases from [implementation specification §16](/documents/en/implementation_spec.md#16-development-stages). Identifiers such as Э0-01 denote tasks in this plan; B/N/X and EXPL01 are checks from the source documents. A development phase does not replace a run manifest stage.

A task is complete once it is implemented and the applicable `pytest` checks pass. Write the test before the code: one rule per test, parameterized boundaries, and properties for general relationships. Reference implementations are independent of the functions under test; tolerances, case counts, and traceability to requirements follow [implementation specification §15.1][checks-common]. Separate test manifests are unnecessary; passing tests does not replace admission of an actual run.

Short tests use synthetic examples; preparation and final evaluation use Coveo. Code acceptance in Э1-09 is separate from admission to training and evaluation in Э1-10.

Main path: data → GCS+Rk → C_smoke → freeze the model and plan → D. C_select is needed for variant selection; the first evaluation on D was performed without tuning. E is reserved and not read. Э6 is a separate service branch after the Э3 contracts; Э7 does not depend on its load testing or readiness.

## 2. Э0 — project foundation

### Э0-01. Environment and package skeleton

- **Dependencies:** architectural boundaries from [architecture §§2–3](/documents/en/architecture.md#2-main-decisions).
- **Deliverable:** an installable package with launch commands; pinned interpreter and libraries for tables, statistics, schemas, HTTP, and testing; environment details and a dependency file with hashes. Neural dependencies are installed separately. Installation, types, and tests are checked automatically; unimplemented commands do not report success.
- **Acceptance:** [implementation specification §2.2](/documents/en/implementation_spec.md#22-environment-and-dependencies), [§15.4][checks-integration]; installation in a clean environment, minimal library calls, and compatibility with the normative LightGBM recipe.

### Э0-02. Types and external boundaries

- **Dependencies:** Э0-01.
- **Deliverable:** data types from [implementation specification §3.2](/documents/en/implementation_spec.md#32-logical-structures) and serialization schemas for early structures: statuses, reasons, time/counter units, and ID order. Interfaces for reading artifacts, state, and time are described without tying them to a database. Prefix inputs are separated from targets and weights; `RequestContext` does not reference a mutable complete session.
- **Acceptance:** the types and representation part of [B03.data][checks-base]; dedicated checks for exact preservation of IDs, time, unknown values, and input/target separation. The remainder of B03.data stays open until Э1.

### Э0-03. Shared numerical and time primitives

- **Dependencies:** Э0-02.
- **Deliverable:** `numerics` with stable operations from [implementation specification §6.2](/documents/en/implementation_spec.md#62-representations-and-numerical-stability), checked time/counter arithmetic, tie-breaking order, masked aggregation of the last group, and independent references. Absent and unusable external vectors are distinguished from each other and from internal computation errors.
- **Acceptance:** the early numerical portion of [B05][checks-base], boundary checks for B01/B03.data; `sec:ops-numerics`, `eq:model-last-group-mean`. B05 is completed after derivative and LightGBM checks in Э4.

### Э0-04. Artifact storage, hashes, and publication

- **Dependencies:** Э0-02.
- **Deliverable:** local storage under [implementation specification §3.3](/documents/en/implementation_spec.md#33-run-directory): pinned table/matrix formats and inputs; staged writing and verification before publication or reuse. Targets are physically separated from prefixes. One run manifest stores references to inputs, code, and settings; results are stored separately, without mandatory per-stage manifest snapshots or a universal artifact registry.
- **Acceptance:** the hashes and compatibility portion of [B13][checks-base]; dedicated checks for interrupted writes, changed inputs, corrupted files, incompatible schemas, and overwrite prevention. Complete secure loading and production-package signatures belong to Э6.

### Э0-05. Run manifest and feature schemas, and check traceability

- **Dependencies:** Э0-02, Э0-04.
- **Deliverable:** run manifest JSON Schema with types and enumerations; `feature_schema` format under [implementation specification §7.2](/documents/en/implementation_spec.md#72-feature-schema); mapping of normative IDs and EXPL01 to tests, with part applicability and deadlines; early synthetic-data recipes and independent references. Unknown values remain unfilled.
- **Acceptance:** the structural portion of [B13][checks-base], [implementation specification §15.1][checks-common], [§15.3][checks-extensions], and [§15.5][checks-explanations].

### Э0-06. Semantic validation and stage conditions

- **Dependencies:** Э0-04, Э0-05.
- **Deliverable:** validation of temporal relationships, pinned inputs, active modules, required fields, and results by their deadlines; manifest commands under [implementation specification §14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior), failure reasons, and valid and invalid examples. Plans are frozen before access to the corresponding evaluation data; a new manifest file is not required at every stage. Test execution and admission decisions are separate; `template_not_runnable` is not accepted as a ready run.
- **Acceptance:** [B13][checks-base], [implementation specification §4](/documents/en/implementation_spec.md#4-run-manifest-and-stage-control), `app:run-stages`: future results are not required early; a missing report required at the current stage blocks admission; a negative result may be saved. Changes to a frozen plan are detected.

### Э0-07. Early test suite and foundation acceptance

- **Dependencies:** Э0-01–Э0-06.
- **Deliverable:** reproducible `pytest` tests without Coveo: schemas, types, and generated properties checked against independent references; results with requirement IDs and limitations. Short checks, Coveo work, and load testing are separate. Future checks are described without replacing them with stubs.
- **Acceptance:** complete B13 as a validator-code check on synthetic cases; implemented early portions of B03.data/B05; [implementation specification §15.4][checks-integration]. Passing this suite does not populate actual-run statuses.

After Э0-07, the environment and schema/validation-code hashes are known. Data and future results may remain unfilled; the exact feature layout and complete ranker configuration are frozen in Э3–Э4 before actual admission.

## 3. Э1 — data, time, and targets

The deliverables below apply to `observed_purchases_v1`. Import and shared transformations are reused; schedule, admission, and target changes are checked first on small examples, then on complete Coveo sessions.

### Э1-01. Source reading and validation

- **Dependencies:** Э0-02, Э0-04, Э0-07.
- **Deliverable:** chunked reading of the three files and a data catalog: validated fields/units, hashes of exact bytes, row provenance, vector dimensions, coverage, and known gaps; a memory/disk plan for grouping. Format checks on a closed period do not expose its purchase metrics in the development report.
- **Acceptance:** [implementation specification §3.1](/documents/en/implementation_spec.md#31-input-files), [B03.data/B13][checks-base]. Dedicated checks for corrupted rows, schema mismatch, and reading with different chunk boundaries.

### Э1-02. Event identity and canonical blocks

- **Dependencies:** Э1-01, Э0-02.
- **Deliverable:** deterministic merging of behavior and search into complete `CanonicalBlock` objects with provenance, `set_v1`, and multiplicities for `multiset_v1`. A stable ID differs from a row reference; verified redeliveries are handled separately. Groups are not split at chunk boundaries; disk sorting is used when needed without loading all tables into memory.
- **Acceptance:** [implementation specification §5.1](/documents/en/implementation_spec.md#51-import-and-canonicalization), [B01/B03.data][checks-base]; independence from row order and chunk sizes, and distinction between a repeated ID and identical rows with different IDs. Search payload validation is completed in Э1-03.

### Э1-03. Search semantics and group aggregates

- **Dependencies:** Э1-02, Э0-03.
- **Deliverable:** distinct search vectors before normalization, with multiplicities and absence/error markers; context from the latest block or a reason for its absence, without reverting to an older block. Shared last-view-group functions account for multiplicities and masks and are used by the future `FeatureEngine`. Search lists and clicks are excluded from baseline computation.
- **Acceptance:** [B02][checks-base] and the B01 portion with normative last-group means; `sec:data-search-canonicalization`, `eq:model-last-group-mean`. Once the complete serving path exists, its independence from search lists is checked again in Э3–Э5.

### Э1-04. Prefix state and request schedule

- **Dependencies:** Э1-02, Э1-03.
- **Deliverable:** a pure `StateReducer`, `PrefixSnapshot`, and immutable prepared requests: observed cart, complete counters, separate bounded histories, and the complete last group. The sole request is created after the first eligible `detail` block, before a recorded purchase, under [implementation specification §5.2](/documents/en/implementation_spec.md#52-prefix-and-schedule); the entire subsequent log is retained. The historical adapter sets a=t, and the snapshot is also checked with different event/availability times. The Э1 record is `PreparedRequest`; `RequestContext` with a catalog hash is created after Э2.
- **Acceptance:** [B01/B02][checks-base] and early cart checks from B03. Changing a session continuation does not change the stored past context. Concurrent service updates and B03.service acknowledgments remain an Э6 task.

### Э1-05. Periods, admission, and reproducible subsampling

- **Dependencies:** Э1-04, Э0-06.
- **Deliverable:** a validated, versioned seven-part plan; session assignment by the first eligible request, normative SHA-256 selection independent of future purchases, selected IDs with all rows, and an exclusion registry. Calendar admission and reading all published rows do not establish telemetry completeness, which remains unknown. A_N and E are not added to baseline training periods.
- **Acceptance:** [B01/B06/B13][checks-base] and the periods portion of B10; [implementation specification §§5.3](/documents/en/implementation_spec.md#53-recorded-targets-and-calendar-admission), [9.1](/documents/en/implementation_spec.md#91-periods-and-result-access), `sec:eval-periods`, `eq:eval-admission`. Actual dates and sample sizes are determined before target analysis using permitted information.

### Э1-06. Recorded targets

- **Dependencies:** Э1-04, Э1-05.
- **Deliverable:** a session record with the first request, last recorded event, and calendar admission; a separate set of all distinct recorded purchases after the request through `min(t0 + H, t_last_obs)`, inclusive. Targets are not filtered by the catalog. An empty target does not become a negative outcome; even a nonempty target does not establish a complete purchase list.
- **Acceptance:** [B01/B06][checks-base], [implementation specification §5.3](/documents/en/implementation_spec.md#53-recorded-targets-and-calendar-admission), `sec:requests`, `sec:eval-maturity`: window boundaries, multiple purchases and repeated purchase of one SKU, purchase in the request block, an empty tail, and a target outside the future catalog. Session continuation does not change the stored prefix.

### Э1-07. Disabling purchase frequency

- **Dependencies:** Э0-05, Э1-06.
- **Deliverable:** preparation does not build a mature purchase cohort or turn unknown outcomes into zeros. The historical purchase-probability feature is absent from the feature schema, rather than replaced with a masked zero. Counts of sessions with a recorded purchase are retained for Э2.
- **Acceptance:** [B09][checks-base], [implementation specification §§6.1](/documents/en/implementation_spec.md#61-catalog-graph-and-statistics), [7.2](/documents/en/implementation_spec.md#72-feature-schema). Manifest validation is already implemented; absence of the feature from actual tables is checked when Э3 is available.

### Э1-08. Static sealed-envelope validation

- **Dependencies:** Э0-02, Э0-06, Э1-02.
- **Deliverable:** shared pure `sealed_block_v1` validation for data and service: consistent fields, a single timestamp and session, IDs, event count, and checksum of the declared bytes. Server time and the allowed range are passed explicitly. No events from a partial or contradictory envelope are accepted.
- **Acceptance:** [B03.data][checks-base], [implementation specification §10.2](/documents/en/implementation_spec.md#102-accepting-sealed_block_v1), `sec:ops-transport-block`, `sec:ops-event-time-validation`. Static validation does not confirm acceptance; receipts, authentication, rate limits, and atomic retries are checked separately in B03.service (local code: Э6-01–Э6-05).

### Э1-09. Preparation workflow and Э1 code acceptance

- **Dependencies:** Э1-01–Э1-08, Э0-04–Э0-07.
- **Deliverable:** `data inspect / prepare`: catalog → blocks → first request → calendar admission → separate recorded targets and exclusion report; small-dataset artifacts and a preparation report. Provenance is retained; period reading is restricted to authorized routes. Targets do not enter feature computation, and closed outcomes cannot be accessed through an alternative command.
- **Acceptance:** parts of [B01/B02/B03.data/B06/B09/B13][checks-base] that apply to preparation, resumption without changed results, and part of the [end-to-end path in implementation specification §15.4][checks-integration]. The full feature schema is checked in Э3, and the path with a model in Э5.

### Coveo preparation adaptation before Э2

The adaptation was completed in Э1-04–Э1-09: CSV files from `data/01_raw/`, the [Coveo profile](/configs/coveo_supply_v1.json), [preparation plan](/configs/coveo_preparation.json), `observed_purchases_v1`, and shared tables. Prepared A_G was expanded from 10 000 to 50 000 sessions; results are in [§7](/documents/en/development_plan.md#7-current-status) and the [reproduction guide](/documents/en/reproduction.md#measured-costs).

Periods and SHA-256 selection of complete sessions are frozen before purchase analysis. All published behavior and search rows, the first request, and separate targets are retained; admission is `t0 + H + g < b_J`, and additionally `t_last_obs + g < b_AG` for A_G. Confirmation of telemetry completeness is not required.

**Acceptance before Э2:** Э1-09, hashes and provenance, row completeness, time boundaries, targets, and nonoverlapping periods; reconciliation with source rows, memory/disk, interruption, and resumption. Applicable checks are [B01/B02/B03.data/B06/B09/B13][checks-base] and B10.data under [implementation specification §5.3](/documents/en/implementation_spec.md#53-recorded-targets-and-calendar-admission).

### Preparing B before Э1-10

**Dependencies:** accepted Э1-09, Э2-08, and Э4-07. Use the existing preparation workflow and CSV files from `/data/01_raw/`. Preserve the completed A_G and historical package without rebuilding.

Conditions for the accepted expansion of B from 10 000 to 50 000 sessions:

1. The plan in `runs/coveo-b50000-v1/` is based on `runs/coveo-b10000-v1/preparation-plan.json`: `B.sample_size=50000`, `max_sessions=50000`, all other sizes are 0. Periods, seed=17, and selection are preserved; the size is frozen before analyzing added purchases. A shortfall is allowed only if fewer eligible sessions exist. Before preparation, estimate `N_B × M`; afterward, estimate `N_B⁺ × M` and Dataset memory under [implementation specification §8.1](/documents/en/implementation_spec.md#81-training-table).
2. This run prepares only B. Permission for B and rejection of all other periods are checked; features and training require Э1-10. C_smoke preparation is performed separately in Э5.
3. `data prepare` with the [supply profile](/configs/coveo_supply_v1.json) stores all published rows for selected sessions, canonical events, admission, first requests, and separate targets in shared tables. Previous results are not overwritten; a full CSV pass is allowed, and a separate incremental computation mechanism is unnecessary. An empty target remains unknown.
4. Check row completeness, admission, windows, and nonoverlap with A_G; the first 10 000 IDs and all records for shared sessions must match the previous run. Select the CSV cross-check sample independently of purchases. Acceptance includes hashes, replay without overwrite, sample composition, resources, interruption, and resumption without changing the plan.

**Accepted-run budgets:** preparation: 4 hours, 8 GiB memory, 4 GiB temporary files, and 1 GiB output; features and training: 4 hours, 8 GiB memory, and 4 GiB output per command. Check free disk space before repeating; exceeding a budget requires agreement on the budget, without truncating data.

**Completion condition:** nonempty B with status `completed`, verified completeness of published rows, and correct separate targets. Checks: [implementation specification §§5.1–5.3](/documents/en/implementation_spec.md#5-preparing-events-requests-and-targets), applicable [B01/B02/B03.data/B06/B09/B10.data][checks-base]. Zero admitted sessions, a failed check, or a budget overrun stops work; absent purchases are not corrected through target-based selection. Readiness of B does not yet authorize feature computation or training.

### Э1-10. Admission of the first actual run

**Dependencies:** verified B from the preceding section, the accepted A_G package from Э2-08, and Э3–Э4 code. A trained model is not required for admission.

1. **Pin the run.** Create one actual run manifest through the standard `manifest init` using the [template](/documents/math/coveo_session_recommender_run_manifest.template.json). Pin current code, environment, source CSV files, B profile and plan, preparation summary, historical package, [feature schema](/configs/feature_schema.json), executable derivatives, and [recipe](/configs/ranker_recipe.json). The code that created the accepted A_G may be older; its inputs, boundaries, and settings must match the run.
2. **Respect evaluation-plan deadlines.** Hit@20 half-width expressed as a proportion, planning variance, assumptions and dependence multiplier, and pilot provenance are not required for Э1-10 and the first model. After training, justify them using the authorized C_smoke pilot and freeze them together with the grid and blocks before opening D under [implementation specification §18](/documents/en/implementation_spec.md#18-decisions-required-before-the-corresponding-stages). Do not present purchase statistics or synthetic examples as measured model quality; leave unknown values unfilled until their deadlines.
3. **Verify the data.** Collect applicable [B01/B02/B03.data/B06/B09/B10.data/B13][checks-base] results on actual inputs using existing result formats. In the evidence whose hash is referenced by `evidence.data`, link the check to the expanded B's `preparation_plan_sha256` and `preparation_report_sha256`; evidence for the previous 10 000 does not replace its verification. Store the `verification.results_by_stage.preflight` report with links to code, configuration, and `run_id`; the validator must be able to access artifacts by their hashes. Distinguish the actual selected-session count from the sample cap. Do not copy synthetic evidence as actual evidence or fill future training, service, or evaluation results.
4. **Check admission.** Run `manifest validate --stage preflight` with external paths to the three CSV files. Prepare the `admission` binding for both commands under the [settings description](/configs/README.md#настройки-обучения-э4). Check it with the [existing admission function](/src/coveo_recsys/cli/admission.py) before feature computation: it additionally binds the manifest to current code, the B plan, selection, and historical package. A success flag or manifest-structure check alone is insufficient.

**Deliverable and acceptance:** complete actual `preflight` and consistent command inputs, or rejection with specific reasons; [implementation specification §4](/documents/en/implementation_spec.md#4-run-manifest-and-stage-control), `app:run-manifest`, `app:run-stages`. After admission, changes to pinned code, inputs, or settings require a new run and repeated admission, without overwriting old results.

Э1-10 is not required for preparing and checking B, preparing and building A_G, or developing Э2–Э5 code. These activities check their own inputs and results but do not assign `preflight=passed`. Features for actual B and training are allowed only after Э1-10.

## 4. Э2–Э10 — development tasks

Unless stated otherwise, dependencies refer to code; actual computations follow the applicable input checks. Э1-10 provides full admission to training; command order is in §5.

### 4.1. Э2: historical package

**Input:** results of [preparation before Э2](/documents/en/development_plan.md#coveo-preparation-adaptation-before-э2). Building on Coveo does not require ranker code or its feature schema; short tests use synthetic examples.

The computational basis is [implementation specification §6](/documents/en/implementation_spec.md#6-historical-package-a_g), `sec:model-graph`, `sec:model-content`, `app:first-config`. Boundaries and formats are in [architecture §11](/documents/en/architecture.md#11-historical-package-design-э2); they are not redefined here.

```text
Э2-01 → Э2-02 → ┬→ Э2-03 ─┐
                ├→ Э2-04 ─┼→ Э2-06 → Э2-07 → Э2-08
                └→ Э2-05 ─┘
```

An arrow denotes an accepted task. Before implementation, verify that the test fails on the specific rule, not merely on a missing import. Mandatory specification examples are preserved.

| Task | Dependencies and owner | Deliverable and acceptance condition |
|---|---|---|
| **Э2-01. Contracts and references** | Updated Э1 contracts (§3); lead | Pin the A_G input, configuration, metadata, package, `historical_package_v1` version, component schemas, and SKU order. Create a synthetic dataset with independent calculations mapped to the matrix below. Check incompatible versions, dimensions, hashes, and import boundaries. All three branches accept shared contracts before starting. [Implementation specification §§2.3](/documents/en/implementation_spec.md#23-main-core-interfaces), [3.3](/documents/en/implementation_spec.md#33-run-directory); part of B13. |
| **Э2-02. Input admission and catalog** | Э2-01; lead | Adapter for the complete A_G log, validating selected sessions, calendar admission, late tails, provenance, and hashes; catalog and row order. Prohibit other periods and dataset substitution; retain a read log. Acceptance covers independence from later data, unknown SKUs, and no catalog expansion from metadata. [Implementation specification §6.1](/documents/en/implementation_spec.md#61-catalog-graph-and-statistics), parts of B10/B13. |
| **Э2-03. Graph** | Э2-02 | Three relations, session contributions, support, sparsification, normalization, and regularization; chunked intermediate aggregation. B04 is checked against direct enumeration on a small dataset; verify numerical boundaries and failure without 0/0, intermediate-data volume, and absence of a dense catalog matrix. |
| **Э2-04. Historical statistics** | Э2-02 | Separate session counts for detail and recorded purchases, smoothed view popularity, and masks. Input includes all retained A_G sessions without target filtering. Purchase probability is not computed. Final tables are checked against a manual reference and the B09 portion concerning disabled frequency computation; [implementation specification §6.1](/documents/en/implementation_spec.md#61-catalog-graph-and-statistics). |
| **Э2-05. Metadata and modalities** | Э2-02 | Verified metadata reading under the profile, separate vectors and masks, reasons for unusable vectors, the A_G dictionary, and price groups. Separately read metadata for an observed SKU outside I_G from the same pinned delivery, without expanding the catalog, historical counters, or A_G dictionary. Reuse shared Э0 normalization. Acceptance covers the corresponding B05 portion, ID order, and no influence of later periods on the dictionary; [implementation specification §6.2](/documents/en/implementation_spec.md#62-representations-and-numerical-stability). |
| **Э2-06. Package build and loading** | Э2-03–Э2-05; lead | `historical build`, publication of consistent components, a loader without a builder import, checkpoints, and a resource report. Before building, validate Э2-02 inputs, code, settings, and Э2 budgets; do not call `preflight` or introduce a new manifest stage. Check resumption after fully published components; rejection requiring a new directory when a component's publication is incomplete; corruption; catalog/configuration/code incompatibility; and absence of partial success. [Implementation specification §15.4][checks-integration]; parts of B13/B10, launch order in §5. |
| **Э2-07. Code acceptance and measurements** | Э2-06; lead and independent reviewer | Э0–Э2 tests, types, style, build, and clean installation; B04 and applicable portions of B05/B09/B10/B13, with independent package reconciliation. Synthetic examples cover a large simultaneous group; the original 10 000 sessions check the entire Э2 path. Budgets are pinned before tests. For each of the three relations, measure edge counts after support filtering and sparsification, and the fraction of catalog SKUs with outgoing links. Retain time, memory, temporary and final disk usage, file count, and replay/recovery results. This checks implementation, not sufficiency for quality evaluation or scalability to all of Coveo. |
| **Э2-08. Expanding A_G before training** | Э2-07; lead | The 50 000-session size and budgets are pinned. Preserve computed sessions, periods, seed, admission, and selection without favoring purchasers. Move records into shared tables under [architecture §10](/documents/en/architecture.md#10-current-implementation), checking equality, row completeness, time boundaries, and targets; preserve source CSV files and previous results. Build the package and compare coverage with 10 000 sessions at unchanged graph thresholds. Acceptance: applicable Э1-09 and Э2-07 checks, resource measurements, replay, and recovery. Pin the package before Э1-10 and B features; no new manifest stage is introduced. |

#### File ownership

After Э2-02, the graph (`historical/graph.py`), statistics (`historical/statistics.py`), and modalities (`historical/content.py`, `data/metadata.py`) are independent. The lead owns contracts, catalog, build/loading, and integration. Shared ID ordering, normalization, and storage are reused. Ownership rules are in §6.

#### Test matrix

Shared check rules and parameters are in §1 and [implementation specification §15.1][checks-common].

| Rule / acceptance | Reference and boundary cases | File |
|---|---|---|
| A_G-only admission; independence from later periods — B10/B13 | Manually selected registry: an inadmissible session, late A_G tail, and another period; replacing A_N/B/C_smoke/C_select/E/D one at a time does not change mathematical tables. Changing exact source bytes changes provenance and does not remove the need to validate new hashes. | `/tests/integration/test_e2_historical.py` |
| Catalog and unknown SKU — §6.1 | SKU set from all events of eligible sessions: an item with only add/remove/purchase, an item only in metadata, an unknown item in a later prefix, and empty input. The dictionary and edges are not expanded. | `/tests/unit/test_e2_catalog.py` |
| Direction, groups, time — B04 | Direct pair enumeration for all three relations; self-loop, one group, distances 1/10/11; delays 0/1/1 800 000/1 800 001 ms; event before/at the snapshot boundary. Non-item groups do not change distance; an item group without detail is counted. | `/tests/unit/test_e2_graph.py` |
| Maximum contribution and support — B04 | Manual list of admissible delays, minimum per session; a repeated pair within one session and the same pair across sessions. Support 2/3/4, checked before neighbor truncation. | `/tests/unit/test_e2_graph.py` |
| top-100 order, normalization, and regularization — B04 | Manual row with 99/100/101 edges, equal weights with different UTF-8 IDs, separate reference for the sum of retained weights; empty row. Session/row permutations and chunk boundaries do not change the result. | `/tests/unit/test_e2_graph.py`, `/tests/properties/test_e2_graph_properties.py` |
| Small decay values — B04/B05 | Decimal reference for exp(−800): mathematical support is positive; expect the declared early numerical failure, not a falsely empty edge or 0/0. Initial τ_G is unchanged. | `/tests/unit/test_e2_graph.py` |
| Popularity and recorded purchases — §6.1 | Sets of session–SKU pairs and manual popularity smoothing; 0/1/repeated detail and purchase events. A purchase without detail and outside the target window contributes to the A_G recorded-purchase counter. Sessions without purchases belong to history. Permuting and repeating an event within a session does not change counters. | `/tests/unit/test_e2_statistics.py`, `/tests/properties/test_e2_statistics_properties.py` |
| Disabled purchase frequency — B09 | Statistics contain no purchase-probability estimate based on unknown outcomes; Э3 has no corresponding feature-schema column. Manifest checks are not repeated. | `/tests/unit/test_e2_statistics.py` |
| Separate spaces, masks, and norm — B05 | Small vectors with known directions and an independent norm calculation: absence, type/dimension/NaN/Inf, norm below/at/above ε, and very large finite coordinates. One unusable modality does not disable another; internal nonfiniteness is an error. Equal dimensions do not establish query/text/image compatibility. | `/tests/unit/test_e2_content.py`, `/tests/properties/test_e2_content_properties.py` |
| Metadata for an observed SKU outside I_G — §6.2, B05/B10 | A usable vector is read from the pinned delivery; the candidate catalog, historical counters, and A_G category dictionary are unchanged. An absent or unusable vector retains its corresponding mask and reason. | `/tests/unit/test_e2_content.py` |
| Dictionaries and price — §6, B05/B10 | Manual UTF-8 order of known A_G categories, code 0 for new/missing categories; input permutations. A price group remains an ordinal value with a separate mask, not a category code. | `/tests/unit/test_e2_content.py` |
| Sparsity, cost, and consistency — B04/B13 | Small direct reference and datasets at the Э2-07 limits: retained edge count, shapes, SKU order, reads/writes, and volumes. Component corruption, incompatible version, and interruption. Intermediate pairs and the entire largest group are accounted for separately; budget overrun causes failure, not truncation. Wall-clock time alone is insufficient. | `/tests/integration/test_e2_historical.py` |

B05/B10 are completed in later phases under [implementation specification §15.2][checks-base].

### 4.2. Э3: candidate retrieval and features

**Input:** the accepted Э2-08 package built from 50 000 A_G sessions, contracts, and prepared Э1 requests. `StateReducer`, numerical functions, the historical-package loader, and shared-table formats are reused. Implemented Э3 modules are described in [architecture §12](/documents/en/architecture.md#12-retrieval-and-features-э3); acceptance status is in §7.

**Phase boundary:** pure computation of normal candidates, separate fallback, and features; batch invocation of the same core. No training, ranking, quality metrics, HTTP, delivery deadlines, ANN, or new sources. Formulas, quotas, and feature-family contents follow [implementation specification §7](/documents/en/implementation_spec.md#7-candidates-and-features), `sec:model-candidates`, `sec:model-features`, and `app:first-config`; they are not repeated here.

The code and command are checked on small test datasets. Actual B is not opened: computing its features requires completed Э4 code and Э1-10 admission. This phase excludes B preparation and expansion of the actual sample. The limited check of the completed A_G package is described in Э3-06 and does not replace evaluation on a later period.

```text
Э3-01 → Э3-02 → ┬→ Э3-03 ─┐
                └→ Э3-04 ─┴→ Э3-05 → Э3-06
```

#### Tasks and acceptance

| Task | Dependencies and owner | Deliverable and acceptance condition |
|---|---|---|
| **Э3-01. Contracts, feature schema, and references** | Э2-08; lead | Refine existing `RequestContext`, `SourceResult`, `CandidateResult`, `FeatureBatch`, and their serialization without parallel formats. Distinguish quota membership, score measurability, rank availability, unavailability reason, and computational evidence. Freeze the concrete `feature_schema` columns under [implementation specification §7.2](/documents/en/implementation_spec.md#72-feature-schema): order, types, masks, formulas, transformations, and provenance; explicitly link the overall order to numeric and categorical `FeatureBatch` components. Existing `/schemas/feature.schema.json` validates the format but does not replace the concrete schema. Before implementation, prepare independent small references under the matrix below. Acceptance: contract and schema checks, no purchase frequency under B09, no incompatible versions or prohibited imports; [implementation specification §§2.3](/documents/en/implementation_spec.md#23-main-core-interfaces), [3.2](/documents/en/implementation_spec.md#32-logical-structures). |
| **Э3-02. Request context and shared scores** | Э3-01; lead | Bind `PreparedRequest` to the pinned catalog through `RequestContext`; verify time cutoff, snapshot, versions, and input hashes. In `retrieval`, implement shared pure functions for history, graph scores, modalities, and search for later use by retrieval and `FeatureEngine`. Complete-prefix counters and the entire last group are not replaced by bounded retrieval history. Metadata for an observed SKU outside I_G are read from the pinned delivery without expanding the catalog or dictionary; the complete CSV is not reread for every request. The historical-package producer's code hash need not match the new Э3 code hash: compatibility is checked through inputs and versions. Acceptance: numerical portion of B05, applicable B01/B02/B10, [implementation specification §§6.2](/documents/en/implementation_spec.md#62-representations-and-numerical-stability), [7.1](/documents/en/implementation_spec.md#71-exact-retrieval). |
| **Э3-03. Sources, union, and fallback** | Э3-02 | Implement exact source lists, eligibility filtering, tie-breaking, provenance union, and separate popularity backfill. Measurable scores are available even for a candidate outside a source's quota. Fallback is computed separately, does not extend the normal set beyond budget, and preserves its own order. Computational evidence enables future explanations to reference the source, item, or link used; public explanations remain in Э6. Acceptance: candidate portions of B07/B08, independent exhaustive enumeration for exact Top_m, and no target injection; [implementation specification §§7.1](/documents/en/implementation_spec.md#71-exact-retrieval), [10.7](/documents/en/implementation_spec.md#107-template-based-explanations), `eq:model-served-candidates`. |
| **Э3-04. Shared FeatureEngine** | Э3-02; parallel to Э3-03 | Implement every column of the accepted schema using shared score computation, without duplicated formulas or target access. Last-group features use the entire group, multiplicities, and masks; category transformations come only from the A_G package. Output validation checks finiteness, SKU/column order, and schema hash; incompatible input causes failure. Acceptance: numerical portion of B05, applicable B01/B02/B09/B10, and reference values for every family under [implementation specification §7.2](/documents/en/implementation_spec.md#72-feature-schema). Before branch integration, use agreed source results from Э3-01. |
| **Э3-05. Batch path and command** | Э3-03, Э3-04; lead | Add a minimal `features build --period B` calling the same functions sequentially and in chunks of complete requests. Store shared candidate and feature tables with pinned inputs and schema under [implementation specification §3.3](/documents/en/implementation_spec.md#33-run-directory), without per-request files. Targets and training weights are excluded from these tables and are not read by the core. On test datasets, verify computation-mode equivalence under §7.2, incompatible package/schema errors, interrupted writes, and retries under existing storage rules. Actual B without Э1-10 is rejected before its tables are read; closed periods are not opened. Acceptance: [implementation specification §§14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior), [15.4][checks-integration], applicable B10/B13. |
| **Э3-06. Code acceptance and limited actual-data check** | Э3-05; lead and independent reviewer | Accept checks from the matrix below, affected Э0–Э2 tests, types, style, build, and ordinary package installation. On the pinned package from 50 000 sessions, check a small prespecified set of test contexts with times after the snapshot: loading, candidate eligibility, feature finiteness, reproducibility, time, and memory. Set size and resource limits before the run. Do not use prefixes from the same A_G as independent later evaluation, read targets, or compute Hit@20. Retain only necessary parameters and measurements; update the brief README status. Complete B05, B07/B08/B10, and B14 remain with their corresponding later phases. |

#### Boundaries for parallel development

After Э3-02, retrieval (`retrieval/sources.py`, `candidates.py`, `fallback.py`) and features (`features/*`) are independent; the lead owns shared scores and contracts. The dependency `retrieval → features` is prohibited. References can be prepared after Э3-01 without using branch implementations. Integration belongs to the lead; ownership rules are in §6.

#### Э3 check matrix

Mandatory examples from the mathematical specification are preserved; the shared procedure is in §1.

| Rule / acceptance | Reference and boundaries |
|---|---|
| Available prefix and complete group — B01 | Snapshot before, at, and after request time; different event and availability times. Permutation of simultaneous rows; set/multiset; history shorter than, equal to, and longer than the limit; last group larger than the history limit. Changing the future or targets does not change candidates or features. Complete counters survive retrieval-history truncation. |
| Latest search — B02 | Age below, at, and above the threshold; missing or unusable latest block without reverting to its predecessor. Replacing search results and clicks does not change the complete Э3 path's result. |
| Scores, modalities, and aggregates — B05 | Direct calculation on small vectors/graphs; empty and old history, zero row, absent vector, zero cosine with mask 1, cancellation of opposing vectors, NaN/Inf. Complete last group with unequal multiplicities and partial masks; zero denominator; maxima are independent of multiplicity. |
| Eligibility and exact retrieval — B07 | Empty and small catalogs, all items purchased, a viewed or added item remains eligible; a SKU outside the catalog does not become a candidate. Quotas 0/1/boundary, equal scores, and UTF-8 order; blockwise Top_m matches exhaustive enumeration at different chunk sizes. Backfill is not limited to the original popularity quota. |
| Provenance and fallback — B08 | One SKU from several sources; its score outside a quota is measurable, but rank is absent; the first position is encoded by the normative formula. An unavailable source does not create arbitrary candidates. Removing one provenance without backfill preserves others; fallback and the normal set are not merged. Delivery and denominator checks belong to Э5–Э6. |
| Computational evidence — §10.7 | Repeated views of one SKU have their graph contributions summed before selecting evidence; ties are resolved by ID. Relation and support match the selected link. Search uses only available context; fallback does not inherit primary-branch provenance. Public templates and complete EXPL01 remain in Э6. |
| Schema and historical data — B09/B10 | Purchase frequency and numeric ID are absent; the recorded-purchase counter is not called a probability. An unknown category is encoded by the A_G dictionary; price remains ordinal. Metadata for an observed SKU outside I_G may affect context but do not change the catalog, edges, or dictionary. Replacing the target, E, or D does not change computation; unauthorized read attempts are blocked. |
| Shared computation — §7.2, applicable B01/B05/B10 | One request set under sequential and batch invocation: candidates, order, masks, and categories match exactly; numeric features follow the separate §7.2 tolerance. Version, SKU order, schema, or hash mismatches are not masked with zeros. Complete service check B14 remains in Э6. |

**Э3 completion:** Э3-01–Э3-06 are accepted; schema and interfaces are ready for Э4 and service branch Э6. This is code acceptance, not `preflight=passed` or demonstrated quality. Accepted Э4 code and Э1-10 are mandatory before actual B computation.

### 4.3. Э4: ranker training and application

**Deliverable:** a training table built from prepared B candidates and features, custom `session_lambdas_v1` derivatives, LightGBM training, a stored model, and its application. Basis: [implementation specification §§8.1–8.4](/documents/en/implementation_spec.md#8-ranking), `sec:model-ranking`, `sec:model-ranker-implementation`; admission order: [Э1-10](/documents/en/development_plan.md#э1-10-admission-of-the-first-actual-run).

Э4-01–Э4-07 are accepted code and checks on test datasets. Э4-08 was completed on Coveo after B preparation and Э1-10 admission; results are in the [reproduction guide](/documents/en/reproduction.md#measured-costs). A_G and the historical package are not rebuilt. Targets, periods, quotas, and the Э3 schema are not changed to make training succeed. Hit@20 evaluation, C_select tuning, the no-metadata variant, calibration, HTTP, and a production service package are outside this block.

**Status:** Э4-01–Э4-07 are accepted as code. Group assembly, derivatives, the LightGBM adapter, loading, application, and the training command are described in [architecture §13](/documents/en/architecture.md#13-training-and-inference-э4). Only the `SupervisionRecord` version changed, to 3; the feature schema and other Э3 contracts are preserved.

Code sequence: Э4-01 → Э4-02 and Э4-03 in parallel → Э4-04 → Э4-05 → Э4-06 → Э4-07. Independent references are prepared alongside contracts; tests precede the implementation being checked. Then: prepared B inputs and Э1-10 → Э4-08. Admission does not require an already trained model.

#### Э4-01. Table, derivative, and model contracts

- **Dependencies:** Э3-01–Э3-06.
- **Deliverable:** consistent interfaces for group assembly, derivative computation, training, and application. Pin row/column order, the relationship of `request_id` and SKU to labels, group sizes, categories, and schema hash. Use existing `SupervisionRecord`, clarifying `observed_purchases_v1` constraints: a single request, unit session weight, and labels in candidate order. Do not add target or weight fields to `FeatureBatch` or the application contract. Define the minimum stored-model contents and its dependencies; increase versions only for changed records.
- **Acceptance:** [implementation specification §§2.3](/documents/en/implementation_spec.md#23-main-core-interfaces), [3.2](/documents/en/implementation_spec.md#32-logical-structures), [7.2](/documents/en/implementation_spec.md#72-feature-schema); small examples for the matrix below. Unknown versions, incompatible schemas, and order violations are detected before training or prediction.

#### Э4-02. Training-group assembly

- **Dependencies:** Э4-01.
- **Deliverable:** chunked joining of prepared primary candidates and Э3 features with separate B targets on verified keys, without repeating candidate retrieval. Include only first requests with a nonempty recorded target. A missing target record is an error; an empty target with status `unknown` is not a negative training group. Do not add purchases outside the primary set or use fallback for training. Preserve group order, labels, sizes, counts of positive/other candidates, and pairs. Sessions with a target but no candidates or pairs enter the summary without being removed from the future evaluation population.
- **Resources:** before materialization, estimate the row-count upper bound under [implementation specification §8.1](/documents/en/implementation_spec.md#81-training-table), matrix, label, and group sizes, and Dataset memory. Set time, memory, and disk limits; terminate a run that exceeds them. Chunks do not split groups or become sequential incremental training. Do not introduce negative sampling or change weights to fit memory.
- **Acceptance:** applicable B06/B07/B09/B10 and §8.1; exact independent group construction on a small example. No pairs in the entire table means `insufficient`, without publishing a ready model. A nonempty group without pairs is retained with zero derivatives.

#### Э4-03. Independent computation and `session_lambdas_v1`

- **Dependencies:** Э4-01; computation needs neither LightGBM nor actual B.
- **Deliverable:** a pure derivative function and a thin adapter for LightGBM's callable objective. Recompute the full within-group order and weights at every iteration under [implementation specification §8.2](/documents/en/implementation_spec.md#82-custom-derivatives); do not replace the function with built-in `lambdarank`. Take ID and group order from the agreed input. Sum in FP64 and return contiguous FP32 arrays. Reuse stable primitives; do not replace zero Hessians with a positive constant.
- **Acceptance:** the derivatives portion of B05; an independent direct pair sum and central differences with ranks and weights fixed. All mandatory cases and tolerances are in the matrix below and [implementation specification §15.1][checks-common]. The reference does not call the function under test or use built-in LambdaRank.

#### Э4-04. Training the pinned LightGBM version

- **Dependencies:** Э4-02, Э4-03.
- **Deliverable:** a `lightgbm.train` adapter under [implementation specification §8.3](/documents/en/implementation_spec.md#83-lightgbm-settings-and-model-persistence). Bind the recipe's string identifier to the accepted callable function. Pass group sizes and the name of the sole categorical column; do not provide external weights or initial scores. Pin the complete actual recipe, including version 4.6.0 defaults, the derivative-function hash, and environment details. Do not tune budget, parameters, or stopping on closed periods.
- **Training results:** actual tree count, leaf sizes from routing training rows, group and pair counts, reasons tree construction stopped, time, memory, and model size. Do not present the iteration budget or `min_data_in_leaf` as a guaranteed tree count or strict minimum leaf row count. Failure to split does not authorize weakening the recipe.
- **Acceptance:** complete B05 for the active baseline: independent derivatives, a one-step numerical example checking the leaf and gain scale, and a grouped categorical split. Relaxed constraints are allowed only in an explicitly identified test example, not the baseline recipe.

#### Э4-05. Saving, loading, and application

- **Dependencies:** Э4-04; the loader and application can be prepared after Э4-01 using a test model.
- **Deliverable:** native LightGBM model format and one summary with necessary parameters, hashes, and results under [implementation specification §§3.3](/documents/en/implementation_spec.md#33-run-directory), [8.3](/documents/en/implementation_spec.md#83-lightgbm-settings-and-model-persistence). Do not copy preparation artifacts, the historical package, or the training table into the model; do not introduce a custom model registry, `pickle`, per-tree snapshots, or per-group logs. Publish the completed set last; do not overwrite an incompatible result. After interruption, a complete training rerun with the same inputs is allowed; a new checkpoint system is unnecessary.
- **Application:** the loader validates contents, versions, hashes, and schema. A pure function receives a verified model and `FeatureBatch`, returning raw scores and primary ordering under §8.3. Use the actual tree count; do not multiply the result by `learning_rate` again. Do not rank fallback or add candidates; handle an empty primary set explicitly. Do not turn scores into probabilities. Application does not import training-table assembly, `targets`, or training code.
- **Acceptance:** matching scores and order before/after saving, unambiguous tie-breaking, small and empty sets; corrupted models, incompatible schema/order/version, and nonfinite values cause failure. Secure service-package publication remains in Э6.

#### Э4-06. Command and data-access boundary

- **Dependencies:** Э4-02–Э4-05.
- **Deliverable:** `train ranker` with explicit pinned inputs, recipe, output directory, and resource limits. On test datasets, connect Э3 → groups → training → saving → loading and application, without reparsing raw CSV files; the existing validator checks pinned input hashes. Distinguish insufficient-data and error statuses from success; report progress during long computations. A repeat uses only a verified compatible result or requires a new run.
- **Admission:** `DatasetAccess` separates prefixes and targets but does not itself establish Э1-10. `features build` and `train ranker` use the [shared admission check](/src/coveo_recsys/cli/admission.py) before reading actual B and before publication: the existing validator, bindings to code, source CSV files, schema, settings, and prepared dataset. A `passed` flag, a `scope` value, or successful tests do not replace admission. Positive and negative scenarios are checked on synthetic artifacts; actual B/C_smoke/C_select/E/D are not opened during Э4-01–Э4-07.
- **Acceptance:** [implementation specification §§4](/documents/en/implementation_spec.md#4-run-manifest-and-stage-control), [14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior), applicable B10/B13. Inputs, derivative function, and schema cannot change unnoticed between admission, training, and publication.

#### Э4-07. Code acceptance

- **Dependencies:** Э4-01–Э4-06.
- **Deliverable:** accepted tests from the matrix below, a small end-to-end run from an ordinary wheel installation, and resource details for that run. The run must contain informative pairs; a model file alone does not demonstrate that training occurred. Check repeated training with pinned inputs, recovery after interruption, and compatible reuse. Do not require bitwise equality across platforms; in the pinned environment, compare numeric results under the implementation specification, and order and identities exactly.
- **Acceptance:** after affected checks, run the shared suite without `coveo`/`load`, Ruff, formatting, mypy, build, and ordinary wheel installation. In the [registry](/configs/test_registry.json), close baseline B05 and the implemented portion of B10.ranker, leaving B10.metadata and later B07/B08/B14 portions open. Update the brief README and architecture status; do not create a separate audit or a new test manifest.
- **Boundary:** code readiness does not mean `preflight=passed`, `post_training=passed` for an actual run, or demonstrated Coveo quality. Stop after this block, before Э4-08.

#### Э4-08. First Coveo training — a separate run

**Dependencies:** accepted Э4-07, verified B, and successful [Э1-10](/documents/en/development_plan.md#э1-10-admission-of-the-first-actual-run). Size and resource limits are already pinned; this phase does not reselect them based on purchases.

1. **Compute features.** Run `features build --period B` with admission, expanded B, the unchanged A_G package, and the Э3 schema. For the current run, set `max_requests=50000` in features and training; update paths, hashes, and `run_id`, and check that `max_observed_skus` is sufficient. For prefix SKUs outside A_G, use the standard `metadata_path` without expanding the historical catalog. Output: three shared tables for candidates, fallback, and features, and a completed summary with hashes and resource usage. The request count must match preparation; targets are not passed to retrieval or features, and fallback is not merged with the primary set. Acceptance: [implementation specification §7](/documents/en/implementation_spec.md#7-candidates-and-features) and Э3 compatibility checks.
2. **Check training input.** At the start of `train ranker`, join prepared primary candidates and features with separate B targets. Check keys and order, counts of all sessions and sessions with targets, positive and other candidates, empty sets, rows, groups without pairs, and pair counts. A positive candidate is a SKU in the intersection of primary retrieval and target; purchase count does not replace this metric. Do not add purchases outside candidates or turn empty targets into negative outcomes. For the current run, set `max_rows=50000 × M` (15 000 000 at M=300); before allocating the complete matrix, check the `N_B⁺ × M` estimate, table and Dataset memory, and disk under [implementation specification §8.1](/documents/en/implementation_spec.md#81-training-table). A row limit does not require allocating memory for the whole limit. This validates the existing command, not a new stage with a copy of the training table.
3. **Train and save the model.** If pairs exist, execute the complete accepted recipe through `train ranker`, without tuning on closed periods or sequential incremental training on chunks. Store the native model and one summary: inputs, group order, schema, recipe, derivative function, actual trees, leaf sizes, time, memory, and disk. An interruption permits a complete repeat with the same inputs; a completed result is not overwritten. Basis: [implementation specification §§8.2–8.3](/documents/en/implementation_spec.md#82-custom-derivatives), [§14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior).
4. **Check the actual result.** Load the model with the standard loader and confirm matching scores and order after saving. On B requests preselected by ID, check finite scores, primary-output eligibility, and its length limit, without claiming quality evaluation. Check replay of the completed command without overwrite, input bindings, and resource compliance. Applicable B05/B10.ranker must refer to this computation; successful synthetic tests do not replace actual checks.

**Completion condition:** a compatible Coveo model, verified loading, and a summary, or explicit `insufficient` if no training pairs exist. Technical failure or budget exhaustion is not insufficient data; it stops computation. Do not weaken the recipe, select additional purchases, or substitute a model from a synthetic example to obtain a successful status.

After validation, update the brief README and plan status, retaining measurements in `runs/`. Stop there: C_smoke/C_select/E/D are not opened here, and Hit@20 is not computed. The quality report and completion of applicable `post_training` conditions belong to Э5 and [§5](/documents/en/development_plan.md#5-run-checkpoints).

**Parallel work:** checking the completed A_G and preparing schema/recipe descriptions are independent of B preparation. B reconciliation, final manifest completion, admission, features, and training are sequential. Independent model review starts after publication; one contributor edits shared plans and the manifest. The single-threaded training recipe is unchanged.

#### Parallel Э4 work

After Э4-01, the table (`ranking/table.py`), derivatives (`ranking/objective.py`), and references are independent. The lead owns shared contracts, admission, `train/load/predict`, and integration after Э4-02–Э4-03. References use neither production functions nor built-in LambdaRank; targets are not passed to Э3. Parallel development does not change the recipe or `num_threads=1`; ownership rules are in §6.

#### Э4 check matrix

Separate derivative and finite-difference tolerances follow [implementation specification §15.1][checks-common]; Э0 checks are not duplicated.

| Rule and basis | Required cases |
|---|---|
| Groups — §8.1, B06/B07/B10 | Multiple sessions, multiple target SKUs, a target outside the catalog or candidates; an empty unknown target and a missing target record are distinguished. Only the primary set; duplicates and mismatched keys are rejected. A session with a target and empty candidates creates no rows; all-positive or all-other-candidate groups yield zero derivatives. No pairs in the entire dataset means `insufficient`. |
| Order and resources — §§7.2, 8.1 | Permuting input after canonicalization and changing chunk size preserve groups, labels, and features. Row count equals the sum of group sizes; masks, categories, schema, and SKU order are retained. Budget overrun does not cause truncation or incremental training on chunks. |
| Ranks and weights — §8.2, B05 | Different group sizes, two positive labels, equal scores with UTF-8 tie-breaking; full order for lengths greater than 20. Both positions below the cutoff depth contribute zero; a pair crossing the boundary is retained. Weight is applied once; the group's gradient sum is zero. Adding a common constant to scores does not change derivatives. |
| Numerical stability — B05 | Normative logits and central differences from `sec:model-ranker-implementation` and B05; ranks and weights are fixed for differences. Direct FP64 reference, FP32 cast and contiguity, zero Hessians, and no pairs. NaN/Inf in input or computation causes failure, not replacement with zero. |
| Actual library — §8.3, B05 | One-step numeric split: leaf and twice the mathematical gain. At least five categories, one with G=H=0; ensure the test checks a grouped split, not only one-hot. `cat_smooth=1`, `cat_l2=0`, no external weights. Actual trees/leaves are retained; test relaxations do not change the baseline recipe. |
| Application — §§7.2, 8.3 | Scores before/after saving match within tolerance, order exactly; `raw_score`, actual tree count, no second `learning_rate` multiplier. Ties, empty/small sets, length limit, only eligible primary candidates. Incompatible schema/catalog/order and model corruption are rejected; fallback is not ranked. |
| Leakage and admission — B01/B02/B09/B10/B13 | Application has no access to targets or training; replacing future events and search lists does not change current computation. Changes to E/D do not change the table, model, or recipe. Purchase frequency and calibration do not appear. No Э1-10 admission means no reading of actual B. Changed inputs/code/schema prevent reuse of the previous result. |

**Readiness for Э1-10:** Э4-01–Э4-07 are accepted, executable derivatives and the full recipe are known, and the test-run model can be saved and applied. **Completion of actual Э4 training:** after Э4-08, a verified Coveo model or explicit insufficient-data result exists; quality is evaluated separately in Э5.

### 4.4. Э5: historical evaluation and the first report

**Deliverable:** reproducible GCS+Rk outputs and two simple controls on C_smoke, session metrics, loss diagnostics, and a report. Basis: [implementation specification §§9.1–9.6](/documents/en/implementation_spec.md#9-temporal-protocol-and-historical-evaluation), [§14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior), [§15.4][checks-integration]; `sec:eval-offline`, `sec:eval-uncertainty`, `app:diagnostic-output`.

**Status:** [§7](/documents/en/development_plan.md#7-current-status); prepared inputs and measurements are in the [reproduction guide](/documents/en/reproduction.md#measured-costs).

**Scope:** only `observed_purchases_v1` and local `ideal_deterministic`. Do not tune the recipe, quotas, or session population based on the result; do not open A_N/C_select/E/D. The service, `execution_trace`, public explanations, the variant without metadata, and the neural network belong to later stages. Do not require them for C_smoke. Do not describe scores as probabilities or historical Hit@20 as an increase in purchases.

Order: Э5-01 → Э5-02–Э5-04 in parallel → Э5-05–Э5-06 → Э5-07 → Э5-08 → Э5-09. The first seven tasks use small examples and prepared artifacts, without opening the real C_smoke.

#### Э5-01. Evaluation contracts and run bindings

- **Dependencies:** Э4-08 and existing Э1/Э3/Э4 contracts.
- **Deliverable:** minimal `SessionEvaluation`, `MetricResult`, and evaluation summary under [implementation specification §§2.3](/documents/en/implementation_spec.md#23-main-core-interfaces), [3.2](/documents/en/implementation_spec.md#32-logical-structures). Reuse `RequestResult`, `CandidateResult`, `PreparedRequest`, and separate `TargetRecord` objects. Separate the target-free recommendation result, session evaluation, and aggregates; define units, denominators, statuses, and reasons for `null`. Fix the common ID set for GCS+Rk, P, and R, breakdown rules, and source measurement rules.
- **Compatibility:** new Э5 code receives its own run and admission. The existing model, schema, historical package, and evidence retain their hashes and producer code. The loader checks the model's fixed identity instead of substituting the new consumer's hash. Changes confined to evaluation code do not require rebuilding B; changes to the computation path require an impact check under [architecture §7](/documents/en/architecture.md#7-artifact-versions-and-dependencies).
- **Acceptance:** exact preservation of IDs/times/statuses; reject unknown formats or a mismatched schema, catalog, or model. Tests confirm that a new consumer can use the unchanged model and that substituted dependencies are rejected. Change versions only for affected contracts.

#### Э5-02. Local replay and controls

- **Dependencies:** Э5-01.
- **Deliverable:** one path from prefix → existing retrieval → FeatureEngine → model or fallback → RequestResult. It processes the first request of every saved session, including sessions without a target; targets are joined only after the output is produced. Do not reimplement Э3/Э4 formulas. Support P, popularity from A_G, and R, recent views filled by popularity, with shared eligibility and tie-breaking rules. Basis: [implementation specification §§1.1](/documents/en/implementation_spec.md#11-deliverables), [7](/documents/en/implementation_spec.md#7-candidates-and-features), [9.3](/documents/en/implementation_spec.md#93-replay-mode), `app:ablation`.
- **Acceptance:** B01/B02 and applicable B07/B10/B11. Main retrieval, features, and ordering agree with direct core calls; check short catalogs, empty responses, fallback, nondelivery, and tied scores. Replacing future targets or search lists/clicks does not change the output. A complete response does not require service storage, delivery clocks, or Э6 explanations.
- **Resources:** load and validate shared dependencies once per run; read whole sessions in batches. For prefix SKUs outside A_G, use the existing `metadata_path` from the original release, without extending the historical catalog or dictionaries from C_smoke. Do not create per-session files or one matrix of all evaluation features. Save short source lists, provenance, and outputs in shared tables; bound the depth of additional diagnostics in advance. Retain immutability and final publication checks.

#### Э5-03. Session metrics, losses, and sources

- **Dependencies:** Э5-01; Э5-02 for integration.
- **Deliverable:** pure metric computation from saved responses and separately joined targets, followed by aggregation under [implementation specification §9.2](/documents/en/implementation_spec.md#92-metrics-and-diagnostic-tables). Include Hit@20 and paired differences against P/R, V/U/Hit and the three losses, conditional candidate coverage, Recall/NDCG, and required breakdowns. Distinguish a target SKU missing from the historical catalog from ineligibility in a specific request. Unknown targets are excluded only from quality calculations, not from data and cost counters.
- **Sources:** sizes and coverage before/after quota limits within the measured depth, overlaps, unique items, and `Delta_fixed`. Removing a source accounts for all provenance, does not fill vacant positions, and does not retrain the model. An unknown full list is not replaced with an empty one; an empty union and an unmeasured value have different reasons. Source diagnostics use already saved retrieval results and do not access targets within the recommendation path.
- **Acceptance:** B06–B09, applicable B10/B11, and an independent reference. The common denominator is preserved for empty outputs, failures, fallback, and targets outside the catalog; if no target-bearing sessions exist, use `null/insufficient`. Check the chain and decomposition for each session and after aggregation, correct breakdowns, and absence of double-counted hits.

#### Э5-04. Block intervals and sufficiency

- **Dependencies:** Э5-01; inputs are the session outcomes from Э5-03.
- **Deliverable:** the calendar grid, block sums, and intervals for one policy and a paired difference strictly under [implementation specification §9.4](/documents/en/implementation_spec.md#94-intervals-and-sufficiency), `sec:eval-time-block`. Use one computation for GCS+Rk and descriptive comparisons with P/R. Preserve empty cells, move the paired policies together, compute the ratio of sums, the primary block length and two prespecified sensitivity settings, and simulation error for the interval bounds.
- **Acceptance:** B12 and an independent small reference before the first report with intervals. Check time-grid boundaries, nearest-rank, a zero resample denominator, insufficient/degenerate blocks and outcomes, and reproducibility of the random stream. An insufficient interval does not erase the point estimate or receive a successful status.
- **Boundary:** C_smoke is a diagnostic pilot, not final D. Specify its grid and blocks before inspecting outcomes. Do not invent D half-width requirements or planning assumptions, or use them as a condition for opening the pilot; the D plan must be justified and frozen by the deadline in [implementation specification §18](/documents/en/implementation_spec.md#18-decisions-required-before-the-corresponding-stages).

#### Э5-05. Sensitivity to event multiplicity

- **Dependencies:** Э5-02–Э5-03.
- **Deliverable:** a diagnostic comparison of `set_v1` and `multiset_v1` on the same C_smoke IDs under [implementation specification §5.1](/documents/en/implementation_spec.md#51-import-and-canonicalization) and `sec:data`. Recompute prefixes and affected features/candidates from saved canonical blocks with multiplicities; report the fraction of affected sessions and changes in features and candidate coverage. First requests, target sets, and session-level edge support retain their meaning.
- **Comparison scope:** the historical package is fixed for the first diagnostic; this tests sensitivity of prefix computations, not training of a new policy. The diagnostic feature schema must explicitly match the semantics. Do not present the application of a set_v1 model to another schema as a usable multiset ranker. If a complete variant with changed A_G/B and training is needed, agree on the computation separately; do not overwrite the original baseline.
- **Acceptance:** B01/B05/B07: without multiplicities, the variants agree; with repeats, only the specified counters, history, and means change. An independent example checks unchanged targets and numbers of distinct supporting sessions. Do not automatically switch the baseline semantics based on C_smoke results.

#### Э5-06. Commands and a report from shared results

- **Dependencies:** Э5-02–Э5-05.
- **Deliverable:** working `replay --period C_smoke` and `report --run ...` under [implementation specification §§9.6](/documents/en/implementation_spec.md#96-report), [14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior). JSON and Markdown are built from the same session results; rerunning the report does not perform retrieval or training. Include the `app:diagnostic-output` blocks, model and data bindings, mode, protocol, exclusions, sufficiency, and reasons for unmeasured quantities. Do not present local time and memory measurements as service load measurements.
- **Admission:** reuse the validator and storage. Build the run manifest's required index of trained variants from the existing `ranker_v1`, preserving its provenance; do not create a new registry or manifest system. C_smoke preparation and `replay` validate the linked `post_training` before opening the period; evaluation with intervals also checks B12 through `operation=evaluate`. A permitted name in `DatasetAccess` alone is insufficient.
- **Acceptance:** B10/B13: fail without admission, with a mismatched model, changed inputs/plans, or an attempt to open A_N/C_select/E/D. One final file marks completion of the set after table validation. Interruption does not publish a completed result; a compatible rerun does not overwrite it. Resource exhaustion and input errors return failure, not `insufficient`; progress reports the stage and request count without a per-session log.

#### Э5-07. Code acceptance before real C_smoke

- **Dependencies:** Э5-01–Э5-06.
- **Deliverable:** the small complete workflow from [implementation specification §15.4][checks-integration], from source rows through training, replay, metrics, and the run manifest, checked against independent calculations. Reuse existing Э1–Э4 examples and checks; update only affected entries in the [registry](/configs/test_registry.json). Check commands from a normal wheel installation: the executable package is imported from the environment, while project sources are available only for provenance checking and do not replace installed code.
- **Acceptance:** the matrix below, then the common suite without `coveo`/`load`, Ruff, formatting, and mypy. Complete applicable B02/B05/B07–B11; B12 for intervals. B10.metadata, service B03/B14–B16, and EXPL01 retain their own deadlines and are not declared passed. Results refer to the final code; known Windows skips remain explicit.
- **Transition condition:** code and settings are fixed; loading the old model and all producer/consumer links are verified; there is no circular requirement to obtain C_smoke results before permission to read it. Passing tests alone does not replace actual `post_training`.

#### Э5-08. Admission and preparation of real C_smoke

- **Dependencies:** Э5-07 and prepared Э4-08 artifacts.
- **Before reading targets:** agree on the maximum number of whole C_smoke sessions, the small trial-run size, and preparation/replay/diagnostic budgets. Preserve the seven calendar boundaries, H/g, seed, and hash-based selection rule; do not choose the sample size based on purchases. Fix controls, breakdowns, list depths, the grid, and blocks. Do not automatically transfer B limits or its 50 000 size to evaluation. Report if fewer sessions are available; expansion after inspecting the result requires a separate decision.
- **Deliverable:** a new consumer run with standard `preflight`/`post_training`, bindings to the unchanged model, and applicable evidence. Preparation admission fixes the C_smoke plan without requiring the not-yet-created dataset. After admission, prepare only C_smoke using the existing process: all published rows of selected sessions, canonical events, the first request, and separate targets in shared tables. Before `replay`, fix the preparation-report hash and check the dataset's link to the plan. Do not change A_G/B or their run manifests; C_select/E/D remain closed. New code may read the original CSV files to extract C_smoke, but may not construct evaluation datasets from closed periods.
- **Acceptance:** applicable B01–B03.data/B06/B10.data and period-opening control. Check row completeness, temporal eligibility, no ID overlap with A_G/B, agreement with the source rows of preselected sessions, hashes, and compatibility of the new dataset. Telemetry completeness remains unknown; empty targets do not become misses. Do not repeat confirmed checks of the original release and A_G/B without a reason.

#### Э5-09. First real report and stage completion

- **Dependencies:** Э5-08; accepted B12 for a report with intervals.
- **Order:** first a small run preselected by ID to check time, memory, disk, and correctness; then the agreed C_smoke volume. The same prefixes serve GCS+Rk, P, and R without reparsing raw data. Run heavy commands sequentially; join a target only to a saved response. Perform Э5-05 sensitivity analysis without changing the baseline based on its results.
- **Acceptance:** primary measures agree with an independent recomputation from responses and targets; policies share a denominator, and exclusions and undefined measures are explained. Validate the integrity of completed data, model/package immutability, and agreement between JSON and the readable report. An integration test confirms a compatible rerun without overwriting; an actual rerun of the old run is not required. B07–B12 diagnostics are computed on the full C_smoke, and the algorithms are validated; a second independent computation of all diagnostics is unnecessary. Sufficiency is checked separately: zero or few observed targets do not permit adding purchasing sessions.
- **Deliverable:** concise measured results and limitations in README, and Э5 status in this plan; shared tables and detailed reports in `runs/`, with no data in Git or separate audit. State where coverage is lost and justify the next step without promising improvement. C_smoke does not provide confirmatory evidence of superiority; `S_offline=not_run` until D, and the business effect is `not_observed`.

##### Closing Э5-09 using completed results

Completed: checked the integrity and composition of all 50 000 sessions and independently verified primary metrics for 657 sessions with a target under [implementation specification §9.2](/documents/en/implementation_spec.md#92-metrics-and-diagnostic-tables), using the tolerances in [§15.1][checks-common]. Preparation, training, recommendation output, and full diagnostics were not repeated. The result is in [§7](/documents/en/development_plan.md#7-current-status); technical verification does not establish statistical admissibility.

#### Э5 check matrix

The common procedure and tolerances are in §1 and [implementation specification §15.1][checks-common].

| Rule | Required cases and basis |
|---|---|
| Output without leakage | Replacing future data/targets and search lists does not change the response; prefix search and cart state are used. Direct and batched paths agree; check UTF-8 ties, empty/small catalogs, fallback, and nondelivery. B01/B02/B07/B10/B11. |
| Denominator | Normative example: 3 saved sessions, 2 with a target, 1 hit → 1/2. Multiple purchases produce at most one Hit; `unknown`, N=0, a target outside the catalog, empty output, fallback, and failure. B06/B07. |
| Chain and breakdowns | Three reasons for losses, target eligibility in the request, a zero conditional denominator; the loss sum and aggregates agree with session values. Popularity groups come from A_G; cart/noncart purchases in one session are not summed as independent outcomes. B07. |
| Sources | Multiple provenance sources for one SKU; another target SKU found by another source; removal without filling; a measured empty list versus an unmeasured list, bounded depth, and empty union. B08. |
| Multiplicities | Agreement without repeats, normative set/multiset examples, unchanged target sets and session support. B01/B05/B07. |
| Intervals | Empty cells, unequal denominators, an incomplete final block, jointly moving the pair; boundaries of 99/100 outcomes and 19/20 blocks, a zero resample denominator, degeneracy, and mismatched assumptions. B12. |
| Admission and saving | Old producer/new consumer; substituted hash, schema, or mode, mismatched IDs, missing admission, a closed period, interruption, and a compatible rerun. B10/B13. |
| Complete workflow | Independent small calculation from raw rows to the report; identical tables when batch size changes, JSON/Markdown agreement, and report generation without retraining. [Implementation specification §15.4][checks-integration]. |

#### Parallel work on Э5

After Э5-01, recommendation output, metrics, and intervals are independent in separate `evaluation` files; references and binding checks are prepared separately. The lead owns shared contracts, CLI, admission, and integration. Real runs are sequential after the code is fixed; ownership rules are in §6.

**Э5 completion:** Э5-01–Э5-09 are accepted, with a verified C_smoke report or a correct result giving specific reasons for statistical insufficiency. A technical error leaves the stage incomplete. Stop after the report: preparation of Э7 and selection of a new study require a separate task; C_select/E/D do not open automatically.

### 4.5. Э6: service and testing

**Goal:** a demonstration API using the prepared model, with correct responses and measured latency. Basis: [implementation specification §10](/documents/en/implementation_spec.md#10-service-state-and-versions) and [architecture §6](/documents/en/architecture.md#6-data-flow-during-serving).

Э5 is complete. Package `runs/coveo-historical-ag50000-v1/` and model `runs/coveo-b50000-ranker-v2/` are preserved. The Э6-01–Э6-05 demonstration scope is accepted: the overall deadline is implemented, and four scenarios passed using the prepared package and model ([API example](/README.md#example-recommendations)). Requirements agreed on 8 October specify one overall timeout of 1000 ms, with no normative deadlines for individual operations. Micro-optimization has stopped.

#### Tasks and acceptance

| Task | Dependencies | Deliverable and acceptance |
|---|---|---|
| **Э6-01. Profile and contracts** | Э3–Э5 | One local process, `local_checksum_only`, memory, and a file log. Session isolation, receipts, atomic state transitions, and retries: [§§10.1–10.3](/documents/en/implementation_spec.md#101-adapters-and-api), applicable B03.service/B14. |
| **Э6-02. Package and shared computation** | Э6-01 | Validated loading of the prepared package and model; shared Э3–Э4 algorithms. Compatibility, publication eligibility before the request, catalog immutability, and agreement of computations: [§10.4](/documents/en/implementation_spec.md#104-publication-and-rollback), applicable B05/B10/B11/B14. |
| **Э6-03. Ingestion and state snapshot** | Э6-01 | Canonicalization and the existing reducer; checks of sender, limits, retries, conflicts, completeness, and both times. Incomplete state is not replaced by empty state: [§§10.2–10.3](/documents/en/implementation_spec.md#102-accepting-sealed_block_v1), B03.service/B14. |
| **Э6-04. API and execution** | Э6-02/Э6-03 | Ingestion, recommendations, and `health`. One first request, retry without recomputation, a bounded queue, fallback, an overall deadline, and mandatory recording. Late computation does not change the response; `L⊆C_served⊆I_q`: [§10.5](/documents/en/implementation_spec.md#105-response-deadline-and-branch-selection), applicable B07/B08/B14. |
| **Э6-05. Explanations and trace** | Э6-02/Э6-04 | Explanations based on actual supporting evidence, a public response without internal diagnostics, and a verifiable link to the log: [§§9.3](/documents/en/implementation_spec.md#93-replay-mode), [10.7](/documents/en/implementation_spec.md#107-template-based-explanations), EXPL01/B14. |
| **Э6-06–Э6-11. Production operation — deferred** | Separate task | Database, multiple processes, production signatures and rollback, extended monitoring, access and deletion management, load, and RTO/RPO. Basis: the production profile in the [mathematical specification](/documents/en/math/coveo_session_recommender_math.tex), `sec:operations-serving`; [implementation specification §11](/documents/en/implementation_spec.md#11-monitoring-storage-and-protection). These are outside the required portfolio scope. |

#### Completion of the demonstration scope

On 8 October, settings, the coordinator, HTTP, and the trace were aligned with the overall deadline in [§10.5](/documents/en/implementation_spec.md#105-response-deadline-and-branch-selection). Tests were refined before the code. The four scenarios in [§10.6](/documents/en/implementation_spec.md#106-demo-api-acceptance) passed with normal core allocation; the complete HTTP call includes recording, and responses matched the on-disk log. A launch example is in [README](/README.md#example-recommendations). Earlier failed results are preserved. Data preparation, training, and quality evaluation were not repeated.

#### Boundaries

The demonstration is available locally only. Data, the model, targets, formulas, and retrieval rules remain unchanged; the service does not read targets. Demonstration acceptance does not open other periods or assign `post_load`, production `S_serve`, or admission to a production pilot. Э6-06–Э6-11 and Э7 evaluation are separate tasks.

#### Parallel work on Э6

One owner changes shared contracts and settings. Independent changes may be split into state (`state/*`), execution (`serving/*`), and explanations (`explanations/*`), with their tests. Perform integration and the four scenarios after merging changes; a separate agent for every item is unnecessary.

### 4.6. Э7: selection and one-time evaluation on D

**Deliverable:** final historical evaluation of a fixed policy or pair on D, with the required variant without supplied metadata. Basis: [implementation specification §§9.1–9.6](/documents/en/implementation_spec.md#9-temporal-protocol-and-historical-evaluation), [§17.2](/documents/en/implementation_spec.md#172-separate-decisions); `sec:eval-primary-comparison`, `sec:metadata-sensitivity`, `app:run-stages`.

**Current status:** Э5, the demonstration service, and Э7-01–Э7-04 are accepted. `evaluate` was run on 60 000 D sessions for fixed GCS+Rk, the controls, and the variant without metadata. `post_D` is recorded: `S_offline=insufficient`, with dependence assumptions unconfirmed. After reviewing D, variant 370 was trained and compared with the baseline through `select` on 50 000 C_select sessions; the fixed rule retained the original model with 300 candidates. Results are in [README](/README.md). A_G, B, and the earlier models are preserved.

**First block — Э7-01–Э7-04:** the variant without metadata and its separate model on already prepared B. Do not open C_select/E/D/A_N. This is a sensitivity check, not a new winner; do not retrain the primary model. A missing D precision plan does not block this work. Production Э6, new studies, and expanded samples are not required.

#### Э7-01. Variant contract and schema

- **Dependencies:** accepted Э3–Э5.
- **Deliverable:** a separate `config_id=graph_recent_popularity_no_provided_metadata`, source composition, feature schema, recipe, and output paths under [implementation specification §9.5](/documents/en/implementation_spec.md#95-selection-and-the-variant-without-metadata). Derive the remaining features from existing formulas; fix their order and provenance before training. Remove excluded features and their masks from the schema instead of filling them with zeros. The shared catalog, sessions, requests, and targets remain unchanged.
- **Acceptance:** [§§3.2](/documents/en/implementation_spec.md#32-logical-structures), [7.2](/documents/en/implementation_spec.md#72-feature-schema), B10: reject unknown profiles, mismatched schemas, and mixing of baseline artifacts with variant artifacts. Limit the change to two explicit profiles in existing contracts; a pluggable model registry is unnecessary.

#### Э7-02. Candidates and features without metadata

- **Dependencies:** Э7-01.
- **Deliverable:** reuse the graph, repeats, popularity, eligibility filter, merging, and filling. Do not compute excluded sources: zeroing quotas after reading vectors does not meet the requirement. Features do not access supplied content or search context, including out-of-catalog history SKUs. Do not rebuild the existing A_G; continue to check its integrity.
- **Acceptance:** [§9.5](/documents/en/implementation_spec.md#95-selection-and-the-variant-without-metadata), B07/B08/B10.metadata: an independent small example of quotas, duplicates, filling, and ordering; a purchase excludes the item, and an unknown SKU does not expand the catalog. Replacing metadata and search contents while permitted events remain unchanged does not change candidates or features; a test prohibits access to excluded data in the computation path. Preserve independence from future events.

#### Э7-03. Variant training, application, and admission

- **Dependencies:** contract Э7-01; integration after Э7-02.
- **Deliverable:** support a schema without categorical columns in the matrix, training table, saving, loading, and application. Preserve the baseline recipe's derivative function, hyperparameters, and budget; change only category parameters required by the schema. Carry the variant identifier and bindings through existing commands, the run manifest, and admission checks. `sensitivity metadata` from [implementation specification §14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior) is a thin command over existing feature construction and training, without a second pipeline. Account for consumers of shared code in evaluation and serving; do not extend the service with this variant.
- **Acceptance:** [§§8.1–8.3](/documents/en/implementation_spec.md#8-ranking), applicable B05–B11, [§15.4][checks-integration]. The small path from candidates → features → its own model → saving/loading → output passes without an external metadata source. B10.metadata also checks scores and output order. The baseline schema and model remain readable; arbitrary recipes, mismatched features, and removal of period restrictions are rejected. Update only affected entries in the [registry](/configs/test_registry.json), without declaring all of B10 passed.

#### Э7-04. A separate model on prepared B

- **Dependencies:** Э7-02/Э7-03 and passing affected tests.
- **Before running:** fix separate settings and directories, original B with 50 000 sessions, the schema, recipe, and bindings. Agree on time, memory, and disk limits for features and training. Previous baseline command limits were 4 hours, 8 GiB of memory, and 4 GiB of output each; use these as a starting point for agreement, not as evidence that they suit the new run.
- **Order:** a small B subset preselected by ID without selecting on purchases → measure features, memory, and disk → the full agreed B → separate training. The small run estimates feature resources, not quality or training cost; no pairs in that subset is not a reason to add purchasing sessions. Before training, separately estimate table and Dataset memory from the full features, column count, and eligible groups; extra training on a small sample is unnecessary. If projected full-scale usage exceeds the limits, stop before the long-running command and report the reason. Run heavy commands sequentially.
- **Deliverable:** the variant's own B candidates/features, training table, model, and one standard summary. Do not copy or overwrite shared B tables, original targets, or the primary model. Account for every selected request; training uses only eligible groups under [§8.1](/documents/en/implementation_spec.md#81-training-table). Do not inherit the primary model's positive-candidate count: it depends on the new retrieval.
- **Acceptance:** standard admission to B and training under [§4](/documents/en/implementation_spec.md#4-run-manifest-and-stage-control), applicable B05–B11. Check group and pair composition, model loading/application, unchanged shared inputs, time, memory, and disk. With no training pairs, return `insufficient`, without a fictitious model or tuning new parameters. README contains the command, result contents, and limitations; successful training is not quality evaluation. Stop after the first block.

#### Э7-05. Selection and final evaluation code

- **Completed portion:** `evaluate` in `ideal_deterministic` mode for one fixed GCS+Rk, P/R controls, and sensitivity without metadata; `select` for comparing the baseline with variant 370. These use shared Э5 computations, each model's own features, and separate admissions with external `before_D`/`before_C_select`. Both commands have been run on Coveo; final evaluation of the selected pair on a new period is not implemented.
- **Dependencies:** Э7-03; real models are needed only for actual runs.
- **Deliverable:** route C_select and D through existing preparation, admission, replay, and reporting, without duplicating the computation pipeline. Implement the commands from [implementation specification §14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior) and separate single-policy/pair modes. Link plans, artifacts, and results; working code alone does not open a period.
- **Acceptance:** [§§9.1–9.6](/documents/en/implementation_spec.md#9-temporal-protocol-and-historical-evaluation), B10/B12/B13: on small examples, check refusal without admission, the common target population, integer Hit sums, intervals, immutable plans, and preservation of negative or insufficient results. D is not read even to check its size before the permitted stage; E remains closed.

#### Э7-06. Fixing the policy and selecting if needed

- **Dependencies:** Э7-04/Э7-05; a separate decision to undertake final evaluation.
- **Deliverable:** one fixed GCS+Rk is permissible for the first evaluation. The variant without metadata remains a required sensitivity analysis and does not automatically become a search participant or second confirmatory hypothesis. When studying an improvement, fix the variants, control, limits, and selection rule before C_select; after `pre_selection`, prepare the agreed number of whole sessions and perform selection. Do not tune parameters afterward.
- **Selection rule:** accept the new variant only if it has a larger hit sum on the common target-bearing sessions and passes the checks; retain the original model on a tie. Publish time and memory separately. The comparison with `gcs_rk_metadata370` is complete: the original model was selected. D, already reviewed, is not independent confirmation of that selection.
- **Acceptance:** §9.5 and applicable B10/B13: reproduce the decision from saved results; if no eligible variants exist, there is no winner. For the mode without tuning, explicitly record that selection will be skipped and check that the run manifest permits it, rather than bypassing the validator.

#### Э7-07. Precision plan and admission to D

- **Dependencies:** Э7-04–Э7-06.
- **Deliverable:** justify precision parameters for the selected outcome using permitted C_smoke; use the existing report without recomputing recommendations. Before D, fix the primary policy or pair, sensitivity without metadata, session volume and selection rule, resources, grid, blocks, and planning parameters from [§9.4](/documents/en/implementation_spec.md#94-intervals-and-sufficiency). Do not transfer B session counts or variance to D without justification.
- **Acceptance:** applicable B10–B12 on actual artifacts and `pre_D`. Do not replace unknown parameters with guesses or resolve missing information by inspecting D. If the earlier pilot cannot justify the selected outcome, first agree on additional C_smoke work. Local evaluation does not require `post_load`.

#### Э7-08. One-time D and result publication

- **Dependencies:** successful Э7-07 and separate permission for the actual run.
- **Deliverable:** prepare the fixed D sample and responses from unchanged models and controls, then join targets and produce the report. Compute the required sensitivity without metadata on the same population. Interruption does not permit changing the plan; compatible completion or reruns preserve inputs and earlier results without becoming new tuning.
- **Acceptance:** [§§9.2–9.6](/documents/en/implementation_spec.md#92-metrics-and-diagnostic-tables), B07–B12, and `post_D`: common denominators, unknown targets, intervals and reasons for insufficiency, and agreement between the report and tables. Independent small references and checks of final summaries, without fully recomputing all diagnostics. README contains quality, cost, and limitations; `S_offline=passed` does not imply superiority or business effect.

#### Parallel work on Э7

After Э7-01, retrieval/features (Э7-02) and training/loading (Э7-03) can be split. One lead changes shared `contracts`, `config`, schemas, CLI commands, and the registry. Independent references can be prepared in parallel against the fixed contract. Integration precedes real computation; feature generation, training, and subsequent evaluation do not compete for resources.

### 4.7. Further development

Optional development after publication. `post_D` and miss analysis are complete; expanding to 370 candidates increased coverage but not Hit@20. GCS+Rk with 300 candidates remains the baseline. Conduct new studies one at a time, with a question, sample, and budget fixed in advance. D has already been reviewed and cannot support a new confirmatory conclusion: an unused final period and a separate plan are required.

| Task | Dependencies | Deliverable | Basis and acceptance |
|---|---|---|---|
| Э8. Neural extension | Validated baseline and measured cost of the local Э0–Э5 path; registered question | Training on A_N, fixed encoders, its own B candidates/features and ranker; diagnostics and cost. | [Implementation specification §§12.1–12.2](/documents/en/implementation_spec.md#121-fixed-paths-and-neural-model), [§15.3][checks-extensions]; N01–N07 and affected B checks. If D has already been used, new confirmation requires a new final period. |
| Э9. Separate study | Baseline diagnostics and dependencies of the selected module; Э8 for SID | One preregistered variant with its own artifacts, control, and budget; reproducible comparison. | [Implementation specification §§12.3–12.7](/documents/en/implementation_spec.md#123-sampled-denominator); applicable X01–X05, B/N, and independent references. X02.support before training; B17 before serving with ANN. |
| Э10. Production pilot | Э6, a suitable Э7 final evaluation, and the operational profile under `Ready_pilot` | Production limits, trusted keys and publishers, owners/deadlines, verifiable deletion, calibrated monitoring, confirmed completeness of outcome observation, the protocol, and assignment accounting; after permission, a separate experiment. | [Implementation specification §§13](/documents/en/implementation_spec.md#13-ab-preparation-and-analysis), [17.2](/documents/en/implementation_spec.md#172-separate-decisions); the software portion of B18 and validation of the actual protocol before A/B, `pre_AB`, then `post_AB`. Local load acceptance does not replace production admission. |

Directions for separate experiments, without promising improvement:

- **Catalog and ranking:** test reducing losses before candidate retrieval or training on rare purchase examples. Compare coverage, Hit@20, and cost with the baseline on one fixed sample; simply expanding the list has already failed to help.
- **Learned representations:** use graphs and metadata for new candidates and features while retaining the graph path. Compare the contribution of representations to coverage and final Hit@20 with the same controls.
- **Graph freshness:** test aging and relation updates using only events available at request time. Compare with a frozen graph under the same retrieval and evaluation.
- **Action order:** compare history aggregation with a compact SASRec-inspired model if enough events exist at request time. The first request currently often has one view; first check whether a useful sequence exists.
- **Indirect relations, SID, and search acceleration:** select these only for a measured problem; compare quality and cost with the original method. These directions are not required for publication.

## 5. Run checkpoints

### First historical report

Under [implementation specification §14](/documents/en/implementation_spec.md#14-commands-and-pipeline-behavior), A_G construction is permitted based on Э2 checks before full `preflight`; admission to B features and training remains required:

`accepted A_G package + B preparation and validation → Э1-10: preflight → features build B → train ranker → post_training → replay C_smoke → report`.

The A_G package was already accepted in Э2-08; B preparation and training do not require rebuilding it. The sequence before obtaining the model follows [B preparation](/documents/en/development_plan.md#preparing-b-before-э1-10), [Э1-10](/documents/en/development_plan.md#э1-10-admission-of-the-first-actual-run), and [Э4-08](/documents/en/development_plan.md#э4-08-first-coveo-training--a-separate-run).

After Э4-08, the development order for evaluation, admission, and the first real C_smoke is specified in [Э5-01–Э5-09](/documents/en/development_plan.md#44-э5-historical-evaluation-and-the-first-report).

Before `post_training`, full B05, B07–B09/B11, and B10 portions for active computations and data separation are required. B12 is checked before the first report with intervals; if C_smoke includes intervals, the check precedes that report. The variant without metadata is prepared by D, and B18 by A/B.

C_smoke includes `multiset_v1` sensitivity analysis; results, loss reasons, and cost are saved. The permitted C_smoke pilot supports D precision planning after training. At this checkpoint, C_select/E/D remain closed and `S_offline=not_run`; the actual state of later runs is in §7.

### Selection and final evaluation

Order and dependencies: [Э7-05–Э7-08](/documents/en/development_plan.md#э7-05-selection-and-final-evaluation-code). Admission is checked with `manifest validate --operation evaluate` and fixed `--plan-reference NAME=SHA256` under [architecture §8](/documents/en/architecture.md#8-configuration-and-acceptance). `evaluate` was run on D; the later `select` was run on C_select for the separate 370 experiment. Local evaluation does not depend on production Э6/`post_load`.

## 6. Parallel work and decisions before stages

- After Э0-02, numerical primitives Э0-03 and storage Э0-04 are independent. Schemas and the validator use shared types; their changes are coordinated before integration.
- After canonicalization, the static envelope Э1-08 can be developed in parallel with prefixes and targets. These branches do not duplicate canonicalization or `StateReducer`.
- In Э2, after contracts and the catalog are accepted, the graph, statistics, and modalities can proceed in parallel. Shared files and subsequent assembly belong to the lead; boundaries are in [§4.1](/documents/en/development_plan.md#41-э2-historical-package).
- In Э3, sources and features are developed in parallel once contracts and shared scores are fixed; file owners and integration order are in [§4.2](/documents/en/development_plan.md#42-э3-candidate-retrieval-and-features). Actual B feature computation and training wait for completed Э4 code and Э1-10 admission.
- In Э4, after shared contracts, the training table, custom derivatives, and independent references can proceed in parallel; integration and file boundaries are in [§4.3](/documents/en/development_plan.md#43-э4-ranker-training-and-application).
- In Э5, after contracts, local recommendation output, metrics, and intervals can proceed in parallel; admission and real computations are sequential. File ownership is in [§4.4](/documents/en/development_plan.md#parallel-work-on-э5).
- In Э6, after shared contracts, the package and state can proceed in parallel, followed by the coordinator and explanations; file ownership is in [§4.5](/documents/en/development_plan.md#parallel-work-on-э6). The first local service is accepted separately from full load testing and the production profile. Э6 does not block independent Э7 work or open C_select/D.
- Independent references and compliance checks can be developed in parallel.

Each shared file has one owner. Contract changes are coordinated with consumers before integration and reflected in artifact dependencies under [architecture §7](/documents/en/architecture.md#7-artifact-versions-and-dependencies).

Remaining decisions and deadlines follow [implementation specification §18](/documents/en/implementation_spec.md#18-decisions-required-before-the-corresponding-stages) and [architecture §9](/documents/en/architecture.md#9-decisions-as-implementation-progresses). An unknown value blocks only work that needs it.

## 7. Current status

| Tasks | Status and limitations |
|---|---|
| Э0-01–Э0-07 | Implemented: environment, contracts, numerical functions, storage, schemas, and the run manifest 2.10 validator. Tests do not replace admission for a real run. |
| Э1-01–Э1-09 | Preparation migrated to `observed_purchases_v1`: one first request, calendar eligibility, observed targets, and unknown telemetry. Changed contracts are version 3; the purchasing cohort was removed. Code checks passed. |
| Preparation for Э2 | Accepted 50 000 A_G sessions in shared tables. Checked equality of saved records, eligibility, row completeness, first requests, and targets; the earlier 10 000-session dataset is preserved for comparison. The cost of subsequent package construction is in the [reproduction guide](/documents/en/reproduction.md#measured-costs). |
| Coveo release validation | The format and local hashes of the three CSV files are confirmed. Reading the full export does not establish telemetry completeness, which remains unknown. |
| B preparation | Expansion to 50 000 is complete: `runs/coveo-b50000-v1/`, with 625 sessions with an observed target. All source rows, eligibility, targets, nonoverlap with A_G, and exact equality of the earlier 10 000 across all four tables were independently checked. A rerun did not change the result; earlier datasets are preserved. Contents and resources are in the [reproduction guide](/documents/en/reproduction.md#measured-costs). |
| Э1-10 | Passed for the current code and prepared B: a new run manifest and evidence in `runs/coveo-b50000-training-v2/`, standard `preflight` with no issues. Preparation code and the earlier row check are preserved; unchanged tables and consumer compatibility are confirmed. Seven symbolic-link checks were skipped due to Windows permissions, not declared passed. Planned precision remains required by pre_D. |
| Э2-01–Э2-07 | Completed on 10 000 A_G sessions: catalog, graphs, statistics, metadata, package publication, and loading. Independent recomputation confirmed every session and part; interruption after graph construction, resumption, and rerunning the command were checked. |
| Э2-08 | Completed: 50 000 sessions were transferred without changing records; the historical package was accepted after independent recomputation, interruption, resumption, and rerunning. Coverage was compared with 10 000 sessions under unchanged thresholds. Purchase relations remain sparse; quality evaluation belongs to Э5. |
| Э3-01–Э3-05 | Implemented: the 93-column schema, context and shared `QueryScores`, exact retrieval, separate fallback, computation evidence, and `FeatureEngine`. Optimization was checked on 64 sessions with exact record agreement. Features for all 50 000 B requests were published; hashes and keys of the three tables were checked. Measurements are in the [reproduction guide](/documents/en/reproduction.md#measured-costs). |
| Э3-06 | Completed: independent references, agreement of batch and sequential computations, interruption, rerunning, code checks, and a normal wheel installation. The real package was checked on 16 synthetic contexts after the snapshot; measurements are in README. This is not quality evaluation or a real `preflight`. |
| Э4-01–Э4-06 | Implemented: training groups, `session_lambdas_v1`, LightGBM, the model and summary, loading and application, `train ranker`, and shared admission for actual B. If there are no pairs, the result is `insufficient`, with no completed model. |
| Э4-07 | Accepted as code: independent references, actual LightGBM splits, the complete workflow, rerunning, interruption, and failures. Common checks and a normal wheel installation passed; measurements are in README. |
| Э4-08 | Completed: a model trained on 625 target-bearing B sessions, 509 positive candidates, and 152 149 pairs; 200 trees under the fixed recipe. Saving, loading, output for eight preselected IDs, and rerunning without overwriting were checked. Contents, resources, and limitations are in the [reproduction guide](/documents/en/reproduction.md#measured-costs). |
| Э5-01–Э5-06 | Implemented local GCS+Rk/P/R outputs, metrics and decomposition, intervals, set/multiset sensitivity, commands, and admission to C_smoke. Responses and targets are joined only during evaluation; earlier producers retain their identities. |
| Э5-07 | Code accepted: independent references, the complete CSV-to-report workflow, interruption, rerunning, admission failures, and compatibility of the earlier model with a new consumer. After evaluation optimization, the common suite, Ruff, mypy, and 61 checks of a normal wheel installation passed. Seven symbolic-link checks were skipped due to Windows permissions; check commands are in [README](/README.md#installation-and-usage). |
| Э5-08 | Completed: volume and budgets agreed, `post_training/evaluate` passed, and 50 000 C_smoke sessions prepared and independently checked. A_G/B and the model are preserved. |
| Э5-09 | Completed: integrity, bindings, and composition of all 50 000 sessions checked; an independent computation of primary measures for 657 target-bearing sessions matched session evaluations and the report. The other 49 343 sessions have unknown targets and are excluded from the quality denominator. Accepted code checks were accounted for; the model and responses are preserved. Evaluation cost is in the [reproduction guide](/documents/en/reproduction.md#measured-costs). |
| Э6-01–Э6-05 | Demonstration scope accepted on 8 October: overall timeout, four scenarios, and verification of HTTP against the on-disk log. Profile: `local_checksum_only`; production acceptance checks have not passed. |
| Э6-06–Э6-11 | Deferred beyond the portfolio scope. A database, production publication, protection, and load acceptance require a separate task. |
| Э7-01–Э7-03 | Code and tests completed: the profile without metadata, a 54-column schema, separate training, loading, and admission; `sensitivity metadata` uses existing commands. The baseline path is preserved. |
| Э7-04 | Completed on 9 October: new admission, measurement on 100 requests, its own features for all 50 000 B requests, and a model without metadata. Resources, training composition, saving, and application checked. Quality on D was measured in Э7-08. |
| Э7-05 | Implemented and executed `evaluate` for the fixed baseline, P/R, and the variant without metadata, as well as `select` for the original model and variant 370 on C_select. Final evaluation of the selected pair on a new period is not implemented. |
| Э7-06–Э7-07 | On 9 October, GCS+Rk was fixed without tuning; the plan, up to 60 000 D sessions, and resource limits were agreed; `pre_D` passed. Settings and evidence: `runs/coveo-d-plan-v2/`. Dependence assumptions are unconfirmed: admission to computation does not establish interval validity. |
| Э7-08 | Computation, verification, and publication completed on 10 October. `post_D`: `S_offline=insufficient`; structure, consistency, and evidence completeness are confirmed, while statistical admission remains closed. Result: `runs/coveo-d-plan-v2/manifest-post_D.json`; checks and loss analysis: `runs/coveo-d-plan-v2/post_D/`. Data, models, and the plan are preserved. |
| Experiment 370 | Its own features and model were obtained on the existing B; comparison on 50 000 C_select sessions is complete. Coverage increased, Hit@20 did not, and computation is more expensive. The fixed rule retained GCS+Rk with 300 candidates. Report: `runs/candidate370-v1/comparison/reports/replay.json`; intervals are `insufficient`, and selection has not been confirmed on an independent period. |
| Э8–Э10 | Not implemented; separate tasks are required. |

B05 for the active baseline and the implemented portion of B10.ranker are included in Э4-07 acceptance; Э4-08 confirmed actual model parameters, saving, and application. Э5 code checks the complete local response, B07/B08, and B12 on small examples. Actual `post_training/evaluate` passed, but does not replace quality evaluation. B03.service and B14 were checked within the local scope; full production scope is deferred together with Э6-06–Э6-11. For B10.metadata, metadata independence on test examples, separate training on Coveo, and sensitivity evaluation on D were checked.

The Э7 outcome, experiment 370, and validation of the small reproduction path are complete; statistical acceptance criteria have not been met, and the business effect is `not_observed`. The primary model with 300 candidates is preserved. Studies in [§4.7](/documents/en/development_plan.md#47-further-development) are optional. A_N/E remain closed. Numbers and limitations are in [README](/README.md).

**Reproduction validation completed on 11 October 2026:** the path from three CSV files to the C_smoke report ran from a separate normal wheel installation, without earlier runs. Direct data checks, `preflight`/`post_training`, and the full pytest suite passed; agreed limits were respected. Known Windows symbolic-link skips retain the `skipped` status when excluded under [implementation specification §15.1][checks-common]. Commands and measurements are in the [reproduction guide](/documents/en/reproduction.md#full-computation-workflow). This validates the baseline path, not a rerun of the published D evaluation; earlier data and models are preserved.

[checks-common]: /documents/en/implementation_spec.md#151-general-procedure
[checks-base]: /documents/en/implementation_spec.md#152-baseline-registry
[checks-extensions]: /documents/en/implementation_spec.md#153-extensions
[checks-integration]: /documents/en/implementation_spec.md#154-end-to-end-tests-and-automation
[checks-explanations]: /documents/en/implementation_spec.md#155-explanation-checks
