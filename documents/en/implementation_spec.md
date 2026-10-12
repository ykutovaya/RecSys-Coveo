# Technical Specification: Coveo Session-Based Recommender System

English | [Русский](/documents/implementation_spec.md)

Specification version: 1.8. Revision: October 8, 2026.

Basis: [mathematical specification 2.10](/documents/en/math/coveo_session_recommender_math.tex) and [run manifest template 2.10](/documents/math/coveo_session_recommender_run_manifest.template.json).

This specification defines the required interfaces, deliverables, and acceptance criteria. Formulas and notation come from the mathematical specification; `sec:...`, `eq:...`, and `app:...` are its labels. See the [README](/README.md) for what is implemented. Use manifest format 2.10 with `evaluation_protocol="observed_purchases_v1"`; earlier formats are unsupported.

In a conflict, the mathematical specification's formulas and normative requirements take precedence, followed by the completed and validated run manifest. Record and resolve any conflict before the affected stage, without silently changing the algorithm. The manifest selects an allowed configuration but does not override requirements.

## 1. Purpose and implementation scope

Implement a reproducible system that retrieves up to 300 candidates from the available prefix of an anonymous session, ranks them, and returns up to 20 distinct eligible items. The protocol evaluates only the first recommendation list against all distinct SKUs of recorded future purchases within the same session's window. The primary metric is Hit@20 among sessions with a nonempty recorded target. Telemetry completeness is unknown; the metric does not measure purchase frequency across all sessions or conversion lift.

### 1.1. Deliverables

| Component | Deliverable |
|---|---|
| First local run | Data preparation, A_G graph and reference data, exact retrieval, features, GCS+Rk, replay, and a C_smoke report. |
| Final historical evaluation | Frozen selection, a variant without metadata, and a one-time evaluation on D with intervals and sufficiency statuses. The service need not be ready. |
| Demo service | One local process: an API over a trained model, input validation, state, retries, fallback, explanations, and logging. Acceptance: §10.6. Production operation is outside the mandatory deliverables. |
| Research and A/B | Only as a separate task after baseline evaluation; entry requirements: §§12–13. |

The first run uses one configuration, `gcs_rk_v2_7`, without parameter tuning. It excludes a neural model, SID, ANN, an updated graph, diversity, IPS/DR, and A/B testing. Periods A_N and E remain reserved and cannot be merged into other periods.

The variant without provided metadata must be ready by D for a sensitivity check, not to select a second winner. It is not a prerequisite for the first C_smoke run. Controls for the first evaluation are A_G popularity and recently viewed items padded with popular items. Diagnostic G, GC, and GCS are allowed on C_smoke; the full set of research comparisons is not required.

### 1.2. Out of scope

- Recovering missing order prices, margins, stable person IDs, impression probabilities, or historical vector versions.
- Training custom text and visual models on raw data absent from Coveo.
- A guaranteed positive business effect.
- Automatic retraining in response to alerts, continuous optimization on D, or searching configurations until a favorable result is found.
- A full distributed platform, cluster orchestrator, and storefront interface. Their integrations remain outside the computational core.

## 2. Architecture and general requirements

### 2.1. Code boundaries

Module boundaries and dependency directions are defined in the [architecture](/documents/en/architecture.md#3-module-boundaries-and-dependencies). Training and serving share feature computation; the computational core does not depend on HTTP, state storage, or report format. Modules for later stages need not be created in advance.

### 2.2. Environment and dependencies

- Core mathematics and metrics: NumPy/FP64; ranker: `lightgbm.train`, LightGBM 4.6.0.
- Neural profile: NumPy 2.1.3, PyTorch 2.6.0, CPU/FP32, one computation thread and one inter-operation thread, deterministic algorithms.
- Pin Python, tabular-data, statistics, JSON Schema, HTTP, and test libraries, the compiler, and numerical libraries at Э0. Check compatibility with a minimal example; save exact versions and hashes in the dependency file. Floating dependencies are prohibited.
- Install optional neural dependencies as a separate set. The baseline run must not import PyTorch or require a GPU.
- Record the OS, CPU, processes, threads, and memory measurement tool for the load-test node separately from the developer environment.
- Reference tests run without Coveo source data. Coveo tests and load tests are marked separately; missing data do not count as a successful skip.

### 2.3. Main core interfaces

```text
canonicalize(raw_events, event_semantics) -> CanonicalSession
make_schedule(session, task_config) -> RequestContext[]
make_targets(session, schedule, period) -> TargetRecord[]
build_historical(admitted_A_G, metadata, config) -> HistoricalPackage
retrieve(context, historical_package, source_runtime) -> CandidateResult
build_features(context, candidates, historical_package) -> FeatureBatch
fit_ranker(training_groups, ranker_config) -> RankerArtifact
recommend(context, package, execution_context) -> RequestResult
explain(trace, recommendation_items, templates) -> Explanation[]
reduce_state(state, accepted_block, config) -> StateTransition
evaluate(request_results, registered_targets, frozen_plan) -> EvaluationResult
validate_manifest(manifest, stage, artifacts) -> ValidationResult
```

`RequestContext` contains the available prefix, times, eligible catalog, and versions, but no future targets or positive-cohort membership. Training weights and labels are supplied separately, only during batch processing. `execution_context` contains the deadline, selected branch, source states, and mandatory logging state, but no future purchases.

Source-list membership and score availability are separate fields. Distinguish reasons for an empty source, missing modality, disabled module, and technical failure.

## 3. Data and persisted artifacts

### 3.1. Input files

Support `browsing_train.csv`, `search_train.csv`, and `sku_to_content.csv`. Define the required column names and meanings in the input schema from the actual files. Confirm time units, vector dimensions, and duplicate semantics; do not guess them.

Normative mappings: `server_timestamp_epoch_ms` is event time in integer milliseconds; `product_action` is the item action; `query_vector` is the search vector; `description_vector` and `image_vector` are separate item modalities. `product_skus_hash` and `clicked_skus_hash` do not replace timestamped events. Support detail, add, remove, purchase, search, and page types; an item may be absent for search and an itemless page. Map ID, category, and price-bucket fields to the actual file schema without guessing their encoding.

Do not modify source files. Retain size, SHA-256 of exact bytes, version, schema, recorded-event range, and known gaps. Read in chunks without requiring all raw tables in memory. Complete reading of released rows and telemetry completeness are separate attributes; the latter remains `unknown`.

Necessary format and integrity checks are allowed before evaluation periods are opened. Outcomes, quality metrics, and policy comparisons on D are not used to select parameters, period boundaries, or statistical block length.

### 3.2. Logical structures

| Structure | Main fields and constraints |
|---|---|
| RawEvent | Source namespace, session, event ID, event time, availability time, type, SKU or search payload; reference to the source row/bytes. |
| CanonicalBlock | Session, exact group time, canonical events, raw multiplicities after repeated-ID removal, search payloads, and reasons they are unusable. |
| SessionRecord | First-request and last-recorded-event times, period, calendar admission, whether all released rows were read, unknown telemetry completeness, exclusion reason. |
| RequestContext | request_id, session_id, trigger_block_id, t_q, a_q, eligible-request ordinal, visible prefix, cart and search state, prefix version. |
| TargetRecord | request_id, recorded-purchase set Y_q, window boundaries, and positive-cohort membership. An empty target does not establish that no purchase occurred. |
| CandidateResult | Ordered source results, final distinct SKUs, ranks and provenance, availability masks, quotas, failure reasons, selected branch. |
| FeatureBatch | request_id, SKU, matrix of numerical and categorical features in fixed order, masks, feature_schema_sha256. |
| SupervisionRecord | Candidate labels and pair weights for the sole request of a session with a nonempty target; unavailable to the service. |
| RequestResult | Package version, actually selected candidates and branch, delivered list, scores, explanations, delivery status, reasons, timing measurements, trace reference. |
| Explanation | SKU, branch, package and template versions, sources, and computational support under §10.7; the safe public representation is separate from internal diagnostics. |
| SessionEvaluation | s, positive-cohort membership, V_s, U_s, Hit_s of the first list, calendar cell, and predefined slices. A session without a target is not counted as a miss. |
| MetricResult | Value or null, status, reason, unit, denominator, counts, interval, and its assumptions/validity. |

Event and availability times use int64 UTC milliseconds; counters use int64. Check counter and time-difference overflow; explicitly convert seconds to milliseconds. Preserve original UTF-8 ID bytes and sort by the normative order, not locale.

Suggested formats: Parquet for normalized tables; arrays with explicit dtype, shape, and row order for matrices. Pin codecs in the environment. Serialization preserves multiplicities, ID order, time precision, and unknown-value indicators.

### 3.3. Run directory

Each run is stored separately. One manifest contains fixed inputs, code version, dependencies, settings, and result references. Selection and evaluation plans are fixed before the corresponding data are opened (§4); results do not change plans.

```text
runs/<run_id>/
  manifest.json            configuration and input identity
  plans/                   frozen selection and evaluation plans
  historical/              catalog, graphs, frequencies, dictionaries, vectors
  datasets/<period>/       requests and features; targets stored separately
  models/                  ranker
  results/                 metrics and check results
```

Directories are created as computation proceeds; the final serving package follows §10.4. Intermediate files do not require separate manifests, schemas, and hashes unless a consumer contract uses them. A general artifact registry, dependency graph, and cache are not mandatory.

A file is written to a temporary path and published after validation without overwriting a previous result. A set is ready once all mandatory files have been written. Reuse requires checking completeness and compatibility of inputs, code, settings, and formats; otherwise, compute a new result.

Source CSV files are not copied into the run directory. Specify their location and the cleanup procedure for intermediate data and traces. Deleted inputs cannot be recovered from a hash. A separate access and deletion system is unnecessary for local computations; live-service requirements are in §11.2.

## 4. Run manifest and stage control

Basis: `app:run-manifest`, `app:run-stages`, B13.

JSON Schema validates manifest structure; semantic checks cover inputs, time boundaries, model compatibility, and requirements for the current step. The `template_not_runnable` template is not executable. Do not invent missing values.

One manifest describes a run. Stage results are stored separately; a new manifest snapshot and a separate SHA-256 at every stage are not required. Changing fixed inputs, code, or settings creates a new run.

| Gate | Checks |
|---|---|
| preflight | Input files, schema, periods, code, settings of active modules, and early data checks. Models, planned precision parameters for D, and future reports are not required. |
| post_training | Model, feature schema, applicable mathematics checks, data separation, and disabled calibration. Service and A/B are not required. |
| pre_selection | Variants, budget, and selection rule are fixed before opening C_select; training and checks relevant to evaluation are complete. |
| pre_D | Policy or pair, evaluation and sensitivity plan are fixed: half-width in proportions, planned variance, assumptions and dependence multiplier, C_smoke pilot provenance, grid, and blocks; selection and applicable checks are complete. No D results exist yet; E is unused. |
| post_D | Estimates, intervals, and actual statuses; the plan is unchanged. |
| post_load | Production profile only: load, memory, latency, failures, and recovery; not required for the demo in §10.6. |
| pre_AB / post_AB | Admission and results of the live experiment under §13; require an accepted service. |

C_select, D, and A/B plans are stored immutably; a result references the plan used and its hash. The D plan links the selected model, selection report, and C_select plan. A hash alone does not prove the plan was adopted before data inspection: execution order and the prohibition on reading closed periods enforce this.

Local `ideal_deterministic` evaluation does not depend on post_load and does not confirm service readiness. Evaluation under `execution_trace` requires a validated service and a complete trace. A general command does not open C_select or D automatically; E remains closed.

Manifest fields `serving` and `operations` describe the deferred production profile. Its deadlines and `post_load` do not apply to the demo in §10.6: the overall timeout is a local-run setting. The manifest format is unchanged; applicable correctness checks run without declaring all production acceptance criteria passed.

A missing mandatory result is `not_run`, a violation is `failed`, and an inapplicable module is `not_used` with a reason. Applicability is determined by the run profile and current computation, not a desire to skip a check. Final states follow §17.2.

## 5. Preparing events, requests, and targets

Basis: `sec:data`, `sec:time`, `sec:requests`, `sec:eval-observed`; B01–B03, B06, B10.data.

### 5.1. Import and canonicalization

1. Check the files and their schema. Merge behavior and search by session and exact time.
2. Use product_action for item behavior; do not create an accompanying pageview as a second item view.
3. Remove confirmed repeated deliveries using stable transport IDs. Without such an ID, matching fields do not prove a retry. A technical source-row ID does not recover event order.
4. In `set_v1`, merge identical (session, time, item, action) tuples into one event, preserving original multiplicity and provenance.
5. Form complete simultaneous blocks. Permuting source rows within a block does not change the result.
6. Use a=t in historical mode. The live adapter passes only accepted complete blocks and server-side availability time.

Compare search vectors before normalization by FP64 coordinates; +0 and −0 are equal. Store distinct original vectors, multiplicities, and missing/invalid markers. Do not merge different vectors with the same normalized direction. Lists product_skus_hash and clicked_skus_hash are outside the primary context.

The latest search block is usable when it contains a single valid value and no invalid markers; repeats of that value are allowed. Otherwise, disable the context without returning to an earlier search.

For a reproducible subsample within each period, compute SHA256(uint64_be(seed) || uint64_be(ID byte length) || utf8(session ID)), sort by hash, then UTF-8 ID. The initial state is 17. Set the session count before sampling; retain all their rows without favoring purchasing sessions.

On C_smoke, run the required `multiset_v1` sensitivity check: multiplicities affect counters, history, and means, but targets remain sets and graph-edge support remains the number of distinct sessions.

### 5.2. Prefix and schedule

The first request follows the first block with a valid item detail, provided no purchase was observed earlier or in that block.

For every method, Q_max=1: later blocks do not create requests in this experiment but remain in the log for target computation. Request time does not depend on a future purchase, metadata availability, or a nonempty response. An unusable search vector disables the source but does not change admission.

The prefix contains only events from the same session with t(e)≤t_q and a(e)≤a_q. The retrieval artifact must have been published strictly before t_q.

Store separately the latest 20 detail events, latest 20 overall blocks, complete latest detail group, visible-prefix counters, latest item views, and search. Resolve tied times during truncation by ID order without treating it as causal.

An item's cart state is add/remove/unknown, initially unknown. The latest unambiguous group sets add/remove; simultaneous add and remove set unknown. Repeated removals do not create negative quantities. Observed state does not guarantee the actual cart balance.

### 5.3. Recorded targets and calendar admission

Initial H=86400 s and g=1800 s. For an admitted session, τ_s=min(t_0,s+H, t_last,s^obs), where t_last,s^obs is the last recorded event in the merged behavior and search stream. Target Y_q contains every distinct SKU of recorded purchases in the same session in (t_0,s,τ_s], not just the first or last purchase. Repeated SKUs do not enlarge the target; the right endpoint is included.

Admission to period [a_J,b_J): t_0,s∈[a_J,b_J) and strict t_0,s+H+g<b_J, regardless of purchase. For A_G, additionally require t_last,s^obs+g<b_AG; exclude a session with a late tail in full. Boundaries and subsample sizes are fixed before target analysis. Read all released rows of a selected session in full; missing rows are a preparation error, not telemetry uncertainty.

The last record and buffer g do not prove session closure. With unknown telemetry, nonempty targets support conditional evaluation, while empty targets do not create a negative session outcome. A purchase outside I_G remains in the target and denominators. Adding a target item to candidates is prohibited.

Stage output: canonical blocks, admission/exclusion reasons, the first request and its context, separate recorded targets, and counts of eligible N_eligible, retained N_obs, and target-bearing N_+ sessions. Selection of N_+ is not passed to retrieval, features, or the recommendation decision. A_G and resource measurements also use sessions without a target.

Criteria: B01, B02, B03.data, B06, B10.data; tests cover time boundaries, a purchase in the first block, multiple purchases and repeated SKUs, an empty target with unknown telemetry, omitted released rows, and changes to the future that leave past inputs unchanged.

## 6. Historical package A_G

Basis: `sec:model-graph`, `sec:model-content`, `app:first-config`; B04, B05, B09, B10.

### 6.1. Catalog, graph, and statistics

Build I_G from all retained A_G sessions after admission under §5.3, not only purchasing sessions. Events from subsequent periods do not expand the baseline's catalog, graph, dictionaries, or historical statistics.

Three directed relations: detail→detail, detail→add, detail→purchase. Different items, a strictly later item group, a distance of 1…10 groups, and delay 0<Δt≤1800 s. Exclude self-loops and simultaneous transitions; groups without item events do not count toward distance.

For each (session, relation, i,j), take the maximum exp(−Δt/600), then sum session contributions. Support is the number of sessions with an eligible pair, not repetitions. Equivalently, the minimum eligible pair delay within a session may be stored.

Processing order:

1. Remove edges with support <3.
2. Retain up to 100 edges per row and relation by descending weight, then ID.
3. Normalize retained weights; an empty row is zero.
4. Retain d_i as the sum of retained weights and ρ_i=d_i/(d_i+10).

Store P, W, supports, row sums, ρ, and the ID mapping sparsely, without a dense catalog matrix. Intermediate aggregation cost and the largest time group are accounted for separately from graph size.

Popularity is the number of distinct A_G sessions with an item detail; smoothing η_P=1. Separately store the number of distinct A_G sessions with a recorded purchase of the item: repeated purchases within a session do not increase the count. These statistics describe records, not purchase probability.

Historical purchase rate is excluded from the feature schema under §8.4. The first-view cohort, its denominators, and smoothing are not computed.

### 6.2. Representations and numerical stability

Normalize description, image, and search vectors separately. Matching dimensions do not permit adding different spaces. Search-to-description similarity is determined by the spaces' provenance contract.

Compute the norm with scaling by the maximum absolute coordinate. Threshold ε_v=10⁻¹²:

- absent vector → mask 0, vector_absent;
- incorrect dimension or type, or nonfinite coordinates → mask 0, vector_invalid;
- small/zero norm → mask 0, vector_zero;
- usable vector → unit direction and mask 1.

An unusable external modality does not disable usable branches. Do not mask internal NaN/Inf: training fails, and the service uses the specified fallback.

Implement shared stable softplus, sigmoid, LSE/log-softmax, a scaled norm, and compensated summation. Apply masks before taking a maximum; handle an empty domain in a separate branch.

B04 tests a short decay time: exp(−800) underflows to zero in FP64, but mathematical support remains. Check support, weights, and normalization together; use stable computation or fail early instead of producing 0/0. Do not change initial τ_G=600 s.

## 7. Candidates and features

### 7.1. Exact retrieval

Basis: `sec:model-candidates`; B07, B08.

Use I_q=I_G excluding already observed purchased items. Viewing an item or adding it to the cart does not exclude it.

| Source | Quota |
|---|---:|
| detail→detail | 50 |
| detail→add | 50 |
| detail→purchase | 50 |
| Description | 40 |
| Image | 30 |
| Search | 50 |
| Recent distinct views | 20 |
| Popularity | 10 |
| Total | 300 |

History is limited to L_H=20, with decay τ_H=300 s. Each relation's graph score is a weighted sum of ρ_i·P_ij. Normalize description and image history representations independently using usable vectors. A zero sum, including from opposite vectors, means the source is absent.

Search uses the latest available search block. An unusable block or age >600 s disables the source without returning to the preceding block. The first run uses exact search over all eligible usable vectors. Blocked matrix multiplication is allowed with an exact global Top_m and a shared tie rule.

After union, deduplicate, preserve all provenance, and fill remaining slots by popularity up to M=300. Filling has separate provenance and is not capped by the popularity quota of 10. Do not populate an unavailable source with items carrying invented scores. A small catalog yields fewer candidates.

For each candidate, distinguish source-list membership, score availability, and rank presence. Compute measurable scores even outside the quota. Encode rank r≥1 as 1/(1+r); a missing rank is 0 with a separate mask.

### 7.2. Feature schema

Basis: `sec:model-features`. Before training, fix `feature_schema`: name, position, type, formula/TeX label, mask, transformation, provenance, and fitting period for each column. Derive the columns from the mathematical specification's families and formulas.

Required families:

- graph scores by relation, links to the latest group, support, and masks;
- description/image similarities to history and the latest group;
- search, its age, and masks;
- candidate view count and age;
- observed cart state and age;
- available-prefix counts and duration;
- category, category similarity, ordinal price bucket, and group comparisons;
- popularity and a separate count of A_G sessions with a recorded purchase of the item;
- source provenance, original scores, and reciprocal ranks.

Group features use the entire latest detail group. Mean: Σn_j·M_ij·φ(i,j)/Σn_j·M_ij; n_j=1 for set_v1, multiplicity for multiset_v1. With a zero denominator, value and mask are zero. Maxima do not depend on multiplicity.

Category similarity is the common initial path length divided by the maximum of the path lengths and one. Category is the baseline's only column treated by the model as categorical: an A_G dictionary in UTF-8 order, unknown value 0, known values 1,2,… . Item ID is not a numerical feature. Price bucket is an ordinal quantity with a mask, not an order price.

Log transforms of counts and ages are allowed only as transformations fixed in advance. Changing the feature schema or retrieval requires new B features and a new ranker.

Check the affected parts of B01/B02/B05/B07–B10; full acceptance follows §§15–16. Batch and sequential computations must match exactly in masks, categories, candidates, and order. For FP64 features, check max_j |x_replay−x_serve|/(10⁻⁸+|x_replay|)≤10⁻⁶.

## 8. Ranking

### 8.1. Training table

Basis: `sec:model-ranking`, `sec:model-ranker-implementation`.

Training uses only the first requests of B sessions with a nonempty recorded target. A row is an actual primary candidate; its label is membership in Y_q. Do not add positive items from outside the candidates. Candidates outside Y_q are surrogate negatives, not confirmed impressions without a response.

Order requests by (session_id,t_q,request_id) and items by UTF-8 ID. One session contributes one group; pass nonempty group sizes to Dataset.group. An empty candidate set creates no rows, but the target-bearing session stays in the evaluation denominator. Groups without positive-negative pairs have zero derivatives. If the entire table has no pairs, report insufficient data, not successful training. The first run has no negative subsampling.

Before building the table, estimate the upper bound of N_B^+·M rows, memory, and disk. The table may be built in chunks, but full training must not silently become sequential training on chunks, the evaluation catalog must not be reduced, and group weights must not change.

### 8.2. Custom derivatives

`session_lambdas_v1` is a callable, not a string naming a built-in LightGBM objective. At every iteration:

1. Check raw-score finiteness and recompute full group order by (−score,ID).
2. Construct every positive-negative pair.
3. Use session weight 1, pair weight 1/max(1,|C⁺|·|C⁻|), binary IDCG@20, and Δ_ij under `sec:model-ranking`.
4. With current ranks and pair weights fixed, accumulate gradients and diagonal second derivatives of the logistic loss, β_R=1.
5. Accumulate in FP64 and return contiguous FP32 arrays. Do not replace zero Hessians with an artificial positive floor.

Pairs below the depth cutoff have zero weight, but ranks use the whole group; pairs crossing the boundary are not discarded. Apply weights once inside the function, with Dataset.weight=None. Built-in lambdarank parameters must not change this computation.

### 8.3. LightGBM settings and model persistence

For profile `gcs_rk_v2_7`, pin the complete recipe from `sec:model-ranker-implementation`: model, Dataset, training, and prediction parameters, including LightGBM 4.6.0 defaults. The recipe identifier is not the protocol version; check correspondence before training.

Save the run configuration, actual tree count and leaf sizes, derivative-function hash, feature schema, group order, and pair statistics. A budget of 200 does not guarantee 200 trees. Do not multiply predict scores by learning_rate again; do not treat min_data_in_leaf as a strict guarantee of actual row count.

Initial score f_0=0. Pass the categorical column name to `categorical_feature`; in the first run, set no monotonic constraints, feature interaction constraints, or forced splits.

The primary list is sorted by descending f, then ID; return the first min(20,|C_main|). Scores are not claimed as purchase probabilities.

### 8.4. Disabled components

Basis: `sec:model-features`, `sec:model-calibration`; B09, B11.

In `observed_purchases_v1`, the calibrator is neither trained nor loaded; `calibration.enabled=false`, `calibration_status="not_used"`. Period E is not read. The probabilistic diversity procedure is disabled, λ_D=0.

With unknown telemetry completeness, historical purchase rate is absent from the feature schema rather than filled with zero and a mask. Recorded-purchase counters are retained under §6.1.

## 9. Temporal protocol and historical evaluation

Basis: `sec:eval-temporal`, `sec:eval-offline`, `sec:eval-uncertainty`, `sec:metadata-sensitivity`, `app:diagnostic-output`.

### 9.1. Periods and result access

Seven disjoint parts: A_G → A_N → B → C_smoke → C_select → E → D. Store exact [a,b) boundaries, whole-session assignment, and a period-opening log.

- A_G: all retained sessions after admission under §5.3; catalog, graph, popularity, recorded-purchase counters, dictionaries, and scales.
- A_N: training for enabled extensions; unused by the baseline.
- B: candidates from fixed retrieval and features; each variant trains its own ranker on sessions with a recorded target.
- C_smoke: debugging and diagnostics of errors, cost, and coverage losses.
- C_select: one selection from a list frozen before opening. New variants after inspection are prohibited.
- E: reserve, unused in the current protocol.
- D: one final evaluation of the frozen plan.

Preliminary calendar proportions are 20/20/25/15/10/10%; C is split equally by time. This grid does not require using unsuitable periods. Check boundaries, H+g losses, and statistical block counts before training using permitted early data.

After changes based on C_select results, that period becomes development data; confirmatory selection requires a new one. An extension after baseline evaluation on D requires a new final period. Unused A_N and E do not become extra baseline data.

### 9.2. Metrics and diagnostic tables

Publish N_eligible, N_obs, N_+, and exclusion reasons under §5.3. The primary evaluation uses the same population of sessions with a nonempty recorded target for every method, N=N_+. An empty list, failure, or fallback does not remove such a session from the denominator. At N=0, the metric is null and status is `insufficient`.

Mandatory calculations:

1. Session Hit_s: intersection of the first delivered list with the recorded target. Multiple hits do not increase a session's contribution.
2. Hit@20 = ΣHit_s/N in proportions; paired differences against the first-evaluation controls from §1.1.
3. V_s: a target exists in the eligible catalog; U_s: a target exists among the selected branch's candidates; candidate bound ΣU_s/N.
4. Check 1 ≥ V_s ≥ U_s ≥ Hit_s and exact decomposition 1−Hit_s = (1−V_s)+(V_s−U_s)+(U_s−Hit_s). Mean losses sum to 1−Hit@20.
5. Distinguish target items missing from the historical catalog from those ineligible for a particular request. A table of individual purchases does not replace the session decomposition.
6. Candidate coverage among sessions with an available target; a zero denominator yields an undefined value with a reason.
7. NDCG@K and Recall@K as auxiliary metrics under `sec:eval-diagnostics`; do not replace Hit@20 with them.
8. Slices by history length, prior search, cart, item popularity and known status, and modality availability. Popularity-group boundaries come from A_G.
9. Separate counts for technical failures, masks, deadline overruns, fallback, and missing traces. Measure cost and failures also on sessions without a recorded target.

For sources, retain pre/post-quota sizes, measured-list depth, intersections, unique items, coverage, and provenance. Do not treat an unmeasured list as empty. Average intersections over defined first requests; show the missing fraction separately.

`Delta_fixed(source)` removes only items with no other provenance, without filling freed slots or retraining. Excluding a source while redistributing quotas is a separate configuration with its own training. Overall coverage cannot be obtained by adding source coverages.

### 9.3. Replay mode

Mode depends on the implemented path:

- `ideal_deterministic`: a=t, fixed data, and healthy sources; results describe ideal replay, not measured live failures.
- `execution_trace`: at the service stage, reproduces versions, source availability, selected branch, deadlines, and delivery from the actual `sec:ops-replay-trace` trace. Not required for local evaluation.

Fix mode names in the schema. Before execution_trace is implemented, a successful recomputation must not be presented as reconstruction of the U_s actually used. An incomplete trace affects report validity.

### 9.4. Intervals and sufficiency

Compute a single policy's Hit@20 or a paired difference on sessions with a recorded target. For the primary estimate, fix the dependence plan and block-sum vector. Intervals are conditional on purchase recording and do not remove incomplete-telemetry bias.

Primary algorithm for Coveo:

1. Construct a dense grid of eligible start times [a_D, b_D−H−g), with a predefined origin and step. Cell count J must be an integer. Retain empty cells; do not add the calendar buffer H+g.
2. Assign a target-bearing session in full by t_0,s. For a pair, store (Σd_s, n_j) per cell, where d_s = Hit_s(π₁)−Hit_s(π₀) and n_j is the number of sessions with a nonempty target. For one policy, replace d_s with Hit_s.
3. With block length m cells, select ceil(J/m) starts uniformly from 1…J−m+1, concatenate consecutive blocks, and retain J indices.
4. Compute a ratio of sums, not a mean of daily Hit@20 values. Carry the paired policies together.
5. Perform 2000 replicates with initial state 17 and α=0.05. Quantiles use nearest-rank: element ceil(pB) of sorted results.
6. If even one replicate has a zero denominator, do not discard it: the whole interval receives `insufficient`.
7. Save the primary interval, two predefined alternative block lengths, an estimate of Monte Carlo error in the bounds, and the time series.

Require at least 20 full nonoverlapping blocks and 20 nonempty blocks. For one policy, require at least 100 target-bearing sessions in each class (hit and miss), with each class occurring in at least 20 blocks. A pair requires at least two categories from {−1,0,1}, each with 100 sessions spread across at least 20 blocks.

Zero empirical variation, coincident bounds, unsuitable assumptions, or insufficient counts yield `insufficient`, even if a numerical result exists. Retain the point estimate and reasons. Additional replicates do not compensate for too few source blocks.

For a usable interval, check the specified half-width in proportions: `single_policy_half_width_fraction` for one policy and `half_width_fraction` for a pair. After training, use an authorized C_smoke pilot to justify a positive threshold, planned variance of the relevant outcome, assumptions, and temporal-dependence multiplier. Retain the pilot's provenance. Before opening D, fix these parameters, the grid, and blocks in an immutable evaluation plan together with the primary policy or pair. Missing planned precision parameters do not block `preflight` or the first training run. Do not invent missing values. IID computation is permitted only as separately labeled sensitivity analysis.

### 9.5. Selection and the variant without metadata

The first primary result evaluates one GCS+Rk. For a later comparison, register configurations, costs, GCS+Rk control, and selection rule before opening C_select. After selection and before opening D, fix one primary pair.

Select only among variants passing computational constraints. To compare the original model with one new variant, compare integer sums of Hit_s over the common population with nonempty targets; retain the original model on a tie. Publish time and memory separately without load runs to break ties. Do not name a winner if targets are absent or results incomplete.

Variant `gcs_rk_metadata370` changes only M to 370 and description/image quotas to 80/60. Catalog A_G, B, feature formulas, and the training recipe remain the same. Recompute B features; a schema with a separate config_id prevents substitution of original artifacts. Prepare C_select once under the fixed plan. An already inspected D does not independently confirm improvement.

Mandatory variant `graph_recent_popularity_no_provided_metadata` excludes text, image, and search vectors, category, price, their masks and derived features, search clicks/lists, and query context. Retain IDs, events, graph, repetitions, and A_G statistics; the neural model and SID are disabled.

Catalog, admission, sessions, the sole request, K, and M remain shared. Quotas: 50 for each of the three graph relations, 20 repeated items, and 10 popularity items; fill remaining slots to 300 by popularity after deduplication. It requires its own B features and ranker.

B10 criterion: arbitrary replacement of provided metadata does not change this variant's candidates, features, or recommendations. Publish V/U/Hit, difference from the baseline, sensitivity intervals, and cost on D. This is not automatically a second confirmatory claim.

### 9.6. Report

Generate JSON and Markdown/HTML from the same results. Include all `app:diagnostic-output` blocks: data, exclusions, coverage stages, three losses, recommendations, sources, slices, cost, statuses, and the rationale for the next study.

Do not replace missing data with invented values. Retain distinct reasons for undefined ratios, unusable intervals, and unmeasured lists. The header identifies period, evaluation protocol, replay mode, run_id, and plan/package hashes. Telemetry completeness remains unknown. In the machine-readable report, `N_obs` is the retained-session count and `N` the target-bearing-session count; check N≤N_obs≤selected subsample size. Primary estimates are `Hit@20` or `DeltaHit@20` in proportions.

Completion criteria: B07–B12, independent recomputation on a small example, and agreement between the report and source session tables; after D, the `post_D` check.

## 10. Service, state, and versions

A separate deliverable, not a prerequisite for evaluation on D in `ideal_deterministic` mode. The portfolio requires one local process with profile `local_checksum_only`. Model-loading, input, recommendation-eligibility, time-boundary, and retry checks remain required. PostgreSQL, multiprocess fault tolerance, RTO/RPO, serving-package signatures, and production load are deferred to a separate task.

Basis: `sec:operations-features`, `sec:operations-serving`, `sec:ops-security`, `sec:ops-recovery`.

### 10.1. Adapters and API

The existing memory and file-replay implementations suffice for the demo. Block acceptance and state changes remain atomic within the process. A production database adapter, a shared transaction with the log, and durability under node failure are outside the current scope.

Proposed routes:

| Route | Purpose |
|---|---|
| POST /v1/blocks | Accept one complete block; return a receipt/status and server decision time. Event acceptance does not imply a successful recommendation. |
| POST /v1/recommendations | Request recommendations for an accepted trigger_block_id using an idempotency key. The service checks first-request admission and Q_max=1 itself. |
| GET /health/live | Check process liveness without changing user state. |
| GET /health/ready | Check readiness of the package, required dependencies, and mandatory logging within the specified deadline. |

A combined block-and-recommendation route is allowed. There, a_q is fixed after atomic acceptance, while the overall deadline starts at the first byte. A block retry does not create a new logical request. A recommendation retry does not extend the original deadline.

The public response contains query_id, package version, items, branch, delivery status, a safe reason code, and explanations under §10.7. Full candidates, features, scores, and raw history are available only through restricted diagnostics. HTTP codes do not remove requests from analytical denominators.

### 10.2. Accepting sealed_block_v1

The envelope contains `source_namespace, session_id, event_time_ms, block_id, event_count, event_ids, events, payload_sha256, closed=true`.

Mandatory checks:

- the sender is authenticated and registered as namespace owner; one aggregator combines behavior and search;
- all events belong to one session and one timestamp;
- event count matches array lengths; event_ids[k] equals events[k].event_id; IDs are unique within the block;
- the service computes SHA-256 of the exact immutable payload bytes itself; the hashed byte boundaries are fixed in the schema;
- the receipt binds namespace, session, time, block_id, count, and hash; block_id alone is insufficient to recognize a retry;
- an incomplete block is not accepted in parts; a closed timestamp cannot be extended with new IDs.

| Limit | Initial value |
|---|---:|
| Envelope / SKU | ≤1 MiB / ≤256 UTF-8 bytes |
| Raw events per envelope | 1–100 |
| Logical events per block / session | ≤100 / ≤1000 |
| Search list | ≤1000 elements |
| Per-session rate | 10 raw events/s, allowed burst 100 |
| Minimum cost of any attempt | 1 |
| Allowed lateness / lead | 1800 s / 0 s |

Check logical limits after repeated-ID removal and canonicalization; rate limits use actual raw events, including retries. On an overrun, reject the entire new block. Connection and sender limits apply before the session is identified.

For a new block, check −1800 s ≤ t(e)−a_block ≤ 0 s and the allowed calendar range. A time error does not change history or t_last; future events are not buffered. Check a confirmed retry before rechecking its time window.

### 10.3. State and transitions

The key is (source_namespace, session_id). A record contains U_s, change number v_s, transformation version, latest arrival time, completeness and gap reasons, required aggregates, a general tail of L_hist=20 blocks, an independent queue of L_H=20 views, the latest search context, and schedule data.

Retain the complete latest view group when features need it; the general tail and L_H queue do not replace each other. Evicting old blocks does not reset aggregates over the entire available prefix. Purchases and assignments are stored independently of the live state record.

| Case | Action |
|---|---|
| New valid block | In one atomic operation, write events, receipt, state, and v_s+1; a(e)=server block-acceptance time. |
| Version conflict | Reread and recompute the transition; at most one repeated atomic update within the original deadline. |
| Confirmed complete retry | Do not change state, completeness, original availability times, or responses. Rate-limit exhaustion may produce duplicate_rate_rejected without a history gap. |
| Existing block_id with a different payload / new IDs at a closed time | closed_block_conflict; preserve old history and quarantine the new unknown portion. |
| Known events only | Do not create a new block. |
| Mixture of old and new IDs | Do not accept individual elements; check the closed timestamp and agreement between IDs and their original fields. |
| Rejected new trusted block | Mark a gap. Without a verified prefix, return failure without recommendations until recovery. |
| Unverifiable retry | retry_unverifiable; do not treat it as a new event. Log checking/recovery determines completeness. |
| Untrusted sender | Reject without changing someone else's session. |
| Unavailable/incompatible/incomplete state | Return failure without recommendations if no verified prefix exists. Fallback is allowed only with a verified prefix and confirmed item eligibility; an empty history cannot substitute for unknown state. |

A request receives one confirmed snapshot with t(e)≤t_q and a(e)≤a_q and one compatible package. If the current record contains later events, reconstruct the required snapshot. A late block does not change a past response.

Recovery: checkpoint before the tail → ordered tail events → the same StateReducer. Limits: ≤1000 events and ≤1 MiB of log per request, within the overall response deadline. Exceeding them produces `state_replay_limit`; background work does not extend the response deadline.

The cache key includes visible-prefix version, t_q, and package, not only the session. A closed transport block does not prove session completeness: a declared stream-completeness signal is required.

### 10.4. Publication and rollback

The package contains a consistent catalog, graph, vectors, dictionaries, feature schema, ranker, policy, explanation templates, and enabled extensions. Before activation, check dimensions and space provenance, schemas, versions, hashes, dependencies, and evaluation protocol.

Choose the latest compatible publication strictly before t_q, not a_q. If none exists, return `version_unavailable`; a newer package cannot be used even for fallback.

Loading must not execute serialized code, access paths outside the package, accept unknown types, or exceed declared sizes. A local checksum is allowed only for the research profile. The live profile requires an Ed25519 signature; its trusted key is supplied independently of the package.

Activation is atomic; a request is pinned to one version. A state-schema change requires a validated migration or recovery. Rollback does not change old responses or permit an expired catalog.

For a future live service: RTO≤60 s and RPO of confirmed assignments and outcomes is 0; recovery tests cover every instance. The demo makes no such promise. It retains failure handling for unavailable state/catalog, logging errors, package corruption, and incompatible schemas; simultaneously loaded versions count toward memory use.

### 10.5. Response deadline and branch selection

Use monotonic clocks for durations and the absolute deadline; use UTC for event and publication times.

One configurable timeout covers the whole request from its first byte, including queuing, computation, serialization, mandatory logging, and return. The demo's initial value is **1000 ms**, fixed before launch. This timeout is not a latency guarantee under load. There are no separate normative stage deadlines: exceeding the former 10/15/25/35 ms does not itself trigger failure.

The primary branch stops with time reserved for the already prepared fallback, logging, and return. Set the reserve as an execution parameter before launch, not as a separate acceptance criterion for each stage. The overall deadline is not extended. A cancellation signal does not guarantee termination of a noninterruptible operation; retain the bounded pool, deadline checks, and prohibition on late publication. A late result does not change the response or state.

The log must link the request, package, selected candidates, public response, and outcome. A two-phase commit for every operation and an RPO=0 guarantee are not required for the demo. A mandatory-logging error cannot be reported as successful completion; unknown delivery cannot be presented as confirmed. A retry returns the saved outcome without recomputation.

Prepare fallback in advance: the most recent distinct viewed items in descending time order, then ID for ties, followed by popularity without duplicates. Up to M=300 candidates and K=20 items; the same eligibility filter. Do not merge it with the primary branch or pass it through the primary ranker.

After branch selection, fix branch and C_served. If delivery or mandatory logging misses the deadline, L is empty but the selected candidates are retained. If no ready branch was selected, both sets are empty. Always L⊆C_served⊆I_q. Unknown eligibility excludes an item; unusable fallback means technical failure. A response error does not undo accepted events.

Minimum failure reasons: `invalid_input`, `overload_rejected`, `backend_timeout`, `cancelled`, `state_replay_limit`, `logging_failure`, block conflicts, unavailable version, and incompleteness reasons.

### 10.6. Demo API acceptance

Using the prepared package and model, check four scenarios: a normal primary-branch list, a block and request retry without recomputation, a computation timeout with delivered fallback, and incomplete state resulting in failure without recommendations. The timeout must actually reach computation. Compare the HTTP response with the reread log; late work must not change the outcome.

Measure the full response including logging, plus package loading and peak memory separately. Specify hardware, request count, timeout, branches, and failures. P95/P99 may be reported as measurements on a sufficiently large set, but are not mandatory thresholds; four requests do not prove load performance. Keep normal CPU-core allocation; computer tuning to pass the check is not required.

Demo acceptance consists of applicable correctness checks B03.service/B14 and EXPL01 plus these four scenarios. It does not assign production `S_serve=passed` or `post_load`. The 100 requests/s profile, three extended runs, B15/B16, and RTO/RPO from the mathematical specification remain deferred live-service requirements, not a barrier to the portfolio.

### 10.7. Template-based explanations

Basis: `sec:explanation`. Module `explanations` explains each recommended item from the selected branch's trace and the pinned package. Diagnostics retain retrieval sources, their scores, and the explanation's support:

| Source | Computational support |
|---|---|
| Graph | Relation, prefix item with the greatest total contribution, contribution value, and support n_ij for that relation; ties use ID order. |
| Content | Modality used and measured similarity. |
| Search | Reference to the available search query, compatible vector-space version, and measured similarity. |
| Recent views, popularity, filling | Actual item provenance, without attributing graph or content relationships. |

Obtain the supporting information during source computation or reproduce it from pinned inputs; an aggregate score does not replace an individual item's contribution. Fallback is explained only through its own sources. Unusable search and unused sources do not become recommendation reasons.

Templates use only verified trace fields. Internal support is stored under §11.2; the public representation uses minimal reason codes without a raw event sequence. It describes the computation, not why a purchase occurred; anonymized vectors cannot establish an item's name, attributes, or compatibility. No language model is used.

Include the template version and hash in the package. Generate explanations within the overall deadline (§10.5). Acceptance follows §15.5.

## 11. Monitoring, storage, and protection

Requirements for the live service, not mandatory infrastructure for the local experiment. Local work retains resource limits, protection of closed periods, input and package validation, and a prohibition on publishing raw data.

### 11.1. Metrics and alerts

Immediately collect input flow, errors, fallback, empty/short responses, unknown items, missing values, candidate sizes and uniqueness, package versions/age, latency, and memory. For extensions, also collect masks, norms, truncations, index agreement, and collisions.

For distribution shifts:

- numerical-bin boundaries and categories are fixed on early data; missing values form a separate category, with histogram smoothing ε_h=1/2;
- the baseline window requires ≥10 000 requests; current nonoverlapping UTC hourly windows require ≥1000 requests and ≥200 sessions;
- threshold tuning requires ≥100 control hourly windows before C_select, separate from the baseline window, with a baseline available by each window;
- A_w=max_j(JS_wj/0.05), h=max(1, nearest-rank 0.99 quantile);
- alert after two consecutive sufficient windows with A_w>h; clear after three windows with A_w<0.8h; repeat notification for a reason no more than once per hour;
- an insufficient window is not considered stable; three in a row generate a separate notification. Evaluate the firing rate of the entire rule, not just individual-window exceedances.

An alert starts diagnostics of completeness, versions, and sources, not automatic model replacement.

Operational safeguards:

- two adjacent five-minute windows with ≥1000 requests and more than 0.1% errors or 1% fallback activate verified fallback/rollback;
- regardless of volume: 20 consecutive technical failures of valid admitted requests or three unsuccessful internal checks;
- internal checks run every 5 s with a 120 ms deadline, separate IDs, and no user-state changes;
- sequence is determined by arrival order; a stuck request receives an outcome at its deadline, and success resets the corresponding counter;
- untrusted input and an individual attempt to load an inauthentic package do not roll back a healthy version; an ineligible recommendation is blocked immediately.

In historical evaluation, Hit@20 uses recorded targets; absence of a purchase in the export does not become zero conversion. Live A/B purchase outcomes are analyzed after maturity and observation-completeness checks. During confirmatory A/B testing, the model and behavioral reference data are frozen unless a different mode is justified in the protocol in advance.

### 11.2. Access and retention

| Entity | Initial retention |
|---|---|
| Live state | 24 hours of inactivity |
| Retry verifiability | At least 24 hours, independent of state TTL |
| Replay log | 48 hours |
| Detailed traces | 7 days |
| Minimal assignments and outcomes | Until analysis closure +30 days |

Assign storage owners; implement role separation, encryption in transit and at rest, and trace-read auditing. The ordinary log contains a pseudonymous key, versions, branch, items, and status, but no raw queries, vectors, or full history.

Do not treat simple ID hashing as anonymization. Identifier transformation uses a secret key and separates namespaces. The assignment key is stable across the experiment horizon; changing diagnostic keys does not change it.

Deletion covers state, logs, diagnostics, and backups. Reapply the deletion register after recovery. Expiration of input retention ends exact replay; aggregated results without session keys may remain. These durations define a project profile, not a legal conclusion.

Completion criteria: B15/B16, deletion tests and recovery tests for deleted keys, no raw data in public responses or ordinary logs, and completed owner assignments and live-profile parameters.

## 12. Optional extensions

Outside the first local version. Each extension is enabled as a separate task after baseline evaluation; its formulas, parameters, and constraints come from the [mathematical specification](/documents/en/math/coveo_session_recommender_math.tex) at the labels below. An active variant needs its own candidates, B features, ranker, compatible package, and applicable checks from §15.3. Returning to the baseline restores its quotas, features, and model.

### 12.1. Fixed paths and neural model

`sec:modern-paths`, `sec:modern-item`, `sec:modern-intents`, `sec:modern-retrieval`, `app:neural-config`; checks N01–N07.

### 12.2. Training and random streams

`sec:modern-objective`, `sec:modern-next`, `sec:modern-contrastive`, `sec:modern-training-recipe`. Random streams, batch order, and numerical modes are reproduced using the specified recipe.

### 12.3. Sampled denominator

`sec:sampled-training`, `sec:sampled-proposal`, `sec:sampled-loss`; X02. Does not replace the exact loss without a separate comparison.

### 12.4. Sequential control

`sec:sequence-control`; X03. Used as a separate comparison model.

### 12.5. Graph aging and updates

`sec:freshness`; X01. Updated reference data are published consistently; a new version does not change past responses.

### 12.6. Semantic codes SID

`sec:sid-overview`, `sec:sid-decoder`, `app:sid-config`; X04/X05.

### 12.7. ANN and additional methods

ANN: `sec:ops-ann-contract`, B17; requires a measured benefit while preserving normative agreement with exact search; the allowed Hit@20 decrease is set before C_select in field `maximum_hit_drop_fraction`. Probabilistic diversity is disabled under §8.4. Position corrections and additional comparisons require suitable data and a predefined research question. Do not reconstruct missing impression probabilities from clicks.

## 13. A/B preparation and analysis

A live pilot after separate service admission, not part of local Coveo evaluation. Normative requirements: `sec:eval-online`, `sec:eval-itt`, and `sec:eval-srm-stop` of the mathematical specification; acceptance: B18 and pre_AB/post_AB.

### 13.1. Assignment and logs

Before the first impression, fix the protocol, assignment unit, cohort, arm, dates, volume, and safeguards. Ensure stable assignment, complete logs, and accounting for every assigned session under the normative protocol. Run B18 checks before pre_AB, including independent synthetic examples.

### 13.2. Monitoring and analysis

ITT, SRM, intervals, stopping rules, and outcome maturity follow the cited specification sections, without tuning after results are inspected. No experiment means `not_observed`, not zero or positive effect. Do not implement IPS/DR on Coveo tables using invented impression probabilities.

## 14. Commands and pipeline behavior

The single interface is `coveo-recsys`. Subcommand names are planned; the implemented set is listed in the README.

| Local-route command | Purpose |
|---|---|
| manifest init / validate | Create and validate a run manifest under §4. |
| data inspect / prepare | Inspect and prepare data under §5. |
| historical build | Build the A_G package under §6. |
| features build --period B / train ranker | Compute features and train the model under §§7–8. |
| replay --period C_smoke | Produce diagnostic recommendations and a report under §9. |
| sensitivity metadata | Build the metadata-free variant required by D under §9.5. |
| selection freeze / run | Freeze the plan and perform selection on C_select. |
| evaluation freeze / run --period D | Check pre_D, freeze the plan, and perform final evaluation. |
| report --run ... | Compile a report from saved results without new selection or training. |

Checks run through the standard `pytest` invocation; a separate test-registration command and test-manifest storage system are not required. Service, load, recovery, extension, A/B, and data-deletion commands are added at the corresponding stages (§§10–13).

First pass: manifest → data inspection and preparation → A_G package → preflight → B features → training → post_training → C_smoke. Preparation and A_G construction check their own inputs and outputs but do not assign `preflight=passed`. Full admission is required before B features and training. Metadata-free sensitivity, service, and A/B do not block this pass; their mandatory check deadlines are in §15.2.

Long-running commands report progress, data volume, peak memory, and failure reasons. Resumption is permitted only from compatible results without changing the sample or random-call order.

A technical error returns a nonzero exit code. An unmet mandatory condition prohibits computation; `insufficient` is not replaced by success. Local commands check allowed periods and record their opening; training and selection do not read closed evaluation data. The live service additionally applies access rules from §11.2.

## 15. Testing and traceability

### 15.1. General procedure

Run checks through `pytest`. For each requirement ID, specify tests and the stage where they apply (§§15.2–15.3); keep inputs, independent references, tolerances, and the generator's initial state with the tests. Store code version, configuration, executed checks, statuses, and failure/skip reasons in run results. Separate preregistration and a hashed manifest for every test are not required.

A narrow platform exception is permitted for local `preflight` and `post_training` on Windows: explicitly listed symbolic-link tests may be skipped only because permission to create them is missing (`WinError 1314`). They remain `skipped` in JUnit and the summary, are excluded from successful counts, and are noted as a run limitation. On Ubuntu, these tests are mandatory. Any other skip or unsuccessful applicable check blocks admission; path protection and the tests themselves are retained.

A reference must be simpler than the primary implementation and must not call the function under test to obtain the expected result.

For each generated property, use at least 1000 cases and initial state 17; save the generator version and minimal counterexample. Also run the specification's mandatory small examples.

Sets, masks, counts, categories, versions, and order require exact equality. For numbers, use declared tolerances:

| Check | Tolerance |
|---|---|
| General FP64 calculations | atol=10⁻¹⁰, rtol=10⁻⁸ |
| General FP32 calculations | atol=10⁻⁶, rtol=10⁻⁵ |
| Derivatives before casting | atol=10⁻¹², rtol=10⁻¹⁰ |
| Derivatives after FP32 | atol=10⁻⁷, rtol=10⁻⁶ |
| Pairwise-loss finite differences, step 10⁻⁴ | absolute gradient error≤10⁻⁷, Hessian error≤10⁻⁶ |
| Agreement between replay and serving features | The formula in Section 7.2, not a generic tolerance substituted for it |

Unexpected NaN/Inf is a failure. The meaning of permitted exceptions, such as +∞ for failed-response latency, is defined separately. Nearly equal scores do not allow changes in item order to be ignored.

### 15.2. Baseline registry

| ID | Verified result | Deadline |
|---|---|---|
| B01 | Time boundaries, invariance to simultaneous-row permutations, set/multiset semantics, and no future information in inputs. | preflight |
| B02 | Search ambiguity, latest unusable context, and independence from search results/clicks. | preflight |
| B03.data | Schema, ID, time, canonicalization, and envelope correctness on data. | preflight |
| B03.service | Retries, closed timestamps, conflicts, limits, atomicity, completeness, and cart. | post_load |
| B04 | Session support and contribution, normalization, empty rows, graph regularization. | post_training |
| B05 | Vectors/numbers, derivatives, zero-contribution groups, FP32, leaf and gain, grouped categorical splitting. | post_training |
| B06 | Calendar admission independent of purchase, recorded targets, unknown outcome with an empty target, out-of-catalog targets, and complete row reading. | preflight |
| B07 | Quotas, eligibility, selected branch, delivery failure, coverage chain and decomposition. | post_training |
| B08 | Multiple provenance, source removal without filling, fallback, and denominators. | post_training |
| B09 | Historical purchase rate is absent from the schema; unknown outcomes are not replaced by zeros to compute it. | preflight |
| B10 | Seven parts, closed D, independence from E, frozen selection, and invariance of the metadata-free variant under metadata replacement. | Data: preflight; ranker: post_training; sensitivity: before its computations; repeat applicable parts before D. |
| B11 | No calibrator required, E unread, λ_D=0, and scores not claimed as purchase probabilities. | Before the first recommendation list, post_training. |
| B12 | Temporal blocks, ratio of sums, rare outcomes, degenerate intervals, boundaries 99/100 and 19/20. | Before the first report with intervals; repeat before D. |
| B13 | Stage-dependent required fields, types, hashes, unknown states, and negative results. | preflight |
| B14 | Computational agreement, publication eligibility, late events, and trace completeness. | post_load |
| B15 | Queue, load, total memory, failures, recovery, attacks on the stream, and RTO/RPO. | post_load |
| B16 | Protective stops, consecutive failures, internal checks, and firing rate of the whole alert rule. | post_load |
| B17 | ANN agreement with exact search, eligibility, empty response, and reproducibility. | post_load when enabled, before serving |
| B18 | Assignment, entry cohort, SRM, stopping, intervals, and distinction between coverage and effect. | Before pre_AB: independent examples and the actual protocol. |

B03.data/B03.service are parts of the same permanent ID B03, not new normative tests. The same applies to X02.support/X02.loss. Actual D and A/B results are recorded only at post_D/post_AB. Early B10–B12 checks use independent small examples; real C_select and D stay closed until admission, and E is unused.

Run B05 in parts: vector and numerical checks by Э3, derivative and LightGBM checks by Э4.

B10.data runs at preflight, B10.ranker by the first model application, and B10.metadata before using the metadata-free variant. B11 confirms disabled components, not calibration quality. Before D, check applicable B10–B12 and actual inputs; B11 settings are checked through the manifest, without a separate E stage. Small examples test implementation, not results on closed data.

B05 and B10 parts may appear in the overall test-run result. `passed` requires every part applicable to the current computation. Unexecuted later parts remain explicitly marked rather than counted as passed.

B18 does not block local training or evaluation. Before A/B, both independent examples and validation of the actual protocol are mandatory. Every report preserves the distinction between historical coverage and causal effect.

### 15.3. Extensions

| ID | Condition and main property | Deadline |
|---|---|---|
| N01 | Neural model: fixed support, exact and truncated paths, original connections. | post_training |
| N02 | Neural model: masks, zero projections, preservation of training examples. | post_training |
| N03 | Neural model: intents, search mixing, permutations, and diagnostics. | post_training |
| N04 | Neural model: full denominator, multiple purchases, next target. | post_training |
| N05 | Neural model: batch with one row per session, contrastive-comparison filters. | post_training |
| N06 | Training enabled extensions: initialization, Adam, batch weights, saturation, independent random streams. | post_training |
| N07 | Neural model: quotas, complete restoration of the baseline, and failure behavior. | post_training |
| X01 | Aging, publication, and enabled multistep graph traversal. | post_training |
| X02.support | Sampled mode: CDF support and generator grid before consuming the random stream. | preflight |
| X02.loss | Sampled mode: loss, derivatives, multiplicities, bias, and repeatability. | post_training |
| X03 | Sequential model: groups, causal mask, and layer normalization. | post_training |
| X04 | SID: soft derivatives, trainable parameters, and stage-specific Adam resets. | post_training |
| X05 | SID: collisions, fixed codes, eligibility, and dictionary consistency. | post_training |

Extensions also require affected baseline checks. Additional tests may have internal names but do not replace the 30 permanent IDs.

### 15.4. End-to-end tests and automation

The mandatory small dataset includes simultaneous purchases/views, multiple future purchases, a repeated SKU, an empty target, missing search and vectors, ambiguous search, duplicates, empty history, a cart conflict, an unknown SKU, a short catalog, a request without a positive candidate, a calendar exclusion, an omitted released row, and every fallback transition point. B06 reproduces the normative example: 3 retained sessions, 2 with a target, and 1 hit give Hit@20=1/2, not 1/3.

Use it to check the full path: source rows → canonicalization → requests/targets → A_G → B candidates/features → training → replay → metrics → manifest. Compare results with independent manual/direct calculations.

Automated change checks include schema validation, static type checking, quick tests, references, and applicable generated properties. Run extended training, load, and A/B analysis as separate jobs with a saved configuration. Missing resources or data do not mean a test passed.

### 15.5. Explanation checks

EXPL01 is an additional technical-specification test, not a replacement for the 30 normative IDs. Requirement trace: `sec:explanation` → `explanations` and the Explanation structure (§10.7) → Э6 / `post_load` → EXPL01. Inputs, independent references, and results follow §15.1.

Check maximum graph contribution and edge support, contribution ties, modality and similarity, unusable search, multiple provenance, filling, and fallback. An explanation must match the item, selected branch, package version, and template version. Check that the public form contains no reasons from unused sources, raw history, or unverified item attributes.

## 16. Development stages

Tasks Э0–Э10, dependencies, deliverables, and acceptance references are defined in the [development plan](/documents/en/development_plan.md). Development stages are not manifest stages.

Local route: Э0–Э5 → Э7. Service stage Э6 runs separately and is required for a live pilot, but not for historical evaluation. Research stages Э8–Э9 and pilot Э10 require separate approval.

Agree on a shared-contract change before parallel development of its consumers. After a change, recompute affected results and repeat applicable checks; reuse conditions are in §3.3.

## 17. Deliverables and final acceptance

### 17.1. Delivered results

For the local version:

- Code, pinned dependencies, and reproduction instructions.
- Validated inputs, run manifest, plans, and results for reached stages under §§3–4.
- Historical package, feature schema, ranker; the metadata-free variant by evaluation on D.
- Synthetic datasets, independent references, mapping of requirement IDs to tests, and their execution results.
- A C_smoke report; after admission, selection and D results under §9.
- Explicit limitations, data insufficiency, and capabilities not yet implemented.

The portfolio delivers the API, explanations, launch command, results of the four scenarios, and measured latency under §10.6. Production load, recovery, and operations under §11, extensions, and A/B are performed only under a separate task.

### 17.2. Separate decisions

Code acceptance confirms that the declared modules, commands, and boundary cases are implemented and satisfy the testable requirements. It does not establish that specific data are suitable, estimates are precise, or the model is useful.

Store final system statuses separately: S_data, S_math, S_offline, S_serve, S_business. Without an experiment, business effect is `not_observed`. While D is closed, S_offline=`not_run`; `calibration_status="not_used"` at every stage of the current protocol.

- A contract violation or technical-threshold overrun is `failed`.
- A correct procedure with too few sessions with a recorded target, blocks, or insufficient precision is `insufficient`.
- If violation and insufficiency coexist, `failed` takes precedence; retain all reasons.
- S_offline=`passed` means the evaluation is usable, not that the model is superior.
- Unknown telemetry completeness does not block conditional historical evaluation but does not permit estimating conversion across all sessions or business effect.
- Pilot readiness requires S_data=S_math=S_offline=S_serve=`passed` specifically in the live profile: signature, limits, access/retention, and validated monitoring thresholds. A/B additionally requires outcome observability under §13.
- Success under local `local_checksum_only` does not authorize a live pilot. Business superiority requires a separately passed causal analysis.

A successful command, report, or hash does not replace mandatory acceptance conditions. Retain negative and insufficient results with their actual statuses; `passed` cannot be promised when information is insufficient.

## 18. Decisions required before the corresponding stages

See the [README](/README.md) for the current implementation and data-delivery status. This section lists decisions required before each stage; future operational parameters do not block a local run.

| Data or decision | When required | What to retain |
|---|---|---|
| Actual version of the three tables, schema, dimensions, reading completeness, and known gaps | Before real preflight | Hashes, validation report, unknown telemetry completeness, source access rights |
| Boundaries of all seven calendar parts, subsample sizes, calendar admission, and first request | Before target analysis and training | Manifest and sampling rule without preference for purchasing sessions |
| Interpreter and supporting libraries | Э0 | Exact versions/hashes and a compatibility check |
| Concrete feature schema and permitted transformations | Before training each ranker | Version, column order, types, formulas, and hash |
| Primary estimate, half-width threshold, planned variance, and dependence | After training using an authorized C_smoke pilot; freeze before opening D, not required for preflight or the first model | Pilot provenance, grid, blocks, assumptions, dependence multiplier, and immutable plan |
| Load-test node and measurement tool, processes/threads, memory sampling rate | Before load testing | Reproducible profile |
| Live storage and atomic operation, checkpoints, and migrations | Before service acceptance | Validated adapter and failure scenarios |
| Namespace owner/aggregator, time range, tenant limits, and SKU contribution limits | Before live launch | Numerical limits and owner registry |
| Trusted publishers, catalog validity period, live keys | Before publishing the live package | Signature and compatibility checks, separate secret storage |
| Monitoring parameters, early windows, alert and storage owners | Before live-profile admission | Threshold validation and action instructions |
| Analysis closure date and copy-expiration rules | Before the corresponding live collection | Retention manifest and deletion register |
| Assignment unit, business threshold, SRM, volume/duration, and safeguards | Before pre_AB | Signed/frozen protocol and responsible owners |

Do not replace unknown values with “typical” measurements. A missing mandatory field at its deadline blocks the corresponding command, but not development and testing of other available modules.

## 19. Source version

This technical specification is based on mathematical specification 2.10 and the `observed_purchases_v1` manifest.

TeX SHA-256: `B96BD37E2C1DAC1610104503288F39EDBC8A3D18E59668ABB2AE5BEC56CE0BF7`.

Run manifest template SHA-256: `FF96AB06042494C08051BF684FC12FED753696CA013AA8119367F45666F93183`.

When these files change, recheck the affected requirements, parameters, interfaces, and acceptance registry. This technical specification does not alter the original mathematical specification.
