# Running and reproducing the project

English | [Русский](/documents/reproduction.md)

Run commands from the repository root in PowerShell. The verified environment
is Windows with Python 3.11; dependencies are pinned in [uv.lock](/uv.lock).
Installation and checks are in the [README](/README.md#installation-and-usage).

The repository includes tools to prepare configurations and admission evidence.
The workflow below builds a model and a C_smoke report from three CSV files,
without relying on previous `runs/`. The end-to-end check on real data was completed on
October 11, 2026, using a separate non-editable wheel installation. The only
external inputs were the three original CSV files. Prepared data, models,
and `runs/` directories are not included in Git.

During a local check, `uv sync` installed a stale cached build of the package.
A non-editable installation of the current wheel was checked separately.
If the CLI still uses old code after a source change, rebuild only this package
from the repository root:

```powershell
uv build --no-build-isolation
uv pip install --python .venv/Scripts/python.exe --no-deps --reinstall `
  dist/coveo_recsys-0.1.0-py3-none-any.whl
```

First install dependencies with the `uv sync` command in the README.
If the project version changes, use the new wheel name; there is no need
to clear the entire cache.

## Local demo

### Required inputs

The [service configuration](/configs/coveo_service.json) pins the completed
300-candidate experiment. The service requires:

| Input | How it is produced |
|---|---|
| `runs/coveo-historical-ag50000-v1/` | `historical build` from prepared A_G |
| `runs/coveo-b50000-ranker-v2/` | `train ranker`: model and `reports/ranker.json` |
| `runs/coveo-b50000-features-v2/reports/features.json` | The `features build` summary bound to the model; the service does not need the full feature tables |
| `data/01_raw/sku_to_content.csv` | Original Coveo release |
| Supply profile, feature schema, and explanation templates | Repository files pinned in the service configuration |

These inputs are already available on the author's computer. In a clean clone,
build them first; no downloadable model has been published. For your own run,
take paths and checksums from its results rather than copying them from an
earlier run. The service still checks the model's bindings to the package,
features, and metadata.

### Starting the service

In the first terminal, set a nonempty secret for the local sender. The second
terminal needs the same value. Do not put it in Git or in a sample response.
Use a new `run-dir` for each run.

```powershell
$env:COVEO_LOCAL_TOKEN = Read-Host 'Local sender secret'
uv run --no-sync coveo-recsys serve `
  --config configs/coveo_service.json `
  --run-dir runs/local-demo-v1 `
  --host 127.0.0.1 --port 8000
```

The current service calendar ends at **January 1, 2027, 00:00 UTC**.
For a later demo, set a valid calendar in a separate local configuration copy
and pass it with `--config`; do not change the historical training periods.
Events outside the calendar are rejected. Stop the server with `Ctrl+C`.

### Request

Run this example in the second terminal. It submits a controlled product view
using a valid SKU from the historical catalog and requests recommendations
from the trained model. It does not replay a shopping session or reevaluate quality.

```powershell
$env:COVEO_LOCAL_TOKEN = Read-Host 'The same local sender secret'
$uri = 'http://127.0.0.1:8000'
$headers = @{Authorization="Bearer $env:COVEO_LOCAL_TOKEN"; 'X-Stream-Complete'='true'}
Invoke-RestMethod "$uri/health/ready"

$sid = [guid]::NewGuid().ToString('N')
$eventTime = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$event = @{
  source_namespace='local-demo'; session_id=$sid; event_id='e1'
  event_time_ms=$eventTime; event_type='detail'
  sku='0005c742138a6bf913d314c829f41ff2d8e665e9b3670b7b337a9d59b7cff34d'
} | ConvertTo-Json -Compress
$events = '[' + $event + ']'
$sha = [Security.Cryptography.SHA256]::Create()
try {
  $digest = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($events)) |
    ForEach-Object { $_.ToString('x2') })
} finally { $sha.Dispose() }
$head = @{
  source_namespace='local-demo'; session_id=$sid; event_time_ms=$eventTime
  block_id='b1'; event_count=1; event_ids=@('e1'); payload_sha256=$digest; closed=$true
} | ConvertTo-Json -Compress
$body = $head.TrimEnd('}') + ',"events":' + $events + '}'
Invoke-RestMethod "$uri/v1/blocks" -Method Post -Headers $headers `
  -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))

$query = @{
  source_namespace='local-demo'; session_id=$sid
  trigger_block_id='b1'; idempotency_key='q1'
} | ConvertTo-Json -Compress
$response = Invoke-WebRequest "$uri/v1/recommendations" -Method Post -Headers $headers `
  -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($query)) `
  -UseBasicParsing
$answer = [Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray()) |
  ConvertFrom-Json
$answer | Select-Object query_id, items, branch, delivery, reasons | ConvertTo-Json -Depth 5
```

`payload_sha256` is computed from the exact bytes of the `events` array sent
in the body. The response is explicitly decoded as UTF-8, preserving Russian
explanation text in Windows PowerShell 5.1. `X-Stream-Complete` confirms that
the sender has finished transmitting the block; it does not assert that
Coveo's historical telemetry is complete.

### Verified response

A real HTTP request was made against the original model on October 11, 2026.
Loading the service took 6.52 s, accepting the block took 3.93 ms, and the client
received recommendations in 91.62 ms. The response contained 20 SKUs; the
first three and selected response fields are shown below:

```json
{
  "items": [
    "0005c742138a6bf913d314c829f41ff2d8e665e9b3670b7b337a9d59b7cff34d",
    "52cf0c63ee45fba1d2d76bf59639f48fcc06233c58b9fd0251d3a3013acc4a1b",
    "0e2c9bc9cdd664a3d5169e29464b450d960d02a05900705438a812a5fbd19ef1"
  ],
  "branch": "primary",
  "delivery": "delivered",
  "reasons": ["search:search_absent"]
}
```

The HTTP response matched the entry in the run directory's `requests.jsonl`.
The full log contains internal diagnostics and is not published. Recommending
a viewed item again is allowed: purchased SKUs are excluded, not all viewed
SKUs. Product names are unknown. A single request's elapsed time does not
measure P99 or guarantee performance under load.

## Full computation workflow

This workflow reproduces the basic computation through C_smoke;
it does not reproduce the published D results. The no-metadata and
370-candidate variants, C_select, D, and the service are outside this workflow.
Existing data and models remain unchanged.

Repeating `historical build` reuses only fully published, compatible parts.
If an artifact write was interrupted before its descriptor and descriptor
checksum were complete, the command requires a new run directory. Remaining
files are not automatically deleted or overwritten; completing that partial
publication in the same directory is unsupported.

### Data and limits

Obtain Public Data Release 1.0.0 using the
[Coveo instructions](https://github.com/coveooss/SIGIR-ecom-data-challenge#license),
accepting the terms yourself. Place `browsing_train.csv`, `search_train.csv`,
and `sku_to_content.csv` in `data/01_raw/`; the archive is not needed there.
The schema and expected checksums are in the
[supply profile](/configs/coveo_supply_v1.json).

The small run approved on October 11, 2026, allows up to **2,000 A_G sessions,
2,000 B sessions, and 1,000 C_smoke sessions**. Each computation command has
limits of 1 hour, 4 GiB of memory, 4 GiB of temporary files, and 2 GiB of output
where the command supports an output limit. All limits were met;
[measurements are below](/documents/en/reproduction.md#small-run-results).
Even a small output requires reading large CSV files.

For `data prepare`, `max_temp_bytes` limits temporary files; there is no
separate limit on the size of the completed tables. Historical construction,
feature computation, training, and `replay` check their own output limits.
Preparation disk usage must therefore be monitored separately.

Run the commands in order; do not proceed after an error. Use a new `$work`
for a new computation. `$csv` may point to an external directory containing
the same three files: there is no need to copy the CSV files into a clean clone.

```powershell
$python = '.\.venv\Scripts\python.exe'
$cli = '.\.venv\Scripts\coveo-recsys.exe'
$work = 'runs/reproduction-small-v1'
$csv = 'data/01_raw'
$helper = @(
  '-m', 'coveo_recsys.verification.reproduce',
  '--work-dir', $work, '--input-dir', $csv,
  '--max-result-bytes', '2147483648'
)
$prepare = @('--input-dir', $csv, '--profile', 'configs/coveo_supply_v1.json')

& $python @helper init --ag-sessions 2000 --b-sessions 2000 --smoke-sessions 1000 `
  --max-seconds 3600 --max-memory-bytes 4294967296 --max-temp-bytes 4294967296
if ($LASTEXITCODE -ne 0) { throw 'Plans were not created' }
```

`init` creates `A_G-plan.json`, `B-plan.json`, and `C_smoke-plan.json`.
Periods, seed 17, the `observed_purchases_v1` target, and whole-session sampling
remain unchanged. Set the size before inspecting purchases. A shortfall is
allowed only when too few sessions are eligible, not to alter the target mix.

### Code checks before reading CSV files

```powershell
& $python @helper tests
if ($LASTEXITCODE -ne 0) { throw 'Mandatory checks are incomplete' }
```

The command runs pytest without Coveo and load tests, saving JUnit,
property-test statistics, and a source inventory in `$work/tests/`. Previous
results are not reused. Do not change repository code or configuration
after this command.

For local admission on Windows, only explicitly listed symbolic-link test
skips caused by `WinError 1314` are allowed. They remain `skipped` in JUnit
and the summary, rather than being treated as passes. On Ubuntu these tests
are mandatory. Other skips, failures, and source changes prevent continuation;
the rule is in [§15.1 of the implementation specification](/documents/en/implementation_spec.md#151-general-procedure).
A non-editable wheel installation was checked in the clean directory
`runs/reproduction-clean-v1/`: the package was loaded from `site-packages`,
and its code matched the sources. The full pytest run excluding Coveo and load
tests took 308.73 s: 2,150 tests passed, 8 symbolic-link tests were skipped
under the approved rule, and 1 test was deselected by marker. Coverage of the
required checks for `preflight` and `post_training` was confirmed; this is
not yet data admission or a C_smoke evaluation result.

### A_G, the historical package, and B

```powershell
& $cli data prepare @prepare --plan "$work/A_G-plan.json" `
  --run-id reproduction-small-ag --run-dir "$work/a_g"
if ($LASTEXITCODE -ne 0) { throw 'A_G preparation did not complete' }
& $python @helper check-ag
if ($LASTEXITCODE -ne 0) { throw 'A_G failed the independent check' }
& $python @helper historical
if ($LASTEXITCODE -ne 0) { throw 'Historical build configuration was not created' }
& $cli historical build --config "$work/historical.json" `
  --run-id reproduction-small-historical --run-dir "$work/historical"
if ($LASTEXITCODE -ne 0) { throw 'Historical build did not complete' }

& $cli data prepare @prepare --plan "$work/B-plan.json" `
  --run-id reproduction-small-b --run-dir "$work/b"
if ($LASTEXITCODE -ne 0) { throw 'B preparation did not complete' }
& $python @helper check-b
if ($LASTEXITCODE -ne 0) { throw 'B failed the independent check' }
```

`check-ag` and `check-b` compare the shared tables with the published rows of
selected sessions: time boundaries, first request, prefix, targets, and period
separation. The production canonicalizer and target computation are not used
as reference implementations. Rechecking does not require the earlier
10,000-session B or per-session files.

### Admission, features, and model

```powershell
& $python @helper features
if ($LASTEXITCODE -ne 0) { throw 'preflight admission was not obtained' }
& $cli features build --period B --config "$work/features.json" `
  --run-id reproduction-small-features --run-dir "$work/b_features"
if ($LASTEXITCODE -ne 0) { throw 'Feature computation did not complete' }

& $python @helper train
if ($LASTEXITCODE -ne 0) { throw 'Training configuration was not created' }
& $cli train ranker --config "$work/train.json" `
  --run-id reproduction-small-ranker --run-dir "$work/ranker"
if ($LASTEXITCODE -ne 0) { throw 'Training did not complete' }
```

`features` assembles `preflight` from the new run's check results and creates
the configuration. `train` pins the completed features; a separate command
trains the model. The 300-candidate quotas, feature schema, and LightGBM recipe
remain unchanged. If there are no targets or training pairs, do not increase
the sample based on the observed purchases: leave the run rejected and agree
on a new sample size separately.

### C_smoke and the report

```powershell
& $python @helper evaluation-preparation
if ($LASTEXITCODE -ne 0) { throw 'post_training/evaluate admission was not obtained' }
& $cli data prepare @prepare --plan "$work/C_smoke-plan.json" `
  --evaluation-config "$work/evaluation-preparation.json" `
  --run-id reproduction-small-csmoke --run-dir "$work/c_smoke"
if ($LASTEXITCODE -ne 0) { throw 'C_smoke preparation did not complete' }
& $python @helper check-csmoke
if ($LASTEXITCODE -ne 0) { throw 'C_smoke failed the independent check' }
& $python @helper replay
if ($LASTEXITCODE -ne 0) { throw 'Evaluation configuration was not created' }
& $cli replay --period C_smoke --config "$work/replay.json" `
  --run-id reproduction-small-replay --run-dir "$work/evaluation"
if ($LASTEXITCODE -ne 0) { throw 'Evaluation did not complete' }
```

`evaluation-preparation` checks the trained model, binds it to its own features
and B, then creates `post_training` with B12 and `operation=evaluate`.
`check-csmoke` repeats the direct check on the new data; `replay` supplies
their actual hash. Outcomes are joined only after recommendations are computed.

Outputs are `$work/evaluation/reports/replay.json` and `replay.md`, plus the
response and session-score tables. For GCS+Rk, P, and R, inspect the number of
sessions with a recorded target and Hit@20 on those sessions. Telemetry
completeness is unknown; an empty target is not a negative outcome. Do not
invent support for interval assumptions: `insufficient` is an allowed status.

In previously saved reports, the `history=...` slices counted at most the last
20 views rather than the full prefix. They must not be used to compare
quality on long histories. These slices have not been recomputed; the fix
does not change overall Hit@20, coverage, Recall@20, or NDCG@20.

[verification/reproduce.py](/src/coveo_recsys/verification/reproduce.py)
assembles all configurations and evidence from current files and check results,
without depending on earlier `runs/`.

### Small-run results

The workflow completed in `runs/reproduction-clean-v1/runs/reproduction-small-v1/`.
The package was loaded from `site-packages`, and installed code matched
the checked sources. Previous data, models, and D results were not changed.

| Step | Volume | Time, s | Peak memory, MiB |
|---|---|---:|---:|
| A_G preparation | 2,000 sessions | 518.09 | 104.05 |
| B preparation | 2,000 sessions | 619.45 | 99.06 |
| C_smoke preparation | 1,000 sessions | 648.97 | 123.74 |
| Historical package | 2,794 SKUs | 5.67 | 139.43 |
| B features | 2,000 requests, 600,000 rows | 291.75 | 150.80 |
| Training | 200 trees | 112.39 | 319.57 |
| C_smoke evaluation | 1,000 requests | 127.38 | 215.66 |

Direct checks of A_G, B, and C_smoke took 87.67, 121.38, and 88.31 s, respectively.
Each preparation peaked at 467.40 MiB of temporary files. The four completed
tables occupy 3.68, 3.36, and 1.60 MiB for A_G, B, and C_smoke, respectively,
excluding auxiliary JSON files. Feature, model, and evaluation directories
occupy 59.46, 0.13, and 22.56 MiB.

A_G, B, and C_smoke contain 22, 24, and 12 sessions with recorded targets,
respectively. Training uses 11 positive candidates, 7,189 negative candidates,
and 3,289 pairs. C_smoke Hit@20 among the 12 sessions with targets is
3/12 (25%) for GCS+Rk, 1/12 (8.33%) for P, and 3/12 (25%) for R.
The other 988 sessions have unknown outcomes. Intervals are marked
`insufficient`: there are too few temporal blocks and categories, dependence
assumptions are unsupported, and the required precision has not been justified.
These figures verify the basic workflow, not the quality of the main model
or the published D results.

## Measured costs

In new comparison reports, `full_recommendation_timing[policy].recommend_seconds`
contains the total time spent in `recommend` calls, excluding subsequent diagnostics,
serialization, and writes. The older `calculation_seconds` retains its original
meaning: computation including different diagnostics for the two models and
their JSON serialization. Old reports have no new timing measurement;
reading them emits a warning rather than substituting the old time.
Neither measure is HTTP latency.

These are measurements of completed local runs on Windows, not predictions
for another computer. Command measurements include loading, checks, and writes
unless stated otherwise. The directories below identify local result
sources, not downloadable artifacts.

| Step | Time | Peak process memory | Source in `runs/` |
|---|---:|---:|---|
| Resumed A_G package build, 50,000 sessions | 370 s | 493 MiB | `coveo-historical-ag50000-v1/` |
| B preparation, 50,000 sessions | 34 min 52 s | 1,725 MiB | `coveo-b50000-v1/` |
| Main-model B features | 1 h 49 min 17 s | 1.12 GiB | `coveo-b50000-features-v2/` |
| Main-model training with checks | 35 min 20 s | 2.00 GiB | `coveo-b50000-ranker-v2/` |
| C_smoke evaluation, 50,000 sessions | 2 h 31 min | 3.81 GiB | `coveo-csmoke50000-replay-v2/` |
| D preparation, 60,000 sessions | 21 min 35 s | 344 MiB | `coveo-d60000-v2/` |
| D evaluation, four methods | 3 h 12 min 16 s | 837 MiB | `coveo-d60000-evaluate-v2/` |
| 300/370 comparison on C_select | 3 h 38 min 5 s | 706 MiB | `candidate370-v1/comparison/` |

The A_G measurement is for a resumed build, not a full build from CSV files.
Main-model features occupy 1.52 GiB, the model and summary occupy 0.72 MiB,
and the two D evaluation tables occupy 1.71 GiB. Preparing 50,000 sessions
has limits of 4 hours, 8 GiB of memory, and 4 GiB of temporary disk space;
these limits are not promises of actual usage.

Quality result sources: `runs/coveo-d60000-evaluate-v2/reports/replay.json`,
`runs/coveo-d-plan-v2/manifest-post_D.json`, and
`runs/candidate370-v1/comparison/reports/replay.json`.
Results without raw events are in the [README](/README.md#measured-results).
Re-evaluating the already examined D reproduces the earlier experiment,
but provides no new independent confirmation of improvements.
