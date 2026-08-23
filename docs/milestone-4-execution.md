# Milestone 4 Execution Plan

Milestone 4 closes the gap between PufferLab's reviewed interview demo and an operator-ready local
workflow. A user must be able to determine what is configured before a live request, create and
later remove only a PufferLab-owned tiny namespace, turn durable evaluation evidence into a
deterministic CI decision, and reproduce the browser/accessibility smoke without oral guidance.

The original hardening checklist in [`implementation-plan.md`](implementation-plan.md) is partly
complete already: live provider integration, resumable ingestion, cancellation/recovery, export,
README guidance, responsive manual QA, and the deterministic offline fallback all shipped through
Milestones 1–3. This plan owns the remaining operator gaps plus the smallest high-value P1 feature,
the provider-free evaluation gate.

## Goal and success story

The finished workflow has two explicit tracks because an ad hoc tiny-fixture comparison does not
create a durable evaluation run:

```text
Provider-free evidence:
demo seed -> gate synthetic durable run -> serve -> inspect dashboard

Credentialed live Playground:
doctor --mode live-tiny -> ingest generated tiny namespace -> show authenticated receipt
                        -> serve -> compare -> clean exact owned namespace
```

The browser must show whether live comparison is locally configured before enabling a comparison.
The readiness path, run dashboard, evaluation gate, and browser CI make no provider call. Only the
explicit ingest, optional live doctor check, comparison, and receipt-bound cleanup may reach
turbopuffer.

Milestone 4 exits when a clean checkout can execute that workflow from documentation, the
provider-free browser smoke and accessibility scan are required CI gates, every delivery PR is
independently reviewed and reviewer-only merged, and protected `main` is green after one reviewed
finalization PR.

## Requirements

### Functional

- Report local capability state for the offline demo, live tiny-fixture Playground, persisted
  dashboard reads, and canonical live evaluation without exposing configuration values.
- Prevent a missing API key, namespace, or optional live-search runtime from producing the first
  actionable signal only after a live comparison POST.
- Provide installed `doctor` and one-worker `serve` commands with stable, scriptable exit behavior.
- Persist one installation-local, authenticated, crash-safe ownership receipt before a generated
  tiny namespace can be created; bind it to the creating credential identity and region, resume
  that exact receipt, and clean only its exact derived namespace.
- Evaluate a completed durable run against aggregate and per-query quality policy from the CLI,
  with a stable nonzero exit for policy failure distinct from invalid evidence or infrastructure
  failure.
- Run the synthetic dashboard journey and accessibility checks in required CI without credentials,
  provider access, licensed data, or tracked screenshots.

### Non-functional

- Readiness is local configuration evidence, not provider authentication or remote health. It never
  auto-probes turbopuffer from API startup, page load, polling, or browser prefetch.
- Provider-free/default doctor, dashboard reads, eval gates, and browser CI construct no provider
  client, embedding model, reranker, or live-search runtime and never transmit a credential. The
  default doctor may read the configured `SecretStr` only inside a local constant-time receipt HMAC
  comparison. Explicit `doctor --live` may unwrap it only into its command-scoped metadata client;
  that client is closed and drained before return. Neither path prints or copies the value into a
  report/error.
- Public responses and output from new Milestone 4 CLI commands contain allowlisted states,
  requirement codes, commands,
  UUIDs, counts, thresholds, and numeric deltas only. They contain no credential, configured
  namespace value, local model/cache path, licensed query/document text, qrel, provider body, or
  traceback. Existing CLI commands retain their reviewed output contracts. The sole configured
  target-output exception is the explicit authenticated `namespace show-tiny` command needed to
  copy its owned region and namespace assignments into `.env`.
- Cleanup never accepts a namespace, receipt path, ownership key, or deletion token from a caller.
  A valid prefix, persisted dataset row, provider lineage field, caller-selected data directory,
  caller-created file, different credential, or current environment region is insufficient
  authority.
- Evaluation policy is deterministic, finite-only, threshold equality is a pass, and incomplete or
  unauthenticated evidence fails closed without being mislabeled a regression.
- Latency remains advisory in this milestone. No noisy single-run p95 becomes a release gate.

## Architecture decision

**Status:** Proposed until M4-0 receives independent review.

Adopt four narrow components over the existing single-process FastAPI/SQLite architecture:

```text
Settings + optional-package metadata
              |
              v
     Local capability inspector ------> GET /api/v1/capabilities ------> Playground guidance
              |
              +------------------------> pufferlab doctor

Generated tiny ingest --> authenticated fixed receipt --> turbopuffer namespace
              ^                                             |
              +----------- exact show/resume/cleanup <-------+

Validated SQLite run + qrels + durable outcomes
              |
              v
       Pure gate evaluator --> safe gate report --> pufferlab eval gate / CI exit

Synthetic seed + one-worker API + built web --> Playwright + axe --> required Frontend check
```

The local capability inspector is shared by HTTP and CLI, but the API exposes only the live
Playground capability needed by the browser. The evaluation gate remains a CLI/process boundary;
adding a second FastAPI policy-execution surface would not improve CI integration. The ownership
receipt is an ignored local capability, not a public contract or database row.

### Options considered

| Option | Value | Risk / limitation | Decision |
|---|---|---|---|
| Documentation-only setup repair | Small | Still allows a doomed comparison and relies on oral sequencing | Reject |
| Local diagnostics plus CI eval gate only | High | Leaves generated interactive namespaces without an installed safe cleanup path | Incomplete |
| Diagnostics, owned tiny lifecycle, eval gate, and automated browser gate | High | Several focused review units, but no migration or arbitrary schema work | Adopt |
| Generic JSONL/qrels importer in the same goal | Very high later | Untrusted parsing, licensing, identity, runtime authentication, and schema mapping make it a separate milestone | Defer |
| Branch an arbitrary customer namespace | Strong demo story | Current queries/configs require a registered compatible dataset; turbopuffer supplies no ownership token | Defer |
| Warm hint or ANN recall panel | Useful diagnostic | Cost-bearing, easy to overclaim, and not required to close setup or deployment decisions | Defer |

## Frozen behavior

### Local capabilities and actionable search failures

Add a versioned provider-free endpoint:

```text
GET /api/v1/capabilities
```

The response reports a `live_playground` capability with:

- state `locally_configured` or `action_required`;
- a deterministic ordered list drawn only from `api_key`, `search_namespace`, `region`,
  `live_search_runtime`, `owned_tiny_receipt_invalid`, `owned_tiny_credential_mismatch`, and
  `owned_tiny_region_mismatch`; invalid fixed receipt bytes fail closed, while credential/region
  mismatch codes apply only when an authenticated receipt's exact namespace is the configured
  Playground namespace;
- an allowlisted next-action code used by the frontend to render checked-in instructions.

It never returns the key, namespace, region, environment-file path, data-directory path, model
path, package version, provider response, or an assertion that the remote namespace exists. The
optional runtime check uses package metadata/spec discovery only; it does not import or initialize
the embedding or reranking model.

`GET /health` continues to mean only that the API process is alive. The top bar must not imply that
live search is ready. The Playground loads capabilities before enabling **Compare results**. When
action is required, it shows the exact relevant local step and sends no comparison POST.

Missing local search configuration uses a direct typed `configuration_required` error before any
provider-capable factory. Provider authentication, namespace readiness, model construction, and
other runtime failures retain their honest separate redacted semantics; they are not guessed from
local settings.

The installed commands are:

```bash
pufferlab doctor --mode demo
pufferlab doctor --mode live-tiny [--live]
pufferlab doctor --mode evaluation [--dataset-version <uuid>] [--live]
pufferlab doctor --mode all [--dataset-version <uuid>] [--live]
```

The default path is read-only and provider-free. It checks the selected capability without
creating a database/directory/sidecar, migrating SQLite, recovering jobs, writing bytes, or
changing database mtime; evaluation inspection opens an already existing database read-only.
`--dataset-version` is accepted only by `evaluation` and
`all`. Without it, evaluation mode selects the sole persisted dataset version; zero or multiple
versions are action-required rather than guessed. It resolves every target namespace from an
authenticated fixed tiny receipt or SQLite dataset row, never from a raw namespace argument.
`--live` is an explicit, potentially billable metadata-only check; it makes at most one metadata
request per selected target through a dedicated pinned turbopuffer 2.9 client. The client pins the
validated official HTTPS region URL, uses `max_retries=0`, a ten-second timeout, an owned HTTP
client with environment proxy inheritance disabled (`trust_env=False`) and redirects disabled, and
a request hook that enforces the exact GET target and reconstructs allowlisted headers with exactly
one Authorization value.
SDK base-URL/custom-header environment overrides therefore cannot redirect or replace the configured
credential. No application retry wraps the client, so retryable status, transport, timeout,
redirect, and close failures still produce exactly one outbound attempt. The check performs no
write, document search/retrieval query, create, or delete, closes and drains the provider under
success/error/repeated cancellation, and reports only `metadata_reachable` plus
`index_up_to_date` or `index_updating`. A blank region is a local configuration failure. Receipt
corruption, credential mismatch, and current-region mismatch are distinct allowlisted
action-required codes and never expose either value. These states do not prove schema, exact corpus
identity, authentication for other operations, or working BM25/ANN retrieval.

Exit codes are frozen as:

- `0`: every requested check is ready;
- `2`: local configuration, dependency, catalog, or evidence is missing/invalid;
- `3`: an explicitly requested metadata request failed/not-found or reported `index_updating`;
- `1`: unexpected internal failure.
- `130`: the command was interrupted; output is one fixed cancellation line and contains no
  partial target or provider detail.

Add `pufferlab serve --host 127.0.0.1 --port 8000`. It accepts only exact loopback hosts
`127.0.0.1`, `::1`, or `localhost` and integer ports 1–65535, always starts exactly one worker, and
rejects invalid input with exit `2` before server construction. `SIGINT`/`SIGTERM` performs bounded
graceful shutdown and returns `0`; startup or unexpected runtime failure returns `1`. The existing
runtime guard remains the authoritative second-process defense.

Uvicorn receives five seconds to drain requests, connections, and tasks before lifespan shutdown;
that Uvicorn timeout does not bound a blocked lifespan handler. A process-level watchdog bounds the
complete synchronous runner, including lifespan shutdown and asyncio runner teardown, to ten
seconds after the first shutdown signal. If cooperative shutdown exceeds that outer bound, the
watchdog uses `os._exit(0)`: this deliberately skips Python cleanup handlers and buffered-output
flushing, so in-flight work may remain interrupted and must be reconciled by the existing SQLite
transaction semantics plus runtime startup migration/recovery on the next launch. The operating
system releases the process lock and sockets. A second signal requests Uvicorn's immediate
force-exit path; neither signal is replayed after handlers are restored.

### Authenticated tiny-namespace ownership

Only `dataset ingest-tiny` with no explicit `--namespace` participates. Before its first provider
operation it creates one production receipt and owner key in an application-chosen absolute state
location resolved from the OS user-account record, independent of `PUFFERLAB_DATA_DIR`, cwd,
`HOME`/XDG overrides, CLI arguments, or other environment-selected paths. On POSIX the frozen
locator is the account-record home plus `.pufferlab/state/owned-tiny-v1`; platform support must use
an equivalent non-environment OS account API or fail closed. Tests may inject paths only into
private helpers. That mode-0700 directory has fixed `owner.key`, `receipt.json`, and
`operation.lock` children; no production path component is caller-selectable. There is at most one
active generated-tiny receipt globally for this local OS user.

Every directory component and file is checked against symlinks; receipt/key files use exclusive
no-follow mode 0600. Missing fixed directories, the owner key, and the initial receipt are fully
prepared under random private staging names, file-synced where applicable, and published only by
native atomic no-replace rename; a fixed occupant is never chmodded, overwritten, or removed after
a stale userspace check. Creation and every state transition use an authenticated atomic
replacement, file plus directory `fsync`, and unchanged owner-key/file identity checks before the
prior receipt can be replaced or removed. A fixed no-follow mode-0600 sibling lock is acquired with a
non-blocking exclusive POSIX `fcntl` lock and held across every ingest/resume/cleanup provider
operation and receipt transition; contention fails safely before provider construction. Platforms
without equivalent process locking fail closed. Each transition writes an `O_EXCL|O_NOFOLLOW`
mode-0600 temporary file in the same directory, flushes and `fsync`s it, atomically exchanges it
with the fixed receipt, inspects the displaced inode, and `fsync`s the directory. Mismatch
restoration first moves the installed replacement to a private quarantine, then restores the
displaced object only into a vacant fixed locator; a new fixed occupant is never clobbered. It
authenticates and reloads the prior bytes and file identity immediately before a compare-and-swap
transition or removal, so a concurrently or manually replaced fixed receipt cannot authorize
work. The receipt binds at least format version, purpose,
creating region, nonce, derived namespace, lifecycle state, a non-output HMAC tag of the creating
credential, and the receipt authentication tag. Neither tag, the credential, nor a
credential-derived fingerprint is printed or returned.

The state machine is finite:

```text
intent -> created -> ready -> cleanup_requested -> not_found_verified
```

- `intent` is durable before any ambiguous remote creation attempt.
- `created` follows the first confirmed namespace-creating write.
- `ready` follows the existing exact schema, ordered-ID/count, and index-readiness checks.
- `cleanup_requested` is durably and atomically committed before the irreversible delete request.
- An interrupted rerun authenticates and resumes the same receipt/namespace; it never mints another
  target while one receipt is active.
- Cleanup may reconcile an authenticated `intent`, `created`, or `ready` receipt so a timed-out
  create cannot orphan a second generated namespace.
- A repeated cleanup resumes an authenticated `cleanup_requested` receipt idempotently: it repeats
  delete-or-already-absent and bounded not-found verification before completing.
- Successful absence verification durably transitions to `not_found_verified` before receipt
  removal. A rerun that authenticates that terminal receipt performs no second provider/delete
  call. Under the same lock, one atomic no-replace rename removes the fixed locator; one held
  descriptor then validates the exact moved inode, canonical bytes, and HMAC. The state directory
  is synced to make fixed absence durable before wiping and syncing only that inode. The owner key
  remains. A crash after the fixed move is remote-safe
  because absence was already durable; random quarantine names are never scanned or restored as
  authority, so a retry performs no provider action and a later generated ingest may start anew.
- Before provider construction for either resume or cleanup, the command must HMAC-verify the
  currently supplied API key against the non-output credential tag and must use the authenticated
  receipt's creating region, never the current environment region. A rotated or different key
  fails closed; recovery requires the exact creating key.
- The receipt is removed only after delete-or-already-absent and a bounded not-found verification;
  the owner key remains. Any provider, verification, cancellation, replacement-file, or close
  failure before the terminal fixed-locator move retains the receipt and exits nonzero. A local
  wipe/fsync failure after exact held-descriptor validation is commit-like: it exits nonzero but
  never promotes an arbitrary quarantine back to authority; fixed absence remains remote-safe.

The fixed state path, every fixed directory component, `owner.key`, `operation.lock`, and
`receipt.json` are the local authority boundary and are protected against replacement at every
mutation. Compliant PufferLab commands are additionally serialized by the stable account-home
lock. Internal staging uses 128-bit random `O_EXCL` names and preserves observed colliders; those
random implementation names are not claimed to resist a malicious same-UID directory watcher,
which can already read `owner.key` and ignore advisory locking. Crash or hostile-interference paths
may leave ignored random zero-byte or private staging artifacts; they are never authority and no
finite-residue claim is made for that out-of-boundary case.

Installed commands expose no target argument:

```bash
pufferlab namespace show-tiny
pufferlab namespace cleanup-tiny
```

`show-tiny` prints the exact `TURBOPUFFER_REGION=...` and
`PUFFERLAB_SEARCH_NAMESPACE=...` assignments from the authenticated receipt. Both resume and
cleanup use that creating region. `cleanup-tiny` acts only on the fixed authenticated receipt.
Explicit `--namespace`
ingestion stays idempotent but is never retroactively treated as cleanup-owned, even when its name
has a valid PufferLab prefix. Legacy, Unix, customer, branched, persisted-only, malformed, copied,
or unrecorded namespaces are outside this cleanup capability.

After verified cleanup, the command and runbook explicitly tell the user to clear the now-stale
`PUFFERLAB_SEARCH_NAMESPACE` assignment and restart the API. Automatic `.env` editing remains
forbidden.

No automatic cleanup runs at API startup, browser navigation, or process exit. Destructive work is
always an explicit installed command. CLI help and the runbook disclose that cleanup's bounded
metadata/not-found verification may itself be billed as zero-row queries.

### Provider-free evaluation gate

The installed command is:

```bash
pufferlab eval gate <run-id> \
  --candidate <config-id> \
  --metric ndcg@10 \
  --min-delta 0 \
  --max-query-drop 0.20 \
  --max-error-rate 0 \
  --min-paired-queries 50 \
  [--format text|json]
```

`metric` is one of `ndcg@10`, `recall@50`, or `mrr@10`. The candidate must be one of the run's
three immutable candidate configs. Policies contain finite values in these closed domains:
`min-delta` is -1 through 1, `max-query-drop` and `max-error-rate` are 0 through 1, and
`min-paired-queries` is 1 through the canonical 50 attempts. Equality at every threshold passes.

The application adapter is a dedicated read-only composition, not `RuntimeCliApplication`, whose
constructor performs migration and recovery. It accepts only a completed durable run and opens an
existing database without creating it, migrating it, recovering jobs, writing sidecars, or changing
database bytes or mtime. It constructs no provider, search runtime, embedding model, or reranker.

Evidence trust is explicit rather than overstated. Unix sources/qrels are re-authenticated with
`authenticate_persisted_unix_query_set`; synthetic sources/qrels are reproduced exactly with
`materialize_synthetic_demo`. Run lifecycle, expected config catalog, query/outcome identity,
ordering, uniqueness, and stored structural/content hashes are validated. Metrics are recomputed
from qrels and the locally persisted ranked document IDs rather than copied summaries or
caller-supplied values. Those ranked IDs remain trusted local durable evidence, not
cryptographically authenticated evidence; protection against a malicious full-database rewrite is
a separate signing/key-management migration outside this milestone.

A pass requires all requested conditions:

- the arithmetic mean of exact per-query paired candidate-minus-baseline metric deltas is at least
  `min-delta`; independently averaged baseline/candidate populations are never subtracted;
- every paired query delta is at least negative `max-query-drop`;
- candidate error rate—failed candidate outcomes divided by the canonical 50 attempted queries—is
  at most `max-error-rate`;
- paired-query count is at least `min-paired-queries`.

A completed run must contain every expected durable outcome identity. A missing outcome is corrupt
evidence, not an ordinary search failure. A typed failure outcome is a valid structurally validated
durable outcome:
it contributes to the candidate error-rate numerator and its pair-status exclusion reduces the
paired count. No-positive-qrel exclusions also reduce the paired count but never become zero-valued
quality. Validated completed evidence that misses error-rate, paired-count, aggregate-delta, or
per-query-drop policy returns a policy failure. Corrupt, unauthenticated, nonterminal, or zero-pair
evidence is not evaluable and returns invalid evidence instead.

The report orders checks as error rate, paired-query coverage, aggregate delta, then per-query drop.
Violations within a check sort by `(observed delta ascending, query UUID ascending)` and the report
bounds per-query violations to the first ten. It includes only run/config/query UUIDs, metric names,
observed values, thresholds, sample/paired/excluded counts, and bounded violating deltas. Query
text, document IDs/text, qrels, namespace, trace/provider payloads, local paths, and credentials are
forbidden from text, JSON, exception context, `repr`, and traceback output.

Exit codes are:

- `0`: policy passed;
- `4`: valid evidence evaluated and policy failed;
- `2`: invalid policy, unknown binding, non-completed/corrupt/unauthenticated/zero-pair evidence;
- `1`: unexpected internal failure.

Synthetic completed runs are gateable for quality and must prove zero provider/model construction.
Latency is excluded because synthetic timing is unavailable and one live run does not establish a
stable performance distribution.

### Automated browser and accessibility gate

Add Playwright Chromium and axe coverage behind `pnpm test:e2e`. The harness creates a fresh
temporary data directory, uses the real provider-free synthetic seeder, starts the one-worker API
and built frontend, and cleans only its own temporary processes/data.

The smoke covers:

- Playground action-required state before any comparison POST, with an assertion that the entire
  provider-free journey emits zero browser POST requests (including search compare, evaluation
  start/cancel, and live replay);
- `/runs`, completed synthetic run detail, candidate/order/limit regression state;
- recorded query deep link and selected document drawer;
- reload and Back/Forward restoration;
- focus movement, dialog focus trap, Escape, and opener focus return;
- desktop and 390-pixel mobile containment;
- no serious or critical axe violations on Playground, run list/detail, and query detail;
- zero browser console errors and poisoned provider/search/model factories on the API process.

The existing required `Frontend` GitHub check installs Python and pinned `uv` dependencies as well
as the pinned browser, seeds through the installed CLI, builds the frontend with an explicit
loopback `VITE_API_BASE_URL`, runs the API and built assets on allocated loopback ports, executes
this smoke, and uploads screenshots/traces only on failure as ephemeral CI artifacts. Successful
screenshots, licensed/provider evidence, databases, exports, and browser artifacts never enter Git.

## Current turbopuffer constraints

Official documentation confirms that namespace metadata exposes approximate count and index state,
but a metadata request is billable as a zero-row query and `up-to-date` does not prove exact corpus
completeness. PufferLab therefore retains its stronger schema plus exact ID/count readiness checks
and makes live doctor metadata explicit rather than automatic:
[`metadata`](https://turbopuffer.com/docs/metadata).

Namespace branching is constant-time copy-on-write and suited to test pipelines, but a branch is
billed separately, begins unpinned, and the provider supplies no PufferLab ownership capability.
The website currently documents optional lineage metadata while the current official OpenAPI and
pinned SDK do not type that field, so it cannot authorize cleanup without a separate live contract
check. Arbitrary customer schemas also do not satisfy the current registered dataset/query/config
contracts. Branching is deferred until a later milestone can accept only a persisted compatible
source and reuse this milestone's authenticated ownership pattern:
[`branching`](https://turbopuffer.com/docs/branching),
[`testing`](https://turbopuffer.com/docs/testing).

Warm-cache calls acknowledge only a hint and may be billed; they do not expose cache state or prove
causal latency improvement. ANN `_debug/recall` is a separately billed index-recall diagnostic,
not judged Recall@50, and may continue after client timeout. Both remain explicit future operator
features rather than implicit readiness work:
[`warm cache`](https://turbopuffer.com/docs/warm-cache),
[`ANN recall`](https://turbopuffer.com/docs/recall).

## Dependency and branch graph

```text
M4-0 -> M4-A
M4-A -> M4-B -> M4-C
M4-A -> M4-D
M4-C + M4-D -> M4-E
M4-B -> M4-F
M4-C + M4-E + M4-F -> M4-G
```

M4-D can implement pure gate math while M4-B owns the first CLI composition edit. M4-C follows
M4-B for the namespace command edit. M4-E integrates the reviewed pure gate after both CLI edits
are merged. M4-F consumes the reviewed capabilities API and can proceed independently of namespace
cleanup and the gate CLI. M4-G is the single goal-finalization PR.

## Review units

### M4-0 — Architecture and gap audit

- **Owner:** root orchestrator
- **Branch:** `codex/m4-plan`
- **Files:** this plan, the detailed milestone summary, and `docs/progress.md`
- **Acceptance:** bring Milestone 3 to canonical verified state; prove which original hardening
  items already shipped; freeze scope, contracts, ownership authority, provider-cost boundaries,
  branch dependencies, rollback, non-goals, and completion criteria before implementation begins.

### M4-A — Contract freeze

- **Owner:** contract worker
- **Branch:** `codex/m4-contracts`
- **Dependencies:** merged M4-0
- **Files:** capability/error/gate Pydantic contracts and contract tests/documentation; OpenAPI and
  generated TypeScript change only for the already reachable error schema at this stage
- **Acceptance:** freeze capability states/requirement/action codes, direct
  `configuration_required` error, finite CLI-only gate policy/report and stable check ordering;
  prove no secret/path/namespace value fields exist; do not publish a capabilities path before it
  is mounted or expose CLI-only gate models through OpenAPI; no provider, CLI, persistence, route,
  or UI behavior.

### M4-B — Readiness, doctor, and one-worker serve

- **Owner:** readiness worker
- **Branch:** `codex/m4-readiness`
- **Dependencies:** merged M4-A
- **Files:** local capability inspector, capabilities route, safe search preflight, doctor/serve CLI,
  OpenAPI/generated TypeScript, focused application/API/CLI tests
- **Acceptance:** capability and default doctor paths perform no provider/model construction,
  database/directory/sidecar creation, migration, recovery, write, byte change, or mtime change;
  `serve` intentionally retains the existing API startup migration/recovery semantics;
  action-required state precedes provider factories; optional metadata-only live check closes under
  every exit and proves one outbound attempt under retryable status/transport/timeout failures;
  stable outputs/exits and exact redaction attacks pass; serve is loopback-only and fixes one
  worker. M4-B installs an injected owned-tiny target resolver that remains action-required until
  M4-C supplies the authenticated receipt; M4-B proves the live one-shot path with a selected
  existing live SQLite dataset and never invents a raw namespace argument.

### M4-C — Owned tiny lifecycle

- **Owner:** namespace lifecycle worker
- **Branch:** `codex/m4-owned-tiny`
- **Dependencies:** merged M4-B
- **Files:** package-owned authenticated receipt, generated ingestion integration, namespace
  show/cleanup CLI, fake/adversarial tests
- **Acceptance:** durable intent before provider access, HMAC-derived exact target, no-follow/0600 and
  file/directory `fsync`, fixed no-follow exclusive process locking, authenticated compare-and-swap
  transitions, credential/creating-region binding, exact resumability including
  `cleanup_requested` and terminal `not_found_verified`, safe concurrent-command rejection, no
  caller target/path/token seam, bounded delete plus not-found verification, receipt retention on
  every ambiguous/failure path, explicit namespaces never cleanup-owned.

### M4-D — Pure gate engine

- **Owner:** evaluation worker
- **Branch:** `codex/m4-gate-core`
- **Dependencies:** merged M4-A
- **Files:** pure gate policy evaluator and exhaustive unit/property tests
- **Acceptance:** no FastAPI, CLI, SQLAlchemy, provider, or frontend imports; hand-calculated metric
  boundaries, hidden large regression, coverage/error, non-finite policy, equality, and deterministic
  ordering cases pass.

### M4-E — Gate application and CLI

- **Owner:** evaluation integration worker
- **Branch:** `codex/m4-gate-cli`
- **Dependencies:** merged M4-C and M4-D
- **Files:** read-only durable adapter, CLI parser/rendering/JSON, focused application/CLI tests
- **Acceptance:** completed runs only; exact Unix or synthetic source/qrel authentication plus
  outcome/config/lifecycle validation before verdict; stored ranked IDs are identified as trusted
  local evidence and metrics are recomputed; provider-free synthetic and live stored paths; no
  database creation/migration/recovery/write/mtime/sidecar change; stable `0/4/2/1` exits; no query
  text or secret values in outputs or retained exception graphs.

### M4-F — Guided frontend and required browser gate

- **Owner:** frontend worker
- **Branch:** `codex/m4-browser-gate`
- **Dependencies:** merged M4-B
- **Files:** generated-type capability client/UI, Playwright/axe harness, package lock, CI, frontend tests
- **Acceptance:** unconfigured Playground is actionable and emits no compare POST; configured state
  never claims remote health; synthetic desktop/mobile/history/dialog journey and axe rules pass;
  required Frontend CI installs `uv`, binds built assets to an explicit loopback API base, rejects
  every unexpected browser POST/provider factory, and stores only failure artifacts.

### M4-G — Interview QA and goal finalization

- **Owner:** root plus dedicated reviewer
- **Branch:** `codex/m4-finalization`
- **Dependencies:** merged M4-C, M4-E, and M4-F
- **Files:** README/operator runbook, `docs/progress.md`, bounded final verification fixes only
- **Acceptance:** clean provider-free seed/gate/dashboard workflow, full
  generated/static/unit/browser/privacy/artifact gates, and at most one independently pre-reviewed
  exact generated tiny live doctor/ingest/show/serve/compare/cleanup rehearsal. Cleanup crash tests
  cover immediately before delete, after provider deletion, during not-found verification, and
  after durable `not_found_verified` but before and after the fixed receipt move and held-inode
  wipe. One final reviewer-only merge and protected-main verification close the goal.

## Rollback and recovery boundaries

- M4-A is additive contract work and can be reverted before dependents. Existing health/search/run
  routes remain versioned.
- M4-B adds no migration revision. Doctor/capability inspection is read-only; `serve` still invokes
  the existing application migration/recovery path. Reverting M4-B removes guidance but not
  provider or SQLite state.
- M4-C never deletes merely because code is being rolled back. If its behavior is suspect, preserve
  the owner key, lock, authenticated receipt, and exact reviewed code revision before reverting;
  use cleanup only from a revision that independently passed the destructive-path review. Failure
  before the terminal fixed move retains the receipt for reviewed recovery rather than guessing a
  manual target. After that move, the namespace is already absence-verified; fixed absence and any
  ignored random quarantine are preserved for review without authorizing another remote action.
- M4-D/E read existing durable state only. A gate verdict is not persisted and rollback cannot
  change a run.
- M4-F adds development/CI dependencies and browser behavior only. Reverting it leaves backend,
  SQLite, and namespaces unchanged.
- No branch force-pushes reviewed history, mutates an existing namespace outside its exact task,
  deletes caller-supplied resources, or makes a credentialed call before independent safety review.

## Explicit non-goals

- Generic BEIR/JSONL/qrels customer import, arbitrary query-set sizes, arbitrary schema/field maps,
  user-supplied vectors, or browser uploads.
- Customer namespace branching, sharded/cross-region/cross-org copy, general namespace deletion, or
  cleanup of legacy/explicit/Unix/customer resources.
- Warm-cache experiments, cache-tier labels, ANN recall diagnostics, recall ground truth, external
  hosted rerankers, or index-profile editing/comparison.
- A gate over latency, nonterminal runs, arbitrary exported JSON, browser-supplied metrics, or
  untrusted summary values.
- API startup/provider preflight, automatic `.env` editing, automatic provider cost, automatic
  cleanup, or exposure of whether a particular secret value authenticates.
- Auth, hosted deployment, distributed workers, queues, Postgres, durable original-stage traces,
  or general config editing.

## Completion criteria

- [ ] M4-0 scope and architecture are independently reviewed and merged.
- [ ] API-exposed capability/error contracts are generated and drift-free; CLI-only gate contracts
  are exhaustively validated without being added to OpenAPI.
- [ ] Doctor/serve and guided Playground prevent the reproduced missing-namespace failure before a
  provider-capable request.
- [ ] A generated tiny namespace can be resumed, shown, and cleaned only through its authenticated
  fixed receipt; adversarial deletion attempts fail closed.
- [ ] `eval gate` passes/fails hand-calculated durable evidence with stable safe output and exits and
  zero provider construction.
- [ ] Required provider-free Playwright/axe CI reproduces the synthetic desktop/mobile workflow.
- [ ] Every M4 PR is independently reviewed and reviewer-only merged; protected-main Backend and
  Frontend checks pass after the reviewed finalization PR.
