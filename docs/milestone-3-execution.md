# Milestone 3 Execution Plan

Milestone 3 turns the durable judged-evaluation core into the interview-facing PufferLab product.
The exit condition is a local browser workflow that lists persisted runs, explains progress and
aggregate trade-offs, ranks per-query regressions, restores one recorded query from a stable URL,
and distinguishes original stored evidence from a separately requested live replay.

The detailed product requirements remain in [`implementation-plan.md`](implementation-plan.md).
This file is the finite architecture, branch, ownership, dependency, and review plan for the active
goal.

## Context and constraints

- Milestone 2 already owns correct metric math, the 50-query/four-config runner, incremental SQLite
  outcomes, cancellation/recovery, canonical export, and dataset-bound retrieval configuration.
- The FastAPI application currently exposes only health, the fixture-bound config catalog, and
  interactive search comparison. It does not compose the SQLite evaluation runtime.
- The web application currently has one Playground view and no router, run dashboard, polling, or
  browser-flow test harness.
- A Milestone 2 success outcome deliberately stores ranked document UUIDs, metrics, timings,
  candidate counts, warnings, and a trace ID, but not query/document text, stage memberships,
  highlights, raw candidates, or provider bodies.
- The verified Milestone 2 namespace was deleted. Its stored run can prove ranks, judgments,
  metrics, and latency, but cannot honestly prove original stage membership after the fact.
- Query text and qrels are licensed local data. They may be returned to the local browser for a
  selected query with attribution, but must never enter tracked fixtures, OpenAPI examples, logs,
  screenshots, analytics, or Git history.
- P0 is a localhost, single-user, one-Uvicorn-worker application. CORS is not authentication;
  provider-writing endpoints remain bounded and must never accept a browser-supplied namespace.

## Architecture decision

**Status:** Proposed until M3-0 receives independent review.

Use SQLite-backed application read models as the authoritative dashboard source, and add a
long-lived API evaluation runtime that resolves every executable search through persisted dataset
and config identities. Recorded run pages perform no provider calls. A user may explicitly request
a cost-bearing live replay for one run/query; the server derives the query, graded qrels, dataset
namespace, and immutable configs and returns new evidence labeled `live_replay`.

### Options considered

| Option | Complexity | Correctness | Offline behavior | Decision |
|---|---:|---:|---:|---|
| Browser reads SQLite/export artifacts directly | Low initially | Poor: duplicates contracts and crosses server-only boundaries | Partial | Reject |
| Reuse the command-scoped CLI runtime directly in FastAPI | Medium | Poor lifecycle fit: command close semantics conflict with multi-request jobs | Good reads | Reject |
| SQLite read services plus a dedicated dataset-bound API runtime | Medium | Strong: one durable source, typed HTTP boundary, exact config binding | Strong | Adopt |
| Backfill original stage traces from ranked IDs or a new replay | Low | Invalid: invents evidence that was never stored | Misleading | Reject |

### Consequences

- Run list/detail, regressions, query detail, and export are provider-free and work for completed,
  partial, cancelled, interrupted, and failed runs.
- The old Milestone 2 run reports `original_stage_evidence_available=false`. Missing stage evidence
  produces `NOT_OBSERVABLE`; it is never inferred from final ranks or a trace ID.
- A replay is a new observation with its own timestamp and trace. It never mutates the immutable
  run/outcomes and is not substituted for recorded evidence.
- If the exact namespace is absent or not ready, replay returns the safe namespace-unavailable
  error while stored evidence remains usable.
- Immutable sanitized trace capture for future runs would require a separate reviewed migration and
  versioned codec; it is outside this P0 goal.

## Frozen P0 behavior

### Canonical evaluation scope

P0 remains the reviewed demo harness: one curated 50-query set and exactly four ordered canonical
configs (BM25 baseline, ANN, server RRF, local reranker). General query-set sizes, arbitrary config
editing, `POST /configs`, distributed workers, and multi-tenant authorization are deferred.

### HTTP surface

All top-level responses are versioned and use generated frontend types.

```text
GET  /api/v1/datasets
GET  /api/v1/datasets/{dataset_version_id}
GET  /api/v1/query-sets?dataset_version_id=...
GET  /api/v1/configs?dataset_version_id=...

POST /api/v1/eval-runs                         -> 202 queued run
GET  /api/v1/eval-runs?limit=...               -> newest first, bounded
GET  /api/v1/eval-runs/{run_id}
POST /api/v1/eval-runs/{run_id}/cancel         -> current durable run
GET  /api/v1/eval-runs/{run_id}/regressions
GET  /api/v1/eval-runs/{run_id}/queries/{query_id}
GET  /api/v1/eval-runs/{run_id}/export
POST /api/v1/eval-runs/{run_id}/queries/{query_id}/replay
```

- `POST /eval-runs` accepts only the canonical 50-by-four suite, persists `queued`, schedules work
  independently of request cancellation, and returns 202. A duplicate equivalent active suite
  returns `409 run_conflict`; a terminal run does not block an intentional new run.
- The API enforces a bounded number of active runs and retains the existing per-run concurrency
  bound. Multiple Uvicorn workers are unsupported until durable task claims/leases exist.
- Cancel is idempotent: queued/running work is cooperatively stopped and drained; a terminal run is
  returned unchanged. Cancel never deletes a run or provider namespace.
- Regression requests require one candidate config from the run, an order of `regressions` or
  `gains`, and a bounded limit. The response contains only paired quality rows plus explicit counts
  for every excluded pair status. Missing/failed queries are never converted to zero.
- Query detail returns the exact judged query, ordered persisted config summaries, one typed durable
  outcome per available config, relevant-document rank changes, attribution, and evidence
  availability. It performs no provider call.
- Replay accepts config IDs only. The server loads the run/query, preserves exact graded qrels,
  resolves the dataset and config hashes, and never accepts query text, expected IDs, or namespace
  from the browser. The response is labeled `evidence_origin=live_replay`, includes `observed_at`, a
  new trace ID, and an explicit non-original-evidence notice.

### Browser routes and state

```text
/                                      existing Playground
/runs                                  run history and create-run controls
/runs/:runId                           durable progress, metrics, regressions
/runs/:runId/queries/:queryId          recorded query and outcomes
/playground?run=...&query=...&left=...&right=...&document=...
```

- Real links and browser history restore every view; route changes focus the page heading.
- The dashboard polls only queued/running runs, stops for terminal states and on unmount, and treats
  `completed_queries` as 0–50 fully durable query groups—not 0–200 attempts.
- Partial runs show progress and durable outcomes but never present absent final summaries as final
  metrics. Metric values always show sample counts; `null` renders as unavailable, never zero.
- Regression ordering, candidate selection, and selected document live in the URL.
- Opening a deep link never triggers provider work. “Live replay” is an explicit action with cost
  and evidence-origin copy.
- The forensic drawer uses only typed returned evidence. It may show observed stages/scores,
  highlights, rank movement, overlap, timings, and hand-calculated RRF contributions when all inputs
  are present. It never claims provider plan, cache state, causal ordering, or generated reranker
  rationale.

## Dependency and branch graph

```text
M3-0 codex/m3-plan
  |
  v
M3-A codex/m3-contracts
  |
  v
M3-B codex/m3-eval-read-api
  |\
  | +--> M3-C codex/m3-run-dashboard
  |
  +----> M3-D codex/m3-eval-control-runtime
             |
             +---- M3-C + M3-D ----> M3-E codex/m3-query-forensics
                                          |
                                          v
                                  M3-F codex/m3-finalization
```

M3-C may build against the reviewed generated M3-B contract while M3-D implements the provider and
job-control runtime. M3-E begins only after both are merged. M3-F is the single goal-finalization
PR; its own merge/check record remains canonical in GitHub.

## Rollback and migration boundaries

- M3-A changes versioned contracts only. It adds no route, provider call, database migration, or
  browser behavior and can be reverted as one contract review unit before dependents merge.
- M3-B adds provider-free reads and HTTP projections over the existing SQLite schema. Removing its
  routes does not alter durable rows or namespaces.
- M3-C is a browser-only consumer of generated M3-B contracts. Its routing/dashboard shell can be
  reverted without changing backend state.
- M3-D reuses the existing run and outcome tables; no trace table or destructive migration is
  authorized. On rollback or process failure, the existing startup recovery marks orphaned running
  work interrupted. No rollback path deletes a namespace.
- M3-E live replay is request-scoped and non-persistent. Reverting it leaves recorded runs and
  outcomes byte-for-byte unchanged.
- Any future durable trace capture, config editing, namespace branching, or job lease requires a
  separate migration/ADR/review unit outside this goal.

## Review units

### M3-0 — Architecture and contract audit

- **Owner:** root orchestrator
- **Branch:** `codex/m3-plan`
- **Files:** this plan and `docs/progress.md`
- **Acceptance:** Milestone 2 canonical completion is current; original-versus-replay evidence is
  explicit; HTTP/UI shapes, contradictions, owners, dependencies, rollback boundaries, and finite
  completion criteria are independently reviewed before implementation branches start.

### M3-A — Contract freeze

- **Owner:** evaluation contract worker
- **Branch:** `codex/m3-contracts`
- **Dependencies:** merged M3-0
- **Files:** evaluation/catalog/forensic Pydantic contracts, contract documentation/tests
- **Acceptance:** freeze versioned catalog, run-list/detail, regression coverage, query-detail,
  cancel, export, and live-replay envelopes; preserve canonical 50-by-four validation; add strict
  size/shape validation to forensic evidence; correct RunEnvironment documentation; no provider,
  persistence, FastAPI, or handwritten TypeScript domain models.

### M3-B — Durable read models and eval HTTP surface

- **Owner:** evaluation application worker
- **Branch:** `codex/m3-eval-read-api`
- **Dependencies:** merged M3-A
- **Files:** repository read methods, provider-free evaluation views, eval/catalog routes and fake
  route tests, OpenAPI and generated TypeScript
- **Acceptance:** bounded deterministic run/catalog listing; strict durable payload decoding;
  regressions/gains use the existing paired engine and exact qrels; explicit excluded-pair coverage;
  relevant rank changes are exact through rank 50; query detail is run-scoped and provider-free;
  all six run statuses export/read; API errors are direct and redacted; generated artifacts do not
  drift.

### M3-C — Run dashboard

- **Owner:** frontend worker
- **Branch:** `codex/m3-run-dashboard`
- **Dependencies:** merged M3-B
- **Files:** application routing/navigation, API client, `web/src/features/evals/**`, frontend tests
- **Acceptance:** `/runs` and `/runs/:id` cover loading/empty/error/retry/not-found and all lifecycle
  states; active-only polling stops correctly; metrics retain value/sample count/null semantics;
  regression ordering and excluded coverage are visible; status is not color-only; semantic tables,
  progress announcements, route focus, 320/390 px layouts, keyboard access, generated types, and
  production build pass.

### M3-D — Dataset-bound execution and control runtime

- **Owner:** root integration worker
- **Branch:** `codex/m3-eval-control-runtime`
- **Dependencies:** merged M3-B
- **Files:** API application runtime/dependencies, evaluation lifecycle integration, focused API/job
  tests
- **Acceptance:** default app migrates SQLite and recovers once; POST persists then schedules at
  202; exact dataset/config resolution replaces fixture binding; active/global concurrency and
  duplicate-run bounds hold; request cancellation does not cancel the job; cancel is idempotent and
  preserves evidence; fatal errors are safe; shutdown drains jobs, closes search runtimes, then
  disposes SQLite; one-worker constraint is enforced/documented.

### M3-E — Regression deep links and observable query forensics

- **Owner:** forensics worker plus frontend worker with disjoint backend/web ownership
- **Branch:** `codex/m3-query-forensics`
- **Dependencies:** merged M3-C and M3-D
- **Files:** pure forensic rules, replay route, query detail/Playground integration, drawer and tests
- **Acceptance:** recorded detail never calls the provider; old M2 evidence is honestly unavailable;
  replay binds the run dataset/configs and is visibly a new observation; absent namespaces degrade
  to stored evidence plus `NOT_OBSERVABLE`; RRF contribution inputs/math are inspectable; reranker
  copy is score/rank only; stable deep links restore run/query/config/document state; drawer keyboard
  focus/Escape/return and forbidden-claim goldens pass.

### M3-F — Interview QA and finalization

- **Owner:** root orchestrator plus dedicated reviewer
- **Branch:** `codex/m3-finalization`
- **Dependencies:** all delivery PRs merged and protected-main checks green
- **Files:** README, demo/observability documentation, synthetic fallback artifacts only, progress
  ledger, browser smoke configuration where required
- **Acceptance:** fresh setup works; local stored-run dashboard and a deliberately authorized live
  replay are verified without exposing licensed text or secrets; a regression deep link survives
  refresh/back navigation at desktop and mobile widths; offline synthetic fallback works; full local
  and GitHub gates pass; one independent reviewer merges the exact final head and verifies protected
  `main`.

## Cross-cutting invariants

1. Every branch follows the independent review/fix/re-review/reviewer-only merge loop.
2. The browser imports domain types only from generated OpenAPI TypeScript.
3. Dashboard reads and deep-link restoration never perform provider work implicitly.
4. Config IDs resolve to one persisted dataset revision; browser input never selects a namespace.
5. Original stored evidence and live replay evidence are never merged or relabeled.
6. Query/document text, qrels, credentials, vectors, provider bodies, database/export files, and
   live screenshots remain outside Git and PR/check output.
7. `EvidenceItem` values are allowlisted and bounded; arbitrary provider payloads never cross the
   API boundary.
8. Latency is labeled observed client wall clock with sample count, never a service benchmark.
9. Failures and missing pairs remain explicit coverage states, never zero-valued quality.
10. UI explanations map to typed evidence; unsupported causes are `NOT_OBSERVABLE`.

## Standard validation

Every branch runs its focused tests plus the repository gates in
[`engineering-loop.md`](engineering-loop.md). Contract-changing branches regenerate and verify
OpenAPI/TypeScript. Browser branches add deterministic fake flows and production-build exposure
scans. Provider replay is opt-in, explicit, dataset-bound, cost-bearing, and never required by
normal CI.
