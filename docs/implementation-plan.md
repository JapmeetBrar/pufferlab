# PufferLab Detailed Implementation Plan

- **Objective:** deliver a thin, credible end-to-end version early, then add hybrid retrieval, regression analysis, and polish without breaking shared contracts.
- **Contract prerequisite:** [contracts.md](./contracts.md) must be reviewed and frozen before parallel work begins.
- **Confirmed environment:** a real turbopuffer account and API credentials are available; the implementation should use live, isolated namespaces from the first vertical slice.
- **No agents have been spawned as part of planning.**

## 1. Delivery milestones

### Milestone 0 — Contract freeze (half day)

- Review `docs/contracts.md`.
- Resolve names, score directions, metric cutoffs, error shapes, and module ownership.
- Create Pydantic skeletons and generate the first OpenAPI snapshot.

Exit: backend and frontend can build against the same checked-in API schema.

### Milestone 1 — Thin vertical slice (days 1–3)

One command ingests a tiny fixture into a throwaway turbopuffer namespace. The browser sends a query and compares BM25 with ANN. Results show ranks, scores, and client wall-clock duration. No eval engine yet.

Exit: a real browser → FastAPI → turbopuffer → browser loop works, with API key server-side.

### Milestone 2 — Useful core (days 4–6)

Add the real Unix dataset pack, server RRF, local reranker, judged evals, persistence, and largest-regression computation.

Exit: one CLI command runs the 50-query suite across four configs and persists a correct summary.

### Milestone 3 — Interview product (days 7–8)

Add eval dashboard, regression deep links, provenance reconstruction, robust empty/error/loading states, and demo queries selected from measured results.

Exit: the 3–5 minute demo works from a fresh documented setup.

### Milestone 4 — Hardening/polish (days 9–10)

Run live integration tests, browser smoke test, accessibility checks, ingestion recovery, cancellation, export, README, and recorded fallback screenshots/data.

Exit: a reviewer can clone, configure, ingest, run, inspect, and clean up without oral guidance.

## 2. Parallel execution map

```text
T0 Contracts/scaffold
    ├── T1 Platform/API/persistence ───────────────┐
    ├── T2 turbopuffer provider ── T5 Retrieval ──┼── T8 Vertical integration
    ├── T3 Dataset/ingestion ──────────────────────┤
    └── T4 Frontend shell ─────────────────────────┘

T6 Eval engine ───────────────┐
T7 CLI ───────────────────────┼── T9 Eval UI/deep links ── T10 Forensics ── T11 QA/demo
T5 Retrieval ─────────────────┘
```

Maximum useful initial parallelism after T0 is four workstreams: platform, turbopuffer provider, dataset, and frontend shell. Retrieval integration starts when T2 has a fake and real adapter contract. Eval math can run in parallel with the real dataset after the qrel contract is frozen.

## 3. Task cards

### T0 — Repository scaffold and contract freeze

**Objective:** establish executable project structure and one type source of truth before parallel work.

**Deliverable:** Python/web scaffolds, Pydantic contract skeletons, FastAPI OpenAPI snapshot, generated TypeScript API types, lint/test commands, `.env.example`.

**Dependencies:** project brief approval.

**Files/modules owned:**

- `pyproject.toml`
- `backend/pufferlab/contracts/**`
- `backend/pufferlab/main.py`
- `web/package.json`, `web/vite.config.ts`, `web/tsconfig*.json`
- `openapi/pufferlab-v1.json`
- `scripts/generate-api-client.*`
- `.env.example`

**Acceptance criteria:**

- `uv run pytest` and frontend unit command execute.
- `GET /api/v1/health` matches the versioned contract.
- OpenAPI generation is deterministic and a diff fails CI.
- Generated frontend types compile.
- No turbopuffer key is exposed in Vite environment variables.
- `.env.example` contains names/placeholders only, while `.env` and local data are ignored.

**Tests:** contract serialization snapshots, health-route test, generated-client typecheck.

**Parallel:** no; this is the gate for all other tasks.

---

### T1 — Persistence, API shell, and job lifecycle

**Objective:** persist immutable revisions and run status without coupling storage to search logic.

**Deliverable:** SQLAlchemy models/repositories, Alembic migration, API dependency wiring, in-process job registry, startup interruption recovery.

**Dependencies:** T0.

**Files/modules owned:**

- `backend/pufferlab/persistence/**`
- `backend/pufferlab/jobs/**`
- `backend/pufferlab/api/dependencies.py`
- `backend/pufferlab/api/routes/health.py`
- `backend/tests/persistence/**`

**Acceptance criteria:**

- Dataset/config/query-set/run revisions round-trip exactly.
- Completed runs cannot be mutated.
- Stale `running` runs become `interrupted` at startup.
- Cancellation preserves completed query outcomes.
- SQLite is created under configurable `PUFFERLAB_DATA_DIR`, ignored by git.

**Tests:** repository tests against temporary SQLite, state-transition property tests, startup recovery test.

**Parallel:** yes, with T2/T3/T4 after T0.

---

<a id="task-t2"></a>
### T2 — turbopuffer provider and live test harness

**Objective:** provide a narrow, fakeable adapter around the official async SDK.

**Deliverable:** client lifecycle/connection reuse, namespace metadata/readiness, write batching, query/multi-query, server RRF, warm hint/recall method stubs for P1, error mapping, live namespace fixture.

**Dependencies:** T0 contracts.

**Files/modules owned:**

- `backend/pufferlab/providers/turbopuffer.py`
- `backend/pufferlab/providers/types.py`
- `backend/pufferlab/providers/errors.py`
- `backend/tests/providers/test_turbopuffer_fake.py`
- `backend/tests/live/test_turbopuffer_live.py`

**Acceptance criteria:**

- One async client instance is reused.
- Neutral Filter AST converts to correct SDK tuples.
- Adapter returns typed rows plus client wall-clock time.
- Multi-query preserves subquery order and consistency.
- Provider/API errors are redacted and mapped to contract codes.
- Live test creates a random namespace, verifies write/read/hybrid, and cleans it in `finally`.

**Tests:** fake SDK unit tests for each query shape/error; opt-in live test marked `live`; filter conversion table tests.

**Parallel:** yes, with T1/T3/T4.

---

<a id="task-t3"></a>
### T3 — Dataset pack, preprocessing, embedding, and ingestion

**Objective:** turn CQADupStack Unix into a deterministic turbopuffer namespace plus local judged query sets.

**Deliverable:** downloader/loader, HTML/code-preserving text cleanup, UUIDv5 mapping, source attribution, BGE embedding batches, async write pipeline, readiness polling, demo/full query-set materialization, tiny offline fixture.

**Dependencies:** T0; uses the T2 adapter interface, but can start with a fake writer.

**Files/modules owned:**

- `backend/pufferlab/datasets/**`
- `backend/pufferlab/providers/embeddings.py`
- `backend/tests/datasets/**`
- `fixtures/tiny-corpus/**`
- `NOTICE-DATASETS.md`

**Acceptance criteria:**

- Same input produces the same corpus hash, IDs, and qrel mappings.
- Every positive qrel in a query set maps to an ingested document.
- Raw dataset and embedding cache are never committed.
- Schema spells out tokenizer/BM25/vector settings explicitly.
- Writes are batched and bounded-concurrent; interrupted ingestion can resume idempotently.
- Namespace is not marked ready until metadata/index checks pass.
- Attribution is retained via source URL/external ID and notice.

**Tests:** golden preprocessing, UUID determinism, qrel integrity, batch/resume behavior with fake writer, 20-document fixture ingestion.

**Parallel:** yes, with T1/T2/T4.

---

### T4 — Frontend shell and API integration primitives

**Objective:** establish a polished, accessible shell without inventing backend shapes.

**Deliverable:** routing, navigation, generated API client wrapper, query cache, design tokens, table/skeleton/error components, empty-state pages for runs/playground/configs/datasets.

**Dependencies:** T0 generated types.

**Files/modules owned:**

- `web/src/app/**`
- `web/src/api/**`
- `web/src/components/**`
- `web/src/styles/**`
- `web/src/test/**`

**Acceptance criteria:**

- No handwritten duplicate of API domain models.
- All four routes render on mobile and desktop.
- Loading, empty, API error, and retry states exist.
- Keyboard focus and color contrast meet basic WCAG AA expectations.

**Tests:** Vitest component tests, TypeScript strict build, route smoke test.

**Parallel:** yes, with T1/T2/T3.

---

<a id="task-t5"></a>
### T5 — Retrieval orchestration, hybrid, reranking, and provenance

**Objective:** execute all four seeded configurations with production-shaped calls and honest instrumentation.

**Deliverable:** config compiler, BM25/ANN/hybrid runners, server RRF, local cross-encoder adapter, debug raw-list probe, RRF reconstruction, overlap/rank movement, score semantics.

**Dependencies:** T0 and T2; T3 tiny fixture helpful.

**Files/modules owned:**

- `backend/pufferlab/retrieval/**`
- `backend/pufferlab/providers/rerankers.py`
- `backend/tests/retrieval/**`

**Acceptance criteria:**

- All modes validate required specs.
- Hybrid uses one same-snapshot multi-query for production-shaped retrieval.
- Debug probe timing is separate from production timing.
- Local reconstructed RRF implements documented rank/weight/constant formula.
- Score directions are correct; raw unlike scores are never merged.
- Reranker receives only configured depth and no vectors.
- Empty results and partial provider failures produce contract-compliant warnings/errors.

**Tests:** golden BM25/vector fake responses, RRF formula/order tests, rank-movement/overlap tests, reranker depth/cache tests, optional live server-vs-local-RRF parity test.

**Parallel:** starts when T2 interface is stable; can overlap late T3 and T1.

---

### T6 — Evaluation and regression engine

**Objective:** make quality calculations auditable, deterministic, and independent of services.

**Deliverable:** pure per-query NDCG@10/Recall@50/MRR@10, aggregation, percentiles, error coverage, paired deltas, regression/gain sorting, subset creation.

**Dependencies:** T0 qrel/result contracts only.

**Files/modules owned:**

- `backend/pufferlab/evals/**`
- `backend/tests/evals/**`

**Acceptance criteria:**

- Metric definitions match documented examples and a trusted reference calculation.
- Positive graded qrels affect NDCG; relevance for Recall/MRR is grade > 0.
- No-positive-qrel returns null/warning, not zero.
- Failed queries affect error/coverage, not quality averages.
- Percentiles include sample counts and use one documented interpolation method.
- Regression sorting is deterministic.

**Tests:** hand-calculated golden cases, empty/tie/duplicate/missing cases, randomized comparison against a reference library in dev tests.

**Parallel:** yes, immediately after T0; no provider dependency.

---

### T7 — CLI workflows

**Objective:** make PufferLab useful in a terminal and automate the exact demo setup.

**Deliverable:** `dataset ingest`, `config seed`, `eval run`, `eval export`, and `serve` commands using application services rather than HTTP duplication.

**Dependencies:** T1, T3, T5, T6 service interfaces.

**Files/modules owned:**

- `backend/pufferlab/cli/**`
- `backend/tests/cli/**`

**Acceptance criteria:**

- `--help` documents environment, cost-bearing actions, and namespace target.
- Ingest prints resolved namespace/schema/doc count before writing.
- Eval streams compact progress and exits nonzero on run failure.
- JSON export validates against the response contract.
- No cleanup command can delete an unowned namespace.

**Tests:** Typer CLI runner tests with fake services, JSON golden export, failure exit codes.

**Parallel:** implementation can start after service signatures stabilize; overlap UI work.

---

<a id="task-t8"></a>
### T8 — Thin vertical integration and compare API

**Objective:** achieve the first real end-to-end slice before building the eval dashboard.

**Deliverable:** datasets/configs/search API routes, seed command, Playground query form, BM25-vs-vector result columns, rank movement and latency.

**Dependencies:** T1–T5.

**Files/modules owned:**

- `backend/pufferlab/api/routes/datasets.py`
- `backend/pufferlab/api/routes/configs.py`
- `backend/pufferlab/api/routes/search.py`
- `web/src/features/playground/**`
- `web/src/features/datasets/**`
- `web/src/features/configs/**`

**Acceptance criteria:**

- Browser compares two configs against the live tiny namespace.
- The existing unscoped `/api/v1/configs` remains the tiny-fixture Playground catalog; later eval
  work adds a separate dataset-scoped catalog rather than changing this route.
- Query parameters can be copied as a stable URL.
- Results show rank, score kind/direction, relevance grade if known, and wall-clock measurement label.
- API key appears nowhere in browser network payloads or built assets.
- Debug-probe duration is visually separate.

**Tests:** API route tests with fake orchestrator, UI component tests, one mock browser flow.

**Parallel:** integration owner coordinates merges; avoid simultaneous edits to owned files.

---

### T9 — Eval runner, dashboard, and regression deep links

**Objective:** complete the “unit tests for search quality” experience.

**Deliverable:** eval routes/job execution/persistence, run list/detail UI, metric comparison table, progress polling, regression table, deep links into Playground.

**Dependencies:** T1, T5, T6, T8.

**Files/modules owned:**

- `backend/pufferlab/api/routes/eval_runs.py`
- `backend/pufferlab/jobs/eval_runner.py`
- `web/src/features/evals/**`
- integration tests for run lifecycle

**Acceptance criteria:**

- A 50-query four-config run completes and persists partial outcomes incrementally.
- Refreshing the page does not lose progress.
- Summary displays metric value and sample count.
- Regressions sort by NDCG delta and show supporting deltas/rank changes.
- Deep link restores query, judgments, baseline, candidate, and run context.
- Run responses carry their dataset revision and four ordered config labels. Dataset-backed config
  discovery uses `/api/v1/datasets/{dataset_version_id}/configs` without changing the existing
  Playground catalog.
- Regression deep links contain UUID-only `run`, `query`, `left`, and `right` parameters (optional
  `document`) and never include licensed query text.
- Cancellation stops new scheduling and leaves run inspectable.

**Tests:** job test with deterministic fake runner, API lifecycle tests, regression-table UI test, Playwright deep-link flow.

**Parallel:** backend job and frontend view can split after endpoint contracts freeze.

---

### T10 — Forensics and explainability guardrails

**Objective:** help answer “why did this move/miss?” using only observable evidence.

**Deliverable:** stage-membership drawer, candidate overlap, RRF contributions, lexical highlights, observability notice, `NOT_OBSERVABLE` states; targeted probe is P1 if schedule remains.

**Dependencies:** T5, T8, T9.

**Files/modules owned:**

- `backend/pufferlab/retrieval/forensics.py`
- `backend/pufferlab/api/routes/forensics.py` (P1)
- `web/src/features/playground/components/ForensicDrawer.tsx`
- forensics tests

**Acceptance criteria:**

- Every displayed explanation maps to a stored evidence item.
- Primary/probe evidence trace and timestamp bind to an actually returned source; probe ranks fit
  returned positive candidate counts, and stored outcomes never fabricate stage evidence.
- Internal plan/cache claims are absent.
- RRF contribution math is inspectable.
- Reranker shows only score/rank change, not generated rationale.
- A missing expected document results in a supported code or `not_observable`.

**Tests:** evidence-to-copy golden tests, forbidden-phrase test, each forensic code, empty/unknown case.

**Parallel:** after T9 feature surfaces stabilize; backend/UI halves can overlap.

---

<a id="task-t11"></a>
### T11 — QA, demo selection, documentation, and release

**Objective:** make the project reliable under interview conditions and understandable to an external engineer.

**Deliverable:** measured demo-query audit, full README/setup, architecture/observability docs, cleanup instructions, sample exported run, screenshots, test/lint/typecheck automation.

**Dependencies:** all P0 tasks.

**Files/modules owned:**

- `README.md`
- `docs/demo-script.md`
- `docs/observability.md`
- `docs/dataset-audit.md`
- `examples/sample-run.json`
- `.github/workflows/ci.yml`
- cross-cutting test fixtures only by coordination

**Acceptance criteria:**

- Fresh-clone setup succeeds from documented commands.
- Three-to-five-minute demo is rehearsed with real measured queries.
- Dataset audit meets exact-token/semantic/hybrid/reranker case coverage.
- No secrets or raw licensed corpus are tracked.
- CI passes Ruff, mypy, pytest, frontend lint/typecheck/unit tests, OpenAPI diff, and browser smoke.
- Live test cleanup is verified; manual cleanup path is documented.
- A sample run allows UI/demo fallback if network access fails.

**Tests:** fresh-directory setup rehearsal, `git grep` secret patterns, link check, full CI, optional live smoke.

**Parallel:** QA can begin early, but final acceptance is serial after integration.

## 4. Shared acceptance matrix

| Capability | Unit | Contract | Fake integration | Live tpuf | Browser |
|---|---:|---:|---:|---:|---:|
| Filter translation | ✓ |  | ✓ | ✓ |  |
| BM25/vector retrieval | ✓ | ✓ | ✓ | ✓ | ✓ |
| Server/local RRF parity | ✓ |  |  | ✓ |  |
| Reranker depth/rank movement | ✓ | ✓ | ✓ | optional | ✓ |
| NDCG/Recall/MRR | ✓ |  | ✓ |  | ✓ |
| Run persistence/recovery | ✓ | ✓ | ✓ |  | ✓ |
| Regression deep link |  | ✓ | ✓ | optional | ✓ |
| Observability guardrails | ✓ | ✓ | ✓ | optional | ✓ |
| Dataset ingestion/readiness | ✓ |  | ✓ | ✓ |  |

## 5. Integration rules for delegated work

1. T0 contracts merge first. No workstream adds a parallel TypeScript domain model.
2. Each workstream edits only its owned modules unless the integration owner explicitly coordinates a cross-cutting change.
3. Provider work returns normalized contracts; retrieval/eval/UI never import SDK response types.
4. Any contract change includes Pydantic tests, OpenAPI regeneration, generated TypeScript update, and an entry in `docs/contracts.md`.
5. Live tests are opt-in and create random names under a dedicated prefix. Cleanup runs in `finally`.
6. Never delete a namespace based on a user-supplied string alone.
7. Persist first, publish progress second; a UI refresh must not invent state.
8. Latency labels must say `client wall clock`; a second debug query is never included in production-shaped timing.
9. Every metric and forensic rule has a hand-calculated or evidence-linked golden test.
10. Merge at each milestone; do not allow long-lived branches to diverge past a contract change.

## 6. Recommended implementation order if working solo

1. T0 contracts/scaffold.
2. T2 fake + minimal live adapter.
3. T3 tiny fixture ingestion only.
4. T8 minimal Playground with BM25/vector.
5. T6 eval math.
6. T1 persistence/jobs.
7. T5 hybrid/reranker/provenance.
8. T9 eval run + regressions.
9. Finish T3 real Unix pack.
10. T7 CLI.
11. T10 guardrails/forensics.
12. T11 polish and measured demo selection.

This order deliberately produces visible, real turbopuffer behavior by day 2–3, before investing in the full dataset or dashboard.

## 7. P0 release definition

P0 is complete only when all of the following are true:

- A fresh user can set `TURBOPUFFER_API_KEY` and `TURBOPUFFER_REGION`, ingest the Unix dataset, and seed four configs.
- The 50-query eval runs to completion and reports correct NDCG@10, Recall@50, MRR@10, p50/p95, errors, and sample counts.
- At least one real regression deep-links into a Playground that displays observable stage membership and rank movement.
- No UI copy asserts an internal plan, causal model rationale, or cold/warm state.
- The entire 3–5 minute story can run live; a checked-in sample run supports a network-failure fallback.
- Tests, lint, typecheck, OpenAPI diff, and browser smoke pass.
