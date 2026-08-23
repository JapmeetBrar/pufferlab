# Milestone 2 Execution Plan

Milestone 2 turns the proven BM25-versus-vector vertical slice into a durable judged-evaluation
core. The exit condition is one CLI command that runs the curated 50-query Unix suite across BM25,
ANN, server RRF, and server RRF plus a local reranker, then persists a contract-valid summary and
per-query evidence.

The detailed product requirements remain in [`implementation-plan.md`](implementation-plan.md).
This file defines the finite branch, ownership, dependency, and review plan for the active goal.

## Scope and boundaries

Milestone 2 includes:

- pure, deterministic NDCG@10, Recall@50, MRR@10, latency, coverage, and paired-delta math;
- SQLite persistence for immutable dataset, configuration, query-set, judgment, and run revisions;
- an in-process run lifecycle with incremental outcomes, cancellation, and startup recovery;
- deterministic CQADupStack Unix preprocessing, attribution, a curated 50-query set, and ignored
  local corpus/embedding caches;
- same-snapshot BM25 plus ANN multi-query, turbopuffer server RRF, observable local RRF
  reconstruction, and a pinned local cross-encoder reranker;
- CLI configuration seeding, evaluation execution, progress, and contract-valid JSON export;
- one credentialed 50-query/four-configuration run with exact namespace cleanup and sanitized
  evidence.

Milestone 2 does not include the eval dashboard, regression deep-link UI, or forensic drawer. Those
remain Milestone 3. It also does not add accounts, hosted services, a distributed queue, inferred
provider internals, or raw licensed corpus data to Git.

## Dependency and branch graph

```text
M2-0 codex/m2-plan (this coordination PR)
  ├── M2-A codex/m2-eval-engine
  ├── M2-B codex/m2-persistence-jobs
  ├── M2-C codex/m2-hybrid-retrieval
  └── M2-D codex/m2-unix-dataset
             │
             └──────────────┐
M2-A + M2-B + M2-C + M2-D ─┴── M2-E codex/m2-eval-cli
                                      │
                                      └── M2-F codex/m2-live-finalization
```

M2-A through M2-D own disjoint modules and may proceed in parallel from the reviewed M2-0 merge.
Each task merges independently through the dedicated reviewer. M2-E starts only after all four
dependencies are on protected `main`; M2-F is the single goal-finalization PR.

## Review units

### M2-0 — Coordination and contract audit

- **Owner:** root orchestrator
- **Branch:** `codex/m2-plan`
- **Files:** `docs/milestone-2-execution.md`, `docs/progress.md`
- **Acceptance:** prior Milestone 1 canonical evidence is current; every Milestone 2 review unit has
  one owner, branch, dependency boundary, acceptance definition, and merge order; no future review
  or merge is predicted.

### M2-A — Pure evaluation engine

- **Owner:** eval worker
- **Branch:** `codex/m2-eval-engine`
- **Files:** `backend/pufferlab/evals/**`, `backend/tests/evals/**`; existing eval contracts only when
  a reviewed contract correction is unavoidable.
- **Acceptance:** hand-calculated and trusted-reference tests cover NDCG@10, Recall@50, MRR@10,
  p50/p95, error coverage, sample counts, paired deltas, deterministic regression ordering,
  duplicate/tie cases, and no-positive-qrel null/warning behavior. The module imports no provider,
  FastAPI, SQLAlchemy, or frontend code.

### M2-B — SQLite persistence and run lifecycle

- **Owner:** persistence worker
- **Branch:** `codex/m2-persistence-jobs`
- **Files:** `backend/pufferlab/persistence/**`, `backend/pufferlab/jobs/**`, migrations, persistence
  tests, and dependency declarations.
- **Acceptance:** immutable revisions and per-query outcomes round-trip exactly; completed runs
  reject mutation; incremental outcomes survive refresh/restart; stale running jobs become
  interrupted at startup; cancellation stops new scheduling without erasing completed outcomes;
  SQLite lives under ignored `PUFFERLAB_DATA_DIR`; the initial migration upgrades and downgrades a
  clean database.

### M2-C — Hybrid retrieval and local reranking

- **Owner:** retrieval worker
- **Branch:** `codex/m2-hybrid-retrieval`
- **Files:** `backend/pufferlab/providers/**`, `backend/pufferlab/retrieval/**`, focused provider and
  retrieval tests.
- **Acceptance:** production hybrid uses one same-snapshot multi-query with server-side weighted
  RRF; debug raw-list probes are separate and their timings cannot contaminate production timing;
  local RRF reconstruction follows the documented rank/weight/constant formula; the pinned
  `cross-encoder/ms-marco-MiniLM-L-6-v2` adapter receives only configured-depth text candidates and
  returns no invented rationale; scores, provenance, partial failures, and stage timings remain
  contract-valid. A fake SDK covers every shape; live RRF parity remains opt-in and self-cleaning.

### M2-D — CQADupStack Unix dataset foundation

- **Owner:** dataset worker
- **Branch:** `codex/m2-unix-dataset`
- **Files:** deterministic dataset adapter/manifests, attribution and audit documentation, dataset
  tests, and CLI ingestion service extensions that do not overlap M2-E command routing.
- **Acceptance:** processing is deterministic and content-addressed; source IDs and URLs preserve
  Stack Exchange attribution; official qrels map only to retained documents; the curated 50-query
  set is deterministic and contains exact-token, semantic, hybrid, and reranker cases; raw corpus,
  generated embeddings, and downloaded archives stay ignored; interruption can resume stable-ID
  upserts without deleting a caller-supplied namespace.

### M2-E — Evaluation application service and CLI

- **Owner:** integration worker
- **Branch:** `codex/m2-eval-cli`
- **Dependencies:** merged M2-A through M2-D
- **Files:** application orchestration, CLI commands/tests, seed manifests, and only the API contract
  wiring required for run execution/export.
- **Acceptance:** `config seed` creates four immutable configs; `eval run` executes the curated 50
  queries across BM25, ANN, hybrid RRF, and hybrid rerank with bounded concurrency and incremental
  persistence; failures affect coverage rather than quality means; compact progress and nonzero
  failure exits are correct; `eval export` emits contract-valid JSON; rerunning an immutable
  completed run cannot mutate it.

### M2-F — Live execution and goal finalization

- **Owner:** root orchestrator plus dedicated reviewer
- **Branch:** `codex/m2-live-finalization`
- **Dependencies:** merged M2-E and all protected-main checks green
- **Acceptance:** a fresh isolated namespace ingests the Unix pack, the 50-query suite completes
  across four configurations, the persisted summary independently recomputes from stored outcomes,
  secrets/raw vectors/corpus data remain outside Git and browser artifacts, exact namespace cleanup
  reaches provider `NOT_FOUND`, full local and GitHub gates pass, and the dedicated reviewer merges
  the exact final head and verifies protected `main`.

## Cross-cutting invariants

1. Existing v1 contracts remain the source of truth. Any unavoidable contract change includes
   Pydantic tests, OpenAPI regeneration, generated TypeScript regeneration, and drift checks.
2. Evaluation math remains pure. Service, persistence, and provider adapters convert at module
   boundaries rather than importing across layers.
3. Persist an outcome before publishing progress. A restart or cancellation never invents or loses
   a completed query result.
4. Quality averages exclude failed queries and include sample counts; failures remain visible in
   coverage and error metrics.
5. All latency is labeled client wall clock. Debug probes and local reranking have separate stages.
6. Only internally generated, pattern-validated namespace identities may be deleted automatically.
7. The API key, request headers, raw vectors, licensed raw corpus, and local database never enter
   tracked files, logs, screenshots, exports, or PR descriptions.
8. Workers never merge their own PRs. Each branch follows the full changes-requested and re-review
   loop before the dedicated reviewer merges it.

## Standard validation

Every branch runs its focused tests plus the repository standard gates from
[`engineering-loop.md`](engineering-loop.md). Contract-changing branches additionally prove
OpenAPI and generated TypeScript are clean. Provider/dataset live checks are opt-in, create their
own random namespace, and confirm exact cleanup in `finally`.
