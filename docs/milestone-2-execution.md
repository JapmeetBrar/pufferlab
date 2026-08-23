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

## Pinned dataset provenance and artifact policy

M2-D uses one acquisition chain and must fail closed if any identifier or checksum drifts:

| Layer | Exact source lock | Governing obligation |
|---|---|---|
| BEIR acquisition code and checksum registry | [`UKPLab/beir` (now `beir-cellar/beir`) commit `ef83d293`](https://github.com/beir-cellar/beir/tree/ef83d29307061c65d04b035b4f4e7c18bd8374af); [`download_dataset.py`](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/examples/dataset/download_dataset.py) and [`md5.csv`](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/examples/dataset/md5.csv) at that revision | Apache-2.0 applies to BEIR software/distribution tooling; retain its license/citation when code or documentation is reused. |
| BEIR archive | [Pinned archive URL](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip); registry MD5 `4e41456d7df8ee7760a7f866133bda78`; select only `cqadupstack/unix` after whole-archive verification | The implementation checks the published MD5 before extraction, computes a SHA-256 over the completed archive, records that SHA-256 in the checked-in source lock, and rejects any later byte drift. |
| Original CQADupStack tooling and dataset description | [`D1Doris/CQADupStack` commit `f73fc5b2`](https://github.com/D1Doris/CQADupStack/tree/f73fc5b2cc708c61d33bc76a3de93de0bf5bf584); [Apache-2.0 license](https://github.com/D1Doris/CQADupStack/blob/f73fc5b2cc708c61d33bc76a3de93de0bf5bf584/LICENSE.md); [paper DOI `10.1145/2838931.2838934`](https://doi.org/10.1145/2838931.2838934); documented source dump date `2014-09-26` | Apache-2.0 applies to the repository tooling, not automatically to the underlying post content. Retain the CQADupStack citation and audit any copied preprocessing logic for license/NOTICE duties. |
| Underlying Unix & Linux Stack Exchange posts | Post identity derived from the pinned 2014 dump and canonical URL `https://unix.stackexchange.com/questions/<post-id>` | [Stack Overflow's license chronology](https://stackoverflow.com/help/licensing) assigns CC BY-SA 2.5 before `2011-04-08` and CC BY-SA 3.0 from that date through the 2014 dump. Preserve source identity and link, identify Unix & Linux Stack Exchange, link both possible licenses, mark transformations, and do not relabel the content Apache-2.0 or CC BY-SA 4.0. |

M2-D checks in `docs/datasets/cqadupstack-unix.md`, a third-party `NOTICE`, and a machine-readable
source-lock manifest containing the exact URLs, repository revisions, upstream MD5, computed
SHA-256, archive member paths, expected counts, preprocessing version/hash, and citations. A
reviewer independently verifies that the lock matches the downloaded bytes before dataset code may
claim readiness.

The dataset adapter retains, at minimum, `source_dataset`, `source_subset`, original post/query ID,
canonical post URL, source site, dump date, applied transformation version, and content hash. It
must inspect and document whether the pinned archive contains enough author/date/revision metadata
for per-post attribution and license selection. Missing attribution fields are a visible audit
limitation, never silently invented. Any locally displayed or exported licensed text includes the
source link and dataset NOTICE; checked-in sample runs use synthetic content only.

The tracked-versus-ignored boundary is explicit:

| Tracked | Ignored and forbidden from Git history |
|---|---|
| Adapter/validation code and synthetic tests | Downloaded archives and partial downloads |
| Source-lock manifest with URLs, revisions, checksums, paths, counts, and hashes—but no post/query text | Extracted corpus, query text, official qrels, and upstream metadata files |
| CQADupStack/BEIR citations, Apache-2.0 notices, Stack Exchange attribution and CC BY-SA 2.5/3.0 links | Processed documents, query/qrel materializations, attribution sidecars containing upstream text or personal fields |
| Deterministic curated-50 manifest containing only source query IDs, PufferLab-authored tags/reasons, selection version, and hashes | Embeddings, raw vectors, model/dataset caches, SQLite databases, run exports, logs, and live evidence |
| Schema/config manifests and synthetic/golden fixtures | Any real title, body, query text, document snippet, author field, or copied upstream README/license body outside the audited NOTICE path |

Tests must prove the ignored paths with `git check-ignore`, reject text-bearing tracked manifests,
and scan the entire Git history for archive signatures and known sampled source text before review.

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
- **Acceptance:** the selected source is exactly the pinned acquisition chain above; the published
  MD5 and locally recorded SHA-256 match before extraction; the source lock and NOTICE pass an
  independent license/citation audit that distinguishes Apache-2.0 tooling from CC BY-SA 2.5/3.0
  post content; retained records carry the required source/transform fields and explicitly report
  unavailable attribution metadata. Processing is deterministic and content-addressed; official
  qrels map only to retained documents; the curated 50-query ID-only manifest is deterministic and
  contains PufferLab-authored exact-token, semantic, hybrid, and reranker tags/reasons. Automated
  tracked-versus-ignored inventory and history scans prove no real corpus/query/qrel text, upstream
  personal metadata, archive, processed row, embedding, vector, cache, database, export, or log is
  tracked. Interruption can resume stable-ID upserts without deleting a caller-supplied namespace.

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
- **Runbook:** [`milestone-2-live-verification.md`](milestone-2-live-verification.md)
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
7. The API key, request headers, raw vectors, licensed source text/qrels/metadata, archives,
   processed rows, local database, and real run exports never enter tracked files, logs,
   screenshots, or PR descriptions. Dataset source locks, ID-only selections, hashes, citations,
   and audited NOTICE material are the only tracked real-dataset artifacts.
8. Workers never merge their own PRs. Each branch follows the full changes-requested and re-review
   loop before the dedicated reviewer merges it.

## Standard validation

Every branch runs its focused tests plus the repository standard gates from
[`engineering-loop.md`](engineering-loop.md). Contract-changing branches additionally prove
OpenAPI and generated TypeScript are clean. Provider/dataset live checks are opt-in, create their
own random namespace, and confirm exact cleanup in `finally`.
