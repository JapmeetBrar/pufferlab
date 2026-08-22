# PufferLab: Project Decision + Implementation Brief

- **Status:** Proposed
- **Research date:** 2026-08-22
- **Working assumption:** one engineer, roughly 8–10 focused working days for a polished local demo
- **Confirmed resource:** access to a real turbopuffer account and API credentials for live development, ingestion, evaluation, and integration testing
- **Decision:** Build PufferLab, but position it as a **customer search-evaluation and query-forensics harness**, not primarily as a playground.

## Executive decision

PufferLab is the right project after one important reframing:

> PufferLab turns a customer's representative queries and relevance judgments into repeatable search-quality tests, identifies the queries that regressed, and opens each failure in an instrumented retrieval debugger.

The core loop is:

> **Configure → Evaluate → Find regressions → Inspect observable stages → Change one variable → Re-run**

That is more useful and more aligned with a turbopuffer Deployed Engineer than a generic search UI. The current role explicitly emphasizes concrete demos and POCs, guiding evaluations, solving hard customer problems, data/schema analysis, and performance benchmarking. PufferLab demonstrates all of those in one coherent artifact ([role posting](https://jobs.ashbyhq.com/turbopuffer/c89ab81b-1fb1-4b6b-8ffb-9926adeeb0f9/)).

The playground remains essential, but as the failure-analysis surface attached to an eval run. This avoids building a handsome toy that cannot answer whether a change made search better.

## 1. Key findings from current turbopuffer research

### Architecture and why it matters to this project

- Durable state lives in object storage; query nodes are stateless and use memory/NVMe as caches. Any query node can serve any namespace, while repeat requests are routed for cache locality. The public architecture page currently gives about **874 ms p50 for a first query** and **14 ms p50 once cached** on a 1M-document namespace. The exact latency is workload-dependent, so PufferLab must measure its workload rather than repeat those numbers as a promise ([architecture](https://turbopuffer.com/docs/architecture), [concepts](https://turbopuffer.com/docs/concepts)).
- A namespace has its own object-storage prefix and WAL. Writes are durable before acknowledgment, concurrent writes are group-committed, and committed-but-unindexed WAL data remains searchable by exhaustive scan. This explains the high-throughput/higher-single-write-latency trade-off and why the ingestion CLI should batch and run concurrently ([architecture](https://turbopuffer.com/docs/architecture), [ingestion](https://turbopuffer.com/docs/ingestion)).
- Vector ANN uses an incrementally maintained, centroid-based SPFresh-inspired index. That access pattern minimizes dependent object-storage round trips and supports online split/merge behavior, unlike a graph traversal optimized for RAM/SSD. This is the architecture story the demo should make visible: PufferLab sends representative first-stage workloads to an engine designed for large, bursty, multi-tenant corpora rather than pretending it is a local HNSW benchmark ([architecture](https://turbopuffer.com/docs/architecture), [native filtering post](https://turbopuffer.com/blog/native-filtering)).
- Strong consistency is the default. It searches unindexed writes and refreshes cache, but incurs an object-storage metadata-check floor. Eventual consistency can reduce warm latency at the cost of bounded-but-real staleness. An eval run must record this setting because results and latency are otherwise not comparable ([query API](https://turbopuffer.com/docs/query), [tradeoffs](https://turbopuffer.com/docs/tradeoffs)).
- Namespaces are the natural tenancy and query boundary. turbopuffer recommends making them as small as possible without routinely querying more than one. PufferLab should use one namespace per immutable dataset/index profile, not one namespace per experiment or query ([performance guide](https://turbopuffer.com/docs/performance), [tradeoffs](https://turbopuffer.com/docs/tradeoffs)).

### Retrieval capabilities relevant to PufferLab

- The v2 query API supports ANN, exact filtered kNN, BM25, sparse vectors, filters, aggregations, and multi-query. A multi-query can run up to 16 subqueries against the same consistent snapshot, which is the correct primitive for hybrid retrieval ([query API](https://turbopuffer.com/docs/query)).
- BM25 is native and object-storage optimized. Full-text fields have explicit tokenizer, case, language, stemming, stopword, ASCII-folding, `k1`, `b`, and `k3` settings. Defaults can change for newly created namespaces, so PufferLab must persist an **explicit schema/index profile** to keep regression runs reproducible ([write/schema API](https://turbopuffer.com/docs/write), [FTS guide](https://turbopuffer.com/docs/fts)).
- Hybrid search is intentionally a first-stage pipeline: vector and BM25 narrow millions of rows to tens or hundreds, then application code fuses and optionally reranks. turbopuffer also provides server-side weighted reciprocal-rank fusion (RRF), with `rank_constant` defaulting to 60 ([hybrid guide](https://turbopuffer.com/docs/hybrid), [RRF API](https://turbopuffer.com/docs/query)).
- Filters are not a side feature. turbopuffer combines filter indexes with ANN natively to preserve recall better than naive pre- or post-filtering. PufferLab should include filter cases in its judged set and report their effect, but it cannot claim access to the server's internal query plan ([native filtering post](https://turbopuffer.com/blog/native-filtering), [query API](https://turbopuffer.com/docs/query)).
- Returned rows expose the overall `$dist` score. `compute_attributes` can add observable clause-level BM25 scores, vector distance, extra reranker features, and highlighted matching passages. These are legitimate debugging signals ([query API](https://turbopuffer.com/docs/query), [FTS highlighting](https://turbopuffer.com/docs/fts)).
- turbopuffer recommends keeping first-stage ranking simple, retrieving roughly 100–1,000 candidates, and applying expensive logic in a second stage. That candidate-count/quality/latency trade-off should be one of PufferLab's main knobs ([performance guide](https://turbopuffer.com/docs/performance), [rank-by-attribute post](https://turbopuffer.com/blog/rank-by-attribute)).
- Native embeddings exist but are still presented as opt-in/beta in the current changelog. They are worth an adapter later, but the P0 demo should not depend on beta access. Use explicit client-side embeddings for reproducibility ([embedding guide](https://turbopuffer.com/docs/embedding), [roadmap/changelog](https://turbopuffer.com/docs/roadmap)).

### Evaluation, cloning, and cache behavior

- turbopuffer's testing guidance recommends using the production service for end-to-end tests, random throwaway namespaces for isolated tests, and copy-on-write namespace branches for tests on real data ([testing guide](https://turbopuffer.com/docs/testing)).
- Namespace branching is constant-time, independent, copy-on-write, and explicitly recommended for test pipelines and snapshots. It is a valuable P1 feature for evaluating a customer's real namespace without mutating it ([branching guide](https://turbopuffer.com/docs/branching)).
- The public `_debug/recall` endpoint compares sampled ANN results against exhaustive vector search. That measures ANN index recall, not semantic relevance or hybrid quality. PufferLab should show it as a separate diagnostic and never conflate it with judged Recall@K ([recall API](https://turbopuffer.com/docs/recall)).
- The public warm-cache endpoint is a **hint**, not a cache-state inspection API. It can prepare a namespace for latency-sensitive work, but there is no public “evict this namespace and prove the next request is cold” control. Therefore PufferLab can report observed end-to-end latency and expose a warm hint, but it must not label a request “cold” based only on being first in a sequence ([warm-cache API](https://turbopuffer.com/docs/warm-cache), [tradeoffs](https://turbopuffer.com/docs/tradeoffs)).

### Honest observability boundary

PufferLab can truthfully show:

- returned document IDs, ranks, `$dist`, and requested attributes;
- BM25 and vector-distance values computed for returned rows;
- lexical and vector candidate-list membership;
- RRF contributions when reconstructed from observed ranks using the documented formula;
- external reranker scores and before/after rank movement;
- exact client wall-clock duration for embedding, turbopuffer requests, fusion, reranking, and total handling;
- exact evaluation judgments and metric calculations;
- whether a known document's locally available attributes pass the configured filter;
- counterfactual results from a clearly labeled second query, such as “same request without filter.”

PufferLab cannot truthfully show:

- turbopuffer's internal query plan, centroid traversal, postings visited, or cache tier used;
- server compute time separate from network time unless the API later exposes it;
- a causal explanation of an embedding or cross-encoder score;
- a guaranteed cold/warm classification from request order;
- “filtered out by the server before ANN” unless this is directly exposed in a future API.

The UI must call its artifacts **observations**, **stage membership**, **score contribution**, and **counterfactual probe**, not “the model's reasoning” or “query plan.”

## 2. Is PufferLab the right project?

### Options considered

Weights: usefulness 20%, Deployed Engineering relevance 20%, turbopuffer specificity 15%, technical depth 15%, demo quality 15%, feasibility 10%, open-source usefulness 5%. Scores are 1–5.

| Option | Useful | DE role | tpuf-specific | Depth | Demo | Feasible | OSS | Weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PufferLab: eval + query forensics** | 5 | 5 | 5 | 5 | 5 | 4 | 5 | **4.90** |
| CLI-first customer workload benchmark | 5 | 5 | 4 | 4 | 3 | 5 | 5 | 4.40 |
| Search migration toolkit | 4 | 5 | 5 | 4 | 4 | 2 | 4 | 4.15 |
| Multi-tenant search reference kit | 4 | 4 | 5 | 4 | 4 | 3 | 5 | 4.10 |
| Cold/warm profiler | 3 | 4 | 5 | 4 | 4 | 4 | 3 | 3.90 |

### Decision and consequences

Choose PufferLab, with the workload benchmark embedded inside it as the eval runner and CLI.

Why it wins:

1. It covers the real evaluation work a Deployed Engineer does while still producing a visual, memorable demo.
2. It is specifically shaped around turbopuffer's multi-query, BM25, ANN, filters, server RRF, first-/second-stage boundary, namespaces, branches, recall endpoint, and cache hint.
3. It produces a reusable artifact: customers can import their own corpus, query set, and qrels after the demo dataset.
4. It has a clean scope boundary. A migration toolkit quickly becomes connector/auth/schema-edge-case work; a multi-tenant reference app becomes generic infrastructure; a cold/warm profiler cannot reliably force the state it claims to benchmark.

What becomes harder: the project needs rigorous contracts and deterministic evaluation, not only UI polish. That is the right difficulty for this interview.

## 3. Exact user and problem

### Primary user

A turbopuffer Deployed Engineer running a customer POC or helping a customer change an existing retrieval pipeline.

### Secondary user

A search engineer who has a representative corpus plus 20–1,000 judged queries and needs to decide whether a proposed retrieval configuration is safe to ship.

### Job to be done

> “Given my real documents, filters, and relevance judgments, tell me whether this configuration is better than the baseline, identify the queries it hurts, and give me enough observable evidence to decide what to change next.”

### Non-functional requirements

- Reproducible: every result records dataset/index/config/query-set revisions.
- Honest: no invented explainability or unverified cache labels.
- Small/local: one process for API/jobs, one React app, SQLite, no queue or auth.
- Useful at demo scale: ~50K documents, 50-query fast set, and up to ~1K queries for a longer run.
- Safe: API key remains server-side; demo never mutates an existing customer namespace.
- Inspectable: run artifacts can be exported as JSON.

### Credential and account setup

The project can use the real turbopuffer service from the first vertical slice. Credentials are supplied only through server-side environment variables:

```bash
TURBOPUFFER_API_KEY=...
TURBOPUFFER_REGION=gcp-us-central1
```

They belong in an ignored local `.env` file or the launching shell, never in frontend environment variables, fixtures, exported runs, logs, screenshots, or git history. Development and live tests should use a dedicated namespace prefix such as `pufferlab-dev--` and random suffixes; cleanup must target only namespaces whose ownership PufferLab recorded.

## 4. Core product experience

### A. Eval dashboard is the home screen

1. Select an immutable dataset version and judged query set.
2. Select baseline and candidate retrieval configs.
3. Run the suite.
4. Compare NDCG@10, Recall@50, MRR@10, errors, p50, and p95.
5. See the largest per-query regressions, not only aggregate averages.
6. Click a regression to open the same query, judgments, baseline, and candidate in the Playground.

### B. Playground is a query-forensics view

The view compares two to four configurations and displays:

- ranked result columns and rank deltas;
- BM25, vector, hybrid, and reranker stage-membership badges;
- candidate-set overlap and unique candidates;
- observable BM25/vector/RRF/reranker scores with score direction;
- end-to-end and client-stage wall-clock latency;
- relevant-document judgments;
- highlighted lexical passages where supported;
- an explicit “Not observable from public API” label for internal plan/cache details.

Selecting a document opens a forensic drawer. P0 shows stage membership and rank/score movement. P1 adds targeted and counterfactual probes.

### C. Change one variable, then re-run

Seed four configs:

1. `bm25-title-body`: weighted BM25 over title and body.
2. `bge-ann`: BGE small cosine ANN.
3. `hybrid-rrf`: BM25 + ANN in one snapshot, server-side RRF.
4. `hybrid-rerank`: same candidates, followed by a local cross-encoder.

The demo changes one high-leverage parameter—candidate count, RRF weight, or reranker depth—then re-runs only the regression subset before the full suite.

## 5. Scope

### P0 — must have

- CQADupStack Unix dataset pack with deterministic preprocessing, attribution, full corpus ingestion, official qrels, and a curated 50-query demo set.
- Explicit turbopuffer schema/index profile and batched async ingestion.
- Four seeded retrieval configurations: BM25, vector, hybrid RRF, hybrid + local reranker.
- Config comparison Playground with rank movement, stage membership, observable scores, candidate overlap, judgments, and client-measured stage latency.
- Evaluation engine: NDCG@10, Recall@50, MRR@10, p50/p95 wall-clock latency, error rate, per-query metrics.
- Baseline/candidate regression table and one-click deep link into Playground.
- Persisted datasets, configs, query sets, judgments, runs, per-query outcomes, and traces in SQLite.
- CLI for dataset ingestion, eval execution, run export, and local server startup.
- FastAPI + generated OpenAPI; React/TypeScript UI.
- Unit, contract, local integration, optional live-turbopuffer integration, and one browser smoke test.
- README with a 3–5 minute demo script and a precise observability disclaimer.

### P1 — high-value additions

- Expected-document probe: direct lookup, filter predicate evidence, computed BM25/vector score, candidate-cutoff comparison, and clearly labeled no-filter counterfactual.
- Generic BEIR/JSONL + qrels importer so customers can use their own data.
- Branch a source namespace into a safe evaluation snapshot; automatically delete only branches created by PufferLab.
- Warm-cache hint action plus before/after **observed latency samples**; never infer cache tier.
- turbopuffer ANN `_debug/recall` panel, separated from judged relevance Recall@K.
- Optional Cohere/Voyage/etc. reranker adapters behind the same interface.
- Eval gates for CI: fail on aggregate threshold or max per-query regression.
- Compare explicit BM25 profiles or vector types in separate namespace/index profiles.

### P2 — stretch

- Paired bootstrap confidence intervals and significance warnings.
- Query tags/slices such as exact-token, semantic paraphrase, filtered, short, and long.
- Namespace metadata/index status and ingestion health panel.
- Experiment artifact bundles with config/schema/git revision.
- Sparse-vector and late-interaction adapters when broadly available.
- Minimal migration importer that transforms exported JSONL, not live vendor connectors.
- Shareable static HTML run report.

### Do not build

- A RAG chatbot, answer generation, agents, or prompt playground.
- A fake `EXPLAIN` plan, causal embedding explanations, or asserted cache-tier labels.
- User accounts, RBAC, billing, teams, hosted SaaS, Redis, Celery, Kubernetes, or microservices.
- Live Pinecone/Qdrant/Weaviate migration connectors.
- A custom vector index, BM25 implementation, or embedding training pipeline.
- LLM-generated judgments presented as ground truth.
- A synthetic “turbopuffer vs competitors” benchmark.
- Arbitrary query-language editors or every turbopuffer API feature.

## 6. Recommended architecture and stack

### Stack

| Layer | Choice | Reason |
|---|---|---|
| Core/backend | Python 3.12, FastAPI, Pydantic v2 | Excellent fit for turbopuffer's async Python SDK, IR metrics, embeddings, CLI reuse, and typed contracts. |
| Search client | Official `turbopuffer` Python SDK, async client | Connection reuse, official request/response types, and production-shaped calls. |
| Embeddings | `sentence-transformers`, `BAAI/bge-small-en-v1.5`, 384d | Small, local, reproducible, and used as a local-model pattern in turbopuffer docs. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small local second-stage baseline; provider interface prevents lock-in. |
| Eval math | Small in-house pure functions using NumPy | NDCG/Recall/MRR are simple enough to audit; avoids library-semantics ambiguity. Golden tests define behavior. |
| Metadata | SQLite + SQLAlchemy/Alembic | Durable local runs with no service dependency. |
| Jobs | In-process async job manager + persisted status | Adequate for a single-user local demo; polling avoids queue/WebSocket complexity. |
| UI | React, TypeScript, Vite, TanStack Query, Tailwind | Fast polished UI, strong table/state ecosystem, API types generated from OpenAPI. |
| CLI | Typer | Reuses the same Python services and supports real DE workflows. |
| Tests | pytest, respx/fakes, Playwright, Vitest | Pure core tests, adapter boundaries, and a thin browser workflow. |
| Tooling | `uv`, Ruff, mypy, npm | Fast deterministic setup and normal quality gates. |

Pin exact compatible dependency versions during scaffold implementation; do not copy versions from this planning document.

### Component flow

```text
CQADupStack/JSONL ──> dataset adapter ──> embed/batch ingest ──> turbopuffer namespace
       │                                                        │
       └── qrels/query sets ───────────────> SQLite             │
                                               ▲                │
React UI / Typer CLI ──> FastAPI ──> retrieval orchestrator ────┘
                              │          │
                              │          ├── client RRF/provenance reconstruction
                              │          └── local/external reranker
                              │
                              └── eval runner ──> metrics + regressions + traces ──> SQLite
```

The FastAPI app is the single composition root. The UI never calls turbopuffer directly, so keys remain server-side and every run uses the same contracts as the CLI.

### What to revisit if it grows

- Move jobs to a real queue only when multiple users or remote workers exist.
- Move SQLite to Postgres only when concurrent writers are real.
- Split embedding/reranking workers only when model inference blocks interactive requests.
- Add auth only if deployed beyond localhost.

## 7. turbopuffer data model

### Namespace strategy

One immutable dataset/index profile per namespace:

```text
pufferlab--{dataset_slug}--{dataset_version}--{index_profile_id}
```

Examples:

```text
pufferlab--cqadup-unix--v1--bge384-bm25v4
pufferlab--customer-acme--2026-08-21--voyage1024-bm25v4
```

Retrieval configs may share a namespace when they only vary query-time behavior. Changes to vector dimensions, vector model, tokenizer, or schema-time BM25 parameters require a distinct index profile/namespace unless current API behavior is explicitly verified to support the change in place.

### Document row

Use a native UUID ID derived with UUIDv5 from `{dataset_version}:{external_id}`. It is stable across re-ingestion, compact, and avoids maintaining an integer-ID allocator.

| Attribute | Type/indexing | Purpose |
|---|---|---|
| `id` | UUID | Stable document identity. |
| `title` | string, FTS enabled, not filterable | Weighted lexical title signal. |
| `body` | string, FTS enabled, not filterable | Main lexical content and display text. |
| `vector` | `[384]f16`, ANN, cosine | BGE embedding of `title + "\n\n" + body`. |
| `external_id` | string, filterable | Trace back to dataset/source. |
| `source` | string, filterable | Dataset/site/repository slice. |
| `source_type` | string, filterable | `question`, `doc`, `issue`, `pr`, etc. |
| `tags` | `[]string`, filterable | Query slices and customer filters. |
| `created_at` | datetime, filterable | Time filters and later recency experiments. |
| `url` | string, not filterable | Human inspection link. |
| `content_hash` | string, not filterable | Deterministic ingestion verification. |

Explicit P0 FTS profile for both `title` and `body`:

```json
{
  "tokenizer": "word_v4",
  "case_sensitive": false,
  "language": "english",
  "stemming": false,
  "remove_stopwords": false,
  "ascii_folding": false,
  "max_token_length": 39,
  "k1": 1.2,
  "b": 0.75,
  "k3": 8.0
}
```

The exact profile is part of `DatasetVersion`; it is not an implicit service default.

### Local metadata model

SQLite stores control-plane and eval artifacts, not the search corpus:

- `dataset_versions`
- `retrieval_configs`
- `query_sets`
- `judged_queries`
- `qrels`
- `eval_runs`
- `run_configs`
- `query_outcomes`
- `search_traces`

Each immutable revision gets a content hash. Editing a config creates a new revision so old runs remain interpretable.

## 8. Retrieval architecture

### BM25

Use a weighted query-time expression:

```text
Sum(
  Product(2.0, BM25(title, query)),
  BM25(body, query)
)
```

Return title/body/source attributes, overall `$dist`, highlighted body fragments, and vector distance as a computed attribute where useful.

### Vector

Embed the query with the exact model/revision in the index profile, then ANN over `vector` with the same filters and `candidate_k`. Return BM25 as a computed attribute for the returned rows. Never return stored vectors to the UI.

### Hybrid

Issue BM25 and ANN as a single `multi_query` so both see one consistent snapshot. In production/eval mode, use turbopuffer's server-side weighted RRF. In debug mode, request the two raw lists and reconstruct RRF from their observed ranks using the documented formula. A contract test verifies reconstructed ordering against server RRF on a small live namespace.

### Hybrid + reranker

Take the fused top `rerank_depth` (default 50), fetch only required text fields, run the local cross-encoder, and return the top `result_k` (default 10). Record turbopuffer request time and reranker time separately.

### Forensic classification

Only emit classifications justified by observations:

- `FILTER_PREDICATE_FAILED`: local document attributes fail the explicit filter AST.
- `NO_LEXICAL_SCORE`: targeted computed BM25 score is zero/absent.
- `OUTSIDE_LEXICAL_CANDIDATES`: score does not clear observed lexical candidate boundary.
- `OUTSIDE_VECTOR_CANDIDATES`: absent from the observed ANN list; do not say why internally.
- `OUTSIDE_FUSION_TOP_K`: present in inputs but below fused cutoff.
- `RERANKED_DOWN`: present before reranking and below result cutoff afterward.
- `NOT_OBSERVABLE`: evidence is insufficient.

For vector misses, if an exact distance would clear the observed ANN boundary but the document was absent, report an **observed ANN candidate miss** and offer the separate turbopuffer recall probe. Do not assert the internal cause.

## 9. Evaluation approach

### Dataset choice

Use the **CQADupStack Unix** retrieval subset for P0.

Why it is better than immediately scraping docs/issues:

- It is an existing duplicate-question retrieval benchmark with real Stack Exchange posts and relevance relationships rather than hand-wavy synthetic “ideal” results.
- The Unix subset is manageable at roughly 47K corpus rows and about 1K queries, while still containing technical commands, rare strings, and semantic paraphrases that should make lexical and dense retrieval diverge.
- It supports credible NDCG/Recall/MRR from day one and is already available in BEIR-style corpus/query/qrels form.
- It is small enough to embed locally and large enough that candidate retrieval is meaningful.

The original CQADupStack project describes 12 Stack Exchange subforums and predefined retrieval splits ([official repository](https://github.com/D1Doris/CQADupStack)); the BEIR-format Unix configuration reports 47,382 corpus rows and 1,072 queries ([dataset card](https://huggingface.co/datasets/BeIR/cqadupstack/blob/main/README.md)). Underlying public contributions require Stack Exchange attribution under CC BY-SA; the repository must preserve per-document URLs/IDs and include a dataset notice ([Stack Overflow licensing](https://stackoverflow.com/help/licensing)).

Why not P0 TechQA: it is an excellent, realistic technical-support dataset with 801,998 Technotes and real forum questions, but full ingestion/embedding is too heavy for the first polished build and the task is partly answer-span QA rather than pure search evaluation ([IBM Research](https://research.ibm.com/publications/the-techqa-dataset), [official repository](https://github.com/IBM/techqa)). Add it later as a scale pack.

Why not P0 GitHub docs/issues: the domain is visually compelling but obtaining defensible judgments is the hard part. It becomes a hand-labeled dataset project before PufferLab itself exists. Add a generic JSONL/qrels importer in P1 and then curate one repository workload deliberately.

### Demo and full query sets

- `unix-demo-v1`: 50 deterministic queries selected for relevance coverage and visible lexical/vector disagreement.
- `unix-full-v1`: all official evaluation queries with qrels.
- `unix-regressions-v1`: generated from the current run's top 10 regressions for rapid iteration.

Dataset acceptance includes a small audit report that identifies at least:

- three exact-token/command queries where BM25 is competitive or best;
- three semantic-paraphrase queries where vector retrieval is competitive or best;
- three cases where hybrid changes the candidate set;
- one reranker win and one reranker regression.

Choose the final demo queries from measured results; do not manufacture them in advance.

### Metrics

- `NDCG@10`: primary quality metric; preserves graded qrels when a dataset supplies them.
- `Recall@50`: candidate coverage; relevance is `grade > 0`.
- `MRR@10`: time-to-first-relevant-result proxy.
- `latency_ms`: client-observed request/stage and total durations.
- `error_rate`: failed queries / attempted queries.

Store per-query values before aggregation. A regression is ranked by candidate-minus-baseline `NDCG@10`, ascending, with MRR and Recall deltas as supporting evidence.

### Run protocol

1. Validate namespace metadata, index status, schema hash, and document count.
2. Warm up each config on an unmeasured query set.
3. Randomize query order with a stored seed.
4. Interleave configs by query for quality comparison; run a separate sequential, bounded-concurrency latency pass if latency is being used for decisions.
5. Use the same consistency level and filters for comparable configs.
6. Record embedding cache state, model revision, candidate counts, concurrency, region, client host, and wall-clock timestamp.
7. Show latency as observed for this run, not a turbopuffer service-level benchmark.
8. Persist partial results; mark cancelled/interrupted runs explicitly.

No aggregate quality number is valid when qrels coverage is zero. No p95 should be treated as stable without showing sample count and distribution.

## 10. Main UI and CLI workflows

### UI routes

- `/runs` — recent eval runs and status.
- `/runs/:runId` — aggregate table, latency, errors, largest gains/regressions.
- `/playground` — ad hoc compare or deep-linked run/query/config state.
- `/configs` — read-only config revisions in P0; duplicate-and-edit form if time permits.
- `/datasets` — dataset/index status and ingestion instructions.

### CLI

```bash
pufferlab dataset ingest cqadupstack-unix --namespace pufferlab--cqadup-unix--v1--bge384-bm25v4
pufferlab config seed
pufferlab eval run --query-set unix-demo-v1 --baseline bge-ann --candidate hybrid-rrf --candidate hybrid-rerank
pufferlab eval export <run-id> --format json
pufferlab serve
```

P1:

```bash
pufferlab namespace branch --source customer-prod --purpose eval
pufferlab eval gate <run-id> --metric ndcg@10 --min-delta 0 --max-query-drop 0.20
pufferlab namespace warm <namespace>
```

## 11. Risks and technical unknowns

| Risk/unknown | Impact | Mitigation / validation spike |
|---|---|---|
| Server-RRF responses do not preserve raw subquery provenance | Debug view could overclaim | Use a separate raw multi-query only in debug mode; verify local RRF ordering against server RRF. |
| Schema-time FTS parameter changes on a branch may not behave as assumed | BM25 experiments may require reingestion | Run a tiny live spike; treat each index profile as a separate namespace until proven otherwise. |
| No public forced-cold or cache-tier API | Cold benchmark would be misleading | Keep cold/warm out of P0; expose only warm hint and observed samples in P1. |
| Local cross-encoder is slow on CPU | Demo latency may be awkward | Rerank only top 50, preload model at startup, cache query/doc scores, record stage time. |
| CQADupStack preprocessing/licensing | Reproducibility and attribution risk | Preserve source IDs/URLs, include CC BY-SA notice, hash preprocessing output, never commit raw data. |
| One client latency sample per query is noisy | False performance conclusions | Warmup, stored random seed, bounded concurrency, sample count, separate quality and latency passes. |
| Index build returns HTTP 202 after schema enablement | Ingestion may look complete prematurely | Poll namespace metadata/index status and fail readiness until `up-to-date`. |
| API limits/429s | Interrupted eval/ingestion | Bounded async concurrency, retry with jitter through official SDK, persist progress. |
| Exact model revision drifts | Regression validity breaks | Pin model/revision and include it in index profile/content hash. |
| Full dataset embedding time | Slow first setup | Cache processed corpus/embeddings locally outside git; provide a 200-document fixture for development. |
| UI scope expands into config/query-language builder | Polish suffers | Seed configs, expose only candidate count/RRF weights/rerank depth as editable P0 knobs. |

## 12. Three-to-five-minute demo story

### 0:00–0:35 — Frame the customer problem

“A customer is deciding whether to replace vector-only retrieval with hybrid + reranking. Their averages improved in a notebook, but they need to know what got worse before shipping.”

Show the dataset/index profile: 47K Unix support questions, explicit BM25 schema, BGE vectors, one turbopuffer namespace.

### 0:35–1:30 — Inspect one query

Open a curated semantic query in Playground. Compare BM25, ANN, hybrid RRF, and hybrid + reranker. Point out:

- exact candidate membership and overlap;
- `$dist`/computed signals and score direction;
- RRF rank contributions;
- client-observed turbopuffer vs reranker latency;
- the observability disclaimer.

Say: “This shows what the pipeline returned and what we instrumented. It does not pretend to expose turbopuffer's internal centroid or cache decisions.”

### 1:30–2:25 — Run the search-quality tests

Open the completed eval run. Baseline is vector-only; candidates are hybrid and hybrid + reranker. Show NDCG@10, Recall@50, MRR@10, p95, errors, and query count. Emphasize that turbopuffer is first-stage retrieval and the reranker cost is separate.

### 2:25–3:30 — Find and debug the regression

Click the largest negative NDCG delta. The deep link opens the exact query, judged relevant documents, baseline/candidate ranks, and stage membership. Show that the relevant document was present in one candidate list but fell below a fusion/rerank cutoff—or, if measured data shows a different failure, use that real failure.

### 3:30–4:20 — Improve one variable

Duplicate the candidate config, change only `candidate_k`, RRF weight, or rerank depth, and run the 10-query regression set. Show the regression recovered without hiding the latency trade-off.

### 4:20–5:00 — Close on turbopuffer architecture

“The corpus lives in one object-storage-backed namespace; BM25 and ANN run as a same-snapshot multi-query; turbopuffer handles first-stage retrieval and filtering; application code owns transparent fusion diagnostics and second-stage reranking. The same harness can branch a customer's namespace, run their judged workload safely, and turn POC opinions into regression tests.”

## Source set

Primary turbopuffer sources used for this brief:

- [Architecture](https://turbopuffer.com/docs/architecture)
- [Concepts](https://turbopuffer.com/docs/concepts)
- [Tradeoffs](https://turbopuffer.com/docs/tradeoffs)
- [Query API](https://turbopuffer.com/docs/query)
- [Write/schema API](https://turbopuffer.com/docs/write)
- [Hybrid guide](https://turbopuffer.com/docs/hybrid)
- [Full-text search guide](https://turbopuffer.com/docs/fts)
- [Performance guide](https://turbopuffer.com/docs/performance)
- [Ingestion guide](https://turbopuffer.com/docs/ingestion)
- [Testing guide](https://turbopuffer.com/docs/testing)
- [Branching guide](https://turbopuffer.com/docs/branching)
- [Warm-cache API](https://turbopuffer.com/docs/warm-cache)
- [Recall API](https://turbopuffer.com/docs/recall)
- [Roadmap/changelog](https://turbopuffer.com/docs/roadmap)
- [Deployed Engineer role](https://jobs.ashbyhq.com/turbopuffer/c89ab81b-1fb1-4b6b-8ffb-9926adeeb0f9/)
