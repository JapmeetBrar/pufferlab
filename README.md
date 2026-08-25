# PufferLab

PufferLab is a local search-evaluation workbench for turbopuffer. It helps a team answer a
deceptively simple question: **which search configuration works best for our judged workload, and
which queries explain the result?**

Instead of choosing a retrieval strategy from a few hand-picked examples, PufferLab runs the same
judged queries across lexical, vector, hybrid, and reranked search. It reports aggregate quality,
then connects every gain or regression back to the relevant documents and ranks.

## What it does

- Compares BM25, vector ANN, server-side hybrid RRF, and hybrid retrieval with a client-side
  cross-encoder.
- Computes NDCG@10, Recall@50, MRR@10, coverage, errors, and client wall-clock latency.
- Persists runs and per-query outcomes in SQLite so dashboards and exports are reproducible.
- Ranks query-level regressions and gains and deep-links into judged-document evidence.
- Provides a side-by-side Playground for live BM25 and vector comparisons.
- Applies a provider-free CLI quality gate to a completed, authenticated local run.

Stored run pages read durable evidence without contacting turbopuffer. Any live replay or document
diagnostic is an explicit, potentially cost-bearing new observation; it is kept separate from what
was recorded in the original run.

## Quickstart: provider-free demo

Prerequisites: Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), Node.js 22+, and pnpm 11.
If pnpm is missing, install the repository's pinned major with `npm install --global pnpm@11`.

From the repository root:

```bash
uv sync --locked
cd web
pnpm install --frozen-lockfile
cd ..

export PUFFERLAB_DATA_DIR="$PWD/data/demo"
export TURBOPUFFER_API_KEY=
export TURBOPUFFER_REGION=
export PUFFERLAB_SEARCH_NAMESPACE=

uv run pufferlab demo seed
uv run pufferlab doctor --mode demo
```

The seed is deterministic and creates one completed run with 50 queries, four configurations, and
200 outcomes. It needs no API key, model download, provider, or network access.

Start the API in terminal 1 from the repository root:

```bash
export PUFFERLAB_DATA_DIR="$PWD/data/demo"
export TURBOPUFFER_API_KEY=
export TURBOPUFFER_REGION=
export PUFFERLAB_SEARCH_NAMESPACE=
uv run pufferlab serve --host 127.0.0.1 --port 8000
```

The server intentionally stays in the foreground. Leave it running, then start the frontend in
terminal 2:

```bash
cd web
VITE_API_BASE_URL=http://127.0.0.1:8000 pnpm dev
```

Open [http://localhost:5173/runs](http://localhost:5173/runs), choose **Synthetic demo**, compare
the four summaries, and inspect a regression. See the
[provider-free operator runbook](docs/synthetic-demo.md) for quality-gate commands, allocated-port
operation, browser checks, idempotence, and cleanup.

## Dataset and recorded workload

The live reference workload uses the Unix subset of CQADupStack in BEIR format:

- 47,382 Unix & Linux Stack Exchange documents
- 50 deterministic curated queries across lexical, semantic, hybrid, and reranking cases
- 83 graded relevance judgments
- four fixed search configurations and 200 successful query/configuration outcomes

PufferLab does **not** ship the licensed corpus or query text. The checked-in source locks,
text-free curation manifest, and attribution notice make the ignored local pack reproducible. See
the [CQADupStack Unix runbook](docs/datasets/cqadupstack-unix.md) and
[dataset notice](NOTICE-DATASETS.md).

One recorded run in `gcp-us-west1` produced:

| Configuration | NDCG@10 | Recall@50 | MRR@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.330995 | 0.488000 | 0.363190 | 201.502 | 275.288 |
| Vector ANN | **0.456537** | **0.696667** | **0.511500** | 504.903 | 959.744 |
| Hybrid RRF | 0.418917 | 0.675000 | 0.455627 | 546.025 | 1,198.534 |
| Hybrid + client cross-encoder | 0.435933 | 0.675000 | 0.498667 | 1,451.313 | 2,460.539 |

These results describe this judged workload, not a universal ranking of search methods. Latency is
client wall-clock time across the complete local/provider path, including embedding and reranking
where applicable; it is not a turbopuffer server benchmark. The useful next step is to inspect the
queries behind each aggregate difference.

## Run the Unix evaluation

The full workflow downloads and verifies the 5.34 GB pinned archive, prepares an ignored local
pack, ingests it into a caller-managed turbopuffer namespace, and runs the four fixed configurations.
Follow the [dataset runbook](docs/datasets/cqadupstack-unix.md); the final commands are:

```bash
uv sync --locked --extra live-search
uv run pufferlab dataset ingest-unix --processed-pack \
  data/cqadupstack-unix/processed/cqadupstack-unix-6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8
uv run pufferlab eval run --seeded-defaults
```

Use a server-only `.env` copied from `.env.example` for `TURBOPUFFER_API_KEY` and
`TURBOPUFFER_REGION`. Never put credentials in a `VITE_*` variable, commit them, or include them in
logs, screenshots, exports, issues, or pull requests.

## Architecture and evidence boundary

- **FastAPI backend:** dataset verification, turbopuffer retrieval, evaluation, run control, and
  generated OpenAPI contracts.
- **React/Vite frontend:** run comparison, regression/gain exploration, query forensics, and the
  live Playground.
- **SQLite control plane:** versioned dataset/query/config catalogs plus durable run outcomes. It
  stores ranked document IDs and judgments, not the remote corpus or raw vectors.
- **turbopuffer data plane:** BM25, ANN, and server RRF over the ingested corpus. The optional
  cross-encoder reranks returned candidates in the PufferLab backend process, client-side relative
  to turbopuffer rather than in browser JavaScript.

The API key and embedding vectors stay in the backend. Stored views never silently replay search.
The application records only facts supported by the stored run or a separately labeled live
observation; it does not infer missing retrieval stages from final ranks.

## Current limits

- Local, single-user application with one API/evaluation worker; no hosted auth or multi-tenancy.
- No generic customer dataset/query-set upload yet. The current paths are the bundled synthetic
  demo, the 20-document tiny fixture, and the curated CQADupStack Unix workflow.
- The judged evaluation uses exactly 50 queries and four seeded configurations; arbitrary suite and
  configuration editing are not implemented.
- Latency is directional client wall time, not a controlled infrastructure benchmark.
- Live diagnostics and replay may incur model and turbopuffer usage and require the original bound
  namespace to remain available.

## Development

Run the full backend and frontend gate:

```bash
make check
```

Run the provider-free desktop/mobile browser journeys separately:

```bash
cd web
pnpm test:e2e
```

API behavior and evidence rules are documented in [contracts](docs/contracts.md). Contributions use
the branch/reviewer process in [the engineering loop](docs/engineering-loop.md), with the active
review unit tracked in [the progress ledger](docs/progress.md).
