# PufferLab

PufferLab is a search evaluation and query-forensics workbench for turbopuffer. It compares lexical, vector, hybrid, and reranked retrieval; runs judged query sets; surfaces regressions; and opens failures in an evidence-based debugger.

The local product now includes a durable SQLite-backed run history, aggregate and per-query
regression analysis, stable query deep links, an evidence-honest forensic drawer, and an explicit
live replay path. Stored-run pages are provider-free; only clearly labeled actions can start new
provider work. See:

- [Project decision and implementation brief](docs/project-decision-and-implementation-brief.md)
- [Shared contracts](docs/contracts.md)
- [Implementation plan](docs/implementation-plan.md)
- [Milestone 3 execution plan](docs/milestone-3-execution.md)
- [Offline synthetic demo](docs/synthetic-demo.md)
- [Observability and demo runbook](docs/observability.md)
- [Milestone 1 live-verification runbook](docs/live-verification.md)
- [CQADupStack Unix local-pack runbook](docs/datasets/cqadupstack-unix.md)

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+
- pnpm 11+

If `pnpm` is not installed, install the repository's pinned major before continuing:

```bash
npm install --global pnpm@11
```

## Five-minute provider-free demo

Install the locked Python and browser dependencies from the repository root:

```bash
uv sync --locked
cd web
pnpm install --frozen-lockfile
cd ..
```

In terminal 1, choose one ignored local data directory, seed the deterministic demo, and start the
API with exactly one worker:

```bash
export PUFFERLAB_DATA_DIR=data/demo
uv run pufferlab demo seed
uv run uvicorn pufferlab.main:app --app-dir backend --workers 1
```

The seed command creates the directory when needed and writes one complete 50-query, four-config,
200-outcome run. It requires no `.env`, API key, model download, provider, or network access.

In terminal 2, start the dashboard:

```bash
cd web
pnpm dev
```

Open `http://localhost:5173/runs`, then use this interview flow:

1. Open the run labeled **Synthetic demo**. Its durable metrics and all provider-free reads come
   from `data/demo/pufferlab.sqlite3`.
2. In **Regressions and gains**, change candidate/order/row controls. The run URL records those
   choices as `candidate`, `order`, and `limit` query parameters.
3. Choose **Inspect recorded query** on a regression. The server-issued URL is
   `/playground?run=<uuid>&query=<uuid>&left=<uuid>&right=<uuid>`; it contains identities, not query
   text.
4. Choose **Inspect document** to open the forensic drawer. Its URL adds only a `document=<uuid>`.
   Refresh, use Back to close it, then Forward to restore it.
5. Confirm that original stage evidence is `NOT_OBSERVABLE`, synthetic timing is unavailable, and
   live replay is disabled for this read-only origin.

The equivalent recorded-query route is
`/runs/<run-uuid>/queries/<query-uuid>?left=<uuid>&right=<uuid>[&document=<uuid>]`. Merely opening,
refreshing, or navigating either form performs GET-only durable reads and never starts provider
work. See [the demo runbook](docs/synthetic-demo.md) for idempotence and cleanup boundaries.

## Ingest the tiny fixture

Copy the server-only settings file, add your turbopuffer API key, and keep the default region or
replace it with the region for your account:

```bash
cp .env.example .env
uv sync --locked --extra live-search
uv run pufferlab dataset ingest-tiny
```

The command explains the target region, generated namespace, schema hash, pinned embedding model,
and 20-document write before constructing the model or provider. It then prints compact progress.
Its final line is the exact server-side setting to copy into `.env`, for example:

```dotenv
PUFFERLAB_SEARCH_NAMESPACE=pufferlab-tiny-0123456789abcdef01234567
```

The default namespace contains a cryptographically random suffix. To resume or verify the same
namespace idempotently, pass back that exact owned name:

```bash
uv run pufferlab dataset ingest-tiny --namespace pufferlab-tiny-0123456789abcdef01234567
```

Explicit targets must be safe `pufferlab-*` names. The ingestion command only performs stable
UUID upserts and readiness reads; it has no deletion or cleanup path. It never prints the API key
or embedding vectors. Run `uv run pufferlab dataset ingest-tiny --help` for batching and bounded
readiness options.

## Run the curated Unix evaluation

First prepare the ignored CQADupStack Unix pack using the
[dataset runbook](docs/datasets/cqadupstack-unix.md). Then ingest its exact content-addressed
directory and persist the READY dataset, curated 50-query set, and four immutable configurations:

```bash
uv sync --locked --extra live-search
uv run pufferlab dataset ingest-unix \
  --processed-pack data/cqadupstack-unix/processed/cqadupstack-unix-6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8
```

The command generates an owned `pufferlab-unix-*` namespace unless `--namespace` names an existing
owned target for an idempotent resume. It verifies the ignored pack, checkpoints stable-ID writes,
waits for exact remote readiness, and prints only safe revision/configuration identities.

Run the persisted 50-query suite across BM25, ANN, server RRF, and server RRF plus the pinned local
reranker with one command:

```bash
uv run pufferlab eval run --seeded-defaults
```

Progress appears only after outcomes commit to `data/pufferlab.sqlite3`. Exit status is `0` only
when all 200 config/query attempts succeed, `3` when the run completes with coverage failures, and
nonzero for cancellation or a systemic failure. `config seed` is an idempotent way to recreate the
canonical config revisions for one persisted dataset:

```bash
uv run pufferlab config seed
```

Export completed or partial durable state beneath the ignored data directory, using the `run_id`
printed by `eval run`:

```bash
uv run pufferlab eval export <run-id> --output exports/<run-id>.json
```

Exports contain typed ranks, metrics, timings, warnings, and redacted failures—never query/document
text, credentials, request bodies, or vectors.

## Serve persisted live runs

For the tiny-fixture Playground, copy the ingestion command's printed namespace assignment into
`.env`. For persisted evaluation runs, point `PUFFERLAB_DATA_DIR` at the same SQLite directory used
by the CLI. Then start the API:

```bash
uv run uvicorn pufferlab.main:app --app-dir backend --workers 1
```

The API is served at `http://localhost:8000`; health is available at `GET /api/v1/health` and
interactive API documentation at `/docs`.

PufferLab's local evaluation controller deliberately supports exactly one Uvicorn worker. Startup
holds an exclusive guard beside the configured `pufferlab.sqlite3`, migrates the database, marks
orphaned running jobs interrupted, and reclaims valid queued jobs oldest-first. A second API worker
fails startup instead of executing the same durable run twice.

Run history, run detail, regressions, query detail, and export are SQLite reads and remain usable
without a provider. The live BM25-versus-vector Playground and explicit query replay use the
optional local embedding runtime and the exact persisted provider namespace. The backend loads the
fixture's pinned `BAAI/bge-small-en-v1.5` revision lazily on the first vector comparison.
`TURBOPUFFER_API_KEY` and query vectors stay inside the backend process.

Live replay is deliberately different from opening a stored run. It is available only for an exact
stored live run/query/config binding and only after the user presses **Run live replay
(cost-bearing)**. The server authenticates the complete persisted 50-query suite against the
checked source anchor before it constructs credential, embedding, reranking, or provider-capable
objects. It derives query text, graded judgments, configs, and namespace server-side; the browser
cannot supply them.

> **Cost and credential warning:** live replay can incur embedding and turbopuffer usage. Confirm
> that `.env` contains the intended server-only key and region and that the run's original namespace
> is still ready. Selecting **Include separate counterfactual provenance probes** makes additional
> provider requests. Never paste the key, licensed query text, qrels, namespace, provider bodies, or
> raw vectors into logs, screenshots, issues, or pull requests.

## Frontend

```bash
cd web
pnpm install --frozen-lockfile
pnpm generate:api
pnpm dev
```

Vite serves the app at `http://localhost:5173` and proxies `/api` to the backend during development.

## Checks

```bash
uv sync --locked
uv run ruff check backend scripts
uv run ruff format --check backend scripts
uv run mypy
uv run pytest
uv run python scripts/generate_openapi.py --check
uv run python scripts/audit_dataset_artifacts.py

cd web
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

## Credentials

`TURBOPUFFER_API_KEY` is read only by the backend. Never put it in a `VITE_*` variable, commit it,
print it in logs, or include it in exported run artifacts.
