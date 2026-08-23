# PufferLab

PufferLab is a search evaluation and query-forensics workbench for turbopuffer. It compares lexical, vector, hybrid, and reranked retrieval; runs judged query sets; surfaces regressions; and opens failures in an evidence-based debugger.

The project has completed its first live vertical slice and is building its durable judged-evaluation
milestone. See:

- [Project decision and implementation brief](docs/project-decision-and-implementation-brief.md)
- [Shared contracts](docs/contracts.md)
- [Implementation plan](docs/implementation-plan.md)
- [Milestone 1 live-verification runbook](docs/live-verification.md)
- [CQADupStack Unix local-pack runbook](docs/datasets/cqadupstack-unix.md)

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+
- pnpm 11+

## Ingest the tiny fixture

Copy the server-only settings file, add your turbopuffer API key, and keep the default region or
replace it with the region for your account:

```bash
cp .env.example .env
uv sync --extra live-search
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
uv sync --extra live-search
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

## Backend

After copying the printed namespace assignment into `.env`, start the API:

```bash
uv run uvicorn pufferlab.main:app --app-dir backend --reload
```

The API is served at `http://localhost:8000`; health is available at `GET /api/v1/health` and
interactive API documentation at `/docs`.

The config catalog is available without provider credentials. The live BM25-versus-vector compare
path uses the optional local embedding runtime and ingested namespace configured above. The backend
loads the fixture's exact pinned `BAAI/bge-small-en-v1.5` revision lazily on the first vector
comparison. `TURBOPUFFER_API_KEY` and the query vector stay inside the backend process.

## Frontend

```bash
cd web
pnpm install
pnpm generate:api
pnpm dev
```

Vite serves the app at `http://localhost:5173` and proxies `/api` to the backend during development.

## Checks

```bash
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
