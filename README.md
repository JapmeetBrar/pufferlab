# PufferLab

PufferLab is a search evaluation and query-forensics workbench for turbopuffer. It compares lexical, vector, hybrid, and reranked retrieval; runs judged query sets; surfaces regressions; and opens failures in an evidence-based debugger.

The project is currently in its contract/scaffold milestone. See:

- [Project decision and implementation brief](docs/project-decision-and-implementation-brief.md)
- [Shared contracts](docs/contracts.md)
- [Implementation plan](docs/implementation-plan.md)

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+
- pnpm 11+

## Backend

```bash
cp .env.example .env
uv sync
uv run uvicorn pufferlab.main:app --app-dir backend --reload
```

The API is served at `http://localhost:8000`; health is available at `GET /api/v1/health` and interactive API documentation at `/docs`.

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

cd web
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

## Credentials

`TURBOPUFFER_API_KEY` is read only by the backend. Never put it in a `VITE_*` variable, commit it, print it in logs, or include it in exported run artifacts.
