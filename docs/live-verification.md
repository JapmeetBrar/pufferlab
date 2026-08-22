# Milestone 1 Live Verification

Use this runbook to prove the complete browser → FastAPI → turbopuffer → browser path against
the checked-in 20-document fixture. The run is successful only when ingestion, an idempotent rerun,
the direct API comparison, the interactive browser comparison, secret scans, cleanup, and protected
GitHub checks all pass.

## Safety invariants

- Put `TURBOPUFFER_API_KEY` only in the ignored local `.env` file. Never paste it into commands,
  screenshots, logs, PR text, or a `VITE_*` variable.
- Let the ingestion command generate the `pufferlab-tiny-*` namespace. Retain that exact immutable
  value for rerun, API startup, browser verification, and cleanup.
- Never delete a namespace supplied by another person or a generic environment. Cleanup is allowed
  only for the exact namespace generated and recorded by this run, after rechecking its reserved
  prefix.
- Capture identifiers, counts, status, ranks, score semantics, and timings. Do not capture API keys,
  model vectors, request headers, or `.env` contents.

## 1. Prerequisites

From a clean checkout of protected `main`:

```bash
cp .env.example .env
uv sync --locked --extra live-search
cd web && pnpm install --frozen-lockfile && cd ..
```

Add only the key and account region to `.env`:

```dotenv
TURBOPUFFER_API_KEY=<local-only>
TURBOPUFFER_REGION=gcp-us-central1
```

## 2. Prove the real embedding boundary

Load the manifest's exact model revision and embed one query plus two passages. Verify every vector
has 384 finite dimensions and unit norm. Do not print vector values.

Current non-secret preflight evidence (2026-08-22):

- model: `BAAI/bge-small-en-v1.5`
- revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- dimensions: `384`
- query norm: `1.000000`
- passage norms: `1.000000`, `1.000000`
- installed CLI/shared-embedding focused tests: `16 passed`

## 3. Run the isolated provider smoke test

```bash
PUFFERLAB_RUN_LIVE=1 uv run --extra live-search pytest \
  backend/tests/live/test_turbopuffer_live.py -q
```

The test must generate its own `pufferlab-live-test-*` namespace, prove write/read behavior, and
delete that exact namespace in `finally`. Cleanup failure is a test failure.

## 4. Ingest and verify idempotence

```bash
uv run --extra live-search pufferlab dataset ingest-tiny
```

Record the generated namespace from the final `PUFFERLAB_SEARCH_NAMESPACE=...` line as
`<exact-live-namespace>`. Confirm the first run reports the explicit schema hash, 20 documents, and
`ready`. Then rerun the same stable-ID upserts and exact remote readiness checks:

```bash
uv run --extra live-search pufferlab dataset ingest-tiny \
  --namespace <exact-live-namespace>
```

Both runs must report the same schema hash, exact 20-document count, exact UUID set, observed ANN
metric, and ready index state.

## 5. Start the server and browser app

In terminal one:

```bash
PUFFERLAB_SEARCH_NAMESPACE=<exact-live-namespace> \
  uv run --extra live-search uvicorn pufferlab.main:app --app-dir backend --port 8000
```

In terminal two:

```bash
cd web
pnpm dev --host 127.0.0.1 --port 5173
```

Confirm `GET http://127.0.0.1:8000/api/v1/health` and
`GET http://127.0.0.1:8000/api/v1/configs` succeed before opening the browser.

## 6. Exercise the real browser path

Open `http://127.0.0.1:5173`, submit one checked-in fixture query, and verify:

1. The browser POST contains only contract version, query text, the BM25/vector config IDs, and the
   debug flag. It contains no API key or vector.
2. Both BM25 and vector columns contain live results.
3. Each displayed hit has document identity, visible 1-based rank, score value/kind/direction, and
   client-wall-clock timing.
4. The URL contains stable `q`, `left`, and `right` parameters and restores the selections.
5. Desktop uses two result columns; a 390-pixel viewport stacks them readably.
6. The observability notice is present and makes no claim about unexposed provider internals.

Record a screenshot only after confirming it contains no key, request headers, or `.env` content.

## 7. Cleanup the browser namespace

Stop both local servers. Delete only `<exact-live-namespace>` using a one-off guarded provider call
that hard-codes the exact value recorded in step 4 and asserts the `pufferlab-tiny-` prefix before
calling `delete_namespace`. Then query metadata for that same exact value and require the provider's
not-found response. Do not turn this into a cleanup command that accepts arbitrary user input.

## 8. Final evidence and protected merge

Update [`progress.md`](progress.md) with the sanitized live namespace fingerprint, ingestion/rerun
counts, smoke-test cleanup result, API/browser assertions, screenshot path, secret scan, full local
gates, reviewer verdict, merge commit, and protected-main CI links. The finalization PR must be
independently reviewed and merged; its GitHub merge/check history is canonical and requires no
recursive ledger PR.

