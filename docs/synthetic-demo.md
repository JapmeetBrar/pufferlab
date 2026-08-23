# Offline synthetic demo

Use this path for an interview, UI review, or fresh-checkout smoke test when no turbopuffer account,
licensed dataset pack, embedding model, or network access should be required.

## Seed and serve an empty data directory

From the repository root, install dependencies once:

```bash
uv sync --locked
cd web
pnpm install --frozen-lockfile
cd ..
```

Choose a new ignored directory and keep the same `PUFFERLAB_DATA_DIR` value for both seeding and the
API process:

```bash
export PUFFERLAB_DATA_DIR=data/demo
uv run pufferlab demo seed
uv run uvicorn pufferlab.main:app --app-dir backend --workers 1
```

In a second terminal:

```bash
cd web
pnpm dev
```

Open `http://localhost:5173/runs`. Vite proxies `/api` to the one-worker API at
`http://localhost:8000`.

The seed command migrates the configured SQLite database and writes one deterministic,
PufferLab-authored dataset, 50 judged queries, the canonical BM25/ANN/server-RRF/local-reranker
configuration order, and 200 successful outcomes. Its output is limited to durable UUIDs,
origin/timing labels, and counts. It does not write an export, vector, credential, namespace,
licensed query, provider response, or tracked database artifact.

## What to exercise

1. Open the run marked **Synthetic demo · read-only** from run history.
2. Verify all four configuration summaries and the 50-of-50 durable query-group progress.
3. Switch the regression candidate, `regressions`/`gains` order, and row limit. The URL should retain
   the selected `candidate`, `order`, and `limit`.
4. Follow **Inspect recorded query**. The deep link contains only `run`, `query`, `left`, `right`,
   and optionally `document` UUIDs; local query text and judgments never enter the URL.
5. Inspect a judged document, then refresh and use Back/Forward. The drawer selection should restore
   from the `document` UUID.
6. Confirm that the stored record reports original stage evidence as `NOT_OBSERVABLE`, latency as
   unavailable, and live replay as disabled.

Opening or restoring these pages performs provider-free GETs. No browser navigation, polling,
refresh, or drawer action can turn a synthetic identity into a create or replay request.

## Idempotence and source of truth

Re-run the same command with the same data directory:

```bash
uv run pufferlab demo seed
```

The second run validates and reuses the same content-addressed dataset, query set, four configs,
run, and canonical export bytes. It does not create a second run or replace existing rows. SQLite
at `$PUFFERLAB_DATA_DIR/pufferlab.sqlite3` is the demo source of truth; generated SQLite and export
files remain ignored runtime artifacts, never repository inputs.

If a different API process appears empty, it is almost always using a different
`PUFFERLAB_DATA_DIR`. Stop that process, export the same value used for the seed, and restart with
one worker.

## Evidence limitations

The synthetic run is explicitly `data_origin=synthetic_demo` and read/export-only. Backend create,
startup recovery, and replay paths reject that origin before credential, embedder, reranker,
retrieval-runtime, or provider construction. The dashboard disables the corresponding cost-bearing
controls.

Quality is real evaluator output over authored ranked document IDs and qrels: per-query metrics are
computed by the normal evaluation engine and run summaries by the normal aggregator. Timing is not
fabricated. Because no searches ran, every success uses
`timing_source=synthetic_unavailable`; total and stage timings are null or absent, latency sample
counts are zero, percentiles are null, and provider traces and candidate-count claims are absent.

The seed demonstrates durable application behavior and search-quality analysis, not provider
latency, namespace readiness, live retrieval parity, or original provider-stage evidence. Use the
[observability runbook](observability.md) when explaining those boundaries.

## Safe troubleshooting

- **`pnpm: command not found`:** install pnpm 11 with `npm install --global pnpm@11`, then rerun the
  frozen install from `web/`.
- **API startup says another worker owns the database:** stop the older PufferLab API. Do not work
  around the guard with multiple workers; P0 supports exactly one.
- **Run history is empty:** compare the API process's `PUFFERLAB_DATA_DIR` with the value used for
  `demo seed`, then rerun the idempotent seed.
- **Latency shows unavailable:** this is expected and required for synthetic evidence; zero
  milliseconds would be a false observation.
- **Replay is disabled:** this is expected. Do not copy synthetic UUIDs into the live replay API;
  synthetic identities are deliberately read/export-only.
- **A deep link returns 404:** the UUID belongs to another local database or the query is not part of
  that run. Return to `/runs` and follow a server-issued link from the current database.

Stop the Vite and Uvicorn processes when finished. The demo database remains under the ignored data
directory for the next idempotence check; removing local runtime data is optional and is not part of
the seed command.
