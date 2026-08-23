# Offline synthetic demo

Seed a complete local evaluation dashboard without a turbopuffer account, model download, or
network access:

```bash
PUFFERLAB_DATA_DIR=./data uv run pufferlab demo seed
```

The command migrates the configured SQLite database and idempotently writes one deterministic
PufferLab-authored dataset, 50 judged queries, the canonical BM25/ANN/server-RRF/local-reranker
configuration order, and 200 successful outcomes. It prints only durable IDs, origin/timing labels,
and counts. It does not write an export file, vectors, credentials, or provider responses.

The seeded run is explicitly `data_origin=synthetic_demo` and read/export-only. Its quality metrics
are recomputed from authored ranked document IDs and qrels with the normal evaluation engine.
Because no searches ran, `timing_source=synthetic_unavailable`: total and stage timings are null or
absent, latency percentiles are null with zero samples, and provider traces and candidate counts are
absent. Re-running the command verifies the same content-addressed rows and canonical export bytes
instead of creating a second run.
