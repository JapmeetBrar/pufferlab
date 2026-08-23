# Observability and interview demo runbook

Use this runbook to explain what PufferLab observed, what it recomputed, and what it deliberately
does not claim. It also covers the safe operator response when stored evidence, replay, or an
optional provenance probe is unavailable.

For a provider-free first run, complete the [offline synthetic demo](synthetic-demo.md). For a live
replay, use a persisted live run whose original dataset namespace still exists and is ready.

## The four evidence sources

PufferLab never collapses these sources into a single story:

| Source | New provider work | What it can support | Forensic trace and time | Certainty |
|---|---|---|---|---|
| `stored_run` | No | Durable final ranks, judgments, metrics, client timing, and explicit original-stage unavailability | Stored `NOT_OBSERVABLE` observations have null trace/time; durable outcomes remain separate | `insufficient` for original stages |
| `live_replay_primary` | Yes, explicit | Only fields returned by the new production-shaped comparison for each selected config | One distinct primary trace per config; observation time equals `primary_observed_at` | `observed` for that new request only |
| `live_replay_counterfactual_probe` | Yes, optional and additional | Bounded BM25/vector candidate counts, ranks, and scores from a separate hybrid probe | Its own config, time, trace, and duration, disjoint from every primary source | `counterfactual` |
| `client_computed` | No additional call | Allowlisted arithmetic such as `weight / (rank_constant + rank)` over exact returned bounded inputs | Retains the one matching returned source trace and timestamp | `counterfactual` whenever a probe supplied an input |

Each `ForensicObservation` also names the exact `config_id` and `document_id` it describes. The
drawer filters on both identities; an observation for another config or document cannot be
substituted into the response.

### Stored run

Run history, run detail, regressions, query detail, export, deep-link refresh, and browser-history
restoration read SQLite only. Stored outcomes can establish durable final ordering, graded qrels,
quality metrics, available PufferLab client-wall timing, and failure/coverage state.

Milestone 2 outcomes did not persist original candidate-stage memberships, stage scores, provider
plan, or cache state. The forensic representation for those facts is therefore a typed
`NOT_OBSERVABLE` observation with `origin=stored_run`, `certainty=insufficient`, and null trace/time.
The absence of stage evidence does not erase the separately displayed final ranks or metrics.

### Primary live replay

Live replay is a new, request-scoped, cost-bearing comparison. It is never presented as a replay of
the provider's historical snapshot and never mutates the durable run. The browser sends only two
persisted config UUIDs and whether optional probes are requested. The server derives the run,
query text, graded qrels, dataset namespace, and immutable config bodies.

Before any credential or provider-capable factory is constructed, the server loads all 50
persisted judged queries and authenticates their source lock, curation order and identities, tags,
qrels, full content hash, query-set UUID, dataset binding, and canonical four-config run binding.
Foreign, duplicated, or tampered stored data fails as a direct redacted error with no provider work.

The primary response is production-shaped. Each selected config has a distinct trace; primary
forensic observations use the response's `primary_observed_at` time and their exact config trace.
They support statements only about fields actually returned by this new request.

### Counterfactual provenance probe

When **Include separate counterfactual provenance probes** is selected, PufferLab may make an
additional request for each selected hybrid config. A successful probe carries its own config UUID,
timestamp, trace, client-wall duration, bounded BM25/vector candidate counts, and returned
candidate memberships. Probe timing is not added to primary latency.

The provider may change between the primary request and the probe. A mismatch is evidence that the
snapshots differ, not evidence that the probe candidates caused the primary result. Probe-derived
observations are always counterfactual. If a probe fails, the primary response remains usable and
the response records a `ReplayFailedCounterfactualProbe` with the exact config, a disjoint trace,
its observation time, and the safe `provenance_probe_failed` warning.

### Client-computed evidence

PufferLab may calculate a bounded RRF contribution from returned rank, weight, and rank constant.
The contract rechecks the exact equation and source bounds. Client computation is transparent
arithmetic, not a provider explanation. It inherits the matching returned source time/trace and
becomes counterfactual if that source was a separate probe.

## How to demonstrate the boundary

1. Open `/runs` and choose a durable run. Point out that the page says its metrics come from SQLite
   and that refresh does not perform provider work.
2. Select a regression and follow **Inspect recorded query**. The UUID-only URL restores the exact
   run/query/config pair without including local licensed text.
3. Inspect one judged document. Start with **Stored run evidence** and the
   `NOT_OBSERVABLE · original stages` notice.
4. On a live-origin run only, confirm the intended server-only key, region, provider cost, and
   namespace readiness. Press **Run live replay (cost-bearing)** only after that deliberate check.
5. Compare the newly labeled primary results with the stored result. Describe them as two
   observations at different times, never as reconstruction of the original provider request.
6. If optional probes were requested, point out their separate timing/trace and counterfactual
   label. Show client-computed RRF arithmetic only when all bounded inputs were returned.

Opening the link, selecting configs, opening the drawer, refreshing, and Back/Forward navigation
remain read-only. The explicit replay button is the only query-detail action that can issue the
cost-bearing replay POST.

## Claims the UI must not make

Do not infer or state any of the following from final ranks, trace UUIDs, a later replay, or a
counterfactual probe:

- which provider cluster executed the request;
- whether a cache was warm or cold;
- the provider's internal query plan;
- that filtering ran before or after ANN;
- that probe-stage membership caused the primary order;
- a generated reranker rationale or hidden model reasoning;
- provider service latency from PufferLab client-wall measurements.

Use “observed in the new primary request,” “observed in a separate counterfactual probe,”
“client-computed from returned inputs,” or `NOT_OBSERVABLE`. Reranker evidence is limited to returned
rank and score movement.

## Safe troubleshooting

### Stored pages or deep links fail

- Confirm the API and seed/live CLI use the same `PUFFERLAB_DATA_DIR`.
- Return to `/runs` and follow a server-issued link from the current database. A UUID from another
  local database correctly returns 404.
- Start Uvicorn with `--workers 1`. If the ownership guard is held, stop the older PufferLab API
  instead of bypassing the guard.
- A partial, failed, cancelled, or interrupted run remains inspectable; missing attempts are
  explicit coverage states and must not be converted to zero-valued quality.

### Stored stages say `NOT_OBSERVABLE`

This is expected for existing durable outcomes. Do not infer original candidate membership from a
final rank or run trace. A new replay may add a separately timestamped observation but cannot fill
the original historical gap.

### Live replay is disabled or unavailable

- `synthetic_demo` is always read/export-only; use its stored workflow and do not try to replay its
  UUIDs.
- A live run needs the original exact namespace to remain ready, a matching server-only region, and
  `TURBOPUFFER_API_KEY` in `.env`. Policy permission on a run does not guarantee current namespace
  readiness.
- Immutable suite authentication rejects altered query text, IDs, tags, qrels, query-set identity,
  or config binding before credentials/provider factories. Restore the canonical local database;
  do not edit rows to make replay proceed.
- A direct `namespace_not_ready` or other redacted error leaves all stored evidence unchanged. Keep
  its safe trace UUID for local diagnosis, but do not attach query text, qrels, namespace, provider
  body, request headers, vectors, database files, or credentials to an issue.

### A counterfactual probe fails

Treat `failed_counterfactual_probes` as a separate bounded outcome. The associated primary result is
still valid for its own request. Do not retry automatically, fold the failed probe into primary
latency, or downgrade the stored evidence. A deliberate retry is additional provider work.

### Evidence looks cross-wired

Stop and report the safe response trace. The generated response contract requires every observation
target config/document to belong to the requested pair, exact-binds rank/score/count/presence/RRF
inputs to one returned primary or probe source, and requires all primary, successful-probe, and
failed-probe traces to be disjoint. Do not work around validation or log the raw response if it
contains local licensed data.

## Verification commands

These checks require no live provider call:

```bash
uv run python scripts/generate_openapi.py --check
uv run pytest backend/tests/contracts/test_forensics.py
uv run pytest backend/tests/application/test_evaluation_runtime.py

cd web
pnpm typecheck
pnpm test
pnpm build
```

Normal CI and the provider-free demo do not perform live replay. Any credentialed replay is an
explicit operator procedure with potential cost, and verification notes must remain sanitized.
