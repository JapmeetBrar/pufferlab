# Milestone 5 Execution Plan

Milestone 5 delivers the highest-value remaining P1 capability from the product brief: an
exact-bound expected-document diagnostic. From one recorded live query, an operator explicitly
selects one positively judged document and one authenticated run configuration. PufferLab then
shows what a new, bounded provider observation can establish about direct lexical/vector score,
stored-query filter predicates, candidate membership/cutoff, and an optional same-request no-filter
counterfactual.

This is not a reconstruction of the stored run. It is a new cost-bearing observation, and it does
not expose turbopuffer's internal plan, server-RRF execution, ANN cause, cache state, or reranker
rationale.

## Goal and success story

The operator can answer a narrower, useful question:

> For this exact judged document, in this exact authenticated query/config binding, what did one
> new same-snapshot diagnostic request directly observe, and what remains unknown?

The intended flow is:

```text
stored live run/query + authenticated qrels/config
  -> choose one grade>0 document and one run config (both blank until selected)
  -> explicit cost confirmation
  -> one strong-consistency diagnostic multi-query
  -> source-bound direct score/filter/candidate/counterfactual evidence
  -> typed finding or NOT_OBSERVABLE
```

Milestone 5 exits after the additive contract, provider/retrieval adapter, pure analysis,
authenticated diagnostic integration, and browser workflow are independently reviewed and merged,
then one finalization PR passes the complete gates and protected `main` is green.

## Why this P1 now

The expected-document probe completes the core interview story already scaffolded in the product:

- `ScoreSource.COMPUTE_ATTRIBUTE` and filter-result forensic values exist but lack an authenticated
  targeted source.
- Query detail already shows exact qrels and preserves a UUID-only selected-document route.
- Replay already supplies reusable complete Unix source/qrel/config authentication before
  provider-capable factories and proves how live sources remain separate.
- The provider already supports narrow typed BM25/ANN multi-query shapes and source-safe failure
  handling.

The generic importer has broader eventual reach but is not the next bounded review unit. It would
reopen schema mapping, `SourceDocument.attributes` ingestion, arbitrary suite sizes/config catalogs,
licensing, untrusted file parsing, identity/signing, and cleanup ownership together. Namespace
branching likewise needs a compatible registered dataset plus a separate ownership capability.

## Requirements

### Functional

- The user explicitly selects one path `document_id` whose authenticated qrel grade is greater
  than zero and one body `config_id` from the run's exact authenticated four-config catalog. There
  is no first-qrel or first-config default.
- The server completes the existing full-suite Unix authentication, dataset/config binding, and
  request validation before reading a credential, constructing an embedder/client, or issuing a
  request.
- One diagnostic provider request returns an exact ID-ranked target lookup plus the candidate lists
  applicable to the selected config. An optional no-filter view is available only when that stored
  query has a validated filter.
- The response distinguishes direct compute scores, locally evaluated filter predicates,
  stored-query candidate evidence, no-filter counterfactual evidence, qualified client-computed
  RRF, and unsupported final/reranker claims.
- Stored outcomes and any separately requested live replay remain visible when a diagnostic request
  fails through the normal fixed redacted API error.
- Synthetic, unauthenticated, unjudged, grade-zero, foreign, or stale selections reach zero
  provider/model factories.

### Non-functional

- The new diagnostic performs exactly one SDK `multi_query` call and at most one HTTP attempt. It
  has exactly two, three, or five ordered subqueries by mode and option, all at root strong
  consistency.
- `max_retries=0` is set on a dedicated diagnostic SDK client and no application retry wraps it.
  A timeout or ambiguous transport result is not replayed automatically.
- The UI describes cost as workload-dependent logical bytes and namespace concurrency, not as one
  billed query or a fixed dollar amount. Every subquery counts toward the namespace's concurrent
  query limit.
- The new diagnostic request/response adds no query text, filter values, returned target
  attributes, query/document vectors, namespace, credential, provider body, local path, model path,
  or exception detail. Existing query-detail fields retain their already reviewed behavior.
- No migration or durable diagnostic table is added. The observation is request-scoped.
- No automatic provider request occurs on route load, qrel selection, config selection, drawer
  open, refresh, polling, Back/Forward, or retry without another explicit action.

## Non-requirements and deferred work

- Generic BEIR/JSONL/qrels import, browser upload, arbitrary schemas/suite sizes, and config editing.
- Customer namespace branching or any new deletion/cleanup authority.
- Warm-cache actions, asserted cache tiers, ANN `_debug/recall`, or causal ANN explanations.
- Durable diagnostic traces, signed whole-database evidence, provider billing history, or reports
  over exported JSON.
- Executing the local reranker, loading target document text for reranking, or generating a model
  rationale during the diagnostic.
- Claiming the diagnostic candidate lists reproduce a stored run or a separate live replay.
- General filter editors or provider expressions beyond the existing validated neutral AST.

## Component and data flow

```text
Browser UUID selection
  |
  v
Dedicated diagnostic request contract
  |
  +--> authenticate all 50 queries/qrels + exact dataset/four-config catalog
  |       (provider/model factories still poisoned)
  |
  +--> validate target grade>0, selected config, no-filter eligibility
  |
  +--> expected-document application service
          |
          +--> query embedder only when vector signal is required
          |
          +--> dedicated diagnostic provider
          |      one strong multi_query / exactly 2, 3, or 5 ordered subqueries
          |
          +--> pure filter + cutoff + RRF analysis
          |
          +--> source-binding response validator --------> diagnostic evidence
```

The diagnostic service is provider-neutral above its adapter. It receives only server-derived
query/config/filter/target inputs. It constructs no reranker and reads no target text beyond the
filter attributes returned by the exact lookup. BM25-only diagnostics construct no embedder;
vector and hybrid modes perform one exact pinned query-embedding operation and never expose the
vector.

## Frozen API contract

Add one dedicated endpoint. Existing replay request/response contracts remain unchanged:

```text
POST /api/v1/eval-runs/{run_id}/queries/{query_id}/documents/{document_id}/diagnostic
```

The body is intentionally smaller than replay:

```python
class ExpectedDocumentDiagnosticRequest(ContractModel):
    contract_version: Literal[1] = 1
    config_id: UUID
    include_no_filter_counterfactual: bool = False
```

- `config_id` must resolve to exactly one config in the authenticated run's canonical four-config
  catalog. It is not restricted to the page's current left/right comparison pair.
- Path `document_id` must equal exactly one qrel document ID for the authenticated query and that
  qrel must have `relevance_grade > 0`.
- `include_no_filter_counterfactual=true` requires non-null `binding.query.filters`. Evaluation
  execution already applies that exact stored query filter as `filter_override`; the canonical
  configs themselves require `config.filters is None`. A client cannot supply or override either.
  Ineligible input is rejected before credential/embedder/provider factories.
- The response repeats the exact run, query, config, and target IDs. Foreign or substituted
  identities are contract-invalid.

The success response has its own source origin:

```python
class ExpectedDocumentDiagnosticResponse(ContractModel):
    contract_version: Literal[1] = 1
    run_id: UUID
    query_id: UUID
    data_origin: Literal["live"]
    origin: Literal["live_expected_document_diagnostic"]
    config_id: UUID
    config_mode: RetrievalMode
    target_document_id: UUID
    included_no_filter_counterfactual: bool
    observed_at: AwareDatetime
    trace_id: UUID
    duration_ms: float
    embedding_duration_ms: float | None
    subqueries: list[DiagnosticSubquerySummary]  # exact legal count 2, 3, or 5
    target: DiagnosticTargetLookup
    filter_evidence: list[FilterPredicateEvidence]
    candidate_evidence: list[CandidateCutoffEvidence]
    observations: list[ForensicObservation]
    observability_notice: Literal["new_live_diagnostic_not_original_run"]
```

`EvidenceOrigin` gains `LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC`. A successful diagnostic is bound to one
provider trace/time. Subqueries do not invent traces; each carries only an exact `ordinal` and
allowlisted `role` under that one source. Provider, response-shape, analysis, or close failures use
the normal fixed redacted API error response; they never return HTTP 200 with partial diagnostic
evidence. Existing stored/replay UI state remains intact.

The response validator uses `config_mode` and `included_no_filter_counterfactual` to require the
exact legal role sequence: BM25/vector modes have two roles without and three with no-filter;
hybrid modes have three without and five with no-filter. It also requires only mode-available direct
scores and candidate stages. M5-D cross-checks both echoes against the authenticated config and
request, requires `data_origin=live`, and emits only the fixed observability notice rather than
free-form copy.

### Subquery and evidence shapes

Subquery roles are an exact mode/option-dependent sequence drawn from:

1. `target_lookup`
2. `stored_query_bm25_candidates`
3. `stored_query_ann_candidates`
4. `no_filter_counterfactual_bm25_candidates`
5. `no_filter_counterfactual_ann_candidates`

The internal M5-B provider result carries each complete ordered candidate list as
`(document_id, rank, observed_score)` with no attributes; exact canonical configs bound each list
to `result_k=50` or `candidate_k=100`. Those rows are input only to M5-C analysis and never enter the
public contract. A public subquery summary contains ordinal, role, requested limit, returned count,
selected-target presence/rank/score, a boundary score only for a full list, and no unrelated
document IDs. Candidate evidence names `stored_query` or `no_filter_counterfactual`, BM25 or ANN,
and repeats only those target-scoped facts plus a qualified client-computed result. M5-D recomputes
and cross-checks every safe summary and qualified RRF against the internal rows before constructing
the response; Pydantic validators enforce the public shape, identities, and exact mode/role
relationships without pretending the browser can authenticate hidden rows. Numbers are finite and
the only document ID in the diagnostic response is the selected canonical target UUID.

When a target appears in a stored-query or no-filter candidate list, its BM25 score or VectorDist
must match the corresponding exact-lookup compute value with
`math.isclose(rel_tol=1e-12, abs_tol=1e-15)`. The same tolerance defines equality at a candidate or
qualified-RRF boundary; a tie is `NOT_OBSERVABLE`. A score mismatch is an invalid provider response,
not a finding.

The exact target lookup reports `available` plus optional direct weighted-BM25 and exact
VectorDist scores. It never returns filter attribute values. Filter evidence contains only a stable
predicate path/ordinal, field, operator, and `matched | not_matched | not_observable`; it omits the
predicate value and observed attribute value.

The existing `ForensicObservation` remains the rendered finding envelope. M5-A may add bounded
evidence value types for direct compute score and cutoff relation rather than pretending a direct
lookup score is candidate-stage membership. Every item is source-bound to the exact diagnostic
config/target/trace/time and cross-checked against the typed diagnostic source.

## Exact provider request

The pinned dependency is turbopuffer Python SDK `2.9.0`. The diagnostic adapter uses one dedicated
async client and exactly the official `https://{region}.turbopuffer.com` endpoint. It supplies an
owned `httpx.AsyncClient` with `trust_env=False`, `follow_redirects=False`, a ten-second timeout,
finite connection limits, and a pre-transport hook that validates the exact POST destination/method
and replaces the complete case-insensitive header set with exactly one Authorization value plus
the exact SDK-required POST protocol headers, including content type and content length. SDK
base/custom-header, proxy, and redirect environment settings cannot alter
the request. SDK construction sets `_strict_response_validation=True`. The SDK has
`max_retries=0`, no application retry, and one
`namespace.multi_query(...)` call:

```text
consistency = {level: strong}              # root only
queries[0] = target lookup                 # always
queries[1..] = mode/filter-dependent raw candidates
rerank_by = omitted                        # raw lists must remain distinct
```

The target lookup is exactly:

```text
rank_by = ("id", "asc")
filters = ("id", "Eq", canonical_target_uuid)
limit = 1
include_attributes = only stored-query-filter fields needed for local evaluation
compute_attributes = weighted BM25 and/or VectorDist required by the selected config
```

Ranking by ID is essential: score-ranked lookup would exclude a BM25 score of zero. Computed
attributes do not affect matching/ranking. Weighted BM25 uses the exact persisted lexical
expression; VectorDist uses the exact persisted vector attribute, distance metric, and newly
computed query vector.

Stored-query candidate subqueries use the exact stored query filter and selected mode's immutable
expression/metric. BM25/vector-only candidate limits equal `result_k`; hybrid candidate limits
equal `candidate_k`. No-filter subqueries are identical except that the stored query filter is
omitted. Any unknown mode or malformed mode/spec/config identity rejects before sensitive
factories. Modes yield these bounds:

| Mode | Normal roles | With eligible no-filter |
|---|---:|---:|
| BM25 | lookup + BM25 = 2; limit `result_k`; no embedder | 3 |
| vector | lookup + ANN = 2; limit `result_k`; one embedding | 3 |
| hybrid RRF | lookup + BM25 + ANN = 3; limit `candidate_k`; one embedding | 5 |
| hybrid rerank | lookup + BM25 + ANN = 3; limit `candidate_k`; one embedding | 5 |

The SDK returns multi-query results in request order. Results carry no role tag, so the adapter
relies on that official guarantee and binds each result to its fixed request ordinal/role; it does
not claim an independent general reorder detector. It rejects a wrong result count,
role-incompatible shape, more than one lookup row, wrong lookup ID, malformed row, missing required
computed field, duplicate candidate ID/rank, or non-finite score. It does not reuse
`_row_to_document`, whose score semantics assume `$dist`; attribute-ranked lookup correctly has no
`$dist`.

Every candidate list must satisfy this complete integrity table before analysis: document IDs are
unique; derived ranks are exactly contiguous `1..returned_count`; returned count does not exceed the
requested limit; BM25 scores are finite, nonnegative, and monotonically non-increasing; VectorDist
values are finite, nonnegative, and monotonically non-decreasing; and ties are allowed. A boundary
is the final row only when returned count equals the requested limit. A present target's score must
agree with its exact-lookup compute value under the frozen tolerance. Any violation fails closed.

Zero lookup rows produce typed `target_unavailable_in_diagnostic_snapshot` and only
`NOT_OBSERVABLE`; this means the UUID was not visible in that diagnostic namespace snapshot, not
that the qrel is false or the stored run was corrupt. Downstream filter/cutoff/RRF claims are
suppressed. If any candidate list contains the target while lookup returned zero rows, the provider
response is invalid rather than self-consistently explained.

The public response omits provider billing fields in M5. Documentation discloses that actual cost
depends on logical bytes queried/returned and namespace configuration. One SDK call is not called
one billed query; each of the mode-bound two, three, or five subqueries consumes namespace query
concurrency.

## Filter and cutoff semantics

The pure filter evaluator supports only the already validated AST (`and`, `or`, `not`, `eq`,
`not_eq`, `lt`, `lte`, `gt`, `gte`, `in`, `contains_any`) and compiled schema types. Logical
evaluation is three-valued so unsupported or type-incomparable data cannot become a false causal
claim. The diagnostic cannot distinguish a missing returned attribute from a present provider null,
so both become one local null-or-missing state and no finding claims which occurred. For that state,
the frozen official-semantics matrix is: `Eq null=true`, `NotEq null=false`, `Eq nonnull=false`,
`NotEq nonnull=true`, `Lt/Lte nonnull=true`, `Gt/Gte=false`, and `In/ContainsAny=false`. Valid
non-null values use schema-typed comparisons only. Tests cover both raw shapes and nested
`And`/`Or`/`Not`; invalid operand types/shapes reject and Python truthiness/coercion is forbidden.
M5-C freezes the truth tables and stable predicate paths. Official semantics plus exhaustive
provider-free fake/SDK tests enable `FILTER_PREDICATE_FAILED`; an isolated live parity test remains
optional and separately authorized rather than a merge prerequisite.

The diagnostic accepts at most 16 predicates, 31 total AST nodes, and depth 8, matching the public
evidence bound while keeping lookup attributes and evaluation finite. An over-bound stored AST
returns a provider-free fixed validation/evidence error before credential/provider work; it does
not create a trace-bearing `NOT_OBSERVABLE` observation. The durable query remains readable.
Predicate values never enter the response.

Cutoff comparisons are direction-aware and only compare values from the one diagnostic trace:

- Target membership and rank in a returned list are observed facts.
- BM25 direct score zero supports `NO_LEXICAL_SCORE` because official query behavior excludes
  zero-score rows.
- A clear worse-than-boundary direct score in a full returned list supports an outside-candidate
  finding. Equality/ties are `NOT_OBSERVABLE`; PufferLab does not invent a tie-break.
- An exact VectorDist that clearly beats the observed ANN boundary while the target is absent is an
  **observed ANN candidate miss in this diagnostic**, never an internal-cause claim.
- A short list, missing boundary, malformed score, or direction mismatch cannot support a cutoff
  claim. Structurally contradictory provider results fail closed.
- Stored-query provider facts use the diagnostic origin with observed certainty. Local filter,
  cutoff, and RRF arithmetic is `client_computed`; no-filter provider facts and computations are
  counterfactual. None proves the provider's internal filter/ANN execution order.

For hybrid configs, PufferLab may reconstruct weighted RRF from the returned raw diagnostic lists
using the persisted weights/rank constant. RRF from stored-query lists is `client_computed` from
current observed provider inputs; RRF from no-filter lists is `client_computed` and counterfactual.
Because `rerank_by` is deliberately omitted, server RRF/final order is not directly observed;
equal reconstructed cutoff scores are `NOT_OBSERVABLE`. For hybrid RRF, the qualified reconstructed
fusion cutoff is `result_k`. For hybrid rerank, only admission to the would-be reranker at
`reranker.depth` is client-computed; post-reranker/final `result_k`, `RERANKED_DOWN`, and reranker
cutoff remain `NOT_OBSERVABLE` because the diagnostic constructs no reranker and returns no target
text. Findings are mode-scoped and never borrow a score or boundary from another config.

Stored-run ranks and separate live-replay ranks come from different snapshots. They may be displayed
alongside the diagnostic but never participate in its cutoff arithmetic or causal wording.

## Validation, error, and cancellation behavior

The operation order is fixed:

1. Decode strict request models.
2. Load run/query and authenticate the complete exact Unix suite; reject synthetic origin.
3. Derive the canonical four configs and validate the selected config ID.
4. Validate one selected config, one positive-qrel target, stored-query-filter bounds, and
   no-filter eligibility.
5. Resolve the namespace only from the authenticated live `DatasetVersion`; require the current
   configured region to exact-match authenticated stored `run.environment.turbopuffer_region` and
   official region syntax, then resolve the required runtime. The browser supplies neither value.
6. Construct only the diagnostic resources needed by the selected mode.
7. Execute one diagnostic SDK call, validate it, analyze it, close resources, then render.

Steps 1–4 poison credential, catalog-runtime, provider, embedder, reranker, and network factories.
Invalid binding returns the existing direct redacted validation/evidence error. A diagnostic
provider/shape/analysis/close failure returns one normal fixed redacted API error; no provider
detail crosses the API and no partial diagnostic response is rendered. Existing stored and replay
evidence already in the browser remains unchanged.

The diagnostic HTTP/SDK provider is request-owned and closes exactly once under success, provider
error, validation error, cancellation, repeated cancellation, `KeyboardInterrupt`, and
`SystemExit`. The current embedder has no close protocol; its query text, embedding, and vector
references are scrubbed/released before crossing the sensitive frame rather than pretending to
close it. Because embedding currently enters `asyncio.to_thread`, a started embedding is shielded
and drained under first or repeated cancellation; its result/vector is discarded, no provider
client is constructed and no SDK call follows, sensitive frames are scrubbed, and a fresh
cancellation is re-raised. Blocked-thread cancellation, repeated cancellation, `SystemExit`, and
marker-retention tests freeze this boundary. Provider cleanup is drained under repeated
cancellation; original cancellation wins and emits no partial response. Sensitive exceptions are
caught, detached, and cleared inside the owning frame, and value-free public results are constructed
after that frame exits. A close failure discards the diagnostic evidence.

## Security and privacy

- Credentials remain `SecretStr`/server-only and are unwrapped only inside the scrubbed diagnostic
  client construction boundary.
- Diagnostic browser requests contain UUIDs and booleans only. Query text, qrels, config/filter
  bodies, namespace, region, and vectors are derived server-side and are not repeated in the new
  diagnostic response; existing query-detail fields retain their current contract.
- New diagnostic evidence contains only the selected target plus run/query/config UUIDs, enums,
  finite target-scoped scores/counts/ranks, booleans, trace IDs, times, and one allowlisted notice.
  It contains no unrelated candidate UUID, raw filter/attribute value, target text, provider body,
  bill, or local path.
- URLs keep only run/query/left/right/document UUIDs. The selected target is visibly a positive
  judgment before the action is enabled.
- Secret/artifact tests scan source, generated browser assets, exception graphs, logs, and failure
  output with synthetic markers. Live evidence and screenshots remain untracked.

## Dependency and branch graph

```text
M5-0 codex/m5-plan
  |
  v
M5-A codex/m5-diagnostic-contracts
  |\
  | +--> M5-B codex/m5-diagnostic-provider
  |
  +----> M5-C codex/m5-diagnostic-analysis
              \                         /
               +------ M5-D -----------+
                       codex/m5-diagnostic-api
                              |
                              v
                    M5-E codex/m5-diagnostic-ui
                              |
                              v
                    M5-F codex/m5-finalization
```

M5-B and M5-C may proceed in parallel after the contract freeze because their production file
ownership is disjoint. M5-D begins only after both are reviewer-merged. M5-F is the one goal-closing
PR.

## Review units and acceptance

### M5-0 — Architecture and contract audit

- **Files:** this plan, `implementation-plan.md`, and `progress.md` only.
- **Acceptance:** close M4 canonically; freeze requirements/nonrequirements, exact request/evidence
  shapes, one-call/subquery/cost bounds, SDK behavior, trust/observability limits, branch ownership,
  rollback, demo, and completion criteria; no implementation or live action.

### M5-A — Additive contract freeze

- **Files:** forensic/common Pydantic contracts and tests plus `contracts.md` only.
- **Acceptance:** strict additive request/response/evidence shapes, distinct diagnostic origin,
  exact source binding and finite bounds, and duplicate/non-finite/cross-trace/tie attacks. Request
  shape requires explicit UUIDs/boolean only; positive-qrel/run/config semantics belong to M5-D.
  Because unreachable models are not emitted, OpenAPI/generated TypeScript must have zero delta.
  No provider, persistence, route, application, or UI behavior.

### M5-B — Provider and retrieval diagnostic adapter

- **Files:** provider-neutral diagnostic types/protocol, turbopuffer adapter, retrieval composition,
  fake/real-SDK serialization tests, and a separately authorized optional isolated live test.
- **Acceptance:** exact one call/one attempt, root strong consistency, exact mode/option-bound 2, 3,
  or 5 ordered roles, hardened official endpoint/HTTP/header boundary, strict SDK response
  validation, ID-ranked lookup, required compute attributes, mode/filter shapes, no `rerank_by`,
  strict internal full-row response decoder,
  zero-row/contradiction behavior, environment-poison attacks, resource drainage, and redacted
  exception graphs. A started threaded embedding drains under cancellation, discards its result,
  constructs no provider client, and leaves no sensitive exception frame. Any live test creates
  and cleans only its internally generated test namespace
  after independent safety review; it is not required by normal CI or merge acceptance.

### M5-C — Pure filter/cutoff/forensic analysis

- **Files:** pure evaluator/analysis modules and exhaustive tests only.
- **Acceptance:** no SDK/FastAPI/SQLAlchemy/filesystem/network/model imports; exact tri-state AST
  truth tables; higher/lower score directions; zero, short/full lists, equality/ties, missing
  boundary, ANN miss, qualified RRF, and reranker/final `NOT_OBSERVABLE` cases; deterministic bounded
  findings from internal bounded rows, target-scoped safe summaries, and forbidden-causal-copy or
  unrelated-ID exposure tests.

### M5-D — Authenticated diagnostic integration

- **Files:** evaluation application/runtime, dedicated route, OpenAPI/generated TypeScript, focused
  application/API tests.
- **Acceptance:** full source/qrel/config authentication and positive-target/no-filter validation
  before every sensitive factory; namespace only from the authenticated live dataset and current
  region exact-bound to the stored run plus official syntax; dedicated diagnostic source
  separation; normal fixed API errors with no partial 200; no writes/migrations;
  cancellation, blocked-embedding, and close matrix; secret/provider-body/exception-graph
  redaction; internal rows are cross-checked into target-only summaries and then discarded;
  synthetic and all tampering paths make zero calls; reachable OpenAPI and generated TypeScript
  move together.

### M5-E — Expected-document UI and browser flow

- **Files:** generated-type consumers under `web/src/features/evals/**`, focused tests, provider-free
  Playwright/axe updates, operator copy.
- **Acceptance:** only grade-positive qrels can be selected; no silent target/config default;
  no-filter is disabled when ineligible; exact cost/call/subquery disclosure; no request before
  explicit action; in-flight work aborts and prior success/error clears when run/query/config/target
  or option changes and before a same-input rerun/retry; stale responses cannot render under a new
  UUID selection; evidence origin and `NOT_OBSERVABLE` copy are accessible/responsive. A fake
  live-origin seeded run exercises initially blank target/config controls followed by explicit
  positive-target/config selection, click/render,
  option reset, and late-response suppression without real network; the synthetic journey remains
  disabled and zero-POST.

### M5-F — QA and goal finalization

- **Files:** focused runbook/README/progress updates and bounded verification fixes only.
- **Acceptance:** full `make check`, generated/schema drift, provider-free E2E/axe, artifact/privacy,
  diff/untracked/residue/process gates and a fake live-origin/offline-SDK exact-shape browser demo
  that covers the application mode matrix and performs zero real network. Live parity
  and a credentialed operator rehearsal remain optional and outside completion unless the user later
  authorizes them; a missing compatible live target is reported, never fabricated. Independent
  reviewer-only merge and green protected `main` close the goal.

## Manual operator demo (separately authorized live path)

1. Open a recorded live query and its exact graded qrels; route load performs provider-free reads.
2. Select one `grade > 0` document and one config explicitly.
3. Review the notice: one diagnostic SDK call, exactly 2, 3, or 5 subqueries by mode/option,
   workload-dependent logical-byte cost, and no original-run reconstruction.
4. Run the explicit action.
5. Inspect direct BM25/VectorDist, stored-query filter results, candidate boundary/membership, and
   qualified RRF or `NOT_OBSERVABLE` states in the selected document drawer.
6. If the stored query has a filter, opt into the no-filter counterfactual and run again; if it has
   no filter, the option stays disabled and no request is sent.
7. Change target/config and verify the prior evidence disappears before another explicit action.

The current canonical Unix queries have no stored filters, so their honest demo covers
direct score and candidate-cutoff evidence while the no-filter control remains disabled. Filter and
no-filter behavior is proven in normal gates with official-semantics unit tests and provider/SDK
fakes. An isolated live fixture is optional, separately safety-reviewed, and not an M5 completion
requirement. M5 does not mutate the canonical catalog merely to make that UI state appear.

## Rollback

- M5-A is additive and unreachable until M5-D mounts the dedicated route. Existing replay remains
  unchanged. Revert it before dependents if the contract is rejected.
- M5-B/C are non-persistent. Reverting removes adapter/analysis behavior without changing SQLite or
  namespaces.
- M5-D adds no migration and writes no diagnostic evidence. In-flight rollback is ordinary request
  cancellation plus exact resource close.
- M5-E can hide/remove the action without changing backend state. Cached browser responses are
  request-scoped and never authoritative.
- No rollback path deletes, branches, rewrites, or reingests a namespace. Any isolated live fixture
  cleanup uses only its independently reviewed ownership harness.

## Risks and tradeoffs

| Risk | Response |
|---|---|
| Diagnostic snapshot differs from stored/replay evidence | Label one diagnostic trace/time; never make cross-source causal comparisons. |
| Multi-query cost is misunderstood | Publish exact call/subquery bound and workload-dependent logical-byte/concurrency wording. |
| Direct-score lookup drops score-zero rows | Rank by ID; compute scores separately. |
| SDK result ordering/shape drifts | Bind fixed ordinal/role and real-SDK serialization/decoder tests. |
| Filter local semantics diverge | Freeze official missing/null/type semantics and exhaustive provider-free parity fakes; keep a live test optional. |
| ANN absence is overexplained | Report only observed miss relative to exact distance/boundary; internal cause remains unknown. |
| RRF/reranker is overclaimed | No `rerank_by`, only qualified client RRF; reranker/final cutoff stays `NOT_OBSERVABLE`. |
| Failure leaks provider detail | Normal fixed API error, detached exception graph, no partial diagnostic response. |
| Current canonical queries have no filter | Keep no-filter disabled; validate with fakes/optional isolated fixture rather than changing the catalog. |

## Completion criteria

- [ ] M5-0 is independently reviewed and merged before implementation starts.
- [ ] Strict contracts plus the dedicated generated route bind one exact positive-qrel target,
  selected config, source, trace/time, and all bounded evidence while the existing replay contract
  remains unchanged.
- [ ] One dedicated strong multi-query performs exactly 2, 3, or 5 ordered subqueries and one HTTP
  attempt, with exact pinned-SDK shapes and resource/error/cancellation coverage.
- [ ] Pure analysis handles filter, score direction, candidate cutoff, ties, ANN miss, qualified
  RRF, and unsupported reranker/final claims without causal overreach.
- [ ] Authenticated integration rejects every foreign/stale/synthetic/ineligible input before
  sensitive factories and preserves source separation and safe failures.
- [ ] Browser selection, stale-state reset, cost disclosure, accessibility, responsive layout, and
  provider-free zero-call behavior pass deterministic tests.
- [ ] Every M5 PR is independently reviewed and reviewer-only merged; protected-main Backend and
  Frontend checks pass after the reviewed finalization PR.

## Current primary sources

- [turbopuffer query API](https://turbopuffer.com/docs/query): rank-by-ID, `compute_attributes`,
  score-zero exclusion, same-snapshot multi-query, ordered results, strong consistency, and the
  16-subquery service limit.
- [turbopuffer hybrid guide](https://turbopuffer.com/docs/hybrid): computed cross-signals without
  extra round trips and first-stage/reranker separation.
- [turbopuffer limits](https://turbopuffer.com/docs/limits): multi-query, concurrency, result, and
  computed-attribute bounds.
- [turbopuffer pricing](https://turbopuffer.com/pricing): workload-dependent query pricing.
- [official Python SDK](https://github.com/turbopuffer/turbopuffer-python): async lifecycle,
  timeout, and retry configuration. Repository lock pins SDK `2.9.0`.
