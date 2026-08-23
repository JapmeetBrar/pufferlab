# PufferLab Shared Contracts v1

- **Status:** Implemented v1 contract; additive changes require regeneration and review
- **Source of truth in code:** Pydantic models under `backend/pufferlab/contracts/`
- **Frontend types:** generated from FastAPI OpenAPI; do not maintain a second handwritten domain model

This document fixes vocabulary, request/response boundaries, identity, score semantics, error behavior, and file ownership. A workstream may extend a contract only through a reviewed change here plus contract tests.

## 1. Conventions

- JSON fields use `snake_case` on both Python and TypeScript sides.
- IDs are opaque UUID strings except turbopuffer document IDs, which are UUIDv5 values serialized as canonical strings.
- Times are ISO-8601 UTC strings. Durations are floating-point milliseconds and end in `_ms`.
- Rankings are 1-based in API/UI contracts.
- `null` means known absence; omitted fields are not requested/not computed.
- Configs, dataset versions, query sets, and completed runs are immutable revisions.
- Every top-level response includes `contract_version: 1`.
- Every catalog, run, regression, query-detail, and export projection carries
  `data_origin: "live" | "synthetic_demo"`. Browser input never selects this value.
- Secrets, raw query vectors, and stored document vectors never cross the API boundary.

Shared JSON value alias:

```python
JsonValue = (
    str | int | float | bool | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
```

## 2. Score semantics

Never compare unlike raw scores directly. Every score carries its meaning and direction.

```python
class ScoreKind(str, Enum):
    BM25 = "bm25"
    VECTOR_DISTANCE = "vector_distance"
    RRF = "rrf"
    RERANKER = "reranker"

class ScoreDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class ObservedScore(BaseModel):
    kind: ScoreKind
    value: float
    direction: ScoreDirection
    source: Literal["turbopuffer_dist", "compute_attribute", "client_computed", "reranker"]
```

Expected direction:

| Kind | Direction |
|---|---|
| `bm25` | higher is better |
| `vector_distance` | lower is better |
| `rrf` | higher is better |
| `reranker` | higher is better |

## 3. Filter AST

The API never accepts raw turbopuffer tuples from the browser. It accepts a validated neutral AST, and the turbopuffer adapter owns translation.

```json
{"kind":"predicate","field":"source","op":"eq","value":"unix"}
```

```json
{
  "kind":"logical",
  "op":"and",
  "children":[
    {"kind":"predicate","field":"source","op":"eq","value":"unix"},
    {"kind":"predicate","field":"created_at","op":"gte","value":"2012-01-01T00:00:00Z"}
  ]
}
```

P0 operators:

- logical: `and`, `or`, `not`
- predicate: `eq`, `not_eq`, `lt`, `lte`, `gt`, `gte`, `in`, `contains_any`

```python
PredicateOp = Literal[
    "eq", "not_eq", "lt", "lte", "gt", "gte", "in", "contains_any"
]

class FilterPredicate(BaseModel):
    kind: Literal["predicate"] = "predicate"
    field: str
    op: PredicateOp
    value: JsonValue

class FilterLogical(BaseModel):
    kind: Literal["logical"] = "logical"
    op: Literal["and", "or", "not"]
    children: list["FilterNode"]

FilterNode = Annotated[FilterPredicate | FilterLogical, Field(discriminator="kind")]
```

`not` has exactly one child; `and` and `or` have at least one. `in` and `contains_any`
require array operands; the other P0 comparison operators require a JSON scalar. All numeric values
must be finite, including values nested inside arrays or objects.

Before any query or embedding runs, the complete AST is validated against the active dataset's
compiled namespace schema. Unknown and non-filterable attributes fail validation; values must match
the attribute's scalar type, and `contains_any` is valid only for array attributes. The tiny fixture
marks only `external_id` as filterable. Invalid filters return the direct `422 validation_error`
contract and are never passed through as arbitrary provider expressions.

## 4. Dataset and index profile

```python
class FtsProfile(BaseModel):
    tokenizer: str = "word_v4"
    case_sensitive: bool = False
    language: str = "english"
    stemming: bool = False
    remove_stopwords: bool = False
    ascii_folding: bool = False
    max_token_length: int = 39
    k1: float = 1.2
    b: float = 0.75
    k3: float = 8.0

class IndexProfile(BaseModel):
    id: str
    embedding_provider: Literal["sentence_transformers"]
    embedding_model: str
    embedding_revision: str
    vector_attribute: str = "vector"
    vector_dimensions: int
    vector_dtype: Literal["f16", "f32", "i8"]
    distance_metric: Literal["cosine_distance", "euclidean_squared"]
    fts_profile: FtsProfile
    schema_hash: str

class DatasetVersion(BaseModel):
    id: UUID
    slug: str
    version: str
    data_origin: Literal["live", "synthetic_demo"] = "live"
    namespace: str
    index_profile: IndexProfile
    document_count: int
    corpus_hash: str
    status: Literal["pending", "ingesting", "indexing", "ready", "failed"]
    created_at: datetime
```

`live` revisions require a non-empty provider namespace. The single `synthetic_demo` revision uses
an empty namespace and may only cross read/export surfaces. It must fail before credentials or any
provider-related factory is constructed on create, recovery, or replay. The default `live` value is
backward compatible with stored Milestone 2 JSON; M3 catalog projections always expose the origin
explicitly.

Versioned dataset list/detail responses wrap the immutable revision and repeat its matching origin.

Identity invariant:

```text
document_uuid = UUIDv5(PUFFERLAB_NAMESPACE_UUID, dataset_version + ":" + external_id)
```

## 5. Retrieval configuration

```python
class RetrievalMode(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"

class LexicalSpec(BaseModel):
    title_weight: float = 2.0
    body_weight: float = 1.0

class VectorSpec(BaseModel):
    attribute: str = "vector"
    embedding_model: str

class RrfSpec(BaseModel):
    execution: Literal["server"] = "server"
    rank_constant: int = 60
    weights: tuple[float, float] = (1.0, 1.0)  # lexical, vector

class RerankerSpec(BaseModel):
    provider: Literal["sentence_transformers"]
    model: str
    revision: str
    depth: int = 50

class RetrievalConfig(BaseModel):
    id: UUID
    revision: int
    name: str
    dataset_version_id: UUID
    mode: RetrievalMode
    result_k: int = 10
    candidate_k: int = 100
    consistency: Literal["strong", "eventual"] = "strong"
    filters: FilterNode | None = None
    lexical: LexicalSpec | None = None
    vector: VectorSpec | None = None
    rrf: RrfSpec | None = None
    reranker: RerankerSpec | None = None
    config_hash: str
    created_at: datetime
```

Validation invariants:

- `bm25` requires `lexical` only.
- `vector` requires `vector` only.
- `hybrid_rrf` requires `lexical`, `vector`, and `rrf`.
- `hybrid_rerank` requires all four specs.
- `candidate_k >= result_k`; `reranker.depth <= candidate_k`.
- All configs in one comparison reference the same dataset version unless explicitly running an index-profile comparison.

## 6. Search comparison

### Request

```python
class SearchCompareRequest(BaseModel):
    contract_version: Literal[1] = 1
    query_text: str
    config_ids: list[UUID]  # 2..4
    query_id: UUID | None = None
    filter_override: FilterNode | None = None
    expected_document_ids: list[UUID] = Field(default_factory=list)
    debug_provenance: bool = True
```

### Hit and provenance

```python
class HighlightOffset(BaseModel):
    start: int
    end: int

class HighlightFragment(BaseModel):
    text: str
    fragment_start: int | None = None
    fragment_end: int | None = None
    match_offsets: list[HighlightOffset] = Field(default_factory=list)

class StageMembership(BaseModel):
    stage: Literal["bm25_candidates", "vector_candidates", "rrf", "reranker", "final"]
    rank: int
    score: ObservedScore | None = None

class SearchHit(BaseModel):
    document_id: UUID
    external_id: str
    title: str
    body_excerpt: str
    url: str | None = None
    relevance_grade: int | None = None
    final_rank: int
    final_score: ObservedScore | None = None
    stage_membership: list[StageMembership]
    highlights: list[HighlightFragment] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

class StageTiming(BaseModel):
    stage: Literal["embed", "turbopuffer", "provenance_probe", "fusion", "rerank", "total"]
    duration_ms: float
    measurement: Literal["client_wall_clock"] = "client_wall_clock"

class ConfigSearchResult(BaseModel):
    config: RetrievalConfigSummary
    hits: list[SearchHit]
    timings: list[StageTiming]
    candidate_counts: dict[str, int]
    warnings: list[ApiWarning]
    trace_id: UUID

class RetrievalConfigSummary(BaseModel):
    id: UUID
    revision: int
    name: str
    mode: RetrievalMode
    config_hash: str

class RetrievalConfigListResponse(BaseModel):
    contract_version: Literal[1] = 1
    configs: list[RetrievalConfigSummary]

class RankMovement(BaseModel):
    document_id: UUID
    ranks_by_config: dict[UUID, int | None]
    max_absolute_delta: int | None

class PairwiseOverlap(BaseModel):
    left_config_id: UUID
    right_config_id: UUID
    left_count: int
    right_count: int
    intersection_count: int
    jaccard: float

class SearchCompareResponse(BaseModel):
    contract_version: Literal[1] = 1
    query_text: str
    query_id: UUID | None
    results: list[ConfigSearchResult]
    rank_movements: list[RankMovement]
    overlap: list[PairwiseOverlap]
    observability_notice: str
```

The M3 catalog response is dataset-scoped and origin-labeled:

```python
class RetrievalConfigCatalogResponse(BaseModel):
    contract_version: Literal[1] = 1
    dataset_version_id: UUID
    data_origin: Literal["live", "synthetic_demo"]
    configs: list[RetrievalConfigSummary]  # exactly four
```

Its order is frozen as BM25 baseline, ANN, server RRF, and local reranker, with four distinct
immutable IDs. P0 has no config mutation endpoint.

If debug provenance requires a second raw multi-query, its duration is `provenance_probe`, never folded into the production-shaped `turbopuffer` timing.

`GET /api/v1/configs` remains the existing tiny-fixture Playground catalog and must not change shape
or meaning during M3. `GET /api/v1/datasets/{dataset_version_id}/configs` is the new persisted,
origin-labeled four-config catalog bound to an immutable dataset revision. M3-B implements the new
route without redirecting or reusing the Playground route. Compare reports only observed 1-based
ranks, typed scores, and separate client-wall-clock embedding/provider timings, and never returns
query vectors or claims unexposed provider internals.

## 7. Judged query sets

```python
class Qrel(BaseModel):
    document_id: UUID
    relevance_grade: int  # >= 0; relevant iff > 0

class JudgedQuery(BaseModel):
    id: UUID
    external_id: str
    text: str
    filters: FilterNode | None = None
    tags: list[str] = Field(default_factory=list)
    qrels: list[Qrel]

class QuerySet(BaseModel):
    id: UUID
    name: str
    version: str
    dataset_version_id: UUID
    query_count: int
    content_hash: str
    created_at: datetime

class QuerySetSummary(BaseModel):
    id: UUID
    name: str
    version: str
    query_count: int
    content_hash: str
```

M3 query-set catalog items pair each immutable `QuerySet` with `data_origin`; the versioned list is
scoped by one required `dataset_version_id`. P0 catalog query sets contain exactly 50 queries.

Metric functions ignore documents with grade 0 as relevant, preserve positive grades for NDCG gain, and return `null` plus a warning when a query has no positive qrels.

## 8. Eval run contracts

### Create run

```python
class CreateEvalRunRequest(BaseModel):
    contract_version: Literal[1] = 1
    query_set_id: UUID
    baseline_config_id: UUID
    candidate_config_ids: list[UUID]  # exactly three
    random_seed: int = 20260822
    max_concurrency: int = Field(default=4, ge=1, le=16)
    warmup_query_count: int = Field(default=5, ge=0, le=50)
```

The baseline and three candidate IDs must be distinct. The application resolves them against one
persisted, canonical 50-query suite and the ordered BM25/ANN/RRF/reranker catalog; neither origin,
namespace, query text, nor config bodies are accepted from the browser. The response is a versioned
`CreateEvalRunResponse` containing the durable `queued` revision and is served with HTTP 202.

### Status and summary

```python
class EvalRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

class MetricAggregate(BaseModel):
    name: Literal["ndcg@10", "recall@50", "mrr@10", "latency_p50_ms", "latency_p95_ms", "error_rate"]
    value: float | None
    sample_count: int

class ConfigRunSummary(BaseModel):
    config_id: UUID
    metrics: list[MetricAggregate]
    completed_queries: int
    failed_queries: int

class RunEnvironment(BaseModel):
    pufferlab_git_revision: str
    turbopuffer_region: str
    python_version: str
    platform: str
    max_concurrency: int  # 1..16
    warmup_query_count: int = 0  # 0..50; unmeasured and excluded from outcomes
    timing_source: Literal["perf_counter", "synthetic_unavailable"] = "perf_counter"
    query_embedding_cache_enabled: bool

class EvalRun(BaseModel):
    contract_version: Literal[1] = 1
    id: UUID
    status: EvalRunStatus
    query_set: QuerySetSummary
    baseline_config_id: UUID
    candidate_config_ids: list[UUID]
    summaries: list[ConfigRunSummary]
    completed_queries: int
    total_queries: int
    random_seed: int
    environment: RunEnvironment
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: ApiErrorDetail | None
```

`MetricAggregate.value` is null if and only if `sample_count == 0`. Completed projections contain
exactly four summaries in config-contract order and exactly these six metrics in order:
`ndcg@10`, `recall@50`, `mrr@10`, `latency_p50_ms`, `latency_p95_ms`, and `error_rate`. Failures and
no-positive-qrel attempts do not enter quality means. Error rate covers all 50 attempts.

`perf_counter` means observed PufferLab client-wall time, not provider service time or a benchmark.
Warmups, configured concurrency, embedding-cache policy, region, platform, and revision are retained
for reproducibility. Synthetic successes use `synthetic_unavailable`, a null total latency, no stage
timings, no candidate-count claim, and no trace. Their p50/p95 summaries are null with zero samples;
zero must never stand in for unavailable timing.

The existing M2 durable success codec remains decodable: omitted outcome `timing_source` means
`perf_counter`, preserving its canonical JSON. The additive synthetic shape is:

```python
class EvalSuccessPayload(BaseModel):
    contract_version: Literal[1] = 1
    kind: Literal["success"] = "success"
    ranked_document_ids: list[UUID]  # 0..50, unique, original order
    metrics: PerQueryMetrics
    timing_source: Literal["perf_counter", "synthetic_unavailable"] = "perf_counter"
    total_client_wall_latency_ms: float | None
    stage_timings: list[StageTiming]
    candidate_counts: dict[str, int]
    warnings: list[EvalOutcomeWarning]
    trace_id: UUID | None
```

Run-list, detail, create, and cancel responses wrap an `EvalRunView` with explicit
`dataset_version_id`, `data_origin`, four ordered `RetrievalConfigSummary` labels matching the
baseline/candidate IDs, `completed_attempts` (0..200), `total_attempts=200`,
`original_stage_evidence_available=false`, and `live_replay_policy_permitted`. That last field means
only that origin policy permits an explicit replay; it never asserts current provider namespace
availability. A completed view requires exact 50-query/four-config durable coverage. A failed view
carries one direct redacted `ApiErrorDetail`; other statuses do not. Cancel returns the current
durable revision and is idempotent for terminal runs. Run lists are bounded to 100 and ordered by
`created_at` descending, then UUID ascending.

### Regression

```python
class RelevantRankChange(BaseModel):
    document_id: UUID
    relevance_grade: int
    baseline_rank: int | None
    candidate_rank: int | None

class RegressionRow(BaseModel):
    query_id: UUID
    query_text: str
    baseline_config_id: UUID
    candidate_config_id: UUID
    baseline_ndcg_at_10: float
    candidate_ndcg_at_10: float
    ndcg_delta: float
    recall_delta: float
    mrr_delta: float
    baseline_latency_ms: float | None
    candidate_latency_ms: float | None
    relevant_rank_changes: list[RelevantRankChange]
    playground_url: str

class RegressionCoverage(BaseModel):
    total_queries: Literal[50] = 50
    paired_queries: int
    excluded: list[ExcludedPairCount]  # exactly six, frozen order

class RegressionResponse(BaseModel):
    contract_version: Literal[1] = 1
    run_id: UUID
    data_origin: Literal["live", "synthetic_demo"]
    baseline_config_id: UUID
    candidate_config_id: UUID
    order: Literal["regressions", "gains"]
    limit: int  # 1..50
    rows: list[RegressionRow]  # paired rows only
    coverage: RegressionCoverage
```

Excluded statuses appear exactly once and in this order: `baseline_missing`, `candidate_missing`,
`baseline_failed`, `candidate_failed`, `both_failed`, `no_positive_qrels`. Paired plus excluded counts
must total 50. Sort regressions by `ndcg_delta`, `mrr_delta`, then UUID ascending. For compatibility
with the Milestone 2 engine, gains use the full inverse—including UUID descending on an exact metric
tie. Missing, failed, and no-positive-qrel pairs never receive zero-valued quality rows.
Synthetic regression rows require null baseline/candidate latency; paired live rows require both
measured client-wall values. Each `playground_url` is a relative `/playground` link containing
exactly one UUID-valued `run`, `query`, `left`, and `right`, plus at most one UUID `document`. M3-B
constructs and tests these IDs against the response run/row identities; query text and all other
parameters are forbidden from the URL.

### Query detail, export, and lifecycle envelopes

`EvalRunQueryDetailResponse` is versioned and run-scoped. It returns the exact judged query, the
baseline plus three candidate IDs, four persisted config summaries in contract order, zero to four
available durable outcomes in that order, three relevant-rank-change groups through rank 50,
dataset attribution, explicit `data_origin`, `original_stage_evidence_available=false`, and replay
availability. It performs no provider work.

`EvalRunExportResponse` wraps the backward-compatible `EvalRunExport` and adds `data_origin`.
Exports are deterministic by `(config_id, query_id)` and safe for partial, cancelled, interrupted,
or failed runs. A completed export requires the identical set of 50 query IDs for each of four run
configs (200 unique outcomes). The synthetic export is completed, contains 200 successes, and
retains unavailable timing without coercion. Export origin is end-to-end: environment timing,
every successful outcome, and completed p50/p95 summaries must agree. Synthetic summaries are
null/zero-sample; completed live summaries contain 50 measured latency samples.

The six durable statuses are `queued`, `running`, `completed`, `failed`, `cancelled`, and
`interrupted`. Recovery first records stale `running -> interrupted`, then claims valid queued runs
oldest first. The exceptional `queued -> failed` transition is allowed only when exact persisted
dataset/query-set/config bindings cannot be reconstructed; it records a safe direct error and makes
no provider call. Partial running work is not automatically resumed.

## 9. Query forensics and explicit replay

```python
class ForensicCode(str, Enum):
    FILTER_PREDICATE_FAILED = "filter_predicate_failed"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_LEXICAL_CANDIDATES = "outside_lexical_candidates"
    OUTSIDE_VECTOR_CANDIDATES = "outside_vector_candidates"
    OUTSIDE_FUSION_TOP_K = "outside_fusion_top_k"
    RERANKED_DOWN = "reranked_down"
    NOT_OBSERVABLE = "not_observable"

class EvidenceOrigin(str, Enum):
    STORED_RUN = "stored_run"
    LIVE_REPLAY_PRIMARY = "live_replay_primary"
    LIVE_REPLAY_COUNTERFACTUAL_PROBE = "live_replay_counterfactual_probe"
    CLIENT_COMPUTED = "client_computed"

class EvidenceItem(BaseModel):
    label: str  # 1..64, machine-readable
    value: ForensicEvidenceValue
    origin: EvidenceOrigin
    observed_at: AwareDatetime | None
    trace_id: UUID | None

class ForensicObservation(BaseModel):
    config_id: UUID
    document_id: UUID
    code: ForensicCode
    statement: str  # 1..512
    origin: EvidenceOrigin
    observed_at: AwareDatetime | None
    trace_id: UUID | None
    evidence: list[EvidenceItem]  # at most 16, unique labels
    certainty: Literal["observed", "counterfactual", "insufficient"]
```

`ForensicEvidenceValue` is a discriminated allowlist only: bounded rank, typed score,
candidate-count, presence, filter-result, RRF-contribution, or warning. Unknown keys/kinds, recursive
JSON, non-finite numbers, boolean-to-number coercion, oversized strings/lists, and arbitrary provider
bodies are rejected. RRF contributions require bounded rank/weight/constant inputs and exact
`weight / (rank_constant + rank)` arithmetic.

Stored M2 outcomes did not retain stage membership or stage scores. They may therefore emit only
`NOT_OBSERVABLE` with `certainty=insufficient` and null trace/time; ranks, metrics, and timing remain
honestly available in the query-detail outcome itself. Primary replay, counterfactual probe, and
client-computed items each retain their exact source timestamp and trace. Probe-derived evidence
cannot claim
`certainty=observed` or causal responsibility for a primary ordering. A non-derived observation
cannot merge origins or traces; a client-computed observation preserves the origin of every bounded
input and becomes counterfactual whenever any input came from the separate probe.

`stored_run` forensic evidence is limited to the typed original-stage-unavailable warning with null
trace/time; stored final ranks, metrics, and timings remain in the durable outcome rather than being
re-cast as stage proof. Every primary forensic trace/time must match an actual primary config result
and the primary replay timestamp. Every counterfactual trace/time must match one returned probe.
Probe membership ranks must be unique per stage and fit a positive returned candidate count;
counterfactual rank/count evidence is checked against that same probe. Client-computed evidence must
name one returned source trace, use exactly that primary/probe timestamp, and apply the same probe
bounds, including positive counts for claimed score/membership inputs.
Primary-derived rank-like inputs must match an actual returned final or stage-membership rank.
Probe-derived client computations inherit counterfactual certainty.

`config_id` and `document_id` are mandatory observation targets. The config must belong to the
requested replay pair. Rank, score, presence, count, filter, and RRF evidence is validated against
that exact config/document in the returned primary or probe source; a valid-shaped observation from
another target is rejected rather than rendered in the forensic drawer.

```python
class EvalRunQueryReplayRequest(BaseModel):
    contract_version: Literal[1] = 1
    config_ids: list[UUID]  # exactly two distinct persisted IDs
    include_counterfactual_probe: bool = False

class ReplayFailedCounterfactualProbe(BaseModel):
    origin: Literal["live_replay_counterfactual_probe"]
    config_id: UUID
    observed_at: AwareDatetime
    trace_id: UUID
    warning: ForensicWarning  # code must be provenance_probe_failed

class EvalRunQueryReplayResponse(BaseModel):
    contract_version: Literal[1] = 1
    run_id: UUID
    query_id: UUID
    data_origin: Literal["live"] = "live"
    config_ids: list[UUID]
    primary_origin: Literal["live_replay_primary"]
    primary_observed_at: AwareDatetime
    primary: SearchCompareResponse
    counterfactual_probes: list[ReplayCounterfactualProbe]
    failed_counterfactual_probes: list[ReplayFailedCounterfactualProbe] = Field(
        default_factory=list,
        max_length=2,
    )
    observations: list[ForensicObservation]
    original_stage_evidence_available: Literal[False] = False
    observability_notice: str
```

A failed optional probe has its own bounded source instead of being folded into a primary result.
Successful and failed probes uniquely target requested config IDs. Primary result traces,
successful-probe traces, and failed-probe traces are all distinct. A failed probe preserves the
primary replay and records only the safe typed warning; it cannot contribute membership, score,
count, timing, or client-computed evidence.

Replay accepts no origin, namespace, query text, qrels, expected document IDs, or config body from
the client. The server derives them from the run and dataset binding. It first authenticates the
complete 50-query persisted suite against the checked source lock and ID-only curation anchor:
exact order, source/query UUIDs, tags, qrels, content hash, query-set UUID, and dataset binding must
all match. It then loads the checked dataset manifest and requires the run configs to equal the
derived canonical suite. Both provider-free validation steps finish before credentials, the bound
catalog, search runtime, provider, embedder, or reranker are constructed. Foreign, duplicated, or
tampered stored content fails with a direct redacted error and zero provider-capable calls. The
primary response contains only production-shaped evidence; BM25/ANN raw candidate memberships and
`provenance_probe` timing remain in separately labeled bounded probes. Probe failure preserves the
primary result.

Forbidden statements include “turbopuffer searched cluster X,” “the cache was cold,” “the filter
ran before ANN,” and counterfactual-probe inputs caused the primary order. Those claims remain
`NOT_OBSERVABLE` unless a future API directly supplies the fact.

## 10. HTTP surface

P0 endpoints:

```text
GET    /api/v1/health
GET    /api/v1/datasets
GET    /api/v1/datasets/{dataset_version_id}
GET    /api/v1/capabilities
GET    /api/v1/query-sets?dataset_version_id=...
GET    /api/v1/configs                                      # existing Playground catalog
GET    /api/v1/datasets/{dataset_version_id}/configs         # persisted eval catalog
POST   /api/v1/search/compare
POST   /api/v1/eval-runs                                      -> 202
GET    /api/v1/eval-runs?limit=...
GET    /api/v1/eval-runs/{run_id}
POST   /api/v1/eval-runs/{run_id}/cancel
GET    /api/v1/eval-runs/{run_id}/regressions
GET    /api/v1/eval-runs/{run_id}/queries/{query_id}
GET    /api/v1/eval-runs/{run_id}/export
POST   /api/v1/eval-runs/{run_id}/queries/{query_id}/replay
```

M4-B mounts `/api/v1/capabilities` only through its provider-free local inspector, so the capability
models are now reachable from OpenAPI. Evaluation-gate policy and report models are CLI-only and
never enter the HTTP schema.

Polling `GET /eval-runs/{run_id}` is the P0 progress mechanism. No WebSocket/SSE contract is reserved.
There is no P0 `POST /configs`.

Every non-2xx response is the direct, redacted `ApiErrorDetail` body—not FastAPI's nested
`{"detail": ...}` shape. Duplicate active suites return `409 run_conflict`; validation and forbidden
synthetic cost paths fail before provider construction. Reads and deep-link restoration never
perform provider work. Live replay is the only explicit query-detail action that may do so.

## 11. Errors and warnings

```python
class ApiErrorDetail(BaseModel):
    code: Literal[
        "validation_error", "configuration_required", "not_found", "namespace_not_ready",
        "provider_error", "rate_limited", "run_conflict", "internal_error"
    ]
    message: str
    retryable: bool
    trace_id: UUID
    details: dict[str, JsonValue] = Field(default_factory=dict)

class ApiWarning(BaseModel):
    code: str
    message: str
```

Provider exceptions are mapped at the adapter boundary. API responses never expose API keys, headers, vectors, stack traces, or unredacted provider bodies.

## 12. Persistence and job invariants

- Normal work follows `queued → running → completed|failed|cancelled`; on startup, stale `running`
  rows become `interrupted`.
- Startup then validates and claims queued rows oldest first within the global active-run bound. A
  queued row whose exact immutable binding cannot be reconstructed becomes `failed` with a direct
  safe error and zero provider calls.
- Query outcomes are upserted by `(run_id, config_id, query_id)` for idempotent persistence.
- A completed run is immutable.
- Cancelling stops scheduling new queries and preserves completed outcomes.
- `completed_queries` means fully durable query groups on 0..50, while outcome attempts are 0..200.
- Persist outcomes before publishing progress; final summaries derive only from durable outcomes.
- Synthetic demo revisions are read/export-only and never enter create, recovery, replay, or any
  provider-capable composition path.
- Deleting a local run never deletes a turbopuffer namespace.
- Only a PufferLab-created branch with a recorded ownership token may be deleted automatically.

## 13. Module boundaries

```text
backend/pufferlab/contracts/       Pydantic/API contracts; no provider imports
backend/pufferlab/providers/       turbopuffer, embedding, reranker adapters
backend/pufferlab/retrieval/       orchestration, RRF/provenance, forensic rules
backend/pufferlab/evals/           pure metrics, aggregation, regression analysis
backend/pufferlab/datasets/        dataset adapters, IDs, preprocessing, ingestion
backend/pufferlab/persistence/     SQLAlchemy models/repositories/migrations
backend/pufferlab/jobs/            in-process runner and cancellation
backend/pufferlab/api/             FastAPI routes and dependency wiring
backend/pufferlab/cli/             Typer commands only; calls services
web/src/api/                       generated client/types only
web/src/features/playground/       comparison UI
web/src/features/evals/            run/regression UI
web/src/features/configs/          config UI
web/src/features/datasets/         dataset UI
```

Dependency direction:

```text
contracts <- datasets/providers/evals <- retrieval <- api/cli/jobs
contracts <- persistence <---------------------^
OpenAPI -> generated web API types -> UI features
```

`evals` remains pure and cannot import turbopuffer, FastAPI, SQLAlchemy, or frontend concepts.

## 14. Local capability and evaluation-gate contracts

Milestone 4 freezes these models before composing routes, commands, provider clients, or durable
adapters. All models forbid extra fields and are frozen after validation. They contain no
free-form string or dictionary value seam for credentials, namespaces, regions, local paths,
provider payloads, query/document text, qrels, or tracebacks.

### Local Playground capability

The versioned capability response has exactly one `live_playground` member. Its state is
`locally_configured` or `action_required`. Requirements are a unique ordered subsequence of:

1. `api_key`
2. `search_namespace`
3. `region`
4. `live_search_runtime`
5. `owned_tiny_receipt_invalid`
6. `owned_tiny_credential_mismatch`
7. `owned_tiny_region_mismatch`

The corresponding action codes, in the same order, are `configure_api_key`,
`configure_search_namespace`, `configure_region`, `install_live_search_runtime`,
`resolve_owned_tiny_receipt`, `use_owned_tiny_credential`, and `use_owned_tiny_region`.
`locally_configured` requires an empty requirement tuple and null action. `action_required`
requires a non-empty tuple, and its action must correspond to the first unmet requirement. The
contract communicates only local configuration state and never remote health.

M4-B maps `configuration_required` to a direct HTTP 503 before provider, embedder, or reranker
factories and teaches the manual frontend error guard to preserve that generated enum value.

### Provider-free evaluation gate

`GatePolicy` is versioned and accepts only `ndcg@10`, `recall@50`, or `mrr@10`. Its strict numeric
domains are finite `min_delta` in `[-1, 1]`, finite `max_query_drop` and `max_error_rate` in
`[0, 1]`, and integer `min_paired_queries` in `[1, 50]`; strings and booleans are not numeric input.

A valid completed-run evaluation returns `GateReport` with verdict `passed` or `policy_failed`,
run/baseline/candidate UUIDs, the selected metric, and exactly four typed checks in this order:

1. `candidate_error_rate`
2. `paired_query_coverage`
3. `aggregate_delta`
4. `per_query_drop`

Every threshold is inclusive. Candidate error rate is exactly the integer failed-candidate count
divided by all 50 attempts; an approximate submitted rate is invalid, and the threshold verdict is
derived from that exact count-based value. Paired plus excluded query counts total 50, every failed
candidate attempt is excluded from that paired population, and aggregate/per-query checks use the
same nonzero paired population. A passing verdict requires all four checks to pass. The per-query
check reports its total violation count and at most the first ten unique violations ordered by
observed delta ascending, then query UUID ascending. Invalid policy or evidence remains a separate
safe CLI error rather than a third gate-report verdict.
