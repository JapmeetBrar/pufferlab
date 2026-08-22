# PufferLab Shared Contracts v1

- **Status:** Freeze before parallel implementation
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

`not` has exactly one child; `and` and `or` have at least one. Empty field names and unknown dataset attributes fail validation.

Unknown fields/operators fail validation; they are never passed through as arbitrary expressions.

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
    distance_metric: Literal["cosine_distance", "euclidean_squared", "dot_product"]
    fts_profile: FtsProfile
    schema_hash: str

class DatasetVersion(BaseModel):
    id: UUID
    slug: str
    version: str
    namespace: str
    index_profile: IndexProfile
    document_count: int
    corpus_hash: str
    status: Literal["pending", "ingesting", "indexing", "ready", "failed"]
    created_at: datetime
```

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

If debug provenance requires a second raw multi-query, its duration is `provenance_probe`, never folded into the production-shaped `turbopuffer` timing.

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

Metric functions ignore documents with grade 0 as relevant, preserve positive grades for NDCG gain, and return `null` plus a warning when a query has no positive qrels.

## 8. Eval run contracts

### Create run

```python
class CreateEvalRunRequest(BaseModel):
    contract_version: Literal[1] = 1
    query_set_id: UUID
    baseline_config_id: UUID
    candidate_config_ids: list[UUID]
    random_seed: int = 20260822
    max_concurrency: int = 4
    warmup_query_count: int = 5
```

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
    max_concurrency: int
    timing_source: Literal["perf_counter"] = "perf_counter"
    query_embedding_cache_enabled: bool

class EvalRun(BaseModel):
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
    baseline_ndcg_at_10: float | None
    candidate_ndcg_at_10: float | None
    ndcg_delta: float | None
    recall_delta: float | None
    mrr_delta: float | None
    baseline_latency_ms: float | None
    candidate_latency_ms: float | None
    relevant_rank_changes: list[RelevantRankChange]
    playground_url: str
```

Sort regressions by non-null `ndcg_delta` ascending, then `mrr_delta`, then `query_id`. Gains use the inverse ordering. A missing/failed query is not silently assigned zero; it is an error outcome and appears in coverage/error metrics.

## 9. Forensic probe contract (P1, reserved now)

```python
class ForensicCode(str, Enum):
    FILTER_PREDICATE_FAILED = "filter_predicate_failed"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_LEXICAL_CANDIDATES = "outside_lexical_candidates"
    OUTSIDE_VECTOR_CANDIDATES = "outside_vector_candidates"
    OUTSIDE_FUSION_TOP_K = "outside_fusion_top_k"
    RERANKED_DOWN = "reranked_down"
    NOT_OBSERVABLE = "not_observable"

class EvidenceItem(BaseModel):
    label: str
    value: JsonValue
    source: Literal[
        "query_response", "compute_attribute", "local_filter_evaluation",
        "counterfactual_query", "client_computation", "reranker"
    ]

class ForensicObservation(BaseModel):
    code: ForensicCode
    statement: str
    evidence: list[EvidenceItem]
    certainty: Literal["observed", "counterfactual", "insufficient"]
```

Forbidden statements include “turbopuffer searched cluster X,” “the cache was cold,” or “the filter ran before ANN” unless a future API directly supplies that fact.

## 10. HTTP surface

P0 endpoints:

```text
GET    /api/v1/health
GET    /api/v1/datasets
GET    /api/v1/datasets/{dataset_version_id}
GET    /api/v1/configs
POST   /api/v1/configs
POST   /api/v1/search/compare
GET    /api/v1/query-sets
POST   /api/v1/eval-runs
GET    /api/v1/eval-runs
GET    /api/v1/eval-runs/{run_id}
GET    /api/v1/eval-runs/{run_id}/regressions
GET    /api/v1/eval-runs/{run_id}/queries/{query_id}
GET    /api/v1/eval-runs/{run_id}/export
```

Polling `GET /eval-runs/{run_id}` is the P0 progress mechanism. No WebSocket/SSE contract is reserved.

## 11. Errors and warnings

```python
class ApiErrorDetail(BaseModel):
    code: Literal[
        "validation_error", "not_found", "namespace_not_ready",
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

- `queued → running → completed|failed|cancelled`; on startup, stale `running` rows become `interrupted`.
- Query outcomes are upserted by `(run_id, config_id, query_id)` for idempotent persistence.
- A completed run is immutable.
- Cancelling stops scheduling new queries and preserves completed outcomes.
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
