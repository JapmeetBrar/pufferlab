"""Judgment, evaluation-run, metric, and regression contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from pufferlab.contracts.common import ContractModel, ContractVersion
from pufferlab.contracts.errors import ApiErrorDetail
from pufferlab.contracts.filters import FilterNode


class Qrel(ContractModel):
    document_id: UUID
    relevance_grade: int = Field(ge=0)


class JudgedQuery(ContractModel):
    id: UUID
    external_id: str
    text: str = Field(min_length=1)
    filters: FilterNode | None = None
    tags: list[str] = Field(default_factory=list)
    qrels: list[Qrel]


class QuerySet(ContractModel):
    id: UUID
    name: str
    version: str
    dataset_version_id: UUID
    query_count: int = Field(ge=0)
    content_hash: str
    created_at: datetime


class QuerySetSummary(ContractModel):
    id: UUID
    name: str
    version: str
    query_count: int = Field(ge=0)
    content_hash: str


class CreateEvalRunRequest(ContractModel):
    contract_version: ContractVersion = 1
    query_set_id: UUID
    baseline_config_id: UUID
    candidate_config_ids: list[UUID] = Field(min_length=1)
    random_seed: int = 20260822
    max_concurrency: int = Field(default=4, ge=1)
    warmup_query_count: int = Field(default=5, ge=0)


class EvalRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class MetricName(StrEnum):
    NDCG_AT_10 = "ndcg@10"
    RECALL_AT_50 = "recall@50"
    MRR_AT_10 = "mrr@10"
    LATENCY_P50_MS = "latency_p50_ms"
    LATENCY_P95_MS = "latency_p95_ms"
    ERROR_RATE = "error_rate"


class MetricAggregate(ContractModel):
    name: MetricName
    value: float | None
    sample_count: int = Field(ge=0)


class ConfigRunSummary(ContractModel):
    config_id: UUID
    metrics: list[MetricAggregate]
    completed_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)


class RunEnvironment(ContractModel):
    pufferlab_git_revision: str
    turbopuffer_region: str
    python_version: str
    platform: str
    max_concurrency: int = Field(ge=1)
    timing_source: Literal["perf_counter"] = "perf_counter"
    query_embedding_cache_enabled: bool


class EvalRun(ContractModel):
    contract_version: ContractVersion = 1
    id: UUID
    status: EvalRunStatus
    query_set: QuerySetSummary
    baseline_config_id: UUID
    candidate_config_ids: list[UUID]
    summaries: list[ConfigRunSummary]
    completed_queries: int = Field(ge=0)
    total_queries: int = Field(ge=0)
    random_seed: int
    environment: RunEnvironment
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: ApiErrorDetail | None


class RelevantRankChange(ContractModel):
    document_id: UUID
    relevance_grade: int = Field(ge=0)
    baseline_rank: int | None = Field(default=None, ge=1)
    candidate_rank: int | None = Field(default=None, ge=1)


class RegressionRow(ContractModel):
    query_id: UUID
    query_text: str
    baseline_config_id: UUID
    candidate_config_id: UUID
    baseline_ndcg_at_10: float | None
    candidate_ndcg_at_10: float | None
    ndcg_delta: float | None
    recall_delta: float | None
    mrr_delta: float | None
    baseline_latency_ms: float | None = Field(default=None, ge=0)
    candidate_latency_ms: float | None = Field(default=None, ge=0)
    relevant_rank_changes: list[RelevantRankChange]
    playground_url: str
