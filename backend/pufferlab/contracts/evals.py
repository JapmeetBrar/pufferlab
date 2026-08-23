"""Judgment, evaluation-run, metric, and regression contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.search import StageTiming, TimingStage


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
    warmup_query_count: int = Field(default=0, ge=0)
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


class PerQueryMetrics(ContractModel):
    """The judged quality values for one successful retrieval attempt."""

    ndcg_at_10: float | None = Field(default=None, ge=0, le=1)
    recall_at_50: float | None = Field(default=None, ge=0, le=1)
    mrr_at_10: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_all_defined_or_null(self) -> "PerQueryMetrics":
        values = (self.ndcg_at_10, self.recall_at_50, self.mrr_at_10)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("per-query quality metrics must all be defined or all be null")
        return self


class EvalOutcomeWarning(ContractModel):
    """Public-safe warning retained with an evaluated query outcome."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class EvalSuccessPayload(ContractModel):
    """Versioned durable evidence for one successful config/query attempt.

    Deliberately excluded fields include query/document text, vectors, credentials, request headers,
    and raw provider responses.
    """

    contract_version: ContractVersion = 1
    kind: Literal["success"] = "success"
    ranked_document_ids: list[UUID] = Field(max_length=50)
    metrics: PerQueryMetrics
    total_client_wall_latency_ms: float = Field(ge=0)
    stage_timings: list[StageTiming]
    candidate_counts: dict[str, int]
    warnings: list[EvalOutcomeWarning]
    trace_id: UUID

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvalSuccessPayload":
        if any(not name or count < 0 for name, count in self.candidate_counts.items()):
            raise ValueError("candidate counts require non-empty names and non-negative values")
        stages = [timing.stage for timing in self.stage_timings]
        if len(stages) != len(set(stages)) or TimingStage.TOTAL in stages:
            raise ValueError(
                "stage timings must be unique and exclude the separately measured total"
            )
        no_positive_qrels = any(warning.code == "no_positive_qrels" for warning in self.warnings)
        metrics_are_null = self.metrics.ndcg_at_10 is None
        if metrics_are_null != no_positive_qrels:
            raise ValueError(
                "null quality metrics and no_positive_qrels warning must occur together"
            )
        return self


class EvalFailurePayload(ContractModel):
    """Redacted durable evidence for an expected operational retrieval failure."""

    contract_version: ContractVersion = 1
    kind: Literal["failure"] = "failure"
    code: ApiErrorCode
    message: str = Field(min_length=1)
    retryable: bool
    operation: str = Field(min_length=1)
    trace_id: UUID
    total_client_wall_latency_ms: float = Field(ge=0)


type EvalOutcomePayload = Annotated[
    EvalSuccessPayload | EvalFailurePayload,
    Field(discriminator="kind"),
]


class EvalOutcomeRecord(ContractModel):
    """Typed identity envelope around one durable payload."""

    run_id: UUID
    config_id: UUID
    query_id: UUID
    created_at: datetime
    outcome: EvalOutcomePayload


class EvalRunExport(ContractModel):
    """Canonical, partial-state-safe JSON export of one run and its durable outcomes."""

    contract_version: ContractVersion = 1
    run: EvalRun
    outcomes: list[EvalOutcomeRecord]

    @model_validator(mode="after")
    def validate_outcome_run_ids(self) -> "EvalRunExport":
        if any(record.run_id != self.run.id for record in self.outcomes):
            raise ValueError("every exported outcome must belong to the exported run")
        identities = [(record.config_id, record.query_id) for record in self.outcomes]
        if len(identities) != len(set(identities)):
            raise ValueError("exported outcome identities must be unique")
        if identities != sorted(identities, key=lambda item: (str(item[0]), str(item[1]))):
            raise ValueError("exported outcomes must be in deterministic identity order")
        return self
