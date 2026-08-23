"""Judgment, evaluation-run, metric, and regression contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import StageTiming, TimingStage

_CANONICAL_QUERY_COUNT = 50
_CANONICAL_CONFIG_COUNT = 4
_CANONICAL_OUTCOME_COUNT = _CANONICAL_QUERY_COUNT * _CANONICAL_CONFIG_COUNT
_MAX_RUN_LIST_ITEMS = 100


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
    candidate_config_ids: list[UUID] = Field(min_length=3, max_length=3)
    random_seed: int = 20260822
    max_concurrency: int = Field(default=4, ge=1, le=16)
    warmup_query_count: int = Field(default=5, ge=0, le=_CANONICAL_QUERY_COUNT)

    @model_validator(mode="after")
    def validate_canonical_config_ids(self) -> "CreateEvalRunRequest":
        config_ids = [self.baseline_config_id, *self.candidate_config_ids]
        if len(set(config_ids)) != _CANONICAL_CONFIG_COUNT:
            raise ValueError(
                "the canonical baseline and three candidate config IDs must be distinct"
            )
        return self


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


_SUMMARY_METRIC_ORDER = (
    MetricName.NDCG_AT_10,
    MetricName.RECALL_AT_50,
    MetricName.MRR_AT_10,
    MetricName.LATENCY_P50_MS,
    MetricName.LATENCY_P95_MS,
    MetricName.ERROR_RATE,
)


class MetricAggregate(ContractModel):
    name: MetricName
    value: float | None
    sample_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_value_sample_pair(self) -> "MetricAggregate":
        if (self.value is None) != (self.sample_count == 0):
            raise ValueError("a metric value is null if and only if its sample count is zero")
        return self


class ConfigRunSummary(ContractModel):
    config_id: UUID
    metrics: list[MetricAggregate]
    completed_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)


class TimingSource(StrEnum):
    PERF_COUNTER = "perf_counter"
    SYNTHETIC_UNAVAILABLE = "synthetic_unavailable"


class RunEnvironment(ContractModel):
    pufferlab_git_revision: str
    turbopuffer_region: str
    python_version: str
    platform: str
    max_concurrency: int = Field(ge=1, le=16)
    warmup_query_count: int = Field(default=0, ge=0, le=_CANONICAL_QUERY_COUNT)
    timing_source: TimingSource = TimingSource.PERF_COUNTER
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


def _validate_completed_summaries(run: EvalRun, *, synthetic: bool) -> None:
    config_ids = [run.baseline_config_id, *run.candidate_config_ids]
    if [summary.config_id for summary in run.summaries] != config_ids:
        raise ValueError("completed summaries must retain config contract order")
    for summary in run.summaries:
        if tuple(metric.name for metric in summary.metrics) != _SUMMARY_METRIC_ORDER:
            raise ValueError("completed summaries require six metrics in contract order")
        if summary.completed_queries + summary.failed_queries != _CANONICAL_QUERY_COUNT:
            raise ValueError("completed summaries must cover all 50 query attempts")
        if synthetic and (
            summary.completed_queries != _CANONICAL_QUERY_COUNT or summary.failed_queries != 0
        ):
            raise ValueError("synthetic summaries require 50 successful query attempts")
        error_rate = summary.metrics[-1]
        if error_rate.sample_count != _CANONICAL_QUERY_COUNT:
            raise ValueError("completed error-rate summaries require 50 samples")
        for name in (MetricName.LATENCY_P50_MS, MetricName.LATENCY_P95_MS):
            latency = next(metric for metric in summary.metrics if metric.name is name)
            if synthetic:
                if latency.value is not None or latency.sample_count != 0:
                    raise ValueError("synthetic latency summaries must be null with zero samples")
            elif latency.value is None or latency.sample_count != _CANONICAL_QUERY_COUNT:
                raise ValueError("live completed latency summaries require 50 measured samples")


class RelevantRankChange(ContractModel):
    document_id: UUID
    relevance_grade: int = Field(ge=0)
    baseline_rank: int | None = Field(default=None, ge=1, le=50)
    candidate_rank: int | None = Field(default=None, ge=1, le=50)


class RegressionOrder(StrEnum):
    REGRESSIONS = "regressions"
    GAINS = "gains"


class RegressionPairStatus(StrEnum):
    PAIRED = "paired"
    BASELINE_MISSING = "baseline_missing"
    CANDIDATE_MISSING = "candidate_missing"
    BASELINE_FAILED = "baseline_failed"
    CANDIDATE_FAILED = "candidate_failed"
    BOTH_FAILED = "both_failed"
    NO_POSITIVE_QRELS = "no_positive_qrels"


_EXCLUDED_PAIR_STATUS_ORDER = (
    RegressionPairStatus.BASELINE_MISSING,
    RegressionPairStatus.CANDIDATE_MISSING,
    RegressionPairStatus.BASELINE_FAILED,
    RegressionPairStatus.CANDIDATE_FAILED,
    RegressionPairStatus.BOTH_FAILED,
    RegressionPairStatus.NO_POSITIVE_QRELS,
)


class RegressionQuery(ContractModel):
    candidate_config_id: UUID
    order: RegressionOrder = RegressionOrder.REGRESSIONS
    limit: int = Field(default=10, ge=1, le=_CANONICAL_QUERY_COUNT)


class RegressionRow(ContractModel):
    query_id: UUID
    query_text: str = Field(min_length=1, max_length=4096)
    baseline_config_id: UUID
    candidate_config_id: UUID
    baseline_ndcg_at_10: float = Field(ge=0, le=1)
    candidate_ndcg_at_10: float = Field(ge=0, le=1)
    ndcg_delta: float = Field(ge=-1, le=1)
    recall_delta: float = Field(ge=-1, le=1)
    mrr_delta: float = Field(ge=-1, le=1)
    baseline_latency_ms: float | None = Field(default=None, ge=0)
    candidate_latency_ms: float | None = Field(default=None, ge=0)
    relevant_rank_changes: list[RelevantRankChange] = Field(max_length=100)
    playground_url: str = Field(
        min_length=1,
        max_length=2048,
        pattern=r"^/playground(?:\?|$)",
    )

    @model_validator(mode="after")
    def validate_playground_deep_link(self) -> "RegressionRow":
        parsed = urlsplit(self.playground_url)
        if parsed.scheme or parsed.netloc or parsed.fragment or parsed.path != "/playground":
            raise ValueError("playground links must be relative /playground URLs without fragments")
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        allowed = {"run", "query", "left", "right", "document"}
        keys = [key for key, _value in pairs]
        required = {"run", "query", "left", "right"}
        if set(keys) - allowed or any(keys.count(key) != 1 for key in required):
            raise ValueError("playground links require one run/query/left/right ID only")
        if keys.count("document") > 1 or len(keys) != len(set(keys)):
            raise ValueError("playground link parameters cannot be duplicated")
        try:
            ids = {key: UUID(value) for key, value in pairs if key in allowed}
        except ValueError as exc:
            raise ValueError("playground link parameters must be UUIDs") from exc
        if (
            ids["query"] != self.query_id
            or ids["left"] != self.baseline_config_id
            or ids["right"] != self.candidate_config_id
        ):
            raise ValueError("playground link query/left/right IDs must match the regression row")
        return self


class ExcludedPairCount(ContractModel):
    status: RegressionPairStatus
    count: int = Field(ge=0, le=_CANONICAL_QUERY_COUNT)


class RegressionCoverage(ContractModel):
    total_queries: int = Field(default=_CANONICAL_QUERY_COUNT, ge=50, le=50)
    paired_queries: int = Field(ge=0, le=_CANONICAL_QUERY_COUNT)
    excluded: list[ExcludedPairCount] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_exact_coverage(self) -> "RegressionCoverage":
        if tuple(item.status for item in self.excluded) != _EXCLUDED_PAIR_STATUS_ORDER:
            raise ValueError("excluded pair counts must use the frozen contract order")
        if self.paired_queries + sum(item.count for item in self.excluded) != self.total_queries:
            raise ValueError("paired and excluded counts must cover all 50 queries")
        return self


class RegressionResponse(ContractModel):
    contract_version: ContractVersion = 1
    run_id: UUID
    data_origin: DataOrigin
    baseline_config_id: UUID
    candidate_config_id: UUID
    order: RegressionOrder
    limit: int = Field(ge=1, le=_CANONICAL_QUERY_COUNT)
    rows: list[RegressionRow] = Field(max_length=_CANONICAL_QUERY_COUNT)
    coverage: RegressionCoverage

    @model_validator(mode="after")
    def validate_rows(self) -> "RegressionResponse":
        if len(self.rows) > self.limit:
            raise ValueError("regression row count cannot exceed the requested limit")
        if any(
            row.baseline_config_id != self.baseline_config_id
            or row.candidate_config_id != self.candidate_config_id
            for row in self.rows
        ):
            raise ValueError("every row must match the requested baseline/candidate pair")
        if any(
            UUID(dict(parse_qsl(urlsplit(row.playground_url).query))["run"]) != self.run_id
            for row in self.rows
        ):
            raise ValueError("every playground link must match the regression run ID")
        query_ids = [row.query_id for row in self.rows]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("regression rows must contain unique query IDs")
        ordered = sorted(
            self.rows,
            key=lambda row: (row.ndcg_delta, row.mrr_delta, str(row.query_id)),
            reverse=self.order is RegressionOrder.GAINS,
        )
        if self.rows != ordered:
            raise ValueError("regression rows must use deterministic quality ordering")
        latency_pairs = [(row.baseline_latency_ms, row.candidate_latency_ms) for row in self.rows]
        if self.data_origin is DataOrigin.SYNTHETIC_DEMO and any(
            baseline is not None or candidate is not None for baseline, candidate in latency_pairs
        ):
            raise ValueError("synthetic regressions cannot claim measured latency")
        if self.data_origin is DataOrigin.LIVE and any(
            baseline is None or candidate is None for baseline, candidate in latency_pairs
        ):
            raise ValueError("live paired regressions require measured latency")
        return self


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

    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1, max_length=512)


class EvalSuccessPayload(ContractModel):
    """Versioned durable evidence for one successful config/query attempt.

    Deliberately excluded fields include query/document text, vectors, credentials, request headers,
    and raw provider responses.
    """

    contract_version: ContractVersion = 1
    kind: Literal["success"] = "success"
    ranked_document_ids: list[UUID] = Field(max_length=50)
    metrics: PerQueryMetrics
    timing_source: TimingSource = Field(
        default=TimingSource.PERF_COUNTER,
        exclude_if=lambda value: value is TimingSource.PERF_COUNTER,
    )
    total_client_wall_latency_ms: float | None = Field(ge=0)
    stage_timings: list[StageTiming] = Field(max_length=10)
    candidate_counts: dict[str, int] = Field(max_length=10)
    warnings: list[EvalOutcomeWarning] = Field(max_length=20)
    trace_id: UUID | None

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvalSuccessPayload":
        if len(self.ranked_document_ids) != len(set(self.ranked_document_ids)):
            raise ValueError("ranked document IDs must be unique")
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
        if self.timing_source is TimingSource.SYNTHETIC_UNAVAILABLE:
            if self.total_client_wall_latency_ms is not None:
                raise ValueError("synthetic outcomes cannot claim total client-wall latency")
            if self.stage_timings:
                raise ValueError("synthetic outcomes cannot claim measured stage timings")
            if self.candidate_counts:
                raise ValueError("synthetic outcomes cannot claim provider candidate counts")
            if self.trace_id is not None:
                raise ValueError("synthetic outcomes cannot claim a provider trace")
        elif self.total_client_wall_latency_ms is None or self.trace_id is None:
            raise ValueError("perf_counter outcomes require measured latency and a trace ID")
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
    outcomes: list[EvalOutcomeRecord] = Field(max_length=_CANONICAL_OUTCOME_COUNT)

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


class EvalRunView(ContractModel):
    """Provider-free projection metadata around one durable run revision."""

    run: EvalRun
    dataset_version_id: UUID
    data_origin: DataOrigin
    configs: list[RetrievalConfigSummary] = Field(min_length=4, max_length=4)
    completed_attempts: int = Field(ge=0, le=_CANONICAL_OUTCOME_COUNT)
    total_attempts: int = Field(
        default=_CANONICAL_OUTCOME_COUNT,
        ge=_CANONICAL_OUTCOME_COUNT,
        le=_CANONICAL_OUTCOME_COUNT,
    )
    original_stage_evidence_available: Literal[False] = False
    live_replay_policy_permitted: bool

    @model_validator(mode="after")
    def validate_canonical_projection(self) -> "EvalRunView":
        config_ids = [self.run.baseline_config_id, *self.run.candidate_config_ids]
        synthetic = self.data_origin is DataOrigin.SYNTHETIC_DEMO
        if (
            self.run.total_queries != _CANONICAL_QUERY_COUNT
            or self.run.query_set.query_count != _CANONICAL_QUERY_COUNT
        ):
            raise ValueError("P0 run views require the canonical 50-query set")
        if len(config_ids) != _CANONICAL_CONFIG_COUNT or len(set(config_ids)) != len(config_ids):
            raise ValueError("P0 run views require four distinct ordered configs")
        if [config.id for config in self.configs] != config_ids:
            raise ValueError("run-view configs must match baseline/candidate contract order")
        if self.completed_attempts < self.run.completed_queries * _CANONICAL_CONFIG_COUNT:
            raise ValueError("attempt progress cannot trail fully durable query-group progress")
        if self.run.status is EvalRunStatus.COMPLETED and (
            self.run.completed_queries != _CANONICAL_QUERY_COUNT
            or self.completed_attempts != _CANONICAL_OUTCOME_COUNT
        ):
            raise ValueError("completed P0 runs require exact 50-by-four durable coverage")
        if self.run.status is EvalRunStatus.FAILED:
            if self.run.error is None:
                raise ValueError("failed run views require one direct redacted error")
        elif self.run.error is not None:
            raise ValueError("only failed run views may carry a run-level error")
        if self.run.status is EvalRunStatus.COMPLETED:
            _validate_completed_summaries(self.run, synthetic=synthetic)
        if synthetic != (self.run.environment.timing_source is TimingSource.SYNTHETIC_UNAVAILABLE):
            raise ValueError("run origin and timing source must agree")
        if self.live_replay_policy_permitted is synthetic:
            raise ValueError("synthetic demo runs are read/export-only")
        if synthetic and self.run.status is not EvalRunStatus.COMPLETED:
            raise ValueError("the synthetic demo projection is one immutable completed run")
        return self


class EvalRunListQuery(ContractModel):
    limit: int = Field(default=50, ge=1, le=_MAX_RUN_LIST_ITEMS)


class EvalRunListResponse(ContractModel):
    contract_version: ContractVersion = 1
    runs: list[EvalRunView] = Field(max_length=_MAX_RUN_LIST_ITEMS)

    @model_validator(mode="after")
    def validate_order(self) -> "EvalRunListResponse":
        expected = sorted(
            self.runs, key=lambda item: (-item.run.created_at.timestamp(), str(item.run.id))
        )
        if self.runs != expected:
            raise ValueError("run lists must be newest first with UUID tie-breaking")
        return self


class EvalRunDetailResponse(ContractModel):
    contract_version: ContractVersion = 1
    result: EvalRunView


class CreateEvalRunResponse(ContractModel):
    contract_version: ContractVersion = 1
    result: EvalRunView

    @model_validator(mode="after")
    def validate_queued(self) -> "CreateEvalRunResponse":
        if self.result.run.status is not EvalRunStatus.QUEUED:
            raise ValueError("create-run responses must return the durable queued revision")
        return self


class CancelEvalRunResponse(ContractModel):
    contract_version: ContractVersion = 1
    result: EvalRunView


class CandidateRelevantRankChanges(ContractModel):
    candidate_config_id: UUID
    changes: list[RelevantRankChange] = Field(max_length=100)


class DatasetAttribution(ContractModel):
    source_name: str = Field(min_length=1, max_length=128)
    source_url: str | None = Field(
        default=None,
        max_length=2048,
        pattern=r"^https://",
    )
    license_name: str | None = Field(default=None, min_length=1, max_length=128)
    license_url: str | None = Field(
        default=None,
        max_length=2048,
        pattern=r"^https://",
    )


class EvalRunQueryDetailResponse(ContractModel):
    contract_version: ContractVersion = 1
    run_id: UUID
    data_origin: DataOrigin
    query: JudgedQuery
    baseline_config_id: UUID
    candidate_config_ids: list[UUID] = Field(min_length=3, max_length=3)
    configs: list[RetrievalConfigSummary] = Field(min_length=4, max_length=4)
    outcomes: list[EvalOutcomeRecord] = Field(max_length=4)
    rank_changes: list[CandidateRelevantRankChanges] = Field(min_length=3, max_length=3)
    attribution: DatasetAttribution
    original_stage_evidence_available: Literal[False] = False
    live_replay_policy_permitted: bool

    @model_validator(mode="after")
    def validate_run_query_scope(self) -> "EvalRunQueryDetailResponse":
        config_ids = [self.baseline_config_id, *self.candidate_config_ids]
        if len(set(config_ids)) != _CANONICAL_CONFIG_COUNT:
            raise ValueError("query detail requires four distinct ordered configs")
        if [config.id for config in self.configs] != config_ids:
            raise ValueError("query-detail configs must use baseline/candidate contract order")
        outcome_ids = [record.config_id for record in self.outcomes]
        if outcome_ids != [config_id for config_id in config_ids if config_id in outcome_ids]:
            raise ValueError("available outcomes must retain config contract order")
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("query detail cannot contain duplicate config outcomes")
        if any(
            record.run_id != self.run_id or record.query_id != self.query.id
            for record in self.outcomes
        ):
            raise ValueError("every outcome must match the requested run and query")
        if [item.candidate_config_id for item in self.rank_changes] != self.candidate_config_ids:
            raise ValueError("rank-change groups must retain candidate contract order")
        synthetic = self.data_origin is DataOrigin.SYNTHETIC_DEMO
        if self.live_replay_policy_permitted is synthetic:
            raise ValueError("synthetic query detail is read/export-only")
        expected_timing = (
            TimingSource.SYNTHETIC_UNAVAILABLE if synthetic else TimingSource.PERF_COUNTER
        )
        if (
            any(
                record.outcome.kind == "failure"
                or record.outcome.timing_source is not expected_timing
                for record in self.outcomes
            )
            and synthetic
        ):
            raise ValueError("synthetic query detail requires unavailable-timing successes")
        if any(
            record.outcome.kind == "success" and record.outcome.timing_source is not expected_timing
            for record in self.outcomes
        ):
            raise ValueError("query-detail outcome timing must agree with its data origin")
        return self


class EvalRunExportResponse(ContractModel):
    contract_version: ContractVersion = 1
    data_origin: DataOrigin
    export: EvalRunExport

    @model_validator(mode="after")
    def validate_origin_timing(self) -> "EvalRunExportResponse":
        run = self.export.run
        config_ids = [run.baseline_config_id, *run.candidate_config_ids]
        if (
            run.total_queries != _CANONICAL_QUERY_COUNT
            or run.query_set.query_count != _CANONICAL_QUERY_COUNT
            or len(config_ids) != _CANONICAL_CONFIG_COUNT
            or len(set(config_ids)) != _CANONICAL_CONFIG_COUNT
        ):
            raise ValueError("P0 exports require the canonical 50-query/four-config suite")
        if any(record.config_id not in config_ids for record in self.export.outcomes):
            raise ValueError("export outcomes must reference one of the run's four configs")
        if run.status is EvalRunStatus.COMPLETED:
            query_ids_by_config = {
                config_id: {
                    record.query_id
                    for record in self.export.outcomes
                    if record.config_id == config_id
                }
                for config_id in config_ids
            }
            query_id_sets = list(query_ids_by_config.values())
            if (
                len(self.export.outcomes) != _CANONICAL_OUTCOME_COUNT
                or any(len(query_ids) != _CANONICAL_QUERY_COUNT for query_ids in query_id_sets)
                or any(query_ids != query_id_sets[0] for query_ids in query_id_sets[1:])
            ):
                raise ValueError("completed exports require exact 50-by-four identity coverage")
        synthetic = self.data_origin is DataOrigin.SYNTHETIC_DEMO
        expected_timing = (
            TimingSource.SYNTHETIC_UNAVAILABLE if synthetic else TimingSource.PERF_COUNTER
        )
        if run.environment.timing_source is not expected_timing:
            raise ValueError("export origin and run timing source must agree")
        if any(
            record.outcome.kind == "success" and record.outcome.timing_source is not expected_timing
            for record in self.export.outcomes
        ):
            raise ValueError("every successful export outcome must match the export timing origin")
        if run.status is EvalRunStatus.COMPLETED:
            _validate_completed_summaries(run, synthetic=synthetic)
        if synthetic:
            if (
                run.status is not EvalRunStatus.COMPLETED
                or len(self.export.outcomes) != _CANONICAL_OUTCOME_COUNT
            ):
                raise ValueError("synthetic exports require one complete 50-by-four run")
            if any(record.outcome.kind != "success" for record in self.export.outcomes):
                raise ValueError("synthetic exports require unavailable-timing success outcomes")
        return self
