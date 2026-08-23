"""Typed, service-independent values used by the evaluation engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class EvaluationWarningCode(StrEnum):
    """Machine-readable warnings emitted while judging one ranking."""

    NO_POSITIVE_QRELS = "no_positive_qrels"
    DUPLICATE_QREL = "duplicate_qrel"
    DUPLICATE_RETRIEVED_DOCUMENT = "duplicate_retrieved_document"


@dataclass(frozen=True, slots=True)
class EvaluationWarning:
    """A non-fatal condition that affects interpretation of query metrics."""

    code: EvaluationWarningCode
    message: str

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("warning message must not be empty")


@dataclass(frozen=True, slots=True)
class Judgment:
    """One document relevance judgment; positive grades are relevant."""

    document_id: UUID
    relevance_grade: int

    def __post_init__(self) -> None:
        if isinstance(self.relevance_grade, bool) or not isinstance(self.relevance_grade, int):
            raise TypeError("relevance_grade must be an integer")
        if self.relevance_grade < 0:
            raise ValueError("relevance_grade must be non-negative")


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Quality metrics for one successful query execution.

    All three values are ``None`` when the query has no positive qrels. Otherwise they are finite
    values in the closed interval ``[0, 1]``.
    """

    ndcg_at_10: float | None
    recall_at_50: float | None
    mrr_at_10: float | None
    warnings: tuple[EvaluationWarning, ...] = ()

    def __post_init__(self) -> None:
        values = (self.ndcg_at_10, self.recall_at_50, self.mrr_at_10)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("quality metrics must all be present or all be null")
        warning_codes = {warning.code for warning in self.warnings}
        has_no_qrel_warning = EvaluationWarningCode.NO_POSITIVE_QRELS in warning_codes
        if all(value is None for value in values) and not has_no_qrel_warning:
            raise ValueError("null quality metrics require a no_positive_qrels warning")
        if all(value is not None for value in values) and has_no_qrel_warning:
            raise ValueError("defined quality metrics cannot carry a no_positive_qrels warning")
        for value in values:
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError("quality metrics must be finite values between zero and one")


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One attempted query for one configuration.

    Successful attempts carry metrics. Failed attempts carry a non-empty error code and no metrics.
    Either may carry an observed client wall-clock duration; aggregate latency includes every
    observed attempt duration, while aggregate quality includes successful judged attempts only.
    """

    query_id: UUID
    metrics: QueryMetrics | None
    latency_ms: float | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if (self.metrics is None) == (self.error_code is None):
            raise ValueError("an outcome must contain exactly one of metrics or error_code")
        if self.error_code == "":
            raise ValueError("error_code must not be empty")
        if self.latency_ms is not None and (
            not math.isfinite(self.latency_ms) or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")

    @classmethod
    def succeeded(
        cls,
        *,
        query_id: UUID,
        metrics: QueryMetrics,
        latency_ms: float | None = None,
    ) -> QueryOutcome:
        return cls(query_id=query_id, metrics=metrics, latency_ms=latency_ms)

    @classmethod
    def failed(
        cls,
        *,
        query_id: UUID,
        error_code: str,
        latency_ms: float | None = None,
    ) -> QueryOutcome:
        return cls(query_id=query_id, metrics=None, latency_ms=latency_ms, error_code=error_code)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """An aggregate value together with the exact number of contributing samples."""

    value: float | None
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("aggregate metric values must be finite")
        if self.value is None and self.sample_count != 0:
            raise ValueError("a null aggregate must have zero samples")
        if self.value is not None and self.sample_count == 0:
            raise ValueError("a defined aggregate must have at least one sample")


@dataclass(frozen=True, slots=True)
class EvaluationAggregate:
    """Quality, latency, error, and coverage aggregates for one configuration."""

    ndcg_at_10: MetricSummary
    recall_at_50: MetricSummary
    mrr_at_10: MetricSummary
    latency_p50_ms: MetricSummary
    latency_p95_ms: MetricSummary
    error_rate: MetricSummary
    coverage_rate: MetricSummary
    quality_coverage_rate: MetricSummary
    attempted_queries: int
    completed_queries: int
    failed_queries: int

    def __post_init__(self) -> None:
        counts = (self.attempted_queries, self.completed_queries, self.failed_queries)
        if any(count < 0 for count in counts):
            raise ValueError("query counts must be non-negative")
        if self.completed_queries + self.failed_queries != self.attempted_queries:
            raise ValueError("completed and failed counts must equal attempted queries")


class PairStatus(StrEnum):
    """Whether a baseline/candidate pair can produce quality deltas."""

    PAIRED = "paired"
    BASELINE_MISSING = "baseline_missing"
    CANDIDATE_MISSING = "candidate_missing"
    BASELINE_FAILED = "baseline_failed"
    CANDIDATE_FAILED = "candidate_failed"
    BOTH_FAILED = "both_failed"
    NO_POSITIVE_QRELS = "no_positive_qrels"


@dataclass(frozen=True, slots=True)
class PairedQueryDelta:
    """Candidate-minus-baseline values for one query."""

    query_id: UUID
    status: PairStatus
    baseline_metrics: QueryMetrics | None
    candidate_metrics: QueryMetrics | None
    ndcg_delta: float | None
    recall_delta: float | None
    mrr_delta: float | None
    latency_delta_ms: float | None

    def __post_init__(self) -> None:
        deltas = (
            self.ndcg_delta,
            self.recall_delta,
            self.mrr_delta,
            self.latency_delta_ms,
        )
        for delta in deltas:
            if delta is not None and not math.isfinite(delta):
                raise ValueError("paired deltas must be finite")
        quality_deltas = (self.ndcg_delta, self.recall_delta, self.mrr_delta)
        if self.status is PairStatus.PAIRED and any(delta is None for delta in quality_deltas):
            raise ValueError("paired quality outcomes must have all quality deltas")
        if self.status is not PairStatus.PAIRED and any(
            delta is not None for delta in quality_deltas
        ):
            raise ValueError("unpaired quality outcomes cannot have quality deltas")
