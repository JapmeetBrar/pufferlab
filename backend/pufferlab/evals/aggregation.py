"""Deterministic aggregation for evaluation outcomes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pufferlab.evals.models import (
    EvaluationAggregate,
    MetricSummary,
    QueryMetrics,
    QueryOutcome,
)


def linear_percentile(values: Sequence[float], percentile: float) -> MetricSummary:
    """Return a percentile using linear interpolation over ``(n - 1) * percentile``.

    This is the Hyndman-Fan type 7 method used by NumPy's default ``method="linear"``: after
    sorting, interpolate between floor and ceiling positions of ``(n - 1) * percentile``. The
    percentile argument is expressed on ``[0, 1]``. Empty input produces a null, zero-sample value.
    """

    if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be a finite value between zero and one")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("percentile samples must be finite and non-negative")
    if not values:
        return MetricSummary(value=None, sample_count=0)

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    value = ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])
    return MetricSummary(value=value, sample_count=len(ordered))


def _mean_metric(
    outcomes: Sequence[QueryOutcome], getter: Callable[[QueryMetrics], float | None]
) -> MetricSummary:
    values = [
        value
        for outcome in outcomes
        if outcome.metrics is not None
        for value in (getter(outcome.metrics),)
        if value is not None
    ]
    if not values:
        return MetricSummary(value=None, sample_count=0)
    return MetricSummary(value=math.fsum(values) / len(values), sample_count=len(values))


def _rate(numerator: int, denominator: int) -> MetricSummary:
    if denominator == 0:
        return MetricSummary(value=None, sample_count=0)
    return MetricSummary(value=numerator / denominator, sample_count=denominator)


def aggregate_outcomes(outcomes: Sequence[QueryOutcome]) -> EvaluationAggregate:
    """Aggregate one configuration's query attempts.

    Failed attempts are excluded from quality means, included in error/completion coverage, and
    included in latency percentiles when they carry an observed duration. Successful queries with
    no positive qrels remain completed but do not contribute to quality means. Duplicate query IDs
    are rejected so no attempt can silently receive extra weight.
    """

    query_ids = [outcome.query_id for outcome in outcomes]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("outcomes must contain unique query IDs")

    attempted = len(outcomes)
    completed = sum(outcome.metrics is not None for outcome in outcomes)
    failed = attempted - completed
    quality_samples = sum(
        outcome.metrics is not None and outcome.metrics.ndcg_at_10 is not None
        for outcome in outcomes
    )
    latencies = [outcome.latency_ms for outcome in outcomes if outcome.latency_ms is not None]

    return EvaluationAggregate(
        ndcg_at_10=_mean_metric(outcomes, lambda metrics: metrics.ndcg_at_10),
        recall_at_50=_mean_metric(outcomes, lambda metrics: metrics.recall_at_50),
        mrr_at_10=_mean_metric(outcomes, lambda metrics: metrics.mrr_at_10),
        latency_p50_ms=linear_percentile(latencies, 0.50),
        latency_p95_ms=linear_percentile(latencies, 0.95),
        error_rate=_rate(failed, attempted),
        coverage_rate=_rate(completed, attempted),
        quality_coverage_rate=_rate(quality_samples, attempted),
        attempted_queries=attempted,
        completed_queries=completed,
        failed_queries=failed,
    )
