from __future__ import annotations

import math
from uuid import UUID

import pytest
from pufferlab.evals import (
    MetricSummary,
    QueryMetrics,
    QueryOutcome,
    aggregate_outcomes,
    evaluate_ranking,
    linear_percentile,
)


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _metrics(ndcg: float, recall: float, mrr: float) -> QueryMetrics:
    return QueryMetrics(ndcg_at_10=ndcg, recall_at_50=recall, mrr_at_10=mrr)


def _no_qrels_metrics() -> QueryMetrics:
    return evaluate_ranking([], [])


def test_linear_percentiles_use_documented_type_seven_interpolation() -> None:
    values = [30.0, 0.0, 20.0, 10.0]

    assert linear_percentile(values, 0.50).value == 15.0
    assert linear_percentile(values, 0.95).value == pytest.approx(28.5)
    assert linear_percentile(values, 0.95).sample_count == 4
    assert linear_percentile([7.0], 0.95).value == 7.0
    assert linear_percentile([], 0.95).value is None
    assert linear_percentile([], 0.95).sample_count == 0


@pytest.mark.parametrize(
    ("values", "percentile"),
    [([math.nan], 0.5), ([math.inf], 0.5), ([-1.0], 0.5), ([1.0], math.nan), ([1.0], 1.1)],
)
def test_linear_percentile_rejects_non_finite_or_out_of_range_inputs(
    values: list[float], percentile: float
) -> None:
    with pytest.raises(ValueError):
        linear_percentile(values, percentile)


def test_aggregate_reports_quality_latency_error_coverage_and_sample_counts() -> None:
    outcomes = [
        QueryOutcome.succeeded(query_id=_uuid(1), metrics=_metrics(1.0, 0.5, 1.0), latency_ms=10.0),
        QueryOutcome.succeeded(
            query_id=_uuid(2), metrics=_metrics(0.5, 1.0, 0.25), latency_ms=20.0
        ),
        QueryOutcome.succeeded(query_id=_uuid(3), metrics=_no_qrels_metrics(), latency_ms=30.0),
        QueryOutcome.failed(query_id=_uuid(4), error_code="provider_error", latency_ms=40.0),
    ]

    aggregate = aggregate_outcomes(outcomes)

    assert aggregate.ndcg_at_10.value == 0.75
    assert aggregate.ndcg_at_10.sample_count == 2
    assert aggregate.recall_at_50.value == 0.75
    assert aggregate.recall_at_50.sample_count == 2
    assert aggregate.mrr_at_10.value == 0.625
    assert aggregate.mrr_at_10.sample_count == 2
    assert aggregate.latency_p50_ms.value == 25.0
    assert aggregate.latency_p95_ms.value == pytest.approx(38.5)
    assert aggregate.latency_p95_ms.sample_count == 4
    assert aggregate.error_rate.value == 0.25
    assert aggregate.error_rate.sample_count == 4
    assert aggregate.coverage_rate.value == 0.75
    assert aggregate.coverage_rate.sample_count == 4
    assert aggregate.quality_coverage_rate.value == 0.5
    assert aggregate.quality_coverage_rate.sample_count == 4
    assert aggregate.attempted_queries == 4
    assert aggregate.completed_queries == 3
    assert aggregate.failed_queries == 1


def test_failed_queries_never_enter_quality_means() -> None:
    aggregate = aggregate_outcomes(
        [QueryOutcome.failed(query_id=_uuid(1), error_code="timeout", latency_ms=8.0)]
    )

    assert aggregate.ndcg_at_10.value is None
    assert aggregate.ndcg_at_10.sample_count == 0
    assert aggregate.latency_p50_ms.value == 8.0
    assert aggregate.error_rate.value == 1.0
    assert aggregate.coverage_rate.value == 0.0
    assert aggregate.quality_coverage_rate.value == 0.0


def test_empty_aggregate_is_null_instead_of_inventing_zero_rates() -> None:
    aggregate = aggregate_outcomes([])

    assert aggregate.ndcg_at_10.value is None
    assert aggregate.latency_p95_ms.value is None
    assert aggregate.error_rate.value is None
    assert aggregate.coverage_rate.value is None
    assert aggregate.quality_coverage_rate.value is None
    assert aggregate.attempted_queries == 0


def test_aggregate_rejects_duplicate_query_attempts() -> None:
    outcome = QueryOutcome.succeeded(query_id=_uuid(1), metrics=_metrics(1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="unique query IDs"):
        aggregate_outcomes([outcome, outcome])


@pytest.mark.parametrize("latency", [-1.0, math.nan, math.inf])
def test_query_outcome_rejects_invalid_latency(latency: float) -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        QueryOutcome.succeeded(
            query_id=_uuid(1), metrics=_metrics(1.0, 1.0, 1.0), latency_ms=latency
        )


def test_query_outcome_requires_exactly_success_or_failure() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        QueryOutcome(query_id=_uuid(1), metrics=None)
    with pytest.raises(ValueError, match="exactly one"):
        QueryOutcome(
            query_id=_uuid(1),
            metrics=_metrics(1.0, 1.0, 1.0),
            error_code="provider_error",
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_query_metrics_reject_non_finite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite values"):
        QueryMetrics(ndcg_at_10=value, recall_at_50=1.0, mrr_at_10=1.0)


def test_null_query_metrics_require_the_no_positive_qrels_warning() -> None:
    with pytest.raises(ValueError, match="no_positive_qrels"):
        QueryMetrics(ndcg_at_10=None, recall_at_50=None, mrr_at_10=None)


def test_metric_summary_requires_sample_count_to_match_nullability() -> None:
    with pytest.raises(ValueError, match="null aggregate"):
        MetricSummary(value=None, sample_count=1)
    with pytest.raises(ValueError, match="defined aggregate"):
        MetricSummary(value=0.0, sample_count=0)
