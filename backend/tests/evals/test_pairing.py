from __future__ import annotations

from uuid import UUID

import pytest
from pufferlab.evals import (
    PairStatus,
    QueryMetrics,
    QueryOutcome,
    evaluate_ranking,
    order_quality_deltas,
    paired_deltas,
    select_query_subset,
)


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _outcome(
    query_id: UUID,
    *,
    ndcg: float,
    recall: float,
    mrr: float,
    latency: float,
) -> QueryOutcome:
    return QueryOutcome.succeeded(
        query_id=query_id,
        metrics=QueryMetrics(ndcg_at_10=ndcg, recall_at_50=recall, mrr_at_10=mrr),
        latency_ms=latency,
    )


def test_paired_deltas_are_candidate_minus_baseline() -> None:
    query_id = _uuid(1)
    baseline = _outcome(query_id, ndcg=0.8, recall=0.5, mrr=1.0, latency=10.0)
    candidate = _outcome(query_id, ndcg=0.6, recall=1.0, mrr=0.5, latency=14.0)

    delta = paired_deltas([baseline], [candidate])[0]

    assert delta.status is PairStatus.PAIRED
    assert delta.ndcg_delta == pytest.approx(-0.2)
    assert delta.recall_delta == 0.5
    assert delta.mrr_delta == -0.5
    assert delta.latency_delta_ms == 4.0


def test_pairing_retains_missing_failed_and_no_qrel_cases_without_zero_imputation() -> None:
    baseline_missing = _uuid(1)
    candidate_missing = _uuid(2)
    baseline_failed = _uuid(3)
    candidate_failed = _uuid(4)
    no_qrels = _uuid(5)
    both_failed = _uuid(6)
    no_qrel_metrics = evaluate_ranking([], [])
    valid_metrics = QueryMetrics(ndcg_at_10=1.0, recall_at_50=1.0, mrr_at_10=1.0)

    baseline = [
        QueryOutcome.succeeded(query_id=candidate_missing, metrics=valid_metrics),
        QueryOutcome.failed(query_id=baseline_failed, error_code="provider_error"),
        QueryOutcome.succeeded(query_id=candidate_failed, metrics=valid_metrics),
        QueryOutcome.succeeded(query_id=no_qrels, metrics=no_qrel_metrics),
        QueryOutcome.failed(query_id=both_failed, error_code="provider_error"),
    ]
    candidate = [
        QueryOutcome.succeeded(query_id=baseline_missing, metrics=valid_metrics),
        QueryOutcome.succeeded(query_id=baseline_failed, metrics=valid_metrics),
        QueryOutcome.failed(query_id=candidate_failed, error_code="provider_error"),
        QueryOutcome.succeeded(query_id=no_qrels, metrics=no_qrel_metrics),
        QueryOutcome.failed(query_id=both_failed, error_code="timeout"),
    ]

    deltas = paired_deltas(baseline, candidate)

    assert [delta.query_id for delta in deltas] == sorted(
        [
            baseline_missing,
            candidate_missing,
            baseline_failed,
            candidate_failed,
            no_qrels,
            both_failed,
        ],
        key=str,
    )
    assert {delta.query_id: delta.status for delta in deltas} == {
        baseline_missing: PairStatus.BASELINE_MISSING,
        candidate_missing: PairStatus.CANDIDATE_MISSING,
        baseline_failed: PairStatus.BASELINE_FAILED,
        candidate_failed: PairStatus.CANDIDATE_FAILED,
        no_qrels: PairStatus.NO_POSITIVE_QRELS,
        both_failed: PairStatus.BOTH_FAILED,
    }
    assert all(delta.ndcg_delta is None for delta in deltas)
    assert order_quality_deltas(deltas, order="regressions") == ()


def test_regression_and_gain_sorting_are_exact_and_deterministic() -> None:
    query_ids = [_uuid(value) for value in range(1, 6)]
    baseline = [
        _outcome(query_id, ndcg=0.5, recall=0.5, mrr=0.5, latency=10.0) for query_id in query_ids
    ]
    candidate_values = [
        (0.3, 0.5, 0.4),  # -0.2 NDCG, -0.1 MRR
        (0.3, 0.5, 0.2),  # -0.2 NDCG, -0.3 MRR: worst regression
        (0.5, 0.5, 0.5),
        (0.7, 0.5, 0.8),  # +0.2 NDCG, +0.3 MRR: best gain
        (0.7, 0.5, 0.6),
    ]
    candidate = [
        _outcome(
            query_id,
            ndcg=ndcg,
            recall=recall,
            mrr=mrr,
            latency=10.0,
        )
        for query_id, (ndcg, recall, mrr) in zip(query_ids, candidate_values, strict=True)
    ]
    deltas = paired_deltas(baseline, candidate)

    regressions = order_quality_deltas(deltas, order="regressions")
    gains = order_quality_deltas(deltas, order="gains")

    assert [delta.query_id for delta in regressions] == [
        query_ids[1],
        query_ids[0],
        query_ids[2],
        query_ids[4],
        query_ids[3],
    ]
    assert [delta.query_id for delta in gains] == list(
        reversed([delta.query_id for delta in regressions])
    )
    assert select_query_subset(deltas, order="regressions", limit=2) == (
        query_ids[1],
        query_ids[0],
    )
    assert select_query_subset(deltas, order="gains", limit=1) == (query_ids[3],)
    assert select_query_subset(deltas, order="gains", limit=0) == ()


def test_query_id_breaks_exact_metric_ties() -> None:
    first, second = _uuid(1), _uuid(2)
    baseline = [
        _outcome(first, ndcg=0.5, recall=0.5, mrr=0.5, latency=1.0),
        _outcome(second, ndcg=0.5, recall=0.5, mrr=0.5, latency=1.0),
    ]
    candidate = [
        _outcome(first, ndcg=0.4, recall=0.5, mrr=0.4, latency=1.0),
        _outcome(second, ndcg=0.4, recall=0.5, mrr=0.4, latency=1.0),
    ]

    deltas = paired_deltas(baseline, candidate)

    assert [row.query_id for row in order_quality_deltas(deltas, order="regressions")] == [
        first,
        second,
    ]
    assert [row.query_id for row in order_quality_deltas(deltas, order="gains")] == [
        second,
        first,
    ]


def test_pairing_rejects_duplicate_query_ids_and_negative_subset_limit() -> None:
    outcome = _outcome(_uuid(1), ndcg=1.0, recall=1.0, mrr=1.0, latency=1.0)

    with pytest.raises(ValueError, match="baseline outcomes"):
        paired_deltas([outcome, outcome], [outcome])
    with pytest.raises(ValueError, match="candidate outcomes"):
        paired_deltas([outcome], [outcome, outcome])
    with pytest.raises(ValueError, match="limit"):
        select_query_subset(paired_deltas([outcome], [outcome]), order="gains", limit=-1)


def test_ordering_rejects_duplicate_delta_rows_and_invalid_order() -> None:
    outcome = _outcome(_uuid(1), ndcg=1.0, recall=1.0, mrr=1.0, latency=1.0)
    delta = paired_deltas([outcome], [outcome])[0]

    with pytest.raises(ValueError, match="unique query IDs"):
        order_quality_deltas([delta, delta], order="regressions")
    with pytest.raises(ValueError, match="order must"):
        order_quality_deltas([delta], order="unexpected")  # type: ignore[arg-type]
