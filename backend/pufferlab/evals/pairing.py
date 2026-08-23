"""Paired candidate-minus-baseline analysis and deterministic ordering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from pufferlab.evals.models import PairedQueryDelta, PairStatus, QueryOutcome


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _pair_one(
    query_id: UUID,
    baseline: QueryOutcome | None,
    candidate: QueryOutcome | None,
) -> PairedQueryDelta:
    baseline_metrics = None if baseline is None else baseline.metrics
    candidate_metrics = None if candidate is None else candidate.metrics
    latency_delta = _delta(
        None if candidate is None else candidate.latency_ms,
        None if baseline is None else baseline.latency_ms,
    )

    if baseline is None:
        status = PairStatus.BASELINE_MISSING
    elif candidate is None:
        status = PairStatus.CANDIDATE_MISSING
    elif baseline.metrics is None and candidate.metrics is None:
        status = PairStatus.BOTH_FAILED
    elif baseline.metrics is None:
        status = PairStatus.BASELINE_FAILED
    elif candidate.metrics is None:
        status = PairStatus.CANDIDATE_FAILED
    elif baseline.metrics.ndcg_at_10 is None or candidate.metrics.ndcg_at_10 is None:
        status = PairStatus.NO_POSITIVE_QRELS
    else:
        status = PairStatus.PAIRED

    if status is PairStatus.PAIRED:
        assert baseline_metrics is not None
        assert candidate_metrics is not None
        return PairedQueryDelta(
            query_id=query_id,
            status=status,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            ndcg_delta=_delta(candidate_metrics.ndcg_at_10, baseline_metrics.ndcg_at_10),
            recall_delta=_delta(candidate_metrics.recall_at_50, baseline_metrics.recall_at_50),
            mrr_delta=_delta(candidate_metrics.mrr_at_10, baseline_metrics.mrr_at_10),
            latency_delta_ms=latency_delta,
        )

    return PairedQueryDelta(
        query_id=query_id,
        status=status,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        ndcg_delta=None,
        recall_delta=None,
        mrr_delta=None,
        latency_delta_ms=latency_delta,
    )


def paired_deltas(
    baseline_outcomes: Sequence[QueryOutcome],
    candidate_outcomes: Sequence[QueryOutcome],
) -> tuple[PairedQueryDelta, ...]:
    """Pair outcomes by query ID, retaining missing and failed attempts as explicit statuses."""

    baseline_by_id = {outcome.query_id: outcome for outcome in baseline_outcomes}
    candidate_by_id = {outcome.query_id: outcome for outcome in candidate_outcomes}
    if len(baseline_by_id) != len(baseline_outcomes):
        raise ValueError("baseline outcomes must contain unique query IDs")
    if len(candidate_by_id) != len(candidate_outcomes):
        raise ValueError("candidate outcomes must contain unique query IDs")

    query_ids = sorted(baseline_by_id.keys() | candidate_by_id.keys(), key=str)
    return tuple(
        _pair_one(query_id, baseline_by_id.get(query_id), candidate_by_id.get(query_id))
        for query_id in query_ids
    )


def order_quality_deltas(
    deltas: Sequence[PairedQueryDelta],
    *,
    order: Literal["regressions", "gains"],
    limit: int | None = None,
) -> tuple[PairedQueryDelta, ...]:
    """Order paired rows by NDCG delta, MRR delta, then query ID.

    Regressions use ascending order and gains use the exact inverse order. Non-paired rows are
    excluded rather than silently receiving zero. ``limit=0`` returns an empty tuple.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if order not in ("regressions", "gains"):
        raise ValueError("order must be regressions or gains")
    query_ids = [delta.query_id for delta in deltas]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("paired deltas must contain unique query IDs")
    paired = [delta for delta in deltas if delta.status is PairStatus.PAIRED]
    for delta in paired:
        if delta.ndcg_delta is None or delta.mrr_delta is None:
            raise ValueError("paired rows must contain NDCG and MRR deltas")
    ordered = sorted(
        paired,
        key=lambda delta: (delta.ndcg_delta, delta.mrr_delta, str(delta.query_id)),
        reverse=order == "gains",
    )
    return tuple(ordered if limit is None else ordered[:limit])


def select_query_subset(
    deltas: Sequence[PairedQueryDelta],
    *,
    order: Literal["regressions", "gains"],
    limit: int,
) -> tuple[UUID, ...]:
    """Return deterministic query IDs for a regression or gain subset."""

    return tuple(delta.query_id for delta in order_quality_deltas(deltas, order=order, limit=limit))
