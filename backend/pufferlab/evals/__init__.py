"""Pure deterministic evaluation metrics and paired analysis."""

from pufferlab.evals.aggregation import aggregate_outcomes, linear_percentile
from pufferlab.evals.gates import (
    GateEvaluationError,
    GateEvaluationErrorCode,
    evaluate_gate,
)
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import (
    EvaluationAggregate,
    EvaluationWarning,
    EvaluationWarningCode,
    Judgment,
    MetricSummary,
    PairedQueryDelta,
    PairStatus,
    QueryMetrics,
    QueryOutcome,
)
from pufferlab.evals.pairing import order_quality_deltas, paired_deltas, select_query_subset

__all__ = [
    "EvaluationAggregate",
    "EvaluationWarning",
    "EvaluationWarningCode",
    "GateEvaluationError",
    "GateEvaluationErrorCode",
    "Judgment",
    "MetricSummary",
    "PairStatus",
    "PairedQueryDelta",
    "QueryMetrics",
    "QueryOutcome",
    "aggregate_outcomes",
    "evaluate_gate",
    "evaluate_ranking",
    "linear_percentile",
    "order_quality_deltas",
    "paired_deltas",
    "select_query_subset",
]
