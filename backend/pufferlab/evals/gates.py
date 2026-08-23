"""Pure deterministic evaluation-gate policy engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID

from pydantic import ValidationError

from pufferlab.contracts.gates import (
    GateAggregateDeltaCheck,
    GateCandidateErrorRateCheck,
    GateMetricName,
    GatePairedQueryCoverageCheck,
    GatePerQueryDropCheck,
    GatePolicy,
    GateQueryViolation,
    GateReport,
    GateVerdict,
)
from pufferlab.evals.models import (
    EvaluationWarning,
    EvaluationWarningCode,
    PairedQueryDelta,
    PairStatus,
    QueryMetrics,
    QueryOutcome,
)
from pufferlab.evals.pairing import paired_deltas

_CANONICAL_QUERY_COUNT = 50
_MAX_QUERY_VIOLATIONS = 10


class GateEvaluationErrorCode(StrEnum):
    """Safe reasons why the pure engine cannot produce a policy verdict."""

    INVALID_POLICY = "invalid_policy"
    INVALID_BINDING = "invalid_binding"
    INVALID_QUERY_CATALOG = "invalid_query_catalog"
    INVALID_BASELINE_EVIDENCE = "invalid_baseline_evidence"
    INVALID_CANDIDATE_EVIDENCE = "invalid_candidate_evidence"
    ZERO_PAIRED_QUERIES = "zero_paired_queries"


class GateEvaluationError(ValueError):
    """A bounded invalid-policy/evidence result with no submitted evidence attached."""

    __slots__ = ("code",)

    def __init__(self, code: GateEvaluationErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _validated_policy(policy: object) -> GatePolicy | None:
    if type(policy) is not GatePolicy:
        return None

    invalid = False
    validated: GatePolicy | None = None
    try:
        # Revalidate a potentially forged/model_construct instance instead of trusting its class.
        validated = GatePolicy.model_validate(policy.model_dump(mode="python"))
    except (TypeError, ValueError, ValidationError):
        invalid = True
    return None if invalid else validated


def _valid_warning(warning: object) -> bool:
    return (
        type(warning) is EvaluationWarning
        and type(warning.code) is EvaluationWarningCode
        and type(warning.message) is str
        and bool(warning.message)
    )


def _valid_metrics(metrics: object) -> bool:
    if type(metrics) is not QueryMetrics:
        return False
    if not all(_valid_warning(warning) for warning in metrics.warnings):
        return False

    values = (metrics.ndcg_at_10, metrics.recall_at_50, metrics.mrr_at_10)
    null_metrics = all(value is None for value in values)
    if any(value is None for value in values) and not null_metrics:
        return False

    warning_codes = {warning.code for warning in metrics.warnings}
    has_no_positive_qrels = EvaluationWarningCode.NO_POSITIVE_QRELS in warning_codes
    if null_metrics:
        return has_no_positive_qrels
    if has_no_positive_qrels:
        return False
    return all(
        type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
    )


def _valid_outcome(outcome: object) -> bool:
    if type(outcome) is not QueryOutcome or type(outcome.query_id) is not UUID:
        return False
    latency = outcome.latency_ms
    if latency is not None and (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        return False
    if outcome.metrics is None:
        return type(outcome.error_code) is str and bool(outcome.error_code)
    return outcome.error_code is None and _valid_metrics(outcome.metrics)


def _validated_expected_query_ids(query_ids: Sequence[UUID]) -> frozenset[UUID] | None:
    materialized = tuple(query_ids)
    if len(materialized) != _CANONICAL_QUERY_COUNT:
        return None
    if any(type(query_id) is not UUID for query_id in materialized):
        return None
    unique = frozenset(materialized)
    return unique if len(unique) == _CANONICAL_QUERY_COUNT else None


def _validated_outcomes(
    outcomes: Sequence[QueryOutcome],
    *,
    expected_query_ids: frozenset[UUID],
) -> tuple[QueryOutcome, ...] | None:
    materialized = tuple(outcomes)
    if len(materialized) != _CANONICAL_QUERY_COUNT:
        return None
    if not all(_valid_outcome(outcome) for outcome in materialized):
        return None
    query_ids = tuple(outcome.query_id for outcome in materialized)
    if len(frozenset(query_ids)) != _CANONICAL_QUERY_COUNT:
        return None
    if frozenset(query_ids) != expected_query_ids:
        return None
    return materialized


def _metric_delta(delta: PairedQueryDelta, metric: GateMetricName) -> float:
    if metric is GateMetricName.NDCG_AT_10:
        value = delta.ndcg_delta
    elif metric is GateMetricName.RECALL_AT_50:
        value = delta.recall_delta
    else:
        value = delta.mrr_delta
    if value is None or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise AssertionError("paired gate delta must be a finite signed-unit value")
    return value


def _evaluate_gate_result(
    *,
    run_id: UUID,
    baseline_config_id: UUID,
    candidate_config_id: UUID,
    expected_query_ids: Sequence[UUID],
    baseline_outcomes: Sequence[QueryOutcome],
    candidate_outcomes: Sequence[QueryOutcome],
    policy: object,
) -> tuple[GateReport | None, GateEvaluationErrorCode | None]:
    if (
        type(run_id) is not UUID
        or type(baseline_config_id) is not UUID
        or type(candidate_config_id) is not UUID
        or baseline_config_id == candidate_config_id
    ):
        return None, GateEvaluationErrorCode.INVALID_BINDING

    validated_policy = _validated_policy(policy)
    if validated_policy is None:
        return None, GateEvaluationErrorCode.INVALID_POLICY

    expected = _validated_expected_query_ids(expected_query_ids)
    if expected is None:
        return None, GateEvaluationErrorCode.INVALID_QUERY_CATALOG
    baseline = _validated_outcomes(baseline_outcomes, expected_query_ids=expected)
    if baseline is None:
        return None, GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE
    candidate = _validated_outcomes(candidate_outcomes, expected_query_ids=expected)
    if candidate is None:
        return None, GateEvaluationErrorCode.INVALID_CANDIDATE_EVIDENCE

    deltas = paired_deltas(baseline, candidate)
    paired = tuple(delta for delta in deltas if delta.status is PairStatus.PAIRED)
    if not paired:
        return None, GateEvaluationErrorCode.ZERO_PAIRED_QUERIES

    metric_deltas = tuple(_metric_delta(delta, validated_policy.metric) for delta in paired)
    observed_mean_delta = math.fsum(metric_deltas) / len(metric_deltas)
    violations = sorted(
        (
            GateQueryViolation(query_id=delta.query_id, observed_delta=observed)
            for delta, observed in zip(paired, metric_deltas, strict=True)
            if observed < -validated_policy.max_query_drop
        ),
        key=lambda violation: (violation.observed_delta, str(violation.query_id)),
    )

    failed_candidate_queries = sum(outcome.metrics is None for outcome in candidate)
    observed_error_rate = failed_candidate_queries / _CANONICAL_QUERY_COUNT
    paired_query_count = len(paired)
    checks = (
        GateCandidateErrorRateCheck(
            passed=observed_error_rate <= validated_policy.max_error_rate,
            failed_candidate_queries=failed_candidate_queries,
            observed_error_rate=observed_error_rate,
            max_error_rate=validated_policy.max_error_rate,
        ),
        GatePairedQueryCoverageCheck(
            passed=paired_query_count >= validated_policy.min_paired_queries,
            paired_query_count=paired_query_count,
            excluded_query_count=_CANONICAL_QUERY_COUNT - paired_query_count,
            min_paired_queries=validated_policy.min_paired_queries,
        ),
        GateAggregateDeltaCheck(
            passed=observed_mean_delta >= validated_policy.min_delta,
            metric=validated_policy.metric,
            paired_query_count=paired_query_count,
            observed_mean_delta=observed_mean_delta,
            min_delta=validated_policy.min_delta,
        ),
        GatePerQueryDropCheck(
            passed=not violations,
            metric=validated_policy.metric,
            paired_query_count=paired_query_count,
            max_query_drop=validated_policy.max_query_drop,
            violating_query_count=len(violations),
            violations=tuple(violations[:_MAX_QUERY_VIOLATIONS]),
        ),
    )
    verdict = (
        GateVerdict.PASSED if all(check.passed for check in checks) else GateVerdict.POLICY_FAILED
    )
    return (
        GateReport(
            verdict=verdict,
            run_id=run_id,
            baseline_config_id=baseline_config_id,
            candidate_config_id=candidate_config_id,
            metric=validated_policy.metric,
            checks=checks,
        ),
        None,
    )


def evaluate_gate(
    *,
    run_id: UUID,
    baseline_config_id: UUID,
    candidate_config_id: UUID,
    expected_query_ids: Sequence[UUID],
    baseline_outcomes: Sequence[QueryOutcome],
    candidate_outcomes: Sequence[QueryOutcome],
    policy: GatePolicy,
) -> GateReport:
    """Evaluate exact canonical evidence against an inclusive finite quality policy.

    Both configurations must contain exactly one outcome for each of the 50 expected query UUIDs.
    Typed failures are valid evidence and affect candidate error rate and paired coverage; missing,
    duplicate, extra, malformed, or zero-pair evidence fails closed without a policy verdict.
    """

    report, error_code = _evaluate_gate_result(
        run_id=run_id,
        baseline_config_id=baseline_config_id,
        candidate_config_id=candidate_config_id,
        expected_query_ids=expected_query_ids,
        baseline_outcomes=baseline_outcomes,
        candidate_outcomes=candidate_outcomes,
        policy=policy,
    )
    if error_code is not None:
        # Do not retain free-form failure/warning strings in the safe exception's library frame.
        del expected_query_ids, baseline_outcomes, candidate_outcomes, policy
        raise GateEvaluationError(error_code) from None
    assert report is not None
    return report
