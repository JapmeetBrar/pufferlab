"""CLI-only evaluation gate policy and report contracts."""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, StrictFloat, StrictInt, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion

_CANONICAL_QUERY_COUNT = 50
_MAX_QUERY_VIOLATIONS = 10


class GateMetricName(StrEnum):
    NDCG_AT_10 = "ndcg@10"
    RECALL_AT_50 = "recall@50"
    MRR_AT_10 = "mrr@10"


class GateVerdict(StrEnum):
    PASSED = "passed"
    POLICY_FAILED = "policy_failed"


class GateCheckCode(StrEnum):
    CANDIDATE_ERROR_RATE = "candidate_error_rate"
    PAIRED_QUERY_COVERAGE = "paired_query_coverage"
    AGGREGATE_DELTA = "aggregate_delta"
    PER_QUERY_DROP = "per_query_drop"


GATE_CHECK_ORDER = tuple(GateCheckCode)

type UnitInterval = Annotated[StrictFloat, Field(ge=0, le=1)]
type SignedUnitInterval = Annotated[StrictFloat, Field(ge=-1, le=1)]
type CanonicalQueryCount = Annotated[StrictInt, Field(ge=1, le=_CANONICAL_QUERY_COUNT)]


class _FrozenGateModel(ContractModel):
    model_config = ConfigDict(frozen=True)


class GatePolicy(_FrozenGateModel):
    """Finite, bounded policy accepted by the installed provider-free gate command."""

    contract_version: ContractVersion = 1
    metric: GateMetricName = GateMetricName.NDCG_AT_10
    min_delta: SignedUnitInterval = 0.0
    max_query_drop: UnitInterval = 0.2
    max_error_rate: UnitInterval = 0.0
    min_paired_queries: CanonicalQueryCount = _CANONICAL_QUERY_COUNT


class GateQueryViolation(_FrozenGateModel):
    query_id: UUID
    observed_delta: SignedUnitInterval


class GateCandidateErrorRateCheck(_FrozenGateModel):
    code: Literal[GateCheckCode.CANDIDATE_ERROR_RATE] = GateCheckCode.CANDIDATE_ERROR_RATE
    passed: bool
    failed_candidate_queries: Annotated[StrictInt, Field(ge=0, le=_CANONICAL_QUERY_COUNT)]
    sample_count: Annotated[
        StrictInt, Field(ge=_CANONICAL_QUERY_COUNT, le=_CANONICAL_QUERY_COUNT)
    ] = _CANONICAL_QUERY_COUNT
    observed_error_rate: UnitInterval
    max_error_rate: UnitInterval

    @model_validator(mode="after")
    def validate_result(self) -> "GateCandidateErrorRateCheck":
        expected_rate = self.failed_candidate_queries / self.sample_count
        if abs(self.observed_error_rate - expected_rate) > 1e-12:
            raise ValueError("candidate error rate must use all 50 attempts")
        if self.passed != (self.observed_error_rate <= self.max_error_rate):
            raise ValueError("error-rate result must honor the inclusive threshold")
        return self


class GatePairedQueryCoverageCheck(_FrozenGateModel):
    code: Literal[GateCheckCode.PAIRED_QUERY_COVERAGE] = GateCheckCode.PAIRED_QUERY_COVERAGE
    passed: bool
    paired_query_count: CanonicalQueryCount
    excluded_query_count: Annotated[StrictInt, Field(ge=0, le=_CANONICAL_QUERY_COUNT - 1)]
    min_paired_queries: CanonicalQueryCount

    @model_validator(mode="after")
    def validate_result(self) -> "GatePairedQueryCoverageCheck":
        if self.paired_query_count + self.excluded_query_count != _CANONICAL_QUERY_COUNT:
            raise ValueError("paired and excluded queries must cover all 50 attempts")
        if self.passed != (self.paired_query_count >= self.min_paired_queries):
            raise ValueError("paired-query result must honor the inclusive threshold")
        return self


class GateAggregateDeltaCheck(_FrozenGateModel):
    code: Literal[GateCheckCode.AGGREGATE_DELTA] = GateCheckCode.AGGREGATE_DELTA
    passed: bool
    metric: GateMetricName
    paired_query_count: CanonicalQueryCount
    observed_mean_delta: SignedUnitInterval
    min_delta: SignedUnitInterval

    @model_validator(mode="after")
    def validate_result(self) -> "GateAggregateDeltaCheck":
        if self.passed != (self.observed_mean_delta >= self.min_delta):
            raise ValueError("aggregate-delta result must honor the inclusive threshold")
        return self


class GatePerQueryDropCheck(_FrozenGateModel):
    code: Literal[GateCheckCode.PER_QUERY_DROP] = GateCheckCode.PER_QUERY_DROP
    passed: bool
    metric: GateMetricName
    paired_query_count: CanonicalQueryCount
    max_query_drop: UnitInterval
    violating_query_count: Annotated[StrictInt, Field(ge=0, le=_CANONICAL_QUERY_COUNT)]
    violations: tuple[GateQueryViolation, ...] = Field(max_length=_MAX_QUERY_VIOLATIONS)

    @model_validator(mode="after")
    def validate_result(self) -> "GatePerQueryDropCheck":
        expected_returned = min(self.violating_query_count, _MAX_QUERY_VIOLATIONS)
        if len(self.violations) != expected_returned:
            raise ValueError("gate report must include the first ten query violations")
        if self.violating_query_count > self.paired_query_count:
            raise ValueError("violating queries cannot exceed paired queries")
        query_ids = [violation.query_id for violation in self.violations]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query violations must be unique")
        if self.passed != (self.violating_query_count == 0):
            raise ValueError("per-query result must match its violation count")
        if any(violation.observed_delta >= -self.max_query_drop for violation in self.violations):
            raise ValueError("equality at the per-query drop threshold must pass")
        ordered = sorted(
            self.violations,
            key=lambda violation: (violation.observed_delta, str(violation.query_id)),
        )
        if self.violations != tuple(ordered):
            raise ValueError("query violations must use delta then UUID contract order")
        return self


class GateReport(_FrozenGateModel):
    """Safe bounded output for a valid completed-run gate evaluation."""

    contract_version: ContractVersion = 1
    verdict: GateVerdict
    run_id: UUID
    baseline_config_id: UUID
    candidate_config_id: UUID
    metric: GateMetricName
    checks: tuple[
        GateCandidateErrorRateCheck,
        GatePairedQueryCoverageCheck,
        GateAggregateDeltaCheck,
        GatePerQueryDropCheck,
    ]

    @model_validator(mode="after")
    def validate_report(self) -> "GateReport":
        if self.baseline_config_id == self.candidate_config_id:
            raise ValueError("gate baseline and candidate configs must be distinct")
        error_rate, coverage, aggregate, per_query = self.checks
        if coverage.excluded_query_count < error_rate.failed_candidate_queries:
            raise ValueError("candidate failures must be excluded from the paired-query population")
        if aggregate.metric is not self.metric or per_query.metric is not self.metric:
            raise ValueError("every quality check must use the report metric")
        if not (
            coverage.paired_query_count
            == aggregate.paired_query_count
            == per_query.paired_query_count
        ):
            raise ValueError("every quality check must use the same paired-query population")
        expected_verdict = (
            GateVerdict.PASSED
            if all(check.passed for check in self.checks)
            else GateVerdict.POLICY_FAILED
        )
        if self.verdict is not expected_verdict:
            raise ValueError("gate verdict must match the ordered check results")
        return self
