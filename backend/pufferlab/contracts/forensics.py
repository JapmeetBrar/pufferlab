"""Bounded observable-evidence and explicit live-replay contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from pufferlab.contracts.common import (
    ContractModel,
    ContractVersion,
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.filters import PredicateOp
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareResponse,
    TimingStage,
)


class ForensicCode(StrEnum):
    FILTER_PREDICATE_FAILED = "filter_predicate_failed"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_LEXICAL_CANDIDATES = "outside_lexical_candidates"
    OUTSIDE_VECTOR_CANDIDATES = "outside_vector_candidates"
    ANN_CANDIDATE_MISS = "ann_candidate_miss"
    OUTSIDE_FUSION_TOP_K = "outside_fusion_top_k"
    RERANKED_DOWN = "reranked_down"
    NOT_OBSERVABLE = "not_observable"


class EvidenceOrigin(StrEnum):
    STORED_RUN = "stored_run"
    LIVE_REPLAY_PRIMARY = "live_replay_primary"
    LIVE_REPLAY_COUNTERFACTUAL_PROBE = "live_replay_counterfactual_probe"
    LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC = "live_expected_document_diagnostic"
    CLIENT_COMPUTED = "client_computed"


class EvidenceCertainty(StrEnum):
    OBSERVED = "observed"
    COUNTERFACTUAL = "counterfactual"
    INSUFFICIENT = "insufficient"


class DiagnosticSignal(StrEnum):
    BM25 = "bm25"
    ANN = "ann"
    RRF = "rrf"


class DiagnosticCandidateScope(StrEnum):
    STORED_QUERY = "stored_query"
    NO_FILTER_COUNTERFACTUAL = "no_filter_counterfactual"


class DiagnosticPredicateResult(StrEnum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    NOT_OBSERVABLE = "not_observable"


class DiagnosticCutoffRelation(StrEnum):
    TARGET_PRESENT = "target_present"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_CANDIDATES = "outside_candidates"
    ANN_CANDIDATE_MISS = "ann_candidate_miss"
    NOT_OBSERVABLE = "not_observable"


class DiagnosticSubqueryRole(StrEnum):
    TARGET_LOOKUP = "target_lookup"
    STORED_QUERY_BM25_CANDIDATES = "stored_query_bm25_candidates"
    STORED_QUERY_ANN_CANDIDATES = "stored_query_ann_candidates"
    NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES = "no_filter_counterfactual_bm25_candidates"
    NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES = "no_filter_counterfactual_ann_candidates"


class DiagnosticTargetUnavailableReason(StrEnum):
    TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT = "target_unavailable_in_diagnostic_snapshot"


class _DiagnosticContractModel(ContractModel):
    """Strict family boundary that revalidates constructed nested diagnostic instances."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ForensicWarningCode(StrEnum):
    ORIGINAL_STAGE_EVIDENCE_UNAVAILABLE = "original_stage_evidence_unavailable"
    PROVENANCE_PROBE_FAILED = "provenance_probe_failed"
    PROVENANCE_SNAPSHOT_DIFFERS = "provenance_snapshot_differs"
    NAMESPACE_UNAVAILABLE = "namespace_unavailable"


class ForensicWarning(ContractModel):
    code: ForensicWarningCode
    message: str = Field(min_length=1, max_length=512)


class RankEvidenceValue(ContractModel):
    kind: Literal["rank"] = "rank"
    stage: RetrievalStage
    rank: int = Field(ge=1, le=10_000, strict=True)


class ScoreEvidenceValue(ContractModel):
    kind: Literal["score"] = "score"
    stage: RetrievalStage
    score: ObservedScore

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_score(cls, value: object) -> object:
        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, dict):
                score_value = score.get("value")
                if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                    raise ValueError("forensic scores require an explicit JSON number")
        return value

    @model_validator(mode="after")
    def validate_stage_score_kind(self) -> ScoreEvidenceValue:
        expected = {
            RetrievalStage.BM25_CANDIDATES: ScoreKind.BM25,
            RetrievalStage.VECTOR_CANDIDATES: ScoreKind.VECTOR_DISTANCE,
            RetrievalStage.RRF: ScoreKind.RRF,
            RetrievalStage.RERANKER: ScoreKind.RERANKER,
        }.get(self.stage)
        if expected is not None and self.score.kind is not expected:
            raise ValueError("forensic score kind must match its retrieval stage")
        if abs(self.score.value) > 1_000_000_000_000:
            raise ValueError("forensic score magnitude exceeds the bounded evidence contract")
        return self


class CandidateCountEvidenceValue(ContractModel):
    kind: Literal["candidate_count"] = "candidate_count"
    stage: RetrievalStage
    count: int = Field(ge=0, le=10_000, strict=True)


class PresenceEvidenceValue(ContractModel):
    kind: Literal["presence"] = "presence"
    stage: RetrievalStage
    present: bool = Field(strict=True)


class FilterResultEvidenceValue(ContractModel):
    kind: Literal["filter_result"] = "filter_result"
    field: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    )
    matched: bool = Field(strict=True)


class RrfContributionEvidenceValue(ContractModel):
    kind: Literal["rrf_contribution"] = "rrf_contribution"
    stage: Literal[
        RetrievalStage.BM25_CANDIDATES,
        RetrievalStage.VECTOR_CANDIDATES,
    ]
    rank: int = Field(ge=1, le=10_000, strict=True)
    weight: float = Field(gt=0, le=100, strict=True)
    rank_constant: int = Field(ge=1, le=10_000, strict=True)
    contribution: float = Field(gt=0, le=100, strict=True)

    @model_validator(mode="after")
    def validate_contribution(self) -> RrfContributionEvidenceValue:
        expected = self.weight / (self.rank_constant + self.rank)
        if not math.isclose(self.contribution, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("RRF contribution must equal weight / (rank_constant + rank)")
        return self


class WarningEvidenceValue(ContractModel):
    kind: Literal["warning"] = "warning"
    code: ForensicWarningCode


class DirectScoreEvidenceValue(_DiagnosticContractModel):
    kind: Literal["diagnostic_direct_score"] = "diagnostic_direct_score"
    signal: Literal[DiagnosticSignal.BM25, DiagnosticSignal.ANN]
    score: ObservedScore

    @model_validator(mode="before")
    @classmethod
    def reject_coerced_score(cls, value: object) -> object:
        return _reject_coerced_score_inputs(value, ("score",))

    @model_validator(mode="after")
    def validate_direct_score(self) -> DirectScoreEvidenceValue:
        _validate_score(self.score, signal=self.signal, direct=True)
        return self


class FilterPredicateEvidenceValue(_DiagnosticContractModel):
    kind: Literal["diagnostic_filter_result"] = "diagnostic_filter_result"
    predicate_ordinal: int = Field(ge=0, le=15, strict=True)
    predicate_path: tuple[Annotated[int, Field(ge=0, le=30, strict=True)], ...] = Field(
        min_length=1,
        max_length=8,
    )
    field: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    )
    operator: PredicateOp
    result: DiagnosticPredicateResult


class CutoffRelationEvidenceValue(_DiagnosticContractModel):
    kind: Literal["diagnostic_cutoff_relation"] = "diagnostic_cutoff_relation"
    scope: DiagnosticCandidateScope
    signal: DiagnosticSignal
    relation: DiagnosticCutoffRelation


type ForensicEvidenceValue = Annotated[
    RankEvidenceValue
    | ScoreEvidenceValue
    | CandidateCountEvidenceValue
    | PresenceEvidenceValue
    | FilterResultEvidenceValue
    | RrfContributionEvidenceValue
    | WarningEvidenceValue
    | DirectScoreEvidenceValue
    | FilterPredicateEvidenceValue
    | CutoffRelationEvidenceValue,
    Field(discriminator="kind"),
]


class EvidenceItem(ContractModel):
    label: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    value: ForensicEvidenceValue
    origin: EvidenceOrigin
    observed_at: AwareDatetime | None
    trace_id: UUID | None

    @model_validator(mode="after")
    def validate_trace_provenance(self) -> EvidenceItem:
        if self.origin in {
            EvidenceOrigin.LIVE_REPLAY_PRIMARY,
            EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
            EvidenceOrigin.CLIENT_COMPUTED,
        } and (self.trace_id is None or self.observed_at is None):
            raise ValueError("live and derived evidence require the exact source trace and time")
        if self.origin is EvidenceOrigin.STORED_RUN:
            if self.trace_id is not None or self.observed_at is not None:
                raise ValueError(
                    "stored-run unavailability evidence cannot claim a source trace/time"
                )
            if not isinstance(self.value, WarningEvidenceValue) or (
                self.value.code is not ForensicWarningCode.ORIGINAL_STAGE_EVIDENCE_UNAVAILABLE
            ):
                raise ValueError(
                    "stored-run forensic evidence is limited to unavailability warnings"
                )
        return self


class ForensicObservation(ContractModel):
    config_id: UUID
    document_id: UUID
    code: ForensicCode
    statement: str = Field(min_length=1, max_length=512)
    origin: EvidenceOrigin
    observed_at: AwareDatetime | None
    trace_id: UUID | None
    evidence: list[EvidenceItem] = Field(max_length=16)
    certainty: EvidenceCertainty

    @model_validator(mode="after")
    def validate_observability(self) -> ForensicObservation:
        labels = [item.label for item in self.evidence]
        if len(labels) != len(set(labels)):
            raise ValueError("forensic evidence labels must be unique per observation")
        if self.origin in {
            EvidenceOrigin.LIVE_REPLAY_PRIMARY,
            EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
            EvidenceOrigin.CLIENT_COMPUTED,
        } and (self.trace_id is None or self.observed_at is None):
            raise ValueError("live and derived observations require a source trace and time")
        if self.origin is not EvidenceOrigin.CLIENT_COMPUTED and any(
            item.origin is not self.origin for item in self.evidence
        ):
            raise ValueError("non-derived observations cannot merge evidence origins")
        if self.trace_id is not None and any(
            item.trace_id is not None and item.trace_id != self.trace_id for item in self.evidence
        ):
            raise ValueError("an observation cannot merge evidence from different traces")
        if (
            self.origin is EvidenceOrigin.STORED_RUN
            and self.code is not ForensicCode.NOT_OBSERVABLE
        ):
            raise ValueError("P0 stored outcomes contain no original stage evidence")
        if self.origin is EvidenceOrigin.STORED_RUN and (
            self.trace_id is not None or self.observed_at is not None
        ):
            raise ValueError("stored-run unavailability observations require null trace/time")
        if self.code is ForensicCode.NOT_OBSERVABLE:
            if self.certainty is not EvidenceCertainty.INSUFFICIENT:
                raise ValueError("NOT_OBSERVABLE requires insufficient certainty")
        elif self.certainty is EvidenceCertainty.INSUFFICIENT:
            raise ValueError(
                "supported forensic codes require observed or counterfactual certainty"
            )
        counterfactual = self.origin is EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE or any(
            item.origin is EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE for item in self.evidence
        )
        if counterfactual and self.certainty is EvidenceCertainty.OBSERVED:
            raise ValueError("counterfactual evidence cannot claim observed certainty")
        return self


_MAX_DIAGNOSTIC_DURATION_MS = 600_000.0
_SCORE_REL_TOL = 1e-12
_SCORE_ABS_TOL = 1e-15
_MISSING = object()


def _validate_score(
    score: ObservedScore,
    *,
    signal: DiagnosticSignal,
    direct: bool,
) -> None:
    value = getattr(score, "value", _MISSING)
    kind = getattr(score, "kind", _MISSING)
    direction = getattr(score, "direction", _MISSING)
    source = getattr(score, "source", _MISSING)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("diagnostic scores require an explicit JSON number")
    if not math.isfinite(value) or value < 0 or value > 1_000_000_000_000:
        raise ValueError("diagnostic scores must be finite, nonnegative, and bounded")
    if not direct and signal is DiagnosticSignal.BM25 and value == 0:
        raise ValueError("score-ranked BM25 rows require a strictly positive score")
    expected_kind = {
        DiagnosticSignal.BM25: ScoreKind.BM25,
        DiagnosticSignal.ANN: ScoreKind.VECTOR_DISTANCE,
        DiagnosticSignal.RRF: ScoreKind.RRF,
    }[signal]
    expected_direction = (
        ScoreDirection.LOWER_IS_BETTER
        if signal is DiagnosticSignal.ANN
        else ScoreDirection.HIGHER_IS_BETTER
    )
    expected_source = (
        ScoreSource.COMPUTE_ATTRIBUTE
        if direct
        else (
            ScoreSource.CLIENT_COMPUTED
            if signal is DiagnosticSignal.RRF
            else ScoreSource.TURBOPUFFER_DIST
        )
    )
    if (
        kind is not expected_kind
        or direction is not expected_direction
        or source is not expected_source
    ):
        raise ValueError("diagnostic score kind, direction, and source must match its signal")


def _reject_coerced_score_inputs(value: object, fields: tuple[str, ...]) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field in fields:
        score = normalized.get(field)
        if isinstance(score, ObservedScore):
            score = score.model_dump(warnings=False)
            normalized[field] = score
        raw_number = score.get("value", _MISSING) if isinstance(score, dict) else None
        if score is not None and (
            isinstance(raw_number, bool) or not isinstance(raw_number, (int, float))
        ):
            raise ValueError(f"{field} requires an explicit JSON number")
    return normalized


def _scores_equal(left: ObservedScore, right: ObservedScore) -> bool:
    return math.isclose(
        left.value,
        right.value,
        rel_tol=_SCORE_REL_TOL,
        abs_tol=_SCORE_ABS_TOL,
    )


class ExpectedDocumentDiagnosticRequest(_DiagnosticContractModel):
    contract_version: ContractVersion = 1
    config_id: UUID
    include_no_filter_counterfactual: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_wire_primitives(cls, value: object) -> object:
        if isinstance(value, dict):
            version = value.get("contract_version", 1)
            if isinstance(version, bool) or not isinstance(version, int) or version != 1:
                raise ValueError("contract_version must be the exact integer 1")
            option = value.get("include_no_filter_counterfactual", False)
            if not isinstance(option, bool):
                raise ValueError("include_no_filter_counterfactual must be a JSON boolean")
        return value


class DiagnosticTargetLookupSubquerySummary(_DiagnosticContractModel):
    kind: Literal["target_lookup"] = "target_lookup"
    ordinal: Literal[0] = 0
    role: Literal[DiagnosticSubqueryRole.TARGET_LOOKUP] = DiagnosticSubqueryRole.TARGET_LOOKUP
    requested_limit: Literal[1] = 1
    returned_count: int = Field(ge=0, le=1, strict=True)
    target_present: bool = Field(strict=True)

    @model_validator(mode="before")
    @classmethod
    def validate_exact_literals(cls, value: object) -> object:
        if isinstance(value, dict):
            for field, expected in (("ordinal", 0), ("requested_limit", 1)):
                actual = value.get(field, expected)
                if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                    raise ValueError(f"{field} must be the exact integer {expected}")
        return value

    @model_validator(mode="after")
    def validate_lookup_presence(self) -> DiagnosticTargetLookupSubquerySummary:
        if self.target_present is not (self.returned_count == 1):
            raise ValueError("target lookup presence must exactly match its zero-or-one row count")
        return self


class DiagnosticCandidateSubquerySummary(_DiagnosticContractModel):
    kind: Literal["candidate"] = "candidate"
    ordinal: int = Field(ge=1, le=4, strict=True)
    role: Literal[
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ]
    requested_limit: Literal[50, 100]
    returned_count: int = Field(ge=0, le=100, strict=True)
    target_present: bool = Field(strict=True)
    target_rank: int | None = Field(default=None, ge=1, le=100, strict=True)
    target_score: ObservedScore | None = None
    boundary_score: ObservedScore | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_exact_limit(cls, value: object) -> object:
        value = _reject_coerced_score_inputs(value, ("target_score", "boundary_score"))
        if isinstance(value, dict):
            limit = value.get("requested_limit")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit not in {50, 100}:
                raise ValueError("requested_limit must be the exact integer 50 or 100")
        return value

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> DiagnosticCandidateSubquerySummary:
        if self.returned_count > self.requested_limit:
            raise ValueError("candidate count cannot exceed the requested limit")
        if self.target_present:
            if self.target_rank is None or self.target_score is None:
                raise ValueError("present target requires an exact rank and candidate score")
            if self.target_rank > self.returned_count:
                raise ValueError("target rank must fit the returned candidate count")
        elif self.target_rank is not None or self.target_score is not None:
            raise ValueError("absent target cannot retain a rank or candidate score")
        full = self.returned_count == self.requested_limit
        if (self.boundary_score is not None) is not full:
            raise ValueError("candidate boundary score exists if and only if the list is full")
        signal = _signal_for_role(self.role)
        for score in (self.target_score, self.boundary_score):
            if score is not None:
                _validate_score(score, signal=signal, direct=False)
        return self


type DiagnosticSubquerySummary = Annotated[
    DiagnosticTargetLookupSubquerySummary | DiagnosticCandidateSubquerySummary,
    Field(discriminator="kind"),
]


class DiagnosticTargetLookup(_DiagnosticContractModel):
    config_id: UUID
    target_document_id: UUID
    origin: Literal[EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC] = (
        EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC
    )
    observed_at: AwareDatetime
    trace_id: UUID
    available: bool = Field(strict=True)
    unavailable_reason: DiagnosticTargetUnavailableReason | None = None
    bm25_score: ObservedScore | None = None
    vector_distance: ObservedScore | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_coerced_scores(cls, value: object) -> object:
        return _reject_coerced_score_inputs(value, ("bm25_score", "vector_distance"))

    @model_validator(mode="after")
    def validate_availability(self) -> DiagnosticTargetLookup:
        if self.available:
            if self.unavailable_reason is not None:
                raise ValueError("available target cannot retain an unavailable reason")
        else:
            if (
                self.unavailable_reason
                is not DiagnosticTargetUnavailableReason.TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT
            ):
                raise ValueError("unavailable target requires the fixed snapshot-unavailable code")
            if self.bm25_score is not None or self.vector_distance is not None:
                raise ValueError("unavailable target cannot retain direct scores")
        if self.bm25_score is not None:
            _validate_score(self.bm25_score, signal=DiagnosticSignal.BM25, direct=True)
        if self.vector_distance is not None:
            _validate_score(self.vector_distance, signal=DiagnosticSignal.ANN, direct=True)
        return self


class FilterPredicateEvidence(_DiagnosticContractModel):
    config_id: UUID
    target_document_id: UUID
    origin: Literal[EvidenceOrigin.CLIENT_COMPUTED] = EvidenceOrigin.CLIENT_COMPUTED
    observed_at: AwareDatetime
    trace_id: UUID
    predicate_ordinal: int = Field(ge=0, le=15, strict=True)
    predicate_path: tuple[Annotated[int, Field(ge=0, le=30, strict=True)], ...] = Field(
        min_length=1,
        max_length=8,
    )
    field: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$",
    )
    operator: PredicateOp
    result: DiagnosticPredicateResult
    certainty: EvidenceCertainty

    @model_validator(mode="after")
    def validate_certainty(self) -> FilterPredicateEvidence:
        expected = (
            EvidenceCertainty.INSUFFICIENT
            if self.result is DiagnosticPredicateResult.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        )
        if self.certainty is not expected:
            raise ValueError("filter evidence certainty must match its tri-state result")
        return self


class CandidateCutoffEvidence(_DiagnosticContractModel):
    config_id: UUID
    target_document_id: UUID
    origin: Literal[EvidenceOrigin.CLIENT_COMPUTED] = EvidenceOrigin.CLIENT_COMPUTED
    observed_at: AwareDatetime
    trace_id: UUID
    subquery_ordinal: int = Field(ge=1, le=4, strict=True)
    role: Literal[
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ]
    scope: DiagnosticCandidateScope
    signal: Literal[DiagnosticSignal.BM25, DiagnosticSignal.ANN]
    requested_limit: Literal[50, 100]
    returned_count: int = Field(ge=0, le=100, strict=True)
    target_present: bool = Field(strict=True)
    target_rank: int | None = Field(default=None, ge=1, le=100, strict=True)
    target_score: ObservedScore | None = None
    direct_score: ObservedScore
    boundary_score: ObservedScore | None = None
    relation: DiagnosticCutoffRelation
    certainty: EvidenceCertainty

    @model_validator(mode="before")
    @classmethod
    def validate_exact_limit(cls, value: object) -> object:
        value = _reject_coerced_score_inputs(
            value,
            ("target_score", "direct_score", "boundary_score"),
        )
        if isinstance(value, dict):
            limit = value.get("requested_limit")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit not in {50, 100}:
                raise ValueError("requested_limit must be the exact integer 50 or 100")
        return value

    @model_validator(mode="after")
    def validate_cutoff(self) -> CandidateCutoffEvidence:
        expected_scope, expected_signal = _scope_signal_for_role(self.role)
        if self.scope is not expected_scope or self.signal is not expected_signal:
            raise ValueError("candidate scope and signal must match its exact subquery role")
        if self.returned_count > self.requested_limit:
            raise ValueError("candidate count cannot exceed the requested limit")
        _validate_score(self.direct_score, signal=self.signal, direct=True)
        for score in (self.target_score, self.boundary_score):
            if score is not None:
                _validate_score(score, signal=self.signal, direct=False)
        if self.target_present:
            if self.target_rank is None or self.target_score is None:
                raise ValueError("present target requires an exact candidate rank and score")
            if self.target_rank > self.returned_count:
                raise ValueError("target rank must fit the returned candidate count")
            if not _scores_equal(self.target_score, self.direct_score):
                raise ValueError("candidate target score must match the direct lookup score")
            if self.signal is DiagnosticSignal.BM25 and self.direct_score.value == 0:
                raise ValueError("zero-score BM25 target cannot appear in a score-ranked list")
        elif self.target_rank is not None or self.target_score is not None:
            raise ValueError("absent target cannot retain a candidate rank or score")
        full = self.returned_count == self.requested_limit
        if (self.boundary_score is not None) is not full:
            raise ValueError("candidate boundary score exists if and only if the list is full")
        if self.target_present and self.boundary_score is not None:
            if self.signal is DiagnosticSignal.BM25 and (
                self.target_score is not None
                and self.target_score.value < self.boundary_score.value
                and not _scores_equal(self.target_score, self.boundary_score)
            ):
                raise ValueError("present BM25 target cannot score below the full-list boundary")
            if self.signal is DiagnosticSignal.ANN and (
                self.target_score is not None
                and self.target_score.value > self.boundary_score.value
                and not _scores_equal(self.target_score, self.boundary_score)
            ):
                raise ValueError("present ANN target cannot score above the full-list boundary")
        expected_relation = self._derive_relation()
        if self.relation is not expected_relation:
            raise ValueError("candidate cutoff relation must equal the bounded score facts")
        expected_certainty = (
            EvidenceCertainty.INSUFFICIENT
            if self.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            else EvidenceCertainty.COUNTERFACTUAL
            if self.scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
            else EvidenceCertainty.OBSERVED
        )
        if self.certainty is not expected_certainty:
            raise ValueError("candidate certainty must match its scope and cutoff relation")
        return self

    def _derive_relation(self) -> DiagnosticCutoffRelation:
        if self.target_present:
            return DiagnosticCutoffRelation.TARGET_PRESENT
        if self.signal is DiagnosticSignal.BM25 and self.direct_score.value == 0:
            return DiagnosticCutoffRelation.NO_LEXICAL_SCORE
        if self.boundary_score is None:
            return DiagnosticCutoffRelation.NOT_OBSERVABLE
        if _scores_equal(self.direct_score, self.boundary_score):
            return DiagnosticCutoffRelation.NOT_OBSERVABLE
        if self.signal is DiagnosticSignal.BM25:
            if self.direct_score.value > self.boundary_score.value:
                raise ValueError("absent BM25 target cannot score clearly above a full boundary")
            return DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
        if self.direct_score.value < self.boundary_score.value:
            return DiagnosticCutoffRelation.ANN_CANDIDATE_MISS
        return DiagnosticCutoffRelation.OUTSIDE_CANDIDATES


class QualifiedRrfEvidence(_DiagnosticContractModel):
    config_id: UUID
    target_document_id: UUID
    origin: Literal[EvidenceOrigin.CLIENT_COMPUTED] = EvidenceOrigin.CLIENT_COMPUTED
    observed_at: AwareDatetime
    trace_id: UUID
    scope: DiagnosticCandidateScope
    cutoff: Literal[50] = 50
    bm25_rank: int | None = Field(default=None, ge=1, le=100, strict=True)
    ann_rank: int | None = Field(default=None, ge=1, le=100, strict=True)
    bm25_weight: float = Field(gt=0, le=100, strict=True)
    ann_weight: float = Field(gt=0, le=100, strict=True)
    rank_constant: int = Field(ge=1, le=10_000, strict=True)
    returned_count: int = Field(ge=0, le=50, strict=True)
    target_present: bool = Field(strict=True)
    target_rank: int | None = Field(default=None, ge=1, le=50, strict=True)
    target_score: ObservedScore
    boundary_score: ObservedScore | None = None
    relation: Literal[
        DiagnosticCutoffRelation.TARGET_PRESENT,
        DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        DiagnosticCutoffRelation.NOT_OBSERVABLE,
    ]
    certainty: EvidenceCertainty

    @model_validator(mode="before")
    @classmethod
    def validate_wire_numbers(cls, value: object) -> object:
        value = _reject_coerced_score_inputs(value, ("target_score", "boundary_score"))
        if isinstance(value, dict):
            cutoff = value.get("cutoff", 50)
            if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff != 50:
                raise ValueError("qualified RRF cutoff must be the exact integer 50")
            for field in ("bm25_weight", "ann_weight"):
                number = value.get(field)
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    raise ValueError(f"{field} requires an explicit JSON number")
        return value

    @model_validator(mode="after")
    def validate_rrf_cutoff(self) -> QualifiedRrfEvidence:
        _validate_score(self.target_score, signal=DiagnosticSignal.RRF, direct=False)
        if self.boundary_score is not None:
            _validate_score(self.boundary_score, signal=DiagnosticSignal.RRF, direct=False)
            if self.boundary_score.value == 0:
                raise ValueError("qualified RRF boundary rows require a positive contribution")
        if self.target_present:
            if self.target_rank is None or self.target_rank > self.returned_count:
                raise ValueError("present fusion target requires a rank within the returned count")
            if self.bm25_rank is None and self.ann_rank is None:
                raise ValueError("present fusion target requires at least one candidate input rank")
        elif self.target_rank is not None:
            raise ValueError("absent fusion target cannot retain a rank")
        if (
            not self.target_present
            and self.returned_count < self.cutoff
            and (self.bm25_rank is not None or self.ann_rank is not None)
        ):
            raise ValueError("short qualified fusion cannot omit a target present in its inputs")
        expected_target_score = sum(
            weight / (self.rank_constant + rank)
            for weight, rank in (
                (self.bm25_weight, self.bm25_rank),
                (self.ann_weight, self.ann_rank),
            )
            if rank is not None
        )
        if not math.isclose(
            self.target_score.value,
            expected_target_score,
            rel_tol=_SCORE_REL_TOL,
            abs_tol=_SCORE_ABS_TOL,
        ):
            raise ValueError("qualified RRF target score must equal its bounded rank inputs")
        full = self.returned_count == self.cutoff
        if (self.boundary_score is not None) is not full:
            raise ValueError("fusion boundary exists if and only if the qualified list is full")
        if (
            self.target_present
            and self.boundary_score is not None
            and self.target_score.value < self.boundary_score.value
            and not _scores_equal(self.target_score, self.boundary_score)
        ):
            raise ValueError("present fusion target cannot score below the full-list boundary")
        expected_relation = self._derive_relation()
        if self.relation is not expected_relation:
            raise ValueError(
                "qualified RRF relation must equal its exact target and boundary facts"
            )
        expected_certainty = (
            EvidenceCertainty.INSUFFICIENT
            if self.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            else EvidenceCertainty.COUNTERFACTUAL
            if self.scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
            else EvidenceCertainty.OBSERVED
        )
        if self.certainty is not expected_certainty:
            raise ValueError("qualified RRF certainty must match its scope and relation")
        return self

    def _derive_relation(self) -> DiagnosticCutoffRelation:
        if self.target_present:
            return DiagnosticCutoffRelation.TARGET_PRESENT
        if self.boundary_score is None or _scores_equal(self.target_score, self.boundary_score):
            return DiagnosticCutoffRelation.NOT_OBSERVABLE
        if self.target_score.value > self.boundary_score.value:
            raise ValueError("absent fusion target cannot score clearly above the full boundary")
        return DiagnosticCutoffRelation.OUTSIDE_CANDIDATES


_ROLE_SCOPE_SIGNAL = {
    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES: (
        DiagnosticCandidateScope.STORED_QUERY,
        DiagnosticSignal.BM25,
    ),
    DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES: (
        DiagnosticCandidateScope.STORED_QUERY,
        DiagnosticSignal.ANN,
    ),
    DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES: (
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
        DiagnosticSignal.BM25,
    ),
    DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES: (
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
        DiagnosticSignal.ANN,
    ),
}


def _scope_signal_for_role(
    role: DiagnosticSubqueryRole,
) -> tuple[DiagnosticCandidateScope, DiagnosticSignal]:
    try:
        return _ROLE_SCOPE_SIGNAL[role]
    except KeyError as error:
        raise ValueError("target lookup has no candidate scope or signal") from error


def _signal_for_role(role: DiagnosticSubqueryRole) -> DiagnosticSignal:
    return _scope_signal_for_role(role)[1]


_ROLE_SEQUENCES = {
    (RetrievalMode.BM25, False): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
    ),
    (RetrievalMode.BM25, True): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
    ),
    (RetrievalMode.VECTOR, False): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
    ),
    (RetrievalMode.VECTOR, True): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RRF, False): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RRF, True): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RERANK, False): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RERANK, True): (
        DiagnosticSubqueryRole.TARGET_LOOKUP,
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
}


_OBSERVATION_STATEMENTS = {
    ForensicCode.FILTER_PREDICATE_FAILED: (
        "The selected target did not match a stored-query filter predicate."
    ),
    ForensicCode.NO_LEXICAL_SCORE: (
        "The selected target had no positive lexical score in this diagnostic."
    ),
    ForensicCode.OUTSIDE_LEXICAL_CANDIDATES: (
        "The selected target scored outside the lexical candidate boundary."
    ),
    ForensicCode.OUTSIDE_VECTOR_CANDIDATES: (
        "The selected target scored outside the vector candidate boundary."
    ),
    ForensicCode.ANN_CANDIDATE_MISS: (
        "The selected target beat the observed ANN boundary but was absent from that list."
    ),
    ForensicCode.OUTSIDE_FUSION_TOP_K: (
        "The selected target scored outside the qualified client-computed fusion boundary."
    ),
    ForensicCode.NOT_OBSERVABLE: (
        "The selected target's exclusion is not observable from this diagnostic."
    ),
}
_UNAVAILABLE_STATEMENT = "The selected target was unavailable in this diagnostic snapshot."


class ExpectedDocumentDiagnosticResponse(_DiagnosticContractModel):
    contract_version: ContractVersion = 1
    run_id: UUID
    query_id: UUID
    data_origin: Literal[DataOrigin.LIVE] = DataOrigin.LIVE
    origin: Literal[EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC] = (
        EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC
    )
    config_id: UUID
    config_mode: RetrievalMode
    target_document_id: UUID
    included_no_filter_counterfactual: bool = Field(strict=True)
    observed_at: AwareDatetime
    trace_id: UUID
    duration_ms: float = Field(ge=0, le=_MAX_DIAGNOSTIC_DURATION_MS, strict=True)
    embedding_duration_ms: float | None = Field(
        default=None,
        ge=0,
        le=_MAX_DIAGNOSTIC_DURATION_MS,
        strict=True,
    )
    subqueries: list[DiagnosticSubquerySummary] = Field(min_length=2, max_length=5)
    target: DiagnosticTargetLookup
    filter_evidence: list[FilterPredicateEvidence] = Field(max_length=16)
    candidate_evidence: list[CandidateCutoffEvidence] = Field(max_length=4)
    qualified_rrf_evidence: list[QualifiedRrfEvidence] = Field(max_length=2)
    observations: list[ForensicObservation] = Field(max_length=32)
    observability_notice: Literal["new_live_diagnostic_not_original_run"] = (
        "new_live_diagnostic_not_original_run"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate_wire(cls, value: object) -> object:
        if isinstance(value, cls):
            value = value.model_dump(warnings=False)
        if isinstance(value, dict):
            normalized = dict(value)
            version = normalized.get("contract_version", 1)
            if isinstance(version, bool) or not isinstance(version, int) or version != 1:
                raise ValueError("contract_version must be the exact integer 1")
            option = normalized.get("included_no_filter_counterfactual")
            if not isinstance(option, bool):
                raise ValueError("included_no_filter_counterfactual must be a JSON boolean")
            for field in ("duration_ms", "embedding_duration_ms"):
                number = normalized.get(field)
                if number is not None and isinstance(number, bool):
                    raise ValueError(f"{field} requires an explicit JSON number")
            raw_observations = normalized.get("observations")
            if isinstance(raw_observations, (list, tuple)):
                observations = []
                for item in raw_observations:
                    observation = (
                        item.model_dump(warnings=False)
                        if isinstance(item, ForensicObservation)
                        else item
                    )
                    raw_evidence = (
                        observation.get("evidence") if isinstance(observation, dict) else None
                    )
                    if isinstance(raw_evidence, (list, tuple)):
                        observation = dict(observation)
                        evidence_payloads = []
                        for evidence in raw_evidence:
                            evidence_payload = (
                                evidence.model_dump(warnings=False)
                                if isinstance(evidence, EvidenceItem)
                                else evidence
                            )
                            if isinstance(evidence_payload, dict):
                                evidence_payload = dict(evidence_payload)
                                evidence_value = evidence_payload.get("value")
                                if isinstance(evidence_value, ContractModel):
                                    evidence_payload["value"] = evidence_value.model_dump(
                                        warnings=False
                                    )
                            evidence_payloads.append(evidence_payload)
                        observation["evidence"] = evidence_payloads
                    observations.append(observation)
                normalized["observations"] = observations
            return normalized
        return value

    @model_validator(mode="after")
    def validate_diagnostic_binding(self) -> ExpectedDocumentDiagnosticResponse:
        if self.embedding_duration_ms is not None and self.embedding_duration_ms > self.duration_ms:
            raise ValueError("embedding duration cannot exceed total diagnostic duration")
        embedding_required = self.config_mode is not RetrievalMode.BM25
        if (self.embedding_duration_ms is not None) is not embedding_required:
            raise ValueError("embedding duration availability must match the selected mode")

        expected_roles = _ROLE_SEQUENCES[(self.config_mode, self.included_no_filter_counterfactual)]
        roles = tuple(item.role for item in self.subqueries)
        ordinals = tuple(item.ordinal for item in self.subqueries)
        if roles != expected_roles or ordinals != tuple(range(len(expected_roles))):
            raise ValueError("diagnostic subqueries require the exact mode/option role sequence")
        candidate_limit = (
            50 if self.config_mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 100
        )
        if any(
            isinstance(item, DiagnosticCandidateSubquerySummary)
            and item.requested_limit != candidate_limit
            for item in self.subqueries
        ):
            raise ValueError("candidate limits must exactly match the selected config mode")

        self._validate_target()
        self._validate_filter_evidence()
        self._validate_candidate_evidence()
        self._validate_qualified_rrf_evidence()
        self._validate_observations()
        return self

    def _validate_target(self) -> None:
        if (
            self.target.config_id != self.config_id
            or self.target.target_document_id != self.target_document_id
            or self.target.origin is not self.origin
            or self.target.trace_id != self.trace_id
            or self.target.observed_at != self.observed_at
        ):
            raise ValueError(
                "diagnostic target must bind to the exact response source and identity"
            )
        lookup = self.subqueries[0]
        if not isinstance(lookup, DiagnosticTargetLookupSubquerySummary):
            raise ValueError("diagnostic ordinal zero must be the exact target lookup")
        if lookup.target_present is not self.target.available:
            raise ValueError("target availability must match the exact lookup result")
        required_scores = {
            RetrievalMode.BM25: (True, False),
            RetrievalMode.VECTOR: (False, True),
            RetrievalMode.HYBRID_RRF: (True, True),
            RetrievalMode.HYBRID_RERANK: (True, True),
        }[self.config_mode]
        actual_scores = (
            self.target.bm25_score is not None,
            self.target.vector_distance is not None,
        )
        if self.target.available and actual_scores != required_scores:
            raise ValueError("available target direct scores must exactly match the selected mode")
        if not self.target.available:
            if self.filter_evidence or self.candidate_evidence or self.qualified_rrf_evidence:
                raise ValueError("unavailable target suppresses filter, cutoff, and RRF evidence")
            if any(
                isinstance(item, DiagnosticCandidateSubquerySummary) and item.target_present
                for item in self.subqueries
            ):
                raise ValueError("candidate target cannot exist when exact lookup is unavailable")

    def _validate_filter_evidence(self) -> None:
        ordinals = [item.predicate_ordinal for item in self.filter_evidence]
        paths = [item.predicate_path for item in self.filter_evidence]
        if ordinals != list(range(len(ordinals))) or len(paths) != len(set(paths)):
            raise ValueError("filter evidence requires contiguous unique ordinals and unique paths")
        for item in self.filter_evidence:
            if (
                item.config_id != self.config_id
                or item.target_document_id != self.target_document_id
                or item.trace_id != self.trace_id
                or item.observed_at != self.observed_at
            ):
                raise ValueError("filter evidence must bind to the exact diagnostic source")

    def _validate_candidate_evidence(self) -> None:
        if not self.target.available and not self.candidate_evidence:
            return
        summaries = {
            (item.ordinal, item.role): item
            for item in self.subqueries
            if isinstance(item, DiagnosticCandidateSubquerySummary)
        }
        evidence = {(item.subquery_ordinal, item.role): item for item in self.candidate_evidence}
        if len(evidence) != len(self.candidate_evidence) or set(evidence) != set(summaries):
            raise ValueError("candidate evidence must uniquely cover every candidate subquery")
        for key, item in evidence.items():
            if (
                item.config_id != self.config_id
                or item.target_document_id != self.target_document_id
                or item.trace_id != self.trace_id
                or item.observed_at != self.observed_at
            ):
                raise ValueError("candidate evidence must bind to the exact diagnostic source")
            summary = summaries[key]
            summary_facts = (
                summary.requested_limit,
                summary.returned_count,
                summary.target_present,
                summary.target_rank,
                summary.target_score,
                summary.boundary_score,
            )
            evidence_facts = (
                item.requested_limit,
                item.returned_count,
                item.target_present,
                item.target_rank,
                item.target_score,
                item.boundary_score,
            )
            if evidence_facts != summary_facts:
                raise ValueError("candidate evidence must repeat its exact safe subquery facts")
            direct = (
                self.target.bm25_score
                if item.signal is DiagnosticSignal.BM25
                else self.target.vector_distance
            )
            if direct is None or not _scores_equal(item.direct_score, direct):
                raise ValueError("candidate evidence must use the exact target lookup score")

    def _validate_qualified_rrf_evidence(self) -> None:
        hybrid = self.config_mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
        expected_scopes = [DiagnosticCandidateScope.STORED_QUERY]
        if self.included_no_filter_counterfactual:
            expected_scopes.append(DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL)
        if not self.target.available:
            expected_scopes = []
        if not hybrid:
            expected_scopes = []
        scopes = [item.scope for item in self.qualified_rrf_evidence]
        if scopes != expected_scopes:
            raise ValueError("qualified RRF evidence must exactly match hybrid diagnostic scopes")
        config_inputs = {
            (item.bm25_weight, item.ann_weight, item.rank_constant, item.cutoff)
            for item in self.qualified_rrf_evidence
        }
        if len(config_inputs) > 1:
            raise ValueError("all qualified RRF scopes must share one exact config input tuple")
        for item in self.qualified_rrf_evidence:
            if (
                item.config_id != self.config_id
                or item.target_document_id != self.target_document_id
                or item.trace_id != self.trace_id
                or item.observed_at != self.observed_at
            ):
                raise ValueError("qualified RRF evidence must bind to the exact diagnostic source")
            candidates = {
                candidate.signal: candidate
                for candidate in self.candidate_evidence
                if candidate.scope is item.scope
            }
            if (
                candidates.get(DiagnosticSignal.BM25) is None
                or candidates.get(DiagnosticSignal.ANN) is None
                or candidates[DiagnosticSignal.BM25].target_rank != item.bm25_rank
                or candidates[DiagnosticSignal.ANN].target_rank != item.ann_rank
            ):
                raise ValueError("qualified RRF ranks must match same-scope candidate evidence")
            source_count = (
                candidates[DiagnosticSignal.BM25].returned_count
                + candidates[DiagnosticSignal.ANN].returned_count
            )
            shared_target_overlap = int(
                candidates[DiagnosticSignal.BM25].target_present
                and candidates[DiagnosticSignal.ANN].target_present
            )
            minimum_union = min(
                item.cutoff,
                max(
                    candidates[DiagnosticSignal.BM25].returned_count,
                    candidates[DiagnosticSignal.ANN].returned_count,
                ),
            )
            maximum_union = min(item.cutoff, source_count - shared_target_overlap)
            if not minimum_union <= item.returned_count <= maximum_union:
                raise ValueError("qualified RRF count must fit its same-scope input union bounds")

    def _validate_observations(self) -> None:
        if not self.target.available:
            if len(self.observations) != 1:
                raise ValueError("unavailable target requires exactly one bounded observation")
            observation = self.observations[0]
            if (
                observation.code is not ForensicCode.NOT_OBSERVABLE
                or observation.statement != _UNAVAILABLE_STATEMENT
                or observation.origin is not EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC
                or observation.certainty is not EvidenceCertainty.INSUFFICIENT
                or observation.evidence
            ):
                raise ValueError("unavailable target permits only the fixed NOT_OBSERVABLE result")
            if (
                observation.config_id != self.config_id
                or observation.document_id != self.target_document_id
                or observation.trace_id != self.trace_id
                or observation.observed_at != self.observed_at
            ):
                raise ValueError("unavailable observation must bind to the exact diagnostic source")
            return
        fingerprints: list[str] = []
        for observation in self.observations:
            if (
                observation.config_id != self.config_id
                or observation.document_id != self.target_document_id
                or observation.trace_id != self.trace_id
                or observation.observed_at != self.observed_at
            ):
                raise ValueError("diagnostic observations must bind to one exact source and target")
            if observation.origin not in {
                EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
                EvidenceOrigin.CLIENT_COMPUTED,
            }:
                raise ValueError("diagnostic observations cannot reuse stored or replay origins")
            if observation.code is ForensicCode.RERANKED_DOWN:
                raise ValueError("the diagnostic cannot claim reranker behavior")
            expected_statement = (
                _UNAVAILABLE_STATEMENT
                if not self.target.available and observation.code is ForensicCode.NOT_OBSERVABLE
                else _OBSERVATION_STATEMENTS.get(observation.code)
            )
            if observation.statement != expected_statement:
                raise ValueError("diagnostic observations require fixed allowlisted statements")
            self._validate_observation_evidence(observation)
            fingerprints.append(self._observation_fingerprint(observation))
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("diagnostic observations cannot contain exact duplicates")

    @staticmethod
    def _observation_fingerprint(observation: ForensicObservation) -> str:
        filter_values = [
            item.value
            for item in observation.evidence
            if isinstance(item.value, FilterPredicateEvidenceValue)
        ]
        cutoff_values = [
            item.value
            for item in observation.evidence
            if isinstance(item.value, CutoffRelationEvidenceValue)
        ]
        triggers: list[tuple[object, ...]]
        if observation.code is ForensicCode.FILTER_PREDICATE_FAILED:
            triggers = [
                (
                    "filter",
                    value.predicate_ordinal,
                    value.predicate_path,
                    value.field,
                    value.operator.value,
                    value.result.value,
                )
                for value in filter_values
                if value.result is DiagnosticPredicateResult.NOT_MATCHED
            ]
        elif observation.code is ForensicCode.NOT_OBSERVABLE:
            triggers = [
                (
                    "filter",
                    value.predicate_ordinal,
                    value.predicate_path,
                    value.field,
                    value.operator.value,
                    value.result.value,
                )
                for value in filter_values
                if value.result is DiagnosticPredicateResult.NOT_OBSERVABLE
            ]
            triggers.extend(
                (
                    "cutoff",
                    value.scope.value,
                    value.signal.value,
                    value.relation.value,
                )
                for value in cutoff_values
                if value.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            )
        else:
            signal_relation = {
                ForensicCode.NO_LEXICAL_SCORE: (
                    DiagnosticSignal.BM25,
                    DiagnosticCutoffRelation.NO_LEXICAL_SCORE,
                ),
                ForensicCode.OUTSIDE_LEXICAL_CANDIDATES: (
                    DiagnosticSignal.BM25,
                    DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                ),
                ForensicCode.OUTSIDE_VECTOR_CANDIDATES: (
                    DiagnosticSignal.ANN,
                    DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                ),
                ForensicCode.ANN_CANDIDATE_MISS: (
                    DiagnosticSignal.ANN,
                    DiagnosticCutoffRelation.ANN_CANDIDATE_MISS,
                ),
                ForensicCode.OUTSIDE_FUSION_TOP_K: (
                    DiagnosticSignal.RRF,
                    DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                ),
            }[observation.code]
            triggers = [
                (
                    "cutoff",
                    value.scope.value,
                    value.signal.value,
                    value.relation.value,
                )
                for value in cutoff_values
                if value.signal is signal_relation[0] and value.relation is signal_relation[1]
            ]
        return repr((observation.code.value, sorted(triggers)))

    def _validate_observation_evidence(self, observation: ForensicObservation) -> None:
        counterfactual = False
        filter_values: list[FilterPredicateEvidenceValue] = []
        cutoff_values: list[CutoffRelationEvidenceValue] = []
        rrf_cutoffs = [
            item.value
            for item in observation.evidence
            if isinstance(item.value, CutoffRelationEvidenceValue)
            and item.value.signal is DiagnosticSignal.RRF
        ]
        if len(rrf_cutoffs) > 1:
            raise ValueError("one observation cannot merge multiple qualified RRF scopes")
        rrf_source = None
        if rrf_cutoffs:
            rrf_source = next(
                (
                    item
                    for item in self.qualified_rrf_evidence
                    if item.scope is rrf_cutoffs[0].scope
                    and item.relation is rrf_cutoffs[0].relation
                ),
                None,
            )
            if rrf_source is None:
                raise ValueError("qualified RRF evidence must reference one exact bounded result")
        for item in observation.evidence:
            if item.trace_id != self.trace_id or item.observed_at != self.observed_at:
                raise ValueError("observation evidence must bind to the diagnostic trace/time")
            value = item.value
            expected_label: str
            if isinstance(value, DirectScoreEvidenceValue):
                if item.origin is not EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC:
                    raise ValueError("direct score evidence requires the diagnostic source origin")
                direct = (
                    self.target.bm25_score
                    if value.signal is DiagnosticSignal.BM25
                    else self.target.vector_distance
                )
                if direct is None or value.score != direct:
                    raise ValueError("direct score evidence must equal the exact target lookup")
                expected_label = f"direct_{value.signal.value}_score"
            elif isinstance(value, FilterPredicateEvidenceValue):
                if item.origin is not EvidenceOrigin.CLIENT_COMPUTED:
                    raise ValueError("filter result evidence must be client-computed")
                matches = [
                    candidate
                    for candidate in self.filter_evidence
                    if candidate.predicate_ordinal == value.predicate_ordinal
                    and candidate.predicate_path == value.predicate_path
                    and candidate.field == value.field
                    and candidate.operator is value.operator
                    and candidate.result is value.result
                ]
                if len(matches) != 1:
                    raise ValueError("filter result evidence must reference one exact safe result")
                filter_values.append(value)
                expected_label = f"filter_predicate_{value.predicate_ordinal}"
            elif isinstance(value, CutoffRelationEvidenceValue):
                if item.origin is not EvidenceOrigin.CLIENT_COMPUTED:
                    raise ValueError("cutoff relation evidence must be client-computed")
                counterfactual = (
                    counterfactual
                    or value.scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
                )
                if value.signal is not DiagnosticSignal.RRF:
                    candidate_matches = [
                        candidate
                        for candidate in self.candidate_evidence
                        if candidate.scope is value.scope
                        and candidate.signal is value.signal
                        and candidate.relation is value.relation
                    ]
                    if len(candidate_matches) != 1:
                        raise ValueError(
                            "cutoff evidence must reference one exact candidate result"
                        )
                elif self.config_mode not in {
                    RetrievalMode.HYBRID_RRF,
                    RetrievalMode.HYBRID_RERANK,
                }:
                    raise ValueError("qualified RRF evidence requires a hybrid config")
                cutoff_values.append(value)
                expected_label = f"cutoff_{value.scope.value}_{value.signal.value}"
            elif isinstance(value, ScoreEvidenceValue):
                if item.origin is not EvidenceOrigin.CLIENT_COMPUTED:
                    raise ValueError("qualified RRF score evidence must be client-computed")
                _validate_score(value.score, signal=DiagnosticSignal.RRF, direct=False)
                if value.stage is not RetrievalStage.RRF:
                    raise ValueError("diagnostic client score evidence is limited to RRF")
                if rrf_source is None or value.score != rrf_source.target_score:
                    raise ValueError("qualified RRF score must equal its bounded target score")
                expected_label = f"qualified_rrf_score_{rrf_source.scope.value}"
            elif isinstance(value, RrfContributionEvidenceValue):
                if item.origin is not EvidenceOrigin.CLIENT_COMPUTED:
                    raise ValueError("RRF contribution evidence must be client-computed")
                if rrf_source is None:
                    raise ValueError("RRF contribution requires one qualified RRF scope")
                signal = (
                    DiagnosticSignal.BM25
                    if value.stage is RetrievalStage.BM25_CANDIDATES
                    else DiagnosticSignal.ANN
                )
                candidate = next(
                    (
                        candidate
                        for candidate in self.candidate_evidence
                        if candidate.scope is rrf_source.scope and candidate.signal is signal
                    ),
                    None,
                )
                if candidate is None or candidate.target_rank != value.rank:
                    raise ValueError("RRF contribution rank must match its exact candidate source")
                expected_weight = (
                    rrf_source.bm25_weight
                    if signal is DiagnosticSignal.BM25
                    else rrf_source.ann_weight
                )
                if (
                    value.weight != expected_weight
                    or value.rank_constant != rrf_source.rank_constant
                ):
                    raise ValueError("RRF contribution inputs must match its qualified source")
                expected_label = f"{rrf_source.scope.value}_{value.stage.value}_rrf_contribution"
            else:
                raise ValueError("diagnostic response contains an unsupported evidence value")
            if item.label != expected_label:
                raise ValueError("diagnostic evidence labels must be fixed and value-derived")

        scopes = {value.scope for value in cutoff_values}
        if len(scopes) > 1 or (
            filter_values and DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL in scopes
        ):
            raise ValueError("one observation cannot merge stored-query and no-filter scopes")
        self._validate_observation_code(observation.code, filter_values, cutoff_values)

        if observation.code is ForensicCode.NOT_OBSERVABLE:
            expected_certainty = EvidenceCertainty.INSUFFICIENT
        elif counterfactual:
            expected_certainty = EvidenceCertainty.COUNTERFACTUAL
        else:
            expected_certainty = EvidenceCertainty.OBSERVED
        if observation.certainty is not expected_certainty:
            raise ValueError("observation certainty must match its code and evidence scope")
        if observation.origin is EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC:
            if observation.code is not ForensicCode.NOT_OBSERVABLE or observation.evidence:
                raise ValueError("direct diagnostic observations are limited to unavailability")
        elif observation.origin is not EvidenceOrigin.CLIENT_COMPUTED:
            raise ValueError("supported diagnostic findings must be client-computed")

    @staticmethod
    def _validate_observation_code(
        code: ForensicCode,
        filter_values: list[FilterPredicateEvidenceValue],
        cutoff_values: list[CutoffRelationEvidenceValue],
    ) -> None:
        expected_cutoff = {
            ForensicCode.NO_LEXICAL_SCORE: (
                DiagnosticSignal.BM25,
                DiagnosticCutoffRelation.NO_LEXICAL_SCORE,
            ),
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES: (
                DiagnosticSignal.BM25,
                DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
            ),
            ForensicCode.OUTSIDE_VECTOR_CANDIDATES: (
                DiagnosticSignal.ANN,
                DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
            ),
            ForensicCode.ANN_CANDIDATE_MISS: (
                DiagnosticSignal.ANN,
                DiagnosticCutoffRelation.ANN_CANDIDATE_MISS,
            ),
            ForensicCode.OUTSIDE_FUSION_TOP_K: (
                DiagnosticSignal.RRF,
                DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
            ),
        }
        if code is ForensicCode.FILTER_PREDICATE_FAILED:
            if not any(
                value.result is DiagnosticPredicateResult.NOT_MATCHED for value in filter_values
            ):
                raise ValueError("filter-failed code requires an exact not-matched predicate")
            return
        if code is ForensicCode.NOT_OBSERVABLE:
            if not any(
                value.result is DiagnosticPredicateResult.NOT_OBSERVABLE for value in filter_values
            ) and not any(
                value.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE for value in cutoff_values
            ):
                raise ValueError("NOT_OBSERVABLE requires one exact insufficient result")
            return
        signal_relation = expected_cutoff.get(code)
        if signal_relation is None or not any(
            value.signal is signal_relation[0] and value.relation is signal_relation[1]
            for value in cutoff_values
        ):
            raise ValueError("forensic code must match its exact diagnostic evidence result")


class EvalRunQueryReplayRequest(ContractModel):
    contract_version: ContractVersion = 1
    config_ids: list[UUID] = Field(min_length=2, max_length=2)
    include_counterfactual_probe: bool = False

    @model_validator(mode="after")
    def validate_config_ids(self) -> EvalRunQueryReplayRequest:
        if len(set(self.config_ids)) != 2:
            raise ValueError("replay config IDs must be distinct")
        return self


class ReplayProbeCandidate(ContractModel):
    document_id: UUID
    stage_membership: list[ProbeStageMembership] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_probe_stages(self) -> ReplayProbeCandidate:
        stages = [membership.stage for membership in self.stage_membership]
        allowed = {
            RetrievalStage.BM25_CANDIDATES,
            RetrievalStage.VECTOR_CANDIDATES,
        }
        if any(stage not in allowed for stage in stages) or len(stages) != len(set(stages)):
            raise ValueError("probe candidates require unique BM25/vector memberships only")
        return self


class ProbeStageMembership(ContractModel):
    stage: Literal[
        RetrievalStage.BM25_CANDIDATES,
        RetrievalStage.VECTOR_CANDIDATES,
    ]
    rank: int = Field(ge=1, le=10_000, strict=True)
    score: ObservedScore | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_score(cls, value: object) -> object:
        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, dict):
                score_value = score.get("value")
                if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                    raise ValueError("probe scores require an explicit JSON number")
        return value

    @model_validator(mode="after")
    def validate_stage_score_kind(self) -> ProbeStageMembership:
        if self.score is None:
            return self
        expected = (
            ScoreKind.BM25
            if self.stage is RetrievalStage.BM25_CANDIDATES
            else ScoreKind.VECTOR_DISTANCE
        )
        if self.score.kind is not expected:
            raise ValueError("probe score kind must match its candidate stage")
        if abs(self.score.value) > 1_000_000_000_000:
            raise ValueError("probe score magnitude exceeds the bounded evidence contract")
        return self


class ReplayCounterfactualProbe(ContractModel):
    origin: Literal[EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE] = (
        EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE
    )
    config_id: UUID
    observed_at: AwareDatetime
    trace_id: UUID
    duration_ms: float = Field(ge=0)
    bm25_candidate_count: int = Field(ge=0, le=10_000, strict=True)
    vector_candidate_count: int = Field(ge=0, le=10_000, strict=True)
    candidates: list[ReplayProbeCandidate] = Field(max_length=200)
    warnings: list[ForensicWarning] = Field(max_length=10)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> ReplayCounterfactualProbe:
        document_ids = [candidate.document_id for candidate in self.candidates]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("a counterfactual probe cannot duplicate candidate documents")
        ranks_by_stage: dict[RetrievalStage, list[int]] = {
            RetrievalStage.BM25_CANDIDATES: [],
            RetrievalStage.VECTOR_CANDIDATES: [],
        }
        counts = {
            RetrievalStage.BM25_CANDIDATES: self.bm25_candidate_count,
            RetrievalStage.VECTOR_CANDIDATES: self.vector_candidate_count,
        }
        for candidate in self.candidates:
            for membership in candidate.stage_membership:
                count = counts[membership.stage]
                if count == 0 or membership.rank > count:
                    raise ValueError("probe membership rank must fit its positive candidate count")
                ranks_by_stage[membership.stage].append(membership.rank)
        if any(len(ranks) != len(set(ranks)) for ranks in ranks_by_stage.values()):
            raise ValueError("probe ranks must be unique within each candidate stage")
        return self


class ReplayFailedCounterfactualProbe(ContractModel):
    origin: Literal[EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE] = (
        EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE
    )
    config_id: UUID
    observed_at: AwareDatetime
    trace_id: UUID
    warning: ForensicWarning

    @model_validator(mode="after")
    def validate_safe_failure_warning(self) -> ReplayFailedCounterfactualProbe:
        if self.warning.code is not ForensicWarningCode.PROVENANCE_PROBE_FAILED:
            raise ValueError(
                "failed counterfactual probes require the safe provenance-probe-failed warning"
            )
        return self


class EvalRunQueryReplayResponse(ContractModel):
    contract_version: ContractVersion = 1
    run_id: UUID
    query_id: UUID
    data_origin: Literal[DataOrigin.LIVE] = DataOrigin.LIVE
    config_ids: list[UUID] = Field(min_length=2, max_length=2)
    primary_origin: Literal[EvidenceOrigin.LIVE_REPLAY_PRIMARY] = EvidenceOrigin.LIVE_REPLAY_PRIMARY
    primary_observed_at: AwareDatetime
    primary: SearchCompareResponse
    counterfactual_probes: list[ReplayCounterfactualProbe] = Field(max_length=2)
    failed_counterfactual_probes: list[ReplayFailedCounterfactualProbe] = Field(
        default_factory=list,
        max_length=2,
    )
    observations: list[ForensicObservation] = Field(max_length=200)
    original_stage_evidence_available: Literal[False] = False
    observability_notice: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_replay_separation(self) -> EvalRunQueryReplayResponse:
        if len(set(self.config_ids)) != 2:
            raise ValueError("replay config IDs must be distinct")
        if self.primary.query_id != self.query_id:
            raise ValueError("primary replay query identity must match the requested run query")
        if [result.config.id for result in self.primary.results] != self.config_ids:
            raise ValueError("primary replay results must retain requested config order")
        primary_by_trace = {result.trace_id: result for result in self.primary.results}
        if len(primary_by_trace) != len(self.primary.results):
            raise ValueError("primary config results require distinct source traces")
        if any(
            timing.stage is TimingStage.PROVENANCE_PROBE
            for result in self.primary.results
            for timing in result.timings
        ):
            raise ValueError("counterfactual probe timing cannot enter the primary result")
        if any(
            membership.stage in {RetrievalStage.BM25_CANDIDATES, RetrievalStage.VECTOR_CANDIDATES}
            for result in self.primary.results
            for hit in result.hits
            for membership in hit.stage_membership
        ):
            raise ValueError("counterfactual candidate membership cannot enter the primary result")
        probe_ids = [
            *(probe.config_id for probe in self.counterfactual_probes),
            *(probe.config_id for probe in self.failed_counterfactual_probes),
        ]
        if len(probe_ids) != len(set(probe_ids)) or any(
            config_id not in self.config_ids for config_id in probe_ids
        ):
            raise ValueError(
                "successful and failed counterfactual probes must uniquely match requested configs"
            )
        probes_by_trace = {probe.trace_id: probe for probe in self.counterfactual_probes}
        if len(probes_by_trace) != len(self.counterfactual_probes):
            raise ValueError("counterfactual probes require distinct source traces")
        failed_probes_by_trace = {
            probe.trace_id: probe for probe in self.failed_counterfactual_probes
        }
        if len(failed_probes_by_trace) != len(self.failed_counterfactual_probes):
            raise ValueError("failed counterfactual probes require distinct source traces")
        all_trace_ids = [
            *primary_by_trace,
            *probes_by_trace,
            *failed_probes_by_trace,
        ]
        if len(all_trace_ids) != len(set(all_trace_ids)):
            raise ValueError("primary and counterfactual sources require disjoint traces")
        for observation in self.observations:
            if observation.config_id not in self.config_ids:
                raise ValueError("forensic observation config must belong to the replay request")
            if observation.code is ForensicCode.ANN_CANDIDATE_MISS:
                raise ValueError("diagnostic-only forensic codes cannot enter legacy replay")
            self._validate_source_binding(
                observation,
                primary_by_trace,
                probes_by_trace,
                failed_probes_by_trace,
            )
        return self

    def _validate_source_binding(
        self,
        observation: ForensicObservation,
        primary_by_trace: dict[UUID, ConfigSearchResult],
        probes_by_trace: dict[UUID, ReplayCounterfactualProbe],
        failed_probes_by_trace: dict[UUID, ReplayFailedCounterfactualProbe],
    ) -> None:
        self._validate_one_source(
            observation.origin,
            observation.trace_id,
            observation.observed_at,
            observation.certainty,
            observation.code,
            observation.config_id,
            observation.document_id,
            None,
            primary_by_trace,
            probes_by_trace,
            failed_probes_by_trace,
        )
        for item in observation.evidence:
            self._validate_one_source(
                item.origin,
                item.trace_id,
                item.observed_at,
                observation.certainty,
                observation.code,
                observation.config_id,
                observation.document_id,
                item.value,
                primary_by_trace,
                probes_by_trace,
                failed_probes_by_trace,
            )

    def _validate_one_source(
        self,
        origin: EvidenceOrigin,
        trace_id: UUID | None,
        observed_at: AwareDatetime | None,
        certainty: EvidenceCertainty,
        code: ForensicCode,
        config_id: UUID,
        document_id: UUID,
        value: ForensicEvidenceValue | None,
        primary_by_trace: dict[UUID, ConfigSearchResult],
        probes_by_trace: dict[UUID, ReplayCounterfactualProbe],
        failed_probes_by_trace: dict[UUID, ReplayFailedCounterfactualProbe],
    ) -> None:
        if origin is EvidenceOrigin.STORED_RUN:
            if trace_id is not None or observed_at is not None:
                raise ValueError("stored-run forensic sources require null trace/time")
            return
        if trace_id is None or observed_at is None:
            raise ValueError("live and derived forensic sources require a trace/time")
        if origin is EvidenceOrigin.LIVE_REPLAY_PRIMARY:
            primary = primary_by_trace.get(trace_id)
            if primary is None or observed_at != self.primary_observed_at:
                raise ValueError(
                    "primary forensic evidence must bind to a primary result trace/time"
                )
            if primary.config.id != config_id:
                raise ValueError("primary forensic evidence must bind to its exact config")
            if value is not None:
                self._validate_primary_evidence_value(value, primary, document_id)
            return
        if origin is EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE:
            probe = probes_by_trace.get(trace_id)
            if probe is not None:
                if observed_at != probe.observed_at:
                    raise ValueError("counterfactual evidence must bind to a probe trace/time")
                if probe.config_id != config_id:
                    raise ValueError("counterfactual evidence must bind to its exact probe config")
                if value is not None:
                    self._validate_probe_evidence_value(value, probe, document_id)
                return
            failed_probe = failed_probes_by_trace.get(trace_id)
            if failed_probe is None or observed_at != failed_probe.observed_at:
                raise ValueError("counterfactual evidence must bind to a probe trace/time")
            if failed_probe.config_id != config_id:
                raise ValueError("counterfactual evidence must bind to its exact probe config")
            if code is not ForensicCode.NOT_OBSERVABLE or (
                certainty is not EvidenceCertainty.INSUFFICIENT
            ):
                raise ValueError(
                    "failed-probe observations must remain NOT_OBSERVABLE with "
                    "insufficient certainty"
                )
            if value is not None and (
                not isinstance(value, WarningEvidenceValue)
                or value.code is not ForensicWarningCode.PROVENANCE_PROBE_FAILED
            ):
                raise ValueError("failed-probe observations are limited to safe failure warnings")
            return
        if origin is EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC:
            raise ValueError("diagnostic evidence cannot enter a legacy replay response")
        if origin is not EvidenceOrigin.CLIENT_COMPUTED:
            raise ValueError("unsupported forensic evidence origin")
        primary = primary_by_trace.get(trace_id)
        if primary is not None:
            if observed_at != self.primary_observed_at:
                raise ValueError(
                    "client-computed evidence must bind to its primary source trace/time"
                )
            if primary.config.id != config_id:
                raise ValueError("client-computed evidence must bind to its exact primary config")
            if value is not None:
                self._validate_primary_evidence_value(value, primary, document_id)
            return
        probe = probes_by_trace.get(trace_id)
        if probe is None:
            raise ValueError("client-computed evidence must bind to a returned source trace")
        if observed_at != probe.observed_at:
            raise ValueError("client-computed evidence must bind to its probe source trace/time")
        if probe.config_id != config_id:
            raise ValueError("client-computed evidence must bind to its exact probe config")
        if certainty is EvidenceCertainty.OBSERVED:
            raise ValueError("probe-derived client computation cannot claim observed certainty")
        if value is not None:
            self._validate_probe_evidence_value(value, probe, document_id)

    @staticmethod
    def _validate_primary_evidence_value(
        value: ForensicEvidenceValue,
        primary: ConfigSearchResult,
        document_id: UUID,
    ) -> None:
        hits = [hit for hit in primary.hits if hit.document_id == document_id]
        if len(hits) > 1:
            raise ValueError("primary source cannot duplicate the forensic target document")
        hit = hits[0] if hits else None
        memberships = (
            [membership for membership in hit.stage_membership if membership.stage is value.stage]
            if hit is not None
            and isinstance(
                value,
                (
                    RankEvidenceValue,
                    ScoreEvidenceValue,
                    PresenceEvidenceValue,
                    RrfContributionEvidenceValue,
                ),
            )
            else []
        )
        if len(memberships) > 1:
            raise ValueError("primary source cannot duplicate a target stage membership")
        membership = memberships[0] if memberships else None

        if isinstance(value, CandidateCountEvidenceValue):
            if primary.candidate_counts.get(value.stage.value) != value.count:
                raise ValueError(
                    "primary candidate-count evidence must equal its exact config source"
                )
            return
        if isinstance(value, PresenceEvidenceValue):
            present = (
                hit is not None if value.stage is RetrievalStage.FINAL else membership is not None
            )
            if value.present is not present:
                raise ValueError(
                    "primary presence evidence must match the exact target document membership"
                )
            return
        if isinstance(value, RankEvidenceValue):
            rank = (
                hit.final_rank if value.stage is RetrievalStage.FINAL and hit is not None else None
            )
            if value.stage is not RetrievalStage.FINAL and membership is not None:
                rank = membership.rank
            if value.rank != rank:
                raise ValueError(
                    "primary-derived rank evidence must match the exact target document "
                    "membership/rank"
                )
            return
        if isinstance(value, ScoreEvidenceValue):
            score = (
                hit.final_score if value.stage is RetrievalStage.FINAL and hit is not None else None
            )
            if value.stage is not RetrievalStage.FINAL and membership is not None:
                score = membership.score
            if value.score != score:
                raise ValueError(
                    "primary score evidence must match the exact target document membership"
                )
            return
        if isinstance(value, RrfContributionEvidenceValue):
            if membership is None or value.rank != membership.rank:
                raise ValueError(
                    "primary RRF input rank must match the exact target document membership"
                )
            return
        raise ValueError("primary forensic evidence value is not authenticated by its source")

    @staticmethod
    def _validate_probe_evidence_value(
        value: ForensicEvidenceValue,
        probe: ReplayCounterfactualProbe,
        document_id: UUID,
    ) -> None:
        counts = {
            RetrievalStage.BM25_CANDIDATES: probe.bm25_candidate_count,
            RetrievalStage.VECTOR_CANDIDATES: probe.vector_candidate_count,
        }
        if isinstance(value, CandidateCountEvidenceValue):
            if value.stage not in counts or value.count != counts[value.stage]:
                raise ValueError("counterfactual count evidence must equal its probe count")
            return
        candidates = [
            candidate for candidate in probe.candidates if candidate.document_id == document_id
        ]
        if len(candidates) > 1:
            raise ValueError("counterfactual probe cannot duplicate the forensic target document")
        candidate = candidates[0] if candidates else None
        memberships = (
            [
                membership
                for membership in candidate.stage_membership
                if membership.stage is value.stage
            ]
            if candidate is not None
            and isinstance(
                value,
                (
                    RankEvidenceValue,
                    ScoreEvidenceValue,
                    PresenceEvidenceValue,
                    RrfContributionEvidenceValue,
                ),
            )
            else []
        )
        membership = memberships[0] if memberships else None
        if (
            isinstance(value, (ScoreEvidenceValue, PresenceEvidenceValue))
            and value.stage not in counts
        ):
            raise ValueError("counterfactual stage evidence is limited to probe candidate stages")
        if isinstance(value, ScoreEvidenceValue) and counts[value.stage] == 0:
            raise ValueError("counterfactual score evidence requires a positive probe count")
        if isinstance(value, PresenceEvidenceValue):
            if value.present and counts[value.stage] == 0:
                raise ValueError(
                    "counterfactual membership evidence requires a positive probe count"
                )
            if value.present is not (membership is not None):
                raise ValueError(
                    "counterfactual presence evidence must match the exact target document "
                    "membership"
                )
        if isinstance(value, ScoreEvidenceValue) and (
            membership is None or value.score != membership.score
        ):
            raise ValueError(
                "counterfactual score evidence must match the exact target document membership"
            )
        if isinstance(value, (RankEvidenceValue, RrfContributionEvidenceValue)):
            count = counts.get(value.stage)
            if count is None or count == 0 or value.rank > count:
                raise ValueError("counterfactual rank evidence must fit its positive probe count")
            if membership is None or value.rank != membership.rank:
                raise ValueError(
                    "counterfactual rank evidence must match the exact target document membership"
                )
        if isinstance(value, (FilterResultEvidenceValue, WarningEvidenceValue)):
            raise ValueError("counterfactual evidence value is not authenticated by its source")
        if isinstance(
            value,
            (
                RankEvidenceValue,
                ScoreEvidenceValue,
                PresenceEvidenceValue,
                RrfContributionEvidenceValue,
            ),
        ):
            return
        raise ValueError("counterfactual evidence value is not supported by legacy replay")
