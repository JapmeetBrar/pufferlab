"""Bounded observable-evidence and explicit live-replay contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion, ObservedScore, ScoreKind
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.search import (
    RetrievalStage,
    SearchCompareResponse,
    TimingStage,
)


class ForensicCode(StrEnum):
    FILTER_PREDICATE_FAILED = "filter_predicate_failed"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_LEXICAL_CANDIDATES = "outside_lexical_candidates"
    OUTSIDE_VECTOR_CANDIDATES = "outside_vector_candidates"
    OUTSIDE_FUSION_TOP_K = "outside_fusion_top_k"
    RERANKED_DOWN = "reranked_down"
    NOT_OBSERVABLE = "not_observable"


class EvidenceOrigin(StrEnum):
    STORED_RUN = "stored_run"
    LIVE_REPLAY_PRIMARY = "live_replay_primary"
    LIVE_REPLAY_COUNTERFACTUAL_PROBE = "live_replay_counterfactual_probe"
    CLIENT_COMPUTED = "client_computed"


class EvidenceCertainty(StrEnum):
    OBSERVED = "observed"
    COUNTERFACTUAL = "counterfactual"
    INSUFFICIENT = "insufficient"


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


type ForensicEvidenceValue = Annotated[
    RankEvidenceValue
    | ScoreEvidenceValue
    | CandidateCountEvidenceValue
    | PresenceEvidenceValue
    | FilterResultEvidenceValue
    | RrfContributionEvidenceValue
    | WarningEvidenceValue,
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
    observed_at: AwareDatetime
    trace_id: UUID | None

    @model_validator(mode="after")
    def validate_trace_provenance(self) -> EvidenceItem:
        if (
            self.origin
            in {
                EvidenceOrigin.LIVE_REPLAY_PRIMARY,
                EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
                EvidenceOrigin.CLIENT_COMPUTED,
            }
            and self.trace_id is None
        ):
            raise ValueError("live and derived evidence require the exact source trace ID")
        if self.origin is EvidenceOrigin.STORED_RUN:
            if self.trace_id is not None:
                raise ValueError("stored-run unavailability evidence cannot claim a source trace")
            if not isinstance(self.value, WarningEvidenceValue) or (
                self.value.code is not ForensicWarningCode.ORIGINAL_STAGE_EVIDENCE_UNAVAILABLE
            ):
                raise ValueError(
                    "stored-run forensic evidence is limited to unavailability warnings"
                )
        return self


class ForensicObservation(ContractModel):
    code: ForensicCode
    statement: str = Field(min_length=1, max_length=512)
    origin: EvidenceOrigin
    observed_at: AwareDatetime
    trace_id: UUID | None
    evidence: list[EvidenceItem] = Field(max_length=16)
    certainty: EvidenceCertainty

    @model_validator(mode="after")
    def validate_observability(self) -> ForensicObservation:
        labels = [item.label for item in self.evidence]
        if len(labels) != len(set(labels)):
            raise ValueError("forensic evidence labels must be unique per observation")
        if (
            self.origin
            in {
                EvidenceOrigin.LIVE_REPLAY_PRIMARY,
                EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
                EvidenceOrigin.CLIENT_COMPUTED,
            }
            and self.trace_id is None
        ):
            raise ValueError("live and derived observations require a trace ID")
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
        primary_traces = {result.trace_id for result in self.primary.results}
        if len(primary_traces) != len(self.primary.results):
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
        probe_ids = [probe.config_id for probe in self.counterfactual_probes]
        if len(probe_ids) != len(set(probe_ids)) or any(
            config_id not in self.config_ids for config_id in probe_ids
        ):
            raise ValueError("counterfactual probes must uniquely match requested configs")
        probes_by_trace = {probe.trace_id: probe for probe in self.counterfactual_probes}
        if len(probes_by_trace) != len(self.counterfactual_probes):
            raise ValueError("counterfactual probes require distinct source traces")
        if primary_traces & probes_by_trace.keys():
            raise ValueError("primary and counterfactual sources cannot share a trace")
        for observation in self.observations:
            self._validate_source_binding(
                observation.origin,
                observation.trace_id,
                observation.observed_at,
                observation.certainty,
                observation.evidence,
                primary_traces,
                probes_by_trace,
            )
        return self

    def _validate_source_binding(
        self,
        origin: EvidenceOrigin,
        trace_id: UUID | None,
        observed_at: AwareDatetime,
        certainty: EvidenceCertainty,
        evidence: list[EvidenceItem],
        primary_traces: set[UUID],
        probes_by_trace: dict[UUID, ReplayCounterfactualProbe],
    ) -> None:
        self._validate_one_source(
            origin,
            trace_id,
            observed_at,
            certainty,
            None,
            primary_traces,
            probes_by_trace,
        )
        for item in evidence:
            self._validate_one_source(
                item.origin,
                item.trace_id,
                item.observed_at,
                certainty,
                item.value,
                primary_traces,
                probes_by_trace,
            )

    def _validate_one_source(
        self,
        origin: EvidenceOrigin,
        trace_id: UUID | None,
        observed_at: AwareDatetime,
        certainty: EvidenceCertainty,
        value: ForensicEvidenceValue | None,
        primary_traces: set[UUID],
        probes_by_trace: dict[UUID, ReplayCounterfactualProbe],
    ) -> None:
        if origin is EvidenceOrigin.STORED_RUN:
            return
        if trace_id is None:
            raise ValueError("live and derived forensic sources require a trace")
        if origin is EvidenceOrigin.LIVE_REPLAY_PRIMARY:
            if trace_id not in primary_traces or observed_at != self.primary_observed_at:
                raise ValueError(
                    "primary forensic evidence must bind to a primary result trace/time"
                )
            return
        if origin is EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE:
            probe = probes_by_trace.get(trace_id)
            if probe is None or observed_at != probe.observed_at:
                raise ValueError("counterfactual evidence must bind to a probe trace/time")
            if value is not None:
                self._validate_probe_evidence_value(value, probe)
            return
        if trace_id not in primary_traces and trace_id not in probes_by_trace:
            raise ValueError("client-computed evidence must bind to a returned source trace")
        if trace_id in probes_by_trace and certainty is EvidenceCertainty.OBSERVED:
            raise ValueError("probe-derived client computation cannot claim observed certainty")

    @staticmethod
    def _validate_probe_evidence_value(
        value: ForensicEvidenceValue,
        probe: ReplayCounterfactualProbe,
    ) -> None:
        counts = {
            RetrievalStage.BM25_CANDIDATES: probe.bm25_candidate_count,
            RetrievalStage.VECTOR_CANDIDATES: probe.vector_candidate_count,
        }
        if isinstance(value, CandidateCountEvidenceValue) and (
            value.stage not in counts or value.count != counts[value.stage]
        ):
            raise ValueError("counterfactual count evidence must equal its probe count")
        if isinstance(value, (ScoreEvidenceValue, PresenceEvidenceValue)) and (
            value.stage not in counts
        ):
            raise ValueError("counterfactual stage evidence is limited to probe candidate stages")
        if isinstance(value, (RankEvidenceValue, RrfContributionEvidenceValue)):
            count = counts.get(value.stage)
            if count is None or count == 0 or value.rank > count:
                raise ValueError("counterfactual rank evidence must fit its positive probe count")
