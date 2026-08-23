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
    observed_at: AwareDatetime | None
    trace_id: UUID | None

    @model_validator(mode="after")
    def validate_trace_provenance(self) -> EvidenceItem:
        if self.origin in {
            EvidenceOrigin.LIVE_REPLAY_PRIMARY,
            EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
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
