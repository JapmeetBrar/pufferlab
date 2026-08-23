"""Pure construction of bounded, source-authenticated live replay evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pufferlab.contracts.evals import Qrel
from pufferlab.contracts.forensics import (
    CandidateCountEvidenceValue,
    EvidenceCertainty,
    EvidenceItem,
    EvidenceOrigin,
    ForensicCode,
    ForensicObservation,
    ForensicWarning,
    ForensicWarningCode,
    PresenceEvidenceValue,
    ProbeStageMembership,
    RankEvidenceValue,
    ReplayCounterfactualProbe,
    ReplayFailedCounterfactualProbe,
    ReplayProbeCandidate,
    RrfContributionEvidenceValue,
    ScoreEvidenceValue,
    WarningEvidenceValue,
)
from pufferlab.contracts.retrieval import RetrievalConfig, RetrievalMode
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareResponse,
    SearchHit,
    StageMembership,
)
from pufferlab.retrieval.rrf import RrfEntry, reconstruct_rrf
from pufferlab.retrieval.types import HybridProbeExecuteResult

_MAX_FORENSIC_TARGETS = 100
_SNAPSHOT_WARNING = (
    "The separate candidate probe did not reconstruct the primary RRF prefix. The requests may "
    "have observed different snapshots or an undocumented server tie-break."
)
_PROBE_FAILURE_WARNING = (
    "The optional candidate probe failed. The primary replay remains available, but candidate "
    "membership is not observable for this probe."
)


@dataclass(frozen=True, slots=True)
class CounterfactualProbeAnalysis:
    probe: ReplayCounterfactualProbe
    reconstruction: tuple[RrfEntry, ...]


def exact_qrel_grades(qrels: list[Qrel]) -> dict[UUID, int]:
    """Return exact stored grades and reject ambiguous or unbounded forensic targets."""
    if len(qrels) > _MAX_FORENSIC_TARGETS:
        raise ValueError("live replay supports at most 100 exact judged documents")
    grades: dict[UUID, int] = {}
    for qrel in qrels:
        if qrel.document_id in grades:
            raise ValueError("live replay qrels must have unique document identities")
        grades[qrel.document_id] = qrel.relevance_grade
    return grades


def annotate_primary_with_exact_grades(
    primary: SearchCompareResponse,
    grades: dict[UUID, int],
) -> SearchCompareResponse:
    """Copy a primary response while replacing binary relevance with exact stored grades."""
    return primary.model_copy(
        update={
            "results": [
                result.model_copy(
                    update={
                        "hits": [
                            hit.model_copy(update={"relevance_grade": grades.get(hit.document_id)})
                            for hit in result.hits
                        ]
                    }
                )
                for result in primary.results
            ]
        }
    )


def analyze_counterfactual_probe(
    execution: HybridProbeExecuteResult,
    *,
    observed_at: datetime,
    config: RetrievalConfig,
    primary: ConfigSearchResult,
) -> CounterfactualProbeAnalysis:
    """Map one explicit raw-list request and compare only its reconstructed prefix."""
    if execution.config_id != config.id or primary.config.id != config.id:
        raise ValueError("counterfactual probe inputs must bind to one exact config")
    if config.rrf is None or config.mode not in {
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }:
        raise ValueError("counterfactual probe analysis requires a hybrid config")

    candidates = [
        ReplayProbeCandidate(
            document_id=candidate.document_id,
            stage_membership=[
                ProbeStageMembership(
                    stage=membership.stage,
                    rank=membership.rank,
                    score=membership.score,
                )
                for membership in candidate.stage_membership
            ],
        )
        for candidate in execution.candidates
    ]
    probe = ReplayCounterfactualProbe(
        config_id=config.id,
        observed_at=observed_at,
        trace_id=execution.trace_id,
        duration_ms=execution.duration_ms,
        bm25_candidate_count=execution.bm25_candidate_count,
        vector_candidate_count=execution.vector_candidate_count,
        candidates=candidates,
        warnings=[],
    )
    ranked_inputs = (
        _probe_stage_ids(probe, RetrievalStage.BM25_CANDIDATES),
        _probe_stage_ids(probe, RetrievalStage.VECTOR_CANDIDATES),
    )
    reconstruction = reconstruct_rrf(
        ranked_inputs,
        rank_constant=config.rrf.rank_constant,
        weights=config.rrf.weights,
    )
    primary_prefix = _primary_rrf_prefix(primary)
    reconstructed_prefix = tuple(
        UUID(str(entry.document_id)) for entry in reconstruction[: len(primary_prefix)]
    )
    if reconstructed_prefix != primary_prefix:
        probe = probe.model_copy(
            update={
                "warnings": [
                    ForensicWarning(
                        code=ForensicWarningCode.PROVENANCE_SNAPSHOT_DIFFERS,
                        message=_SNAPSHOT_WARNING,
                    )
                ]
            }
        )
    return CounterfactualProbeAnalysis(probe=probe, reconstruction=reconstruction)


def failed_counterfactual_probe(
    *,
    config_id: UUID,
    observed_at: datetime,
    trace_id: UUID,
) -> ReplayFailedCounterfactualProbe:
    return ReplayFailedCounterfactualProbe(
        config_id=config_id,
        observed_at=observed_at,
        trace_id=trace_id,
        warning=ForensicWarning(
            code=ForensicWarningCode.PROVENANCE_PROBE_FAILED,
            message=_PROBE_FAILURE_WARNING,
        ),
    )


def build_forensic_observations(
    *,
    primary: SearchCompareResponse,
    primary_observed_at: datetime,
    config_ids: tuple[UUID, UUID],
    configs: dict[UUID, RetrievalConfig],
    target_document_ids: tuple[UUID, ...],
    probe_analyses: dict[UUID, CounterfactualProbeAnalysis],
    failed_probes: dict[UUID, ReplayFailedCounterfactualProbe],
) -> list[ForensicObservation]:
    """Classify each bounded judged target using exactly one returned source at a time."""
    if len(target_document_ids) > _MAX_FORENSIC_TARGETS:
        raise ValueError("live replay supports at most 100 forensic targets")
    primary_by_config = {result.config.id: result for result in primary.results}
    if tuple(primary_by_config) != config_ids or set(configs) != set(config_ids):
        raise ValueError("forensic inputs must retain exact replay config order")

    observations: list[ForensicObservation] = []
    for config_id in config_ids:
        config = configs[config_id]
        result = primary_by_config[config_id]
        for document_id in target_document_ids:
            reranked = _reranked_down_observation(
                result,
                config=config,
                document_id=document_id,
                observed_at=primary_observed_at,
            )
            if reranked is not None:
                observations.append(reranked)
                continue
            if any(hit.document_id == document_id for hit in result.hits):
                continue
            failed = failed_probes.get(config_id)
            if failed is not None:
                observations.append(_failed_probe_observation(failed, document_id=document_id))
                continue
            analysis = probe_analyses.get(config_id)
            if analysis is not None:
                observation = _probe_observation(
                    analysis,
                    config=config,
                    primary=result,
                    document_id=document_id,
                )
                if observation is not None:
                    observations.append(observation)
                continue
            observations.append(
                _primary_not_observable(
                    result,
                    document_id=document_id,
                    observed_at=primary_observed_at,
                )
            )
    if len(observations) > 200:
        raise ValueError("live replay produced too many forensic observations")
    return observations


def _reranked_down_observation(
    primary: ConfigSearchResult,
    *,
    config: RetrievalConfig,
    document_id: UUID,
    observed_at: datetime,
) -> ForensicObservation | None:
    if config.mode is not RetrievalMode.HYBRID_RERANK:
        return None
    hit = next((item for item in primary.hits if item.document_id == document_id), None)
    if hit is None:
        return None
    rrf = _one_membership(hit, RetrievalStage.RRF)
    reranker = _one_membership(hit, RetrievalStage.RERANKER)
    if rrf is None or reranker is None or reranker.rank <= rrf.rank:
        return None
    evidence = [
        _evidence(
            "rrf_rank",
            RankEvidenceValue(stage=RetrievalStage.RRF, rank=rrf.rank),
            origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
            observed_at=observed_at,
            trace_id=primary.trace_id,
        ),
        _evidence(
            "reranker_rank",
            RankEvidenceValue(stage=RetrievalStage.RERANKER, rank=reranker.rank),
            origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
            observed_at=observed_at,
            trace_id=primary.trace_id,
        ),
    ]
    if rrf.score is not None:
        evidence.append(
            _evidence(
                "rrf_score",
                ScoreEvidenceValue(stage=RetrievalStage.RRF, score=rrf.score),
                origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
                observed_at=observed_at,
                trace_id=primary.trace_id,
            )
        )
    if reranker.score is not None:
        evidence.append(
            _evidence(
                "reranker_score",
                ScoreEvidenceValue(stage=RetrievalStage.RERANKER, score=reranker.score),
                origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
                observed_at=observed_at,
                trace_id=primary.trace_id,
            )
        )
    return ForensicObservation(
        config_id=config.id,
        document_id=document_id,
        code=ForensicCode.RERANKED_DOWN,
        statement=(
            "The primary replay returned this document at a lower rank after local reranking. "
            "Only the observed ranks and scores are reported."
        ),
        origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
        observed_at=observed_at,
        trace_id=primary.trace_id,
        evidence=evidence,
        certainty=EvidenceCertainty.OBSERVED,
    )


def _failed_probe_observation(
    failed: ReplayFailedCounterfactualProbe,
    *,
    document_id: UUID,
) -> ForensicObservation:
    warning = _evidence(
        "provenance_probe_failure",
        WarningEvidenceValue(code=ForensicWarningCode.PROVENANCE_PROBE_FAILED),
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=failed.observed_at,
        trace_id=failed.trace_id,
    )
    return ForensicObservation(
        config_id=failed.config_id,
        document_id=document_id,
        code=ForensicCode.NOT_OBSERVABLE,
        statement=(
            "The separate candidate probe failed, so candidate membership for this document is "
            "not observable."
        ),
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=failed.observed_at,
        trace_id=failed.trace_id,
        evidence=[warning],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )


def _probe_observation(
    analysis: CounterfactualProbeAnalysis,
    *,
    config: RetrievalConfig,
    primary: ConfigSearchResult,
    document_id: UUID,
) -> ForensicObservation | None:
    probe = analysis.probe
    candidate = next(
        (item for item in probe.candidates if item.document_id == document_id),
        None,
    )
    memberships = (
        {} if candidate is None else {item.stage: item for item in candidate.stage_membership}
    )
    bm25 = memberships.get(RetrievalStage.BM25_CANDIDATES)
    vector = memberships.get(RetrievalStage.VECTOR_CANDIDATES)
    base_evidence = [
        _evidence(
            "bm25_candidate_count",
            CandidateCountEvidenceValue(
                stage=RetrievalStage.BM25_CANDIDATES,
                count=probe.bm25_candidate_count,
            ),
            origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            observed_at=probe.observed_at,
            trace_id=probe.trace_id,
        ),
        _evidence(
            "vector_candidate_count",
            CandidateCountEvidenceValue(
                stage=RetrievalStage.VECTOR_CANDIDATES,
                count=probe.vector_candidate_count,
            ),
            origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            observed_at=probe.observed_at,
            trace_id=probe.trace_id,
        ),
    ]
    if bm25 is None and vector is None:
        evidence = [
            *base_evidence,
            _presence(probe, document_id, RetrievalStage.BM25_CANDIDATES, False),
            _presence(probe, document_id, RetrievalStage.VECTOR_CANDIDATES, False),
        ]
        return _counterfactual_observation(
            probe,
            document_id=document_id,
            code=ForensicCode.NOT_OBSERVABLE,
            statement=(
                "The separate bounded probe returned this document in neither candidate list; "
                "the primary miss cause remains unobservable."
            ),
            evidence=evidence,
            certainty=EvidenceCertainty.INSUFFICIENT,
        )
    if bm25 is None:
        assert vector is not None
        return _counterfactual_observation(
            probe,
            document_id=document_id,
            code=ForensicCode.OUTSIDE_LEXICAL_CANDIDATES,
            statement=(
                "The separate bounded probe did not return this document in its lexical "
                "candidate list. This does not establish the cause of the primary result."
            ),
            evidence=[
                *base_evidence,
                _presence(probe, document_id, RetrievalStage.BM25_CANDIDATES, False),
                *_membership_evidence(probe, document_id, vector),
            ],
            certainty=EvidenceCertainty.COUNTERFACTUAL,
        )
    if vector is None:
        assert bm25 is not None
        return _counterfactual_observation(
            probe,
            document_id=document_id,
            code=ForensicCode.OUTSIDE_VECTOR_CANDIDATES,
            statement=(
                "The separate bounded probe did not return this document in its vector candidate "
                "list. This does not establish the cause of the primary result."
            ),
            evidence=[
                *base_evidence,
                _presence(probe, document_id, RetrievalStage.VECTOR_CANDIDATES, False),
                *_membership_evidence(probe, document_id, bm25),
            ],
            certainty=EvidenceCertainty.COUNTERFACTUAL,
        )
    if config.rrf is None:
        raise ValueError("hybrid observation requires an exact RRF specification")
    reconstruction = {
        UUID(str(entry.document_id)): (rank, entry)
        for rank, entry in enumerate(analysis.reconstruction, start=1)
    }
    reconstructed = reconstruction.get(document_id)
    if reconstructed is None:
        raise ValueError("probe reconstruction omitted a returned candidate")
    reconstructed_rank, _entry = reconstructed
    evidence = [*base_evidence]
    for membership, weight in zip(
        (bm25, vector),
        config.rrf.weights,
        strict=True,
    ):
        evidence.extend(_membership_evidence(probe, document_id, membership))
        evidence.append(
            _evidence(
                f"{membership.stage.value}_rrf_contribution",
                RrfContributionEvidenceValue(
                    stage=membership.stage,
                    rank=membership.rank,
                    weight=weight,
                    rank_constant=config.rrf.rank_constant,
                    contribution=weight / (config.rrf.rank_constant + membership.rank),
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=probe.observed_at,
                trace_id=probe.trace_id,
            )
        )
    if reconstructed_rank <= len(_primary_rrf_prefix(primary)):
        return ForensicObservation(
            config_id=config.id,
            document_id=document_id,
            code=ForensicCode.NOT_OBSERVABLE,
            statement=(
                "The separate probe reconstruction and the primary returned prefix differ for "
                "this document, so the primary miss cause is not observable."
            ),
            origin=EvidenceOrigin.CLIENT_COMPUTED,
            observed_at=probe.observed_at,
            trace_id=probe.trace_id,
            evidence=evidence,
            certainty=EvidenceCertainty.INSUFFICIENT,
        )
    return ForensicObservation(
        config_id=config.id,
        document_id=document_id,
        code=ForensicCode.OUTSIDE_FUSION_TOP_K,
        statement=(
            "The separate probe inputs place this document outside the reconstructed returned "
            "prefix. The computation is counterfactual and does not explain the primary result."
        ),
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=probe.observed_at,
        trace_id=probe.trace_id,
        evidence=evidence,
        certainty=EvidenceCertainty.COUNTERFACTUAL,
    )


def _primary_not_observable(
    primary: ConfigSearchResult,
    *,
    document_id: UUID,
    observed_at: datetime,
) -> ForensicObservation:
    return ForensicObservation(
        config_id=primary.config.id,
        document_id=document_id,
        code=ForensicCode.NOT_OBSERVABLE,
        statement=(
            "The primary replay did not return this document, but it exposes no evidence that "
            "establishes why."
        ),
        origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
        observed_at=observed_at,
        trace_id=primary.trace_id,
        evidence=[
            _evidence(
                "final_presence",
                PresenceEvidenceValue(stage=RetrievalStage.FINAL, present=False),
                origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
                observed_at=observed_at,
                trace_id=primary.trace_id,
            )
        ],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )


def _counterfactual_observation(
    probe: ReplayCounterfactualProbe,
    *,
    document_id: UUID,
    code: ForensicCode,
    statement: str,
    evidence: list[EvidenceItem],
    certainty: EvidenceCertainty,
) -> ForensicObservation:
    return ForensicObservation(
        config_id=probe.config_id,
        document_id=document_id,
        code=code,
        statement=statement,
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=probe.observed_at,
        trace_id=probe.trace_id,
        evidence=evidence,
        certainty=certainty,
    )


def _presence(
    probe: ReplayCounterfactualProbe,
    document_id: UUID,
    stage: RetrievalStage,
    present: bool,
) -> EvidenceItem:
    del document_id
    return _evidence(
        f"{stage.value}_presence",
        PresenceEvidenceValue(stage=stage, present=present),
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=probe.observed_at,
        trace_id=probe.trace_id,
    )


def _membership_evidence(
    probe: ReplayCounterfactualProbe,
    document_id: UUID,
    membership: ProbeStageMembership,
) -> list[EvidenceItem]:
    del document_id
    evidence = [
        _evidence(
            f"{membership.stage.value}_presence",
            PresenceEvidenceValue(stage=membership.stage, present=True),
            origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            observed_at=probe.observed_at,
            trace_id=probe.trace_id,
        ),
        _evidence(
            f"{membership.stage.value}_rank",
            RankEvidenceValue(stage=membership.stage, rank=membership.rank),
            origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            observed_at=probe.observed_at,
            trace_id=probe.trace_id,
        ),
    ]
    if membership.score is not None:
        evidence.append(
            _evidence(
                f"{membership.stage.value}_score",
                ScoreEvidenceValue(stage=membership.stage, score=membership.score),
                origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
                observed_at=probe.observed_at,
                trace_id=probe.trace_id,
            )
        )
    return evidence


def _evidence(
    label: str,
    value: object,
    *,
    origin: EvidenceOrigin,
    observed_at: datetime,
    trace_id: UUID,
) -> EvidenceItem:
    return EvidenceItem.model_validate(
        {
            "label": label,
            "value": value,
            "origin": origin,
            "observed_at": observed_at,
            "trace_id": trace_id,
        }
    )


def _probe_stage_ids(
    probe: ReplayCounterfactualProbe,
    stage: RetrievalStage,
) -> tuple[str, ...]:
    ranked = sorted(
        (
            (membership.rank, candidate.document_id)
            for candidate in probe.candidates
            for membership in candidate.stage_membership
            if membership.stage is stage
        ),
        key=lambda item: item[0],
    )
    return tuple(str(document_id) for _rank, document_id in ranked)


def _primary_rrf_prefix(primary: ConfigSearchResult) -> tuple[UUID, ...]:
    ranked: list[tuple[int, UUID]] = []
    for hit in primary.hits:
        membership = _one_membership(hit, RetrievalStage.RRF)
        if membership is None:
            raise ValueError("hybrid primary result lacks exact RRF membership")
        ranked.append((membership.rank, hit.document_id))
    if len({rank for rank, _document_id in ranked}) != len(ranked):
        raise ValueError("hybrid primary result duplicates an RRF rank")
    return tuple(document_id for _rank, document_id in sorted(ranked))


def _one_membership(hit: SearchHit, stage: RetrievalStage) -> StageMembership | None:
    memberships = [
        membership
        for membership in getattr(hit, "stage_membership", ())
        if membership.stage is stage
    ]
    if len(memberships) > 1:
        raise ValueError("primary result duplicates a stage membership")
    return memberships[0] if memberships else None
