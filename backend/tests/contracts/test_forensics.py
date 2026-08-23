from datetime import UTC, datetime
from uuid import UUID

import pytest
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.forensics import (
    CandidateCountEvidenceValue,
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
    EvidenceCertainty,
    EvidenceItem,
    EvidenceOrigin,
    ForensicCode,
    ForensicObservation,
    ForensicWarningCode,
    PresenceEvidenceValue,
    ProbeStageMembership,
    RankEvidenceValue,
    ReplayCounterfactualProbe,
    ReplayProbeCandidate,
    RrfContributionEvidenceValue,
    ScoreEvidenceValue,
    WarningEvidenceValue,
)
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareResponse,
    SearchHit,
    StageTiming,
    TimingStage,
)
from pydantic import ValidationError

_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
_LATER = datetime(2026, 8, 23, 13, tzinfo=UTC)
_TRACE_ID = UUID(int=100)
_QUERY_ID = UUID(int=101)
_CONFIG_IDS = [UUID(int=102), UUID(int=103)]


def _score() -> ObservedScore:
    return ObservedScore(
        kind=ScoreKind.BM25,
        value=2.5,
        direction=ScoreDirection.HIGHER_IS_BETTER,
        source=ScoreSource.TURBOPUFFER_DIST,
    )


def _evidence(
    *,
    origin: EvidenceOrigin = EvidenceOrigin.LIVE_REPLAY_PRIMARY,
    trace_id: UUID = _TRACE_ID,
    observed_at: datetime = _NOW,
) -> EvidenceItem:
    return EvidenceItem(
        label="final_score",
        value=ScoreEvidenceValue(stage="final", score=_score()),
        origin=origin,
        observed_at=observed_at,
        trace_id=trace_id,
    )


def _primary() -> SearchCompareResponse:
    results = [
        ConfigSearchResult(
            config=RetrievalConfigSummary(
                id=config_id,
                revision=1,
                name=f"config-{index}",
                mode=RetrievalMode.BM25 if index == 0 else RetrievalMode.VECTOR,
                config_hash=str(index),
            ),
            hits=[],
            timings=[],
            candidate_counts={},
            warnings=[],
            trace_id=UUID(int=200 + index),
        )
        for index, config_id in enumerate(_CONFIG_IDS)
    ]
    return SearchCompareResponse(
        query_text="server-derived query",
        query_id=_QUERY_ID,
        results=results,
        rank_movements=[],
        overlap=[],
        observability_notice="Primary replay contains production-shaped final results.",
    )


def _primary_with_final_hit() -> SearchCompareResponse:
    primary = _primary()
    primary.results[0].hits = [
        SearchHit(
            document_id=UUID(int=400),
            external_id="doc-400",
            title="Bounded result",
            body_excerpt="Server-returned excerpt.",
            final_rank=1,
            final_score=_score(),
            stage_membership=[],
        )
    ]
    return primary


def _probe() -> ReplayCounterfactualProbe:
    return ReplayCounterfactualProbe(
        config_id=_CONFIG_IDS[0],
        observed_at=_NOW,
        trace_id=_TRACE_ID,
        duration_ms=1.0,
        bm25_candidate_count=1,
        vector_candidate_count=0,
        candidates=[
            ReplayProbeCandidate(
                document_id=UUID(int=400),
                stage_membership=[ProbeStageMembership(stage="bm25_candidates", rank=1)],
            )
        ],
        warnings=[],
    )


def test_replay_request_accepts_only_two_server_resolved_config_ids() -> None:
    request = EvalRunQueryReplayRequest(config_ids=_CONFIG_IDS)

    assert request.include_counterfactual_probe is False
    for forbidden_field in ("query_text", "namespace", "expected_document_ids", "origin"):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EvalRunQueryReplayRequest.model_validate(
                {
                    "config_ids": [str(config_id) for config_id in _CONFIG_IDS],
                    forbidden_field: "browser override",
                }
            )
    with pytest.raises(ValidationError, match="must be distinct"):
        EvalRunQueryReplayRequest(config_ids=[_CONFIG_IDS[0], _CONFIG_IDS[0]])


def test_evidence_is_allowlisted_bounded_and_rejects_provider_payloads() -> None:
    payload = _evidence().model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvidenceItem.model_validate(
            {
                **payload,
                "value": {
                    "kind": "warning",
                    "code": ForensicWarningCode.PROVENANCE_PROBE_FAILED,
                    "provider_body": {"opaque": "x" * 100_000},
                },
            }
        )
    with pytest.raises(ValidationError, match="Input tag 'provider_body'"):
        EvidenceItem.model_validate(
            {
                **payload,
                "value": {"kind": "provider_body", "payload": "opaque"},
            }
        )
    with pytest.raises(ValidationError, match="finite number"):
        EvidenceItem.model_validate(
            {
                **payload,
                "value": {
                    "kind": "score",
                    "stage": "final",
                    "score": {
                        "kind": "bm25",
                        "value": float("nan"),
                        "direction": "higher_is_better",
                        "source": "turbopuffer_dist",
                    },
                },
            }
        )
    with pytest.raises(ValidationError, match="explicit JSON number"):
        ScoreEvidenceValue.model_validate(
            {
                "stage": "final",
                "score": {
                    "kind": "bm25",
                    "value": True,
                    "direction": "higher_is_better",
                    "source": "turbopuffer_dist",
                },
            }
        )
    with pytest.raises(ValidationError, match="valid integer"):
        CandidateCountEvidenceValue(stage="final", count=True)


def test_counterfactual_evidence_cannot_claim_observed_causality() -> None:
    counterfactual = _evidence(origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE)

    with pytest.raises(ValidationError, match="cannot claim observed certainty"):
        ForensicObservation(
            code=ForensicCode.OUTSIDE_VECTOR_CANDIDATES,
            statement="The document was absent from a separately requested vector candidate list.",
            origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            observed_at=_NOW,
            trace_id=_TRACE_ID,
            evidence=[counterfactual],
            certainty=EvidenceCertainty.OBSERVED,
        )

    observation = ForensicObservation(
        code=ForensicCode.OUTSIDE_VECTOR_CANDIDATES,
        statement="The document was absent from a separately requested vector candidate list.",
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=_NOW,
        trace_id=_TRACE_ID,
        evidence=[counterfactual],
        certainty=EvidenceCertainty.COUNTERFACTUAL,
    )
    assert observation.certainty is EvidenceCertainty.COUNTERFACTUAL


def test_stored_run_forensics_are_honestly_not_observable() -> None:
    unavailable = EvidenceItem(
        label="original_stage_evidence",
        value=WarningEvidenceValue(code=ForensicWarningCode.ORIGINAL_STAGE_EVIDENCE_UNAVAILABLE),
        origin=EvidenceOrigin.STORED_RUN,
        observed_at=None,
        trace_id=None,
    )
    observation = ForensicObservation(
        code=ForensicCode.NOT_OBSERVABLE,
        statement="The M2 outcome did not persist stage membership or scores.",
        origin=EvidenceOrigin.STORED_RUN,
        observed_at=None,
        trace_id=None,
        evidence=[unavailable],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )

    assert observation.trace_id is None
    assert observation.observed_at is None
    assert observation.evidence[0].observed_at is None
    for invented_source in (
        {"trace_id": _TRACE_ID, "observed_at": None},
        {"trace_id": None, "observed_at": _NOW},
    ):
        with pytest.raises(ValidationError, match="require null trace/time"):
            ForensicObservation.model_validate({**observation.model_dump(), **invented_source})
        with pytest.raises(ValidationError, match="cannot claim a source trace/time"):
            EvidenceItem.model_validate({**unavailable.model_dump(), **invented_source})
    with pytest.raises(ValidationError, match="no original stage evidence"):
        ForensicObservation.model_validate(
            {
                **observation.model_dump(),
                "code": ForensicCode.RERANKED_DOWN,
                "certainty": EvidenceCertainty.OBSERVED,
            }
        )
    with pytest.raises(ValidationError, match="limited to unavailability warnings"):
        EvidenceItem(
            label="fabricated_bm25_rank",
            value=RankEvidenceValue(stage="bm25_candidates", rank=1),
            origin=EvidenceOrigin.STORED_RUN,
            observed_at=None,
            trace_id=None,
        )
    with pytest.raises(ValidationError, match="limited to unavailability warnings"):
        ForensicObservation.model_validate(
            {
                **observation.model_dump(),
                "evidence": [
                    {
                        "label": "fabricated_bm25_rank",
                        "value": {"kind": "rank", "stage": "bm25_candidates", "rank": 1},
                        "origin": "stored_run",
                        "observed_at": None,
                        "trace_id": None,
                    }
                ],
            }
        )


def test_rrf_evidence_and_probe_membership_are_strict_and_bounded() -> None:
    contribution = RrfContributionEvidenceValue(
        stage="bm25_candidates",
        rank=4,
        weight=1.0,
        rank_constant=60,
        contribution=1.0 / 64.0,
    )

    assert contribution.contribution == 1.0 / 64.0
    with pytest.raises(ValidationError, match="must equal"):
        RrfContributionEvidenceValue(
            stage="bm25_candidates",
            rank=4,
            weight=1.0,
            rank_constant=60,
            contribution=0.5,
        )
    with pytest.raises(ValidationError, match="probe scores require an explicit JSON number"):
        ProbeStageMembership.model_validate(
            {
                "stage": "bm25_candidates",
                "rank": 1,
                "score": {
                    "kind": "bm25",
                    "value": True,
                    "direction": "higher_is_better",
                    "source": "turbopuffer_dist",
                },
            }
        )


def test_replay_envelope_separates_primary_and_probe_evidence() -> None:
    response = EvalRunQueryReplayResponse(
        run_id=UUID(int=300),
        query_id=_QUERY_ID,
        config_ids=_CONFIG_IDS,
        primary_observed_at=_NOW,
        primary=_primary(),
        counterfactual_probes=[],
        observations=[],
        observability_notice="Original stage evidence is unavailable; this is a live replay.",
    )

    payload = response.model_dump(mode="json")
    assert payload["primary_origin"] == "live_replay_primary"
    assert payload["data_origin"] == "live"
    assert payload["original_stage_evidence_available"] is False

    primary = _primary()
    primary.results[0].timings.append(
        StageTiming(stage=TimingStage.PROVENANCE_PROBE, duration_ms=1.0)
    )
    with pytest.raises(ValidationError, match="probe timing cannot enter"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=primary,
            counterfactual_probes=[],
            observations=[],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )


def test_replay_rejects_primary_evidence_without_an_actual_result_trace() -> None:
    observation = ForensicObservation(
        code=ForensicCode.RERANKED_DOWN,
        statement="The primary replay showed a lower final rank after reranking.",
        origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
        observed_at=_NOW,
        trace_id=_TRACE_ID,
        evidence=[_evidence()],
        certainty=EvidenceCertainty.OBSERVED,
    )

    with pytest.raises(ValidationError, match="bind to a primary result trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[],
            observations=[observation],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    actual_trace = _primary().results[0].trace_id
    wrong_time = ForensicObservation(
        code=ForensicCode.RERANKED_DOWN,
        statement="The primary replay showed a lower final rank after reranking.",
        origin=EvidenceOrigin.LIVE_REPLAY_PRIMARY,
        observed_at=_LATER,
        trace_id=actual_trace,
        evidence=[_evidence(trace_id=actual_trace, observed_at=_LATER)],
        certainty=EvidenceCertainty.OBSERVED,
    )
    with pytest.raises(ValidationError, match="bind to a primary result trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[],
            observations=[wrong_time],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )


def test_counterfactual_probe_rejects_rank_beyond_candidate_count() -> None:
    with pytest.raises(ValidationError, match="fit its positive candidate count"):
        ReplayCounterfactualProbe(
            config_id=_CONFIG_IDS[0],
            observed_at=_NOW,
            trace_id=_TRACE_ID,
            duration_ms=1.0,
            bm25_candidate_count=0,
            vector_candidate_count=0,
            candidates=[
                ReplayProbeCandidate(
                    document_id=UUID(int=400),
                    stage_membership=[ProbeStageMembership(stage="bm25_candidates", rank=100)],
                )
            ],
            warnings=[],
        )


def test_counterfactual_observation_binds_exact_probe_trace_time_and_count() -> None:
    probe = _probe()
    rank = EvidenceItem(
        label="bm25_rank",
        value=RankEvidenceValue(stage="bm25_candidates", rank=1),
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=_NOW,
        trace_id=_TRACE_ID,
    )
    observation = ForensicObservation(
        code=ForensicCode.OUTSIDE_FUSION_TOP_K,
        statement="The separate probe observed one bounded BM25 candidate rank.",
        origin=EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
        observed_at=_NOW,
        trace_id=_TRACE_ID,
        evidence=[rank],
        certainty=EvidenceCertainty.COUNTERFACTUAL,
    )

    response = EvalRunQueryReplayResponse(
        run_id=UUID(int=300),
        query_id=_QUERY_ID,
        config_ids=_CONFIG_IDS,
        primary_observed_at=_NOW,
        primary=_primary(),
        counterfactual_probes=[probe],
        observations=[observation],
        observability_notice="Original stage evidence is unavailable; this is a live replay.",
    )

    assert response.observations[0].trace_id == probe.trace_id

    impossible_count = rank.model_copy(
        update={
            "label": "bm25_count",
            "value": CandidateCountEvidenceValue(stage="bm25_candidates", count=0),
        }
    )
    with pytest.raises(ValidationError, match="must equal its probe count"):
        EvalRunQueryReplayResponse(
            run_id=response.run_id,
            query_id=response.query_id,
            config_ids=response.config_ids,
            primary_observed_at=response.primary_observed_at,
            primary=response.primary,
            counterfactual_probes=response.counterfactual_probes,
            observations=[observation.model_copy(update={"evidence": [impossible_count]})],
            observability_notice=response.observability_notice,
        )


def test_client_computed_origin_cannot_be_supplied_without_source_trace() -> None:
    with pytest.raises(ValidationError, match="require the exact source trace"):
        EvidenceItem(
            label="rrf_contribution",
            value=RrfContributionEvidenceValue(
                stage="vector_candidates",
                rank=2,
                weight=1.0,
                rank_constant=60,
                contribution=1.0 / 62.0,
            ),
            origin=EvidenceOrigin.CLIENT_COMPUTED,
            observed_at=_NOW,
            trace_id=None,
        )


def test_client_computed_primary_source_requires_exact_time_for_observation_and_evidence() -> None:
    primary = _primary()
    actual_trace = primary.results[0].trace_id
    wrong_observation_time = ForensicObservation(
        code=ForensicCode.RERANKED_DOWN,
        statement="A client computation referenced the primary replay.",
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=_LATER,
        trace_id=actual_trace,
        evidence=[],
        certainty=EvidenceCertainty.OBSERVED,
    )

    with pytest.raises(ValidationError, match="primary source trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=primary,
            counterfactual_probes=[],
            observations=[wrong_observation_time],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    wrong_evidence_time = _evidence(
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        trace_id=actual_trace,
        observed_at=_LATER,
    )
    right_observation_time = wrong_observation_time.model_copy(
        update={"observed_at": _NOW, "evidence": [wrong_evidence_time]}
    )
    with pytest.raises(ValidationError, match="primary source trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=primary,
            counterfactual_probes=[],
            observations=[right_observation_time],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )


def test_client_computed_probe_source_requires_exact_time_and_rank_bounds() -> None:
    probe = _probe()
    impossible_contribution = EvidenceItem(
        label="rrf_contribution",
        value=RrfContributionEvidenceValue(
            stage="bm25_candidates",
            rank=100,
            weight=1.0,
            rank_constant=60,
            contribution=1.0 / 160.0,
        ),
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=probe.observed_at,
        trace_id=probe.trace_id,
    )
    observation = ForensicObservation(
        code=ForensicCode.OUTSIDE_FUSION_TOP_K,
        statement="A client computation used the separately returned probe.",
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=probe.observed_at,
        trace_id=probe.trace_id,
        evidence=[impossible_contribution],
        certainty=EvidenceCertainty.COUNTERFACTUAL,
    )

    with pytest.raises(ValidationError, match="fit its positive probe count"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[probe],
            observations=[observation],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    impossible_rank = impossible_contribution.model_copy(
        update={
            "label": "bm25_rank",
            "value": RankEvidenceValue(stage="bm25_candidates", rank=100),
        }
    )
    with pytest.raises(ValidationError, match="fit its positive probe count"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[probe],
            observations=[observation.model_copy(update={"evidence": [impossible_rank]})],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    impossible_membership = impossible_contribution.model_copy(
        update={
            "label": "vector_membership",
            "value": PresenceEvidenceValue(stage="vector_candidates", present=True),
        }
    )
    with pytest.raises(
        ValidationError, match="membership evidence requires a positive probe count"
    ):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[probe],
            observations=[observation.model_copy(update={"evidence": [impossible_membership]})],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    wrong_time = observation.model_copy(update={"observed_at": _LATER, "evidence": []})
    with pytest.raises(ValidationError, match="probe source trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[probe],
            observations=[wrong_time],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )

    wrong_evidence_time = impossible_contribution.model_copy(
        update={
            "observed_at": _LATER,
            "value": RankEvidenceValue(stage="bm25_candidates", rank=1),
        }
    )
    with pytest.raises(ValidationError, match="probe source trace/time"):
        EvalRunQueryReplayResponse(
            run_id=UUID(int=300),
            query_id=_QUERY_ID,
            config_ids=_CONFIG_IDS,
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[probe],
            observations=[observation.model_copy(update={"evidence": [wrong_evidence_time]})],
            observability_notice="Original stage evidence is unavailable; this is a live replay.",
        )


def test_client_computed_primary_rank_must_exist_in_returned_hits() -> None:
    primary = _primary_with_final_hit()
    actual_trace = primary.results[0].trace_id
    actual_rank = EvidenceItem(
        label="final_rank",
        value=RankEvidenceValue(stage=RetrievalStage.FINAL, rank=1),
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=_NOW,
        trace_id=actual_trace,
    )
    observation = ForensicObservation(
        code=ForensicCode.RERANKED_DOWN,
        statement="A client computation used the final rank returned by the primary replay.",
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=_NOW,
        trace_id=actual_trace,
        evidence=[actual_rank],
        certainty=EvidenceCertainty.OBSERVED,
    )
    response = EvalRunQueryReplayResponse(
        run_id=UUID(int=300),
        query_id=_QUERY_ID,
        config_ids=_CONFIG_IDS,
        primary_observed_at=_NOW,
        primary=primary,
        counterfactual_probes=[],
        observations=[observation],
        observability_notice="Original stage evidence is unavailable; this is a live replay.",
    )

    assert response.observations[0].evidence[0].value == actual_rank.value

    fabricated_rank = actual_rank.model_copy(
        update={"value": RankEvidenceValue(stage=RetrievalStage.FINAL, rank=2)}
    )
    with pytest.raises(ValidationError, match="actual returned hit membership/rank"):
        EvalRunQueryReplayResponse(
            run_id=response.run_id,
            query_id=response.query_id,
            config_ids=response.config_ids,
            primary_observed_at=response.primary_observed_at,
            primary=response.primary,
            counterfactual_probes=[],
            observations=[observation.model_copy(update={"evidence": [fabricated_rank]})],
            observability_notice=response.observability_notice,
        )
