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
    ProbeStageMembership,
    RrfContributionEvidenceValue,
    ScoreEvidenceValue,
)
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.contracts.search import (
    ConfigSearchResult,
    SearchCompareResponse,
    StageTiming,
    TimingStage,
)
from pydantic import ValidationError

_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
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
) -> EvidenceItem:
    return EvidenceItem(
        label="final_score",
        value=ScoreEvidenceValue(stage="final", score=_score()),
        origin=origin,
        observed_at=_NOW,
        trace_id=_TRACE_ID,
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
    observation = ForensicObservation(
        code=ForensicCode.NOT_OBSERVABLE,
        statement="The M2 outcome did not persist stage membership or scores.",
        origin=EvidenceOrigin.STORED_RUN,
        observed_at=_NOW,
        trace_id=None,
        evidence=[],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )

    assert observation.trace_id is None
    with pytest.raises(ValidationError, match="no original stage evidence"):
        ForensicObservation.model_validate(
            {
                **observation.model_dump(),
                "code": ForensicCode.RERANKED_DOWN,
                "certainty": EvidenceCertainty.OBSERVED,
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
