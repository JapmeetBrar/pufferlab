from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.forensics import (
    CandidateCutoffEvidence,
    CutoffRelationEvidenceValue,
    DiagnosticCandidateScope,
    DiagnosticCandidateSubquerySummary,
    DiagnosticCutoffRelation,
    DiagnosticPredicateResult,
    DiagnosticSignal,
    DiagnosticSubqueryRole,
    DiagnosticTargetLookup,
    DiagnosticTargetLookupSubquerySummary,
    DiagnosticTargetUnavailableReason,
    DirectScoreEvidenceValue,
    EvalRunQueryReplayResponse,
    EvidenceCertainty,
    EvidenceItem,
    EvidenceOrigin,
    ExpectedDocumentDiagnosticRequest,
    ExpectedDocumentDiagnosticResponse,
    FilterPredicateEvidence,
    FilterPredicateEvidenceValue,
    ForensicCode,
    ForensicObservation,
    QualifiedRrfEvidence,
    RrfContributionEvidenceValue,
    ScoreEvidenceValue,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.contracts.search import RetrievalStage
from pydantic import ValidationError

_NOW = datetime(2026, 8, 23, 20, tzinfo=UTC)
_RUN = UUID(int=1)
_QUERY = UUID(int=2)
_CONFIG = UUID(int=3)
_TARGET = UUID(int=4)
_TRACE = UUID(int=5)
_ROOT = Path(__file__).resolve().parents[3]


def _score(signal: DiagnosticSignal, value: float, *, direct: bool) -> ObservedScore:
    return ObservedScore(
        kind={
            DiagnosticSignal.BM25: ScoreKind.BM25,
            DiagnosticSignal.ANN: ScoreKind.VECTOR_DISTANCE,
            DiagnosticSignal.RRF: ScoreKind.RRF,
        }[signal],
        value=value,
        direction=(
            ScoreDirection.LOWER_IS_BETTER
            if signal is DiagnosticSignal.ANN
            else ScoreDirection.HIGHER_IS_BETTER
        ),
        source=(
            ScoreSource.COMPUTE_ATTRIBUTE
            if direct
            else ScoreSource.CLIENT_COMPUTED
            if signal is DiagnosticSignal.RRF
            else ScoreSource.TURBOPUFFER_DIST
        ),
    )


def _roles(mode: RetrievalMode, include_no_filter: bool) -> list[DiagnosticSubqueryRole]:
    stored = {
        RetrievalMode.BM25: [DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES],
        RetrievalMode.VECTOR: [DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES],
        RetrievalMode.HYBRID_RRF: [
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ],
        RetrievalMode.HYBRID_RERANK: [
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ],
    }[mode]
    if not include_no_filter:
        return [DiagnosticSubqueryRole.TARGET_LOOKUP, *stored]
    no_filter = [
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES
        if "bm25" in role.value
        else DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES
        for role in stored
    ]
    return [DiagnosticSubqueryRole.TARGET_LOOKUP, *stored, *no_filter]


def _scope_signal(
    role: DiagnosticSubqueryRole,
) -> tuple[DiagnosticCandidateScope, DiagnosticSignal]:
    return (
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
        if role.value.startswith("no_filter")
        else DiagnosticCandidateScope.STORED_QUERY,
        DiagnosticSignal.BM25 if "bm25" in role.value else DiagnosticSignal.ANN,
    )


def _response(
    mode: RetrievalMode = RetrievalMode.BM25,
    *,
    include_no_filter: bool = False,
    target_available: bool = True,
) -> ExpectedDocumentDiagnosticResponse:
    direct = {
        DiagnosticSignal.BM25: _score(DiagnosticSignal.BM25, 3.0, direct=True),
        DiagnosticSignal.ANN: _score(DiagnosticSignal.ANN, 0.2, direct=True),
    }
    role_sequence = _roles(mode, include_no_filter)
    candidate_limit = 50 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 100
    subqueries: list[Any] = [
        DiagnosticTargetLookupSubquerySummary(
            returned_count=1 if target_available else 0,
            target_present=target_available,
        )
    ]
    candidate_evidence: list[CandidateCutoffEvidence] = []
    for ordinal, role in enumerate(role_sequence[1:], start=1):
        scope, signal = _scope_signal(role)
        target_score = (
            _score(signal, direct[signal].value, direct=False) if target_available else None
        )
        subqueries.append(
            DiagnosticCandidateSubquerySummary(
                ordinal=ordinal,
                role=role,
                requested_limit=candidate_limit,
                returned_count=1,
                target_present=target_available,
                target_rank=1 if target_available else None,
                target_score=target_score,
                boundary_score=None,
            )
        )
        if target_available:
            candidate_evidence.append(
                CandidateCutoffEvidence(
                    config_id=_CONFIG,
                    target_document_id=_TARGET,
                    observed_at=_NOW,
                    trace_id=_TRACE,
                    subquery_ordinal=ordinal,
                    role=role,
                    scope=scope,
                    signal=signal,
                    requested_limit=candidate_limit,
                    returned_count=1,
                    target_present=True,
                    target_rank=1,
                    target_score=target_score,
                    direct_score=direct[signal],
                    boundary_score=None,
                    relation=DiagnosticCutoffRelation.TARGET_PRESENT,
                    certainty=(
                        EvidenceCertainty.COUNTERFACTUAL
                        if scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
                        else EvidenceCertainty.OBSERVED
                    ),
                )
            )
    qualified_rrf: list[QualifiedRrfEvidence] = []
    if target_available and mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}:
        scopes = [DiagnosticCandidateScope.STORED_QUERY]
        if include_no_filter:
            scopes.append(DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL)
        for scope in scopes:
            qualified_rrf.append(
                QualifiedRrfEvidence(
                    config_id=_CONFIG,
                    target_document_id=_TARGET,
                    observed_at=_NOW,
                    trace_id=_TRACE,
                    scope=scope,
                    bm25_rank=1,
                    ann_rank=1,
                    bm25_weight=1.0,
                    ann_weight=1.0,
                    rank_constant=60,
                    returned_count=1,
                    target_present=True,
                    target_rank=1,
                    target_score=_score(DiagnosticSignal.RRF, 2 / 61, direct=False),
                    boundary_score=None,
                    relation=DiagnosticCutoffRelation.TARGET_PRESENT,
                    certainty=(
                        EvidenceCertainty.COUNTERFACTUAL
                        if scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
                        else EvidenceCertainty.OBSERVED
                    ),
                )
            )
    target = DiagnosticTargetLookup(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        available=target_available,
        unavailable_reason=(
            None
            if target_available
            else DiagnosticTargetUnavailableReason.TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT
        ),
        bm25_score=(
            direct[DiagnosticSignal.BM25]
            if target_available
            and mode in {RetrievalMode.BM25, RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
            else None
        ),
        vector_distance=(
            direct[DiagnosticSignal.ANN]
            if target_available
            and mode
            in {RetrievalMode.VECTOR, RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
            else None
        ),
    )
    observations = (
        []
        if target_available
        else [
            ForensicObservation(
                config_id=_CONFIG,
                document_id=_TARGET,
                code=ForensicCode.NOT_OBSERVABLE,
                statement="The selected target was unavailable in this diagnostic snapshot.",
                origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
                observed_at=_NOW,
                trace_id=_TRACE,
                evidence=[],
                certainty=EvidenceCertainty.INSUFFICIENT,
            )
        ]
    )
    return ExpectedDocumentDiagnosticResponse(
        run_id=_RUN,
        query_id=_QUERY,
        config_id=_CONFIG,
        config_mode=mode,
        target_document_id=_TARGET,
        included_no_filter_counterfactual=include_no_filter,
        observed_at=_NOW,
        trace_id=_TRACE,
        duration_ms=20.0,
        embedding_duration_ms=None if mode is RetrievalMode.BM25 else 10.0,
        subqueries=subqueries,
        target=target,
        filter_evidence=[],
        candidate_evidence=candidate_evidence,
        qualified_rrf_evidence=qualified_rrf,
        observations=observations,
    )


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("include_no_filter", [False, True])
def test_exact_mode_option_role_limit_matrix(
    mode: RetrievalMode,
    include_no_filter: bool,
) -> None:
    response = _response(mode, include_no_filter=include_no_filter)

    assert [item.role for item in response.subqueries] == _roles(mode, include_no_filter)
    assert [item.ordinal for item in response.subqueries] == list(range(len(response.subqueries)))
    expected_limit = 50 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 100
    assert all(
        item.requested_limit == expected_limit
        for item in response.subqueries[1:]
        if isinstance(item, DiagnosticCandidateSubquerySummary)
    )


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_request_rejects_coerced_contract_version(version: object) -> None:
    with pytest.raises(ValidationError, match="exact integer 1"):
        ExpectedDocumentDiagnosticRequest.model_validate(
            {"contract_version": version, "config_id": str(_CONFIG)}
        )


@pytest.mark.parametrize("option", [0, 1, "false", None])
def test_request_rejects_non_boolean_option(option: object) -> None:
    with pytest.raises(ValidationError, match="JSON boolean"):
        ExpectedDocumentDiagnosticRequest.model_validate(
            {
                "config_id": str(_CONFIG),
                "include_no_filter_counterfactual": option,
            }
        )


def test_request_contains_only_explicit_config_and_option() -> None:
    request = ExpectedDocumentDiagnosticRequest(config_id=_CONFIG)
    assert request.model_dump(mode="json") == {
        "contract_version": 1,
        "config_id": str(_CONFIG),
        "include_no_filter_counterfactual": False,
    }
    for field in ("query_text", "namespace", "region", "document_id", "filter"):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ExpectedDocumentDiagnosticRequest.model_validate(
                {"config_id": str(_CONFIG), field: "forbidden"}
            )


def test_role_order_limit_and_option_attacks_reject() -> None:
    payload = _response(RetrievalMode.BM25).model_dump(mode="json")
    payload["subqueries"] = list(reversed(payload["subqueries"]))
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response(RetrievalMode.BM25).model_dump(mode="json")
    payload["subqueries"][1]["requested_limit"] = 100
    payload["candidate_evidence"][0]["requested_limit"] = 100
    with pytest.raises(ValidationError, match="selected config mode"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response(RetrievalMode.BM25).model_dump(mode="json")
    payload["included_no_filter_counterfactual"] = True
    with pytest.raises(ValidationError, match="role sequence"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("duration_ms", True), ("duration_ms", "20"), ("embedding_duration_ms", False)],
)
def test_response_rejects_coerced_durations(field: str, value: object) -> None:
    payload = _response(RetrievalMode.VECTOR).model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_response_rejects_duration_and_embedding_mode_conflicts() -> None:
    payload = _response(RetrievalMode.VECTOR).model_dump(mode="json")
    payload["embedding_duration_ms"] = 21.0
    with pytest.raises(ValidationError, match="cannot exceed"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)
    payload = _response(RetrievalMode.BM25).model_dump(mode="json")
    payload["embedding_duration_ms"] = 1.0
    with pytest.raises(ValidationError, match="selected mode"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_candidate_presence_rank_count_and_boundary_iff_rules() -> None:
    payload = _response().model_dump(mode="json")
    candidate = payload["subqueries"][1]
    candidate["target_rank"] = None
    with pytest.raises(ValidationError, match="requires an exact rank"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["subqueries"][1]["returned_count"] = 50
    payload["candidate_evidence"][0]["returned_count"] = 50
    with pytest.raises(ValidationError, match="if and only if"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["subqueries"][1]["returned_count"] = True
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize("bad_value", [True, "3.0", float("nan"), float("inf")])
def test_nested_scores_reject_coercion_and_nonfinite_values(bad_value: object) -> None:
    payload = _response().model_dump(mode="json")
    payload["target"]["bm25_score"]["value"] = bad_value
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_direct_candidate_and_rrf_score_sources_are_exact() -> None:
    payload = _response().model_dump(mode="json")
    payload["target"]["bm25_score"]["source"] = "turbopuffer_dist"
    with pytest.raises(ValidationError, match="kind, direction, and source"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["subqueries"][1]["target_score"]["source"] = "compute_attribute"
    with pytest.raises(ValidationError, match="kind, direction, and source"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    payload["qualified_rrf_evidence"][0]["target_score"]["source"] = "reranker"
    with pytest.raises(ValidationError, match="kind, direction, and source"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_target_unavailable_is_bounded_and_suppresses_downstream_claims() -> None:
    response = _response(target_available=False)

    assert response.candidate_evidence == []
    assert response.filter_evidence == []
    assert response.qualified_rrf_evidence == []
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE

    payload = response.model_dump(mode="json")
    payload["subqueries"][1]["target_present"] = True
    payload["subqueries"][1]["target_rank"] = 1
    payload["subqueries"][1]["target_score"] = _score(
        DiagnosticSignal.BM25, 3.0, direct=False
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="cannot exist"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_candidate_tolerance_tie_is_not_observable_and_false_claim_rejects() -> None:
    evidence = CandidateCutoffEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        subquery_ordinal=1,
        role=DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        signal=DiagnosticSignal.BM25,
        requested_limit=50,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        direct_score=_score(DiagnosticSignal.BM25, 1.0 + 5e-16, direct=True),
        boundary_score=_score(DiagnosticSignal.BM25, 1.0, direct=False),
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    assert evidence.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    with pytest.raises(ValidationError, match="must equal"):
        CandidateCutoffEvidence.model_validate(
            {**evidence.model_dump(), "relation": "outside_candidates"}
        )


def test_score_ranked_bm25_rows_require_positive_scores() -> None:
    payload = _response().model_dump(mode="json")
    zero_direct = _score(DiagnosticSignal.BM25, 0.0, direct=True).model_dump(mode="json")
    zero_candidate = _score(DiagnosticSignal.BM25, 0.0, direct=False).model_dump(mode="json")
    payload["target"]["bm25_score"] = zero_direct
    payload["subqueries"][1]["target_score"] = zero_candidate
    payload["candidate_evidence"][0]["direct_score"] = zero_direct
    payload["candidate_evidence"][0]["target_score"] = zero_candidate
    with pytest.raises(ValidationError, match="strictly positive"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    summary = _response().subqueries[1]
    assert isinstance(summary, DiagnosticCandidateSubquerySummary)
    with pytest.raises(ValidationError, match="strictly positive"):
        DiagnosticCandidateSubquerySummary.model_validate(
            {
                **summary.model_dump(),
                "returned_count": 50,
                "boundary_score": zero_candidate,
            }
        )

    evidence = _response().candidate_evidence[0]
    with pytest.raises(ValidationError, match="strictly positive"):
        CandidateCutoffEvidence.model_validate(
            {
                **evidence.model_dump(),
                "returned_count": 50,
                "boundary_score": zero_candidate,
            }
        )

    payload = _response().model_dump(mode="json")
    payload["subqueries"][1].update(
        returned_count=50,
        boundary_score=zero_candidate,
    )
    with pytest.raises(ValidationError, match="strictly positive"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_candidate_evidence_requires_exact_unique_role_facts_and_source() -> None:
    payload = _response().model_dump(mode="json")
    payload["candidate_evidence"].append(payload["candidate_evidence"][0])
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["candidate_evidence"][0]["trace_id"] = str(UUID(int=99))
    with pytest.raises(ValidationError, match="exact diagnostic source"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_filter_evidence_is_unique_bounded_and_contains_no_values() -> None:
    item = FilterPredicateEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        predicate_ordinal=0,
        predicate_path=(0, 1),
        field="language",
        operator="eq",
        result=DiagnosticPredicateResult.MATCHED,
        certainty=EvidenceCertainty.OBSERVED,
    )
    assert "value" not in item.model_dump(mode="json")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FilterPredicateEvidence.model_validate({**item.model_dump(), "observed_value": "secret"})

    payload = _response().model_dump(mode="json")
    payload["filter_evidence"] = [item.model_dump(mode="json"), item.model_dump(mode="json")]
    with pytest.raises(ValidationError, match="contiguous unique"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_qualified_rrf_score_is_recomputed_and_ties_are_not_observable() -> None:
    source = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        bm25_rank=None,
        ann_rank=None,
        bm25_weight=1.0,
        ann_weight=1.0,
        rank_constant=60,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False),
        boundary_score=_score(DiagnosticSignal.RRF, 5e-16, direct=False),
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    assert source.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    payload = source.model_dump()
    payload["target_score"] = _score(DiagnosticSignal.RRF, 0.1, direct=False)
    with pytest.raises(ValidationError, match="bounded rank inputs"):
        QualifiedRrfEvidence.model_validate(payload)


def test_qualified_rrf_scopes_share_config_inputs_and_cannot_exceed_source_rows() -> None:
    payload = _response(RetrievalMode.HYBRID_RRF, include_no_filter=True).model_dump(mode="json")
    payload["qualified_rrf_evidence"][1]["bm25_weight"] = 2.0
    payload["qualified_rrf_evidence"][1]["target_score"]["value"] = 3 / 61
    with pytest.raises(ValidationError, match="one exact config input tuple"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    for candidate in payload["subqueries"][1:]:
        candidate.update(
            returned_count=0,
            target_present=False,
            target_rank=None,
            target_score=None,
            boundary_score=None,
        )
    for candidate in payload["candidate_evidence"]:
        candidate.update(
            returned_count=0,
            target_present=False,
            target_rank=None,
            target_score=None,
            boundary_score=None,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        )
    payload["qualified_rrf_evidence"][0].update(
        bm25_rank=None,
        ann_rank=None,
        returned_count=1,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False).model_dump(mode="json"),
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    with pytest.raises(ValidationError, match="input union bounds"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    for impossible_count, message in ((0, "short qualified fusion"), (2, "input union bounds")):
        payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
        payload["qualified_rrf_evidence"][0].update(
            returned_count=impossible_count,
            target_present=impossible_count != 0,
            target_rank=None if impossible_count == 0 else 1,
            relation=(
                DiagnosticCutoffRelation.NOT_OBSERVABLE
                if impossible_count == 0
                else DiagnosticCutoffRelation.TARGET_PRESENT
            ),
            certainty=(
                EvidenceCertainty.INSUFFICIENT
                if impossible_count == 0
                else EvidenceCertainty.OBSERVED
            ),
        )
        with pytest.raises(ValidationError, match=message):
            ExpectedDocumentDiagnosticResponse.model_validate(payload)

    source = _response(RetrievalMode.HYBRID_RRF).qualified_rrf_evidence[0]
    with pytest.raises(ValidationError, match="short qualified fusion"):
        QualifiedRrfEvidence.model_validate(
            {
                **source.model_dump(),
                "target_present": False,
                "target_rank": None,
                "relation": DiagnosticCutoffRelation.NOT_OBSERVABLE,
                "certainty": EvidenceCertainty.INSUFFICIENT,
            }
        )


def test_qualified_rrf_boundary_row_requires_positive_contribution() -> None:
    source = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        bm25_rank=None,
        ann_rank=None,
        bm25_weight=1.0,
        ann_weight=1.0,
        rank_constant=60,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False),
        boundary_score=_score(DiagnosticSignal.RRF, 0.1, direct=False),
        relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        certainty=EvidenceCertainty.OBSERVED,
    )
    payload = source.model_dump()
    payload["boundary_score"] = _score(DiagnosticSignal.RRF, 0.0, direct=False)
    with pytest.raises(ValidationError, match="positive contribution"):
        QualifiedRrfEvidence.model_validate(payload)


def _outside_rrf_response() -> dict[str, Any]:
    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    for candidate in payload["subqueries"][1:]:
        signal = DiagnosticSignal.BM25 if "bm25" in candidate["role"] else DiagnosticSignal.ANN
        candidate.update(
            returned_count=100,
            boundary_score=_score(
                signal,
                1.0 if signal is DiagnosticSignal.BM25 else 0.5,
                direct=False,
            ).model_dump(mode="json"),
        )
    for candidate in payload["candidate_evidence"]:
        signal = DiagnosticSignal(candidate["signal"])
        candidate.update(
            returned_count=100,
            boundary_score=_score(
                signal,
                1.0 if signal is DiagnosticSignal.BM25 else 0.5,
                direct=False,
            ).model_dump(mode="json"),
        )
    qualified = payload["qualified_rrf_evidence"][0]
    qualified.update(
        returned_count=50,
        target_present=False,
        target_rank=None,
        boundary_score=_score(DiagnosticSignal.RRF, 0.04, direct=False).model_dump(mode="json"),
        relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        certainty=EvidenceCertainty.OBSERVED,
    )
    payload["observations"] = [
        ForensicObservation(
            config_id=_CONFIG,
            document_id=_TARGET,
            code=ForensicCode.OUTSIDE_FUSION_TOP_K,
            statement=(
                "The selected target scored outside the qualified client-computed fusion boundary."
            ),
            origin=EvidenceOrigin.CLIENT_COMPUTED,
            observed_at=_NOW,
            trace_id=_TRACE,
            evidence=[
                EvidenceItem(
                    label="cutoff_stored_query_rrf",
                    value=CutoffRelationEvidenceValue(
                        scope=DiagnosticCandidateScope.STORED_QUERY,
                        signal=DiagnosticSignal.RRF,
                        relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                    ),
                    origin=EvidenceOrigin.CLIENT_COMPUTED,
                    observed_at=_NOW,
                    trace_id=_TRACE,
                ),
                EvidenceItem(
                    label="qualified_rrf_score_stored_query",
                    value=ScoreEvidenceValue(
                        stage=RetrievalStage.RRF,
                        score=_score(DiagnosticSignal.RRF, 2 / 61, direct=False),
                    ),
                    origin=EvidenceOrigin.CLIENT_COMPUTED,
                    observed_at=_NOW,
                    trace_id=_TRACE,
                ),
                *[
                    EvidenceItem(
                        label=f"stored_query_{stage.value}_rrf_contribution",
                        value=RrfContributionEvidenceValue(
                            stage=stage,
                            rank=1,
                            weight=1.0,
                            rank_constant=60,
                            contribution=1 / 61,
                        ),
                        origin=EvidenceOrigin.CLIENT_COMPUTED,
                        observed_at=_NOW,
                        trace_id=_TRACE,
                    )
                    for stage in (
                        RetrievalStage.BM25_CANDIDATES,
                        RetrievalStage.VECTOR_CANDIDATES,
                    )
                ],
            ],
            certainty=EvidenceCertainty.OBSERVED,
        ).model_dump(mode="json")
    ]
    ExpectedDocumentDiagnosticResponse.model_validate(payload)
    return payload


@pytest.mark.parametrize("attack", ["cutoff", "score", "contribution"])
def test_rrf_observation_values_cannot_be_injected_outside_qualified_source(
    attack: str,
) -> None:
    payload = _outside_rrf_response()
    values = payload["observations"][0]["evidence"]
    if attack == "cutoff":
        values[0]["value"]["scope"] = DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
    elif attack == "score":
        values[1]["value"]["score"]["value"] = 0.1
    else:
        values[2]["value"].update(rank=2, contribution=1 / 62)

    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def _outside_bm25_response(*, include_no_filter: bool = False) -> dict[str, Any]:
    payload = _response(RetrievalMode.BM25, include_no_filter=include_no_filter).model_dump(
        mode="json"
    )
    for index, candidate in enumerate(payload["subqueries"][1:]):
        candidate.update(
            returned_count=50,
            target_present=False,
            target_rank=None,
            target_score=None,
            boundary_score=_score(DiagnosticSignal.BM25, 4.0, direct=False).model_dump(mode="json"),
        )
        evidence = payload["candidate_evidence"][index]
        evidence.update(
            returned_count=50,
            target_present=False,
            target_rank=None,
            target_score=None,
            boundary_score=_score(DiagnosticSignal.BM25, 4.0, direct=False).model_dump(mode="json"),
            relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        )
    observations = []
    for evidence in payload["candidate_evidence"]:
        scope = evidence["scope"]
        certainty = (
            EvidenceCertainty.COUNTERFACTUAL
            if scope == DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
            else EvidenceCertainty.OBSERVED
        )
        observations.append(
            ForensicObservation(
                config_id=_CONFIG,
                document_id=_TARGET,
                code=ForensicCode.OUTSIDE_LEXICAL_CANDIDATES,
                statement="The selected target scored outside the lexical candidate boundary.",
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=_NOW,
                trace_id=_TRACE,
                evidence=[
                    EvidenceItem(
                        label=f"cutoff_{scope}_bm25",
                        value=CutoffRelationEvidenceValue(
                            scope=scope,
                            signal=DiagnosticSignal.BM25,
                            relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                        ),
                        origin=EvidenceOrigin.CLIENT_COMPUTED,
                        observed_at=_NOW,
                        trace_id=_TRACE,
                    )
                ],
                certainty=certainty,
            ).model_dump(mode="json")
        )
    payload["observations"] = observations
    return payload


def test_observation_code_must_match_exact_evidence_result() -> None:
    payload = _outside_bm25_response()
    payload["observations"][0]["code"] = ForensicCode.OUTSIDE_VECTOR_CANDIDATES
    payload["observations"][0]["statement"] = (
        "The selected target scored outside the vector candidate boundary."
    )
    with pytest.raises(ValidationError, match="must match"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_every_supported_code_has_one_exact_typed_trigger() -> None:
    predicate = FilterPredicateEvidenceValue(
        predicate_ordinal=0,
        predicate_path=(0,),
        field="language",
        operator="eq",
        result=DiagnosticPredicateResult.NOT_MATCHED,
    )
    cases = [
        (ForensicCode.FILTER_PREDICATE_FAILED, [predicate], []),
        (
            ForensicCode.NO_LEXICAL_SCORE,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.BM25,
                    relation=DiagnosticCutoffRelation.NO_LEXICAL_SCORE,
                )
            ],
        ),
        (
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.BM25,
                    relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                )
            ],
        ),
        (
            ForensicCode.OUTSIDE_VECTOR_CANDIDATES,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.ANN,
                    relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                )
            ],
        ),
        (
            ForensicCode.ANN_CANDIDATE_MISS,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.ANN,
                    relation=DiagnosticCutoffRelation.ANN_CANDIDATE_MISS,
                )
            ],
        ),
        (
            ForensicCode.OUTSIDE_FUSION_TOP_K,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.RRF,
                    relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
                )
            ],
        ),
        (
            ForensicCode.NOT_OBSERVABLE,
            [],
            [
                CutoffRelationEvidenceValue(
                    scope=DiagnosticCandidateScope.STORED_QUERY,
                    signal=DiagnosticSignal.BM25,
                    relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
                )
            ],
        ),
    ]
    for code, filters, cutoffs in cases:
        ExpectedDocumentDiagnosticResponse._validate_observation_code(code, filters, cutoffs)
        for foreign_code, _, _ in cases:
            if foreign_code is code:
                continue
            with pytest.raises(ValueError):
                ExpectedDocumentDiagnosticResponse._validate_observation_code(
                    foreign_code,
                    filters,
                    cutoffs,
                )


def test_stored_and_no_filter_same_code_are_distinct_but_exact_duplicates_reject() -> None:
    payload = _outside_bm25_response(include_no_filter=True)
    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert [item.certainty for item in response.observations] == [
        EvidenceCertainty.OBSERVED,
        EvidenceCertainty.COUNTERFACTUAL,
    ]

    payload["observations"][1] = payload["observations"][0]
    with pytest.raises(ValidationError, match="exact duplicates"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_reordered_evidence_cannot_hide_an_exact_duplicate_observation() -> None:
    payload = _outside_bm25_response()
    direct = EvidenceItem(
        label="direct_bm25_score",
        value=DirectScoreEvidenceValue(
            signal=DiagnosticSignal.BM25,
            score=_score(DiagnosticSignal.BM25, 3.0, direct=True),
        ),
        origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
        observed_at=_NOW,
        trace_id=_TRACE,
    ).model_dump(mode="json")
    payload["observations"][0]["evidence"].append(direct)
    duplicate = {**payload["observations"][0]}
    duplicate["evidence"] = list(reversed(payload["observations"][0]["evidence"]))
    payload["observations"].append(duplicate)
    with pytest.raises(ValidationError, match="exact duplicates"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _outside_bm25_response()
    padded = {**payload["observations"][0]}
    padded["evidence"] = [*payload["observations"][0]["evidence"], direct]
    payload["observations"].append(padded)
    with pytest.raises(ValidationError, match="exact duplicates"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_observations_reject_foreign_identity_trace_origin_and_reranker_claim() -> None:
    payload = _outside_bm25_response()
    payload["observations"][0]["document_id"] = str(UUID(int=99))
    with pytest.raises(ValidationError, match="one exact source and target"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _outside_bm25_response()
    payload["observations"][0]["origin"] = EvidenceOrigin.LIVE_REPLAY_PRIMARY
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _outside_bm25_response()
    payload["observations"][0]["code"] = ForensicCode.RERANKED_DOWN
    payload["observations"][0]["statement"] = "unsafe"
    with pytest.raises(ValidationError, match="reranker"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_nested_constructed_and_copied_instances_are_revalidated() -> None:
    response = _response()
    bad_score = response.target.bm25_score.model_copy(update={"value": "3.0"})
    bad_target = response.target.model_copy(update={"bm25_score": bad_score})
    attacked = response.model_copy(update={"target": bad_target})
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    candidate = response.subqueries[1]
    assert isinstance(candidate, DiagnosticCandidateSubquerySummary)
    bad_candidate = DiagnosticCandidateSubquerySummary.model_construct(
        **{**candidate.__dict__, "requested_limit": True}
    )
    attacked = response.model_copy(update={"subqueries": [response.subqueries[0], bad_candidate]})
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    payload = _outside_bm25_response()
    observation = ForensicObservation.model_validate(payload["observations"][0])
    bad_item = EvidenceItem.model_construct(
        **{**observation.evidence[0].__dict__, "label": "provider secret"}
    )
    bad_observation = ForensicObservation.model_construct(
        **{**observation.__dict__, "evidence": [bad_item]}
    )
    payload["observations"] = [bad_observation]
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize("missing", ["kind", "value", "direction", "source"])
def test_constructed_scores_missing_fields_fail_as_validation_errors(missing: str) -> None:
    response = _response()
    score_fields: dict[str, object] = {
        "kind": ScoreKind.BM25,
        "value": 3.0,
        "direction": ScoreDirection.HIGHER_IS_BETTER,
        "source": ScoreSource.COMPUTE_ATTRIBUTE,
    }
    score_fields.pop(missing)
    bad_score = ObservedScore.model_construct(**score_fields)
    bad_target = response.target.model_copy(update={"bm25_score": bad_score})
    attacked = response.model_copy(update={"target": bad_target})

    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


@pytest.mark.parametrize("value", ["3.0", True, object()])
def test_constructed_scores_with_malformed_values_fail_as_validation_errors(value: object) -> None:
    response = _response()
    bad_score = ObservedScore.model_construct(
        kind=ScoreKind.BM25,
        value=value,
        direction=ScoreDirection.HIGHER_IS_BETTER,
        source=ScoreSource.COMPUTE_ATTRIBUTE,
    )
    bad_target = response.target.model_copy(update={"bm25_score": bad_score})
    attacked = response.model_copy(update={"target": bad_target})

    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


@pytest.mark.parametrize(
    "value",
    [
        ScoreEvidenceValue.model_construct(
            kind="score",
            stage=RetrievalStage.RRF,
        ),
        RrfContributionEvidenceValue.model_construct(
            kind="rrf_contribution",
            rank=1,
            weight=1.0,
            rank_constant=60,
            contribution=1 / 61,
        ),
    ],
)
def test_constructed_legacy_evidence_values_fail_as_validation_errors(value: object) -> None:
    payload = _outside_bm25_response()
    payload["observations"][0]["evidence"][0]["value"] = value

    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_legacy_replay_explicitly_rejects_diagnostic_origin() -> None:
    from backend.tests.contracts.test_forensics import _primary  # type: ignore[import-not-found]

    observation = ForensicObservation(
        config_id=UUID(int=102),
        document_id=_TARGET,
        code=ForensicCode.NOT_OBSERVABLE,
        statement="bounded",
        origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
        observed_at=_NOW,
        trace_id=_TRACE,
        evidence=[],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    with pytest.raises(ValidationError, match="cannot enter a legacy replay"):
        EvalRunQueryReplayResponse(
            run_id=_RUN,
            query_id=UUID(int=101),
            config_ids=[UUID(int=102), UUID(int=103)],
            primary_observed_at=_NOW,
            primary=_primary(),
            counterfactual_probes=[],
            observations=[observation],
            observability_notice="bounded",
        )


@pytest.mark.parametrize(
    "value",
    [
        DirectScoreEvidenceValue(
            signal=DiagnosticSignal.BM25,
            score=_score(DiagnosticSignal.BM25, 1.0, direct=True),
        ),
        FilterPredicateEvidenceValue(
            predicate_ordinal=0,
            predicate_path=(0,),
            field="language",
            operator="eq",
            result=DiagnosticPredicateResult.MATCHED,
        ),
        CutoffRelationEvidenceValue(
            scope=DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
            signal=DiagnosticSignal.BM25,
            relation=DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        ),
    ],
)
@pytest.mark.parametrize("source", ["primary", "probe"])
def test_legacy_replay_rejects_every_shared_diagnostic_value(
    value: DirectScoreEvidenceValue | FilterPredicateEvidenceValue | CutoffRelationEvidenceValue,
    source: str,
) -> None:
    from backend.tests.contracts.test_forensics import (  # type: ignore[import-not-found]
        _primary,
        _probe,
    )

    primary = _primary()
    probe = _probe()
    trace_id = primary.results[0].trace_id if source == "primary" else probe.trace_id
    observation = ForensicObservation.model_validate(
        {
            "config_id": str(UUID(int=102)),
            "document_id": str(UUID(int=400)),
            "code": ForensicCode.NOT_OBSERVABLE,
            "statement": "bounded",
            "origin": EvidenceOrigin.CLIENT_COMPUTED,
            "observed_at": _NOW,
            "trace_id": str(trace_id),
            "evidence": [
                {
                    "label": "bounded_value",
                    "value": value.model_dump(mode="json"),
                    "origin": EvidenceOrigin.CLIENT_COMPUTED,
                    "observed_at": _NOW.isoformat(),
                    "trace_id": str(trace_id),
                }
            ],
            "certainty": EvidenceCertainty.INSUFFICIENT,
        }
    )
    with pytest.raises(ValidationError):
        EvalRunQueryReplayResponse(
            run_id=_RUN,
            query_id=UUID(int=101),
            config_ids=[UUID(int=102), UUID(int=103)],
            primary_observed_at=_NOW,
            primary=primary,
            counterfactual_probes=[probe] if source == "probe" else [],
            observations=[observation],
            observability_notice="bounded",
        )


@pytest.mark.parametrize(
    ("origin", "source", "certainty"),
    [
        (EvidenceOrigin.LIVE_REPLAY_PRIMARY, "primary", EvidenceCertainty.OBSERVED),
        (
            EvidenceOrigin.LIVE_REPLAY_COUNTERFACTUAL_PROBE,
            "probe",
            EvidenceCertainty.COUNTERFACTUAL,
        ),
        (EvidenceOrigin.CLIENT_COMPUTED, "primary", EvidenceCertainty.OBSERVED),
    ],
)
def test_legacy_replay_rejects_diagnostic_only_forensic_code(
    origin: EvidenceOrigin,
    source: str,
    certainty: EvidenceCertainty,
) -> None:
    from backend.tests.contracts.test_forensics import (  # type: ignore[import-not-found]
        _primary,
        _probe,
    )

    primary = _primary()
    probe = _probe()
    trace_id = primary.results[0].trace_id if source == "primary" else probe.trace_id
    observation = ForensicObservation(
        config_id=UUID(int=102),
        document_id=UUID(int=400),
        code=ForensicCode.ANN_CANDIDATE_MISS,
        statement="legacy replay must not widen",
        origin=origin,
        observed_at=_NOW,
        trace_id=trace_id,
        evidence=[],
        certainty=certainty,
    )
    with pytest.raises(ValidationError, match="diagnostic-only"):
        EvalRunQueryReplayResponse(
            run_id=_RUN,
            query_id=UUID(int=101),
            config_ids=[UUID(int=102), UUID(int=103)],
            primary_observed_at=_NOW,
            primary=primary,
            counterfactual_probes=[probe] if source == "probe" else [],
            observations=[observation],
            observability_notice="bounded",
        )


def test_public_response_is_target_scoped_and_contains_no_unrelated_document_ids() -> None:
    marker = str(UUID(int=999))
    dumped = _response(RetrievalMode.HYBRID_RRF, include_no_filter=True).model_dump_json()

    assert str(_TARGET) in dumped
    assert marker not in dumped
    for forbidden in (
        "namespace",
        "query_text",
        "filter_value",
        "provider_body",
        "raw_vector",
        "query_vector",
    ):
        assert forbidden not in dumped


def test_generated_schema_contains_only_reachable_shared_diagnostic_additions() -> None:
    schema = json.loads((_ROOT / "openapi" / "pufferlab-v1.json").read_text())
    components = schema["components"]["schemas"]

    assert "live_expected_document_diagnostic" in components["EvidenceOrigin"]["enum"]
    assert "ann_candidate_miss" in components["ForensicCode"]["enum"]
    discriminator = components["ForensicEvidenceValue"]["discriminator"]["mapping"]
    assert set(discriminator) >= {
        "diagnostic_direct_score",
        "diagnostic_filter_result",
        "diagnostic_cutoff_relation",
    }
    for unreachable in (
        "ExpectedDocumentDiagnosticRequest",
        "ExpectedDocumentDiagnosticResponse",
        "DiagnosticSubquerySummary",
        "DiagnosticTargetLookup",
        "CandidateCutoffEvidence",
        "QualifiedRrfEvidence",
    ):
        assert unreachable not in components
    assert all("/diagnostic" not in path for path in schema["paths"])
