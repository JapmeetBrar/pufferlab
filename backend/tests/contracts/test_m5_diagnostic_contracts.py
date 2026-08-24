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
                    stored_filter_result=None,
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
                    stored_filter_result=None,
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
        stored_filter_result=None,
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


def _filter_item(
    result: DiagnosticPredicateResult,
    *,
    ordinal: int = 0,
) -> FilterPredicateEvidence:
    return FilterPredicateEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        predicate_ordinal=ordinal,
        predicate_path=(0, ordinal),
        field=f"field_{ordinal}",
        operator="eq",
        result=result,
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if result is DiagnosticPredicateResult.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )


def _filter_observation(
    result: DiagnosticPredicateResult,
    *,
    ordinal: int = 0,
) -> ForensicObservation:
    code = (
        ForensicCode.FILTER_PREDICATE_FAILED
        if result is DiagnosticPredicateResult.NOT_MATCHED
        else ForensicCode.NOT_OBSERVABLE
    )
    item = _filter_item(result, ordinal=ordinal)
    return ForensicObservation(
        config_id=_CONFIG,
        document_id=_TARGET,
        code=code,
        statement={
            ForensicCode.FILTER_PREDICATE_FAILED: (
                "The selected target did not match a stored-query filter predicate."
            ),
            ForensicCode.NOT_OBSERVABLE: (
                "The selected target's exclusion is not observable from this diagnostic."
            ),
        }[code],
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=_NOW,
        trace_id=_TRACE,
        evidence=[
            EvidenceItem(
                label=f"filter_predicate_{ordinal}",
                value=FilterPredicateEvidenceValue(
                    predicate_ordinal=ordinal,
                    predicate_path=item.predicate_path,
                    field=item.field,
                    operator=item.operator,
                    result=result,
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=_NOW,
                trace_id=_TRACE,
            )
        ],
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if code is ForensicCode.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )


def _cutoff_observation(
    *,
    scope: DiagnosticCandidateScope,
    signal: DiagnosticSignal,
    relation: DiagnosticCutoffRelation,
    code: ForensicCode,
    certainty: EvidenceCertainty,
) -> ForensicObservation:
    return ForensicObservation(
        config_id=_CONFIG,
        document_id=_TARGET,
        code=code,
        statement={
            ForensicCode.NO_LEXICAL_SCORE: (
                "The selected target had no positive lexical score in this diagnostic."
            ),
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES: (
                "The selected target scored outside the lexical candidate boundary."
            ),
            ForensicCode.OUTSIDE_VECTOR_CANDIDATES: (
                "The selected target scored outside the vector candidate boundary."
            ),
            ForensicCode.OUTSIDE_FUSION_TOP_K: (
                "The selected target scored outside the qualified client-computed fusion boundary."
            ),
            ForensicCode.NOT_OBSERVABLE: (
                "The selected target's exclusion is not observable from this diagnostic."
            ),
        }[code],
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=_NOW,
        trace_id=_TRACE,
        evidence=[
            EvidenceItem(
                label=f"cutoff_{scope.value}_{signal.value}",
                value=CutoffRelationEvidenceValue(
                    scope=scope,
                    signal=signal,
                    relation=relation,
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=_NOW,
                trace_id=_TRACE,
            )
        ],
        certainty=certainty,
    )


def _bind_filter_root(
    payload: dict[str, Any],
    root: DiagnosticPredicateResult,
    leaves: list[DiagnosticPredicateResult],
) -> None:
    payload["stored_filter_result"] = root
    payload["filter_evidence"] = [
        _filter_item(result, ordinal=ordinal).model_dump(mode="json")
        for ordinal, result in enumerate(leaves)
    ]
    for evidence in payload["candidate_evidence"]:
        evidence["stored_filter_result"] = (
            root if evidence["scope"] == DiagnosticCandidateScope.STORED_QUERY else None
        )
    for evidence in payload["qualified_rrf_evidence"]:
        evidence["stored_filter_result"] = (
            root if evidence["scope"] == DiagnosticCandidateScope.STORED_QUERY else None
        )


def _make_bm25_stored_target_absent(payload: dict[str, Any]) -> None:
    payload["subqueries"][1].update(
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=None,
    )
    payload["candidate_evidence"][0].update(
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )


def _make_rrf_scope_target_absent_not_observable(
    payload: dict[str, Any],
    scope: DiagnosticCandidateScope,
) -> None:
    roles = {
        DiagnosticCandidateScope.STORED_QUERY: {
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        },
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL: {
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
        },
    }[scope]
    for summary in payload["subqueries"]:
        if summary["role"] in roles:
            summary.update(
                returned_count=0,
                target_present=False,
                target_rank=None,
                target_score=None,
                boundary_score=None,
            )
    for candidate in payload["candidate_evidence"]:
        if candidate["scope"] == scope:
            candidate.update(
                returned_count=0,
                target_present=False,
                target_rank=None,
                target_score=None,
                boundary_score=None,
                relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
                certainty=EvidenceCertainty.INSUFFICIENT,
            )
    rrf = next(item for item in payload["qualified_rrf_evidence"] if item["scope"] == scope)
    rrf.update(
        bm25_rank=None,
        ann_rank=None,
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False).model_dump(mode="json"),
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
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
        stored_filter_result=None,
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


@pytest.mark.parametrize(
    ("signal", "direct_value", "boundary_value", "expected_relation"),
    [
        (
            DiagnosticSignal.BM25,
            2.0,
            1.0,
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            DiagnosticSignal.BM25,
            1.0,
            1.0,
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            DiagnosticSignal.BM25,
            0.5,
            1.0,
            DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        ),
        (
            DiagnosticSignal.ANN,
            0.5,
            1.0,
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            DiagnosticSignal.ANN,
            1.0,
            1.0,
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            DiagnosticSignal.ANN,
            2.0,
            1.0,
            DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        ),
    ],
)
@pytest.mark.parametrize(
    "stored_filter_result",
    [
        DiagnosticPredicateResult.NOT_MATCHED,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
    ],
)
def test_filter_ineligible_absent_full_candidate_preserves_independent_base_facts(
    signal: DiagnosticSignal,
    direct_value: float,
    boundary_value: float,
    expected_relation: DiagnosticCutoffRelation,
    stored_filter_result: DiagnosticPredicateResult,
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if signal is DiagnosticSignal.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    evidence = CandidateCutoffEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        subquery_ordinal=1,
        role=role,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=stored_filter_result,
        signal=signal,
        requested_limit=50,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        direct_score=_score(signal, direct_value, direct=True),
        boundary_score=_score(signal, boundary_value, direct=False),
        relation=expected_relation,
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if expected_relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )

    assert evidence.relation is expected_relation
    false_relation = (
        DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
        if expected_relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
        else DiagnosticCutoffRelation.NOT_OBSERVABLE
    )
    with pytest.raises(ValidationError, match="must equal"):
        CandidateCutoffEvidence.model_validate(
            {
                **evidence.model_dump(),
                "relation": false_relation,
                "certainty": (
                    EvidenceCertainty.OBSERVED
                    if false_relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
                    else EvidenceCertainty.INSUFFICIENT
                ),
            }
        )


@pytest.mark.parametrize(
    "stored_filter_result",
    [
        DiagnosticPredicateResult.NOT_MATCHED,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
    ],
)
def test_filter_ineligible_zero_bm25_preserves_no_lexical_score(
    stored_filter_result: DiagnosticPredicateResult,
) -> None:
    evidence = CandidateCutoffEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        subquery_ordinal=1,
        role=DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=stored_filter_result,
        signal=DiagnosticSignal.BM25,
        requested_limit=50,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        direct_score=_score(DiagnosticSignal.BM25, 0.0, direct=True),
        boundary_score=_score(DiagnosticSignal.BM25, 1.0, direct=False),
        relation=DiagnosticCutoffRelation.NO_LEXICAL_SCORE,
        certainty=EvidenceCertainty.OBSERVED,
    )

    assert evidence.relation is DiagnosticCutoffRelation.NO_LEXICAL_SCORE


@pytest.mark.parametrize("signal", [DiagnosticSignal.BM25, DiagnosticSignal.ANN])
@pytest.mark.parametrize(
    "stored_filter_result",
    [
        DiagnosticPredicateResult.NOT_MATCHED,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
    ],
)
def test_filter_ineligible_absent_short_candidate_is_not_observable(
    signal: DiagnosticSignal,
    stored_filter_result: DiagnosticPredicateResult,
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if signal is DiagnosticSignal.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    evidence = CandidateCutoffEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        subquery_ordinal=1,
        role=role,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=stored_filter_result,
        signal=signal,
        requested_limit=50,
        returned_count=49,
        target_present=False,
        target_rank=None,
        target_score=None,
        direct_score=_score(signal, 3.0 if signal is DiagnosticSignal.BM25 else 0.2, direct=True),
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )

    assert evidence.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE


@pytest.mark.parametrize("signal", [DiagnosticSignal.BM25, DiagnosticSignal.ANN])
def test_filter_result_controls_present_stored_candidate_consistency(
    signal: DiagnosticSignal,
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if signal is DiagnosticSignal.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    direct_value = 3.0 if signal is DiagnosticSignal.BM25 else 0.2
    payload = {
        "config_id": _CONFIG,
        "target_document_id": _TARGET,
        "observed_at": _NOW,
        "trace_id": _TRACE,
        "subquery_ordinal": 1,
        "role": role,
        "scope": DiagnosticCandidateScope.STORED_QUERY,
        "stored_filter_result": DiagnosticPredicateResult.NOT_OBSERVABLE,
        "signal": signal,
        "requested_limit": 50,
        "returned_count": 1,
        "target_present": True,
        "target_rank": 1,
        "target_score": _score(signal, direct_value, direct=False),
        "direct_score": _score(signal, direct_value, direct=True),
        "boundary_score": None,
        "relation": DiagnosticCutoffRelation.TARGET_PRESENT,
        "certainty": EvidenceCertainty.OBSERVED,
    }
    accepted = CandidateCutoffEvidence.model_validate(payload)
    assert accepted.relation is DiagnosticCutoffRelation.TARGET_PRESENT

    payload["stored_filter_result"] = DiagnosticPredicateResult.NOT_MATCHED
    with pytest.raises(ValidationError, match="filter-ineligible"):
        CandidateCutoffEvidence.model_validate(payload)


def test_matched_and_no_filter_candidates_keep_the_original_direction_derivation() -> None:
    base = {
        "config_id": _CONFIG,
        "target_document_id": _TARGET,
        "observed_at": _NOW,
        "trace_id": _TRACE,
        "subquery_ordinal": 1,
        "requested_limit": 50,
        "returned_count": 50,
        "target_present": False,
        "target_rank": None,
        "target_score": None,
        "boundary_score": _score(DiagnosticSignal.BM25, 1.0, direct=False),
        "relation": DiagnosticCutoffRelation.NOT_OBSERVABLE,
        "certainty": EvidenceCertainty.INSUFFICIENT,
    }
    for stored_filter_result in (DiagnosticPredicateResult.MATCHED, None):
        with pytest.raises(ValidationError, match="clearly above"):
            CandidateCutoffEvidence.model_validate(
                {
                    **base,
                    "role": DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    "scope": DiagnosticCandidateScope.STORED_QUERY,
                    "stored_filter_result": stored_filter_result,
                    "signal": DiagnosticSignal.BM25,
                    "direct_score": _score(DiagnosticSignal.BM25, 2.0, direct=True),
                }
            )

    counterfactual = CandidateCutoffEvidence.model_validate(
        {
            **base,
            "role": DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
            "scope": DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
            "stored_filter_result": None,
            "signal": DiagnosticSignal.ANN,
            "direct_score": _score(DiagnosticSignal.ANN, 0.5, direct=True),
            "boundary_score": _score(DiagnosticSignal.ANN, 1.0, direct=False),
            "relation": DiagnosticCutoffRelation.ANN_CANDIDATE_MISS,
            "certainty": EvidenceCertainty.COUNTERFACTUAL,
        }
    )
    assert counterfactual.relation is DiagnosticCutoffRelation.ANN_CANDIDATE_MISS

    with pytest.raises(ValidationError, match="cannot retain"):
        CandidateCutoffEvidence.model_validate(
            {
                **counterfactual.model_dump(),
                "stored_filter_result": DiagnosticPredicateResult.NOT_MATCHED,
            }
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
    payload["stored_filter_result"] = DiagnosticPredicateResult.MATCHED
    payload["candidate_evidence"][0]["stored_filter_result"] = DiagnosticPredicateResult.MATCHED
    payload["filter_evidence"] = [item.model_dump(mode="json"), item.model_dump(mode="json")]
    with pytest.raises(ValidationError, match="contiguous unique"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_response_requires_nullable_aggregate_filter_result_exactly_with_filter_evidence() -> None:
    payload = _response().model_dump(mode="json")
    payload.pop("stored_filter_result")
    with pytest.raises(ValidationError, match="Field required"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    for malformed in (True, 0, "unknown", {}, []):
        payload = _response().model_dump(mode="json")
        payload["stored_filter_result"] = malformed
        with pytest.raises(ValidationError):
            ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["filter_evidence"] = [_filter_item(DiagnosticPredicateResult.MATCHED).model_dump()]
    with pytest.raises(ValidationError, match="if and only if"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["stored_filter_result"] = DiagnosticPredicateResult.MATCHED
    payload["candidate_evidence"][0]["stored_filter_result"] = DiagnosticPredicateResult.MATCHED
    with pytest.raises(ValidationError, match="if and only if"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_response_binds_stored_filter_result_to_stored_candidate_and_rrf_only() -> None:
    payload = _response(
        RetrievalMode.HYBRID_RRF,
        include_no_filter=True,
    ).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.stored_filter_result is DiagnosticPredicateResult.MATCHED
    assert [item.stored_filter_result for item in response.candidate_evidence] == [
        DiagnosticPredicateResult.MATCHED,
        DiagnosticPredicateResult.MATCHED,
        None,
        None,
    ]
    assert [item.stored_filter_result for item in response.qualified_rrf_evidence] == [
        DiagnosticPredicateResult.MATCHED,
        None,
    ]

    attacked = response.model_dump(mode="json")
    attacked["candidate_evidence"][0]["stored_filter_result"] = None
    with pytest.raises(ValidationError, match="scope-bound"):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    attacked = response.model_dump(mode="json")
    attacked["qualified_rrf_evidence"][0]["stored_filter_result"] = (
        DiagnosticPredicateResult.NOT_OBSERVABLE
    )
    with pytest.raises(ValidationError, match="scope-bound"):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    attacked = response.model_dump(mode="json")
    attacked["candidate_evidence"][2]["stored_filter_result"] = DiagnosticPredicateResult.MATCHED
    with pytest.raises(ValidationError, match="cannot retain"):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    attacked = response.model_dump(mode="json")
    attacked["qualified_rrf_evidence"][1]["stored_filter_result"] = (
        DiagnosticPredicateResult.MATCHED
    )
    with pytest.raises(ValidationError, match="cannot retain"):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


@pytest.mark.parametrize(
    ("mode", "root", "boundary_value"),
    [
        (RetrievalMode.BM25, DiagnosticPredicateResult.NOT_MATCHED, 2.0),
        (RetrievalMode.BM25, DiagnosticPredicateResult.NOT_OBSERVABLE, 2.0),
        (RetrievalMode.VECTOR, DiagnosticPredicateResult.NOT_MATCHED, 0.3),
        (RetrievalMode.VECTOR, DiagnosticPredicateResult.NOT_OBSERVABLE, 0.3),
    ],
)
def test_response_accepts_scope_bound_filter_qualified_better_but_absent_candidate(
    mode: RetrievalMode,
    root: DiagnosticPredicateResult,
    boundary_value: float,
) -> None:
    payload = _response(mode).model_dump(mode="json")
    leaf = (
        DiagnosticPredicateResult.MATCHED
        if root is DiagnosticPredicateResult.NOT_MATCHED
        else DiagnosticPredicateResult.NOT_OBSERVABLE
    )
    _bind_filter_root(payload, root, [leaf])
    signal = DiagnosticSignal.BM25 if mode is RetrievalMode.BM25 else DiagnosticSignal.ANN
    boundary = _score(signal, boundary_value, direct=False).model_dump(mode="json")
    payload["subqueries"][1].update(
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=boundary,
    )
    payload["candidate_evidence"][0].update(
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=boundary,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.candidate_evidence[0].stored_filter_result is root
    assert response.candidate_evidence[0].relation is DiagnosticCutoffRelation.NOT_OBSERVABLE


def test_nested_filter_result_missing_malformed_and_constructed_attacks_reject() -> None:
    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    payload["candidate_evidence"][0].pop("stored_filter_result")
    with pytest.raises(ValidationError, match="Field required"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    payload["qualified_rrf_evidence"][0].pop("stored_filter_result")
    with pytest.raises(ValidationError, match="Field required"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    for malformed in (True, 0, "invalid", {}, []):
        payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
        payload["candidate_evidence"][0]["stored_filter_result"] = malformed
        with pytest.raises(ValidationError):
            ExpectedDocumentDiagnosticResponse.model_validate(payload)

        payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
        payload["qualified_rrf_evidence"][0]["stored_filter_result"] = malformed
        with pytest.raises(ValidationError):
            ExpectedDocumentDiagnosticResponse.model_validate(payload)

    response = _response(RetrievalMode.HYBRID_RRF)
    candidate = response.candidate_evidence[0]
    bad_candidate = CandidateCutoffEvidence.model_construct(
        **{key: value for key, value in candidate.__dict__.items() if key != "stored_filter_result"}
    )
    attacked = response.model_copy(
        update={"candidate_evidence": [bad_candidate, *response.candidate_evidence[1:]]}
    )
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)

    rrf = response.qualified_rrf_evidence[0]
    bad_rrf = QualifiedRrfEvidence.model_construct(
        **{key: value for key, value in rrf.__dict__.items() if key != "stored_filter_result"}
    )
    attacked = response.model_copy(update={"qualified_rrf_evidence": [bad_rrf]})
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


def test_filter_result_is_required_nullable_in_every_contract_schema() -> None:
    for model in (
        ExpectedDocumentDiagnosticResponse,
        CandidateCutoffEvidence,
        QualifiedRrfEvidence,
    ):
        schema = model.model_json_schema()
        assert "stored_filter_result" in schema["required"]
        field_schema = schema["properties"]["stored_filter_result"]
        assert any(item.get("type") == "null" for item in field_schema["anyOf"])


@pytest.mark.parametrize(
    ("root", "leaves", "valid_result", "invalid_result"),
    [
        (
            DiagnosticPredicateResult.MATCHED,
            [DiagnosticPredicateResult.MATCHED, DiagnosticPredicateResult.NOT_MATCHED],
            None,
            DiagnosticPredicateResult.NOT_MATCHED,
        ),
        (
            DiagnosticPredicateResult.NOT_MATCHED,
            [DiagnosticPredicateResult.MATCHED, DiagnosticPredicateResult.NOT_MATCHED],
            DiagnosticPredicateResult.NOT_MATCHED,
            DiagnosticPredicateResult.MATCHED,
        ),
        (
            DiagnosticPredicateResult.NOT_OBSERVABLE,
            [DiagnosticPredicateResult.NOT_MATCHED, DiagnosticPredicateResult.NOT_OBSERVABLE],
            DiagnosticPredicateResult.NOT_OBSERVABLE,
            DiagnosticPredicateResult.NOT_MATCHED,
        ),
    ],
)
def test_filter_observations_must_be_exact_root_aligned_witness_leaves(
    root: DiagnosticPredicateResult,
    leaves: list[DiagnosticPredicateResult],
    valid_result: DiagnosticPredicateResult | None,
    invalid_result: DiagnosticPredicateResult,
) -> None:
    payload = _response().model_dump(mode="json")
    _bind_filter_root(payload, root, leaves)
    if root is DiagnosticPredicateResult.NOT_MATCHED:
        _make_bm25_stored_target_absent(payload)
    if valid_result is not None:
        valid_ordinal = leaves.index(valid_result)
        payload["observations"] = [
            _filter_observation(valid_result, ordinal=valid_ordinal).model_dump(mode="json")
        ]
    ExpectedDocumentDiagnosticResponse.model_validate(payload)

    invalid_ordinal = leaves.index(invalid_result)
    payload["observations"] = [
        _filter_observation(invalid_result, ordinal=invalid_ordinal).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="aggregate stored-filter result"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_not_true_false_root_is_evidence_only_without_false_atomic_finding() -> None:
    payload = _response().model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    _make_bm25_stored_target_absent(payload)
    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations == []

    payload["observations"] = [
        _filter_observation(DiagnosticPredicateResult.MATCHED).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="aggregate stored-filter result"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


def test_false_filter_suppresses_stored_candidate_not_observable_observation() -> None:
    payload = _response().model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    _make_bm25_stored_target_absent(payload)
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.STORED_QUERY,
            signal=DiagnosticSignal.BM25,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            code=ForensicCode.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        ).model_dump(mode="json")
    ]

    with pytest.raises(ValidationError, match="known filter failure suppresses"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "signal"),
    [
        (RetrievalMode.BM25, DiagnosticSignal.BM25),
        (RetrievalMode.HYBRID_RRF, DiagnosticSignal.RRF),
    ],
)
def test_false_filter_rejects_stored_not_observable_cutoff_padded_into_filter_failure(
    mode: RetrievalMode,
    signal: DiagnosticSignal,
) -> None:
    payload = _response(mode).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.NOT_MATCHED],
    )
    if signal is DiagnosticSignal.BM25:
        _make_bm25_stored_target_absent(payload)
    else:
        _make_rrf_scope_target_absent_not_observable(
            payload,
            DiagnosticCandidateScope.STORED_QUERY,
        )
    observation = _filter_observation(DiagnosticPredicateResult.NOT_MATCHED).model_dump(mode="json")
    cutoff = _cutoff_observation(
        scope=DiagnosticCandidateScope.STORED_QUERY,
        signal=signal,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        code=ForensicCode.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    ).model_dump(mode="json")
    observation["evidence"].append(cutoff["evidence"][0])
    payload["observations"] = [observation]

    with pytest.raises(ValidationError, match="known filter failure suppresses"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "signal"),
    [
        (RetrievalMode.BM25, DiagnosticSignal.BM25),
        (RetrievalMode.HYBRID_RRF, DiagnosticSignal.RRF),
    ],
)
def test_unknown_filter_allows_stored_not_observable_cutoff_with_root_witness(
    mode: RetrievalMode,
    signal: DiagnosticSignal,
) -> None:
    payload = _response(mode).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
        [DiagnosticPredicateResult.NOT_OBSERVABLE],
    )
    if signal is DiagnosticSignal.BM25:
        _make_bm25_stored_target_absent(payload)
    else:
        _make_rrf_scope_target_absent_not_observable(
            payload,
            DiagnosticCandidateScope.STORED_QUERY,
        )
    observation = _filter_observation(DiagnosticPredicateResult.NOT_OBSERVABLE).model_dump(
        mode="json"
    )
    cutoff = _cutoff_observation(
        scope=DiagnosticCandidateScope.STORED_QUERY,
        signal=signal,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        code=ForensicCode.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    ).model_dump(mode="json")
    observation["evidence"].append(cutoff["evidence"][0])
    payload["observations"] = [observation]

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE
    assert len(response.observations[0].evidence) == 2


def test_unknown_filter_permits_stored_candidate_not_observable_observation() -> None:
    payload = _response().model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
        [DiagnosticPredicateResult.NOT_OBSERVABLE],
    )
    _make_bm25_stored_target_absent(payload)
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.STORED_QUERY,
            signal=DiagnosticSignal.BM25,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            code=ForensicCode.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        ).model_dump(mode="json")
    ]

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE


def test_false_filter_does_not_suppress_counterfactual_cutoff_uncertainty() -> None:
    payload = _response(include_no_filter=True).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    _make_bm25_stored_target_absent(payload)
    payload["subqueries"][2].update(
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=None,
    )
    payload["candidate_evidence"][1].update(
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
            signal=DiagnosticSignal.BM25,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            code=ForensicCode.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        ).model_dump(mode="json")
    ]

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE


def test_false_filter_does_not_suppress_counterfactual_rrf_uncertainty() -> None:
    payload = _response(RetrievalMode.HYBRID_RRF, include_no_filter=True).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.NOT_MATCHED],
    )
    _make_rrf_scope_target_absent_not_observable(
        payload,
        DiagnosticCandidateScope.STORED_QUERY,
    )
    _make_rrf_scope_target_absent_not_observable(
        payload,
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
    )
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
            signal=DiagnosticSignal.RRF,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            code=ForensicCode.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        ).model_dump(mode="json")
    ]

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE


def test_false_filter_suppresses_stored_rrf_not_observable_observation() -> None:
    payload = _response(RetrievalMode.HYBRID_RRF).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    for summary in payload["subqueries"][1:]:
        summary.update(
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
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False).model_dump(mode="json"),
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.STORED_QUERY,
            signal=DiagnosticSignal.RRF,
            relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
            code=ForensicCode.NOT_OBSERVABLE,
            certainty=EvidenceCertainty.INSUFFICIENT,
        ).model_dump(mode="json")
    ]

    with pytest.raises(ValidationError, match="known filter failure suppresses"):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)


@pytest.mark.parametrize("case", ["zero", "outside"])
def test_false_filter_preserves_independent_bm25_global_observations(case: str) -> None:
    payload = _response().model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.NOT_MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    direct = 0.0 if case == "zero" else 3.0
    relation = (
        DiagnosticCutoffRelation.NO_LEXICAL_SCORE
        if case == "zero"
        else DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
    )
    code = (
        ForensicCode.NO_LEXICAL_SCORE if case == "zero" else ForensicCode.OUTSIDE_LEXICAL_CANDIDATES
    )
    payload["target"]["bm25_score"] = _score(
        DiagnosticSignal.BM25,
        direct,
        direct=True,
    ).model_dump(mode="json")
    payload["subqueries"][1].update(
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        boundary_score=_score(DiagnosticSignal.BM25, 4.0, direct=False).model_dump(mode="json"),
    )
    payload["candidate_evidence"][0].update(
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=None,
        direct_score=_score(DiagnosticSignal.BM25, direct, direct=True).model_dump(mode="json"),
        boundary_score=_score(DiagnosticSignal.BM25, 4.0, direct=False).model_dump(mode="json"),
        relation=relation,
        certainty=EvidenceCertainty.OBSERVED,
    )
    payload["observations"] = [
        _cutoff_observation(
            scope=DiagnosticCandidateScope.STORED_QUERY,
            signal=DiagnosticSignal.BM25,
            relation=relation,
            code=code,
            certainty=EvidenceCertainty.OBSERVED,
        ).model_dump(mode="json")
    ]

    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    assert response.observations[0].code is code


def test_qualified_rrf_score_is_recomputed_and_ties_are_not_observable() -> None:
    source = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=None,
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
        stored_filter_result=None,
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


@pytest.mark.parametrize(
    ("boundary_value", "expected_relation"),
    [
        (0.01, DiagnosticCutoffRelation.OUTSIDE_CANDIDATES),
        (5e-16, DiagnosticCutoffRelation.NOT_OBSERVABLE),
    ],
)
def test_filter_false_absent_full_rrf_preserves_zero_score_arithmetic(
    boundary_value: float,
    expected_relation: DiagnosticCutoffRelation,
) -> None:
    evidence = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=DiagnosticPredicateResult.NOT_MATCHED,
        bm25_rank=None,
        ann_rank=None,
        bm25_weight=1.0,
        ann_weight=1.0,
        rank_constant=60,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False),
        boundary_score=_score(DiagnosticSignal.RRF, boundary_value, direct=False),
        relation=expected_relation,
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if expected_relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )

    assert evidence.relation is expected_relation


@pytest.mark.parametrize(
    ("boundary_value", "expected_relation"),
    [
        (1 / 61, DiagnosticCutoffRelation.NOT_OBSERVABLE),
        (0.02, DiagnosticCutoffRelation.OUTSIDE_CANDIDATES),
    ],
)
def test_filter_unknown_absent_full_rrf_preserves_existing_arithmetic(
    boundary_value: float,
    expected_relation: DiagnosticCutoffRelation,
) -> None:
    evidence = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=DiagnosticPredicateResult.NOT_OBSERVABLE,
        bm25_rank=1,
        ann_rank=None,
        bm25_weight=1.0,
        ann_weight=1.0,
        rank_constant=60,
        returned_count=50,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 1 / 61, direct=False),
        boundary_score=_score(DiagnosticSignal.RRF, boundary_value, direct=False),
        relation=expected_relation,
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if expected_relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )

    assert evidence.relation is expected_relation

    if expected_relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES:
        with pytest.raises(ValidationError, match="clearly above"):
            QualifiedRrfEvidence.model_validate(
                {
                    **evidence.model_dump(),
                    "boundary_score": _score(DiagnosticSignal.RRF, 0.01, direct=False),
                    "relation": DiagnosticCutoffRelation.NOT_OBSERVABLE,
                    "certainty": EvidenceCertainty.INSUFFICIENT,
                }
            )


@pytest.mark.parametrize("target_present", [False, True])
def test_filter_false_rrf_rejects_any_stored_input_rank(target_present: bool) -> None:
    with pytest.raises(ValidationError, match="input ranks"):
        QualifiedRrfEvidence(
            config_id=_CONFIG,
            target_document_id=_TARGET,
            observed_at=_NOW,
            trace_id=_TRACE,
            scope=DiagnosticCandidateScope.STORED_QUERY,
            stored_filter_result=DiagnosticPredicateResult.NOT_MATCHED,
            bm25_rank=1,
            ann_rank=None,
            bm25_weight=1.0,
            ann_weight=1.0,
            rank_constant=60,
            returned_count=50 if not target_present else 1,
            target_present=target_present,
            target_rank=1 if target_present else None,
            target_score=_score(DiagnosticSignal.RRF, 1 / 61, direct=False),
            boundary_score=(
                _score(DiagnosticSignal.RRF, 0.02, direct=False) if not target_present else None
            ),
            relation=(
                DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
                if not target_present
                else DiagnosticCutoffRelation.TARGET_PRESENT
            ),
            certainty=EvidenceCertainty.OBSERVED,
        )


@pytest.mark.parametrize(
    "stored_filter_result",
    [
        DiagnosticPredicateResult.NOT_MATCHED,
        DiagnosticPredicateResult.NOT_OBSERVABLE,
    ],
)
def test_filter_ineligible_absent_short_rrf_is_not_observable(
    stored_filter_result: DiagnosticPredicateResult,
) -> None:
    evidence = QualifiedRrfEvidence(
        config_id=_CONFIG,
        target_document_id=_TARGET,
        observed_at=_NOW,
        trace_id=_TRACE,
        scope=DiagnosticCandidateScope.STORED_QUERY,
        stored_filter_result=stored_filter_result,
        bm25_rank=None,
        ann_rank=None,
        bm25_weight=1.0,
        ann_weight=1.0,
        rank_constant=60,
        returned_count=0,
        target_present=False,
        target_rank=None,
        target_score=_score(DiagnosticSignal.RRF, 0.0, direct=False),
        boundary_score=None,
        relation=DiagnosticCutoffRelation.NOT_OBSERVABLE,
        certainty=EvidenceCertainty.INSUFFICIENT,
    )

    assert evidence.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE


def test_filter_result_controls_present_stored_rrf_consistency() -> None:
    payload = {
        "config_id": _CONFIG,
        "target_document_id": _TARGET,
        "observed_at": _NOW,
        "trace_id": _TRACE,
        "scope": DiagnosticCandidateScope.STORED_QUERY,
        "stored_filter_result": DiagnosticPredicateResult.NOT_OBSERVABLE,
        "bm25_rank": 1,
        "ann_rank": None,
        "bm25_weight": 1.0,
        "ann_weight": 1.0,
        "rank_constant": 60,
        "returned_count": 1,
        "target_present": True,
        "target_rank": 1,
        "target_score": _score(DiagnosticSignal.RRF, 1 / 61, direct=False),
        "boundary_score": None,
        "relation": DiagnosticCutoffRelation.TARGET_PRESENT,
        "certainty": EvidenceCertainty.OBSERVED,
    }
    accepted = QualifiedRrfEvidence.model_validate(payload)
    assert accepted.relation is DiagnosticCutoffRelation.TARGET_PRESENT

    payload["stored_filter_result"] = DiagnosticPredicateResult.NOT_MATCHED
    with pytest.raises(ValidationError, match="filter-ineligible"):
        QualifiedRrfEvidence.model_validate(payload)


def test_no_filter_rrf_cannot_retain_stored_filter_result() -> None:
    source = _response(
        RetrievalMode.HYBRID_RRF,
        include_no_filter=True,
    ).qualified_rrf_evidence[1]
    assert source.scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
    assert source.stored_filter_result is None

    with pytest.raises(ValidationError, match="cannot retain"):
        QualifiedRrfEvidence.model_validate(
            {
                **source.model_dump(),
                "stored_filter_result": DiagnosticPredicateResult.MATCHED,
            }
        )


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


@pytest.mark.parametrize("malformed", [None, object(), 3, "invalid", {}])
def test_malformed_observation_containers_fail_as_validation_errors(
    malformed: object,
) -> None:
    payload = _response().model_dump(mode="json")
    payload["observations"] = malformed
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    attacked = _response().model_copy(update={"observations": malformed})
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


@pytest.mark.parametrize("malformed", [None, object(), 3, "invalid", {}])
def test_malformed_nested_evidence_containers_fail_as_validation_errors(
    malformed: object,
) -> None:
    payload = _outside_bm25_response()
    payload["observations"][0]["evidence"] = malformed
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    source = ForensicObservation.model_validate(_outside_bm25_response()["observations"][0])
    attacked_observation = source.model_copy(update={"evidence": malformed})
    attacked = _response().model_copy(update={"observations": [attacked_observation]})
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(attacked)


def test_constructed_observation_and_evidence_missing_fields_fail_as_validation_errors() -> None:
    payload = _response().model_dump(mode="json")
    payload["observations"] = [ForensicObservation.model_construct(config_id=_CONFIG)]
    with pytest.raises(ValidationError):
        ExpectedDocumentDiagnosticResponse.model_validate(payload)

    payload = _outside_bm25_response()
    payload["observations"][0]["evidence"] = [
        EvidenceItem.model_construct(label="cutoff_stored_query_bm25")
    ]
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
    payload = _response(RetrievalMode.HYBRID_RRF, include_no_filter=True).model_dump(mode="json")
    _bind_filter_root(
        payload,
        DiagnosticPredicateResult.MATCHED,
        [DiagnosticPredicateResult.MATCHED],
    )
    response = ExpectedDocumentDiagnosticResponse.model_validate(payload)
    dumped = response.model_dump_json()

    assert str(_TARGET) in dumped
    assert marker not in dumped
    assert '"stored_filter_result":"matched"' in dumped
    for forbidden in (
        "namespace",
        "query_text",
        "filter_value",
        "provider_body",
        "raw_vector",
        "query_vector",
    ):
        assert forbidden not in dumped

    for container in (
        payload,
        payload["candidate_evidence"][0],
        payload["qualified_rrf_evidence"][0],
    ):
        attacked = json.loads(json.dumps(payload))
        if container is payload:
            attacked["stored_filter_value"] = "PRIVATE_FILTER_VALUE"
        elif container is payload["candidate_evidence"][0]:
            attacked["candidate_evidence"][0]["stored_filter_value"] = "PRIVATE_FILTER_VALUE"
        else:
            attacked["qualified_rrf_evidence"][0]["stored_filter_value"] = "PRIVATE_FILTER_VALUE"
        with pytest.raises(ValidationError):
            ExpectedDocumentDiagnosticResponse.model_validate(attacked)


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
