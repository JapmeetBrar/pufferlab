from __future__ import annotations

import ast
import math
import traceback
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import NoReturn
from uuid import UUID

import pytest
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.contracts.forensics import (
    DiagnosticCutoffRelation,
    DiagnosticPredicateResult,
    DiagnosticSignal,
    DiagnosticSubqueryRole,
    EvidenceCertainty,
    ExpectedDocumentDiagnosticResponse,
    ForensicCode,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.contracts.search import RetrievalStage
from pufferlab.evals import diagnostic_analysis as analysis_module
from pufferlab.evals.diagnostic_analysis import analyze_diagnostic, preflight_filter_definition
from pufferlab.evals.diagnostic_models import (
    AttributePresence,
    CandidateListInput,
    CandidateRow,
    DiagnosticAnalysisError,
    DiagnosticAnalysisErrorCode,
    DiagnosticAnalysisInput,
    DiagnosticAnalysisResult,
    DiagnosticBinding,
    FilterAnalysisInput,
    FilterDefinitionInput,
    FilterFieldSchema,
    FilterValueType,
    ObservedFilterAttribute,
    PreservedAttribute,
    RrfInputs,
    TargetLookupInput,
    TruthValue,
)

_ROOT = Path(__file__).resolve().parents[3]
_RUN = UUID(int=201)
_QUERY = UUID(int=202)
_CONFIG = UUID(int=203)
_TARGET = UUID(int=204)
_TRACE = UUID(int=205)
_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
_DEFAULT_RRF = object()


def _roles(mode: RetrievalMode, include_no_filter: bool) -> tuple[DiagnosticSubqueryRole, ...]:
    stored = {
        RetrievalMode.BM25: (DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,),
        RetrievalMode.VECTOR: (DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,),
        RetrievalMode.HYBRID_RRF: (
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ),
        RetrievalMode.HYBRID_RERANK: (
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ),
    }[mode]
    if not include_no_filter:
        return stored
    counterfactual = tuple(
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES
        if "bm25" in role.value
        else DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES
        for role in stored
    )
    return (*stored, *counterfactual)


def _signal(role: DiagnosticSubqueryRole) -> DiagnosticSignal:
    return DiagnosticSignal.BM25 if "bm25" in role.value else DiagnosticSignal.ANN


def _trivial_filter(
    *,
    right: str | None = None,
    observed: PreservedAttribute | None = None,
) -> FilterAnalysisInput:
    return FilterAnalysisInput(
        node=FilterPredicate(field="filter_field", op=PredicateOp.EQ, value=right),
        schema=(FilterFieldSchema("filter_field", FilterValueType.STRING, True),),
        attributes=(
            ObservedFilterAttribute(
                "filter_field",
                observed or PreservedAttribute(AttributePresence.MISSING),
            ),
        ),
    )


def _default_candidates(
    mode: RetrievalMode,
    include_no_filter: bool,
) -> tuple[CandidateListInput, ...]:
    limit = 50 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 100
    candidates = []
    for ordinal, role in enumerate(_roles(mode, include_no_filter), start=1):
        score = 3.0 if _signal(role) is DiagnosticSignal.BM25 else 0.2
        candidates.append(
            CandidateListInput(
                ordinal=ordinal,
                role=role,
                requested_limit=limit,
                rows=(CandidateRow(document_id=_TARGET, rank=1, score=score),),
            )
        )
    return tuple(candidates)


def _input(
    mode: RetrievalMode = RetrievalMode.BM25,
    *,
    include_no_filter: bool = False,
    target: TargetLookupInput | None = None,
    candidates: tuple[CandidateListInput, ...] | None = None,
    stored_filter: FilterAnalysisInput | None = None,
    rrf: RrfInputs | object | None = _DEFAULT_RRF,
    binding: DiagnosticBinding | None = None,
) -> DiagnosticAnalysisInput:
    vector_required = mode is not RetrievalMode.BM25
    bm25_required = mode is not RetrievalMode.VECTOR
    if rrf is _DEFAULT_RRF:
        rrf_value = (
            RrfInputs(1.0, 1.0, 60)
            if mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
            else None
        )
    else:
        rrf_value = rrf
    return DiagnosticAnalysisInput(
        binding=binding
        or DiagnosticBinding(
            config_id=_CONFIG,
            target_document_id=_TARGET,
            observed_at=_NOW,
            trace_id=_TRACE,
        ),
        mode=mode,
        include_no_filter_counterfactual=include_no_filter,
        target=target
        or TargetLookupInput(
            available=True,
            bm25_score=3.0 if bm25_required else None,
            vector_distance=0.2 if vector_required else None,
        ),
        candidate_lists=candidates or _default_candidates(mode, include_no_filter),
        stored_filter=(
            _trivial_filter() if include_no_filter and stored_filter is None else stored_filter
        ),
        rrf=rrf_value,  # type: ignore[arg-type]
    )


def _response(
    result: DiagnosticAnalysisResult,
    *,
    mode: RetrievalMode,
    include_no_filter: bool = False,
) -> ExpectedDocumentDiagnosticResponse:
    return ExpectedDocumentDiagnosticResponse(
        run_id=_RUN,
        query_id=_QUERY,
        config_id=_CONFIG,
        config_mode=mode,
        target_document_id=_TARGET,
        included_no_filter_counterfactual=include_no_filter,
        stored_filter_result=(
            None
            if result.filter_root_result is None
            else {
                "true": DiagnosticPredicateResult.MATCHED,
                "false": DiagnosticPredicateResult.NOT_MATCHED,
                "unknown": DiagnosticPredicateResult.NOT_OBSERVABLE,
            }[result.filter_root_result.value]
        ),
        observed_at=_NOW,
        trace_id=_TRACE,
        duration_ms=20.0,
        embedding_duration_ms=None if mode is RetrievalMode.BM25 else 10.0,
        subqueries=list(result.subqueries),
        target=result.target,
        filter_evidence=list(result.filter_evidence),
        candidate_evidence=list(result.candidate_evidence),
        qualified_rrf_evidence=list(result.qualified_rrf_evidence),
        observations=list(result.observations),
    )


@pytest.mark.parametrize("mode", tuple(RetrievalMode))
@pytest.mark.parametrize("include_no_filter", [False, True])
def test_all_modes_emit_exact_target_only_contract_compatible_shapes(
    mode: RetrievalMode,
    include_no_filter: bool,
) -> None:
    result = analyze_diagnostic(_input(mode, include_no_filter=include_no_filter))

    expected_roles = (DiagnosticSubqueryRole.TARGET_LOOKUP, *_roles(mode, include_no_filter))
    assert tuple(item.role for item in result.subqueries) == expected_roles
    assert tuple(item.ordinal for item in result.subqueries) == tuple(range(len(expected_roles)))
    assert len(result.candidate_evidence) == len(expected_roles) - 1
    assert len(result.qualified_rrf_evidence) == (
        (2 if include_no_filter else 1)
        if mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
        else 0
    )
    response = _response(result, mode=mode, include_no_filter=include_no_filter)
    assert response.config_mode is mode
    expected_filter_result = response.stored_filter_result
    assert all(
        item.stored_filter_result
        is (expected_filter_result if item.scope.value == "stored_query" else None)
        for item in result.candidate_evidence
    )
    assert all(
        item.stored_filter_result
        is (expected_filter_result if item.scope.value == "stored_query" else None)
        for item in result.qualified_rrf_evidence
    )


@pytest.mark.parametrize(
    "attack",
    [
        "role",
        "ordinal",
        "limit",
        "list_container",
        "rows_container",
        "duplicate_id",
        "rank_gap",
        "rank_bool",
        "nan",
        "infinity",
        "negative",
        "score_bound",
        "oversized",
        "bm25_zero",
        "bm25_order",
        "ann_order",
    ],
)
def test_candidate_integrity_attacks_fail_closed(attack: str) -> None:
    base = _input(RetrievalMode.BM25)
    candidate = base.candidate_lists[0]
    rows: object = candidate.rows
    candidate_lists: object = base.candidate_lists
    if attack == "role":
        candidate = replace(candidate, role=DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES)
    elif attack == "ordinal":
        candidate = replace(candidate, ordinal=2)
    elif attack == "limit":
        candidate = replace(candidate, requested_limit=100)
    elif attack == "list_container":
        candidate_lists = [candidate]
    elif attack == "rows_container":
        rows = list(candidate.rows)
    elif attack == "duplicate_id":
        rows = (
            CandidateRow(_TARGET, 1, 3.0),
            CandidateRow(_TARGET, 2, 2.0),
        )
    elif attack == "rank_gap":
        rows = (CandidateRow(_TARGET, 2, 3.0),)
    elif attack == "rank_bool":
        rows = (CandidateRow(_TARGET, True, 3.0),)  # type: ignore[arg-type]
    elif attack == "nan":
        rows = (CandidateRow(_TARGET, 1, float("nan")),)
    elif attack == "infinity":
        rows = (CandidateRow(_TARGET, 1, float("inf")),)
    elif attack == "negative":
        rows = (CandidateRow(_TARGET, 1, -1.0),)
    elif attack == "score_bound":
        rows = (CandidateRow(_TARGET, 1, 1_000_000_000_001.0),)
    elif attack == "oversized":
        rows = tuple(
            CandidateRow(UUID(int=1000 + index), index + 1, 100 - index) for index in range(51)
        )
    elif attack == "bm25_zero":
        rows = (CandidateRow(_TARGET, 1, 0.0),)
    elif attack == "bm25_order":
        rows = (CandidateRow(UUID(int=1), 1, 1.0), CandidateRow(_TARGET, 2, 2.0))
    else:
        base = _input(RetrievalMode.VECTOR)
        candidate = base.candidate_lists[0]
        rows = (CandidateRow(UUID(int=1), 1, 0.2), CandidateRow(_TARGET, 2, 0.1))
    if attack != "list_container":
        candidate = replace(candidate, rows=rows)  # type: ignore[arg-type]
        candidate_lists = (candidate,)
    attacked = replace(base, candidate_lists=candidate_lists)  # type: ignore[arg-type]

    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(attacked)
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_CANDIDATES


def _bm25_rows(count: int) -> tuple[CandidateRow, ...]:
    return tuple(
        CandidateRow(document_id=UUID(int=1000 + rank), rank=rank, score=float(101 - rank))
        for rank in range(1, count + 1)
    )


def _ann_rows(count: int) -> tuple[CandidateRow, ...]:
    return tuple(
        CandidateRow(document_id=UUID(int=2000 + rank), rank=rank, score=rank / 100.0)
        for rank in range(1, count + 1)
    )


@pytest.mark.parametrize(
    ("mode", "direct", "rows", "expected"),
    [
        (RetrievalMode.BM25, 0.0, (), DiagnosticCutoffRelation.NO_LEXICAL_SCORE),
        (RetrievalMode.BM25, 50.0, _bm25_rows(49), DiagnosticCutoffRelation.NOT_OBSERVABLE),
        (RetrievalMode.BM25, 50.0, _bm25_rows(50), DiagnosticCutoffRelation.OUTSIDE_CANDIDATES),
        (RetrievalMode.BM25, 51.0, _bm25_rows(50), DiagnosticCutoffRelation.NOT_OBSERVABLE),
        (RetrievalMode.VECTOR, 0.49, _ann_rows(50), DiagnosticCutoffRelation.ANN_CANDIDATE_MISS),
        (RetrievalMode.VECTOR, 0.50, _ann_rows(50), DiagnosticCutoffRelation.NOT_OBSERVABLE),
        (RetrievalMode.VECTOR, 0.51, _ann_rows(50), DiagnosticCutoffRelation.OUTSIDE_CANDIDATES),
    ],
)
def test_direction_aware_candidate_cutoff_matrix(
    mode: RetrievalMode,
    direct: float,
    rows: tuple[CandidateRow, ...],
    expected: DiagnosticCutoffRelation,
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if mode is RetrievalMode.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    target = TargetLookupInput(
        available=True,
        bm25_score=direct if mode is RetrievalMode.BM25 else None,
        vector_distance=direct if mode is RetrievalMode.VECTOR else None,
    )
    result = analyze_diagnostic(
        _input(
            mode,
            target=target,
            candidates=(CandidateListInput(1, role, 50, rows),),
        )
    )

    evidence = result.candidate_evidence[0]
    assert evidence.relation is expected
    assert (evidence.boundary_score is None) is (len(rows) < 50)
    code = {
        DiagnosticCutoffRelation.NO_LEXICAL_SCORE: ForensicCode.NO_LEXICAL_SCORE,
        DiagnosticCutoffRelation.OUTSIDE_CANDIDATES: (
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES
            if mode is RetrievalMode.BM25
            else ForensicCode.OUTSIDE_VECTOR_CANDIDATES
        ),
        DiagnosticCutoffRelation.ANN_CANDIDATE_MISS: ForensicCode.ANN_CANDIDATE_MISS,
        DiagnosticCutoffRelation.NOT_OBSERVABLE: ForensicCode.NOT_OBSERVABLE,
    }[expected]
    assert result.observations[0].code is code
    _response(result, mode=mode)


@pytest.mark.parametrize(
    ("root", "stored_filter"),
    [
        (
            "false",
            _trivial_filter(
                right="wanted",
                observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
            ),
        ),
        (
            "unknown",
            _trivial_filter(
                right=None,
                observed=PreservedAttribute(AttributePresence.PRESENT_NULL),
            ),
        ),
    ],
)
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    ("mode", "direct", "rows"),
    [
        (RetrievalMode.BM25, 52.0, _bm25_rows(50)),
        (RetrievalMode.VECTOR, 0.005, _ann_rows(50)),
    ],
)
def test_stored_filter_ineligible_absence_makes_cutoff_conservative_without_duplicate_finding(
    root: str,
    stored_filter: FilterAnalysisInput,
    full: bool,
    mode: RetrievalMode,
    direct: float,
    rows: tuple[CandidateRow, ...],
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if mode is RetrievalMode.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    selected_rows = rows if full else rows[:-1]
    target = TargetLookupInput(
        available=True,
        bm25_score=direct if mode is RetrievalMode.BM25 else None,
        vector_distance=direct if mode is RetrievalMode.VECTOR else None,
    )

    result = analyze_diagnostic(
        _input(
            mode,
            target=target,
            candidates=(CandidateListInput(1, role, 50, selected_rows),),
            stored_filter=stored_filter,
        )
    )

    assert result.filter_root_result.value == root
    assert result.candidate_evidence[0].relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    assert result.candidate_evidence[0].certainty is EvidenceCertainty.INSUFFICIENT
    assert len(result.observations) == (1 if root == "false" else 2)
    assert result.observations[0].code is (
        ForensicCode.FILTER_PREDICATE_FAILED if root == "false" else ForensicCode.NOT_OBSERVABLE
    )
    assert result.observations[0].evidence[0].label.startswith("filter_predicate_")
    if root == "unknown":
        assert result.observations[1].evidence[0].label.startswith("cutoff_stored_query_")
    _response(result, mode=mode)


@pytest.mark.parametrize(
    "stored_filter",
    [
        _trivial_filter(
            right="wanted",
            observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
        ),
        _trivial_filter(
            right=None,
            observed=PreservedAttribute(AttributePresence.PRESENT_NULL),
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "direct", "rows", "counterfactual_relation", "counterfactual_code"),
    [
        (
            RetrievalMode.BM25,
            50.0,
            _bm25_rows(50),
            DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES,
        ),
        (
            RetrievalMode.VECTOR,
            0.49,
            _ann_rows(50),
            DiagnosticCutoffRelation.ANN_CANDIDATE_MISS,
            ForensicCode.ANN_CANDIDATE_MISS,
        ),
    ],
)
def test_no_filter_counterfactual_cutoff_remains_independent_of_stored_filter_eligibility(
    stored_filter: FilterAnalysisInput,
    mode: RetrievalMode,
    direct: float,
    rows: tuple[CandidateRow, ...],
    counterfactual_relation: DiagnosticCutoffRelation,
    counterfactual_code: ForensicCode,
) -> None:
    stored_role, counterfactual_role = _roles(mode, True)
    target = TargetLookupInput(
        available=True,
        bm25_score=direct if mode is RetrievalMode.BM25 else None,
        vector_distance=direct if mode is RetrievalMode.VECTOR else None,
    )
    result = analyze_diagnostic(
        _input(
            mode,
            include_no_filter=True,
            target=target,
            candidates=(
                CandidateListInput(1, stored_role, 50, rows),
                CandidateListInput(2, counterfactual_role, 50, rows),
            ),
            stored_filter=stored_filter,
        )
    )

    stored, counterfactual = result.candidate_evidence
    expected_stored = (
        DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
        if mode is RetrievalMode.BM25
        else DiagnosticCutoffRelation.NOT_OBSERVABLE
    )
    assert stored.relation is expected_stored
    assert counterfactual.relation is counterfactual_relation
    assert counterfactual.certainty is EvidenceCertainty.COUNTERFACTUAL
    cutoff_observations = [
        item
        for item in result.observations
        if item.evidence and item.evidence[0].label.startswith("cutoff_")
    ]
    expected_cutoff_count = (
        2
        if expected_stored is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
        or result.filter_root_result is TruthValue.UNKNOWN
        else 1
    )
    assert len(cutoff_observations) == expected_cutoff_count
    assert any(
        item.code is counterfactual_code and "no_filter_counterfactual" in item.evidence[0].label
        for item in cutoff_observations
    )
    _response(result, mode=mode, include_no_filter=True)


@pytest.mark.parametrize(
    ("root", "stored_filter"),
    [
        (
            TruthValue.FALSE,
            _trivial_filter(
                right="wanted",
                observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
            ),
        ),
        (
            TruthValue.UNKNOWN,
            _trivial_filter(
                right=None,
                observed=PreservedAttribute(AttributePresence.PRESENT_NULL),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "direct", "rows", "expected"),
    [
        (RetrievalMode.BM25, 0.0, _bm25_rows(50), DiagnosticCutoffRelation.NO_LEXICAL_SCORE),
        (
            RetrievalMode.BM25,
            50.0,
            _bm25_rows(50),
            DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        ),
        (
            RetrievalMode.BM25,
            51.0,
            _bm25_rows(50),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            RetrievalMode.BM25,
            52.0,
            _bm25_rows(49),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            RetrievalMode.BM25,
            52.0,
            _bm25_rows(50),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            RetrievalMode.VECTOR,
            0.51,
            _ann_rows(50),
            DiagnosticCutoffRelation.OUTSIDE_CANDIDATES,
        ),
        (
            RetrievalMode.VECTOR,
            0.50,
            _ann_rows(50),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            RetrievalMode.VECTOR,
            0.49,
            _ann_rows(49),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
        (
            RetrievalMode.VECTOR,
            0.49,
            _ann_rows(50),
            DiagnosticCutoffRelation.NOT_OBSERVABLE,
        ),
    ],
)
def test_stored_filter_eligibility_preserves_all_noncontradictory_candidate_relations(
    root: TruthValue,
    stored_filter: FilterAnalysisInput,
    mode: RetrievalMode,
    direct: float,
    rows: tuple[CandidateRow, ...],
    expected: DiagnosticCutoffRelation,
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if mode is RetrievalMode.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    result = analyze_diagnostic(
        _input(
            mode,
            target=TargetLookupInput(
                available=True,
                bm25_score=direct if mode is RetrievalMode.BM25 else None,
                vector_distance=direct if mode is RetrievalMode.VECTOR else None,
            ),
            candidates=(CandidateListInput(1, role, 50, rows),),
            stored_filter=stored_filter,
        )
    )
    evidence = result.candidate_evidence[0]
    assert result.filter_root_result is root
    assert evidence.relation is expected
    assert (
        evidence.stored_filter_result
        is {
            TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
            TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
        }[root]
    )
    cutoff_findings = [
        item
        for item in result.observations
        if item.evidence and item.evidence[0].label.startswith("cutoff_")
    ]
    if root is TruthValue.FALSE and expected is DiagnosticCutoffRelation.NOT_OBSERVABLE:
        assert cutoff_findings == []
    else:
        assert len(cutoff_findings) == 1
    _response(result, mode=mode)


@pytest.mark.parametrize(
    ("mode", "role", "score"),
    [
        (
            RetrievalMode.BM25,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            3.0,
        ),
        (
            RetrievalMode.VECTOR,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            0.2,
        ),
    ],
)
def test_stored_target_present_while_filter_is_false_fails_structural_consistency(
    mode: RetrievalMode,
    role: DiagnosticSubqueryRole,
    score: float,
) -> None:
    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(
            _input(
                mode,
                candidates=(CandidateListInput(1, role, 50, (CandidateRow(_TARGET, 1, score),)),),
                stored_filter=_trivial_filter(
                    right="wanted",
                    observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
                ),
            )
        )
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_CANDIDATES


@pytest.mark.parametrize(
    ("mode", "direct", "rows"),
    [
        (RetrievalMode.BM25, 52.0, _bm25_rows(50)),
        (RetrievalMode.BM25, 3.0, (CandidateRow(_TARGET, 1, 3.0 + 4e-12),)),
        (RetrievalMode.VECTOR, 0.2, (CandidateRow(_TARGET, 1, 0.2 + 2e-12),)),
    ],
)
def test_candidate_direction_contradiction_or_score_mismatch_rejects(
    mode: RetrievalMode,
    direct: float,
    rows: tuple[CandidateRow, ...],
) -> None:
    role = (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES
        if mode is RetrievalMode.BM25
        else DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES
    )
    target = TargetLookupInput(
        available=True,
        bm25_score=direct if mode is RetrievalMode.BM25 else None,
        vector_distance=direct if mode is RetrievalMode.VECTOR else None,
    )
    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(
            _input(
                mode,
                target=target,
                candidates=(CandidateListInput(1, role, 50, rows),),
            )
        )
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_CANDIDATES


def test_candidate_target_score_uses_exact_frozen_tolerance() -> None:
    within_relative = CandidateRow(_TARGET, 1, 3.0 + 2e-12)
    relative = analyze_diagnostic(
        _input(
            candidates=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    50,
                    (within_relative,),
                ),
            )
        )
    )
    assert relative.candidate_evidence[0].target_present is True

    within_absolute = CandidateRow(_TARGET, 1, 1e-12 + 5e-16)
    absolute = analyze_diagnostic(
        _input(
            target=TargetLookupInput(available=True, bm25_score=1e-12),
            candidates=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    50,
                    (within_absolute,),
                ),
            ),
        )
    )
    assert absolute.candidate_evidence[0].target_present is True


def test_unavailable_target_keeps_safe_summaries_and_suppresses_all_downstream_evidence() -> None:
    rows = _bm25_rows(50)
    result = analyze_diagnostic(
        _input(
            target=TargetLookupInput(available=False),
            candidates=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    50,
                    rows,
                ),
            ),
        )
    )

    assert result.target.available is False
    assert result.subqueries[1].returned_count == 50
    assert result.subqueries[1].boundary_score is not None
    assert (
        result.filter_evidence == result.candidate_evidence == result.qualified_rrf_evidence == ()
    )
    assert [item.code for item in result.observations] == [ForensicCode.NOT_OBSERVABLE]
    assert result.observations[0].origin.value == "live_expected_document_diagnostic"
    _response(result, mode=RetrievalMode.BM25)

    with pytest.raises(DiagnosticAnalysisError):
        analyze_diagnostic(
            _input(
                target=TargetLookupInput(available=False),
                candidates=(
                    CandidateListInput(
                        1,
                        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                        50,
                        (CandidateRow(_TARGET, 1, 3.0),),
                    ),
                ),
            )
        )


def _rrf_tie_lists(
    shared_count: int,
    target_rank: int,
) -> tuple[tuple[CandidateRow, ...], tuple[CandidateRow, ...]]:
    shared = tuple(UUID(int=3000 + index) for index in range(1, shared_count + 1))
    bm25 = (
        *(
            CandidateRow(document_id=document_id, rank=rank, score=float(1000 - rank))
            for rank, document_id in enumerate(shared, start=1)
        ),
        CandidateRow(_TARGET, target_rank, float(1000 - target_rank)),
    )
    ann_other = UUID(int=9999)
    ann = (
        *(
            CandidateRow(document_id=document_id, rank=rank, score=rank / 1000.0)
            for rank, document_id in enumerate(shared, start=1)
        ),
        CandidateRow(ann_other, target_rank, target_rank / 1000.0),
    )
    return bm25, ann


@pytest.mark.parametrize(
    ("shared_count", "target_rank", "expected_present", "expected_relation", "expected_rank"),
    [
        (48, 49, True, DiagnosticCutoffRelation.TARGET_PRESENT, 49),
        (49, 50, False, DiagnosticCutoffRelation.NOT_OBSERVABLE, None),
        (50, 51, False, DiagnosticCutoffRelation.OUTSIDE_CANDIDATES, None),
    ],
)
def test_qualified_rrf_competition_rank_and_cutoff_tie_groups(
    shared_count: int,
    target_rank: int,
    expected_present: bool,
    expected_relation: DiagnosticCutoffRelation,
    expected_rank: int | None,
) -> None:
    bm25, ann = _rrf_tie_lists(shared_count, target_rank)
    target = TargetLookupInput(
        available=True,
        bm25_score=float(1000 - target_rank),
        vector_distance=0.25,
    )
    candidates = (
        CandidateListInput(
            1,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            100,
            bm25,
        ),
        CandidateListInput(
            2,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            100,
            ann,
        ),
    )
    result = analyze_diagnostic(
        _input(RetrievalMode.HYBRID_RRF, target=target, candidates=candidates)
    )

    qualified = result.qualified_rrf_evidence[0]
    assert qualified.returned_count == min(shared_count + 2, 50)
    assert qualified.target_present is expected_present
    assert qualified.target_rank == expected_rank
    assert qualified.relation is expected_relation
    assert qualified.target_score.value == 1.0 / (60 + target_rank)
    if expected_relation is DiagnosticCutoffRelation.NOT_OBSERVABLE:
        assert qualified.boundary_score is not None
        assert math.isclose(
            qualified.target_score.value,
            qualified.boundary_score.value,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    if expected_relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES:
        rrf_observation = next(
            item for item in result.observations if item.code is ForensicCode.OUTSIDE_FUSION_TOP_K
        )
        assert "qualified client-computed fusion boundary" in rrf_observation.statement
        assert "server" not in rrf_observation.statement.lower()
    _response(result, mode=RetrievalMode.HYBRID_RRF)


def test_rrf_union_overlap_weights_contributions_and_counterfactual_certainty_are_exact() -> None:
    result = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            include_no_filter=True,
            rrf=RrfInputs(2.0, 3.0, 10),
        )
    )

    stored, counterfactual = result.qualified_rrf_evidence
    assert stored.returned_count == counterfactual.returned_count == 1
    assert stored.target_score.value == counterfactual.target_score.value == 5 / 11
    assert stored.certainty is EvidenceCertainty.OBSERVED
    assert counterfactual.certainty is EvidenceCertainty.COUNTERFACTUAL
    assert stored.bm25_rank == stored.ann_rank == 1


def test_outside_rrf_finding_binds_both_exact_target_rank_contributions() -> None:
    shared = tuple(UUID(int=30_000 + rank) for rank in range(1, 51))
    bm25_rows = (
        *(
            CandidateRow(document_id, rank, float(1000 - rank))
            for rank, document_id in enumerate(shared, start=1)
        ),
        CandidateRow(_TARGET, 51, 949.0),
    )
    ann_rows = (
        *(
            CandidateRow(document_id, rank, rank / 1000.0)
            for rank, document_id in enumerate(shared, start=1)
        ),
        CandidateRow(_TARGET, 51, 0.051),
    )
    result = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            target=TargetLookupInput(
                available=True,
                bm25_score=949.0,
                vector_distance=0.051,
            ),
            candidates=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    100,
                    bm25_rows,
                ),
                CandidateListInput(
                    2,
                    DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
                    100,
                    ann_rows,
                ),
            ),
            rrf=RrfInputs(2.0, 3.0, 10),
        )
    )

    qualified = result.qualified_rrf_evidence[0]
    assert qualified.bm25_rank == qualified.ann_rank == 51
    assert math.isclose(
        qualified.target_score.value,
        5 / 61,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
    assert qualified.target_present is False
    assert qualified.relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
    observation = next(
        item for item in result.observations if item.code is ForensicCode.OUTSIDE_FUSION_TOP_K
    )
    contributions = [
        item.value for item in observation.evidence if item.value.kind == "rrf_contribution"
    ]
    assert [(item.rank, item.weight, item.contribution) for item in contributions] == [
        (51, 2.0, 2 / 61),
        (51, 3.0, 3 / 61),
    ]
    _response(result, mode=RetrievalMode.HYBRID_RRF)


def test_qualified_rrf_zero_short_and_full_target_absence_semantics() -> None:
    empty_candidates = (
        CandidateListInput(
            1,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            100,
            (),
        ),
        CandidateListInput(
            2,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            100,
            (),
        ),
    )
    short = analyze_diagnostic(
        _input(RetrievalMode.HYBRID_RRF, candidates=empty_candidates)
    ).qualified_rrf_evidence[0]
    assert short.returned_count == 0
    assert short.target_score.value == 0
    assert short.boundary_score is None
    assert short.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE

    documents = tuple(UUID(int=20_000 + rank) for rank in range(1, 101))
    full_candidates = (
        CandidateListInput(
            1,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            100,
            tuple(
                CandidateRow(document_id, rank, float(201 - rank))
                for rank, document_id in enumerate(documents, start=1)
            ),
        ),
        CandidateListInput(
            2,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            100,
            tuple(
                CandidateRow(document_id, rank, rank / 100.0)
                for rank, document_id in enumerate(documents, start=1)
            ),
        ),
    )
    full_result = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            target=TargetLookupInput(
                available=True,
                bm25_score=3.0,
                vector_distance=2.0,
            ),
            candidates=full_candidates,
        )
    )
    full = full_result.qualified_rrf_evidence[0]
    assert full.returned_count == 50
    assert full.target_score.value == 0
    assert full.boundary_score is not None
    assert full.relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
    _response(full_result, mode=RetrievalMode.HYBRID_RRF)


def _hybrid_absent_lists(*, full: bool) -> tuple[CandidateListInput, CandidateListInput]:
    documents = tuple(UUID(int=80_000 + rank) for rank in range(1, 101))
    count = 100 if full else 0
    return (
        CandidateListInput(
            1,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            100,
            tuple(
                CandidateRow(document_id, rank, float(201 - rank))
                for rank, document_id in enumerate(documents[:count], start=1)
            ),
        ),
        CandidateListInput(
            2,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            100,
            tuple(
                CandidateRow(document_id, rank, rank / 100.0)
                for rank, document_id in enumerate(documents[:count], start=1)
            ),
        ),
    )


@pytest.mark.parametrize(
    ("root", "stored_filter"),
    [
        (
            TruthValue.FALSE,
            _trivial_filter(
                right="wanted",
                observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
            ),
        ),
        (
            TruthValue.UNKNOWN,
            _trivial_filter(
                right=None,
                observed=PreservedAttribute(AttributePresence.PRESENT_NULL),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("full", "expected"),
    [
        (False, DiagnosticCutoffRelation.NOT_OBSERVABLE),
        (True, DiagnosticCutoffRelation.OUTSIDE_CANDIDATES),
    ],
)
def test_qualified_rrf_preserves_exact_arithmetic_under_false_or_unknown_filter(
    root: TruthValue,
    stored_filter: FilterAnalysisInput,
    full: bool,
    expected: DiagnosticCutoffRelation,
) -> None:
    result = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            target=TargetLookupInput(available=True, bm25_score=3.0, vector_distance=2.0),
            candidates=_hybrid_absent_lists(full=full),
            stored_filter=stored_filter,
        )
    )
    qualified = result.qualified_rrf_evidence[0]
    assert result.filter_root_result is root
    assert (
        qualified.stored_filter_result
        is {
            TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
            TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
        }[root]
    )
    assert qualified.bm25_rank is qualified.ann_rank is None
    assert qualified.target_score.value == 0
    assert qualified.relation is expected
    rrf_findings = [
        item
        for item in result.observations
        if item.evidence and item.evidence[0].label == "cutoff_stored_query_rrf"
    ]
    if root is TruthValue.FALSE and expected is DiagnosticCutoffRelation.NOT_OBSERVABLE:
        assert rrf_findings == []
    else:
        assert len(rrf_findings) == 1
    _response(result, mode=RetrievalMode.HYBRID_RRF)


def test_false_filter_forbids_stored_rrf_input_rank_while_unknown_preserves_membership() -> None:
    false_filter = _trivial_filter(
        right="wanted",
        observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
    )
    with pytest.raises(DiagnosticAnalysisError) as false_rank:
        analyze_diagnostic(_input(RetrievalMode.HYBRID_RRF, stored_filter=false_filter))
    assert false_rank.value.code is DiagnosticAnalysisErrorCode.INVALID_CANDIDATES

    unknown = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            stored_filter=_trivial_filter(
                right=None,
                observed=PreservedAttribute(AttributePresence.PRESENT_NULL),
            ),
        )
    )
    assert unknown.qualified_rrf_evidence[0].target_present is True
    assert unknown.qualified_rrf_evidence[0].relation is DiagnosticCutoffRelation.TARGET_PRESENT
    _response(unknown, mode=RetrievalMode.HYBRID_RRF)


def test_false_filter_rrf_counterfactual_remains_independent_and_unbound() -> None:
    false_filter = _trivial_filter(
        right="wanted",
        observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, "other"),
    )
    stored_bm25, stored_ann = _hybrid_absent_lists(full=False)
    counter_bm25 = replace(
        stored_bm25,
        ordinal=3,
        role=DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        rows=(CandidateRow(_TARGET, 1, 3.0),),
    )
    counter_ann = replace(
        stored_ann,
        ordinal=4,
        role=DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
        rows=(CandidateRow(_TARGET, 1, 0.2),),
    )
    result = analyze_diagnostic(
        _input(
            RetrievalMode.HYBRID_RRF,
            include_no_filter=True,
            candidates=(stored_bm25, stored_ann, counter_bm25, counter_ann),
            stored_filter=false_filter,
        )
    )
    stored, counterfactual = result.qualified_rrf_evidence
    assert stored.stored_filter_result is DiagnosticPredicateResult.NOT_MATCHED
    assert stored.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    assert counterfactual.stored_filter_result is None
    assert counterfactual.target_present is True
    assert counterfactual.relation is DiagnosticCutoffRelation.TARGET_PRESENT
    _response(result, mode=RetrievalMode.HYBRID_RRF, include_no_filter=True)


@pytest.mark.parametrize(
    "rrf",
    [
        None,
        RrfInputs(True, 1.0, 60),  # type: ignore[arg-type]
        RrfInputs(0.0, 1.0, 60),
        RrfInputs(1.0, float("nan"), 60),
        RrfInputs(1.0, 1.0, True),  # type: ignore[arg-type]
        RrfInputs(1.0, 1.0, 0),
        RrfInputs(1.0, 1.0, 60, 49),
    ],
)
def test_hybrid_rrf_inputs_fail_closed(rrf: RrfInputs | None) -> None:
    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(_input(RetrievalMode.HYBRID_RRF, rrf=rrf))
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_RRF


def test_hybrid_rerank_reports_only_would_be_top50_admission_without_final_claim() -> None:
    result = analyze_diagnostic(_input(RetrievalMode.HYBRID_RERANK))

    assert len(result.qualified_rrf_evidence) == 1
    assert result.qualified_rrf_evidence[0].cutoff == 50
    assert all(item.code is not ForensicCode.RERANKED_DOWN for item in result.observations)
    assert all(
        getattr(evidence.value, "stage", None)
        not in {RetrievalStage.RERANKER, RetrievalStage.FINAL}
        for observation in result.observations
        for evidence in observation.evidence
    )


def test_not_observable_filter_leaf_is_top_level_contract_compatible() -> None:
    result = analyze_diagnostic(
        _input(
            stored_filter=FilterAnalysisInput(
                node=FilterPredicate(
                    field="filter_field",
                    op=PredicateOp.LT,
                    value="value",
                ),
                schema=(FilterFieldSchema("filter_field", FilterValueType.STRING, True),),
                attributes=(
                    ObservedFilterAttribute(
                        "filter_field",
                        PreservedAttribute(AttributePresence.MISSING),
                    ),
                ),
            )
        )
    )
    response = _response(result, mode=RetrievalMode.BM25)
    assert response.filter_evidence[0].result.value == "not_observable"
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE


def test_false_root_created_only_by_not_is_contract_compatible_without_leaf_finding() -> None:
    result = analyze_diagnostic(
        _input(
            stored_filter=FilterAnalysisInput(
                node=FilterLogical(
                    op=LogicalOp.NOT,
                    children=[FilterPredicate(field="filter_field", op=PredicateOp.EQ, value=None)],
                ),
                schema=(FilterFieldSchema("filter_field", FilterValueType.STRING, True),),
                attributes=(
                    ObservedFilterAttribute(
                        "filter_field",
                        PreservedAttribute(AttributePresence.MISSING),
                    ),
                ),
            ),
            candidates=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    50,
                    (),
                ),
            ),
        )
    )
    assert result.filter_root_result is TruthValue.FALSE
    assert result.filter_evidence[0].result is DiagnosticPredicateResult.MATCHED
    assert all(
        item.code is not ForensicCode.FILTER_PREDICATE_FAILED for item in result.observations
    )
    response = _response(result, mode=RetrievalMode.BM25)
    assert response.stored_filter_result is DiagnosticPredicateResult.NOT_MATCHED


def test_output_and_input_repr_exclude_filter_values_and_unrelated_candidate_ids() -> None:
    unrelated = UUID(int=987654)
    marker_filter = "PRIVATE_FILTER_VALUE"
    marker_attribute = "PRIVATE_ATTRIBUTE_VALUE"
    value = _input(
        stored_filter=FilterAnalysisInput(
            node=FilterPredicate(
                field="filter_field",
                op=PredicateOp.NOT_EQ,
                value=marker_filter,
            ),
            schema=(FilterFieldSchema("filter_field", FilterValueType.STRING, True),),
            attributes=(
                ObservedFilterAttribute(
                    "filter_field",
                    PreservedAttribute(
                        AttributePresence.PRESENT_VALUE,
                        marker_attribute,
                    ),
                ),
            ),
        ),
        candidates=(
            CandidateListInput(
                1,
                DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                50,
                (
                    CandidateRow(_TARGET, 1, 3.0),
                    CandidateRow(unrelated, 2, 2.0),
                ),
            ),
        ),
    )
    result = analyze_diagnostic(value)
    response = _response(result, mode=RetrievalMode.BM25)
    dumped = response.model_dump_json()

    for forbidden in (
        marker_filter,
        marker_attribute,
        str(unrelated),
        "query_text",
        "namespace",
        "provider",
        "filter_value",
        "observed_value",
        "reranked_down",
    ):
        assert forbidden not in dumped
        assert forbidden not in repr(value)


class _ExplodingTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        raise RuntimeError("PRIVATE_TIMEZONE_MARKER")

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None


def test_malformed_inputs_raise_only_value_free_error_with_scrubbed_public_frame() -> None:
    marker = "PRIVATE_FILTER_TRACE_MARKER"
    value = _input(
        stored_filter=_trivial_filter(
            right=marker,
            observed=PreservedAttribute(AttributePresence.PRESENT_VALUE, 1),
        )
    )
    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(value)

    error = raised.value
    rendered = "".join(traceback.format_exception(error))
    assert error.code is DiagnosticAnalysisErrorCode.INVALID_FILTER
    assert error.__cause__ is None
    assert error.__context__ is None
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in rendered
    current = error.__traceback__
    while current is not None:
        if current.tb_frame.f_code.co_filename.endswith("diagnostic_analysis.py"):
            assert marker not in repr(current.tb_frame.f_locals)
        current = current.tb_next

    huge = _input(target=TargetLookupInput(available=True, bm25_score=10**10000))
    with pytest.raises(DiagnosticAnalysisError) as huge_error:
        analyze_diagnostic(huge)
    assert huge_error.value.code is DiagnosticAnalysisErrorCode.INVALID_TARGET

    over_contract_bound = _input(
        target=TargetLookupInput(available=True, bm25_score=1_000_000_000_001.0)
    )
    with pytest.raises(DiagnosticAnalysisError) as bounded_error:
        analyze_diagnostic(over_contract_bound)
    assert bounded_error.value.code is DiagnosticAnalysisErrorCode.INVALID_TARGET

    exploding_time = datetime(2026, 8, 23, tzinfo=_ExplodingTimezone())
    with pytest.raises(DiagnosticAnalysisError) as time_error:
        analyze_diagnostic(
            _input(
                binding=DiagnosticBinding(_CONFIG, _TARGET, exploding_time, _TRACE),
            )
        )
    assert time_error.value.code is DiagnosticAnalysisErrorCode.INVALID_OUTPUT


def _partial_instance(model: type[object], **fields: object) -> object:
    instance = object.__new__(model)
    for name, value in fields.items():
        object.__setattr__(instance, name, value)
    return instance


def test_missing_slotted_dataclass_attributes_always_map_to_fixed_errors() -> None:
    predicate = FilterPredicate(field="filter_field", op=PredicateOp.EQ, value=None)
    valid_schema = FilterFieldSchema("filter_field", FilterValueType.STRING, True)
    valid_attribute = ObservedFilterAttribute(
        "filter_field",
        PreservedAttribute(AttributePresence.MISSING),
    )
    attacks = [
        replace(
            _input(),
            binding=_partial_instance(
                DiagnosticBinding,
                config_id=_CONFIG,
                target_document_id=_TARGET,
                trace_id=_TRACE,
            ),  # type: ignore[arg-type]
        ),
        replace(
            _input(),
            target=_partial_instance(
                TargetLookupInput,
                bm25_score=3.0,
                vector_distance=None,
            ),  # type: ignore[arg-type]
        ),
        replace(
            _input(),
            candidate_lists=(
                _partial_instance(
                    CandidateListInput,
                    role=DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    requested_limit=50,
                    rows=(),
                ),  # type: ignore[arg-type]
            ),
        ),
        replace(
            _input(),
            candidate_lists=(
                CandidateListInput(
                    1,
                    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    50,
                    (
                        _partial_instance(
                            CandidateRow,
                            rank=1,
                            score=3.0,
                        ),  # type: ignore[arg-type]
                    ),
                ),
            ),
        ),
        replace(
            _input(),
            stored_filter=_partial_instance(
                FilterAnalysisInput,
                node=predicate,
                attributes=(valid_attribute,),
            ),  # type: ignore[arg-type]
        ),
        replace(
            _input(),
            stored_filter=FilterAnalysisInput(
                node=predicate,
                schema=(
                    _partial_instance(
                        FilterFieldSchema,
                        value_type=FilterValueType.STRING,
                    ),  # type: ignore[arg-type]
                ),
                attributes=(valid_attribute,),
            ),
        ),
        replace(
            _input(),
            stored_filter=FilterAnalysisInput(
                node=predicate,
                schema=(valid_schema,),
                attributes=(
                    _partial_instance(
                        ObservedFilterAttribute,
                        attribute=PreservedAttribute(AttributePresence.MISSING),
                    ),  # type: ignore[arg-type]
                ),
            ),
        ),
        replace(
            _input(),
            stored_filter=FilterAnalysisInput(
                node=predicate,
                schema=(valid_schema,),
                attributes=(
                    ObservedFilterAttribute(
                        "filter_field",
                        _partial_instance(PreservedAttribute, value=None),  # type: ignore[arg-type]
                    ),
                ),
            ),
        ),
        replace(
            _input(RetrievalMode.HYBRID_RRF),
            rrf=_partial_instance(
                RrfInputs,
                ann_weight=1.0,
                rank_constant=60,
                cutoff=50,
            ),  # type: ignore[arg-type]
        ),
    ]

    for attacked in attacks:
        with pytest.raises(DiagnosticAnalysisError) as raised:
            analyze_diagnostic(attacked)
        assert raised.value.code in {
            DiagnosticAnalysisErrorCode.INVALID_BINDING,
            DiagnosticAnalysisErrorCode.INVALID_FILTER,
            DiagnosticAnalysisErrorCode.INVALID_CANDIDATES,
            DiagnosticAnalysisErrorCode.INVALID_RRF,
            DiagnosticAnalysisErrorCode.INVALID_OUTPUT,
        }
        assert raised.value.__cause__ is raised.value.__context__ is None


@pytest.mark.parametrize("failure", [AttributeError("x"), OverflowError("x"), RuntimeError("x")])
def test_ordinary_malformed_raw_exceptions_are_mapped_to_fixed_error(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    def fail(_: object) -> NoReturn:
        raise failure

    monkeypatch.setattr(analysis_module, "_analyze", fail)
    with pytest.raises(DiagnosticAnalysisError) as raised:
        analyze_diagnostic(_input())
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_OUTPUT
    assert raised.value.__cause__ is raised.value.__context__ is None


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (KeyboardInterrupt("PRIVATE_KEYBOARD_MARKER"), None),
        (SystemExit("PRIVATE_SYSTEM_EXIT_MARKER"), 1),
        (MemoryError("PRIVATE_MEMORY_MARKER"), None),
    ],
)
def test_process_control_and_memory_failures_are_fresh_value_only_controls(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_code: int | None,
) -> None:
    def fail(_: object) -> NoReturn:
        raise failure

    monkeypatch.setattr(analysis_module, "_analyze", fail)
    with pytest.raises(type(failure)) as raised:
        analyze_diagnostic(_input())

    public = raised.value
    assert public is not failure
    assert public.__cause__ is public.__context__ is None
    assert failure.__traceback__ is failure.__cause__ is failure.__context__ is None
    if isinstance(public, SystemExit):
        assert public.code == expected_code == 1
    else:
        assert public.args == ()
    rendered = "".join(traceback.format_exception(public))
    for marker in (
        "PRIVATE_KEYBOARD_MARKER",
        "PRIVATE_SYSTEM_EXIT_MARKER",
        "PRIVATE_MEMORY_MARKER",
    ):
        assert marker not in rendered
        assert marker not in str(public)
        current = public.__traceback__
        while current is not None:
            if current.tb_frame.f_code.co_filename.endswith("diagnostic_analysis.py"):
                assert marker not in repr(current.tb_frame.f_locals)
            current = current.tb_next


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt("PRIVATE_PREFLIGHT_KEYBOARD"),
        SystemExit("PRIVATE_PREFLIGHT_EXIT"),
        MemoryError("PRIVATE_PREFLIGHT_MEMORY"),
    ],
)
def test_definition_preflight_uses_the_same_value_only_control_boundary(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    def fail(_: object) -> NoReturn:
        raise failure

    monkeypatch.setattr(analysis_module, "_validate_filter_definition", fail)
    definition = FilterDefinitionInput(
        node=FilterPredicate(field="filter_field", op=PredicateOp.EQ, value=None),
        schema=(FilterFieldSchema("filter_field", FilterValueType.STRING, True),),
    )
    with pytest.raises(type(failure)) as raised:
        preflight_filter_definition(definition)
    public = raised.value
    assert public is not failure
    assert public.__cause__ is public.__context__ is None
    assert failure.__traceback__ is failure.__cause__ is failure.__context__ is None
    if isinstance(public, SystemExit):
        assert public.code == 1
    else:
        assert public.args == ()
    rendered = "".join(traceback.format_exception(public))
    assert "PRIVATE_PREFLIGHT" not in rendered


def test_diagnostic_modules_have_pure_imports_and_no_forbidden_causal_copy() -> None:
    modules = (
        _ROOT / "backend/pufferlab/evals/diagnostic_models.py",
        _ROOT / "backend/pufferlab/evals/diagnostic_analysis.py",
    )
    forbidden_imports = (
        "fastapi",
        "sqlalchemy",
        "turbopuffer",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "pathlib",
        "os",
        "io",
        "sentence_transformers",
        "pufferlab.api",
        "pufferlab.application",
        "pufferlab.cli",
        "pufferlab.config",
        "pufferlab.jobs",
        "pufferlab.persistence",
        "pufferlab.providers",
        "pufferlab.retrieval",
    )
    forbidden_copy = (
        "searched cluster",
        "cache was cold",
        "filter ran before ann",
        "provider rationale",
        "reranked_down",
    )
    violations: list[str] = []
    combined = ""
    for path in modules:
        source = path.read_text(encoding="utf-8")
        combined += source.lower()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if module.startswith(forbidden_imports):
                    violations.append(f"{path.name}:{node.lineno}:{module}")
    assert violations == []
    assert all(copy not in combined for copy in forbidden_copy)
