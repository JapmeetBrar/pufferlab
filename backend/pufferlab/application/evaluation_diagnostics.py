"""Sensitive, request-scoped composition for one authenticated diagnostic."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from time import perf_counter
from typing import NoReturn, cast
from uuid import UUID

from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.forensics import (
    DiagnosticCandidateScope,
    DiagnosticCandidateSubquerySummary,
    DiagnosticPredicateResult,
    DiagnosticSignal,
    DiagnosticSubqueryRole,
    DiagnosticTargetLookupSubquerySummary,
    DiagnosticTargetUnavailableReason,
    EvidenceOrigin,
    ExpectedDocumentDiagnosticResponse,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.evals.diagnostic_analysis import analyze_diagnostic
from pufferlab.evals.diagnostic_models import (
    AttributePresence,
    CandidateListInput,
    CandidateRow,
    DiagnosticAnalysisInput,
    DiagnosticAnalysisResult,
    DiagnosticBinding,
    FilterAnalysisInput,
    FilterFieldSchema,
    ObservedFilterAttribute,
    PreservedAttribute,
    RrfInputs,
    TargetLookupInput,
    TruthValue,
)
from pufferlab.retrieval.config import SeededSearchConfig
from pufferlab.retrieval.diagnostic import (
    DiagnosticProviderFactory,
    DiagnosticRetrievalInput,
    execute_expected_document_diagnostic,
)
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticProviderResult,
    diagnostic_roles,
)
from pufferlab.retrieval.types import QueryEmbedder

_MAX_DIAGNOSTIC_DURATION_MS = 600_000.0
_DURATION_TOLERANCE_MS = 1e-6


class ExpectedDocumentDiagnosticFailure(RuntimeError):
    """Value-free ordinary failure at the application diagnostic boundary."""

    def __init__(self) -> None:
        super().__init__("expected-document diagnostic failed")


@dataclass(frozen=True, slots=True, repr=False)
class ExpectedDocumentDiagnosticBinding:
    """One authenticated server-derived binding; its sensitive values are never represented."""

    run_id: UUID
    query_id: UUID
    query_text: str
    config: SeededSearchConfig
    target_document_id: UUID
    namespace: str
    stored_filter: FilterNode | None
    filter_schema: tuple[FilterFieldSchema, ...]
    include_no_filter_counterfactual: bool


class _ControlOutcome(StrEnum):
    NONE = "none"
    CANCELLED = "cancelled"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    SYSTEM_EXIT = "system_exit"


@dataclass(frozen=True, slots=True)
class _CompositionOutcome:
    response: ExpectedDocumentDiagnosticResponse | None = None
    failed: bool = False
    control: _ControlOutcome = _ControlOutcome.NONE


async def compose_expected_document_diagnostic(
    binding: ExpectedDocumentDiagnosticBinding,
    *,
    provider_factory: DiagnosticProviderFactory,
    query_embedder: QueryEmbedder | None,
    now: Callable[[], datetime],
    trace_id_factory: Callable[[], UUID],
    monotonic: Callable[[], float] = perf_counter,
) -> ExpectedDocumentDiagnosticResponse:
    """Retrieve, analyze, revalidate, and release one sensitive diagnostic binding."""

    outcome = await _compose_sensitive(
        binding,
        provider_factory=provider_factory,
        query_embedder=query_embedder,
        now=now,
        trace_id_factory=trace_id_factory,
        monotonic=monotonic,
    )
    binding = cast(ExpectedDocumentDiagnosticBinding, None)
    provider_factory = cast(DiagnosticProviderFactory, None)
    query_embedder = cast(QueryEmbedder, None)
    now = cast(Callable[[], datetime], None)
    trace_id_factory = cast(Callable[[], UUID], None)
    monotonic = cast(Callable[[], float], None)
    if outcome.control is not _ControlOutcome.NONE:
        _raise_control(outcome.control)
    if outcome.failed or outcome.response is None:
        raise ExpectedDocumentDiagnosticFailure() from None
    return outcome.response


async def _compose_sensitive(
    binding: object,
    *,
    provider_factory: DiagnosticProviderFactory,
    query_embedder: QueryEmbedder | None,
    now: Callable[[], datetime],
    trace_id_factory: Callable[[], UUID],
    monotonic: Callable[[], float],
) -> _CompositionOutcome:
    retrieval_result = None
    provider_result: DiagnosticProviderResult | None = None
    analysis_input: DiagnosticAnalysisInput | None = None
    analysis = None
    response: ExpectedDocumentDiagnosticResponse | None = None
    error: BaseException | None = None
    try:
        checked = _validate_binding(binding)
        trace_id = trace_id_factory()
        if type(trace_id) is not UUID:
            raise ValueError("diagnostic trace identity is invalid")
        started = _finite_clock(monotonic())
        retrieval_result = await execute_expected_document_diagnostic(
            DiagnosticRetrievalInput(
                namespace=checked.namespace,
                query_text=checked.query_text,
                target_document_id=checked.target_document_id,
                config=checked.config,
                stored_filter=checked.stored_filter,
                include_no_filter_counterfactual=checked.include_no_filter_counterfactual,
            ),
            provider_factory=provider_factory,
            query_embedder=query_embedder,
        )
        provider_result = retrieval_result.provider_result
        observed_at = now()
        finished = _finite_clock(monotonic())
        duration_ms = _duration_ms(started, finished)
        _validate_component_durations(
            duration_ms,
            embedding_duration_ms=retrieval_result.embedding_duration_ms,
            provider_duration_ms=provider_result.client_duration_ms,
        )
        analysis_input = _analysis_input(
            checked,
            provider_result,
            observed_at=observed_at,
            trace_id=trace_id,
        )
        analysis = analyze_diagnostic(analysis_input)
        _crosscheck_analysis_against_provider(
            analysis,
            checked,
            provider_result,
            observed_at=observed_at,
            trace_id=trace_id,
        )
        stored_filter_result = _public_truth(analysis.filter_root_result)
        response = ExpectedDocumentDiagnosticResponse(
            run_id=checked.run_id,
            query_id=checked.query_id,
            data_origin=DataOrigin.LIVE,
            origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
            config_id=checked.config.summary.id,
            config_mode=checked.config.mode,
            target_document_id=checked.target_document_id,
            included_no_filter_counterfactual=checked.include_no_filter_counterfactual,
            stored_filter_result=stored_filter_result,
            observed_at=observed_at,
            trace_id=trace_id,
            duration_ms=duration_ms,
            embedding_duration_ms=retrieval_result.embedding_duration_ms,
            subqueries=list(analysis.subqueries),
            target=analysis.target,
            filter_evidence=list(analysis.filter_evidence),
            candidate_evidence=list(analysis.candidate_evidence),
            qualified_rrf_evidence=list(analysis.qualified_rrf_evidence),
            observations=list(analysis.observations),
        )
        response = ExpectedDocumentDiagnosticResponse.model_validate(
            response.model_dump(mode="python", warnings=False)
        )
        _crosscheck_public_response(
            response,
            checked,
            analysis,
            observed_at=observed_at,
            trace_id=trace_id,
            duration_ms=duration_ms,
            embedding_duration_ms=retrieval_result.embedding_duration_ms,
        )
    except asyncio.CancelledError as caught:
        error = caught
    except KeyboardInterrupt as caught:
        error = caught
    except SystemExit as caught:
        error = caught
    except BaseException as caught:
        error = caught

    control = _classify_control(error)
    if error is not None:
        _detach_exception(error)
    binding = None
    checked = cast(ExpectedDocumentDiagnosticBinding, None)
    provider_factory = cast(DiagnosticProviderFactory, None)
    query_embedder = cast(QueryEmbedder, None)
    now = cast(Callable[[], datetime], None)
    trace_id_factory = cast(Callable[[], UUID], None)
    monotonic = cast(Callable[[], float], None)
    retrieval_result = None
    provider_result = None
    analysis_input = None
    analysis = None
    if error is not None:
        error = None
        return _CompositionOutcome(failed=control is _ControlOutcome.NONE, control=control)
    if response is None:
        return _CompositionOutcome(failed=True)
    return _CompositionOutcome(response=response)


def _validate_binding(value: object) -> ExpectedDocumentDiagnosticBinding:
    if type(value) is not ExpectedDocumentDiagnosticBinding:
        raise ValueError("diagnostic application binding is invalid")
    if (
        type(value.run_id) is not UUID
        or type(value.query_id) is not UUID
        or type(value.query_text) is not str
        or not value.query_text
        or type(value.config) is not SeededSearchConfig
        or type(value.config.mode) is not RetrievalMode
        or type(value.target_document_id) is not UUID
        or type(value.namespace) is not str
        or type(value.filter_schema) is not tuple
        or not all(type(item) is FilterFieldSchema for item in value.filter_schema)
        or type(value.include_no_filter_counterfactual) is not bool
    ):
        raise ValueError("diagnostic application binding is invalid")
    return value


def _analysis_input(
    binding: ExpectedDocumentDiagnosticBinding,
    provider_result: DiagnosticProviderResult,
    *,
    observed_at: datetime,
    trace_id: UUID,
) -> DiagnosticAnalysisInput:
    target = provider_result.target
    target_input = TargetLookupInput(
        available=target.available,
        bm25_score=None if target.bm25_score is None else target.bm25_score.value,
        vector_distance=(None if target.vector_distance is None else target.vector_distance.value),
    )
    candidates = tuple(
        CandidateListInput(
            ordinal=ordinal,
            role=item.role,
            requested_limit=item.requested_limit,
            rows=tuple(
                CandidateRow(
                    document_id=row.document_id,
                    rank=row.rank,
                    score=row.score.value,
                )
                for row in item.rows
            ),
        )
        for ordinal, item in enumerate(provider_result.candidate_lists, start=1)
    )
    stored_filter = None
    if binding.stored_filter is not None:
        stored_filter = FilterAnalysisInput(
            node=binding.stored_filter,
            schema=binding.filter_schema,
            attributes=tuple(
                ObservedFilterAttribute(
                    field=item.field,
                    attribute=PreservedAttribute(
                        presence=_attribute_presence(item.state),
                        value=_analysis_attribute_value(item.value),
                    ),
                )
                for item in target.attributes
            ),
        )
    return DiagnosticAnalysisInput(
        binding=DiagnosticBinding(
            config_id=binding.config.summary.id,
            target_document_id=binding.target_document_id,
            observed_at=observed_at,
            trace_id=trace_id,
        ),
        mode=binding.config.mode,
        include_no_filter_counterfactual=binding.include_no_filter_counterfactual,
        target=target_input,
        candidate_lists=candidates,
        stored_filter=stored_filter,
        rrf=_rrf_inputs(binding.config),
    )


def _rrf_inputs(config: SeededSearchConfig) -> RrfInputs | None:
    if config.mode not in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}:
        return None
    if config.rrf_weights is None or config.rrf_rank_constant is None:
        raise ValueError("hybrid diagnostic config is missing RRF inputs")
    cutoff = config.result_k if config.mode is RetrievalMode.HYBRID_RRF else config.reranker_depth
    if cutoff is None:
        raise ValueError("hybrid rerank diagnostic config is missing its admission depth")
    return RrfInputs(
        bm25_weight=config.rrf_weights[0],
        ann_weight=config.rrf_weights[1],
        rank_constant=config.rrf_rank_constant,
        cutoff=cutoff,
    )


def _attribute_presence(value: DiagnosticAttributeState) -> AttributePresence:
    return {
        DiagnosticAttributeState.MISSING: AttributePresence.MISSING,
        DiagnosticAttributeState.PRESENT_NULL: AttributePresence.PRESENT_NULL,
        DiagnosticAttributeState.PRESENT_VALUE: AttributePresence.PRESENT_VALUE,
    }[value]


def _analysis_attribute_value(value: object) -> object:
    if type(value) is list:
        return tuple(_analysis_attribute_value(item) for item in value)
    return value


def _public_truth(value: TruthValue | None) -> DiagnosticPredicateResult | None:
    if value is None:
        return None
    return {
        TruthValue.TRUE: DiagnosticPredicateResult.MATCHED,
        TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
        TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
    }[value]


def _crosscheck_public_response(
    response: ExpectedDocumentDiagnosticResponse,
    binding: ExpectedDocumentDiagnosticBinding,
    analysis: DiagnosticAnalysisResult,
    *,
    observed_at: datetime,
    trace_id: UUID,
    duration_ms: float,
    embedding_duration_ms: float | None,
) -> None:
    # Attribute access happens only after the pure result is known to be contract-valid.
    if (
        response.run_id != binding.run_id
        or response.query_id != binding.query_id
        or response.config_id != binding.config.summary.id
        or response.config_mode is not binding.config.mode
        or response.target_document_id != binding.target_document_id
        or response.included_no_filter_counterfactual
        is not binding.include_no_filter_counterfactual
        or response.data_origin is not DataOrigin.LIVE
        or response.origin is not EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC
        or response.observed_at != observed_at
        or response.trace_id != trace_id
        or response.duration_ms != duration_ms
        or response.embedding_duration_ms != embedding_duration_ms
        or response.observability_notice != "new_live_diagnostic_not_original_run"
    ):
        raise ValueError("diagnostic response identity does not match its authenticated binding")
    if (
        response.target != analysis.target
        or tuple(response.subqueries) != analysis.subqueries
        or tuple(response.filter_evidence) != analysis.filter_evidence
        or tuple(response.candidate_evidence) != analysis.candidate_evidence
        or tuple(response.qualified_rrf_evidence) != analysis.qualified_rrf_evidence
        or tuple(response.observations) != analysis.observations
        or response.stored_filter_result != _public_truth(analysis.filter_root_result)
    ):
        raise ValueError("diagnostic response evidence does not match pure analysis")


def _crosscheck_analysis_against_provider(
    analysis: DiagnosticAnalysisResult,
    binding: ExpectedDocumentDiagnosticBinding,
    provider_result: DiagnosticProviderResult,
    *,
    observed_at: datetime,
    trace_id: UUID,
) -> None:
    expected_roles = diagnostic_roles(
        binding.config.mode,
        binding.include_no_filter_counterfactual,
    )
    if (
        provider_result.namespace != binding.namespace
        or provider_result.target.target_document_id != binding.target_document_id
        or tuple(item.role for item in provider_result.candidate_lists) != expected_roles[1:]
    ):
        raise ValueError("diagnostic provider rows do not match their authenticated binding")
    target = provider_result.target
    expected_unavailable = (
        None
        if target.available
        else DiagnosticTargetUnavailableReason.TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT
    )
    if (
        analysis.target.config_id != binding.config.summary.id
        or analysis.target.target_document_id != binding.target_document_id
        or analysis.target.observed_at != observed_at
        or analysis.target.trace_id != trace_id
        or analysis.target.available is not target.available
        or analysis.target.unavailable_reason is not expected_unavailable
        or analysis.target.bm25_score != target.bm25_score
        or analysis.target.vector_distance != target.vector_distance
    ):
        raise ValueError("diagnostic target summary differs from provider rows")

    expected_summaries: list[
        DiagnosticTargetLookupSubquerySummary | DiagnosticCandidateSubquerySummary
    ] = [
        DiagnosticTargetLookupSubquerySummary(
            returned_count=1 if target.available else 0,
            target_present=target.available,
        )
    ]
    for ordinal, candidate_list in enumerate(provider_result.candidate_lists, start=1):
        rows = tuple(
            row for row in candidate_list.rows if row.document_id == binding.target_document_id
        )
        if len(rows) > 1:
            raise ValueError("diagnostic target occurs more than once in a candidate list")
        target_row = None if not rows else rows[0]
        boundary = (
            candidate_list.rows[-1]
            if len(candidate_list.rows) == candidate_list.requested_limit
            else None
        )
        expected_summaries.append(
            DiagnosticCandidateSubquerySummary(
                ordinal=ordinal,
                role=candidate_list.role,
                requested_limit=candidate_list.requested_limit,
                returned_count=len(candidate_list.rows),
                target_present=target_row is not None,
                target_rank=None if target_row is None else target_row.rank,
                target_score=None if target_row is None else target_row.score,
                boundary_score=None if boundary is None else boundary.score,
            )
        )
    if analysis.subqueries != tuple(expected_summaries):
        raise ValueError("diagnostic subquery summaries differ from provider rows")

    if target.available:
        if len(analysis.candidate_evidence) != len(provider_result.candidate_lists):
            raise ValueError("diagnostic candidate evidence does not cover provider rows")
        for evidence, summary in zip(
            analysis.candidate_evidence,
            expected_summaries[1:],
            strict=True,
        ):
            assert isinstance(summary, DiagnosticCandidateSubquerySummary)
            direct = (
                target.bm25_score
                if evidence.signal is DiagnosticSignal.BM25
                else target.vector_distance
            )
            if (
                evidence.subquery_ordinal != summary.ordinal
                or evidence.role is not summary.role
                or evidence.requested_limit != summary.requested_limit
                or evidence.returned_count != summary.returned_count
                or evidence.target_present is not summary.target_present
                or evidence.target_rank != summary.target_rank
                or evidence.target_score != summary.target_score
                or evidence.boundary_score != summary.boundary_score
                or evidence.direct_score != direct
            ):
                raise ValueError("diagnostic candidate evidence differs from provider rows")
    elif analysis.candidate_evidence:
        raise ValueError("unavailable diagnostic target cannot retain candidate evidence")

    _crosscheck_qualified_rrf(analysis, binding, provider_result)


def _crosscheck_qualified_rrf(
    analysis: DiagnosticAnalysisResult,
    binding: ExpectedDocumentDiagnosticBinding,
    provider_result: DiagnosticProviderResult,
) -> None:
    if binding.config.mode not in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}:
        if analysis.qualified_rrf_evidence:
            raise ValueError("nonhybrid diagnostic cannot retain qualified RRF evidence")
        return
    if not provider_result.target.available:
        if analysis.qualified_rrf_evidence:
            raise ValueError("unavailable diagnostic target cannot retain qualified RRF evidence")
        return
    inputs = _rrf_inputs(binding.config)
    assert inputs is not None
    by_role = {item.role: item for item in provider_result.candidate_lists}
    role_pairs = [
        (
            DiagnosticCandidateScope.STORED_QUERY,
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        )
    ]
    if binding.include_no_filter_counterfactual:
        role_pairs.append(
            (
                DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
                DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
                DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
            )
        )
    if len(analysis.qualified_rrf_evidence) != len(role_pairs):
        raise ValueError("qualified RRF scopes do not match provider rows")
    for evidence, (scope, bm25_role, ann_role) in zip(
        analysis.qualified_rrf_evidence,
        role_pairs,
        strict=True,
    ):
        bm25 = by_role[bm25_role]
        ann = by_role[ann_role]
        totals: dict[UUID, list[float]] = {}
        ranks: dict[UUID, tuple[int | None, int | None]] = {}
        for row in bm25.rows:
            totals.setdefault(row.document_id, []).append(
                inputs.bm25_weight / (inputs.rank_constant + row.rank)
            )
            ranks[row.document_id] = (
                row.rank,
                ranks.get(row.document_id, (None, None))[1],
            )
        for row in ann.rows:
            totals.setdefault(row.document_id, []).append(
                inputs.ann_weight / (inputs.rank_constant + row.rank)
            )
            ranks[row.document_id] = (
                ranks.get(row.document_id, (None, None))[0],
                row.rank,
            )
        fused = {document_id: math.fsum(parts) for document_id, parts in totals.items()}
        ordered = sorted(fused.values(), reverse=True)
        returned_count = min(len(ordered), inputs.cutoff)
        boundary = ordered[inputs.cutoff - 1] if len(ordered) >= inputs.cutoff else None
        target_score = fused.get(binding.target_document_id, 0.0)
        target_ranks = ranks.get(binding.target_document_id, (None, None))
        strictly_higher = sum(
            score > target_score and not _scores_close(score, target_score)
            for score in fused.values()
        )
        tied = sum(_scores_close(score, target_score) for score in fused.values())
        target_in_union = binding.target_document_id in fused
        target_present = target_in_union and (
            len(fused) <= inputs.cutoff or strictly_higher + tied <= inputs.cutoff
        )
        target_rank = strictly_higher + 1 if target_present else None
        if (
            evidence.scope is not scope
            or evidence.bm25_rank != target_ranks[0]
            or evidence.ann_rank != target_ranks[1]
            or evidence.bm25_weight != inputs.bm25_weight
            or evidence.ann_weight != inputs.ann_weight
            or evidence.rank_constant != inputs.rank_constant
            or evidence.cutoff != inputs.cutoff
            or evidence.returned_count != returned_count
            or evidence.target_present is not target_present
            or evidence.target_rank != target_rank
            or not _scores_close(evidence.target_score.value, target_score)
            or (
                (evidence.boundary_score is None) is not (boundary is None)
                or (
                    boundary is not None
                    and evidence.boundary_score is not None
                    and not _scores_close(evidence.boundary_score.value, boundary)
                )
            )
        ):
            raise ValueError("qualified RRF evidence differs from provider rows")


def _scores_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _finite_clock(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValueError("diagnostic monotonic clock is invalid")
    number = cast(int | float, value)
    if not math.isfinite(number):
        raise ValueError("diagnostic monotonic clock is invalid")
    return float(number)


def _duration_ms(started: float, finished: float) -> float:
    duration = (finished - started) * 1000.0
    if not math.isfinite(duration) or duration < 0 or duration > _MAX_DIAGNOSTIC_DURATION_MS:
        raise ValueError("diagnostic duration is invalid")
    return duration


def _validate_component_durations(
    duration_ms: float,
    *,
    embedding_duration_ms: float | None,
    provider_duration_ms: float,
) -> None:
    components = [provider_duration_ms]
    if embedding_duration_ms is not None:
        components.append(embedding_duration_ms)
    if any(
        type(value) not in {int, float}
        or not math.isfinite(value)
        or value < 0
        or value > _MAX_DIAGNOSTIC_DURATION_MS
        for value in components
    ):
        raise ValueError("diagnostic component duration is invalid")
    if math.fsum(float(value) for value in components) > duration_ms + _DURATION_TOLERANCE_MS:
        raise ValueError("diagnostic component durations exceed total duration")


def _classify_control(error: BaseException | None) -> _ControlOutcome:
    if isinstance(error, asyncio.CancelledError):
        return _ControlOutcome.CANCELLED
    if isinstance(error, KeyboardInterrupt):
        return _ControlOutcome.KEYBOARD_INTERRUPT
    if isinstance(error, SystemExit):
        return _ControlOutcome.SYSTEM_EXIT
    return _ControlOutcome.NONE


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None


def _raise_control(control: _ControlOutcome) -> NoReturn:
    if control is _ControlOutcome.CANCELLED:
        raise asyncio.CancelledError() from None
    if control is _ControlOutcome.KEYBOARD_INTERRUPT:
        raise KeyboardInterrupt() from None
    if control is _ControlOutcome.SYSTEM_EXIT:
        raise SystemExit(1) from None
    raise AssertionError("unreachable control outcome")
