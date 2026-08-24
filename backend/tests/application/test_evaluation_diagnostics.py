from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pufferlab.application import evaluation_diagnostics as diagnostic_module
from pufferlab.application.evaluation_diagnostics import (
    ExpectedDocumentDiagnosticBinding,
    ExpectedDocumentDiagnosticFailure,
    compose_expected_document_diagnostic,
)
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import FilterPredicate, PredicateOp
from pufferlab.contracts.forensics import (
    CandidateCutoffEvidence,
    DiagnosticCandidateSubquerySummary,
    DiagnosticTargetUnavailableReason,
    ForensicCode,
    QualifiedRrfEvidence,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.evals.diagnostic_models import FilterFieldSchema, FilterValueType
from pufferlab.retrieval.config import SeededSearchConfig, build_search_catalog
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticAttributeValue,
    DiagnosticCandidateList,
    DiagnosticCandidateRow,
    DiagnosticProviderRequest,
    DiagnosticProviderResult,
    DiagnosticTargetObservation,
)
from pufferlab.retrieval.types import QueryEmbedding
from pufferlab.synthetic_demo import AUTHORED_SYNTHETIC_DEMO

_RUN_ID = UUID("10000000-0000-0000-0000-000000000001")
_QUERY_ID = UUID("20000000-0000-0000-0000-000000000001")
_TARGET_ID = UUID("30000000-0000-0000-0000-000000000001")
_TRACE_ID = UUID("40000000-0000-0000-0000-000000000001")
_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _score(signal: str, value: float, *, direct: bool) -> ObservedScore:
    bm25 = signal == "bm25"
    return ObservedScore(
        kind=ScoreKind.BM25 if bm25 else ScoreKind.VECTOR_DISTANCE,
        value=value,
        direction=(ScoreDirection.HIGHER_IS_BETTER if bm25 else ScoreDirection.LOWER_IS_BETTER),
        source=(ScoreSource.COMPUTE_ATTRIBUTE if direct else ScoreSource.TURBOPUFFER_DIST),
    )


class _Embedder:
    def __init__(self, config: SeededSearchConfig) -> None:
        assert config.embedding_model is not None
        assert config.embedding_revision is not None
        assert config.embedding_dimensions is not None
        self.model = config.embedding_model
        self.revision = config.embedding_revision
        self.dimensions = config.embedding_dimensions
        self.calls = 0

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        assert query_text == "sensitive diagnostic query"
        self.calls += 1
        return QueryEmbedding(vector=(0.0,) * self.dimensions, client_duration_ms=0.0)


class _Provider:
    def __init__(
        self,
        *,
        target_available: bool = True,
        client_duration_ms: float = 0.0,
    ) -> None:
        self.target_available = target_available
        self.client_duration_ms = client_duration_ms
        self.requests: list[DiagnosticProviderRequest] = []
        self.close_calls = 0

    async def query(self, request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        self.requests.append(request)
        attributes = (
            tuple(
                DiagnosticAttributeValue(
                    field=field,
                    state=DiagnosticAttributeState.PRESENT_VALUE,
                    value="doc-1",
                )
                for field in request.filter_fields
            )
            if self.target_available
            else ()
        )
        target = DiagnosticTargetObservation(
            target_document_id=request.target_document_id,
            available=self.target_available,
            bm25_score=(
                _score("bm25", 4.0, direct=True)
                if self.target_available and request.lexical_fields is not None
                else None
            ),
            vector_distance=(
                _score("ann", 0.2, direct=True)
                if self.target_available and request.query_vector is not None
                else None
            ),
            attributes=attributes,
        )
        candidate_lists = tuple(
            DiagnosticCandidateList(
                role=role,
                requested_limit=request.candidate_limit,
                rows=(
                    (
                        DiagnosticCandidateRow(
                            document_id=request.target_document_id,
                            rank=1,
                            score=(
                                _score("bm25", 4.0, direct=False)
                                if "bm25" in role.value
                                else _score("ann", 0.2, direct=False)
                            ),
                        ),
                    )
                    if self.target_available
                    else ()
                ),
            )
            for role in request.roles[1:]
        )
        return DiagnosticProviderResult(
            namespace=request.namespace,
            target=target,
            candidate_lists=candidate_lists,
            client_duration_ms=self.client_duration_ms,
        )

    async def close(self) -> None:
        self.close_calls += 1


def _config(mode: RetrievalMode) -> SeededSearchConfig:
    return next(
        config
        for config in build_search_catalog(
            AUTHORED_SYNTHETIC_DEMO.manifest,
            result_k=50,
            candidate_k=100,
            reranker_depth=50,
        ).configs
        if config.mode is mode
    )


def _binding(
    mode: RetrievalMode,
    *,
    filtered: bool,
    include_no_filter: bool,
) -> ExpectedDocumentDiagnosticBinding:
    return ExpectedDocumentDiagnosticBinding(
        run_id=_RUN_ID,
        query_id=_QUERY_ID,
        query_text="sensitive diagnostic query",
        config=_config(mode),
        target_document_id=_TARGET_ID,
        namespace="pufferlab-diagnostic-test",
        stored_filter=(
            FilterPredicate(field="external_id", op=PredicateOp.EQ, value="doc-1")
            if filtered
            else None
        ),
        filter_schema=(
            (FilterFieldSchema("external_id", FilterValueType.STRING, True),) if filtered else ()
        ),
        include_no_filter_counterfactual=include_no_filter,
    )


def _monotonic() -> Callable[[], float]:
    values = iter((1.0, 1.001))
    return lambda: next(values)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("include_no_filter", [False, True])
async def test_composer_binds_every_mode_and_optional_no_filter_to_one_safe_trace(
    mode: RetrievalMode,
    include_no_filter: bool,
) -> None:
    provider = _Provider()
    factory_calls = 0

    async def factory() -> _Provider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    embedder = None if mode is RetrievalMode.BM25 else _Embedder(_config(mode))
    response = await compose_expected_document_diagnostic(
        _binding(mode, filtered=True, include_no_filter=include_no_filter),
        provider_factory=factory,
        query_embedder=embedder,
        now=lambda: _NOW,
        trace_id_factory=lambda: _TRACE_ID,
        monotonic=_monotonic(),
    )

    expected_count = 2 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 3
    if include_no_filter:
        expected_count = 3 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 5
    assert response.run_id == _RUN_ID
    assert response.query_id == _QUERY_ID
    assert response.target_document_id == _TARGET_ID
    assert response.config_mode is mode
    assert response.trace_id == _TRACE_ID
    assert response.observed_at == _NOW
    assert response.duration_ms == pytest.approx(1.0)
    assert len(response.subqueries) == expected_count
    assert response.stored_filter_result.value == "matched"
    assert response.filter_evidence[0].field == "external_id"
    assert response.included_no_filter_counterfactual is include_no_filter
    assert factory_calls == 1
    assert len(provider.requests) == 1
    assert provider.close_calls == 1
    assert (0 if embedder is None else embedder.calls) == (0 if mode is RetrievalMode.BM25 else 1)
    assert "sensitive diagnostic query" not in response.model_dump_json()
    assert "doc-1" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_unavailable_target_is_a_safe_success_without_fabricated_evidence() -> None:
    provider = _Provider(target_available=False)

    async def factory() -> _Provider:
        return provider

    response = await compose_expected_document_diagnostic(
        _binding(RetrievalMode.BM25, filtered=False, include_no_filter=False),
        provider_factory=factory,
        query_embedder=None,
        now=lambda: _NOW,
        trace_id_factory=lambda: _TRACE_ID,
        monotonic=_monotonic(),
    )

    assert response.target.available is False
    assert (
        response.target.unavailable_reason
        is DiagnosticTargetUnavailableReason.TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT
    )
    assert response.filter_evidence == []
    assert response.candidate_evidence == []
    assert response.qualified_rrf_evidence == []
    assert len(response.observations) == 1
    assert response.observations[0].code is ForensicCode.NOT_OBSERVABLE
    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_component_duration_forgery_fails_after_provider_close() -> None:
    provider = _Provider(client_duration_ms=2.0)

    async def factory() -> _Provider:
        return provider

    with pytest.raises(ExpectedDocumentDiagnosticFailure) as raised:
        await compose_expected_document_diagnostic(
            _binding(RetrievalMode.BM25, filtered=False, include_no_filter=False),
            provider_factory=factory,
            query_embedder=None,
            now=lambda: _NOW,
            trace_id_factory=lambda: _TRACE_ID,
            monotonic=_monotonic(),
        )

    assert str(raised.value) == "expected-document diagnostic failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert provider.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("forgery", ["summary", "candidate", "rrf"])
async def test_independent_provider_row_crosscheck_rejects_contract_valid_analysis_forgery(
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    provider = _Provider()
    original = diagnostic_module.analyze_diagnostic

    def forged_analysis(value: object) -> object:
        result = original(value)  # type: ignore[arg-type]
        if forgery == "summary":
            summary = result.subqueries[1]
            assert isinstance(summary, DiagnosticCandidateSubquerySummary)
            payload = summary.model_dump(mode="python")
            payload["returned_count"] = 2
            forged = DiagnosticCandidateSubquerySummary.model_validate(payload)
            return replace(
                result,
                subqueries=(result.subqueries[0], forged, *result.subqueries[2:]),
            )
        if forgery == "candidate":
            evidence = result.candidate_evidence[0]
            payload = evidence.model_dump(mode="python")
            payload["returned_count"] = 2
            forged = CandidateCutoffEvidence.model_validate(payload)
            return replace(
                result,
                candidate_evidence=(forged, *result.candidate_evidence[1:]),
            )
        evidence = result.qualified_rrf_evidence[0]
        payload = evidence.model_dump(mode="python")
        payload["bm25_weight"] = 2.0
        target_score = payload["target_score"]
        target_score["value"] = 3.0 / 61.0
        forged = QualifiedRrfEvidence.model_validate(payload)
        return replace(
            result,
            qualified_rrf_evidence=(forged, *result.qualified_rrf_evidence[1:]),
        )

    monkeypatch.setattr(diagnostic_module, "analyze_diagnostic", forged_analysis)

    async def factory() -> _Provider:
        return provider

    with pytest.raises(ExpectedDocumentDiagnosticFailure):
        await compose_expected_document_diagnostic(
            _binding(
                RetrievalMode.HYBRID_RRF,
                filtered=False,
                include_no_filter=False,
            ),
            provider_factory=factory,
            query_embedder=_Embedder(_config(RetrievalMode.HYBRID_RRF)),
            now=lambda: _NOW,
            trace_id_factory=lambda: _TRACE_ID,
            monotonic=_monotonic(),
        )

    assert provider.close_calls == 1
