import json
import traceback
from uuid import UUID, uuid4

import pytest
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import FilterLogical, FilterPredicate, LogicalOp, PredicateOp
from pufferlab.contracts.retrieval import LexicalSpec, RetrievalMode
from pufferlab.contracts.search import RetrievalStage, SearchCompareRequest, TimingStage
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.providers.rerankers import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    RerankCandidate,
    RerankResult,
    RerankScore,
)
from pufferlab.providers.types import (
    ProviderDocument,
    ProviderHybridProbeResult,
    ProviderQueryResult,
)
from pufferlab.retrieval.config import build_search_catalog
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.service import SearchCompareService
from pufferlab.retrieval.types import QueryEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TRACE_ID = UUID("0f7b308a-c5ac-4438-a1b2-6901002386fd")
DOCUMENT_A = UUID("0e0ee431-1a4b-5ca6-9e98-4033c8d44498")
DOCUMENT_B = UUID("307a0026-1cd9-58a5-8131-729c329fd068")
DOCUMENT_C = UUID("fd00f34b-9d23-5655-81f2-b05a1b9f4ea8")


def _manifest() -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "format_version": 1,
            "slug": "pufferlab-tiny-unix",
            "version": "tiny-unix-v1",
            "title": "Test corpus",
            "license": "CC0-1.0",
            "source_url": "https://example.test/corpus",
            "embedding": {
                "provider": "sentence_transformers",
                "model": MODEL,
                "revision": REVISION,
                "dimensions": 3,
            },
            "vector": {
                "attribute": "vector",
                "dtype": "f16",
                "distance_metric": "cosine_distance",
            },
            "fts": {
                "attributes": ["title", "body"],
                "tokenizer": "word_v4",
                "case_sensitive": False,
                "language": "english",
                "stemming": False,
                "remove_stopwords": False,
                "ascii_folding": False,
                "max_token_length": 39,
                "k1": 1.2,
                "b": 0.75,
                "k3": 8.0,
            },
        }
    )


def _catalog():
    return build_search_catalog(_manifest(), result_k=3)


def _document(
    document_id: UUID,
    *,
    external_id: str,
    score: float,
    kind: ScoreKind,
) -> ProviderDocument:
    return ProviderDocument(
        id=str(document_id),
        attributes={
            "external_id": external_id,
            "title": f"Title {external_id}",
            "body": f"Body {external_id}",
            "source_url": f"https://example.test/{external_id}",
        },
        score=ObservedScore(
            kind=kind,
            value=score,
            direction=(
                ScoreDirection.LOWER_IS_BETTER
                if kind is ScoreKind.VECTOR_DISTANCE
                else ScoreDirection.HIGHER_IS_BETTER
            ),
            source=ScoreSource.TURBOPUFFER_DIST,
        ),
    )


class FakeProvider:
    def __init__(self) -> None:
        self.bm25_result = ProviderQueryResult(
            documents=(
                _document(DOCUMENT_A, external_id="a", score=7.5, kind=ScoreKind.BM25),
                _document(DOCUMENT_B, external_id="b", score=4.0, kind=ScoreKind.BM25),
            ),
            client_duration_ms=2.25,
        )
        self.ann_result = ProviderQueryResult(
            documents=(
                _document(
                    DOCUMENT_B,
                    external_id="b",
                    score=0.08,
                    kind=ScoreKind.VECTOR_DISTANCE,
                ),
                _document(
                    DOCUMENT_C,
                    external_id="c",
                    score=0.3,
                    kind=ScoreKind.VECTOR_DISTANCE,
                ),
            ),
            client_duration_ms=3.5,
        )
        self.hybrid_result = ProviderQueryResult(
            documents=(
                _document(DOCUMENT_B, external_id="b", score=0.0325, kind=ScoreKind.RRF),
                _document(DOCUMENT_A, external_id="a", score=0.0164, kind=ScoreKind.RRF),
                _document(DOCUMENT_C, external_id="c", score=0.0161, kind=ScoreKind.RRF),
            ),
            client_duration_ms=4.5,
        )
        self.hybrid_probe_result = ProviderHybridProbeResult(
            bm25_documents=self.bm25_result.documents,
            ann_documents=self.ann_result.documents,
            client_duration_ms=2.0,
        )
        self.bm25_calls: list[dict[str, object]] = []
        self.ann_calls: list[dict[str, object]] = []
        self.hybrid_calls: list[dict[str, object]] = []
        self.probe_calls: list[dict[str, object]] = []
        self.closed = False

    async def query_bm25(self, **kwargs: object) -> ProviderQueryResult:
        self.bm25_calls.append(kwargs)
        return self.bm25_result

    async def query_ann(self, **kwargs: object) -> ProviderQueryResult:
        self.ann_calls.append(kwargs)
        return self.ann_result

    async def query_hybrid_rrf(self, **kwargs: object) -> ProviderQueryResult:
        self.hybrid_calls.append(kwargs)
        return self.hybrid_result

    async def probe_hybrid_candidates(self, **kwargs: object) -> ProviderHybridProbeResult:
        self.probe_calls.append(kwargs)
        return self.hybrid_probe_result

    async def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    model = MODEL
    revision = REVISION
    dimensions = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        self.queries.append(query_text)
        return QueryEmbedding(vector=(0.125, -0.25, 0.5), client_duration_ms=1.75)


class FakeReranker:
    model = DEFAULT_RERANKER_MODEL
    revision = DEFAULT_RERANKER_REVISION

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    async def rerank(
        self,
        *,
        query_text: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> RerankResult:
        resolved = tuple(candidates)
        self.calls.append((query_text, resolved))
        scores = {str(DOCUMENT_A): 0.9, str(DOCUMENT_B): 0.1, str(DOCUMENT_C): 0.2}
        return RerankResult(
            scores=tuple(
                RerankScore(
                    document_id=candidate.document_id, score=scores[str(candidate.document_id)]
                )
                for candidate in resolved
            ),
            client_duration_ms=6.25,
        )


def _service(
    provider: FakeProvider | None = None,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[SearchCompareService, FakeProvider, FakeEmbedder, FakeReranker]:
    resolved_provider = provider or FakeProvider()
    resolved_embedder = embedder or FakeEmbedder()
    resolved_reranker = reranker or FakeReranker()
    service = SearchCompareService(
        namespace="pufferlab-test",
        catalog=_catalog(),
        write_spec=compile_namespace_write_spec(_manifest()),
        provider=resolved_provider,
        query_embedder=resolved_embedder,
        reranker=resolved_reranker,
        trace_id_factory=lambda: TRACE_ID,
    )
    return service, resolved_provider, resolved_embedder, resolved_reranker


def test_catalog_is_deterministic_and_describes_all_four_modes() -> None:
    first = _catalog().summaries()
    second = _catalog().summaries()

    assert first == second
    assert [summary.mode for summary in first] == [
        RetrievalMode.BM25,
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    ]
    assert len({summary.id for summary in first}) == 4
    assert all(len(summary.config_hash) == 64 for summary in first)
    rerank_config = _catalog().configs[-1]
    assert rerank_config.reranker_model == DEFAULT_RERANKER_MODEL
    assert rerank_config.reranker_revision == DEFAULT_RERANKER_REVISION
    assert rerank_config.reranker_depth == 50
    assert _catalog().configs[0].lexical_fields == (("title", 2.0), ("body", 1.0))

    manifest = _manifest()
    changed_manifest = manifest.model_copy(
        update={"vector": manifest.vector.model_copy(update={"dtype": "f32"})}
    )
    changed_schema = build_search_catalog(changed_manifest, result_k=3).summaries()
    assert [summary.config_hash for summary in changed_schema] != [
        summary.config_hash for summary in first
    ]


def test_catalog_lexical_weights_are_executable_and_part_of_identity() -> None:
    original = _catalog()
    changed = build_search_catalog(
        _manifest(),
        result_k=3,
        lexical=LexicalSpec(title_weight=3.0, body_weight=0.5),
    )

    assert changed.configs[0].lexical_fields == (("title", 3.0), ("body", 0.5))
    assert changed.summaries()[1] == original.summaries()[1]
    for index in (0, 2, 3):
        assert changed.summaries()[index].id != original.summaries()[index].id
        assert changed.summaries()[index].config_hash != original.summaries()[index].config_hash

    body_only = build_search_catalog(
        _manifest(),
        result_k=3,
        lexical=LexicalSpec(title_weight=0.0, body_weight=1.0),
    )
    assert body_only.configs[0].lexical_fields == (("body", 1.0),)

    with pytest.raises(ValueError, match="at least one lexical weight"):
        build_search_catalog(
            _manifest(),
            result_k=3,
            lexical=LexicalSpec(title_weight=0.0, body_weight=0.0),
        )

    body_fts_only = _manifest().model_copy(
        update={
            "fts": _manifest().fts.model_copy(update={"attributes": ["body"]}),
        }
    )
    with pytest.raises(ValueError, match="title"):
        build_search_catalog(body_fts_only, result_k=3)
    assert build_search_catalog(
        body_fts_only,
        result_k=3,
        lexical=LexicalSpec(title_weight=0.0, body_weight=1.0),
    ).configs[0].lexical_fields == (("body", 1.0),)


@pytest.mark.asyncio
async def test_compare_preserves_config_order_and_reports_observed_evidence() -> None:
    service, provider, embedder, _ = _service()
    bm25, vector, _, _ = service.list_configs()
    filter_override = FilterPredicate(field="external_id", op=PredicateOp.NOT_EQ, value="ignore")

    response = await service.compare(
        SearchCompareRequest(
            query_text="find terminal basics",
            config_ids=[bm25.id, vector.id],
            filter_override=filter_override,
            expected_document_ids=[DOCUMENT_B],
        )
    )

    assert [result.config.id for result in response.results] == [bm25.id, vector.id]
    assert [hit.external_id for hit in response.results[0].hits] == ["a", "b"]
    assert [hit.final_rank for hit in response.results[0].hits] == [1, 2]
    assert response.results[0].hits[0].body_excerpt == "Body a"
    assert response.results[0].hits[0].url == "https://example.test/a"
    assert response.results[0].hits[1].relevance_grade == 1
    assert response.results[0].hits[0].relevance_grade == 0
    assert response.results[0].hits[0].stage_membership[0].stage is (RetrievalStage.BM25_CANDIDATES)
    assert response.results[1].hits[0].stage_membership[0].stage is (
        RetrievalStage.VECTOR_CANDIDATES
    )
    assert response.results[0].hits[0].final_score is not None
    assert response.results[0].hits[0].final_score.direction is (ScoreDirection.HIGHER_IS_BETTER)
    assert response.results[1].hits[0].final_score is not None
    assert response.results[1].hits[0].final_score.direction is ScoreDirection.LOWER_IS_BETTER
    assert response.results[0].timings[0].stage is TimingStage.TURBOPUFFER
    assert response.results[0].timings[0].duration_ms == 2.25
    assert [(timing.stage, timing.duration_ms) for timing in response.results[1].timings] == [
        (TimingStage.EMBED, 1.75),
        (TimingStage.TURBOPUFFER, 3.5),
    ]
    assert response.results[0].candidate_counts == {"bm25_candidates": 2}
    assert response.results[1].candidate_counts == {"vector_candidates": 2}
    assert response.overlap[0].intersection_count == 1
    assert response.overlap[0].jaccard == pytest.approx(1 / 3)
    movement_by_document = {movement.document_id: movement for movement in response.rank_movements}
    assert movement_by_document[DOCUMENT_B].ranks_by_config == {bm25.id: 2, vector.id: 1}
    assert movement_by_document[DOCUMENT_B].max_absolute_delta == 1
    assert movement_by_document[DOCUMENT_A].max_absolute_delta is None
    assert response.results[0].trace_id == TRACE_ID == response.results[1].trace_id

    assert provider.bm25_calls == [
        {
            "namespace": "pufferlab-test",
            "lexical_fields": (("title", 2.0), ("body", 1.0)),
            "query_text": "find terminal basics",
            "top_k": 3,
            "include_attributes": ("external_id", "title", "body", "source_url"),
            "filters": filter_override,
            "consistency": "strong",
            "vector_attributes": ("vector",),
        }
    ]
    assert provider.ann_calls[0]["query_vector"] == (0.125, -0.25, 0.5)
    assert provider.ann_calls[0]["distance_metric"] == "cosine_distance"
    assert provider.ann_calls[0]["filters"] == filter_override
    assert embedder.queries == ["find terminal basics"]

    serialized = json.dumps(response.model_dump(mode="json"))
    assert "query_vector" not in serialized
    assert "[0.125, -0.25, 0.5]" not in serialized


@pytest.mark.asyncio
async def test_hybrid_modes_separate_server_probe_fusion_and_reranker_evidence() -> None:
    service, provider, embedder, reranker = _service()
    _, _, hybrid, hybrid_rerank = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="find shell permissions",
            config_ids=[hybrid.id, hybrid_rerank.id],
            expected_document_ids=[DOCUMENT_A],
            debug_provenance=True,
        )
    )

    hybrid_result, reranked_result = response.results
    assert [hit.external_id for hit in hybrid_result.hits] == ["b", "a", "c"]
    assert [hit.external_id for hit in reranked_result.hits] == ["a", "c", "b"]
    assert [membership.stage for membership in hybrid_result.hits[0].stage_membership] == [
        RetrievalStage.BM25_CANDIDATES,
        RetrievalStage.VECTOR_CANDIDATES,
        RetrievalStage.RRF,
        RetrievalStage.FINAL,
    ]
    assert hybrid_result.hits[0].stage_membership[2].score is not None
    assert hybrid_result.hits[0].stage_membership[2].score.source is ScoreSource.CLIENT_COMPUTED
    assert hybrid_result.hits[0].final_score is not None
    assert hybrid_result.hits[0].final_score.source is ScoreSource.TURBOPUFFER_DIST
    assert [membership.stage for membership in reranked_result.hits[0].stage_membership] == [
        RetrievalStage.BM25_CANDIDATES,
        RetrievalStage.RRF,
        RetrievalStage.RERANKER,
        RetrievalStage.FINAL,
    ]
    assert reranked_result.hits[0].final_score is not None
    assert reranked_result.hits[0].final_score.kind is ScoreKind.RERANKER
    assert reranked_result.hits[0].final_score.source is ScoreSource.RERANKER
    assert reranked_result.hits[0].relevance_grade == 1
    assert [(timing.stage, timing.duration_ms) for timing in hybrid_result.timings[:3]] == [
        (TimingStage.EMBED, 1.75),
        (TimingStage.TURBOPUFFER, 4.5),
        (TimingStage.PROVENANCE_PROBE, 2.0),
    ]
    assert hybrid_result.timings[3].stage is TimingStage.FUSION
    assert [timing.stage for timing in reranked_result.timings] == [
        TimingStage.EMBED,
        TimingStage.TURBOPUFFER,
        TimingStage.PROVENANCE_PROBE,
        TimingStage.FUSION,
        TimingStage.RERANK,
    ]
    assert reranked_result.timings[-1].duration_ms == 6.25
    assert hybrid_result.candidate_counts == {
        "rrf": 3,
        "bm25_candidates": 2,
        "vector_candidates": 2,
    }
    assert reranked_result.candidate_counts == {
        "rrf": 3,
        "bm25_candidates": 2,
        "vector_candidates": 2,
        "reranker": 3,
    }

    assert len(provider.hybrid_calls) == 2
    assert [call["result_k"] for call in provider.hybrid_calls] == [3, 50]
    assert all(call["candidate_k"] == 100 for call in provider.hybrid_calls)
    assert all(call["rank_constant"] == 60 for call in provider.hybrid_calls)
    assert all(call["weights"] == (1.0, 1.0) for call in provider.hybrid_calls)
    assert len(provider.probe_calls) == 2
    assert embedder.queries == ["find shell permissions", "find shell permissions"]
    assert len(reranker.calls) == 1
    assert [candidate.document_id for candidate in reranker.calls[0][1]] == [
        str(DOCUMENT_B),
        str(DOCUMENT_A),
        str(DOCUMENT_C),
    ]
    assert all(
        not hasattr(candidate, "vector") and candidate.body.startswith("Body")
        for candidate in reranker.calls[0][1]
    )
    serialized = json.dumps(response.model_dump(mode="json"))
    assert "query_vector" not in serialized
    assert "rationale" not in serialized


@pytest.mark.asyncio
async def test_reranker_receives_only_configured_depth() -> None:
    provider = FakeProvider()
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    catalog = build_search_catalog(
        _manifest(),
        result_k=2,
        candidate_k=3,
        reranker_depth=2,
    )
    service = SearchCompareService(
        namespace="pufferlab-test",
        catalog=catalog,
        write_spec=compile_namespace_write_spec(_manifest()),
        provider=provider,
        query_embedder=embedder,
        reranker=reranker,
    )
    bm25, _, _, hybrid_rerank = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[bm25.id, hybrid_rerank.id],
            debug_provenance=False,
        )
    )

    assert provider.hybrid_calls[0]["candidate_k"] == 3
    assert provider.hybrid_calls[0]["result_k"] == 2
    assert provider.probe_calls == []
    assert [candidate.document_id for candidate in reranker.calls[0][1]] == [
        str(DOCUMENT_B),
        str(DOCUMENT_A),
    ]
    assert [hit.external_id for hit in response.results[1].hits] == ["a", "b"]
    assert [timing.stage for timing in response.results[1].timings] == [
        TimingStage.EMBED,
        TimingStage.TURBOPUFFER,
        TimingStage.RERANK,
    ]


class SecretReranker(FakeReranker):
    async def rerank(
        self,
        *,
        query_text: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> RerankResult:
        del query_text, candidates
        raise RuntimeError("reranker-secret-value")


@pytest.mark.asyncio
async def test_reranker_failures_are_detached_and_redacted() -> None:
    service, _, _, _ = _service(reranker=SecretReranker())
    bm25, _, _, hybrid_rerank = service.list_configs()

    with pytest.raises(SearchError) as caught:
        await service.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[bm25.id, hybrid_rerank.id],
                debug_provenance=False,
            )
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "reranker-secret-value" not in repr(caught.value)
    assert "reranker-secret-value" not in formatted


class FailingProbeProvider(FakeProvider):
    async def probe_hybrid_candidates(self, **kwargs: object) -> ProviderHybridProbeResult:
        del kwargs
        raise RuntimeError("debug-probe-secret")


@pytest.mark.asyncio
async def test_optional_provenance_probe_failure_keeps_production_result_contract_valid() -> None:
    service, _, _, _ = _service(FailingProbeProvider())
    bm25, _, hybrid, _ = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[bm25.id, hybrid.id],
            debug_provenance=True,
        )
    )

    hybrid_result = response.results[1]
    assert [hit.external_id for hit in hybrid_result.hits] == ["b", "a", "c"]
    assert [warning.code for warning in hybrid_result.warnings] == ["provenance_probe_failed"]
    assert [timing.stage for timing in hybrid_result.timings] == [
        TimingStage.EMBED,
        TimingStage.TURBOPUFFER,
    ]
    assert all(
        membership.stage
        not in {
            RetrievalStage.BM25_CANDIDATES,
            RetrievalStage.VECTOR_CANDIDATES,
        }
        for hit in hybrid_result.hits
        for membership in hit.stage_membership
    )
    assert "debug-probe-secret" not in json.dumps(response.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_separate_probe_order_difference_is_explicitly_warned() -> None:
    provider = FakeProvider()
    provider.hybrid_result = ProviderQueryResult(
        documents=(
            provider.hybrid_result.documents[1],
            provider.hybrid_result.documents[0],
            provider.hybrid_result.documents[2],
        ),
        client_duration_ms=4.5,
    )
    service, _, _, _ = _service(provider)
    bm25, _, hybrid, _ = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[bm25.id, hybrid.id],
            debug_provenance=True,
        )
    )

    hybrid_result = response.results[1]
    assert [warning.code for warning in hybrid_result.warnings] == ["provenance_snapshot_differs"]
    assert hybrid_result.hits[0].final_rank == 1
    rrf_membership = next(
        membership
        for membership in hybrid_result.hits[0].stage_membership
        if membership.stage is RetrievalStage.RRF
    )
    assert rrf_membership.rank == 2
    assert rrf_membership.score is not None
    assert rrf_membership.score.source is ScoreSource.CLIENT_COMPUTED


@pytest.mark.asyncio
async def test_compare_rejects_duplicate_or_unknown_configs() -> None:
    service, _, _, _ = _service()
    bm25, vector, _, _ = service.list_configs()

    with pytest.raises(SearchError, match="distinct") as duplicate:
        await service.compare(
            SearchCompareRequest(query_text="query", config_ids=[bm25.id, bm25.id])
        )
    assert duplicate.value.details.http_status == 422

    with pytest.raises(SearchError, match="not found") as missing:
        await service.compare(
            SearchCompareRequest(query_text="query", config_ids=[bm25.id, uuid4()])
        )
    assert missing.value.details.http_status == 404
    assert vector.id != bm25.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_override",
    [
        FilterPredicate.model_construct(
            field="external_id", op=PredicateOp.IN, value="not-an-array"
        ),
        FilterPredicate(field="unknown-secret-field", op=PredicateOp.EQ, value="value"),
        FilterPredicate(field="title", op=PredicateOp.EQ, value="value"),
        FilterPredicate(field="external_id", op=PredicateOp.CONTAINS_ANY, value=["value"]),
        FilterPredicate(field="external_id", op=PredicateOp.IN, value=[1]),
        FilterPredicate(field="external_id", op=PredicateOp.EQ, value=5),
        FilterPredicate(field="external_id", op=PredicateOp.LTE, value=True),
        FilterPredicate.model_construct(field="external_id", op=PredicateOp.EQ, value=["value"]),
        FilterPredicate.model_construct(
            field="external_id", op=PredicateOp.EQ, value={"nested": "value"}
        ),
        FilterLogical(
            op=LogicalOp.AND,
            children=[
                FilterPredicate(field="external_id", op=PredicateOp.EQ, value="allowed"),
                FilterLogical(
                    op=LogicalOp.OR,
                    children=[
                        FilterPredicate(field="body", op=PredicateOp.EQ, value="blocked"),
                    ],
                ),
            ],
        ),
    ],
)
async def test_invalid_filters_are_rejected_before_embedding_or_provider_calls(
    filter_override: FilterPredicate | FilterLogical,
) -> None:
    service, provider, embedder, _ = _service()
    summaries = service.list_configs()

    with pytest.raises(SearchError) as caught:
        await service.compare(
            SearchCompareRequest.model_construct(
                query_text="query",
                config_ids=[summary.id for summary in summaries],
                query_id=None,
                filter_override=filter_override,
                expected_document_ids=[],
                debug_provenance=True,
            )
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.details.http_status == 422
    assert caught.value.details.code.value == "validation_error"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "unknown-secret-field" not in repr(caught.value)
    assert "unknown-secret-field" not in formatted
    assert provider.bm25_calls == []
    assert provider.ann_calls == []
    assert embedder.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
async def test_nonfinite_nested_filter_values_are_rejected_before_dependencies(
    nonfinite: float,
) -> None:
    service, provider, embedder, _ = _service()
    summaries = service.list_configs()
    bypassed_contract = FilterPredicate.model_construct(
        field="external_id",
        op=PredicateOp.IN,
        value=["valid", {"nested": [nonfinite]}],
    )
    request = SearchCompareRequest.model_construct(
        query_text="query",
        config_ids=[summary.id for summary in summaries],
        query_id=None,
        filter_override=bypassed_contract,
        expected_document_ids=[],
        debug_provenance=True,
    )

    with pytest.raises(SearchError, match="finite JSON") as caught:
        await service.compare(request)

    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert provider.bm25_calls == []
    assert provider.ann_calls == []
    assert embedder.queries == []


@pytest.mark.asyncio
async def test_valid_nested_filter_is_forwarded_unchanged_to_both_queries() -> None:
    service, provider, embedder, _ = _service()
    summaries = service.list_configs()
    filter_override = FilterLogical(
        op=LogicalOp.AND,
        children=[
            FilterPredicate(field="external_id", op=PredicateOp.IN, value=["a", "b"]),
            FilterLogical(
                op=LogicalOp.NOT,
                children=[
                    FilterPredicate(
                        field="external_id",
                        op=PredicateOp.GTE,
                        value="z",
                    )
                ],
            ),
        ],
    )

    await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[summary.id for summary in summaries],
            filter_override=filter_override,
        )
    )

    assert provider.bm25_calls[0]["filters"] == filter_override
    assert provider.ann_calls[0]["filters"] == filter_override
    assert all(call["filters"] == filter_override for call in provider.hybrid_calls)
    assert all(call["filters"] == filter_override for call in provider.probe_calls)
    assert embedder.queries == ["query", "query", "query"]


class SecretProvider(FakeProvider):
    async def query_bm25(self, **kwargs: object) -> ProviderQueryResult:
        del kwargs
        raise RuntimeError("provider-secret-value")


class SecretEmbedder(FakeEmbedder):
    async def embed_query(self, query_text: str) -> QueryEmbedding:
        del query_text
        raise RuntimeError("embedder-secret-value")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "embedder", "config_index", "secret"),
    [
        (SecretProvider(), FakeEmbedder(), 0, "provider-secret-value"),
        (FakeProvider(), SecretEmbedder(), 1, "embedder-secret-value"),
    ],
)
async def test_unexpected_dependency_errors_are_detached_and_redacted(
    provider: FakeProvider,
    embedder: FakeEmbedder,
    config_index: int,
    secret: str,
) -> None:
    service, _, _, _ = _service(provider, embedder)
    summaries = service.list_configs()
    other_index = 1 - config_index

    with pytest.raises(SearchError) as caught:
        await service.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[summaries[config_index].id, summaries[other_index].id],
            )
        )

    error = caught.value
    formatted = "".join(traceback.format_exception(error))
    assert error.__context__ is None
    assert error.__cause__ is None
    assert secret not in repr(error)
    assert secret not in formatted


@pytest.mark.asyncio
async def test_invalid_provider_document_is_replaced_with_safe_error() -> None:
    provider = FakeProvider()
    provider.bm25_result = ProviderQueryResult(
        documents=(
            ProviderDocument(
                id="not-a-uuid",
                attributes={"external_id": "secret-ish-raw-payload"},
                score=_document(
                    DOCUMENT_A,
                    external_id="a",
                    score=1.0,
                    kind=ScoreKind.BM25,
                ).score,
            ),
        ),
        client_duration_ms=1.0,
    )
    service, _, _, _ = _service(provider)
    bm25, vector, _, _ = service.list_configs()

    with pytest.raises(SearchError, match="invalid result") as caught:
        await service.compare(
            SearchCompareRequest(query_text="query", config_ids=[bm25.id, vector.id])
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "secret-ish-raw-payload" not in formatted


@pytest.mark.asyncio
async def test_nonfinite_provider_evidence_is_rejected() -> None:
    provider = FakeProvider()
    provider.bm25_result = ProviderQueryResult(
        documents=provider.bm25_result.documents,
        client_duration_ms=float("inf"),
    )
    service, _, _, _ = _service(provider)
    bm25, vector, _, _ = service.list_configs()

    with pytest.raises(SearchError, match="invalid result"):
        await service.compare(
            SearchCompareRequest(query_text="query", config_ids=[bm25.id, vector.id])
        )


@pytest.mark.asyncio
async def test_hybrid_rejects_wrong_provider_score_semantics() -> None:
    provider = FakeProvider()
    provider.hybrid_result = ProviderQueryResult(
        documents=provider.bm25_result.documents,
        client_duration_ms=1.0,
    )
    service, _, _, _ = _service(provider)
    bm25, _, hybrid, _ = service.list_configs()

    with pytest.raises(SearchError, match="invalid result"):
        await service.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[bm25.id, hybrid.id],
                debug_provenance=False,
            )
        )


@pytest.mark.asyncio
async def test_empty_results_have_warnings_and_defined_overlap() -> None:
    provider = FakeProvider()
    empty = ProviderQueryResult(documents=(), client_duration_ms=0.5)
    provider.bm25_result = empty
    provider.ann_result = empty
    provider.hybrid_result = empty
    service, _, _, _ = _service(provider)
    summaries = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[summary.id for summary in summaries],
            debug_provenance=False,
        )
    )

    assert response.overlap[0].jaccard == 1.0
    assert [result.warnings[0].code for result in response.results] == ["empty_result"] * 4
    await service.close()
    assert provider.closed
