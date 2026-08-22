import json
import traceback
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import FilterPredicate, PredicateOp
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.contracts.search import RetrievalStage, SearchCompareRequest, TimingStage
from pufferlab.providers.types import ProviderDocument, ProviderQueryResult
from pufferlab.retrieval.config import SearchCatalogProfile, build_search_catalog
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.service import SearchCompareService
from pufferlab.retrieval.types import QueryEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TRACE_ID = UUID("0f7b308a-c5ac-4438-a1b2-6901002386fd")
DOCUMENT_A = UUID("0e0ee431-1a4b-5ca6-9e98-4033c8d44498")
DOCUMENT_B = UUID("307a0026-1cd9-58a5-8131-729c329fd068")
DOCUMENT_C = UUID("fd00f34b-9d23-5655-81f2-b05a1b9f4ea8")


def _profile() -> SearchCatalogProfile:
    return SearchCatalogProfile(
        dataset_slug="pufferlab-tiny-unix",
        dataset_version="tiny-unix-v1",
        namespace_schema_hash="fixture-schema-hash",
        text_attribute="body",
        vector_attribute="vector",
        embedding_model=MODEL,
        embedding_revision=REVISION,
        embedding_dimensions=3,
        distance_metric="cosine_distance",
    )


def _catalog():
    return build_search_catalog(_profile(), result_k=3)


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
        self.bm25_calls: list[dict[str, object]] = []
        self.ann_calls: list[dict[str, object]] = []
        self.closed = False

    async def query_bm25(self, **kwargs: object) -> ProviderQueryResult:
        self.bm25_calls.append(kwargs)
        return self.bm25_result

    async def query_ann(self, **kwargs: object) -> ProviderQueryResult:
        self.ann_calls.append(kwargs)
        return self.ann_result

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


def _service(
    provider: FakeProvider | None = None,
    embedder: FakeEmbedder | None = None,
) -> tuple[SearchCompareService, FakeProvider, FakeEmbedder]:
    resolved_provider = provider or FakeProvider()
    resolved_embedder = embedder or FakeEmbedder()
    service = SearchCompareService(
        namespace="pufferlab-test",
        catalog=_catalog(),
        provider=resolved_provider,
        query_embedder=resolved_embedder,
        trace_id_factory=lambda: TRACE_ID,
    )
    return service, resolved_provider, resolved_embedder


def test_catalog_is_deterministic_and_describes_bm25_and_vector() -> None:
    first = _catalog().summaries()
    second = _catalog().summaries()

    assert first == second
    assert [summary.mode for summary in first] == [RetrievalMode.BM25, RetrievalMode.VECTOR]
    assert len({summary.id for summary in first}) == 2
    assert all(len(summary.config_hash) == 64 for summary in first)

    changed_schema = build_search_catalog(
        replace(_profile(), namespace_schema_hash="changed-schema-hash"), result_k=3
    ).summaries()
    assert [summary.config_hash for summary in changed_schema] != [
        summary.config_hash for summary in first
    ]


@pytest.mark.asyncio
async def test_compare_preserves_config_order_and_reports_observed_evidence() -> None:
    service, provider, embedder = _service()
    bm25, vector = service.list_configs()
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
            "text_attribute": "body",
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
async def test_compare_rejects_duplicate_or_unknown_configs() -> None:
    service, _, _ = _service()
    bm25, vector = service.list_configs()

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
    service, _, _ = _service(provider, embedder)
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
    service, _, _ = _service(provider)
    bm25, vector = service.list_configs()

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
    service, _, _ = _service(provider)
    bm25, vector = service.list_configs()

    with pytest.raises(SearchError, match="invalid result"):
        await service.compare(
            SearchCompareRequest(query_text="query", config_ids=[bm25.id, vector.id])
        )


@pytest.mark.asyncio
async def test_empty_results_have_warnings_and_defined_overlap() -> None:
    provider = FakeProvider()
    empty = ProviderQueryResult(documents=(), client_duration_ms=0.5)
    provider.bm25_result = empty
    provider.ann_result = empty
    service, _, _ = _service(provider)
    summaries = service.list_configs()

    response = await service.compare(
        SearchCompareRequest(
            query_text="query",
            config_ids=[summary.id for summary in summaries],
            debug_provenance=False,
        )
    )

    assert response.overlap[0].jaccard == 1.0
    assert [result.warnings[0].code for result in response.results] == [
        "empty_result",
        "empty_result",
    ]
    await service.close()
    assert provider.closed
