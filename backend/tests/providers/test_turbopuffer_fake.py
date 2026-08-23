from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
from pufferlab.contracts.common import ScoreDirection, ScoreKind, ScoreSource
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.filters import FilterLogical, FilterPredicate, LogicalOp, PredicateOp
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider, filter_to_turbopuffer
from pufferlab.providers.types import ProviderSchema, WriteDocument
from turbopuffer import (
    APIError,
    APIResponseValidationError,
    AsyncTurbopuffer,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)


@dataclass
class FakeIndex:
    status: str = "up-to-date"
    unindexed_bytes: int | None = None


@dataclass
class FakeMetadata:
    approx_row_count: int
    index: FakeIndex
    schema_: dict[str, object]


@dataclass
class FakeResponse:
    rows: list[dict[str, object]] | None = None
    rows_affected: int = 0
    aggregations: dict[str, object] | None = None


@dataclass
class FakeMultiResult:
    rows: list[dict[str, object]] | None = None


@dataclass
class FakeMultiResponse:
    results: list[FakeMultiResult]


class FakeNamespace:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.query_response = FakeResponse()
        self.query_responses: list[FakeResponse] = []
        self.multi_query_response = FakeMultiResponse(results=[])
        self.write_response = FakeResponse(rows_affected=0)
        self.metadata_response = FakeMetadata(
            approx_row_count=0,
            index=FakeIndex(),
            schema_={},
        )
        self.write_error: APIError | None = None
        self.query_error: APIError | None = None
        self.multi_query_error: APIError | None = None
        self.metadata_error: APIError | None = None
        self.delete_error: APIError | None = None

    async def write(self, **kwargs: object) -> object:
        self.calls.append(("write", kwargs))
        if self.write_error is not None:
            raise self.write_error
        return self.write_response

    async def query(self, **kwargs: object) -> object:
        self.calls.append(("query", kwargs))
        if self.query_error is not None:
            raise self.query_error
        if self.query_responses:
            return self.query_responses.pop(0)
        return self.query_response

    async def multi_query(self, **kwargs: object) -> object:
        self.calls.append(("multi_query", kwargs))
        if self.multi_query_error is not None:
            raise self.multi_query_error
        return self.multi_query_response

    async def metadata(self, **kwargs: object) -> object:
        self.calls.append(("metadata", kwargs))
        if self.metadata_error is not None:
            raise self.metadata_error
        return self.metadata_response

    async def delete_all(self, **kwargs: object) -> object:
        self.calls.append(("delete_all", kwargs))
        if self.delete_error is not None:
            raise self.delete_error
        return object()


class FakeClient:
    def __init__(self, namespace: FakeNamespace) -> None:
        self.fake_namespace = namespace
        self.namespace_calls: list[str] = []
        self.close_calls = 0
        self.close_error: APIError | None = None

    def namespace(self, namespace: str) -> FakeNamespace:
        self.namespace_calls.append(namespace)
        return self.fake_namespace

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def clock(*values: float) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


def unit_vector(dimensions: int, hot_dimension: int) -> list[float]:
    return [float(position == hot_dimension) for position in range(dimensions)]


def inventory_rows(start: int, stop: int) -> list[dict[str, object]]:
    return [{"id": f"doc-{value:05d}"} for value in range(start, stop)]


def make_provider(
    namespace: FakeNamespace,
    *,
    timer: Callable[[], float] | None = None,
) -> tuple[TurbopufferProvider, FakeClient]:
    client = FakeClient(namespace)
    provider = TurbopufferProvider(
        api_key="not-a-real-key",
        region="gcp-us-central1",
        client=client,
        clock=timer or clock(1.0, 1.001),
    )
    return provider, client


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (PredicateOp.EQ, ("source", "Eq", "unix")),
        (PredicateOp.NOT_EQ, ("source", "NotEq", "unix")),
        (PredicateOp.LT, ("source", "Lt", 10)),
        (PredicateOp.LTE, ("source", "Lte", 10)),
        (PredicateOp.GT, ("source", "Gt", 10)),
        (PredicateOp.GTE, ("source", "Gte", 10)),
        (PredicateOp.IN, ("source", "In", ["unix", "stats"])),
        (PredicateOp.CONTAINS_ANY, ("source", "ContainsAny", ["unix", "stats"])),
    ],
)
def test_filter_predicates_convert_to_sdk_tuples(
    operation: PredicateOp,
    expected: tuple[object, ...],
) -> None:
    value: object = (
        ["unix", "stats"] if operation in {PredicateOp.IN, PredicateOp.CONTAINS_ANY} else 10
    )
    if operation in {PredicateOp.EQ, PredicateOp.NOT_EQ}:
        value = "unix"
    predicate = FilterPredicate.model_validate(
        {"field": "source", "op": operation.value, "value": value}
    )

    assert filter_to_turbopuffer(predicate) == expected


def test_logical_filters_preserve_nesting_and_not_arity() -> None:
    source = FilterPredicate(field="source", op=PredicateOp.EQ, value="unix")
    public = FilterPredicate(field="public", op=PredicateOp.EQ, value=True)
    filter_node = FilterLogical(
        op=LogicalOp.AND,
        children=[source, FilterLogical(op=LogicalOp.NOT, children=[public])],
    )

    assert filter_to_turbopuffer(filter_node) == (
        "And",
        (("source", "Eq", "unix"), ("Not", ("public", "Eq", True))),
    )


@pytest.mark.parametrize("operation", [PredicateOp.IN, PredicateOp.CONTAINS_ANY])
def test_collection_filter_rejects_scalar_values(operation: PredicateOp) -> None:
    predicate = FilterPredicate.model_construct(field="source", op=operation, value="unix")

    with pytest.raises(ValueError, match="array value"):
        filter_to_turbopuffer(predicate)


@pytest.mark.asyncio
async def test_write_always_sends_explicit_schema_and_complete_rows() -> None:
    namespace = FakeNamespace()
    namespace.write_response = FakeResponse(rows_affected=2)
    provider, _ = make_provider(namespace, timer=clock(10.0, 10.0025))
    schema = cast(
        ProviderSchema,
        {
            "text": {
                "type": "string",
                "full_text_search": {"k1": 1.2, "b": 0.75, "k3": 8.0},
            },
            "vector": {"type": "[2]f32", "ann": True},
        },
    )
    first_vector = unit_vector(2, 0)
    second_vector = unit_vector(2, 1)

    result = await provider.write_documents(
        namespace="fixture",
        documents=(
            WriteDocument(id="one", attributes={"text": "first", "vector": first_vector}),
            WriteDocument(id="two", attributes={"text": "second", "vector": second_vector}),
        ),
        schema=schema,
        distance_metric="cosine_distance",
    )

    assert result.rows_affected == 2
    assert result.client_duration_ms == pytest.approx(2.5)
    assert namespace.calls == [
        (
            "write",
            {
                "upsert_rows": [
                    {"id": "one", "text": "first", "vector": first_vector},
                    {"id": "two", "text": "second", "vector": second_vector},
                ],
                "schema": {
                    "text": {
                        "type": "string",
                        "full_text_search": {"k1": 1.2, "b": 0.75, "k3": 8.0},
                    },
                    "vector": {"type": "[2]f32", "ann": True},
                },
                "distance_metric": "cosine_distance",
            },
        )
    ]


@pytest.mark.asyncio
async def test_bm25_query_shape_and_score_semantics() -> None:
    namespace = FakeNamespace()
    hidden_vector = unit_vector(2, 0)
    namespace.query_response = FakeResponse(
        rows=[
            {
                "id": "doc-1",
                "$dist": 3.75,
                "title": "Pufferfish",
                "published_at": datetime(2026, 8, 22, tzinfo=UTC),
                "vector": hidden_vector,
            }
        ]
    )
    provider, _ = make_provider(namespace, timer=clock(5.0, 5.012))
    filter_node = FilterPredicate(field="source", op=PredicateOp.EQ, value="unix")

    result = await provider.query_bm25(
        namespace="fixture",
        lexical_fields=(("title", 2.0), ("body", 1.0)),
        query_text="pufferfish",
        top_k=5,
        include_attributes=("title", "published_at"),
        filters=filter_node,
    )

    assert namespace.calls == [
        (
            "query",
            {
                "rank_by": (
                    "Sum",
                    [
                        ("Product", 2.0, ("title", "BM25", "pufferfish")),
                        ("Product", 1.0, ("body", "BM25", "pufferfish")),
                    ],
                ),
                "top_k": 5,
                "include_attributes": ["title", "published_at"],
                "consistency": {"level": "strong"},
                "filters": ("source", "Eq", "unix"),
            },
        )
    ]
    assert result.client_duration_ms == pytest.approx(12.0)
    assert result.documents[0].attributes == {
        "title": "Pufferfish",
        "published_at": "2026-08-22T00:00:00+00:00",
    }
    score = result.documents[0].score
    assert score.kind is ScoreKind.BM25
    assert score.direction is ScoreDirection.HIGHER_IS_BETTER
    assert score.source is ScoreSource.TURBOPUFFER_DIST
    assert score.value == 3.75


@pytest.mark.asyncio
async def test_weighted_bm25_shape_serializes_through_real_sdk() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads((await request.aread()).decode()))
        return httpx.Response(200, json={"rows": []}, request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://sdk.test",
    )
    sdk = AsyncTurbopuffer(
        api_key="not-a-real-key",
        base_url="https://sdk.test",
        http_client=http_client,
        max_retries=0,
    )
    provider = TurbopufferProvider(
        api_key="not-a-real-key",
        region="gcp-us-central1",
        client=sdk,
    )

    try:
        await provider.query_bm25(
            namespace="fixture",
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="pufferfish",
            top_k=5,
            include_attributes=("title",),
        )
    finally:
        await provider.close()

    assert captured["rank_by"] == [
        "Sum",
        [
            ["Product", 2.0, ["title", "BM25", "pufferfish"]],
            ["Product", 1.0, ["body", "BM25", "pufferfish"]],
        ],
    ]


@pytest.mark.asyncio
async def test_ann_query_shape_and_distance_score_semantics() -> None:
    namespace = FakeNamespace()
    query_vector = unit_vector(2, 1)
    namespace.query_response = FakeResponse(
        rows=[{"id": "doc-2", "$dist": 0.125, "title": "Nearest", "embedding": query_vector}]
    )
    provider, _ = make_provider(namespace)

    result = await provider.query_ann(
        namespace="fixture",
        vector_attribute="embedding",
        query_vector=query_vector,
        top_k=3,
        include_attributes=("title",),
        consistency="eventual",
        distance_metric="cosine_distance",
    )

    assert namespace.calls == [
        (
            "query",
            {
                "rank_by": ("embedding", "ANN", query_vector),
                "top_k": 3,
                "include_attributes": ["title"],
                "consistency": {"level": "eventual"},
                "distance_metric": "cosine_distance",
            },
        )
    ]
    assert result.documents[0].attributes == {"title": "Nearest"}
    score = result.documents[0].score
    assert score.kind is ScoreKind.VECTOR_DISTANCE
    assert score.direction is ScoreDirection.LOWER_IS_BETTER
    assert score.value == 0.125


@pytest.mark.asyncio
async def test_hybrid_rrf_uses_one_weighted_same_snapshot_multi_query() -> None:
    namespace = FakeNamespace()
    query_vector = unit_vector(2, 1)
    namespace.multi_query_response = FakeMultiResponse(
        results=[
            FakeMultiResult(
                rows=[
                    {
                        "id": "doc-2",
                        "$dist": 0.0315,
                        "title": "Fused",
                        "vector": query_vector,
                    }
                ]
            )
        ]
    )
    provider, _ = make_provider(namespace, timer=clock(2.0, 2.007))
    filter_node = FilterPredicate(field="source", op=PredicateOp.EQ, value="unix")

    result = await provider.query_hybrid_rrf(
        namespace="fixture",
        lexical_fields=(("title", 2.0), ("body", 1.0)),
        query_text="shell pipe",
        vector_attribute="vector",
        query_vector=query_vector,
        candidate_k=50,
        result_k=10,
        include_attributes=("title",),
        rank_constant=60,
        weights=(1.5, 0.75),
        filters=filter_node,
        consistency="strong",
        distance_metric="cosine_distance",
    )

    assert namespace.calls == [
        (
            "multi_query",
            {
                "queries": [
                    {
                        "limit": 50,
                        "include_attributes": ["title"],
                        "filters": ("source", "Eq", "unix"),
                        "rank_by": (
                            "Sum",
                            [
                                ("Product", 2.0, ("title", "BM25", "shell pipe")),
                                ("Product", 1.0, ("body", "BM25", "shell pipe")),
                            ],
                        ),
                    },
                    {
                        "limit": 50,
                        "include_attributes": ["title"],
                        "filters": ("source", "Eq", "unix"),
                        "rank_by": ("vector", "ANN", query_vector),
                        "distance_metric": "cosine_distance",
                    },
                ],
                "consistency": {"level": "strong"},
                "limit": {"total": 10},
                "rerank_by": (
                    "RRF",
                    {"rank_constant": 60, "weights": [1.5, 0.75]},
                ),
            },
        )
    ]
    assert result.client_duration_ms == pytest.approx(7.0)
    assert result.documents[0].attributes == {"title": "Fused"}
    score = result.documents[0].score
    assert score.kind is ScoreKind.RRF
    assert score.direction is ScoreDirection.HIGHER_IS_BETTER
    assert score.source is ScoreSource.TURBOPUFFER_DIST
    assert score.value == 0.0315


@pytest.mark.asyncio
async def test_debug_hybrid_probe_is_separate_and_preserves_raw_score_semantics() -> None:
    namespace = FakeNamespace()
    query_vector = unit_vector(2, 0)
    namespace.multi_query_response = FakeMultiResponse(
        results=[
            FakeMultiResult(rows=[{"id": "lexical", "$dist": 8.0, "title": "Lexical"}]),
            FakeMultiResult(
                rows=[
                    {
                        "id": "semantic",
                        "$dist": 0.125,
                        "title": "Semantic",
                        "vector": query_vector,
                    }
                ]
            ),
        ]
    )
    provider, _ = make_provider(namespace, timer=clock(4.0, 4.003))

    result = await provider.probe_hybrid_candidates(
        namespace="fixture",
        lexical_fields=(("title", 2.0), ("body", 1.0)),
        query_text="chmod command",
        vector_attribute="vector",
        query_vector=query_vector,
        candidate_k=25,
        include_attributes=("title",),
        consistency="eventual",
        distance_metric="cosine_distance",
    )

    assert namespace.calls == [
        (
            "multi_query",
            {
                "queries": [
                    {
                        "limit": 25,
                        "include_attributes": ["title"],
                        "rank_by": (
                            "Sum",
                            [
                                ("Product", 2.0, ("title", "BM25", "chmod command")),
                                ("Product", 1.0, ("body", "BM25", "chmod command")),
                            ],
                        ),
                    },
                    {
                        "limit": 25,
                        "include_attributes": ["title"],
                        "rank_by": ("vector", "ANN", query_vector),
                        "distance_metric": "cosine_distance",
                    },
                ],
                "consistency": {"level": "eventual"},
            },
        )
    ]
    assert result.client_duration_ms == pytest.approx(3.0)
    assert result.bm25_documents[0].score.kind is ScoreKind.BM25
    assert result.ann_documents[0].score.kind is ScoreKind.VECTOR_DISTANCE


@pytest.mark.asyncio
@pytest.mark.parametrize("result_count", [0, 2])
async def test_hybrid_rrf_rejects_unexpected_server_result_shapes(result_count: int) -> None:
    namespace = FakeNamespace()
    namespace.multi_query_response = FakeMultiResponse(
        results=[FakeMultiResult(rows=[]) for _ in range(result_count)]
    )
    provider, _ = make_provider(namespace)

    with pytest.raises(ValueError, match="unexpected result count"):
        await provider.query_hybrid_rrf(
            namespace="fixture",
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="query",
            vector_attribute="vector",
            query_vector=unit_vector(2, 0),
            candidate_k=10,
            result_k=5,
            include_attributes=(),
            rank_constant=60,
            weights=(1.0, 1.0),
        )


@pytest.mark.asyncio
async def test_document_id_inventory_is_strong_ordered_and_limit_safe() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=[{"id": "doc-1"}, {"id": "doc-2"}]),
        FakeResponse(aggregations={"count": 2}),
    ]
    provider, _ = make_provider(namespace, timer=clock(7.0, 7.003))

    inventory = await provider.namespace_document_ids("fixture", max_documents=20)

    assert inventory.document_ids == ("doc-1", "doc-2")
    assert inventory.document_count == 2
    assert not inventory.truncated
    assert inventory.client_duration_ms == pytest.approx(3.0)
    assert namespace.calls == [
        (
            "query",
            {
                "rank_by": ("id", "asc"),
                "top_k": 21,
                "include_attributes": [],
                "consistency": {"level": "strong"},
            },
        ),
        (
            "query",
            {
                "aggregate_by": {"count": ("Count",)},
                "consistency": {"level": "strong"},
            },
        ),
    ]


@pytest.mark.asyncio
async def test_document_id_inventory_detects_more_than_expected_limit() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=[{"id": "doc-1"}, {"id": "doc-2"}, {"id": "doc-3"}]),
        FakeResponse(aggregations={"count": 40}),
    ]
    provider, _ = make_provider(namespace)

    inventory = await provider.namespace_document_ids("fixture", max_documents=2)

    assert inventory.document_ids == ("doc-1", "doc-2", "doc-3")
    assert inventory.document_count == 40
    assert inventory.truncated


@pytest.mark.asyncio
async def test_document_id_inventory_paginates_more_than_ten_thousand_exact_ids() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=inventory_rows(0, 10_000)),
        FakeResponse(rows=inventory_rows(10_000, 20_000)),
        FakeResponse(rows=inventory_rows(20_000, 30_000)),
        FakeResponse(rows=inventory_rows(30_000, 40_000)),
        FakeResponse(rows=inventory_rows(40_000, 47_382)),
        FakeResponse(aggregations={"count": 47_382}),
    ]
    provider, _ = make_provider(namespace)

    inventory = await provider.namespace_document_ids("fixture", max_documents=47_382)

    assert inventory.document_ids == tuple(f"doc-{value:05d}" for value in range(47_382))
    assert inventory.document_count == 47_382
    assert not inventory.truncated
    id_queries = [kwargs for operation, kwargs in namespace.calls if "rank_by" in kwargs]
    assert id_queries == [
        {
            "rank_by": ("id", "asc"),
            "top_k": 10_000,
            "include_attributes": [],
            "consistency": {"level": "strong"},
        },
        {
            "rank_by": ("id", "asc"),
            "top_k": 10_000,
            "include_attributes": [],
            "consistency": {"level": "strong"},
            "filters": ("id", "Gt", "doc-09999"),
        },
        {
            "rank_by": ("id", "asc"),
            "top_k": 10_000,
            "include_attributes": [],
            "consistency": {"level": "strong"},
            "filters": ("id", "Gt", "doc-19999"),
        },
        {
            "rank_by": ("id", "asc"),
            "top_k": 10_000,
            "include_attributes": [],
            "consistency": {"level": "strong"},
            "filters": ("id", "Gt", "doc-29999"),
        },
        {
            "rank_by": ("id", "asc"),
            "top_k": 7_383,
            "include_attributes": [],
            "consistency": {"level": "strong"},
            "filters": ("id", "Gt", "doc-39999"),
        },
    ]
    assert namespace.calls[-1] == (
        "query",
        {
            "aggregate_by": {"count": ("Count",)},
            "consistency": {"level": "strong"},
        },
    )


@pytest.mark.asyncio
async def test_document_id_inventory_preserves_ordered_integer_ids() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=[{"id": 1}, {"id": 2}]),
        FakeResponse(aggregations={"count": 2}),
    ]
    provider, _ = make_provider(namespace)

    inventory = await provider.namespace_document_ids("fixture", max_documents=20)

    assert inventory.document_ids == (1, 2)
    assert not inventory.truncated


@pytest.mark.asyncio
async def test_document_id_inventory_collects_one_over_expected_across_page_boundary() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=inventory_rows(0, 10_000)),
        FakeResponse(rows=inventory_rows(10_000, 10_001)),
        FakeResponse(aggregations={"count": 10_001}),
    ]
    provider, _ = make_provider(namespace)

    inventory = await provider.namespace_document_ids("fixture", max_documents=10_000)

    assert len(inventory.document_ids) == 10_001
    assert inventory.document_ids[-1] == "doc-10000"
    assert inventory.document_count == 10_001
    assert inventory.truncated
    assert namespace.calls[1] == (
        "query",
        {
            "rank_by": ("id", "asc"),
            "top_k": 1,
            "include_attributes": [],
            "consistency": {"level": "strong"},
            "filters": ("id", "Gt", "doc-09999"),
        },
    )


@pytest.mark.asyncio
async def test_document_id_inventory_handles_an_exact_full_page_with_empty_successor() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=inventory_rows(0, 10_000)),
        FakeResponse(rows=[]),
        FakeResponse(aggregations={"count": 10_000}),
    ]
    provider, _ = make_provider(namespace)

    inventory = await provider.namespace_document_ids("fixture", max_documents=10_000)

    assert len(inventory.document_ids) == 10_000
    assert not inventory.truncated
    assert namespace.calls[1][1]["top_k"] == 1
    assert namespace.calls[1][1]["filters"] == ("id", "Gt", "doc-09999")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "doc-2"}, {"id": "doc-1"}], "duplicate or out of order"),
        ([{"id": "doc-1"}, {"id": "doc-1"}], "duplicate or out of order"),
        ([{"id": 1}, {"id": "doc-2"}], "mixed ID types"),
        ([{"id": True}], "valid id"),
    ],
)
async def test_document_id_inventory_rejects_invalid_page_ordering(
    rows: list[dict[str, object]],
    message: str,
) -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [FakeResponse(rows=rows)]
    provider, _ = make_provider(namespace)

    with pytest.raises(ValueError, match=message):
        await provider.namespace_document_ids("fixture", max_documents=20)


@pytest.mark.asyncio
async def test_document_id_inventory_rejects_non_progress_across_pages() -> None:
    namespace = FakeNamespace()
    namespace.query_responses = [
        FakeResponse(rows=inventory_rows(0, 10_000)),
        FakeResponse(rows=[{"id": "doc-09999"}]),
    ]
    provider, _ = make_provider(namespace)

    with pytest.raises(ValueError, match="duplicate or out of order"):
        await provider.namespace_document_ids("fixture", max_documents=10_000)


@pytest.mark.asyncio
async def test_document_id_inventory_rejects_oversized_and_count_inconsistent_pages() -> None:
    oversized = FakeNamespace()
    oversized.query_responses = [FakeResponse(rows=inventory_rows(0, 10_001))]
    oversized_provider, _ = make_provider(oversized)
    with pytest.raises(ValueError, match="invalid document inventory page"):
        await oversized_provider.namespace_document_ids("fixture", max_documents=20_000)

    short = FakeNamespace()
    short.query_responses = [
        FakeResponse(rows=inventory_rows(0, 9_999)),
        FakeResponse(aggregations={"count": 10_000}),
    ]
    short_provider, _ = make_provider(short)
    with pytest.raises(ValueError, match="invalid document inventory"):
        await short_provider.namespace_document_ids("fixture", max_documents=20_000)


@pytest.mark.asyncio
async def test_client_and_namespace_are_reused_and_close_is_idempotent() -> None:
    namespace = FakeNamespace()
    namespace.query_response = FakeResponse(rows=[])
    provider, client = make_provider(namespace, timer=clock(1.0, 1.1, 2.0, 2.1))

    await provider.query_bm25(
        namespace="same",
        lexical_fields=(("title", 2.0), ("body", 1.0)),
        query_text="one",
        top_k=1,
        include_attributes=(),
    )
    await provider.query_ann(
        namespace="same",
        vector_attribute="vector",
        query_vector=unit_vector(2, 0),
        top_k=1,
        include_attributes=(),
    )
    await provider.close()
    await provider.close()

    assert client.namespace_calls == ["same"]
    assert client.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await provider.namespace_metadata("same")


@pytest.mark.asyncio
async def test_metadata_exposes_index_readiness_without_sdk_models() -> None:
    namespace = FakeNamespace()
    namespace.metadata_response = FakeMetadata(
        approx_row_count=3,
        index=FakeIndex(status="updating", unindexed_bytes=128),
        schema_={"body": {"type": "string", "full_text_search": True}},
    )
    provider, _ = make_provider(namespace)

    metadata = await provider.namespace_metadata("fixture")

    assert metadata.approx_row_count == 3
    assert metadata.index_status == "updating"
    assert metadata.unindexed_bytes == 128
    assert metadata.ready is False
    assert metadata.schema == {"body": {"type": "string", "full_text_search": True}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_factory", "expected_code", "retryable"),
    [
        (
            lambda request, response, secret: AuthenticationError(
                f"invalid credential {secret}",
                response=response,
                body={"api_key": secret},
            ),
            ApiErrorCode.PROVIDER_ERROR,
            False,
        ),
        (
            lambda request, response, secret: RateLimitError(
                f"rate limit for {secret}",
                response=httpx.Response(429, request=request),
                body={"token": secret},
            ),
            ApiErrorCode.RATE_LIMITED,
            True,
        ),
        (
            lambda request, response, secret: NotFoundError(
                f"missing namespace for {secret}",
                response=httpx.Response(404, request=request),
                body={"credential": secret},
            ),
            ApiErrorCode.NOT_FOUND,
            False,
        ),
        (
            lambda request, response, secret: APIResponseValidationError(
                response=httpx.Response(202, request=request),
                body={"credential": secret},
                message=f"indexing for {secret}",
            ),
            ApiErrorCode.NAMESPACE_NOT_READY,
            True,
        ),
    ],
)
async def test_provider_errors_are_mapped_without_secret_material(
    error_factory: Callable[[httpx.Request, httpx.Response, str], APIError],
    expected_code: ApiErrorCode,
    retryable: bool,
) -> None:
    secret = "credential-material-that-must-not-leak"
    request = httpx.Request("POST", "https://api.turbopuffer.com/v2/namespaces/test/query")
    response = httpx.Response(401, request=request)
    namespace = FakeNamespace()
    namespace.query_error = error_factory(request, response, secret)
    provider, _ = make_provider(namespace)

    with pytest.raises(ProviderError) as raised:
        await provider.query_bm25(
            namespace="fixture",
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="secret test",
            top_k=1,
            include_attributes=(),
        )

    assert raised.value.details.code is expected_code
    assert raised.value.details.retryable is retryable
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    formatted_traceback = "".join(traceback.format_exception(raised.value, chain=True))
    assert secret not in formatted_traceback


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["query", "write", "metadata", "delete", "close"])
async def test_every_sdk_path_detaches_secret_bearing_exception_context(operation: str) -> None:
    secret = "path-specific-credential-that-must-not-leak"
    request = httpx.Request("POST", "https://api.turbopuffer.com/v2/namespaces/test")
    sdk_error = AuthenticationError(
        f"invalid credential {secret}",
        response=httpx.Response(401, request=request),
        body={"credential": secret},
    )
    namespace = FakeNamespace()
    provider, client = make_provider(namespace)

    if operation == "query":
        namespace.query_error = sdk_error
    elif operation == "write":
        namespace.write_error = sdk_error
    elif operation == "metadata":
        namespace.metadata_error = sdk_error
    elif operation == "delete":
        namespace.delete_error = sdk_error
    else:
        client.close_error = sdk_error

    with pytest.raises(ProviderError) as raised:
        if operation == "query":
            await provider.query_bm25(
                namespace="fixture",
                lexical_fields=(("title", 2.0), ("body", 1.0)),
                query_text="safe query",
                top_k=1,
                include_attributes=(),
            )
        elif operation == "write":
            await provider.write_documents(
                namespace="fixture",
                documents=(WriteDocument(id="one", attributes={"body": "safe document"}),),
                schema=cast(ProviderSchema, {"body": {"type": "string"}}),
                distance_metric="cosine_distance",
            )
        elif operation == "metadata":
            await provider.namespace_metadata("fixture")
        elif operation == "delete":
            await provider.delete_namespace("fixture")
        else:
            await provider.close()

    error = raised.value
    assert error.details.code is ApiErrorCode.PROVIDER_ERROR
    assert error.details.retryable is False
    assert error.__context__ is None
    assert error.__cause__ is None
    assert secret not in str(error)
    assert secret not in repr(error)
    formatted_traceback = "".join(traceback.format_exception(error, chain=True))
    assert secret not in formatted_traceback
