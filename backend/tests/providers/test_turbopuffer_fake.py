from __future__ import annotations

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
from turbopuffer import APIError, AuthenticationError, RateLimitError


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


class FakeNamespace:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.query_response = FakeResponse()
        self.write_response = FakeResponse(rows_affected=0)
        self.metadata_response = FakeMetadata(
            approx_row_count=0,
            index=FakeIndex(),
            schema_={},
        )
        self.query_error: APIError | None = None

    async def write(self, **kwargs: object) -> object:
        self.calls.append(("write", kwargs))
        return self.write_response

    async def query(self, **kwargs: object) -> object:
        self.calls.append(("query", kwargs))
        if self.query_error is not None:
            raise self.query_error
        return self.query_response

    async def metadata(self, **kwargs: object) -> object:
        self.calls.append(("metadata", kwargs))
        return self.metadata_response

    async def delete_all(self, **kwargs: object) -> object:
        self.calls.append(("delete_all", kwargs))
        return object()


class FakeClient:
    def __init__(self, namespace: FakeNamespace) -> None:
        self.fake_namespace = namespace
        self.namespace_calls: list[str] = []
        self.close_calls = 0

    def namespace(self, namespace: str) -> FakeNamespace:
        self.namespace_calls.append(namespace)
        return self.fake_namespace

    async def close(self) -> None:
        self.close_calls += 1


def clock(*values: float) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


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
    predicate = FilterPredicate(field="source", op=operation, value="unix")

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
            "text": {"type": "string", "full_text_search": True},
            "vector": {"type": "[2]f32", "ann": True},
        },
    )

    result = await provider.write_documents(
        namespace="fixture",
        documents=(
            WriteDocument(id="one", attributes={"text": "first", "vector": [1.0, 0.0]}),
            WriteDocument(id="two", attributes={"text": "second", "vector": [0.0, 1.0]}),
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
                    {"id": "one", "text": "first", "vector": [1.0, 0.0]},
                    {"id": "two", "text": "second", "vector": [0.0, 1.0]},
                ],
                "schema": schema,
                "distance_metric": "cosine_distance",
            },
        )
    ]


@pytest.mark.asyncio
async def test_bm25_query_shape_and_score_semantics() -> None:
    namespace = FakeNamespace()
    namespace.query_response = FakeResponse(
        rows=[
            {
                "id": "doc-1",
                "$dist": 3.75,
                "title": "Pufferfish",
                "published_at": datetime(2026, 8, 22, tzinfo=UTC),
                "vector": [1.0, 0.0],
            }
        ]
    )
    provider, _ = make_provider(namespace, timer=clock(5.0, 5.012))
    filter_node = FilterPredicate(field="source", op=PredicateOp.EQ, value="unix")

    result = await provider.query_bm25(
        namespace="fixture",
        text_attribute="body",
        query_text="pufferfish",
        top_k=5,
        include_attributes=("title", "published_at"),
        filters=filter_node,
    )

    assert namespace.calls == [
        (
            "query",
            {
                "rank_by": ("body", "BM25", "pufferfish"),
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
async def test_ann_query_shape_and_distance_score_semantics() -> None:
    namespace = FakeNamespace()
    namespace.query_response = FakeResponse(
        rows=[{"id": "doc-2", "$dist": 0.125, "title": "Nearest", "embedding": [0.0, 1.0]}]
    )
    provider, _ = make_provider(namespace)

    result = await provider.query_ann(
        namespace="fixture",
        vector_attribute="embedding",
        query_vector=(0.0, 1.0),
        top_k=3,
        include_attributes=("title",),
        consistency="eventual",
        distance_metric="cosine_distance",
    )

    assert namespace.calls == [
        (
            "query",
            {
                "rank_by": ("embedding", "ANN", [0.0, 1.0]),
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
async def test_client_and_namespace_are_reused_and_close_is_idempotent() -> None:
    namespace = FakeNamespace()
    namespace.query_response = FakeResponse(rows=[])
    provider, client = make_provider(namespace, timer=clock(1.0, 1.1, 2.0, 2.1))

    await provider.query_bm25(
        namespace="same",
        text_attribute="body",
        query_text="one",
        top_k=1,
        include_attributes=(),
    )
    await provider.query_ann(
        namespace="same",
        vector_attribute="vector",
        query_vector=(1.0, 0.0),
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
    ],
)
async def test_provider_errors_are_mapped_without_secret_material(
    error_factory: Callable[[httpx.Request, httpx.Response, str], APIError],
    expected_code: ApiErrorCode,
    retryable: bool,
) -> None:
    secret = "tpuf-secret-that-must-not-leak"
    request = httpx.Request("POST", "https://api.turbopuffer.com/v2/namespaces/test/query")
    response = httpx.Response(401, request=request)
    namespace = FakeNamespace()
    namespace.query_error = error_factory(request, response, secret)
    provider, _ = make_provider(namespace)

    with pytest.raises(ProviderError) as raised:
        await provider.query_bm25(
            namespace="fixture",
            text_attribute="body",
            query_text="secret test",
            top_k=1,
            include_attributes=(),
        )

    assert raised.value.details.code is expected_code
    assert raised.value.details.retryable is retryable
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    assert raised.value.__cause__ is None
