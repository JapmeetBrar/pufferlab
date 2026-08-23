"""Small async adapter around the official turbopuffer SDK."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date, datetime
from time import perf_counter
from typing import Literal, Protocol, cast
from uuid import UUID

from turbopuffer import APIError, AsyncTurbopuffer

from pufferlab.contracts.common import (
    JsonValue,
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.filters import (
    FilterNode,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails, map_turbopuffer_error
from pufferlab.providers.types import (
    ConsistencyLevel,
    DistanceMetric,
    LexicalFieldWeights,
    ProviderDeleteResult,
    ProviderDocument,
    ProviderDocumentIdInventory,
    ProviderHybridProbeResult,
    ProviderNamespaceMetadata,
    ProviderQueryResult,
    ProviderSchema,
    ProviderWriteResult,
    WriteDocument,
)

type SdkFilter = tuple[object, ...]

_DOCUMENT_ID_PAGE_SIZE = 10_000


class _AsyncNamespace(Protocol):
    async def write(self, **kwargs: object) -> object: ...

    async def query(self, **kwargs: object) -> object: ...

    async def multi_query(self, **kwargs: object) -> object: ...

    async def metadata(self, **kwargs: object) -> object: ...

    async def delete_all(self, **kwargs: object) -> object: ...


class _AsyncClient(Protocol):
    def namespace(self, namespace: str) -> _AsyncNamespace: ...

    async def close(self) -> None: ...


class _DumpableModel(Protocol):
    def model_dump(self, *, by_alias: bool = False) -> dict[str, object]: ...


_PREDICATE_OPERATORS: dict[PredicateOp, str] = {
    PredicateOp.EQ: "Eq",
    PredicateOp.NOT_EQ: "NotEq",
    PredicateOp.LT: "Lt",
    PredicateOp.LTE: "Lte",
    PredicateOp.GT: "Gt",
    PredicateOp.GTE: "Gte",
    PredicateOp.IN: "In",
    PredicateOp.CONTAINS_ANY: "ContainsAny",
}


def filter_to_turbopuffer(node: FilterNode) -> SdkFilter:
    """Translate a validated neutral filter AST to the SDK tuple grammar."""

    if isinstance(node, FilterPredicate):
        value = node.value
        if node.op in {PredicateOp.IN, PredicateOp.CONTAINS_ANY} and not isinstance(value, list):
            raise ValueError(f"{node.op.value} filters require an array value")
        return (node.field, _PREDICATE_OPERATORS[node.op], value)

    children = tuple(filter_to_turbopuffer(child) for child in node.children)
    if node.op is LogicalOp.NOT:
        return ("Not", children[0])
    if node.op is LogicalOp.AND:
        return ("And", children)
    return ("Or", children)


def _weighted_bm25_rank_by(
    lexical_fields: LexicalFieldWeights,
    query_text: str,
) -> tuple[object, ...]:
    if not lexical_fields:
        raise ValueError("lexical_fields must not be empty")
    clauses: list[tuple[object, ...]] = []
    seen: set[str] = set()
    for attribute, weight in lexical_fields:
        if not attribute.strip() or attribute in seen:
            raise ValueError("lexical field names must be non-blank and unique")
        if isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("lexical field weights must be finite and positive")
        seen.add(attribute)
        clauses.append(("Product", weight, (attribute, "BM25", query_text)))
    if len(clauses) == 1:
        return clauses[0]
    return ("Sum", clauses)


class TurbopufferProvider:
    """Own and reuse one async SDK client for all provider calls."""

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        client: _AsyncClient | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._client = client or cast(
            _AsyncClient,
            AsyncTurbopuffer(api_key=api_key, region=region),
        )
        self._clock = clock
        self._namespaces: dict[str, _AsyncNamespace] = {}
        self._closed = False

    async def __aenter__(self) -> TurbopufferProvider:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await _call_sdk(self._client.close(), operation="close")
        finally:
            self._closed = True
            self._namespaces.clear()

    async def write_documents(
        self,
        *,
        namespace: str,
        documents: Sequence[WriteDocument],
        schema: ProviderSchema,
        distance_metric: DistanceMetric,
    ) -> ProviderWriteResult:
        """Upsert complete rows while always sending an explicit schema."""

        rows: list[dict[str, object]] = [
            {"id": document.id, **dict(document.attributes)} for document in documents
        ]
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        response = await _call_sdk(
            provider_namespace.write(
                upsert_rows=rows,
                schema=dict(schema),
                distance_metric=distance_metric,
            ),
            operation="write",
        )
        return ProviderWriteResult(
            rows_affected=_required_int(response, "rows_affected"),
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    async def query_bm25(
        self,
        *,
        namespace: str,
        lexical_fields: LexicalFieldWeights,
        query_text: str,
        top_k: int,
        include_attributes: Sequence[str],
        filters: FilterNode | None = None,
        consistency: ConsistencyLevel = "strong",
        vector_attributes: Sequence[str] = ("vector",),
    ) -> ProviderQueryResult:
        kwargs = self._query_kwargs(
            rank_by=_weighted_bm25_rank_by(lexical_fields, query_text),
            top_k=top_k,
            include_attributes=include_attributes,
            filters=filters,
            consistency=consistency,
        )
        return await self._query(
            namespace=namespace,
            kwargs=kwargs,
            score_kind=ScoreKind.BM25,
            vector_attributes=vector_attributes,
            operation="query_bm25",
        )

    async def query_ann(
        self,
        *,
        namespace: str,
        vector_attribute: str,
        query_vector: Sequence[float],
        top_k: int,
        include_attributes: Sequence[str],
        filters: FilterNode | None = None,
        consistency: ConsistencyLevel = "strong",
        distance_metric: DistanceMetric | None = None,
    ) -> ProviderQueryResult:
        kwargs = self._query_kwargs(
            rank_by=(vector_attribute, "ANN", list(query_vector)),
            top_k=top_k,
            include_attributes=include_attributes,
            filters=filters,
            consistency=consistency,
        )
        if distance_metric is not None:
            kwargs["distance_metric"] = distance_metric
        return await self._query(
            namespace=namespace,
            kwargs=kwargs,
            score_kind=ScoreKind.VECTOR_DISTANCE,
            vector_attributes=(vector_attribute,),
            operation="query_ann",
        )

    async def query_hybrid_rrf(
        self,
        *,
        namespace: str,
        lexical_fields: LexicalFieldWeights,
        query_text: str,
        vector_attribute: str,
        query_vector: Sequence[float],
        candidate_k: int,
        result_k: int,
        include_attributes: Sequence[str],
        rank_constant: int,
        weights: tuple[float, float],
        filters: FilterNode | None = None,
        consistency: ConsistencyLevel = "strong",
        distance_metric: DistanceMetric | None = None,
    ) -> ProviderQueryResult:
        """Execute production hybrid retrieval as one same-snapshot server RRF call."""

        queries = self._hybrid_subqueries(
            lexical_fields=lexical_fields,
            query_text=query_text,
            vector_attribute=vector_attribute,
            query_vector=query_vector,
            candidate_k=candidate_k,
            include_attributes=include_attributes,
            filters=filters,
            distance_metric=distance_metric,
        )
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        response = await _call_sdk(
            provider_namespace.multi_query(
                queries=queries,
                consistency={"level": consistency},
                limit={"total": result_k},
                rerank_by=(
                    "RRF",
                    {
                        "rank_constant": rank_constant,
                        "weights": list(weights),
                    },
                ),
            ),
            operation="query_hybrid_rrf",
        )
        result_rows = _multi_query_rows(response, expected_results=1)
        documents = tuple(
            _row_to_document(
                row,
                score_kind=ScoreKind.RRF,
                vector_attributes=(vector_attribute,),
            )
            for row in result_rows[0]
        )
        return ProviderQueryResult(
            documents=documents,
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    async def probe_hybrid_candidates(
        self,
        *,
        namespace: str,
        lexical_fields: LexicalFieldWeights,
        query_text: str,
        vector_attribute: str,
        query_vector: Sequence[float],
        candidate_k: int,
        include_attributes: Sequence[str],
        filters: FilterNode | None = None,
        consistency: ConsistencyLevel = "strong",
        distance_metric: DistanceMetric | None = None,
    ) -> ProviderHybridProbeResult:
        """Return raw hybrid lists through a separate, explicitly debug-only request."""

        queries = self._hybrid_subqueries(
            lexical_fields=lexical_fields,
            query_text=query_text,
            vector_attribute=vector_attribute,
            query_vector=query_vector,
            candidate_k=candidate_k,
            include_attributes=include_attributes,
            filters=filters,
            distance_metric=distance_metric,
        )
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        response = await _call_sdk(
            provider_namespace.multi_query(
                queries=queries,
                consistency={"level": consistency},
            ),
            operation="probe_hybrid_candidates",
        )
        bm25_rows, ann_rows = _multi_query_rows(response, expected_results=2)
        return ProviderHybridProbeResult(
            bm25_documents=tuple(
                _row_to_document(
                    row,
                    score_kind=ScoreKind.BM25,
                    vector_attributes=(vector_attribute,),
                )
                for row in bm25_rows
            ),
            ann_documents=tuple(
                _row_to_document(
                    row,
                    score_kind=ScoreKind.VECTOR_DISTANCE,
                    vector_attributes=(vector_attribute,),
                )
                for row in ann_rows
            ),
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        response = await _call_sdk(provider_namespace.metadata(), operation="metadata")

        index = _required_attribute(response, "index")
        status = _required_str(index, "status")
        if status not in {"updating", "up-to-date"}:
            details = ProviderErrorDetails(
                code=ApiErrorCode.PROVIDER_ERROR,
                retryable=False,
                operation="metadata",
            )
            raise ProviderError("turbopuffer returned invalid metadata", details)

        schema_value = _required_attribute(response, "schema_")
        schema = _mapping_to_json(schema_value)
        unindexed = _optional_int(index, "unindexed_bytes")
        return ProviderNamespaceMetadata(
            approx_row_count=_required_int(response, "approx_row_count"),
            index_status=cast(Literal["updating", "up-to-date"], status),
            unindexed_bytes=unindexed,
            schema=schema,
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    async def namespace_document_ids(
        self,
        namespace: str,
        *,
        max_documents: int,
    ) -> ProviderDocumentIdInventory:
        """Observe a bounded exact ID inventory with strong-consistency keyset pages.

        Requesting one row beyond the expected maximum proves whether the bounded result is
        complete without exceeding the provider's per-query result limit.
        """
        if max_documents < 1:
            raise ValueError("max_documents must be at least 1")

        provider_namespace = self._namespace(namespace)
        start = self._clock()
        inventory_limit = max_documents + 1
        document_ids: list[str | int] = []
        last_id: str | int | None = None
        while len(document_ids) < inventory_limit:
            page_size = min(_DOCUMENT_ID_PAGE_SIZE, inventory_limit - len(document_ids))
            query: dict[str, object] = {
                "rank_by": ("id", "asc"),
                "top_k": page_size,
                "include_attributes": [],
                "consistency": {"level": "strong"},
            }
            if last_id is not None:
                query["filters"] = ("id", "Gt", last_id)
            ids_response = await _call_sdk(
                provider_namespace.query(**query),
                operation="namespace_document_ids",
            )
            rows = _query_rows(ids_response)
            if len(rows) > page_size:
                raise ValueError("turbopuffer returned an invalid document inventory page")
            page_ids = tuple(_row_id(row) for row in rows)
            for document_id in page_ids:
                if last_id is not None:
                    _require_document_id_progress(last_id, document_id)
                document_ids.append(document_id)
                last_id = document_id
            if len(page_ids) < page_size:
                break

        count_response = await _call_sdk(
            provider_namespace.query(
                aggregate_by={"count": ("Count",)},
                consistency={"level": "strong"},
            ),
            operation="namespace_document_count",
        )
        document_count = _required_aggregation_count(count_response, "count")
        expected_returned_ids = min(document_count, max_documents + 1)
        if len(document_ids) != expected_returned_ids or len(set(document_ids)) != len(
            document_ids
        ):
            raise ValueError("turbopuffer returned an invalid document inventory")
        return ProviderDocumentIdInventory(
            document_ids=tuple(document_ids),
            document_count=document_count,
            truncated=document_count > max_documents,
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        await _call_sdk(provider_namespace.delete_all(), operation="delete_namespace")
        self._namespaces.pop(namespace, None)
        return ProviderDeleteResult(client_duration_ms=_elapsed_ms(start, self._clock()))

    async def _query(
        self,
        *,
        namespace: str,
        kwargs: Mapping[str, object],
        score_kind: ScoreKind,
        vector_attributes: Sequence[str],
        operation: str,
    ) -> ProviderQueryResult:
        provider_namespace = self._namespace(namespace)
        start = self._clock()
        response = await _call_sdk(
            provider_namespace.query(**dict(kwargs)),
            operation=operation,
        )

        rows_value = getattr(response, "rows", None)
        rows = () if rows_value is None else cast(Sequence[object], rows_value)
        documents = tuple(
            _row_to_document(row, score_kind=score_kind, vector_attributes=vector_attributes)
            for row in rows
        )
        return ProviderQueryResult(
            documents=documents,
            client_duration_ms=_elapsed_ms(start, self._clock()),
        )

    def _namespace(self, namespace: str) -> _AsyncNamespace:
        if self._closed:
            raise RuntimeError("turbopuffer provider is closed")
        if namespace not in self._namespaces:
            self._namespaces[namespace] = self._client.namespace(namespace)
        return self._namespaces[namespace]

    @staticmethod
    def _hybrid_subqueries(
        *,
        lexical_fields: LexicalFieldWeights,
        query_text: str,
        vector_attribute: str,
        query_vector: Sequence[float],
        candidate_k: int,
        include_attributes: Sequence[str],
        filters: FilterNode | None,
        distance_metric: DistanceMetric | None,
    ) -> list[dict[str, object]]:
        common: dict[str, object] = {
            "limit": candidate_k,
            "include_attributes": list(include_attributes),
        }
        if filters is not None:
            common["filters"] = filter_to_turbopuffer(filters)
        lexical = {
            **common,
            "rank_by": _weighted_bm25_rank_by(lexical_fields, query_text),
        }
        vector = {
            **common,
            "rank_by": (vector_attribute, "ANN", list(query_vector)),
        }
        if distance_metric is not None:
            vector["distance_metric"] = distance_metric
        return [lexical, vector]

    @staticmethod
    def _query_kwargs(
        *,
        rank_by: tuple[object, ...],
        top_k: int,
        include_attributes: Sequence[str],
        filters: FilterNode | None,
        consistency: ConsistencyLevel,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "rank_by": rank_by,
            "top_k": top_k,
            "include_attributes": list(include_attributes),
            "consistency": {"level": consistency},
        }
        if filters is not None:
            kwargs["filters"] = filter_to_turbopuffer(filters)
        return kwargs


def _row_to_document(
    row: object,
    *,
    score_kind: ScoreKind,
    vector_attributes: Sequence[str],
) -> ProviderDocument:
    data = _object_to_mapping(row)
    document_id = data.pop("id", None)
    if not isinstance(document_id, str | int) or isinstance(document_id, bool):
        raise ValueError("turbopuffer row is missing a valid id")
    score_value = data.pop("$dist", None)
    if not isinstance(score_value, int | float) or isinstance(score_value, bool):
        raise ValueError("turbopuffer row is missing a numeric $dist score")

    data.pop("vector", None)
    for vector_attribute in vector_attributes:
        data.pop(vector_attribute, None)

    direction = (
        ScoreDirection.LOWER_IS_BETTER
        if score_kind is ScoreKind.VECTOR_DISTANCE
        else ScoreDirection.HIGHER_IS_BETTER
    )
    return ProviderDocument(
        id=document_id,
        attributes={key: _to_json_value(value) for key, value in data.items()},
        score=ObservedScore(
            kind=score_kind,
            value=float(score_value),
            direction=direction,
            source=ScoreSource.TURBOPUFFER_DIST,
        ),
    )


def _row_id(row: object) -> str | int:
    document_id = _object_to_mapping(row).get("id")
    if not isinstance(document_id, str | int) or isinstance(document_id, bool):
        raise ValueError("turbopuffer row is missing a valid id")
    return document_id


def _query_rows(response: object) -> tuple[object, ...]:
    rows = getattr(response, "rows", None)
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes | bytearray):
        raise ValueError("turbopuffer document inventory response has invalid rows")
    return tuple(rows)


def _require_document_id_progress(previous: str | int, current: str | int) -> None:
    if isinstance(previous, str):
        if not isinstance(current, str):
            raise ValueError("turbopuffer document inventory contains mixed ID types")
        progresses = current > previous
    else:
        if not isinstance(current, int):
            raise ValueError("turbopuffer document inventory contains mixed ID types")
        progresses = current > previous
    if not progresses:
        raise ValueError("turbopuffer document inventory IDs are duplicate or out of order")


def _multi_query_rows(
    response: object,
    *,
    expected_results: int,
) -> tuple[tuple[object, ...], ...]:
    results = getattr(response, "results", None)
    if not isinstance(results, Sequence) or isinstance(results, str | bytes | bytearray):
        raise ValueError("turbopuffer multi-query response is missing results")
    if len(results) != expected_results:
        raise ValueError("turbopuffer multi-query returned an unexpected result count")
    result_rows: list[tuple[object, ...]] = []
    for result in results:
        rows = getattr(result, "rows", None)
        if rows is None:
            result_rows.append(())
        elif isinstance(rows, Sequence) and not isinstance(rows, str | bytes | bytearray):
            result_rows.append(tuple(rows))
        else:
            raise ValueError("turbopuffer multi-query returned invalid rows")
    return tuple(result_rows)


def _object_to_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return cast(_DumpableModel, value).model_dump(by_alias=True)
    return {str(key): item for key, item in vars(value).items()}


def _mapping_to_json(value: object) -> dict[str, JsonValue]:
    return {key: _to_json_value(item) for key, item in _object_to_mapping(value).items()}


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, UUID | date | datetime):
        return value.isoformat() if isinstance(value, date | datetime) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _mapping_to_json(value)
    return str(value)


def _required_attribute(value: object, name: str) -> object:
    result = getattr(value, name, None)
    if result is None:
        raise ValueError(f"turbopuffer response is missing {name}")
    return result


def _required_int(value: object, name: str) -> int:
    result = _required_attribute(value, name)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"turbopuffer response has invalid {name}")
    return result


def _optional_int(value: object, name: str) -> int | None:
    result = getattr(value, name, None)
    if result is None:
        return None
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"turbopuffer response has invalid {name}")
    return result


def _required_str(value: object, name: str) -> str:
    result = _required_attribute(value, name)
    if not isinstance(result, str):
        raise ValueError(f"turbopuffer response has invalid {name}")
    return result


def _required_aggregation_count(value: object, name: str) -> int:
    aggregations = getattr(value, "aggregations", None)
    if not isinstance(aggregations, Mapping):
        raise ValueError("turbopuffer response is missing aggregations")
    count = aggregations.get(name)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    if isinstance(count, float) and count.is_integer() and count >= 0:
        return int(count)
    raise ValueError(f"turbopuffer response has invalid {name} aggregation")


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


async def _call_sdk(awaitable: Awaitable[object], *, operation: str) -> object:
    """Detach secret-bearing SDK exceptions before raising their safe replacement."""

    safe_error: ProviderError | None = None
    try:
        return await awaitable
    except APIError as error:
        safe_error = map_turbopuffer_error(error, operation=operation)

    if safe_error is None:  # pragma: no cover - the try either returns or maps an APIError
        raise RuntimeError("provider error mapping failed")
    raise safe_error from None
