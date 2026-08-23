"""Narrow, injectable boundaries for interactive retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.providers.types import (
    ConsistencyLevel,
    DistanceMetric,
    LexicalFieldWeights,
    ProviderHybridProbeResult,
    ProviderQueryResult,
)


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """An in-process query embedding and its measured client-side duration.

    The vector is deliberately absent from every public API contract.
    """

    vector: tuple[float, ...]
    client_duration_ms: float


class QueryEmbedder(Protocol):
    model: str
    revision: str
    dimensions: int

    async def embed_query(self, query_text: str) -> QueryEmbedding: ...


class RetrievalProvider(Protocol):
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
    ) -> ProviderQueryResult: ...

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
    ) -> ProviderQueryResult: ...

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
    ) -> ProviderQueryResult: ...

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
    ) -> ProviderHybridProbeResult: ...

    async def close(self) -> None: ...


class SearchBackend(Protocol):
    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]: ...

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse: ...

    async def close(self) -> None: ...
