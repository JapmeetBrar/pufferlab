"""Narrow, injectable boundaries for interactive retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pufferlab.contracts.common import ObservedScore
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareRequest,
    SearchCompareResponse,
)
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


@dataclass(frozen=True, slots=True)
class SearchExecuteRequest:
    """Internal single-config execution input; evaluation keeps provenance disabled by default."""

    namespace: str
    query_text: str
    config_id: UUID
    query_id: UUID | None = None
    filter_override: FilterNode | None = None
    expected_document_ids: tuple[UUID, ...] = ()
    debug_provenance: bool = False


@dataclass(frozen=True, slots=True)
class SearchExecuteResult:
    """One config result retaining the caller's query and configuration identities."""

    config_id: UUID
    query_id: UUID | None
    result: ConfigSearchResult


@dataclass(frozen=True, slots=True)
class HybridProbeExecuteRequest:
    """Internal explicit counterfactual-probe input with a caller-owned source trace."""

    namespace: str
    query_text: str
    config_id: UUID
    trace_id: UUID
    query_id: UUID | None = None
    filter_override: FilterNode | None = None


@dataclass(frozen=True, slots=True)
class HybridProbeStageMembership:
    """One bounded raw-list membership with no provider attributes or vector payload."""

    stage: RetrievalStage
    rank: int
    score: ObservedScore


@dataclass(frozen=True, slots=True)
class HybridProbeCandidate:
    document_id: UUID
    stage_membership: tuple[HybridProbeStageMembership, ...]


@dataclass(frozen=True, slots=True)
class HybridProbeExecuteResult:
    """Safe explicit probe result kept separate from production-shaped search results."""

    config_id: UUID
    query_id: UUID | None
    trace_id: UUID
    duration_ms: float
    bm25_candidate_count: int
    vector_candidate_count: int
    candidates: tuple[HybridProbeCandidate, ...]


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

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult: ...

    async def close(self) -> None: ...


class ReplaySearchBackend(SearchBackend, Protocol):
    """Dataset-bound search runtime with an explicit, separately traced debug probe."""

    async def probe_hybrid_candidates(
        self,
        request: HybridProbeExecuteRequest,
    ) -> HybridProbeExecuteResult: ...
