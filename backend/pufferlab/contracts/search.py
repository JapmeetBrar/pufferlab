"""Interactive search comparison contracts."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from pufferlab.contracts.common import ContractModel, ContractVersion, JsonValue, ObservedScore
from pufferlab.contracts.errors import ApiWarning
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary


class HighlightOffset(ContractModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class HighlightFragment(ContractModel):
    text: str
    fragment_start: int | None = Field(default=None, ge=0)
    fragment_end: int | None = Field(default=None, ge=0)
    match_offsets: list[HighlightOffset] = Field(default_factory=list)


class RetrievalStage(StrEnum):
    BM25_CANDIDATES = "bm25_candidates"
    VECTOR_CANDIDATES = "vector_candidates"
    RRF = "rrf"
    RERANKER = "reranker"
    FINAL = "final"


class StageMembership(ContractModel):
    stage: RetrievalStage
    rank: int = Field(ge=1)
    score: ObservedScore | None = None


class SearchHit(ContractModel):
    document_id: UUID
    external_id: str
    title: str
    body_excerpt: str
    url: str | None = None
    relevance_grade: int | None = Field(default=None, ge=0)
    final_rank: int = Field(ge=1)
    final_score: ObservedScore | None = None
    stage_membership: list[StageMembership]
    highlights: list[HighlightFragment] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class TimingStage(StrEnum):
    EMBED = "embed"
    TURBOPUFFER = "turbopuffer"
    PROVENANCE_PROBE = "provenance_probe"
    FUSION = "fusion"
    RERANK = "rerank"
    TOTAL = "total"


class StageTiming(ContractModel):
    stage: TimingStage
    duration_ms: float = Field(ge=0)
    measurement: Literal["client_wall_clock"] = "client_wall_clock"


class ConfigSearchResult(ContractModel):
    config: RetrievalConfigSummary
    hits: list[SearchHit]
    timings: list[StageTiming]
    candidate_counts: dict[str, int]
    warnings: list[ApiWarning]
    trace_id: UUID


class RankMovement(ContractModel):
    document_id: UUID
    ranks_by_config: dict[UUID, int | None]
    max_absolute_delta: int | None = Field(default=None, ge=0)


class PairwiseOverlap(ContractModel):
    left_config_id: UUID
    right_config_id: UUID
    left_count: int = Field(ge=0)
    right_count: int = Field(ge=0)
    intersection_count: int = Field(ge=0)
    jaccard: float = Field(ge=0, le=1)


class SearchCompareRequest(ContractModel):
    contract_version: ContractVersion = 1
    query_text: str = Field(min_length=1)
    config_ids: list[UUID] = Field(min_length=2, max_length=4)
    query_id: UUID | None = None
    filter_override: FilterNode | None = None
    expected_document_ids: list[UUID] = Field(default_factory=list)
    debug_provenance: bool = True


class SearchCompareResponse(ContractModel):
    contract_version: ContractVersion = 1
    query_text: str
    query_id: UUID | None
    results: list[ConfigSearchResult]
    rank_movements: list[RankMovement]
    overlap: list[PairwiseOverlap]
    observability_notice: str
