"""Immutable retrieval configuration contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion
from pufferlab.contracts.filters import FilterNode


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"


class LexicalSpec(ContractModel):
    title_weight: float = Field(default=2.0, ge=0)
    body_weight: float = Field(default=1.0, ge=0)


class VectorSpec(ContractModel):
    attribute: str = Field(default="vector", min_length=1)
    embedding_model: str = Field(min_length=1)


class RrfSpec(ContractModel):
    execution: Literal["server"] = "server"
    rank_constant: int = Field(default=60, gt=0)
    weights: tuple[float, float] = (1.0, 1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "RrfSpec":
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("RRF weights must be greater than zero")
        return self


class RerankerSpec(ContractModel):
    provider: Literal["sentence_transformers"]
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    depth: int = Field(default=50, gt=0)


class RetrievalConfig(ContractModel):
    id: UUID
    revision: int = Field(ge=1)
    name: str = Field(min_length=1)
    dataset_version_id: UUID
    mode: RetrievalMode
    result_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)
    consistency: Literal["strong", "eventual"] = "strong"
    filters: FilterNode | None = None
    lexical: LexicalSpec | None = None
    vector: VectorSpec | None = None
    rrf: RrfSpec | None = None
    reranker: RerankerSpec | None = None
    config_hash: str = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def validate_mode_specs(self) -> "RetrievalConfig":
        if self.candidate_k < self.result_k:
            raise ValueError("candidate_k must be greater than or equal to result_k")

        requirements = {
            RetrievalMode.BM25: (True, False, False, False),
            RetrievalMode.VECTOR: (False, True, False, False),
            RetrievalMode.HYBRID_RRF: (True, True, True, False),
            RetrievalMode.HYBRID_RERANK: (True, True, True, True),
        }
        actual = (
            self.lexical is not None,
            self.vector is not None,
            self.rrf is not None,
            self.reranker is not None,
        )
        if actual != requirements[self.mode]:
            raise ValueError(f"retrieval specs do not match mode {self.mode.value}")
        if self.reranker is not None and self.reranker.depth > self.candidate_k:
            raise ValueError("reranker depth cannot exceed candidate_k")
        return self


class RetrievalConfigSummary(ContractModel):
    id: UUID
    revision: int = Field(ge=1)
    name: str
    mode: RetrievalMode
    config_hash: str


class RetrievalConfigListResponse(ContractModel):
    contract_version: ContractVersion = 1
    configs: list[RetrievalConfigSummary] = Field(max_length=100)
