"""Dataset and turbopuffer index-profile contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from pufferlab.contracts.common import ContractModel


class FtsProfile(ContractModel):
    tokenizer: str = "word_v4"
    case_sensitive: bool = False
    language: str = "english"
    stemming: bool = False
    remove_stopwords: bool = False
    ascii_folding: bool = False
    max_token_length: int = Field(default=39, ge=1, le=254)
    k1: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    k3: float = Field(default=8.0, gt=0)


class IndexProfile(ContractModel):
    id: str = Field(min_length=1)
    embedding_provider: Literal["sentence_transformers"]
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(min_length=1)
    vector_attribute: str = "vector"
    vector_dimensions: int = Field(gt=0)
    vector_dtype: Literal["f16", "f32", "i8"]
    distance_metric: Literal["cosine_distance", "euclidean_squared", "dot_product"]
    fts_profile: FtsProfile
    schema_hash: str = Field(min_length=1)


class DatasetStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DatasetVersion(ContractModel):
    id: UUID
    slug: str = Field(min_length=1)
    version: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    index_profile: IndexProfile
    document_count: int = Field(ge=0)
    corpus_hash: str = Field(min_length=1)
    status: DatasetStatus
    created_at: datetime
