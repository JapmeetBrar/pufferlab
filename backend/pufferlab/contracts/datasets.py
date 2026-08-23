"""Dataset and turbopuffer index-profile contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel


class FtsProfile(ContractModel):
    tokenizer: str = "word_v4"
    case_sensitive: bool = False
    language: str = "english"
    stemming: bool = False
    remove_stopwords: bool = False
    ascii_folding: bool = False
    max_token_length: int = Field(default=39, ge=1, le=255)
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
    distance_metric: Literal["cosine_distance", "euclidean_squared"]
    fts_profile: FtsProfile
    schema_hash: str = Field(min_length=1)


class DatasetStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DataOrigin(StrEnum):
    """Whether persisted catalog data came from provider-backed work or the offline demo."""

    LIVE = "live"
    SYNTHETIC_DEMO = "synthetic_demo"


class DatasetVersion(ContractModel):
    id: UUID
    slug: str = Field(min_length=1)
    version: str = Field(min_length=1)
    # Live M2 payloads predate ``data_origin``. Omitting the live default during serialization
    # preserves their canonical immutable JSON, while synthetic rows retain an explicit marker.
    data_origin: DataOrigin = Field(
        default=DataOrigin.LIVE,
        exclude_if=lambda value: value is DataOrigin.LIVE,
    )
    namespace: str
    index_profile: IndexProfile
    document_count: int = Field(ge=0)
    corpus_hash: str = Field(min_length=1)
    status: DatasetStatus
    created_at: datetime

    @model_validator(mode="after")
    def validate_origin_namespace(self) -> "DatasetVersion":
        if self.data_origin is DataOrigin.LIVE and not self.namespace.strip():
            raise ValueError("live dataset revisions require a provider namespace")
        if self.data_origin is DataOrigin.SYNTHETIC_DEMO and self.namespace != "":
            raise ValueError("synthetic demo revisions cannot claim a provider namespace")
        return self
