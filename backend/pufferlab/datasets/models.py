"""Strict local models for the checked-in dataset pack."""

from collections.abc import Mapping
from typing import Annotated, Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.identity import document_uuid


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, Field(min_length=1), AfterValidator(_require_non_blank)]


class StrictModel(BaseModel):
    """Reject unknown fields and scalar coercion in source data."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class EmbeddingProfile(StrictModel):
    provider: Literal["sentence_transformers"]
    model: NonBlank
    revision: NonBlank
    dimensions: int = Field(gt=0)


class VectorProfile(StrictModel):
    attribute: NonBlank
    dtype: Literal["f16", "f32", "i8"]
    distance_metric: Literal["cosine_distance", "euclidean_squared"]


class FtsProfile(StrictModel):
    attributes: list[Literal["title", "body"]] = Field(min_length=1)
    tokenizer: NonBlank
    case_sensitive: bool
    language: NonBlank
    stemming: bool
    remove_stopwords: bool
    ascii_folding: bool
    max_token_length: int = Field(ge=1, le=254)
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    k3: float = Field(gt=0)

    @model_validator(mode="after")
    def attributes_are_unique(self) -> "FtsProfile":
        if len(self.attributes) != len(set(self.attributes)):
            raise ValueError("FTS attributes must be unique")
        return self


class DatasetManifest(StrictModel):
    format_version: Literal[1]
    slug: NonBlank
    version: NonBlank
    title: NonBlank
    license: NonBlank
    source_url: NonBlank
    embedding: EmbeddingProfile
    vector: VectorProfile
    fts: FtsProfile

    @model_validator(mode="after")
    def vector_attribute_does_not_collide(self) -> "DatasetManifest":
        reserved = {"id", "external_id", "title", "body", "source_url"}
        if self.vector.attribute in reserved:
            raise ValueError("vector attribute collides with a document attribute")
        return self

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        _validate_http_url(value, field_name="source_url")
        return value


class SourceDocument(StrictModel):
    external_id: NonBlank
    title: NonBlank
    body: NonBlank
    source_url: NonBlank
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("external_id", "title", "body")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        _validate_http_url(value, field_name="source_url")
        return value


class FixtureQuery(StrictModel):
    external_id: NonBlank
    text: NonBlank
    expected_external_ids: list[NonBlank] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def expected_ids_are_unique(self) -> "FixtureQuery":
        if len(self.expected_external_ids) != len(set(self.expected_external_ids)):
            raise ValueError("expected_external_ids must be unique")
        return self


class FixtureCorpus(StrictModel):
    manifest: DatasetManifest
    documents: tuple[SourceDocument, ...]
    queries: tuple[FixtureQuery, ...]
    corpus_hash: NonBlank

    def document_id(self, external_id: str) -> UUID:
        return document_uuid(self.manifest.version, external_id)

    @property
    def expected_document_ids(self) -> Mapping[str, tuple[UUID, ...]]:
        return {
            query.external_id: tuple(
                self.document_id(value) for value in query.expected_external_ids
            )
            for query in self.queries
        }


def _validate_http_url(value: str, *, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
