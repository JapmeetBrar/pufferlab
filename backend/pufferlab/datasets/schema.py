"""Compile a dataset manifest into the exact namespace write shape."""

from dataclasses import dataclass
from typing import Literal

from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.identity import canonical_schema_hash
from pufferlab.datasets.models import DatasetManifest, FtsProfile

type DistanceMetric = Literal["cosine_distance", "euclidean_squared"]


@dataclass(frozen=True, slots=True)
class FullTextSearchWriteSpec:
    tokenizer: str
    case_sensitive: bool
    language: str
    stemming: bool
    remove_stopwords: bool
    ascii_folding: bool
    max_token_length: int
    k1: float
    b: float
    k3: float

    @classmethod
    def from_profile(cls, profile: FtsProfile) -> "FullTextSearchWriteSpec":
        return cls(
            tokenizer=profile.tokenizer,
            case_sensitive=profile.case_sensitive,
            language=profile.language,
            stemming=profile.stemming,
            remove_stopwords=profile.remove_stopwords,
            ascii_folding=profile.ascii_folding,
            max_token_length=profile.max_token_length,
            k1=profile.k1,
            b=profile.b,
            k3=profile.k3,
        )

    def provider_value(self) -> dict[str, JsonValue]:
        return {
            "tokenizer": self.tokenizer,
            "case_sensitive": self.case_sensitive,
            "language": self.language,
            "stemming": self.stemming,
            "remove_stopwords": self.remove_stopwords,
            "ascii_folding": self.ascii_folding,
            "max_token_length": self.max_token_length,
            "k1": self.k1,
            "b": self.b,
            "k3": self.k3,
        }


@dataclass(frozen=True, slots=True)
class NamespaceAttributeWriteSpec:
    type: str
    filterable: bool | None = None
    full_text_search: FullTextSearchWriteSpec | None = None
    ann: bool | None = None

    def provider_value(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {"type": self.type}
        if self.filterable is not None:
            value["filterable"] = self.filterable
        if self.full_text_search is not None:
            value["full_text_search"] = self.full_text_search.provider_value()
        if self.ann is not None:
            value["ann"] = self.ann
        return value


@dataclass(frozen=True, slots=True)
class NamespaceWriteSpec:
    attributes: tuple[tuple[str, NamespaceAttributeWriteSpec], ...]
    vector_attribute: str
    distance_metric: DistanceMetric

    @property
    def provider_schema(self) -> dict[str, dict[str, JsonValue]]:
        return {name: specification.provider_value() for name, specification in self.attributes}

    @property
    def schema_hash(self) -> str:
        return canonical_schema_hash(
            {
                "schema": self.provider_schema,
                "distance_metric": self.distance_metric,
            }
        )


def compile_namespace_write_spec(manifest: DatasetManifest) -> NamespaceWriteSpec:
    fts = FullTextSearchWriteSpec.from_profile(manifest.fts)
    fts_attributes = set(manifest.fts.attributes)
    attributes = (
        (
            "external_id",
            NamespaceAttributeWriteSpec(type="string", filterable=True),
        ),
        (
            "title",
            NamespaceAttributeWriteSpec(
                type="string",
                filterable=False,
                full_text_search=fts if "title" in fts_attributes else None,
            ),
        ),
        (
            "body",
            NamespaceAttributeWriteSpec(
                type="string",
                filterable=False,
                full_text_search=fts if "body" in fts_attributes else None,
            ),
        ),
        (
            "source_url",
            NamespaceAttributeWriteSpec(type="string", filterable=False),
        ),
        (
            manifest.vector.attribute,
            NamespaceAttributeWriteSpec(
                type=f"[{manifest.embedding.dimensions}]{manifest.vector.dtype}",
                ann=True,
            ),
        ),
    )
    return NamespaceWriteSpec(
        attributes=attributes,
        vector_attribute=manifest.vector.attribute,
        distance_metric=manifest.vector.distance_metric,
    )
