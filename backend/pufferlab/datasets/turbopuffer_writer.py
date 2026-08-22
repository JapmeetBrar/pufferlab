"""Typed integration from dataset ingestion to the turbopuffer provider."""

from collections.abc import Mapping, Sequence
from typing import Protocol, Self
from uuid import UUID

from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.identity import canonical_schema_hash
from pufferlab.datasets.ingestion import EmbeddedDocument, NamespaceReadiness
from pufferlab.datasets.schema import NamespaceAttributeWriteSpec, NamespaceWriteSpec
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import (
    AttributeSchema,
    DistanceMetric,
    FullTextSearchSchema,
    ProviderDocumentIdInventory,
    ProviderNamespaceMetadata,
    ProviderSchema,
    ProviderWriteResult,
    WriteDocument,
)


class _WriteProvider(Protocol):
    async def write_documents(
        self,
        *,
        namespace: str,
        documents: Sequence[WriteDocument],
        schema: ProviderSchema,
        distance_metric: DistanceMetric,
    ) -> ProviderWriteResult: ...

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata: ...

    async def namespace_document_ids(
        self,
        namespace: str,
        *,
        max_documents: int,
    ) -> ProviderDocumentIdInventory: ...


class TurbopufferNamespaceWriter:
    """Adapt compiled dataset writes without exposing SDK types to the ingestion service."""

    def __init__(self, provider: _WriteProvider) -> None:
        self._provider = provider
        self._specifications: dict[str, NamespaceWriteSpec] = {}

    @classmethod
    def from_provider(cls, provider: TurbopufferProvider) -> Self:
        """Construct from the real adapter while statically checking its protocol compatibility."""
        return cls(provider)

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        write_spec: NamespaceWriteSpec,
    ) -> None:
        existing = self._specifications.setdefault(namespace, write_spec)
        if existing != write_spec:
            raise ValueError("namespace cannot be written with conflicting specifications")

        provider_documents = tuple(
            WriteDocument(
                id=str(document.id),
                attributes=document.provider_attributes(
                    vector_attribute=write_spec.vector_attribute
                ),
            )
            for document in documents
        )
        await self._provider.write_documents(
            namespace=namespace,
            documents=provider_documents,
            schema=_compile_provider_schema(write_spec),
            distance_metric=write_spec.distance_metric,
        )

    async def inspect_readiness(
        self,
        namespace: str,
        *,
        expected_document_ids: frozenset[UUID],
    ) -> NamespaceReadiness:
        write_spec = self._specifications.get(namespace)
        if write_spec is None:
            raise RuntimeError("namespace has no dataset write specification")

        metadata = await self._provider.namespace_metadata(namespace)
        inventory = await self._provider.namespace_document_ids(
            namespace,
            max_documents=len(expected_document_ids),
        )
        expected_schema = write_spec.provider_schema
        actual_schema, actual_distance_metric = _normalize_observed_schema(
            metadata.schema,
            vector_attribute=write_spec.vector_attribute,
        )
        schema_matches = (
            actual_schema == expected_schema
            and actual_distance_metric == write_spec.distance_metric
        )
        actual_schema_hash = canonical_schema_hash(
            {
                "schema": actual_schema,
                "distance_metric": actual_distance_metric,
            }
        )
        remote_document_ids = frozenset(UUID(str(value)) for value in inventory.document_ids)
        return NamespaceReadiness(
            document_count=inventory.document_count,
            document_ids=remote_document_ids,
            schema_hash=actual_schema_hash,
            metadata_ready=schema_matches and not inventory.truncated,
            indexes_ready=metadata.ready,
        )


def _compile_provider_schema(write_spec: NamespaceWriteSpec) -> ProviderSchema:
    schema: dict[str, AttributeSchema] = {}
    for name, specification in write_spec.attributes:
        schema[name] = _compile_attribute(specification)
    return schema


def _compile_attribute(specification: NamespaceAttributeWriteSpec) -> AttributeSchema:
    attribute: AttributeSchema = {"type": specification.type}
    if specification.filterable is not None:
        attribute["filterable"] = specification.filterable
    if specification.full_text_search is not None:
        fts = specification.full_text_search
        full_text_search: FullTextSearchSchema = {
            "tokenizer": fts.tokenizer,
            "case_sensitive": fts.case_sensitive,
            "language": fts.language,
            "stemming": fts.stemming,
            "remove_stopwords": fts.remove_stopwords,
            "ascii_folding": fts.ascii_folding,
            "max_token_length": fts.max_token_length,
            "k1": fts.k1,
            "b": fts.b,
            "k3": fts.k3,
        }
        attribute["full_text_search"] = full_text_search
    if specification.ann is not None:
        attribute["ann"] = specification.ann
    return attribute


def _normalize_observed_schema(
    schema: Mapping[str, JsonValue],
    *,
    vector_attribute: str,
) -> tuple[dict[str, JsonValue], DistanceMetric | None]:
    normalized = _normalize_mapping(schema)
    if normalized.get("id") == {"type": "string"}:
        normalized.pop("id")
    vector = normalized.get(vector_attribute)
    if not isinstance(vector, dict):
        return normalized, None
    ann = vector.get("ann")
    if not isinstance(ann, dict):
        return normalized, None

    metric_value = ann.get("distance_metric")
    metric: DistanceMetric | None = None
    if metric_value == "cosine_distance":
        metric = "cosine_distance"
    elif metric_value == "euclidean_squared":
        metric = "euclidean_squared"

    remaining_ann = {key: value for key, value in ann.items() if key != "distance_metric"}
    normalized_vector = dict(vector)
    normalized_vector["ann"] = remaining_ann or True
    normalized[vector_attribute] = normalized_vector
    return normalized, metric


def _normalize_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _normalize_value(item) for key, item in value.items() if item is not None}


def _normalize_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value if item is not None]
    if value is None:  # pragma: no cover - callers filter mapping/list nulls
        raise ValueError("cannot normalize an unowned null schema value")
    return value
