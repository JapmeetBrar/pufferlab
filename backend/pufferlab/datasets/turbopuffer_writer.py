"""Typed integration from dataset ingestion to the turbopuffer provider."""

from collections.abc import Sequence
from typing import Protocol, Self
from uuid import UUID

from pufferlab.datasets.identity import canonical_schema_hash
from pufferlab.datasets.ingestion import EmbeddedDocument, NamespaceReadiness
from pufferlab.datasets.schema import NamespaceAttributeWriteSpec, NamespaceWriteSpec
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import (
    AttributeSchema,
    DistanceMetric,
    FullTextSearchSchema,
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


class TurbopufferNamespaceWriter:
    """Adapt compiled dataset writes without exposing SDK types to the ingestion service."""

    def __init__(self, provider: _WriteProvider) -> None:
        self._provider = provider
        self._specifications: dict[str, NamespaceWriteSpec] = {}
        self._acknowledged_ids: dict[str, set[UUID]] = {}

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
        acknowledged = self._acknowledged_ids.setdefault(namespace, set())
        acknowledged.update(document.id for document in documents)

    async def inspect_readiness(self, namespace: str) -> NamespaceReadiness:
        write_spec = self._specifications.get(namespace)
        if write_spec is None:
            raise RuntimeError("namespace has no acknowledged dataset write specification")

        metadata = await self._provider.namespace_metadata(namespace)
        expected_schema = write_spec.provider_schema
        actual_schema = dict(metadata.schema)
        schema_matches = actual_schema == expected_schema
        actual_schema_hash = canonical_schema_hash(
            {
                "schema": actual_schema,
                "distance_metric": write_spec.distance_metric,
            }
        )
        return NamespaceReadiness(
            document_count=metadata.approx_row_count,
            document_ids=frozenset(self._acknowledged_ids.get(namespace, set())),
            schema_hash=actual_schema_hash,
            metadata_ready=schema_matches,
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
