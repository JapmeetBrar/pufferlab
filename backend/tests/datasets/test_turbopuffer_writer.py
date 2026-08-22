from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pufferlab.datasets.ingestion import EmbeddedDocument
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.datasets.turbopuffer_writer import TurbopufferNamespaceWriter
from pufferlab.providers.types import (
    DistanceMetric,
    ProviderNamespaceMetadata,
    ProviderSchema,
    ProviderWriteResult,
    WriteDocument,
)

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "tiny-corpus"


@dataclass(frozen=True, slots=True)
class WriteCall:
    namespace: str
    documents: tuple[WriteDocument, ...]
    schema: ProviderSchema
    distance_metric: DistanceMetric


class FakeProvider:
    def __init__(self) -> None:
        self.write_calls: list[WriteCall] = []
        self.metadata: ProviderNamespaceMetadata | None = None

    async def write_documents(
        self,
        *,
        namespace: str,
        documents: Sequence[WriteDocument],
        schema: ProviderSchema,
        distance_metric: DistanceMetric,
    ) -> ProviderWriteResult:
        self.write_calls.append(
            WriteCall(
                namespace=namespace,
                documents=tuple(documents),
                schema=schema,
                distance_metric=distance_metric,
            )
        )
        return ProviderWriteResult(rows_affected=len(documents), client_duration_ms=1.0)

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
        del namespace
        if self.metadata is None:
            raise RuntimeError("test metadata was not configured")
        return self.metadata


@pytest.mark.asyncio
async def test_adapter_forwards_exact_typed_schema_documents_and_distance_metric() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    writer = TurbopufferNamespaceWriter(provider)
    document = _embedded_first_document()

    await writer.upsert_batch("test-tiny", (document,), write_spec=specification)

    call = provider.write_calls[0]
    assert call.namespace == "test-tiny"
    assert call.distance_metric == "cosine_distance"
    assert call.schema == specification.provider_schema
    assert call.schema["vector"] == {"type": "[384]f16", "ann": True}
    assert call.schema["title"] == {
        "type": "string",
        "filterable": False,
        "full_text_search": {
            "tokenizer": "word_v4",
            "case_sensitive": False,
            "language": "english",
            "stemming": False,
            "remove_stopwords": False,
            "ascii_folding": False,
            "max_token_length": 39,
            "k1": 1.2,
            "b": 0.75,
            "k3": 8.0,
        },
    }
    assert call.documents == (
        WriteDocument(
            id=str(document.id),
            attributes=document.provider_attributes(vector_attribute="vector"),
        ),
    )


@pytest.mark.asyncio
async def test_adapter_reports_remote_schema_and_acknowledged_exact_ids() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    writer = TurbopufferNamespaceWriter(provider)
    document = _embedded_first_document()
    await writer.upsert_batch("test-tiny", (document,), write_spec=specification)
    provider.metadata = ProviderNamespaceMetadata(
        approx_row_count=1,
        index_status="up-to-date",
        unindexed_bytes=0,
        schema=specification.provider_schema,
        client_duration_ms=1.0,
    )

    readiness = await writer.inspect_readiness("test-tiny")

    assert readiness.document_count == 1
    assert readiness.document_ids == frozenset({document.id})
    assert readiness.schema_hash == specification.schema_hash
    assert readiness.metadata_ready
    assert readiness.indexes_ready


@pytest.mark.asyncio
async def test_adapter_does_not_accept_drifted_remote_schema() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    writer = TurbopufferNamespaceWriter(provider)
    document = _embedded_first_document()
    await writer.upsert_batch("test-tiny", (document,), write_spec=specification)
    provider.metadata = ProviderNamespaceMetadata(
        approx_row_count=1,
        index_status="up-to-date",
        unindexed_bytes=0,
        schema={"body": {"type": "string", "full_text_search": True}},
        client_duration_ms=1.0,
    )

    readiness = await writer.inspect_readiness("test-tiny")

    assert not readiness.metadata_ready
    assert readiness.schema_hash != specification.schema_hash
    assert readiness.indexes_ready


@pytest.mark.asyncio
async def test_adapter_requires_a_successful_write_before_readiness() -> None:
    writer = TurbopufferNamespaceWriter(FakeProvider())

    with pytest.raises(RuntimeError, match="no acknowledged dataset write specification"):
        await writer.inspect_readiness("test-tiny")


def _embedded_first_document() -> EmbeddedDocument:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    source = corpus.documents[0]
    return EmbeddedDocument(
        id=corpus.document_id(source.external_id),
        external_id=source.external_id,
        title=source.title,
        body=source.body,
        source_url=source.source_url,
        vector=(0.0,) * corpus.manifest.embedding.dimensions,
    )
