from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.ingestion import EmbeddedDocument, IngestionService, ReadinessError
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.models import FixtureCorpus
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec
from pufferlab.datasets.turbopuffer_writer import (
    TurbopufferNamespaceWriter,
    _normalize_observed_schema,
)
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import (
    DistanceMetric,
    ProviderDocumentIdInventory,
    ProviderNamespaceMetadata,
    ProviderSchema,
    ProviderWriteResult,
    WriteDocument,
)
from turbopuffer.types import NamespaceMetadata

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
        self.inventory_requests: list[tuple[str, int]] = []
        self.metadata: ProviderNamespaceMetadata | None = None
        self.inventory: ProviderDocumentIdInventory | None = None

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

    async def namespace_document_ids(
        self,
        namespace: str,
        *,
        max_documents: int,
    ) -> ProviderDocumentIdInventory:
        self.inventory_requests.append((namespace, max_documents))
        if self.inventory is None:
            raise RuntimeError("test inventory was not configured")
        return self.inventory


class FakeEmbedder:
    @property
    def dimensions(self) -> int:
        return 384

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.0] * self.dimensions for _ in texts]


@dataclass
class SdkResponse:
    rows: list[dict[str, object]] | None = None
    rows_affected: int = 0
    aggregations: dict[str, object] | None = None


class SdkNamespace:
    def __init__(
        self,
        *,
        metadata: NamespaceMetadata,
        document_ids: Sequence[UUID],
    ) -> None:
        self.metadata_response = metadata
        self.document_ids = tuple(document_ids)

    async def write(self, **kwargs: object) -> object:
        rows = kwargs.get("upsert_rows")
        return SdkResponse(rows_affected=len(rows) if isinstance(rows, list) else 0)

    async def query(self, **kwargs: object) -> object:
        if "aggregate_by" in kwargs:
            return SdkResponse(aggregations={"count": len(self.document_ids)})
        return SdkResponse(rows=[{"id": str(value)} for value in self.document_ids])

    async def metadata(self, **kwargs: object) -> object:
        del kwargs
        return self.metadata_response

    async def delete_all(self, **kwargs: object) -> object:
        del kwargs
        return object()


class SdkClient:
    def __init__(self, namespace: SdkNamespace) -> None:
        self.namespace_value = namespace

    def namespace(self, namespace: str) -> SdkNamespace:
        del namespace
        return self.namespace_value

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_adapter_forwards_exact_typed_schema_documents_and_distance_metric() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    writer = TurbopufferNamespaceWriter(provider)
    document = _embedded_documents(corpus)[0]

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
async def test_real_sdk_metadata_noise_is_normalized_and_reaches_ready() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    report = await _ingest_with_sdk_metadata(corpus)

    assert report.ready
    assert report.readiness is not None
    assert report.readiness.document_count == 20
    assert report.readiness.document_ids == _expected_ids(corpus)
    assert report.readiness.schema_hash == compile_namespace_write_spec(corpus.manifest).schema_hash


def test_only_the_observed_string_document_id_schema_is_provider_noise() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    observed = _observed_schema(specification, distance_metric="cosine_distance")

    normalized, _ = _normalize_observed_schema(observed, vector_attribute="vector")
    assert normalized == specification.provider_schema

    observed["id"] = {"type": "uint"}
    normalized, _ = _normalize_observed_schema(observed, vector_attribute="vector")
    assert normalized["id"] == {"type": "uint"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("distance_metric", "body_k1"),
    [
        ("euclidean_squared", 1.2),
        ("cosine_distance", 1.3),
    ],
)
async def test_real_sdk_metric_or_schema_drift_never_reaches_ready(
    distance_metric: DistanceMetric,
    body_k1: float,
) -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)

    with pytest.raises(ReadinessError) as caught:
        await _ingest_with_sdk_metadata(
            corpus,
            distance_metric=distance_metric,
            body_k1=body_k1,
        )

    assert not caught.value.report.ready
    assert caught.value.report.readiness is not None
    assert not caught.value.report.readiness.metadata_ready
    assert caught.value.report.readiness.schema_hash != (
        compile_namespace_write_spec(corpus.manifest).schema_hash
    )


@pytest.mark.asyncio
async def test_same_count_wrong_remote_ids_fail_readiness() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    provider.metadata = _provider_metadata(specification, approx_row_count=20)
    provider.inventory = ProviderDocumentIdInventory(
        document_ids=tuple(str(UUID(int=value)) for value in range(1, 21)),
        document_count=20,
        truncated=False,
        client_duration_ms=1.0,
    )
    writer = TurbopufferNamespaceWriter(provider)
    service = IngestionService(
        FakeEmbedder(),
        writer,
        batch_size=20,
        readiness_attempts=1,
    )

    with pytest.raises(ReadinessError) as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert not caught.value.report.ready
    assert caught.value.report.readiness is not None
    assert caught.value.report.readiness.document_count == 20
    assert caught.value.report.readiness.document_ids != _expected_ids(corpus)
    assert provider.inventory_requests == [("test-tiny", 20)]


@pytest.mark.asyncio
async def test_remote_inventory_over_expected_limit_fails_readiness() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    specification = compile_namespace_write_spec(corpus.manifest)
    provider = FakeProvider()
    provider.metadata = _provider_metadata(specification, approx_row_count=20)
    provider.inventory = ProviderDocumentIdInventory(
        document_ids=tuple([*(str(value) for value in _expected_ids(corpus)), str(UUID(int=1))]),
        document_count=40,
        truncated=True,
        client_duration_ms=1.0,
    )
    writer = TurbopufferNamespaceWriter(provider)
    service = IngestionService(
        FakeEmbedder(),
        writer,
        batch_size=20,
        readiness_attempts=1,
    )

    with pytest.raises(ReadinessError) as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert not caught.value.report.ready
    assert caught.value.report.readiness is not None
    assert caught.value.report.readiness.document_count == 40
    assert not caught.value.report.readiness.metadata_ready


@pytest.mark.asyncio
async def test_adapter_requires_a_write_specification_before_readiness() -> None:
    writer = TurbopufferNamespaceWriter(FakeProvider())

    with pytest.raises(RuntimeError, match="no dataset write specification"):
        await writer.inspect_readiness("test-tiny", expected_document_ids=frozenset({UUID(int=1)}))


async def _ingest_with_sdk_metadata(
    corpus: FixtureCorpus,
    *,
    distance_metric: DistanceMetric = "cosine_distance",
    body_k1: float = 1.2,
):
    specification = compile_namespace_write_spec(corpus.manifest)
    metadata = _sdk_metadata(
        specification,
        distance_metric=distance_metric,
        body_k1=body_k1,
    )
    namespace = SdkNamespace(
        metadata=metadata,
        document_ids=tuple(sorted(_expected_ids(corpus))),
    )
    provider = TurbopufferProvider(
        api_key="not-a-real-key",
        region="gcp-us-central1",
        client=SdkClient(namespace),
    )
    writer = TurbopufferNamespaceWriter.from_provider(provider)
    service = IngestionService(
        FakeEmbedder(),
        writer,
        batch_size=5,
        max_concurrency=2,
        readiness_attempts=1,
    )
    return await service.ingest(corpus, namespace="test-tiny")


def _sdk_metadata(
    specification: NamespaceWriteSpec,
    *,
    distance_metric: DistanceMetric,
    body_k1: float,
) -> NamespaceMetadata:
    schema = _observed_schema(specification, distance_metric=distance_metric)
    body = schema["body"]
    assert isinstance(body, dict)
    full_text_search = body["full_text_search"]
    assert isinstance(full_text_search, dict)
    full_text_search["k1"] = body_k1
    return NamespaceMetadata.model_validate(
        {
            "approx_logical_bytes": 12345,
            "approx_row_count": 999,
            "created_at": "2026-08-22T00:00:00Z",
            "updated_at": "2026-08-22T00:00:01Z",
            "encryption": {"mode": "default"},
            "index": {"status": "up-to-date"},
            "schema": schema,
        }
    )


def _provider_metadata(
    specification: NamespaceWriteSpec,
    *,
    approx_row_count: int,
) -> ProviderNamespaceMetadata:
    return ProviderNamespaceMetadata(
        approx_row_count=approx_row_count,
        index_status="up-to-date",
        unindexed_bytes=0,
        schema=_observed_schema(specification, distance_metric="cosine_distance"),
        client_duration_ms=1.0,
    )


def _observed_schema(
    specification: NamespaceWriteSpec,
    *,
    distance_metric: DistanceMetric,
) -> dict[str, JsonValue]:
    schema = deepcopy(specification.provider_schema)
    vector = schema[specification.vector_attribute]
    vector["ann"] = {
        "distance_metric": distance_metric,
        "late_interaction": None,
    }
    for attribute in schema.values():
        attribute["embed"] = None
        attribute["fuzzy"] = None
        attribute["glob"] = None
        attribute["regex"] = None
        attribute["sparse_knn"] = None
    schema["id"] = {"type": "string"}
    return schema


def _embedded_documents(corpus: FixtureCorpus) -> tuple[EmbeddedDocument, ...]:
    return tuple(
        EmbeddedDocument(
            id=corpus.document_id(source.external_id),
            external_id=source.external_id,
            title=source.title,
            body=source.body,
            source_url=source.source_url,
            vector=(0.0,) * corpus.manifest.embedding.dimensions,
        )
        for source in corpus.documents
    )


def _expected_ids(corpus: FixtureCorpus) -> frozenset[UUID]:
    return frozenset(document.id for document in _embedded_documents(corpus))
