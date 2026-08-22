import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from pufferlab.datasets.identity import canonical_schema_hash
from pufferlab.datasets.ingestion import (
    BatchWriteError,
    EmbeddedDocument,
    EmbeddingCountMismatch,
    EmbeddingDimensionMismatch,
    IngestionProgress,
    IngestionService,
    IngestionState,
    NamespaceReadiness,
    ReadinessError,
)
from pufferlab.datasets.loader import load_fixture_corpus

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "tiny-corpus"


class FakeEmbedder:
    def __init__(
        self,
        dimensions: int = 384,
        *,
        returned_dimensions: int | None = None,
        omit_last: bool = False,
    ) -> None:
        self._dimensions = dimensions
        self._returned_dimensions = returned_dimensions or dimensions
        self._omit_last = omit_last
        self.active = 0
        self.max_active = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        count = len(texts) - 1 if self._omit_last else len(texts)
        vectors = [[float(index)] * self._returned_dimensions for index in range(count)]
        self.active -= 1
        return vectors


class FakeWriter:
    def __init__(self, *, fail_on_external_id: str | None = None) -> None:
        self.documents: dict[str, EmbeddedDocument] = {}
        self.calls: list[tuple[str, ...]] = []
        self.fail_on_external_id = fail_on_external_id
        self.schema_hash = ""
        self.active = 0
        self.max_active = 0
        self.readiness_checks = 0
        self.readiness_override: NamespaceReadiness | None = None
        self.readiness_sequence: list[NamespaceReadiness] = []

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        schema: dict[str, object],
    ) -> None:
        del namespace
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        external_ids = tuple(document.external_id for document in documents)
        self.calls.append(external_ids)
        if self.fail_on_external_id in external_ids:
            self.active -= 1
            raise RuntimeError("provider detail that must not become the public ingestion error")
        self.schema_hash = canonical_schema_hash(schema)
        self.documents.update({str(document.id): document for document in documents})
        self.active -= 1

    async def inspect_readiness(self, namespace: str) -> NamespaceReadiness:
        del namespace
        self.readiness_checks += 1
        if self.readiness_sequence:
            return self.readiness_sequence.pop(0)
        if self.readiness_override is not None:
            return self.readiness_override
        return NamespaceReadiness(
            document_count=len(self.documents),
            schema_hash=self.schema_hash,
            metadata_ready=True,
            indexes_ready=True,
        )


@pytest.mark.asyncio
async def test_ingests_twenty_documents_in_deterministic_bounded_batches() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    embedder = FakeEmbedder()
    writer = FakeWriter()
    progress: list[IngestionProgress] = []
    service = IngestionService(embedder, writer, batch_size=6, max_concurrency=2)

    report = await service.ingest(corpus, namespace="test-tiny", on_progress=progress.append)

    assert report.ready
    assert report.state is IngestionState.READY
    assert report.documents_completed == 20
    assert report.batches_completed == 4
    assert [len(result.document_ids) for result in report.batch_results] == [6, 6, 6, 2]
    assert sorted(writer.calls) == [
        ("tiny-001", "tiny-002", "tiny-003", "tiny-004", "tiny-005", "tiny-006"),
        ("tiny-007", "tiny-008", "tiny-009", "tiny-010", "tiny-011", "tiny-012"),
        ("tiny-013", "tiny-014", "tiny-015", "tiny-016", "tiny-017", "tiny-018"),
        ("tiny-019", "tiny-020"),
    ]
    assert embedder.max_active == 2
    assert writer.max_active == 2
    assert progress[0].state is IngestionState.INGESTING
    assert progress[-2].state is IngestionState.VERIFYING
    assert progress[-1].state is IngestionState.READY


@pytest.mark.asyncio
async def test_rerun_is_idempotent_because_upserts_reuse_stable_ids() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    service = IngestionService(FakeEmbedder(), writer, batch_size=5, max_concurrency=3)

    first = await service.ingest(corpus, namespace="test-tiny")
    first_snapshot = writer.documents.copy()
    second = await service.ingest(corpus, namespace="test-tiny")

    assert first.ready and second.ready
    assert len(writer.documents) == 20
    assert writer.documents == first_snapshot
    assert first.batch_results == second.batch_results
    assert len(writer.calls) == 8


@pytest.mark.asyncio
async def test_rejects_embedder_schema_dimension_mismatch_before_writing() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    service = IngestionService(FakeEmbedder(dimensions=16), writer)

    with pytest.raises(EmbeddingDimensionMismatch) as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert caught.value.report.state is IngestionState.FAILED
    assert not caught.value.report.ready
    assert not writer.calls
    assert writer.readiness_checks == 0


@pytest.mark.asyncio
async def test_rejects_wrong_vector_count_without_writing_batch() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    service = IngestionService(FakeEmbedder(omit_last=True), writer, batch_size=20)

    with pytest.raises(
        EmbeddingCountMismatch, match="returned 19 vectors for 20 documents"
    ) as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert not caught.value.report.ready
    assert not writer.calls
    assert writer.readiness_checks == 0


@pytest.mark.asyncio
async def test_rejects_wrong_vector_dimensions_without_writing_batch() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    service = IngestionService(
        FakeEmbedder(returned_dimensions=383), writer, batch_size=20, max_concurrency=1
    )

    with pytest.raises(EmbeddingDimensionMismatch, match="has 383 dimensions; expected 384"):
        await service.ingest(corpus, namespace="test-tiny")

    assert not writer.calls
    assert writer.readiness_checks == 0


@pytest.mark.asyncio
async def test_batch_failure_propagates_without_readiness_claim() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter(fail_on_external_id="tiny-009")
    service = IngestionService(FakeEmbedder(), writer, batch_size=5, max_concurrency=2)

    with pytest.raises(BatchWriteError, match="a dataset batch failed to ingest") as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert caught.value.report.state is IngestionState.FAILED
    assert not caught.value.report.ready
    assert caught.value.report.batches_completed == 1
    assert caught.value.report.documents_completed == 5
    assert writer.readiness_checks == 0
    assert "provider detail" not in str(caught.value)

    writer.fail_on_external_id = None
    resumed = await service.ingest(corpus, namespace="test-tiny")
    assert resumed.ready
    assert len(writer.documents) == 20


@pytest.mark.asyncio
async def test_namespace_is_not_ready_until_metadata_and_indexes_match() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    writer.readiness_override = NamespaceReadiness(
        document_count=20,
        schema_hash=corpus.manifest.schema_hash,
        metadata_ready=True,
        indexes_ready=False,
    )
    service = IngestionService(FakeEmbedder(), writer, batch_size=20, readiness_attempts=1)

    with pytest.raises(ReadinessError, match="did not satisfy readiness checks") as caught:
        await service.ingest(corpus, namespace="test-tiny")

    assert caught.value.report.state is IngestionState.FAILED
    assert not caught.value.report.ready
    assert caught.value.report.readiness is not None
    assert not caught.value.report.readiness.indexes_ready


@pytest.mark.asyncio
async def test_readiness_is_polled_until_indexes_are_available() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = FakeWriter()
    writer.readiness_sequence = [
        NamespaceReadiness(
            document_count=20,
            schema_hash=corpus.manifest.schema_hash,
            metadata_ready=True,
            indexes_ready=False,
        ),
        NamespaceReadiness(
            document_count=20,
            schema_hash=corpus.manifest.schema_hash,
            metadata_ready=True,
            indexes_ready=True,
        ),
    ]
    service = IngestionService(
        FakeEmbedder(),
        writer,
        batch_size=20,
        readiness_attempts=2,
        readiness_poll_interval=0,
    )

    report = await service.ingest(corpus, namespace="test-tiny")

    assert report.ready
    assert writer.readiness_checks == 2
