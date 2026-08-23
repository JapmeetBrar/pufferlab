from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from pufferlab.datasets.checkpoints import IngestionCheckpointStore
from pufferlab.datasets.ingestion import (
    BatchWriteError,
    EmbeddedDocument,
    IngestionCheckpoint,
    IngestionService,
    NamespaceReadiness,
)
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "tiny-corpus"


class _Embedder:
    dimensions = 384

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]


class _InterruptibleWriter:
    def __init__(self) -> None:
        self.documents: dict[UUID, EmbeddedDocument] = {}
        self.calls: list[tuple[str, ...]] = []
        self.interrupt_on_second_batch = True
        self.schema_hash = ""

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        write_spec: NamespaceWriteSpec,
    ) -> None:
        del namespace
        external_ids = tuple(document.external_id for document in documents)
        self.calls.append(external_ids)
        if self.interrupt_on_second_batch and "tiny-006" in external_ids:
            raise RuntimeError("synthetic interruption")
        self.schema_hash = write_spec.schema_hash
        self.documents.update({document.id: document for document in documents})

    async def inspect_readiness(
        self,
        namespace: str,
        *,
        expected_document_ids: frozenset[UUID],
    ) -> NamespaceReadiness:
        del namespace
        del expected_document_ids
        return NamespaceReadiness(
            document_count=len(self.documents),
            document_ids=frozenset(self.documents),
            schema_hash=self.schema_hash,
            metadata_ready=True,
            indexes_ready=True,
        )


@pytest.mark.asyncio
async def test_interrupted_upserts_resume_from_atomic_checkpoint_without_deletion(
    tmp_path: Path,
) -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    writer = _InterruptibleWriter()
    service = IngestionService(_Embedder(), writer, batch_size=5, max_concurrency=1)
    store = IngestionCheckpointStore(tmp_path.resolve())

    with pytest.raises(BatchWriteError, match="dataset batch failed"):
        await service.ingest(
            corpus,
            namespace="caller-supplied-safe-namespace",
            on_checkpoint=store.save,
        )

    write_spec = compile_namespace_write_spec(corpus.manifest)
    checkpoint = store.load(
        namespace="caller-supplied-safe-namespace",
        dataset_version=corpus.manifest.version,
        corpus_hash=corpus.corpus_hash,
        schema_hash=write_spec.schema_hash,
    )
    assert checkpoint is not None
    assert len(checkpoint.completed_document_ids) == 5
    assert writer.calls.count(("tiny-001", "tiny-002", "tiny-003", "tiny-004", "tiny-005")) == 1
    assert not hasattr(writer, "delete_namespace")

    writer.interrupt_on_second_batch = False
    report = await service.ingest(
        corpus,
        namespace="caller-supplied-safe-namespace",
        resume_from=checkpoint,
        on_checkpoint=store.save,
    )

    assert report.ready
    assert report.resumed_documents == 5
    assert report.documents_completed == 20
    assert writer.calls.count(("tiny-001", "tiny-002", "tiny-003", "tiny-004", "tiny-005")) == 1


def test_checkpoint_filename_cannot_escape_data_directory(tmp_path: Path) -> None:
    store = IngestionCheckpointStore(tmp_path.resolve())
    checkpoint = IngestionCheckpoint(
        format_version=1,
        namespace="../../caller-owned-production",
        dataset_version="dataset-v1",
        corpus_hash="a" * 64,
        schema_hash="b" * 64,
        completed_document_ids=(UUID(int=1),),
    )

    path = store.save(checkpoint)
    loaded = store.load(
        namespace=checkpoint.namespace,
        dataset_version=checkpoint.dataset_version,
        corpus_hash=checkpoint.corpus_hash,
        schema_hash=checkpoint.schema_hash,
    )

    assert path.parent == tmp_path / "ingestion-checkpoints"
    assert path.name.endswith(".json")
    assert "caller-owned" not in path.name
    assert loaded == checkpoint


def test_checkpoint_data_directory_must_be_explicit_and_absolute() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        IngestionCheckpointStore(Path("data"))
