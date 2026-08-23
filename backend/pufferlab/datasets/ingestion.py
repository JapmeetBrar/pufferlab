"""Provider-neutral, deterministic asynchronous corpus ingestion."""

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.identity import document_uuid
from pufferlab.datasets.models import FixtureCorpus, SourceDocument
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    id: UUID
    external_id: str
    title: str
    body: str
    source_url: str
    vector: tuple[float, ...]

    def provider_attributes(self, *, vector_attribute: str) -> dict[str, JsonValue]:
        """Return exactly the attributes represented by the compiled namespace schema."""
        return {
            "external_id": self.external_id,
            "title": self.title,
            "body": self.body,
            "source_url": self.source_url,
            vector_attribute: list(self.vector),
        }


class Embedder(Protocol):
    """Concrete local embedding models implement this integration boundary."""

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class NamespaceReadiness:
    document_count: int
    document_ids: frozenset[UUID]
    schema_hash: str
    metadata_ready: bool
    indexes_ready: bool


class NamespaceWriter(Protocol):
    """A namespace provider must implement stable-ID upserts and readiness inspection."""

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        write_spec: NamespaceWriteSpec,
    ) -> None: ...

    async def inspect_readiness(
        self,
        namespace: str,
        *,
        expected_document_ids: frozenset[UUID],
    ) -> NamespaceReadiness: ...


class IngestionState(StrEnum):
    INGESTING = "ingesting"
    VERIFYING = "verifying"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestionProgress:
    state: IngestionState
    batches_completed: int
    batches_total: int
    documents_completed: int
    documents_total: int


@dataclass(frozen=True, slots=True)
class BatchResult:
    index: int
    document_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class IngestionCheckpoint:
    """Durable resume capability bound to one namespace and immutable corpus revision."""

    format_version: int
    namespace: str
    dataset_version: str
    corpus_hash: str
    schema_hash: str
    completed_document_ids: tuple[UUID, ...]

    @property
    def completed_ids(self) -> frozenset[UUID]:
        return frozenset(self.completed_document_ids)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    namespace: str
    dataset_version: str
    corpus_hash: str
    schema_hash: str
    state: IngestionState
    batches_total: int
    batches_completed: int
    documents_total: int
    documents_completed: int
    batch_results: tuple[BatchResult, ...]
    resumed_documents: int = 0
    readiness: NamespaceReadiness | None = None

    @property
    def ready(self) -> bool:
        return self.state is IngestionState.READY and self.readiness is not None


class IngestionError(RuntimeError):
    """Base error carrying a non-ready progress report."""

    def __init__(self, message: str, report: IngestionReport) -> None:
        super().__init__(message)
        self.report = report


class EmbeddingCountMismatch(IngestionError):
    pass


class EmbeddingDimensionMismatch(IngestionError):
    pass


class EmbeddingValueMismatch(IngestionError):
    pass


class BatchWriteError(IngestionError):
    pass


class ReadinessError(IngestionError):
    pass


type ProgressObserver = Callable[[IngestionProgress], None]
type CheckpointObserver = Callable[[IngestionCheckpoint], None]


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        writer: NamespaceWriter,
        *,
        batch_size: int = 32,
        max_concurrency: int = 4,
        readiness_attempts: int = 10,
        readiness_poll_interval: float = 0.5,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if readiness_attempts < 1:
            raise ValueError("readiness_attempts must be at least 1")
        if readiness_poll_interval < 0:
            raise ValueError("readiness_poll_interval must not be negative")
        self._embedder = embedder
        self._writer = writer
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._readiness_attempts = readiness_attempts
        self._readiness_poll_interval = readiness_poll_interval

    async def ingest(
        self,
        corpus: FixtureCorpus,
        *,
        namespace: str,
        on_progress: ProgressObserver | None = None,
        resume_from: IngestionCheckpoint | None = None,
        on_checkpoint: CheckpointObserver | None = None,
    ) -> IngestionReport:
        if not namespace.strip():
            raise ValueError("namespace must not be blank")
        if not corpus.documents:
            raise ValueError("corpus must contain at least one document")
        write_spec = compile_namespace_write_spec(corpus.manifest)
        expected_document_ids = frozenset(
            document_uuid(corpus.manifest.version, document.external_id)
            for document in corpus.documents
        )
        resumed_ids = self._validate_checkpoint(
            corpus,
            namespace=namespace,
            write_spec=write_spec,
            expected_document_ids=expected_document_ids,
            checkpoint=resume_from,
        )
        if self._embedder.dimensions != corpus.manifest.embedding.dimensions:
            report = self._report(corpus, namespace=namespace, state=IngestionState.FAILED)
            raise EmbeddingDimensionMismatch(
                "embedder dimensions do not match the dataset schema",
                report,
            )

        pending_documents = tuple(
            document
            for document in corpus.documents
            if document_uuid(corpus.manifest.version, document.external_id) not in resumed_ids
        )
        batches = _make_batches(pending_documents, self._batch_size)
        completed: list[BatchResult] = []
        checkpoint_ids = set(resumed_ids)
        self._emit(
            on_progress,
            self._progress(
                corpus,
                batches,
                completed,
                IngestionState.INGESTING,
                resumed_documents=len(resumed_ids),
            ),
        )

        for wave_start in range(0, len(batches), self._max_concurrency):
            wave = batches[wave_start : wave_start + self._max_concurrency]
            tasks = [
                self._ingest_batch(
                    corpus,
                    namespace=namespace,
                    write_spec=write_spec,
                    index=index,
                    documents=documents,
                )
                for index, documents in wave
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            first_error: BaseException | None = None
            for outcome in outcomes:
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                if isinstance(outcome, BaseException):
                    if first_error is None:
                        first_error = outcome
                else:
                    completed.append(outcome)
                    checkpoint_ids.update(outcome.document_ids)
                    if on_checkpoint is not None:
                        try:
                            on_checkpoint(
                                self._checkpoint(
                                    corpus,
                                    namespace=namespace,
                                    write_spec=write_spec,
                                    completed_document_ids=checkpoint_ids,
                                )
                            )
                        except Exception:
                            report = self._report(
                                corpus,
                                namespace=namespace,
                                state=IngestionState.FAILED,
                                batches=batches,
                                completed=completed,
                                resumed_documents=len(resumed_ids),
                            )
                            raise BatchWriteError(
                                "ingestion checkpoint could not be persisted", report
                            ) from None
                    self._emit(
                        on_progress,
                        self._progress(
                            corpus,
                            batches,
                            completed,
                            IngestionState.INGESTING,
                            resumed_documents=len(resumed_ids),
                        ),
                    )
            if first_error is not None:
                report = self._report(
                    corpus,
                    namespace=namespace,
                    state=IngestionState.FAILED,
                    batches=batches,
                    completed=completed,
                    resumed_documents=len(resumed_ids),
                )
                if isinstance(first_error, _EmbeddingCountError):
                    raise EmbeddingCountMismatch(str(first_error), report) from None
                if isinstance(first_error, _EmbeddingDimensionError):
                    raise EmbeddingDimensionMismatch(str(first_error), report) from None
                if isinstance(first_error, _EmbeddingValueError):
                    raise EmbeddingValueMismatch(str(first_error), report) from None
                raise BatchWriteError("a dataset batch failed to ingest", report) from None

        self._emit(
            on_progress,
            self._progress(
                corpus,
                batches,
                completed,
                IngestionState.VERIFYING,
                resumed_documents=len(resumed_ids),
            ),
        )
        readiness: NamespaceReadiness | None = None
        ready = False
        for attempt in range(self._readiness_attempts):
            inspection_failed = False
            try:
                readiness = await self._writer.inspect_readiness(
                    namespace,
                    expected_document_ids=expected_document_ids,
                )
            except Exception:
                inspection_failed = True
            if inspection_failed:
                report = self._report(
                    corpus,
                    namespace=namespace,
                    state=IngestionState.FAILED,
                    batches=batches,
                    completed=completed,
                    resumed_documents=len(resumed_ids),
                )
                raise ReadinessError("namespace readiness inspection failed", report) from None
            assert readiness is not None
            ready = self._is_ready(
                corpus,
                write_spec,
                expected_document_ids,
                readiness,
            )
            if ready:
                break
            if attempt + 1 < self._readiness_attempts:
                await asyncio.sleep(self._readiness_poll_interval)

        assert readiness is not None
        state = IngestionState.READY if ready else IngestionState.FAILED
        report = self._report(
            corpus,
            namespace=namespace,
            state=state,
            batches=batches,
            completed=completed,
            resumed_documents=len(resumed_ids),
            readiness=readiness,
        )
        self._emit(
            on_progress,
            self._progress(
                corpus,
                batches,
                completed,
                state,
                resumed_documents=len(resumed_ids),
            ),
        )
        if not ready:
            raise ReadinessError("namespace did not satisfy readiness checks", report)
        return report

    @staticmethod
    def _is_ready(
        corpus: FixtureCorpus,
        write_spec: NamespaceWriteSpec,
        expected_document_ids: frozenset[UUID],
        readiness: NamespaceReadiness,
    ) -> bool:
        return (
            readiness.metadata_ready
            and readiness.indexes_ready
            and readiness.document_count == len(corpus.documents)
            and readiness.document_ids == expected_document_ids
            and readiness.schema_hash == write_spec.schema_hash
        )

    async def _ingest_batch(
        self,
        corpus: FixtureCorpus,
        *,
        namespace: str,
        write_spec: NamespaceWriteSpec,
        index: int,
        documents: tuple[SourceDocument, ...],
    ) -> BatchResult:
        texts = tuple(f"{document.title}\n\n{document.body}" for document in documents)
        vectors = await self._embedder.embed(texts)
        if len(vectors) != len(documents):
            raise _EmbeddingCountError(
                f"embedding batch {index} returned {len(vectors)} vectors "
                f"for {len(documents)} documents"
            )
        expected_dimensions = corpus.manifest.embedding.dimensions
        normalized_vectors: list[tuple[float, ...]] = []
        for offset, vector in enumerate(vectors):
            if len(vector) != expected_dimensions:
                raise _EmbeddingDimensionError(
                    f"embedding batch {index} vector {offset} has {len(vector)} dimensions; "
                    f"expected {expected_dimensions}"
                )
            try:
                normalized = tuple(float(value) for value in vector)
            except (TypeError, ValueError, OverflowError):
                raise _EmbeddingValueError(
                    f"embedding batch {index} vector {offset} contains a non-numeric component"
                ) from None
            if not all(math.isfinite(value) for value in normalized):
                raise _EmbeddingValueError(
                    f"embedding batch {index} vector {offset} contains a non-finite component"
                )
            normalized_vectors.append(normalized)

        embedded = tuple(
            EmbeddedDocument(
                id=document_uuid(corpus.manifest.version, document.external_id),
                external_id=document.external_id,
                title=document.title,
                body=document.body,
                source_url=document.source_url,
                vector=vector,
            )
            for document, vector in zip(documents, normalized_vectors, strict=True)
        )
        await self._writer.upsert_batch(
            namespace,
            embedded,
            write_spec=write_spec,
        )
        return BatchResult(index=index, document_ids=tuple(document.id for document in embedded))

    @staticmethod
    def _emit(observer: ProgressObserver | None, progress: IngestionProgress) -> None:
        if observer is not None:
            observer(progress)

    @staticmethod
    def _progress(
        corpus: FixtureCorpus,
        batches: tuple[tuple[int, tuple[SourceDocument, ...]], ...],
        completed: list[BatchResult],
        state: IngestionState,
        *,
        resumed_documents: int,
    ) -> IngestionProgress:
        return IngestionProgress(
            state=state,
            batches_completed=len(completed),
            batches_total=len(batches),
            documents_completed=resumed_documents
            + sum(len(result.document_ids) for result in completed),
            documents_total=len(corpus.documents),
        )

    def _report(
        self,
        corpus: FixtureCorpus,
        *,
        namespace: str,
        state: IngestionState,
        batches: tuple[tuple[int, tuple[SourceDocument, ...]], ...] | None = None,
        completed: list[BatchResult] | None = None,
        resumed_documents: int = 0,
        readiness: NamespaceReadiness | None = None,
    ) -> IngestionReport:
        actual_batches = batches or _make_batches(corpus.documents, self._batch_size)
        actual_completed = tuple(sorted(completed or [], key=lambda item: item.index))
        return IngestionReport(
            namespace=namespace,
            dataset_version=corpus.manifest.version,
            corpus_hash=corpus.corpus_hash,
            schema_hash=compile_namespace_write_spec(corpus.manifest).schema_hash,
            state=state,
            batches_total=len(actual_batches),
            batches_completed=len(actual_completed),
            documents_total=len(corpus.documents),
            documents_completed=resumed_documents
            + sum(len(result.document_ids) for result in actual_completed),
            batch_results=actual_completed,
            resumed_documents=resumed_documents,
            readiness=readiness,
        )

    @staticmethod
    def _validate_checkpoint(
        corpus: FixtureCorpus,
        *,
        namespace: str,
        write_spec: NamespaceWriteSpec,
        expected_document_ids: frozenset[UUID],
        checkpoint: IngestionCheckpoint | None,
    ) -> frozenset[UUID]:
        if checkpoint is None:
            return frozenset()
        if checkpoint.format_version != 1:
            raise ValueError("ingestion checkpoint format is unsupported")
        if checkpoint.namespace != namespace:
            raise ValueError("ingestion checkpoint belongs to a different namespace")
        if (
            checkpoint.dataset_version != corpus.manifest.version
            or checkpoint.corpus_hash != corpus.corpus_hash
            or checkpoint.schema_hash != write_spec.schema_hash
        ):
            raise ValueError("ingestion checkpoint belongs to a different corpus revision")
        if len(checkpoint.completed_document_ids) != len(checkpoint.completed_ids):
            raise ValueError("ingestion checkpoint contains duplicate document IDs")
        if not checkpoint.completed_ids <= expected_document_ids:
            raise ValueError("ingestion checkpoint contains unknown document IDs")
        return checkpoint.completed_ids

    @staticmethod
    def _checkpoint(
        corpus: FixtureCorpus,
        *,
        namespace: str,
        write_spec: NamespaceWriteSpec,
        completed_document_ids: set[UUID],
    ) -> IngestionCheckpoint:
        return IngestionCheckpoint(
            format_version=1,
            namespace=namespace,
            dataset_version=corpus.manifest.version,
            corpus_hash=corpus.corpus_hash,
            schema_hash=write_spec.schema_hash,
            completed_document_ids=tuple(sorted(completed_document_ids, key=str)),
        )


class _EmbeddingCountError(RuntimeError):
    pass


class _EmbeddingDimensionError(RuntimeError):
    pass


class _EmbeddingValueError(RuntimeError):
    pass


def _make_batches(
    documents: tuple[SourceDocument, ...], batch_size: int
) -> tuple[tuple[int, tuple[SourceDocument, ...]], ...]:
    ordered = tuple(sorted(documents, key=lambda document: document.external_id))
    return tuple(
        (index, ordered[offset : offset + batch_size])
        for index, offset in enumerate(range(0, len(ordered), batch_size))
    )
