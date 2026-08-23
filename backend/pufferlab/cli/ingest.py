"""Single-command ingestion for the checked-in tiny fixture."""

from __future__ import annotations

import asyncio
import importlib.util
import re
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.datasets.embeddings import SentenceTransformerDocumentEmbedder
from pufferlab.datasets.ingestion import (
    Embedder,
    IngestionError,
    IngestionProgress,
    IngestionReport,
    IngestionService,
    NamespaceWriter,
)
from pufferlab.datasets.loader import DatasetLoadError, load_fixture_corpus
from pufferlab.datasets.models import FixtureCorpus
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec
from pufferlab.datasets.turbopuffer_writer import TurbopufferNamespaceWriter
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import (
    DistanceMetric,
    ProviderDocumentIdInventory,
    ProviderNamespaceMetadata,
    ProviderSchema,
    ProviderWriteResult,
    WriteDocument,
)

_OWNED_NAMESPACE_PATTERN = re.compile(r"pufferlab-[A-Za-z0-9](?:[A-Za-z0-9._-]{0,116}[A-Za-z0-9])?")
_GENERATED_NAMESPACE_PREFIX = "pufferlab-tiny-"
_EXPECTED_DATASET_SLUG = "pufferlab-tiny-unix"
_EXPECTED_DATASET_VERSION = "tiny-unix-v1"
_EXPECTED_CORPUS_HASH = "fc7817ade91368cef13c52e48d4dc154189ec3ba803a4e293407dd7c09242512"
_EXPECTED_SCHEMA_HASH = "0251f57f6166bf8f1ab8351ae0a4a797cfcf691fb0699bcfc59a4083945eea1d"
_EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"
_EXPECTED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_EXPECTED_DIMENSIONS = 384


def _token_hex(byte_count: int) -> str:
    return secrets.token_hex(byte_count)


@dataclass(frozen=True, slots=True)
class IngestTinyOptions:
    namespace: str | None = None
    batch_size: int = 20
    max_concurrency: int = 2
    readiness_attempts: int = 180
    readiness_poll_interval: float = 0.5


class TinyIngestionCommandError(RuntimeError):
    """A safe, user-facing ingestion failure with a stable process exit code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _IngestionProvider(Protocol):
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

    async def close(self) -> None: ...


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> _IngestionProvider: ...


class _EmbedderFactory(Protocol):
    def __call__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        batch_size: int,
    ) -> Embedder: ...


class _WriterFactory(Protocol):
    def __call__(self, provider: _IngestionProvider) -> NamespaceWriter: ...


class _TokenFactory(Protocol):
    def __call__(self, byte_count: int) -> str: ...


class TinyFixtureIngestor:
    """Wire the manifest, local embedder, provider, and ingestion service without HTTP."""

    def __init__(
        self,
        *,
        provider_factory: _ProviderFactory = TurbopufferProvider,
        embedder_factory: _EmbedderFactory = SentenceTransformerDocumentEmbedder,
        writer_factory: _WriterFactory | None = None,
        token_factory: _TokenFactory = _token_hex,
        optional_runtime_available: Callable[[], bool] | None = None,
    ) -> None:
        self._provider_factory = provider_factory
        self._embedder_factory = embedder_factory
        self._writer_factory = writer_factory or _make_writer
        self._token_factory = token_factory
        self._optional_runtime_available = (
            optional_runtime_available or _sentence_transformers_available
        )

    async def run(
        self,
        settings: Settings,
        options: IngestTinyOptions,
        *,
        emit: Callable[[str], None],
    ) -> IngestionReport:
        namespace = resolve_owned_namespace(options.namespace, token_factory=self._token_factory)
        corpus = _load_corpus(settings)
        api_key = _required_api_key(settings)
        if not self._optional_runtime_available():
            raise TinyIngestionCommandError(
                "sentence-transformers is not installed; run `uv sync --extra live-search`",
                exit_code=2,
            )
        write_spec = compile_namespace_write_spec(corpus.manifest)
        _emit_plan(
            emit,
            settings=settings,
            corpus=corpus,
            namespace=namespace,
            write_spec=write_spec,
        )

        provider: _IngestionProvider | None = None
        report: IngestionReport | None = None
        failure: TinyIngestionCommandError | None = None
        try:
            embedder = self._embedder_factory(
                model=corpus.manifest.embedding.model,
                revision=corpus.manifest.embedding.revision,
                dimensions=corpus.manifest.embedding.dimensions,
                batch_size=options.batch_size,
            )
            provider = self._provider_factory(
                api_key=api_key,
                region=settings.turbopuffer_region,
            )
            writer = self._writer_factory(provider)
            service = IngestionService(
                embedder,
                writer,
                batch_size=options.batch_size,
                max_concurrency=options.max_concurrency,
                readiness_attempts=options.readiness_attempts,
                readiness_poll_interval=options.readiness_poll_interval,
            )
            report = await service.ingest(
                corpus,
                namespace=namespace,
                on_progress=_CompactProgress(emit),
            )
        except asyncio.CancelledError:
            raise
        except IngestionError:
            failure = TinyIngestionCommandError(
                "tiny fixture ingestion failed before readiness was verified"
            )
        except Exception:
            failure = TinyIngestionCommandError("tiny fixture ingestion runtime failed")
        finally:
            if provider is not None:
                try:
                    await provider.close()
                except Exception:
                    if failure is None:
                        failure = TinyIngestionCommandError(
                            "tiny fixture ingestion runtime did not close cleanly"
                        )

        if failure is not None:
            raise failure from None
        assert report is not None
        readiness = report.readiness
        assert readiness is not None
        exact_document_ids = len(readiness.document_ids) == len(corpus.documents)
        emit(
            f"verified remote_documents={readiness.document_count} "
            f"exact_document_ids={str(exact_document_ids).lower()} "
            f"observed_schema_hash={readiness.schema_hash} "
            f"distance_metric={corpus.manifest.vector.distance_metric} "
            f"metadata_ready={str(readiness.metadata_ready).lower()} "
            f"indexes_ready={str(readiness.indexes_ready).lower()}"
        )
        emit(
            f"ready namespace={report.namespace} documents={report.documents_completed} "
            f"schema_hash={report.schema_hash}"
        )
        emit(f"PUFFERLAB_SEARCH_NAMESPACE={report.namespace}")
        return report


class _CompactProgress:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._last: tuple[object, ...] | None = None

    def __call__(self, progress: IngestionProgress) -> None:
        current = (
            progress.state,
            progress.batches_completed,
            progress.documents_completed,
        )
        if current == self._last:
            return
        self._last = current
        self._emit(
            f"progress state={progress.state.value} "
            f"batches={progress.batches_completed}/{progress.batches_total} "
            f"documents={progress.documents_completed}/{progress.documents_total}"
        )


def resolve_owned_namespace(
    namespace: str | None,
    *,
    token_factory: _TokenFactory = _token_hex,
    generated_prefix: str = _GENERATED_NAMESPACE_PREFIX,
) -> str:
    resolved = f"{generated_prefix}{token_factory(12)}" if namespace is None else namespace
    if _OWNED_NAMESPACE_PATTERN.fullmatch(resolved) is None:
        raise TinyIngestionCommandError(
            "namespace must be an owned pufferlab-* name using 1-128 letters, digits, '-', '_', "
            "or '.' characters",
            exit_code=2,
        )
    return resolved


def _load_corpus(settings: Settings) -> FixtureCorpus:
    try:
        corpus = load_fixture_corpus(settings.pufferlab_fixture_dir)
    except (DatasetLoadError, OSError):
        raise TinyIngestionCommandError(
            "the checked-in tiny fixture could not be loaded",
            exit_code=2,
        ) from None
    manifest = corpus.manifest
    exact_identity = (
        manifest.slug == _EXPECTED_DATASET_SLUG
        and manifest.version == _EXPECTED_DATASET_VERSION
        and corpus.corpus_hash == _EXPECTED_CORPUS_HASH
        and len(corpus.documents) == 20
        and manifest.embedding.model == _EXPECTED_MODEL
        and manifest.embedding.revision == _EXPECTED_REVISION
        and manifest.embedding.dimensions == _EXPECTED_DIMENSIONS
        and compile_namespace_write_spec(manifest).schema_hash == _EXPECTED_SCHEMA_HASH
    )
    if not exact_identity:
        raise TinyIngestionCommandError(
            "the tiny fixture does not match the expected pinned corpus, schema, and embedding "
            "profile",
            exit_code=2,
        )
    return corpus


def _required_api_key(settings: Settings) -> str:
    secret = settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        raise TinyIngestionCommandError(
            "TURBOPUFFER_API_KEY is required for dataset ingestion",
            exit_code=2,
        )
    return secret.get_secret_value()


def _emit_plan(
    emit: Callable[[str], None],
    *,
    settings: Settings,
    corpus: FixtureCorpus,
    namespace: str,
    write_spec: NamespaceWriteSpec,
) -> None:
    emit("ingestion plan (local model execution and remote writes follow)")
    emit(f"region={settings.turbopuffer_region}")
    emit(f"namespace={namespace}")
    emit(f"schema_hash={write_spec.schema_hash}")
    emit(f"documents={len(corpus.documents)}")
    emit(
        f"embedding_model={corpus.manifest.embedding.model} "
        f"revision={corpus.manifest.embedding.revision} "
        f"dimensions={corpus.manifest.embedding.dimensions}"
    )


def _make_writer(provider: _IngestionProvider) -> NamespaceWriter:
    return TurbopufferNamespaceWriter(provider)


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None
