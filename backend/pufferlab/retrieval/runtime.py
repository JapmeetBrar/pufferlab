"""Lazy production wiring for the checked-in fixture and real provider."""

from __future__ import annotations

import asyncio
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.retrieval.config import SearchConfigCatalog, build_search_catalog
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder
from pufferlab.retrieval.errors import SearchError, search_unavailable
from pufferlab.retrieval.service import SearchCompareService
from pufferlab.retrieval.types import QueryEmbedder, RetrievalProvider


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> RetrievalProvider: ...


class _EmbedderFactory(Protocol):
    def __call__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
    ) -> QueryEmbedder: ...


class RuntimeSearchBackend:
    """Expose config discovery eagerly and initialize network/model clients on first compare."""

    def __init__(
        self,
        *,
        settings: Settings,
        manifest: DatasetManifest,
        provider_factory: _ProviderFactory | None = None,
        embedder_factory: _EmbedderFactory | None = None,
    ) -> None:
        self._settings = settings
        self._manifest = manifest
        self._catalog: SearchConfigCatalog = build_search_catalog(manifest)
        self._provider_factory: _ProviderFactory = provider_factory or TurbopufferProvider
        self._embedder_factory: _EmbedderFactory = (
            embedder_factory or SentenceTransformerQueryEmbedder
        )
        self._service: SearchCompareService | None = None
        self._service_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings) -> RuntimeSearchBackend:
        corpus = load_fixture_corpus(settings.pufferlab_fixture_dir)
        return cls(settings=settings, manifest=corpus.manifest)

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return self._catalog.summaries()

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        service = await self._get_service()
        return await service.compare(request)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._service is not None:
            await self._service.close()

    async def _get_service(self) -> SearchCompareService:
        if self._closed:
            raise search_unavailable()
        if self._service is not None:
            return self._service
        async with self._service_lock:
            if self._service is None:
                self._service = self._build_service()
        return self._service

    def _build_service(self) -> SearchCompareService:
        namespace = self._settings.pufferlab_search_namespace
        api_key = self._settings.turbopuffer_api_key
        if namespace is None or not namespace.strip() or api_key is None:
            raise search_unavailable()
        secret = api_key.get_secret_value()
        if not secret:
            raise search_unavailable()

        failure: SearchError | None = None
        service: SearchCompareService | None = None
        try:
            embedder = self._embedder_factory(
                model=self._manifest.embedding.model,
                revision=self._manifest.embedding.revision,
                dimensions=self._manifest.embedding.dimensions,
            )
            provider = self._provider_factory(
                api_key=secret,
                region=self._settings.turbopuffer_region,
            )
            service = SearchCompareService(
                namespace=namespace,
                catalog=self._catalog,
                write_spec=compile_namespace_write_spec(self._manifest),
                provider=provider,
                query_embedder=embedder,
            )
        except Exception:
            failure = search_unavailable()
        if failure is not None:
            raise failure from None
        assert service is not None
        return service
