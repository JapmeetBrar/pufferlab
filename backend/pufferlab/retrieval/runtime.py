"""Lazy production wiring for the checked-in fixture and real provider."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.providers.rerankers import (
    Reranker,
    SentenceTransformersReranker,
)
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.retrieval.config import BoundSearchCatalog, build_search_catalog
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder
from pufferlab.retrieval.errors import SearchError, invalid_search, search_unavailable
from pufferlab.retrieval.filter_validation import FixtureFilterValidator
from pufferlab.retrieval.service import SearchCompareService
from pufferlab.retrieval.types import (
    HybridProbeExecuteRequest,
    HybridProbeExecuteResult,
    QueryEmbedder,
    RetrievalProvider,
    SearchExecuteRequest,
    SearchExecuteResult,
)


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


class _RerankerFactory(Protocol):
    def __call__(self, *, model: str, revision: str) -> Reranker: ...


class RuntimeSearchBackend:
    """Expose config discovery eagerly and initialize network/model clients on first compare."""

    def __init__(
        self,
        *,
        settings: Settings,
        manifest: DatasetManifest,
        bound_catalog: BoundSearchCatalog | None = None,
        provider_factory: _ProviderFactory | None = None,
        embedder_factory: _EmbedderFactory | None = None,
        reranker_factory: _RerankerFactory | None = None,
    ) -> None:
        self._settings = settings
        self._manifest = manifest
        self._write_spec = compile_namespace_write_spec(manifest)
        self._filter_validator = FixtureFilterValidator(self._write_spec)
        if bound_catalog is not None:
            if bound_catalog.manifest != manifest:
                raise ValueError("bound retrieval catalog manifest does not match runtime manifest")
            configured_namespace = settings.pufferlab_search_namespace
            if configured_namespace != bound_catalog.dataset_version.namespace:
                raise ValueError(
                    "bound retrieval catalog namespace does not match runtime namespace"
                )
            self._catalog = bound_catalog.catalog
        else:
            self._catalog = build_search_catalog(manifest)
        self._bound_catalog = bound_catalog
        reranker_config = self._catalog.configs[-1]
        if reranker_config.reranker_model is None or reranker_config.reranker_revision is None:
            raise ValueError("retrieval catalog is missing its reranker model identity")
        self._reranker_model = reranker_config.reranker_model
        self._reranker_revision = reranker_config.reranker_revision
        self._provider_factory: _ProviderFactory = provider_factory or TurbopufferProvider
        self._embedder_factory: _EmbedderFactory = (
            embedder_factory or SentenceTransformerQueryEmbedder
        )
        self._reranker_factory: _RerankerFactory = reranker_factory or SentenceTransformersReranker
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
        if request.filter_override is not None:
            self._filter_validator.validate(request.filter_override)
        service = await self._get_service()
        return await service.compare(request)

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        if request.filter_override is not None:
            self._filter_validator.validate(request.filter_override)
        configured_namespace = self._settings.pufferlab_search_namespace
        if configured_namespace is None or not configured_namespace.strip():
            raise search_unavailable()
        if request.namespace != configured_namespace:
            raise invalid_search("execution namespace does not match the configured namespace")
        service = await self._get_service()
        return await service.search_one(request)

    async def probe_hybrid_candidates(
        self,
        request: HybridProbeExecuteRequest,
    ) -> HybridProbeExecuteResult:
        if request.filter_override is not None:
            self._filter_validator.validate(request.filter_override)
        configured_namespace = self._settings.pufferlab_search_namespace
        if configured_namespace is None or not configured_namespace.strip():
            raise search_unavailable()
        if request.namespace != configured_namespace:
            raise invalid_search("execution namespace does not match the configured namespace")
        service = await self._get_service()
        return await service.probe_hybrid_candidates(request)

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
                self._service = await self._build_service()
        return self._service

    async def _build_service(self) -> SearchCompareService:
        namespace = self._settings.pufferlab_search_namespace
        api_key = self._settings.turbopuffer_api_key
        if namespace is None or not namespace.strip() or api_key is None:
            raise search_unavailable()
        secret = api_key.get_secret_value()
        if not secret:
            raise search_unavailable()

        failure: SearchError | None = None
        service: SearchCompareService | None = None
        provider: RetrievalProvider | None = None
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
            reranker = self._reranker_factory(
                model=self._reranker_model,
                revision=self._reranker_revision,
            )
            service = SearchCompareService(
                namespace=namespace,
                catalog=self._catalog,
                write_spec=self._write_spec,
                provider=provider,
                query_embedder=embedder,
                reranker=reranker,
            )
        except Exception:
            failure = search_unavailable()
        if failure is not None:
            if provider is not None:
                with suppress(Exception):
                    await provider.close()
            raise failure from None
        assert service is not None
        return service
