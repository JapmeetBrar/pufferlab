import traceback
from pathlib import Path

import pytest
from pufferlab.config import Settings
from pufferlab.contracts.search import SearchCompareRequest
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.providers.types import ProviderQueryResult
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from pufferlab.retrieval.types import QueryEmbedding

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "fixtures" / "tiny-corpus"


class FakeProvider:
    def __init__(self) -> None:
        self.bm25_calls: list[dict[str, object]] = []
        self.ann_calls: list[dict[str, object]] = []
        self.closed = False

    async def query_bm25(self, **kwargs: object) -> ProviderQueryResult:
        self.bm25_calls.append(kwargs)
        return ProviderQueryResult(documents=(), client_duration_ms=1.0)

    async def query_ann(self, **kwargs: object) -> ProviderQueryResult:
        self.ann_calls.append(kwargs)
        return ProviderQueryResult(documents=(), client_duration_ms=2.0)

    async def close(self) -> None:
        self.closed = True


class FakeProviderFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.provider = FakeProvider()

    def __call__(self, *, api_key: str, region: str) -> FakeProvider:
        self.calls.append((api_key, region))
        return self.provider


class FakeEmbedder:
    def __init__(self, *, model: str, revision: str, dimensions: int) -> None:
        self.model = model
        self.revision = revision
        self.dimensions = dimensions
        self.queries: list[str] = []

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        self.queries.append(query_text)
        return QueryEmbedding(
            vector=tuple(0.0 for _ in range(self.dimensions)),
            client_duration_ms=0.5,
        )


class FakeEmbedderFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.embedder: FakeEmbedder | None = None

    def __call__(self, *, model: str, revision: str, dimensions: int) -> FakeEmbedder:
        self.calls.append((model, revision, dimensions))
        self.embedder = FakeEmbedder(model=model, revision=revision, dimensions=dimensions)
        return self.embedder


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "pufferlab_fixture_dir": FIXTURE_DIR,
        "pufferlab_search_namespace": "pufferlab-runtime-test",
        "turbopuffer_api_key": "server-only-test-value",
        "turbopuffer_region": "gcp-us-central1",
    }
    values.update(overrides)
    return Settings.model_validate(values)


@pytest.mark.asyncio
async def test_runtime_uses_exact_manifest_and_lazily_reuses_clients() -> None:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    provider_factory = FakeProviderFactory()
    embedder_factory = FakeEmbedderFactory()
    runtime = RuntimeSearchBackend(
        settings=_settings(),
        manifest=corpus.manifest,
        provider_factory=provider_factory,
        embedder_factory=embedder_factory,
    )
    bm25, vector = runtime.list_configs()

    assert provider_factory.calls == []
    assert embedder_factory.calls == []
    request = SearchCompareRequest(
        query_text="how do pipes work",
        config_ids=[bm25.id, vector.id],
    )
    first = await runtime.compare(request)
    second = await runtime.compare(request)

    assert provider_factory.calls == [("server-only-test-value", "gcp-us-central1")]
    assert embedder_factory.calls == [
        (
            corpus.manifest.embedding.model,
            corpus.manifest.embedding.revision,
            corpus.manifest.embedding.dimensions,
        )
    ]
    assert len(provider_factory.provider.bm25_calls) == 2
    assert len(provider_factory.provider.ann_calls) == 2
    assert provider_factory.provider.ann_calls[0]["vector_attribute"] == (
        corpus.manifest.vector.attribute
    )
    assert provider_factory.provider.ann_calls[0]["distance_metric"] == (
        corpus.manifest.vector.distance_metric
    )
    assert first.model_dump(mode="json") != {}
    assert second.results[1].timings[0].duration_ms == 0.5
    assert embedder_factory.embedder is not None
    assert embedder_factory.embedder.queries == ["how do pipes work", "how do pipes work"]

    await runtime.close()
    assert provider_factory.provider.closed


@pytest.mark.asyncio
async def test_runtime_discovers_configs_without_server_credentials() -> None:
    runtime = RuntimeSearchBackend.from_settings(
        _settings(turbopuffer_api_key=None, pufferlab_search_namespace=None)
    )
    summaries = runtime.list_configs()

    assert len(summaries) == 2
    with pytest.raises(SearchError, match="not configured") as caught:
        await runtime.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[summary.id for summary in summaries],
            )
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


class SecretProviderFactory(FakeProviderFactory):
    def __call__(self, *, api_key: str, region: str) -> FakeProvider:
        del api_key, region
        raise RuntimeError("factory-secret-marker")


@pytest.mark.asyncio
async def test_runtime_factory_failures_are_detached_and_redacted() -> None:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    runtime = RuntimeSearchBackend(
        settings=_settings(),
        manifest=corpus.manifest,
        provider_factory=SecretProviderFactory(),
        embedder_factory=FakeEmbedderFactory(),
    )
    summaries = runtime.list_configs()

    with pytest.raises(SearchError) as caught:
        await runtime.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[summary.id for summary in summaries],
            )
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "factory-secret-marker" not in repr(caught.value)
    assert "factory-secret-marker" not in formatted
