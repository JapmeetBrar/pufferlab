import traceback
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pufferlab.config import Settings
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.filters import FilterLogical, FilterPredicate, LogicalOp, PredicateOp
from pufferlab.contracts.search import SearchCompareRequest
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.main import create_app
from pufferlab.providers.rerankers import Reranker
from pufferlab.providers.types import ProviderQueryResult
from pufferlab.retrieval.config import bind_retrieval_catalog
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from pufferlab.retrieval.types import QueryEmbedding, SearchExecuteRequest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "fixtures" / "tiny-corpus"


class FakeProvider:
    def __init__(self) -> None:
        self.bm25_calls: list[dict[str, object]] = []
        self.ann_calls: list[dict[str, object]] = []
        self.closed = False
        self.close_calls = 0

    async def query_bm25(self, **kwargs: object) -> ProviderQueryResult:
        self.bm25_calls.append(kwargs)
        return ProviderQueryResult(documents=(), client_duration_ms=1.0)

    async def query_ann(self, **kwargs: object) -> ProviderQueryResult:
        self.ann_calls.append(kwargs)
        return ProviderQueryResult(documents=(), client_duration_ms=2.0)

    async def close(self) -> None:
        self.close_calls += 1
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


def _bound_dataset_version() -> tuple[DatasetManifest, DatasetVersion]:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    manifest = corpus.manifest
    write_spec = compile_namespace_write_spec(manifest)
    dataset = DatasetVersion(
        id=UUID("8f8248a2-ef60-4c57-861e-5e94782e9907"),
        slug=manifest.slug,
        version=manifest.version,
        namespace="pufferlab-bound-runtime",
        index_profile=IndexProfile(
            id=f"{manifest.slug}-{write_spec.schema_hash[:16]}",
            embedding_provider=manifest.embedding.provider,
            embedding_model=manifest.embedding.model,
            embedding_revision=manifest.embedding.revision,
            vector_attribute=manifest.vector.attribute,
            vector_dimensions=manifest.embedding.dimensions,
            vector_dtype=manifest.vector.dtype,
            distance_metric=manifest.vector.distance_metric,
            fts_profile=FtsProfile(
                tokenizer=manifest.fts.tokenizer,
                case_sensitive=manifest.fts.case_sensitive,
                language=manifest.fts.language,
                stemming=manifest.fts.stemming,
                remove_stopwords=manifest.fts.remove_stopwords,
                ascii_folding=manifest.fts.ascii_folding,
                max_token_length=manifest.fts.max_token_length,
                k1=manifest.fts.k1,
                b=manifest.fts.b,
                k3=manifest.fts.k3,
            ),
            schema_hash=write_spec.schema_hash,
        ),
        document_count=len(corpus.documents),
        corpus_hash=corpus.corpus_hash,
        status=DatasetStatus.READY,
        created_at=datetime(2014, 9, 26, tzinfo=UTC),
    )
    return manifest, dataset


def _runtime_without_credentials() -> tuple[
    RuntimeSearchBackend,
    FakeProviderFactory,
    FakeEmbedderFactory,
]:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    provider_factory = FakeProviderFactory()
    embedder_factory = FakeEmbedderFactory()
    runtime = RuntimeSearchBackend(
        settings=_settings(turbopuffer_api_key=None, pufferlab_search_namespace=None),
        manifest=corpus.manifest,
        provider_factory=provider_factory,
        embedder_factory=embedder_factory,
    )
    return runtime, provider_factory, embedder_factory


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
    bm25, vector, _, _ = runtime.list_configs()

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

    assert len(summaries) == 4
    with pytest.raises(SearchError, match="not configured") as caught:
        await runtime.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[summary.id for summary in summaries],
            )
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_runtime_executes_one_config_and_rejects_a_different_namespace() -> None:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    provider_factory = FakeProviderFactory()
    runtime = RuntimeSearchBackend(
        settings=_settings(),
        manifest=corpus.manifest,
        provider_factory=provider_factory,
        embedder_factory=FakeEmbedderFactory(),
    )
    bm25, _, _, _ = runtime.list_configs()

    result = await runtime.search_one(
        SearchExecuteRequest(
            namespace="pufferlab-runtime-test",
            query_text="query",
            config_id=bm25.id,
        )
    )

    assert result.config_id == bm25.id
    assert result.result.config == bm25
    assert provider_factory.provider.bm25_calls[0]["top_k"] == 10
    with pytest.raises(SearchError, match="namespace"):
        await runtime.search_one(
            SearchExecuteRequest(
                namespace="pufferlab-another-namespace",
                query_text="query",
                config_id=bm25.id,
            )
        )
    assert len(provider_factory.provider.bm25_calls) == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_accepts_only_the_complete_bound_evaluation_catalog() -> None:
    manifest, dataset = _bound_dataset_version()
    bound = bind_retrieval_catalog(dataset, manifest)
    provider_factory = FakeProviderFactory()
    runtime = RuntimeSearchBackend(
        settings=_settings(pufferlab_search_namespace=dataset.namespace),
        manifest=manifest,
        bound_catalog=bound,
        provider_factory=provider_factory,
        embedder_factory=FakeEmbedderFactory(),
    )
    bm25, _, _, _ = runtime.list_configs()

    result = await runtime.search_one(
        SearchExecuteRequest(
            namespace=dataset.namespace,
            query_text="query",
            config_id=bm25.id,
        )
    )

    assert result.result.config == bound.catalog.summaries()[0]
    assert provider_factory.provider.bm25_calls[0]["top_k"] == 50
    assert "catalog" not in signature(RuntimeSearchBackend).parameters
    assert "bound_catalog" in signature(RuntimeSearchBackend).parameters

    with pytest.raises(ValueError, match="namespace"):
        RuntimeSearchBackend(
            settings=_settings(pufferlab_search_namespace="pufferlab-wrong"),
            manifest=manifest,
            bound_catalog=bound,
        )
    with pytest.raises(ValueError, match="manifest"):
        RuntimeSearchBackend(
            settings=_settings(pufferlab_search_namespace=dataset.namespace),
            manifest=manifest.model_copy(update={"title": "different manifest"}),
            bound_catalog=bound,
        )
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_override",
    [
        FilterPredicate(field="runtime-secret-field", op=PredicateOp.EQ, value="value"),
        FilterPredicate(field="title", op=PredicateOp.EQ, value="value"),
        FilterLogical(
            op=LogicalOp.AND,
            children=[
                FilterPredicate(field="external_id", op=PredicateOp.EQ, value="valid"),
                FilterLogical(
                    op=LogicalOp.NOT,
                    children=[FilterPredicate(field="body", op=PredicateOp.EQ, value="invalid")],
                ),
            ],
        ),
    ],
)
async def test_runtime_validates_filters_before_credentials_or_factories(
    filter_override: FilterPredicate | FilterLogical,
) -> None:
    runtime, provider_factory, embedder_factory = _runtime_without_credentials()
    summaries = runtime.list_configs()

    with pytest.raises(SearchError) as caught:
        await runtime.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[summary.id for summary in summaries],
                filter_override=filter_override,
            )
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.details.http_status == 422
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "runtime-secret-field" not in repr(caught.value)
    assert "runtime-secret-field" not in formatted
    assert provider_factory.calls == []
    assert embedder_factory.calls == []


def test_runtime_api_orders_filter_validation_before_missing_configuration() -> None:
    runtime, provider_factory, embedder_factory = _runtime_without_credentials()
    config_ids = [str(summary.id) for summary in runtime.list_configs()]
    invalid_body = {
        "query_text": "query",
        "config_ids": config_ids,
        "filter_override": {
            "kind": "predicate",
            "field": "runtime-api-secret-field",
            "op": "eq",
            "value": "value",
        },
    }
    valid_body = {
        **invalid_body,
        "filter_override": {
            "kind": "predicate",
            "field": "external_id",
            "op": "eq",
            "value": "doc-001",
        },
    }

    with TestClient(create_app(search_backend=runtime)) as client:
        invalid_response = client.post("/api/v1/search/compare", json=invalid_body)
        valid_response = client.post("/api/v1/search/compare", json=valid_body)

    assert invalid_response.status_code == 422
    assert invalid_response.json()["code"] == "validation_error"
    assert invalid_response.json()["message"] == "filter field is not available"
    assert invalid_response.json()["details"] == {"operation": "compare"}
    assert "detail" not in invalid_response.json()
    assert "runtime-api-secret-field" not in invalid_response.text
    assert valid_response.status_code == 503
    assert valid_response.json()["message"] == "search backend is not configured"
    assert provider_factory.calls == []
    assert embedder_factory.calls == []


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


class SecretRerankerFactory:
    def __call__(self, *, model: str, revision: str) -> Reranker:
        del model, revision
        raise RuntimeError("reranker-factory-secret-marker")


@pytest.mark.asyncio
async def test_runtime_closes_partially_constructed_provider_exactly_once() -> None:
    corpus = load_fixture_corpus(FIXTURE_DIR)
    provider_factory = FakeProviderFactory()
    runtime = RuntimeSearchBackend(
        settings=_settings(),
        manifest=corpus.manifest,
        provider_factory=provider_factory,
        embedder_factory=FakeEmbedderFactory(),
        reranker_factory=SecretRerankerFactory(),
    )
    bm25, vector, _, _ = runtime.list_configs()

    with pytest.raises(SearchError) as caught:
        await runtime.compare(
            SearchCompareRequest(
                query_text="query",
                config_ids=[bm25.id, vector.id],
            )
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "reranker-factory-secret-marker" not in repr(caught.value)
    assert "reranker-factory-secret-marker" not in formatted
    assert provider_factory.provider.close_calls == 1

    await runtime.close()
    assert provider_factory.provider.close_calls == 1
