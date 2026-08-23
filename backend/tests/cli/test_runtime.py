from __future__ import annotations

import asyncio
import io
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pufferlab.cli.evaluation import EvalRunOptions, UnixIngestOptions
from pufferlab.cli.main import main
from pufferlab.cli.runtime import RuntimeCliApplication
from pufferlab.config import Settings
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import EvalRun, JudgedQuery, Qrel, QuerySet
from pufferlab.contracts.retrieval import RetrievalConfigSummary
from pufferlab.contracts.search import (
    ConfigSearchResult,
    SearchCompareRequest,
    SearchCompareResponse,
)
from pufferlab.datasets import load_unix_dataset_manifest
from pufferlab.datasets.ingestion import (
    IngestionProgress,
    IngestionReport,
    IngestionState,
    NamespaceReadiness,
)
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.datasets.unix_application import (
    CuratedJudgedQuerySeed,
    UnixEvaluationSeed,
    UnixIngestionResult,
)
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.retrieval.config import BoundSearchCatalog
from pufferlab.retrieval.types import SearchExecuteRequest, SearchExecuteResult

ROOT = Path(__file__).resolve().parents[3]
UNIX_MANIFEST = ROOT / "datasets" / "cqadupstack-unix" / "dataset-manifest.json"
IDENTITY_NAMESPACE = UUID("50a825c2-56fa-4a83-a5f5-565e1b45b690")
CREATED_AT = datetime(2014, 9, 26, tzinfo=UTC)


def _id(name: str) -> UUID:
    return uuid5(IDENTITY_NAMESPACE, name)


def _settings(tmp_path: Path, *, api_key: str | None = "server-secret") -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_data_dir": tmp_path / "data",
            "turbopuffer_api_key": api_key,
            "turbopuffer_region": "gcp-us-west1",
        }
    )


def _unix_seed(manifest: DatasetManifest, *, namespace: str) -> UnixEvaluationSeed:
    write_spec = compile_namespace_write_spec(manifest)
    dataset = DatasetVersion(
        id=_id(f"dataset:{namespace}"),
        slug=manifest.slug,
        version=manifest.version,
        namespace=namespace,
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
        document_count=100,
        corpus_hash="synthetic-corpus-hash",
        status=DatasetStatus.READY,
        created_at=CREATED_AT,
    )
    curated: list[CuratedJudgedQuerySeed] = []
    for index in range(50):
        query = JudgedQuery(
            id=_id(f"{namespace}:query:{index:02d}"),
            external_id=f"synthetic-{index:02d}",
            text=f"synthetic Unix query {index:02d}",
            tags=["hybrid"],
            qrels=[
                Qrel(
                    document_id=_id(f"{namespace}:document:{index:02d}"),
                    relevance_grade=2,
                )
            ],
        )
        curated.append(
            CuratedJudgedQuerySeed(
                judged_query=query,
                primary_tag="hybrid",
                tags=("hybrid",),
                reason="synthetic runtime composition coverage",
            )
        )
    query_set = QuerySet(
        id=_id(f"query-set:{namespace}"),
        name="synthetic curated 50",
        version="synthetic-v1",
        dataset_version_id=dataset.id,
        query_count=50,
        content_hash="synthetic-query-set-hash",
        created_at=CREATED_AT,
    )
    return UnixEvaluationSeed(
        dataset_version=dataset,
        query_set=query_set,
        curated_queries=tuple(curated),
    )


def _persist_unconfigured_seed(settings: Settings, seed: UnixEvaluationSeed) -> None:
    with Database.from_settings(settings) as database:
        database.migrate()
        repository = PufferLabRepository(database.session_factory)
        repository.put_dataset_version(seed.dataset_version)
        repository.put_query_set(seed.query_set, seed.judged_queries)


class FakeSearchBackend:
    def __init__(self, bound: BoundSearchCatalog) -> None:
        self._summaries = bound.catalog.summaries()
        self.calls: list[SearchExecuteRequest] = []
        self.closed = False

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return self._summaries

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise NotImplementedError

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        self.calls.append(request)
        summary = next(config for config in self._summaries if config.id == request.config_id)
        return SearchExecuteResult(
            config_id=request.config_id,
            query_id=request.query_id,
            result=ConfigSearchResult(
                config=summary,
                hits=[],
                timings=[],
                candidate_counts={},
                warnings=[],
                trace_id=_id(f"trace:{request.config_id}:{request.query_id}"),
            ),
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_default_cli_seed_then_runtime_run_and_export_use_persisted_bound_suite(
    tmp_path: Path,
) -> None:
    manifest = load_unix_dataset_manifest(UNIX_MANIFEST)
    seed = _unix_seed(manifest, namespace="pufferlab-unix-runtime")
    settings = _settings(tmp_path)
    _persist_unconfigured_seed(settings, seed)

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = await asyncio.to_thread(
        main,
        ["config", "seed"],
        settings_factory=lambda: settings,
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert stdout.getvalue().count("config ordinal=") == 4

    backends: list[FakeSearchBackend] = []

    def make_search_backend(
        runtime_settings: Settings,
        runtime_manifest: DatasetManifest,
        bound: BoundSearchCatalog,
    ) -> FakeSearchBackend:
        assert runtime_settings.pufferlab_search_namespace == seed.dataset_version.namespace
        assert runtime_manifest == manifest
        backend = FakeSearchBackend(bound)
        backends.append(backend)
        return backend

    application = RuntimeCliApplication(
        settings,
        unix_manifest_path=UNIX_MANIFEST,
        search_backend_factory=make_search_backend,
        git_revision_factory=lambda: "a" * 40,
        optional_runtime_available=lambda: True,
    )
    progress: list[int] = []

    async def observe(run: EvalRun) -> None:
        progress.append(run.completed_queries)

    completed = await application.run(
        EvalRunOptions(
            query_set_id=None,
            baseline_config_id=None,
            candidate_config_ids=(),
            seeded_defaults=True,
            random_seed=7,
            max_concurrency=3,
            warmup_query_count=0,
        ),
        run_id=_id("runtime-run"),
        on_progress=observe,
    )
    exported = application.export(completed.id)
    await application.close()

    assert completed.status.value == "completed"
    assert completed.completed_queries == 50
    assert len(completed.summaries) == 4
    assert len(exported.outcomes) == 200
    assert progress[-1] == 50
    assert len(backends) == 1
    assert len(backends[0].calls) == 200
    assert backends[0].closed
    assert all(record.outcome.kind == "success" for record in exported.outcomes)


class FakeProvider:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.0] * self.dimensions for _ in texts]


@pytest.mark.asyncio
async def test_unix_ingest_composition_persists_seed_and_closes_both_provider_boundaries(
    tmp_path: Path,
) -> None:
    manifest = load_unix_dataset_manifest(UNIX_MANIFEST)
    seed = _unix_seed(manifest, namespace="pufferlab-unix-ingest-runtime")
    readiness = NamespaceReadiness(
        document_count=seed.dataset_version.document_count,
        document_ids=frozenset(),
        schema_hash=seed.dataset_version.index_profile.schema_hash,
        metadata_ready=True,
        indexes_ready=True,
    )
    report = IngestionReport(
        namespace=seed.dataset_version.namespace,
        dataset_version=seed.dataset_version.version,
        corpus_hash=seed.dataset_version.corpus_hash,
        schema_hash=seed.dataset_version.index_profile.schema_hash,
        state=IngestionState.READY,
        batches_total=1,
        batches_completed=1,
        documents_total=seed.dataset_version.document_count,
        documents_completed=seed.dataset_version.document_count,
        batch_results=(),
        readiness=readiness,
    )
    provider = FakeProvider()
    backends: list[FakeSearchBackend] = []
    emitted: list[str] = []

    class FakeUnixService:
        async def ingest(
            self,
            *,
            namespace: str,
            on_progress: Callable[[IngestionProgress], None] | None = None,
        ) -> UnixIngestionResult:
            assert namespace == seed.dataset_version.namespace
            if on_progress is not None:
                on_progress(
                    IngestionProgress(
                        state=IngestionState.READY,
                        batches_completed=1,
                        batches_total=1,
                        documents_completed=100,
                        documents_total=100,
                    )
                )
            return UnixIngestionResult(report=report, evaluation_seed=seed)

    def make_unix_service(
        ingestion: object,
        checkpoints: object,
        options: UnixIngestOptions,
    ) -> FakeUnixService:
        assert ingestion is not None
        assert checkpoints is not None
        assert options.namespace == seed.dataset_version.namespace
        return FakeUnixService()

    def make_search_backend(
        runtime_settings: Settings,
        runtime_manifest: DatasetManifest,
        bound: BoundSearchCatalog,
    ) -> FakeSearchBackend:
        assert runtime_settings.pufferlab_search_namespace == seed.dataset_version.namespace
        assert runtime_manifest == manifest
        backend = FakeSearchBackend(bound)
        backends.append(backend)
        return backend

    application = RuntimeCliApplication(
        _settings(tmp_path),
        unix_manifest_path=UNIX_MANIFEST,
        ingestion_provider_factory=lambda **_: provider,  # type: ignore[arg-type]
        document_embedder_factory=lambda **values: FakeEmbedder(values["dimensions"]),
        unix_service_factory=make_unix_service,  # type: ignore[arg-type]
        search_backend_factory=make_search_backend,
        optional_runtime_available=lambda: True,
    )
    options = UnixIngestOptions(
        namespace=seed.dataset_version.namespace,
        processed_pack_path=tmp_path / "ignored-processed-pack",
        source_lock_path=tmp_path / "source-lock.json",
        processed_pack_lock_path=tmp_path / "processed-lock.json",
        dataset_manifest_path=UNIX_MANIFEST,
        curated_manifest_path=tmp_path / "curated.json",
    )

    result = await application.ingest_unix(options, emit=emitted.append)
    await application.close()

    assert result.dataset_version == seed.dataset_version
    assert result.query_set == seed.query_set
    assert len(result.configs) == 4
    assert provider.closed
    assert len(backends) == 1 and backends[0].closed
    assert emitted == ["progress state=ready batches=1/1 documents=100/100"]
    with Database.from_settings(_settings(tmp_path)) as database:
        repository = PufferLabRepository(database.session_factory)
        assert repository.get_query_set(seed.query_set.id)[0] == seed.query_set
        assert (
            len(repository.list_retrieval_configs(dataset_version_id=seed.dataset_version.id)) == 4
        )
