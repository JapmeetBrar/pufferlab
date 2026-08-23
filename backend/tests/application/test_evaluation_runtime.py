from __future__ import annotations

import asyncio
import sqlite3
import threading
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from fastapi.testclient import TestClient
from pufferlab.application import evaluation_runtime as evaluation_runtime_module
from pufferlab.application.evaluation_runtime import EvaluationApiRuntime
from pufferlab.application.evaluations import create_evaluation_run
from pufferlab.application.view_errors import (
    EvaluationViewError,
    evaluation_conflict,
)
from pufferlab.config import Settings
from pufferlab.contracts.common import ObservedScore, ScoreDirection, ScoreKind, ScoreSource
from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion, FtsProfile, IndexProfile
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    CreateEvalRunRequest,
    EvalRun,
    EvalRunStatus,
    JudgedQuery,
    Qrel,
    QuerySet,
    RunEnvironment,
)
from pufferlab.contracts.forensics import EvalRunQueryReplayRequest, ForensicCode
from pufferlab.contracts.retrieval import (
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
)
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareRequest,
    SearchCompareResponse,
    SearchHit,
    StageMembership,
)
from pufferlab.datasets.cqadupstack import (
    CuratedQuery,
    CuratedQueryManifest,
    SourceLock,
    curated_selection_sha256,
    load_source_lock,
    source_lock_sha256,
)
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.datasets.unix_application import (
    UNIX_REVISION_CREATED_AT,
    unix_query_set_content_sha256,
)
from pufferlab.main import create_app
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.errors import PersistenceValidationError, RecordNotFoundError
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.retrieval.config import BoundSearchCatalog, derive_bound_retrieval_configs
from pufferlab.retrieval.errors import provider_failed
from pufferlab.retrieval.types import (
    HybridProbeCandidate,
    HybridProbeExecuteRequest,
    HybridProbeExecuteResult,
    HybridProbeStageMembership,
    ReplaySearchBackend,
    SearchExecuteRequest,
    SearchExecuteResult,
)
from pufferlab.synthetic_demo import AUTHORED_SYNTHETIC_DEMO
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo

_TEST_NAMESPACE = UUID("cc1bc5f7-0f4e-4b99-a8ad-8cc647027700")
_NOW = datetime(2026, 8, 23, 0, 0, tzinfo=UTC)
_REPOSITORY_ROOT = Path(__file__).parents[3]


def _id(name: str) -> UUID:
    return uuid5(_TEST_NAMESPACE, name)


@dataclass(frozen=True, slots=True)
class _LiveSuite:
    dataset: DatasetVersion
    query_set: QuerySet
    configs: tuple[RetrievalConfig, ...]
    manifest: DatasetManifest

    def request(self, *, query_set_id: UUID | None = None) -> CreateEvalRunRequest:
        return CreateEvalRunRequest(
            query_set_id=query_set_id or self.query_set.id,
            baseline_config_id=self.configs[0].id,
            candidate_config_ids=[config.id for config in self.configs[1:]],
            max_concurrency=4,
            warmup_query_count=0,
        )


class _BlockingBackend:
    def __init__(
        self,
        configs: tuple[RetrievalConfig, ...],
        *,
        close_order: list[str] | None = None,
        close_label: str = "eval-close",
    ) -> None:
        self._configs = configs
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self._close_order = close_order
        self._close_label = close_label

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return tuple(
            RetrievalConfigSummary(
                id=config.id,
                revision=config.revision,
                name=config.name,
                mode=config.mode,
                config_hash=config.config_hash,
            )
            for config in self._configs
        )

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise AssertionError(f"unexpected compare request: {request.query_text}")

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.001)
        raise provider_failed("evaluation_runtime_test")

    async def probe_hybrid_candidates(
        self,
        request: HybridProbeExecuteRequest,
    ) -> HybridProbeExecuteResult:
        raise AssertionError(f"unexpected probe request: {request.query_text}")

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._close_order is not None:
            self._close_order.append(self._close_label)


class _PlaygroundBackend:
    def __init__(self, close_order: list[str]) -> None:
        self._close_order = close_order

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return ()

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise AssertionError(f"unexpected compare request: {request.query_text}")

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        raise AssertionError(f"unexpected search request: {request.query_text}")

    async def close(self) -> None:
        self._close_order.append("playground-close")


class _TrackingDatabase(Database):
    def __init__(self, path: Path, order: list[str]) -> None:
        super().__init__(path)
        self._order = order
        self.migrate_calls = 0

    def migrate(self) -> None:
        self.migrate_calls += 1
        super().migrate()

    def dispose(self) -> None:
        self._order.append("database-dispose")
        super().dispose()


class _ContextBearingGuard:
    def __init__(self, marker: str) -> None:
        self._marker = marker

    def acquire(self) -> None:
        try:
            raise RuntimeError(self._marker)
        except RuntimeError:
            raise evaluation_conflict(
                message="evaluation worker guard is unavailable",
                operation="start_evaluation_runtime",
            ) from None

    def release(self) -> None:
        return


class _FactoryProbe:
    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        close_order: list[str] | None = None,
        fail_runtime: bool = False,
    ) -> None:
        self.manifest = manifest
        self.close_order = close_order
        self.fail_runtime = fail_runtime
        self.calls = {
            "manifest": 0,
            "credential": 0,
            "catalog": 0,
            "runtime": 0,
            "provider": 0,
            "embedder": 0,
            "reranker": 0,
        }
        self.backends: list[_BlockingBackend] = []

    def load_manifest(self, _path: Path) -> DatasetManifest:
        self.calls["manifest"] += 1
        return self.manifest

    def check_credential(self, _settings: Settings) -> None:
        self.calls["credential"] += 1

    def make_catalog(
        self,
        dataset: DatasetVersion,
        manifest: DatasetManifest,
        configs: tuple[RetrievalConfig, ...],
    ) -> BoundSearchCatalog:
        self.calls["catalog"] += 1
        return BoundSearchCatalog(
            dataset_version=dataset,
            manifest=manifest,
            configs=configs,
        )

    def make_runtime(
        self,
        _settings: Settings,
        _manifest: DatasetManifest,
        bound: BoundSearchCatalog,
        provider_factory: Any,
        embedder_factory: Any,
        reranker_factory: Any,
    ) -> ReplaySearchBackend:
        self.calls["runtime"] += 1
        if self.fail_runtime:
            raise RuntimeError("private runtime factory detail")
        backend = _BlockingBackend(bound.configs, close_order=self.close_order)
        self.backends.append(backend)
        # The factories are retained by the real runtime and may only execute after a claim.
        assert callable(provider_factory)
        assert callable(embedder_factory)
        assert callable(reranker_factory)
        return backend

    def provider(self, **_kwargs: object) -> Any:
        self.calls["provider"] += 1
        raise AssertionError("provider construction was not expected")

    def embedder(self, **_kwargs: object) -> Any:
        self.calls["embedder"] += 1
        raise AssertionError("embedder construction was not expected")

    def reranker(self, **_kwargs: object) -> Any:
        self.calls["reranker"] += 1
        raise AssertionError("reranker construction was not expected")


def _observed_score(kind: ScoreKind, value: float) -> ObservedScore:
    return ObservedScore(
        kind=kind,
        value=value,
        direction=(
            ScoreDirection.LOWER_IS_BETTER
            if kind is ScoreKind.VECTOR_DISTANCE
            else ScoreDirection.HIGHER_IS_BETTER
        ),
        source=(
            ScoreSource.RERANKER if kind is ScoreKind.RERANKER else ScoreSource.TURBOPUFFER_DIST
        ),
    )


class _ReplayBackend:
    def __init__(
        self,
        configs: tuple[RetrievalConfig, ...],
        document_ids: tuple[UUID, ...],
        *,
        fail_probe: bool = False,
        fail_compare: bool = False,
        fail_close: bool = False,
        block_compare: bool = False,
        block_close: bool = False,
        mismatch_probe: bool = False,
        corrupt_primary_binding: bool = False,
        corrupt_probe_binding: bool = False,
    ) -> None:
        self._configs = {config.id: config for config in configs}
        self._document_ids = document_ids
        self._fail_probe = fail_probe
        self._fail_compare = fail_compare
        self._fail_close = fail_close
        self._block_compare = block_compare
        self._block_close = block_close
        self._mismatch_probe = mismatch_probe
        self._corrupt_primary_binding = corrupt_primary_binding
        self._corrupt_probe_binding = corrupt_probe_binding
        self.compare_calls: list[SearchCompareRequest] = []
        self.probe_calls: list[HybridProbeExecuteRequest] = []
        self.compare_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.closed = False

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return tuple(
            RetrievalConfigSummary(
                id=config.id,
                revision=config.revision,
                name=config.name,
                mode=config.mode,
                config_hash=config.config_hash,
            )
            for config in self._configs.values()
        )

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        self.compare_calls.append(request)
        self.compare_started.set()
        if self._block_compare:
            await asyncio.Event().wait()
        if self._fail_compare:
            try:
                raise RuntimeError("PRIVATE_NAMESPACE_FAILURE_MARKER")
            except RuntimeError as cause:
                raise ProviderError(
                    "turbopuffer namespace was not found",
                    ProviderErrorDetails(
                        code=ApiErrorCode.NOT_FOUND,
                        retryable=False,
                        operation="query",
                        status_code=404,
                    ),
                ) from cause
        results: list[ConfigSearchResult] = []
        for result_index, config_id in enumerate(request.config_ids):
            config = self._configs[config_id]
            score_kind = {
                RetrievalMode.BM25: ScoreKind.BM25,
                RetrievalMode.VECTOR: ScoreKind.VECTOR_DISTANCE,
                RetrievalMode.HYBRID_RRF: ScoreKind.RRF,
                RetrievalMode.HYBRID_RERANK: ScoreKind.RERANKER,
            }[config.mode]
            hits: list[SearchHit] = []
            for rank, document_id in enumerate(self._document_ids, start=1):
                score = _observed_score(score_kind, float(rank))
                memberships: list[StageMembership] = []
                if config.mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}:
                    memberships.append(
                        StageMembership(
                            stage=RetrievalStage.RRF,
                            rank=rank,
                            score=_observed_score(ScoreKind.RRF, 1.0 / (60 + rank)),
                        )
                    )
                if config.mode is RetrievalMode.HYBRID_RERANK:
                    memberships.append(
                        StageMembership(
                            stage=RetrievalStage.RERANKER,
                            rank=rank,
                            score=score,
                        )
                    )
                memberships.append(
                    StageMembership(stage=RetrievalStage.FINAL, rank=rank, score=score)
                )
                hits.append(
                    SearchHit(
                        document_id=document_id,
                        external_id=f"authored-{rank}",
                        title="Authored replay result",
                        body_excerpt="Bounded authored replay excerpt.",
                        relevance_grade=1,
                        final_rank=rank,
                        final_score=score,
                        stage_membership=memberships,
                    )
                )
            results.append(
                ConfigSearchResult(
                    config=RetrievalConfigSummary(
                        id=config.id,
                        revision=config.revision,
                        name=config.name,
                        mode=config.mode,
                        config_hash=config.config_hash,
                    ),
                    hits=hits,
                    timings=[],
                    candidate_counts={
                        (
                            RetrievalStage.RRF.value
                            if config.mode
                            in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
                            else RetrievalStage.FINAL.value
                        ): len(hits)
                    },
                    warnings=[],
                    trace_id=UUID(int=1_000 + result_index),
                )
            )
        return SearchCompareResponse(
            query_text=(
                "wrong replay query" if self._corrupt_primary_binding else request.query_text
            ),
            query_id=request.query_id,
            results=results,
            rank_movements=[],
            overlap=[],
            observability_notice="Primary replay fake.",
        )

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        raise AssertionError(f"unexpected search request: {request.query_text}")

    async def probe_hybrid_candidates(
        self,
        request: HybridProbeExecuteRequest,
    ) -> HybridProbeExecuteResult:
        self.probe_calls.append(request)
        if self._fail_probe:
            raise RuntimeError("PRIVATE_PROBE_FAILURE_MARKER")
        ordered = (
            tuple(reversed(self._document_ids)) if self._mismatch_probe else self._document_ids
        )
        candidates = tuple(
            HybridProbeCandidate(
                document_id=document_id,
                stage_membership=(
                    HybridProbeStageMembership(
                        stage=RetrievalStage.BM25_CANDIDATES,
                        rank=rank,
                        score=_observed_score(ScoreKind.BM25, float(10 - rank)),
                    ),
                    HybridProbeStageMembership(
                        stage=RetrievalStage.VECTOR_CANDIDATES,
                        rank=rank,
                        score=_observed_score(ScoreKind.VECTOR_DISTANCE, rank / 10.0),
                    ),
                ),
            )
            for rank, document_id in enumerate(ordered, start=1)
        )
        return HybridProbeExecuteResult(
            config_id=request.config_id,
            query_id=request.query_id,
            trace_id=UUID(int=request.trace_id.int + 1)
            if self._corrupt_probe_binding
            else request.trace_id,
            duration_ms=2.0,
            bm25_candidate_count=len(candidates),
            vector_candidate_count=len(candidates),
            candidates=candidates,
        )

    async def close(self) -> None:
        self.close_started.set()
        if self._block_close:
            await self.close_release.wait()
        self.closed = True
        if self._fail_close:
            raise RuntimeError("PRIVATE_REPLAY_CLOSE_MARKER")


class _ReplayFactoryProbe(_FactoryProbe):
    def __init__(
        self,
        manifest: DatasetManifest,
        document_ids: tuple[UUID, ...],
        **behavior: bool,
    ) -> None:
        super().__init__(manifest)
        self.document_ids = document_ids
        self.behavior = behavior
        self.replay_backends: list[_ReplayBackend] = []

    def make_runtime(
        self,
        _settings: Settings,
        _manifest: DatasetManifest,
        bound: BoundSearchCatalog,
        provider_factory: Any,
        embedder_factory: Any,
        reranker_factory: Any,
    ) -> _ReplayBackend:
        self.calls["runtime"] += 1
        assert callable(provider_factory)
        assert callable(embedder_factory)
        assert callable(reranker_factory)
        backend = _ReplayBackend(bound.configs, self.document_ids, **self.behavior)
        self.replay_backends.append(backend)
        return backend


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        pufferlab_data_dir=tmp_path,
        turbopuffer_api_key="test-only-secret",
        turbopuffer_region="gcp-us-west1",
    )


def _seed_live_suite(
    repository: PufferLabRepository,
    *,
    judged_queries: list[JudgedQuery] | None = None,
    curated_manifest: CuratedQueryManifest | None = None,
) -> _LiveSuite:
    manifest = AUTHORED_SYNTHETIC_DEMO.manifest
    write_spec = compile_namespace_write_spec(manifest)
    dataset = DatasetVersion(
        id=_id("live-dataset"),
        slug=manifest.slug,
        version=manifest.version,
        namespace="pufferlab-live-runtime-test",
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
        document_count=len(AUTHORED_SYNTHETIC_DEMO.documents),
        corpus_hash="pufferlab-authored-live-test-corpus",
        status=DatasetStatus.READY,
        created_at=_NOW,
    )
    configs = derive_bound_retrieval_configs(dataset, manifest)
    if curated_manifest is None:
        query_set = QuerySet(
            id=_id("live-query-set"),
            name="PufferLab-authored runtime test queries",
            version="v1",
            dataset_version_id=dataset.id,
            query_count=50,
            content_hash="pufferlab-authored-runtime-query-hash",
            created_at=_NOW,
        )
    else:
        assert curated_manifest.query_set_content_sha256 is not None
        query_set = QuerySet(
            id=uuid5(
                PUFFERLAB_NAMESPACE_UUID,
                (f"query-set:{dataset.id}:{curated_manifest.query_set_content_sha256}"),
            ),
            name="CQADupStack Unix curated 50",
            version=curated_manifest.selection_version,
            dataset_version_id=dataset.id,
            query_count=curated_manifest.query_count,
            content_hash=curated_manifest.query_set_content_sha256,
            created_at=UNIX_REVISION_CREATED_AT,
        )
    repository.put_dataset_version(dataset)
    for config in configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(
        query_set,
        judged_queries
        if judged_queries is not None
        else [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
    )
    return _LiveSuite(dataset, query_set, configs, manifest)


def _environment() -> RunEnvironment:
    return RunEnvironment(
        pufferlab_git_revision="test-revision",
        turbopuffer_region="gcp-us-west1",
        python_version="3.13",
        platform="test",
        max_concurrency=4,
        warmup_query_count=0,
        query_embedding_cache_enabled=False,
    )


def _runtime(
    settings: Settings,
    database: Database,
    probe: _FactoryProbe,
    *,
    worker_guard_factory: Callable[[Path], Any] | None = None,
    curated_manifest: CuratedQueryManifest | None = None,
    source_lock: SourceLock | None = None,
) -> EvaluationApiRuntime:
    def load_curated(_path: Path) -> CuratedQueryManifest:
        assert curated_manifest is not None
        return curated_manifest

    def load_checked_source(_path: Path) -> SourceLock:
        assert source_lock is not None
        return source_lock

    return EvaluationApiRuntime(
        settings,
        database=database,
        manifest_loader=probe.load_manifest,
        curated_manifest_loader=load_curated,
        source_lock_loader=load_checked_source,
        query_set_authenticator=(
            None
            if curated_manifest is not None and source_lock is not None
            else lambda _dataset, _query_set, _queries: None
        ),
        credential_check=probe.check_credential,
        bound_catalog_factory=probe.make_catalog,
        search_backend_factory=probe.make_runtime,
        provider_factory=probe.provider,
        embedder_factory=probe.embedder,
        reranker_factory=probe.reranker,
        worker_guard_factory=worker_guard_factory,
        git_revision_factory=lambda: "a" * 40,
        now=lambda: datetime.now(UTC),
    )


def _assert_detached_error(
    error: EvaluationViewError,
    *,
    marker: str,
    http_status: int,
) -> None:
    assert error.http_status == http_status
    assert error.__cause__ is None
    assert error.__context__ is None
    assert marker not in str(error)
    assert marker not in repr(error)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert marker not in rendered


async def _assert_detached_async_error(
    action: Callable[[], Awaitable[object]],
    *,
    marker: str,
    http_status: int,
) -> None:
    with pytest.raises(EvaluationViewError) as raised:
        await action()
    _assert_detached_error(raised.value, marker=marker, http_status=http_status)


async def _wait_for_status(
    repository: PufferLabRepository,
    run_id: UUID,
    statuses: set[EvalRunStatus],
) -> EvalRun:
    for _ in range(500):
        run = repository.get_run(run_id)
        if run.status in statuses:
            return run
        await asyncio.sleep(0.002)
    raise AssertionError(f"run {run_id} did not reach {statuses}")


async def _wait_for_backend_start(backend: _BlockingBackend) -> None:
    for _ in range(500):
        if backend.started.is_set():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("evaluation backend did not receive work")


@pytest.mark.asyncio
async def test_create_is_durable_scheduled_and_independent_of_request_scope_cancellation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)
    await runtime.start()

    request_returned = asyncio.Event()

    async def request_scope() -> None:
        response = await runtime.create_eval_run(suite.request())
        assert response.result.run.status is EvalRunStatus.QUEUED
        request_returned.set()
        await asyncio.Event().wait()

    scope = asyncio.create_task(request_scope())
    await request_returned.wait()
    scope.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scope

    [run] = repository.list_active_runs()
    running = await _wait_for_status(repository, run.id, {EvalRunStatus.RUNNING})
    assert running.started_at is not None
    await _wait_for_backend_start(probe.backends[0])
    probe.backends[0].release.set()
    cancelled = await runtime.cancel_eval_run(run.id)
    assert cancelled.result.run.status is EvalRunStatus.CANCELLED
    assert (await runtime.cancel_eval_run(run.id)) == cancelled
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_duplicate_precedes_capacity_and_only_one_run_claims(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    second_query_set = suite.query_set.model_copy(
        update={"id": _id("second-query-set"), "content_hash": "second-query-set-hash"}
    )
    repository.put_query_set(
        second_query_set,
        [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
    )
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    first = await runtime.create_eval_run(suite.request())

    with pytest.raises(EvaluationViewError, match="equivalent") as duplicate:
        await runtime.create_eval_run(suite.request())
    assert duplicate.value.http_status == 409
    with pytest.raises(EvaluationViewError, match="capacity") as capacity:
        await runtime.create_eval_run(suite.request(query_set_id=second_query_set.id))
    assert capacity.value.http_status == 409

    await _wait_for_status(repository, first.result.run.id, {EvalRunStatus.RUNNING})
    assert len(probe.backends) == 1
    probe.backends[0].release.set()
    await runtime.cancel_eval_run(first.result.run.id)
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_startup_interrupts_stale_running_and_reclaims_queued_oldest_first(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    stale = create_evaluation_run(
        repository,
        suite.request(),
        _environment(),
        run_id=_id("stale-running"),
        now=lambda: _NOW - timedelta(minutes=3),
    )
    repository.claim_queued_run(stale.id, at=_NOW - timedelta(minutes=2))
    oldest = create_evaluation_run(
        repository,
        suite.request(),
        _environment(),
        run_id=_id("oldest-queued"),
        now=lambda: _NOW - timedelta(minutes=1),
    )
    newest = create_evaluation_run(
        repository,
        suite.request(),
        _environment(),
        run_id=_id("newest-queued"),
        now=lambda: _NOW,
    )
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)

    await runtime.start()
    assert repository.get_run(stale.id).status is EvalRunStatus.INTERRUPTED
    await _wait_for_status(repository, oldest.id, {EvalRunStatus.RUNNING})
    assert repository.get_run(newest.id).status is EvalRunStatus.QUEUED
    assert len(probe.backends) == 1

    probe.backends[0].release.set()
    await runtime.cancel_eval_run(oldest.id)
    await _wait_for_status(repository, newest.id, {EvalRunStatus.RUNNING})
    assert len(probe.backends) == 2
    probe.backends[1].release.set()
    await runtime.cancel_eval_run(newest.id)
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_invalid_queued_binding_fails_before_claim_or_provider_factories(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    tampered = tuple(
        config.model_copy(
            update={
                "id": _id(f"tampered-config-{index}"),
                "name": f"tampered {index}",
                "config_hash": f"tampered-hash-{index}",
            }
        )
        for index, config in enumerate(suite.configs)
    )
    for config in tampered:
        repository.put_retrieval_config(config)
    request = CreateEvalRunRequest(
        query_set_id=suite.query_set.id,
        baseline_config_id=tampered[0].id,
        candidate_config_ids=[config.id for config in tampered[1:]],
        max_concurrency=4,
        warmup_query_count=0,
    )
    queued = create_evaluation_run(
        repository,
        request,
        _environment(),
        run_id=_id("invalid-queued"),
        now=lambda: _NOW,
    )
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)

    await runtime.start()
    failed = await _wait_for_status(repository, queued.id, {EvalRunStatus.FAILED})
    assert failed.started_at is None
    assert failed.error is not None
    assert failed.error.message == "evaluation runtime could not execute the durable run"
    assert probe.calls == {
        "manifest": 1,
        "credential": 0,
        "catalog": 0,
        "runtime": 0,
        "provider": 0,
        "embedder": 0,
        "reranker": 0,
    }
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_synthetic_queued_recovery_rejects_every_live_factory_boundary(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    synthetic = materialize_synthetic_demo()
    repository.put_dataset_version(synthetic.dataset_version)
    for config in synthetic.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(
        synthetic.query_set,
        [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
    )
    repository.create_run(synthetic.queued_run)
    probe = _FactoryProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    runtime = _runtime(settings, database, probe)

    await runtime.start()
    failed = await _wait_for_status(
        repository,
        synthetic.queued_run.id,
        {EvalRunStatus.FAILED},
    )
    assert failed.started_at is None
    assert failed.error is not None
    assert all(value == 0 for value in probe.calls.values())
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_synthetic_create_rejects_before_every_live_factory_boundary(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    synthetic = materialize_synthetic_demo()
    repository.put_dataset_version(synthetic.dataset_version)
    for config in synthetic.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(
        synthetic.query_set,
        [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
    )
    probe = _FactoryProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    request = CreateEvalRunRequest(
        query_set_id=synthetic.query_set.id,
        baseline_config_id=synthetic.configs[0].id,
        candidate_config_ids=[config.id for config in synthetic.configs[1:]],
        max_concurrency=1,
        warmup_query_count=0,
    )

    with pytest.raises(EvaluationViewError, match="read/export-only") as conflict:
        await runtime.create_eval_run(request)

    assert conflict.value.http_status == 409
    assert repository.list_active_runs() == []
    assert all(value == 0 for value in probe.calls.values())
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_post_claim_factory_failure_is_safe_and_detached(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    probe = _FactoryProbe(suite.manifest, fail_runtime=True)
    runtime = _runtime(settings, database, probe)
    await runtime.start()

    response = await runtime.create_eval_run(suite.request())
    failed = await _wait_for_status(
        repository,
        response.result.run.id,
        {EvalRunStatus.FAILED},
    )
    assert failed.started_at is not None
    assert failed.error is not None
    assert "private" not in failed.error.model_dump_json()
    assert probe.calls == {
        "manifest": 1,
        "credential": 1,
        "catalog": 1,
        "runtime": 1,
        "provider": 0,
        "embedder": 0,
        "reranker": 0,
    }
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_worker_guard_rejects_a_second_runtime_and_start_is_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first_database = Database.from_settings(settings)
    second_database = Database.from_settings(settings)
    empty_probe = _FactoryProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    first = _runtime(settings, first_database, empty_probe)
    second = _runtime(settings, second_database, empty_probe)

    await first.start()
    await first.start()
    with pytest.raises(EvaluationViewError, match="another PufferLab API worker") as conflict:
        await second.start()
    assert conflict.value.http_status == 503
    await first.shutdown_execution()
    first.dispose()
    await second.shutdown_execution()
    second.dispose()
    assert all(value == 0 for value in empty_probe.calls.values())


@pytest.mark.asyncio
async def test_all_runtime_error_translations_detach_private_exception_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_marker = "PRIVATE_START_CONTEXT_MARKER"
    start_settings = _settings(tmp_path / "start")
    start_database = Database.from_settings(start_settings)
    start_probe = _FactoryProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    start_runtime = _runtime(
        start_settings,
        start_database,
        start_probe,
        worker_guard_factory=lambda _path: _ContextBearingGuard(start_marker),
    )
    await _assert_detached_async_error(
        start_runtime.start,
        marker=start_marker,
        http_status=409,
    )
    start_runtime.dispose()

    settings = _settings(tmp_path / "controls")
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)
    await runtime.start()

    binding_failures: tuple[tuple[type[BaseException], str, int], ...] = (
        (RecordNotFoundError, "PRIVATE_CREATE_MISSING_MARKER", 422),
        (PersistenceValidationError, "PRIVATE_CREATE_VALIDATION_MARKER", 422),
        (RuntimeError, "PRIVATE_CREATE_RUNTIME_MARKER", 503),
    )
    for exception_type, marker, status in binding_failures:

        def fail_binding(
            _request: CreateEvalRunRequest,
            *,
            failure_type: type[BaseException] = exception_type,
            failure_marker: str = marker,
        ) -> None:
            raise failure_type(failure_marker)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_resolve_request_binding", fail_binding)
            await _assert_detached_async_error(
                lambda: runtime.create_eval_run(suite.request()),
                marker=marker,
                http_status=status,
            )

    nested_marker = "PRIVATE_CREATE_VIEW_CONTEXT_MARKER"

    def fail_with_context(_request: CreateEvalRunRequest) -> None:
        try:
            raise RuntimeError(nested_marker)
        except RuntimeError:
            raise evaluation_conflict(
                message="evaluation request conflicts with stored state",
                operation="create_eval_run",
            ) from None

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_resolve_request_binding", fail_with_context)
        await _assert_detached_async_error(
            lambda: runtime.create_eval_run(suite.request()),
            marker=nested_marker,
            http_status=409,
        )

    persisted_marker = "PRIVATE_CREATE_PERSIST_MARKER"

    def fail_persist(*_args: object, **_kwargs: object) -> None:
        raise PersistenceValidationError(persisted_marker)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_runtime_module, "create_evaluation_run", fail_persist)
        await _assert_detached_async_error(
            lambda: runtime.create_eval_run(suite.request()),
            marker=persisted_marker,
            http_status=422,
        )

    cancel_failures: tuple[tuple[type[BaseException], str, int], ...] = (
        (RecordNotFoundError, "PRIVATE_CANCEL_MISSING_MARKER", 404),
        (PersistenceValidationError, "PRIVATE_CANCEL_STORAGE_MARKER", 503),
    )
    for exception_type, marker, status in cancel_failures:

        def fail_get_run(
            _run_id: UUID,
            *,
            failure_type: type[BaseException] = exception_type,
            failure_marker: str = marker,
        ) -> None:
            raise failure_type(failure_marker)

        with monkeypatch.context() as patch:
            patch.setattr(runtime._repository, "get_run", fail_get_run)
            await _assert_detached_async_error(
                lambda: runtime.cancel_eval_run(_id("missing-runtime-run")),
                marker=marker,
                http_status=status,
            )

    await runtime.shutdown_execution()
    runtime.dispose()


def test_file_worker_guard_errors_detach_os_and_lock_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_type = evaluation_runtime_module._FileWorkerGuard

    open_marker = "PRIVATE_GUARD_OPEN_MARKER"

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise OSError(open_marker)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_runtime_module.os, "open", fail_open)
        with pytest.raises(EvaluationViewError, match="unavailable") as raised:
            guard_type(tmp_path / "guard-open.lock").acquire()
    _assert_detached_error(raised.value, marker=open_marker, http_status=503)

    blocked_marker = "PRIVATE_GUARD_BLOCK_MARKER"

    def fail_blocked(*_args: object, **_kwargs: object) -> None:
        raise BlockingIOError(blocked_marker)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_runtime_module.fcntl, "flock", fail_blocked)
        with pytest.raises(EvaluationViewError, match="another PufferLab API worker") as raised:
            guard_type(tmp_path / "guard-blocked.lock").acquire()
    _assert_detached_error(raised.value, marker=blocked_marker, http_status=503)

    lock_marker = "PRIVATE_GUARD_LOCK_MARKER"

    def fail_lock(*_args: object, **_kwargs: object) -> None:
        raise OSError(lock_marker)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_runtime_module.fcntl, "flock", fail_lock)
        with pytest.raises(EvaluationViewError, match="unavailable") as raised:
            guard_type(tmp_path / "guard-failed.lock").acquire()
    _assert_detached_error(raised.value, marker=lock_marker, http_status=503)


def test_http_control_errors_never_expose_private_runtime_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    probe = _FactoryProbe(suite.manifest)
    runtime = _runtime(settings, database, probe)
    app = create_app(
        settings,
        search_backend=_PlaygroundBackend([]),
        evaluation_runtime=runtime,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        create_marker = "PRIVATE_HTTP_CREATE_MARKER"

        def fail_manifest(_path: Path) -> DatasetManifest:
            raise RuntimeError(create_marker)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_manifest_loader", fail_manifest)
            response = client.post(
                "/api/v1/eval-runs",
                json=suite.request().model_dump(mode="json"),
            )
        assert response.status_code == 503
        assert create_marker not in response.text

        cancel_marker = "PRIVATE_HTTP_CANCEL_MARKER"

        def fail_cancel(_run_id: UUID) -> None:
            raise PersistenceValidationError(cancel_marker)

        with monkeypatch.context() as patch:
            patch.setattr(runtime._repository, "get_run", fail_cancel)
            response = client.post(f"/api/v1/eval-runs/{_id('http-cancel')}/cancel")
        assert response.status_code == 503
        assert cancel_marker not in response.text


def test_app_lifespan_orders_eval_drain_playground_close_then_database_dispose(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    settings = _settings(tmp_path)
    database = _TrackingDatabase(settings.database_path, order)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    suite = _seed_live_suite(repository)
    probe = _FactoryProbe(suite.manifest, close_order=order)
    runtime = _runtime(settings, database, probe)
    playground = _PlaygroundBackend(order)
    app = create_app(
        settings,
        search_backend=playground,
        evaluation_runtime=runtime,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/eval-runs",
            json=suite.request().model_dump(mode="json"),
        )
        assert response.status_code == 202
        run_id = UUID(response.json()["result"]["run"]["id"])
        for _ in range(500):
            detail = client.get(f"/api/v1/eval-runs/{run_id}")
            if detail.json()["result"]["run"]["status"] == "running":
                break
        else:
            raise AssertionError("API-created run did not start")
        assert probe.backends[0].started.wait(timeout=1)
        probe.backends[0].release.set()

    assert order[-3:] == ["eval-close", "playground-close", "database-dispose"]
    assert database.migrate_calls == 2


def test_default_app_starts_migrated_guarded_evaluation_runtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert app.state.evaluation_runtime is not None
        assert settings.database_path.is_file()

    assert (tmp_path / ".pufferlab-api.lock").is_file()


def _replay_queries() -> tuple[list[JudgedQuery], tuple[UUID, UUID, UUID, UUID]]:
    queries = [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries]
    first = queries[0]
    zero_grade_document = queries[1].qrels[0].document_id
    unjudged_document = queries[2].qrels[0].document_id
    queries[0] = first.model_copy(
        update={
            "qrels": [
                first.qrels[0].model_copy(update={"relevance_grade": 2}),
                first.qrels[1].model_copy(update={"relevance_grade": 1}),
                Qrel(document_id=zero_grade_document, relevance_grade=0),
            ]
        }
    )
    return queries, (
        first.qrels[0].document_id,
        first.qrels[1].document_id,
        zero_grade_document,
        unjudged_document,
    )


def _authenticated_replay_source(
    queries: list[JudgedQuery],
) -> tuple[list[JudgedQuery], CuratedQueryManifest, SourceLock]:
    source_lock = load_source_lock(_REPOSITORY_ROOT / "datasets/cqadupstack-unix/source-lock.json")
    tag_order = ("exact_token", "semantic", "hybrid", "reranker")
    authenticated_queries: list[JudgedQuery] = []
    entries: list[CuratedQuery] = []
    for index, query in enumerate(queries):
        tag = tag_order[index % len(tag_order)]
        authenticated_queries.append(query.model_copy(update={"tags": [tag]}))
        entries.append(
            CuratedQuery(
                query_id=query.external_id,
                primary_tag=tag,
                tags=(tag,),
                reason="Selected as an authored provider-free authentication fixture.",
            )
        )
    entry_tuple = tuple(entries)
    unanchored = CuratedQueryManifest(
        format_version=1,
        selection_version="pufferlab-curated-50-v1",
        source_lock_sha256=source_lock_sha256(source_lock),
        query_count=50,
        selection_sha256=curated_selection_sha256(entry_tuple),
        query_set_content_sha256=None,
        entries=entry_tuple,
    )
    content_hash = unix_query_set_content_sha256(
        tuple(authenticated_queries),
        unanchored,
    )
    return (
        authenticated_queries,
        unanchored.model_copy(update={"query_set_content_sha256": content_hash}),
        source_lock,
    )


def _config_for_mode(suite: _LiveSuite, mode: RetrievalMode) -> RetrievalConfig:
    return next(config for config in suite.configs if config.mode is mode)


def _create_replay_run(repository: PufferLabRepository, suite: _LiveSuite, name: str) -> EvalRun:
    return create_evaluation_run(
        repository,
        suite.request(),
        _environment(),
        run_id=_id(name),
        now=lambda: _NOW,
    )


@pytest.mark.asyncio
async def test_replay_uses_exact_stored_binding_distinct_sources_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(suite.manifest, document_ids)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "exact-replay-run")
    before_bytes = database.path.read_bytes()
    before_run = repository.get_run(run.id).model_dump_json()
    before_query = repository.get_judged_query(suite.query_set.id, queries[0].id).model_dump_json()
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    hybrid = _config_for_mode(suite, RetrievalMode.HYBRID_RRF)

    response = await runtime.replay_eval_query(
        run.id,
        queries[0].id,
        EvalRunQueryReplayRequest(
            config_ids=[bm25.id, hybrid.id],
            include_counterfactual_probe=True,
        ),
    )

    backend = probe.replay_backends[0]
    assert len(backend.compare_calls) == 1
    primary_request = backend.compare_calls[0]
    assert primary_request.query_text == queries[0].text
    assert primary_request.query_id == queries[0].id
    assert primary_request.config_ids == [bm25.id, hybrid.id]
    assert primary_request.filter_override == queries[0].filters
    assert primary_request.expected_document_ids == []
    assert primary_request.debug_provenance is False
    assert [result.config.id for result in response.primary.results] == [bm25.id, hybrid.id]
    assert len({result.trace_id for result in response.primary.results}) == 2
    for result in response.primary.results:
        assert [hit.relevance_grade for hit in result.hits] == [2, 1, 0, None]
        assert all(
            membership.stage
            not in {RetrievalStage.BM25_CANDIDATES, RetrievalStage.VECTOR_CANDIDATES}
            for hit in result.hits
            for membership in hit.stage_membership
        )
        assert all(timing.stage.value != "provenance_probe" for timing in result.timings)
    assert len(backend.probe_calls) == 1
    explicit_probe = backend.probe_calls[0]
    assert explicit_probe.config_id == hybrid.id
    assert explicit_probe.query_text == queries[0].text
    assert explicit_probe.query_id == queries[0].id
    assert explicit_probe.namespace == suite.dataset.namespace
    assert [item.config_id for item in response.counterfactual_probes] == [hybrid.id]
    all_traces = {
        *(result.trace_id for result in response.primary.results),
        *(item.trace_id for item in response.counterfactual_probes),
    }
    assert len(all_traces) == 3
    assert response.failed_counterfactual_probes == []
    assert backend.closed is True
    assert database.path.read_bytes() == before_bytes
    assert repository.get_run(run.id).model_dump_json() == before_run
    assert (
        repository.get_judged_query(suite.query_set.id, queries[0].id).model_dump_json()
        == before_query
    )
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("behavior", "expected_successes", "expected_failures"),
    [
        ({"mismatch_probe": True}, 1, 0),
        ({"fail_probe": True}, 0, 1),
        ({"corrupt_probe_binding": True}, 0, 1),
    ],
)
async def test_replay_probe_mismatch_and_failure_remain_noncausal_and_safe(
    tmp_path: Path,
    behavior: dict[str, bool],
    expected_successes: int,
    expected_failures: int,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(suite.manifest, document_ids[:2], **behavior)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"probe-{next(iter(behavior))}")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    hybrid = _config_for_mode(suite, RetrievalMode.HYBRID_RRF)

    response = await runtime.replay_eval_query(
        run.id,
        queries[0].id,
        EvalRunQueryReplayRequest(
            config_ids=[bm25.id, hybrid.id],
            include_counterfactual_probe=True,
        ),
    )

    assert len(response.counterfactual_probes) == expected_successes
    assert len(response.failed_counterfactual_probes) == expected_failures
    serialized = response.model_dump_json()
    assert "PRIVATE_PROBE_FAILURE_MARKER" not in serialized
    for forbidden_claim in (
        "this caused",
        "the cause is",
        "provider cache",
        "query plan",
        "reranker rationale",
    ):
        assert forbidden_claim not in serialized.lower()
    if expected_failures:
        assert response.failed_counterfactual_probes[0].config_id == hybrid.id
        failed_observations = [
            item for item in response.observations if item.config_id == hybrid.id
        ]
        assert failed_observations
        assert all(item.code is ForensicCode.NOT_OBSERVABLE for item in failed_observations)
        assert all(item.certainty.value == "insufficient" for item in failed_observations)
    else:
        [successful_probe] = response.counterfactual_probes
        assert [warning.code.value for warning in successful_probe.warnings] == [
            "provenance_snapshot_differs"
        ]
        assert all(item.certainty.value != "observed" for item in response.observations)
    assert probe.replay_backends[0].closed is True
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_nonhybrid_replay_never_runs_a_counterfactual_probe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(suite.manifest, document_ids)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "nonhybrid-replay")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)

    response = await runtime.replay_eval_query(
        run.id,
        queries[0].id,
        EvalRunQueryReplayRequest(
            config_ids=[bm25.id, vector.id],
            include_counterfactual_probe=True,
        ),
    )

    assert response.counterfactual_probes == []
    assert response.failed_counterfactual_probes == []
    assert probe.replay_backends[0].probe_calls == []
    assert probe.replay_backends[0].closed is True
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_replay_fails_closed_on_a_backend_response_with_a_foreign_binding(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(
        suite.manifest,
        document_ids,
        corrupt_primary_binding=True,
    )
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "foreign-primary-binding")
    before_bytes = database.path.read_bytes()
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[bm25.id, vector.id]),
        )

    assert raised.value.http_status == 503
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert probe.replay_backends[0].closed is True
    assert database.path.read_bytes() == before_bytes
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_replay_namespace_error_and_close_failure_are_safe_detached_and_nonpersistent(
    tmp_path: Path,
) -> None:
    for behavior, marker, expected_type in (
        ({"fail_compare": True}, "PRIVATE_NAMESPACE_FAILURE_MARKER", ProviderError),
        ({"fail_close": True}, "PRIVATE_REPLAY_CLOSE_MARKER", EvaluationViewError),
    ):
        case_path = tmp_path / marker.lower()
        settings = _settings(case_path)
        database = Database.from_settings(settings)
        database.migrate()
        repository = PufferLabRepository(database.session_factory)
        queries, document_ids = _replay_queries()
        suite = _seed_live_suite(repository, judged_queries=queries)
        probe = _ReplayFactoryProbe(suite.manifest, document_ids, **behavior)
        runtime = _runtime(settings, database, probe)
        await runtime.start()
        run = _create_replay_run(repository, suite, marker)
        before_bytes = database.path.read_bytes()
        bm25 = _config_for_mode(suite, RetrievalMode.BM25)
        vector = _config_for_mode(suite, RetrievalMode.VECTOR)

        with pytest.raises(expected_type) as raised:
            await runtime.replay_eval_query(
                run.id,
                queries[0].id,
                EvalRunQueryReplayRequest(config_ids=[bm25.id, vector.id]),
            )

        error = raised.value
        assert error.__cause__ is None
        assert error.__context__ is None
        rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        assert marker not in rendered
        if isinstance(error, ProviderError):
            assert error.details.code is ApiErrorCode.NOT_FOUND
            assert error.details.status_code == 404
        else:
            assert isinstance(error, EvaluationViewError)
            assert error.http_status == 503
        assert probe.replay_backends[0].closed is True
        assert database.path.read_bytes() == before_bytes
        await runtime.shutdown_execution()
        runtime.dispose()


@pytest.mark.asyncio
async def test_replay_cancellation_still_closes_request_backend(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(suite.manifest, document_ids, block_compare=True)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "cancelled-replay")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)

    task = asyncio.create_task(
        runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[bm25.id, vector.id]),
        )
    )
    for _ in range(500):
        if probe.replay_backends and probe.replay_backends[0].compare_started.is_set():
            break
        await asyncio.sleep(0.002)
    else:
        raise AssertionError("replay did not begin")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert probe.replay_backends[0].closed is True
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_replay_repeated_cancellation_drains_owned_backend_close(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _ReplayFactoryProbe(
        suite.manifest,
        document_ids,
        block_compare=True,
        block_close=True,
    )
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "double-cancelled-replay")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)

    task = asyncio.create_task(
        runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[bm25.id, vector.id]),
        )
    )
    for _ in range(500):
        if probe.replay_backends and probe.replay_backends[0].compare_started.is_set():
            break
        await asyncio.sleep(0.002)
    else:
        raise AssertionError("replay did not begin")
    backend = probe.replay_backends[0]
    task.cancel("first request cancellation")
    await asyncio.wait_for(backend.close_started.wait(), timeout=1)
    task.cancel("second request cancellation during close")
    await asyncio.sleep(0)
    assert task.done() is False
    assert backend.closed is False

    backend.close_release.set()
    with pytest.raises(asyncio.CancelledError, match="first request cancellation"):
        await task
    assert backend.closed is True
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["document", "grade", "document_and_grade"])
async def test_replay_rejects_valid_shaped_sqlite_qrel_substitutions_before_factories(
    tmp_path: Path,
    mutation: str,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    replay_queries, document_ids = _replay_queries()
    queries, curated_manifest, source_lock = _authenticated_replay_source(replay_queries)
    suite = _seed_live_suite(
        repository,
        judged_queries=queries,
        curated_manifest=curated_manifest,
    )
    probe = _ReplayFactoryProbe(suite.manifest, document_ids)
    runtime = _runtime(
        settings,
        database,
        probe,
        curated_manifest=curated_manifest,
        source_lock=source_lock,
    )
    await runtime.start()
    run = _create_replay_run(repository, suite, "sqlite-qrel-substitution")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    hybrid = _config_for_mode(suite, RetrievalMode.HYBRID_RRF)

    with sqlite3.connect(database.path) as connection:
        binding = (str(suite.query_set.id), str(queries[0].id))
        if mutation == "document":
            cursor = connection.execute(
                """
                UPDATE qrels SET document_id = ?
                WHERE query_set_id = ? AND query_id = ? AND ordinal = 0
                """,
                (str(UUID(int=999_999)), *binding),
            )
        elif mutation == "grade":
            cursor = connection.execute(
                """
                UPDATE qrels SET relevance_grade = ?
                WHERE query_set_id = ? AND query_id = ? AND ordinal = 0
                """,
                (3, *binding),
            )
        else:
            assert mutation == "document_and_grade"
            cursor = connection.execute(
                """
                UPDATE qrels SET document_id = ?, relevance_grade = ?
                WHERE query_set_id = ? AND query_id = ? AND ordinal = 0
                """,
                (str(UUID(int=999_999)), 3, *binding),
            )
        assert cursor.rowcount == 1

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[bm25.id, hybrid.id]),
        )

    assert raised.value.http_status == 422
    _assert_no_live_factory_calls(probe)
    assert probe.replay_backends == []
    await runtime.shutdown_execution()
    runtime.dispose()


def _assert_no_live_factory_calls(probe: _FactoryProbe) -> None:
    assert probe.calls == {
        "manifest": 0,
        "credential": 0,
        "catalog": 0,
        "runtime": 0,
        "provider": 0,
        "embedder": 0,
        "reranker": 0,
    }


@pytest.mark.asyncio
async def test_replay_rejects_synthetic_foreign_and_duplicate_qrels_before_all_factories(
    tmp_path: Path,
) -> None:
    synthetic_settings = _settings(tmp_path / "synthetic")
    synthetic_database = Database.from_settings(synthetic_settings)
    synthetic_database.migrate()
    synthetic_repository = PufferLabRepository(synthetic_database.session_factory)
    synthetic = materialize_synthetic_demo()
    synthetic_probe = _FactoryProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    synthetic_runtime = _runtime(
        synthetic_settings,
        synthetic_database,
        synthetic_probe,
    )
    await synthetic_runtime.start()
    synthetic_repository.put_dataset_version(synthetic.dataset_version)
    for config in synthetic.configs:
        synthetic_repository.put_retrieval_config(config)
    synthetic_repository.put_query_set(
        synthetic.query_set,
        [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
    )
    synthetic_repository.create_run(synthetic.queued_run)
    with pytest.raises(EvaluationViewError) as synthetic_error:
        await synthetic_runtime.replay_eval_query(
            synthetic.queued_run.id,
            AUTHORED_SYNTHETIC_DEMO.queries[0].judged_query.id,
            EvalRunQueryReplayRequest(
                config_ids=[synthetic.configs[0].id, synthetic.configs[2].id]
            ),
        )
    assert synthetic_error.value.http_status == 409
    _assert_no_live_factory_calls(synthetic_probe)
    await synthetic_runtime.shutdown_execution()
    synthetic_runtime.dispose()

    live_settings = _settings(tmp_path / "foreign")
    live_database = Database.from_settings(live_settings)
    live_database.migrate()
    live_repository = PufferLabRepository(live_database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(live_repository, judged_queries=queries)
    live_probe = _ReplayFactoryProbe(suite.manifest, document_ids)
    live_runtime = _runtime(live_settings, live_database, live_probe)
    await live_runtime.start()
    run = _create_replay_run(live_repository, suite, "foreign-replay")
    bm25 = _config_for_mode(suite, RetrievalMode.BM25)
    hybrid = _config_for_mode(suite, RetrievalMode.HYBRID_RRF)

    with pytest.raises(EvaluationViewError) as foreign_config:
        await live_runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[bm25.id, _id("foreign-config")]),
        )
    assert foreign_config.value.http_status == 422
    _assert_no_live_factory_calls(live_probe)
    with pytest.raises(EvaluationViewError) as foreign_query:
        await live_runtime.replay_eval_query(
            run.id,
            _id("foreign-query"),
            EvalRunQueryReplayRequest(config_ids=[bm25.id, hybrid.id]),
        )
    assert foreign_query.value.http_status == 404
    _assert_no_live_factory_calls(live_probe)
    await live_runtime.shutdown_execution()
    live_runtime.dispose()

    duplicate_settings = _settings(tmp_path / "duplicate-qrel")
    duplicate_database = Database.from_settings(duplicate_settings)
    duplicate_database.migrate()
    duplicate_repository = PufferLabRepository(duplicate_database.session_factory)
    duplicate_queries, duplicate_document_ids = _replay_queries()
    first_query = duplicate_queries[0]
    duplicate_queries[0] = first_query.model_copy(
        update={
            "qrels": [
                *first_query.qrels,
                Qrel(
                    document_id=first_query.qrels[0].document_id,
                    relevance_grade=0,
                ),
            ]
        }
    )
    duplicate_suite = _seed_live_suite(
        duplicate_repository,
        judged_queries=duplicate_queries,
    )
    duplicate_probe = _ReplayFactoryProbe(
        duplicate_suite.manifest,
        duplicate_document_ids,
    )
    duplicate_runtime = _runtime(
        duplicate_settings,
        duplicate_database,
        duplicate_probe,
    )
    await duplicate_runtime.start()
    duplicate_run = _create_replay_run(
        duplicate_repository,
        duplicate_suite,
        "duplicate-qrel-replay",
    )
    duplicate_bm25 = _config_for_mode(duplicate_suite, RetrievalMode.BM25)
    duplicate_hybrid = _config_for_mode(duplicate_suite, RetrievalMode.HYBRID_RRF)
    with pytest.raises(EvaluationViewError) as duplicate_error:
        await duplicate_runtime.replay_eval_query(
            duplicate_run.id,
            duplicate_queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[duplicate_bm25.id, duplicate_hybrid.id]),
        )
    assert duplicate_error.value.http_status == 422
    _assert_no_live_factory_calls(duplicate_probe)
    await duplicate_runtime.shutdown_execution()
    duplicate_runtime.dispose()


@pytest.mark.asyncio
async def test_tampered_run_config_fails_after_manifest_before_provider_capable_factories(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    tampered = tuple(
        config.model_copy(
            update={
                "id": _id(f"replay-tampered-{index}"),
                "name": f"replay tampered {index}",
                "config_hash": f"replay-tampered-hash-{index}",
            }
        )
        for index, config in enumerate(suite.configs)
    )
    for config in tampered:
        repository.put_retrieval_config(config)
    request = CreateEvalRunRequest(
        query_set_id=suite.query_set.id,
        baseline_config_id=tampered[0].id,
        candidate_config_ids=[config.id for config in tampered[1:]],
        max_concurrency=4,
        warmup_query_count=0,
    )
    probe = _ReplayFactoryProbe(suite.manifest, document_ids)
    runtime = _runtime(settings, database, probe)
    await runtime.start()
    run = create_evaluation_run(
        repository,
        request,
        _environment(),
        run_id=_id("tampered-replay-run"),
        now=lambda: _NOW,
    )

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.replay_eval_query(
            run.id,
            queries[0].id,
            EvalRunQueryReplayRequest(config_ids=[tampered[0].id, tampered[2].id]),
        )

    assert raised.value.http_status == 422
    assert probe.calls == {
        "manifest": 1,
        "credential": 0,
        "catalog": 0,
        "runtime": 0,
        "provider": 0,
        "embedder": 0,
        "reranker": 0,
    }
    assert probe.replay_backends == []
    await runtime.shutdown_execution()
    runtime.dispose()
