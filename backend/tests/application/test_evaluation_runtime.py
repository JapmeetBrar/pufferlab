from __future__ import annotations

import asyncio
import sqlite3
import threading
import traceback
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
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
from pufferlab.contracts.filters import FilterPredicate, PredicateOp
from pufferlab.contracts.forensics import (
    DiagnosticSubqueryRole,
    EvalRunQueryReplayRequest,
    ExpectedDocumentDiagnosticRequest,
    ForensicCode,
)
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
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticAttributeValue,
    DiagnosticCandidateList,
    DiagnosticCandidateRow,
    DiagnosticProviderRequest,
    DiagnosticProviderResult,
    DiagnosticTargetObservation,
)
from pufferlab.retrieval.errors import provider_failed
from pufferlab.retrieval.types import (
    HybridProbeCandidate,
    HybridProbeExecuteRequest,
    HybridProbeExecuteResult,
    HybridProbeStageMembership,
    QueryEmbedding,
    ReplaySearchBackend,
    SearchExecuteRequest,
    SearchExecuteResult,
)
from pufferlab.synthetic_demo import AUTHORED_SYNTHETIC_DEMO
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo
from pydantic import SecretStr

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
    namespace: str = "pufferlab-live-runtime-test",
) -> _LiveSuite:
    manifest = AUTHORED_SYNTHETIC_DEMO.manifest
    write_spec = compile_namespace_write_spec(manifest)
    dataset = DatasetVersion(
        id=_id("live-dataset"),
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
    diagnostic_credential_getter: Callable[[Settings], SecretStr] | None = None,
    diagnostic_provider_factory: Callable[..., Awaitable[Any]] | None = None,
    diagnostic_embedder_factory: Callable[..., Any] | None = None,
    query_set_authenticator: Callable[..., None] | None = None,
) -> EvaluationApiRuntime:
    def load_curated(_path: Path) -> CuratedQueryManifest:
        assert curated_manifest is not None
        return curated_manifest

    def load_checked_source(_path: Path) -> SourceLock:
        assert source_lock is not None
        return source_lock

    diagnostic_options: dict[str, object] = {}
    if diagnostic_credential_getter is not None:
        diagnostic_options["diagnostic_credential_getter"] = diagnostic_credential_getter
    if diagnostic_provider_factory is not None:
        diagnostic_options["diagnostic_provider_factory"] = diagnostic_provider_factory
    return EvaluationApiRuntime(
        settings,
        database=database,
        manifest_loader=probe.load_manifest,
        curated_manifest_loader=load_curated,
        source_lock_loader=load_checked_source,
        query_set_authenticator=(
            query_set_authenticator
            if query_set_authenticator is not None
            else None
            if curated_manifest is not None and source_lock is not None
            else lambda _dataset, _query_set, _queries: None
        ),
        credential_check=probe.check_credential,
        bound_catalog_factory=probe.make_catalog,
        search_backend_factory=probe.make_runtime,
        provider_factory=probe.provider,
        embedder_factory=diagnostic_embedder_factory or probe.embedder,
        reranker_factory=probe.reranker,
        worker_guard_factory=worker_guard_factory,
        git_revision_factory=lambda: "a" * 40,
        now=lambda: datetime.now(UTC),
        **diagnostic_options,
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
    current = error.__traceback__
    while current is not None:
        if "/backend/pufferlab/" in current.tb_frame.f_code.co_filename:
            assert marker not in repr(current.tb_frame.f_locals)
        current = current.tb_next


def _assert_fresh_process_control(
    error: BaseException,
    *,
    original: BaseException,
    marker: str,
) -> None:
    assert error is not original
    assert error.__cause__ is None
    assert error.__context__ is None
    assert marker not in str(error)
    assert marker not in repr(error)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert marker not in rendered
    current = error.__traceback__
    while current is not None:
        if "/backend/pufferlab/" in current.tb_frame.f_code.co_filename:
            assert marker not in repr(current.tb_frame.f_locals)
        current = current.tb_next


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


def _diagnostic_observed_score(
    kind: ScoreKind,
    value: float,
    *,
    direct: bool,
) -> ObservedScore:
    return ObservedScore(
        kind=kind,
        value=value,
        direction=(
            ScoreDirection.HIGHER_IS_BETTER
            if kind is ScoreKind.BM25
            else ScoreDirection.LOWER_IS_BETTER
        ),
        source=(ScoreSource.COMPUTE_ATTRIBUTE if direct else ScoreSource.TURBOPUFFER_DIST),
    )


class _DiagnosticRuntimeEmbedder:
    def __init__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        events: list[str],
        error: BaseException | None = None,
        block: bool = False,
    ) -> None:
        self.model = model
        self.revision = revision
        self.dimensions = dimensions
        self._events = events
        self.error = error
        self.block = block
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        assert query_text
        self.calls += 1
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return QueryEmbedding(vector=(0.0,) * self.dimensions, client_duration_ms=0.0)


class _DiagnosticRuntimeProvider:
    def __init__(
        self,
        *,
        query_error: BaseException | None = None,
        close_error: BaseException | None = None,
        block_query: bool = False,
        result_mutation: str | None = None,
    ) -> None:
        self.requests: list[DiagnosticProviderRequest] = []
        self.close_calls = 0
        self.query_error = query_error
        self.close_error = close_error
        self.block_query = block_query
        self.result_mutation = result_mutation
        self.query_started = asyncio.Event()

    async def query(self, request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        self.requests.append(request)
        self.query_started.set()
        if self.block_query:
            await asyncio.Event().wait()
        if self.query_error is not None:
            raise self.query_error
        target = DiagnosticTargetObservation(
            target_document_id=request.target_document_id,
            available=True,
            bm25_score=(
                _diagnostic_observed_score(ScoreKind.BM25, 5.0, direct=True)
                if request.lexical_fields is not None
                else None
            ),
            vector_distance=(
                _diagnostic_observed_score(
                    ScoreKind.VECTOR_DISTANCE,
                    0.25,
                    direct=True,
                )
                if request.query_vector is not None
                else None
            ),
            attributes=tuple(
                DiagnosticAttributeValue(
                    field=field,
                    state=DiagnosticAttributeState.PRESENT_VALUE,
                    value="doc-1",
                )
                for field in request.filter_fields
            ),
        )
        candidates = tuple(
            DiagnosticCandidateList(
                role=role,
                requested_limit=request.candidate_limit,
                rows=(
                    DiagnosticCandidateRow(
                        document_id=request.target_document_id,
                        rank=1,
                        score=(
                            _diagnostic_observed_score(
                                ScoreKind.BM25,
                                5.0,
                                direct=False,
                            )
                            if "bm25" in role.value
                            else _diagnostic_observed_score(
                                ScoreKind.VECTOR_DISTANCE,
                                0.25,
                                direct=False,
                            )
                        ),
                    ),
                ),
            )
            for role in request.roles[1:]
        )
        result = DiagnosticProviderResult(
            namespace=request.namespace,
            target=target,
            candidate_lists=candidates,
            client_duration_ms=0.0,
        )
        if self.result_mutation == "namespace":
            return replace(result, namespace="forged-diagnostic-namespace")
        if self.result_mutation == "role":
            candidate = result.candidate_lists[0]
            return replace(
                result,
                candidate_lists=(
                    replace(
                        candidate,
                        role=DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
                    ),
                    *result.candidate_lists[1:],
                ),
            )
        if self.result_mutation == "limit":
            candidate = result.candidate_lists[0]
            return replace(
                result,
                candidate_lists=(
                    replace(candidate, requested_limit=candidate.requested_limit + 1),
                    *result.candidate_lists[1:],
                ),
            )
        return result

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _DiagnosticRuntimeProbe(_FactoryProbe):
    def __init__(self, manifest: DatasetManifest) -> None:
        super().__init__(manifest)
        self.events: list[str] = []
        self.providers: list[_DiagnosticRuntimeProvider] = []
        self.embedders: list[_DiagnosticRuntimeEmbedder] = []
        self.provider_bindings: list[tuple[str, str]] = []
        self.catalog_mutation: str | None = None
        self.catalog_error: BaseException | None = None
        self.forged_mode: RetrievalMode | None = None
        self.provider_factory_error: BaseException | None = None
        self.provider_query_error: BaseException | None = None
        self.provider_close_error: BaseException | None = None
        self.provider_result_mutation: str | None = None
        self.block_provider_query = False
        self.embedder_factory_error: BaseException | None = None
        self.embedder_error: BaseException | None = None
        self.block_embedder = False

    def load_manifest(self, path: Path) -> DatasetManifest:
        self.events.append("manifest")
        return super().load_manifest(path)

    def diagnostic_credential(self, settings: Settings) -> SecretStr:
        self.events.append("credential")
        credential = settings.turbopuffer_api_key
        assert credential is not None
        return credential

    def make_catalog(
        self,
        dataset: DatasetVersion,
        manifest: DatasetManifest,
        configs: tuple[RetrievalConfig, ...],
    ) -> BoundSearchCatalog:
        self.events.append("catalog")
        if self.catalog_error is not None:
            raise self.catalog_error
        bound = super().make_catalog(dataset, manifest, configs)
        if self.catalog_mutation == "namespace":
            object.__setattr__(dataset, "namespace", "mutated-catalog-namespace")
        elif self.catalog_mutation == "extra":
            bound.catalog._configs = (  # type: ignore[attr-defined]
                *bound.catalog.configs,
                bound.catalog.configs[0],
            )
        elif self.catalog_mutation == "missing":
            bound.catalog._configs = bound.catalog.configs[:-1]  # type: ignore[attr-defined]
        elif self.catalog_mutation == "reordered":
            bound.catalog._configs = tuple(  # type: ignore[attr-defined]
                reversed(bound.catalog.configs)
            )
        elif self.catalog_mutation == "getter":
            bound.catalog.get = lambda _config_id: bound.catalog.configs[0]  # type: ignore[method-assign]
        if self.forged_mode is not None:
            executable = list(bound.catalog.configs)
            index = next(
                value for value, config in enumerate(executable) if config.mode is self.forged_mode
            )
            config = executable[index]
            changes: dict[str, object] = {
                RetrievalMode.BM25.value: {"lexical_fields": (("title", 99.0),)},
                RetrievalMode.VECTOR.value: {"vector_attribute": "forged_vector"},
                RetrievalMode.HYBRID_RRF.value: {"rrf_weights": (2.0, 1.0)},
                RetrievalMode.HYBRID_RERANK.value: {"reranker_depth": 49},
            }[self.forged_mode.value]
            executable[index] = replace(config, **changes)
            bound.catalog._configs = tuple(executable)  # type: ignore[attr-defined]
        return bound

    def make_embedder(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
    ) -> _DiagnosticRuntimeEmbedder:
        self.events.append("embedder")
        if self.embedder_factory_error is not None:
            raise self.embedder_factory_error
        embedder = _DiagnosticRuntimeEmbedder(
            model=model,
            revision=revision,
            dimensions=dimensions,
            events=self.events,
            error=self.embedder_error,
            block=self.block_embedder,
        )
        self.embedders.append(embedder)
        return embedder

    async def make_provider(
        self,
        *,
        api_key: str,
        region: str,
        namespace: str,
    ) -> _DiagnosticRuntimeProvider:
        assert api_key == "test-only-secret"
        self.events.append("provider")
        if self.provider_factory_error is not None:
            raise self.provider_factory_error
        self.provider_bindings.append((region, namespace))
        provider = _DiagnosticRuntimeProvider(
            query_error=self.provider_query_error,
            close_error=self.provider_close_error,
            block_query=self.block_provider_query,
            result_mutation=self.provider_result_mutation,
        )
        self.providers.append(provider)
        return provider


def _diagnostic_runtime(
    settings: Settings,
    database: Database,
    probe: _DiagnosticRuntimeProbe,
) -> EvaluationApiRuntime:
    def authenticate(*_values: object) -> None:
        probe.events.append("auth")

    return _runtime(
        settings,
        database,
        probe,
        diagnostic_credential_getter=probe.diagnostic_credential,
        diagnostic_provider_factory=probe.make_provider,
        diagnostic_embedder_factory=probe.make_embedder,
        query_set_authenticator=authenticate,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("target_index", [0, 1])
async def test_diagnostic_runtime_all_modes_accept_positive_grades_and_are_read_only(
    tmp_path: Path,
    mode: RetrievalMode,
    target_index: int,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"pufferlab_search_namespace": "foreign-playground-setting"}
    )
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"diagnostic-{mode.value}-{target_index}")
    config = _config_for_mode(suite, mode)
    before = database.path.read_bytes()

    response = await runtime.diagnose_expected_document(
        run.id,
        queries[0].id,
        document_ids[target_index],
        ExpectedDocumentDiagnosticRequest(config_id=config.id),
    )

    expected_events = ["auth", "manifest", "credential", "catalog"]
    if mode is not RetrievalMode.BM25:
        expected_events.append("embedder")
    expected_events.append("provider")
    assert probe.events == expected_events
    assert response.config_mode is mode
    assert response.target_document_id == document_ids[target_index]
    assert response.stored_filter_result is None
    assert probe.provider_bindings == [("gcp-us-west1", "pufferlab-live-runtime-test")]
    assert len(probe.providers) == 1
    assert probe.providers[0].close_calls == 1
    assert len(probe.embedders) == (0 if mode is RetrievalMode.BM25 else 1)
    assert probe.calls["runtime"] == probe.calls["reranker"] == 0
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_no_filter", [False, True])
async def test_diagnostic_runtime_authenticates_filtered_suite_with_unrelated_schema_fields(
    tmp_path: Path,
    include_no_filter: bool,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    queries[0] = queries[0].model_copy(
        update={
            "filters": FilterPredicate(
                field="external_id",
                op=PredicateOp.EQ,
                value="doc-1",
            )
        }
    )
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"filtered-{include_no_filter}")
    config = _config_for_mode(suite, RetrievalMode.HYBRID_RRF)

    response = await runtime.diagnose_expected_document(
        run.id,
        queries[0].id,
        document_ids[0],
        ExpectedDocumentDiagnosticRequest(
            config_id=config.id,
            include_no_filter_counterfactual=include_no_filter,
        ),
    )

    assert response.stored_filter_result is not None
    assert response.stored_filter_result.value == "matched"
    assert response.included_no_filter_counterfactual is include_no_filter
    assert [item.field for item in response.filter_evidence] == ["external_id"]
    assert len(response.subqueries) == (5 if include_no_filter else 3)
    assert probe.events == [
        "auth",
        "manifest",
        "credential",
        "catalog",
        "embedder",
        "provider",
    ]
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["grade_zero", "unjudged"])
async def test_diagnostic_runtime_rejects_nonpositive_or_unjudged_target_before_sensitive_work(
    tmp_path: Path,
    target_kind: str,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"invalid-target-{target_kind}")
    target = document_ids[2 if target_kind == "grade_zero" else 3]

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            target,
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    assert raised.value.http_status == 422
    assert raised.value.operation == "diagnose_expected_document"
    assert probe.events == ["auth", "manifest"]
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_rejects_unchecked_direct_request_values_before_repository_work(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    probe = _DiagnosticRuntimeProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    malformed = ExpectedDocumentDiagnosticRequest.model_construct(
        config_id="forged-config",
        include_no_filter_counterfactual=False,
    )

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(  # type: ignore[arg-type]
            _id("run"),
            _id("query"),
            _id("document"),
            malformed,
        )

    assert raised.value.http_status == 422
    assert probe.events == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["run", "query", "document", "request"])
async def test_diagnostic_runtime_rejects_path_or_request_subclasses_before_repository_work(
    tmp_path: Path,
    attack: str,
) -> None:
    class _UuidSubclass(UUID):
        pass

    class _RequestSubclass(ExpectedDocumentDiagnosticRequest):
        pass

    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    probe = _DiagnosticRuntimeProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run_id: UUID = _id("subclass-run")
    query_id: UUID = _id("subclass-query")
    document_id: UUID = _id("subclass-document")
    request: ExpectedDocumentDiagnosticRequest = ExpectedDocumentDiagnosticRequest(
        config_id=_id("subclass-config")
    )
    if attack == "run":
        run_id = _UuidSubclass(str(run_id))
    elif attack == "query":
        query_id = _UuidSubclass(str(query_id))
    elif attack == "document":
        document_id = _UuidSubclass(str(document_id))
    else:
        request = _RequestSubclass(config_id=request.config_id)

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(run_id, query_id, document_id, request)

    assert raised.value.http_status == 422
    assert raised.value.operation == "diagnose_expected_document"
    assert probe.events == []
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_uses_region_captured_before_credential_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)

    def mutate_region(current: Settings) -> SecretStr:
        probe.events.append("credential")
        credential = current.turbopuffer_api_key
        assert credential is not None
        current.turbopuffer_region = "gcp-us-east1"
        return credential

    def authenticate(*_values: object) -> None:
        probe.events.append("auth")

    runtime = _runtime(
        settings,
        database,
        probe,
        diagnostic_credential_getter=mutate_region,
        diagnostic_provider_factory=probe.make_provider,
        diagnostic_embedder_factory=probe.make_embedder,
        query_set_authenticator=authenticate,
    )
    await runtime.start()
    run = _create_replay_run(repository, suite, "captured-region")
    response = await runtime.diagnose_expected_document(
        run.id,
        queries[0].id,
        document_ids[0],
        ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
    )

    assert response.target_document_id == document_ids[0]
    assert probe.provider_bindings == [("gcp-us-west1", "pufferlab-live-runtime-test")]
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "forged_mode", "selected_mode"),
    [
        ("namespace", None, RetrievalMode.BM25),
        ("extra", None, RetrievalMode.BM25),
        ("missing", None, RetrievalMode.BM25),
        ("reordered", None, RetrievalMode.BM25),
        ("getter", None, RetrievalMode.BM25),
        ("forged", RetrievalMode.BM25, RetrievalMode.BM25),
        ("forged", RetrievalMode.VECTOR, RetrievalMode.VECTOR),
        ("forged", RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RRF),
        ("forged", RetrievalMode.HYBRID_RERANK, RetrievalMode.HYBRID_RERANK),
        ("forged", RetrievalMode.VECTOR, RetrievalMode.BM25),
    ],
)
async def test_diagnostic_runtime_rejects_catalog_alias_and_internal_config_forgery(
    tmp_path: Path,
    mutation: str,
    forged_mode: RetrievalMode | None,
    selected_mode: RetrievalMode,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    if forged_mode is None:
        probe.catalog_mutation = mutation
    else:
        probe.forged_mode = forged_mode
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(
        repository,
        suite,
        f"catalog-{mutation}-{forged_mode}-{selected_mode.value}",
    )

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=_config_for_mode(suite, selected_mode).id),
        )

    assert raised.value.http_status == 503
    assert probe.events == ["auth", "manifest", "credential", "catalog"]
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["catalog", "embedder_factory"])
async def test_diagnostic_runtime_factory_failures_are_fixed_before_provider_construction(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    marker = f"PRIVATE_DIAGNOSTIC_{failure_stage.upper()}_MARKER"
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    if failure_stage == "catalog":
        probe.catalog_error = RuntimeError(marker)
    else:
        probe.embedder_factory_error = RuntimeError(marker)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"factory-{failure_stage}")
    config = _config_for_mode(
        suite,
        RetrievalMode.BM25 if failure_stage == "catalog" else RetrievalMode.VECTOR,
    )
    before = database.path.read_bytes()

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=config.id),
        )

    _assert_detached_error(raised.value, marker=marker, http_status=503)
    expected = ["auth", "manifest", "credential", "catalog"]
    if failure_stage == "embedder_factory":
        expected.append("embedder")
    assert probe.events == expected
    assert probe.providers == probe.embedders == []
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        "source",
        "duplicate_qrel",
        "foreign_config",
        "missing_filter_field",
        "nonfilterable_field",
        "wrong_filter_type",
        "ineligible_no_filter",
        "namespace",
        "region",
    ],
)
async def test_diagnostic_runtime_tamper_matrix_reaches_zero_sensitive_factories(
    tmp_path: Path,
    tamper: str,
) -> None:
    settings = _settings(tmp_path)
    if tamper == "region":
        settings.turbopuffer_region = "gcp-us-east1"
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    if tamper == "duplicate_qrel":
        queries[0] = queries[0].model_copy(
            update={"qrels": [*queries[0].qrels, queries[0].qrels[0]]}
        )
    filter_by_tamper = {
        "missing_filter_field": FilterPredicate(
            field="missing_field",
            op=PredicateOp.EQ,
            value="doc-1",
        ),
        "nonfilterable_field": FilterPredicate(
            field="body",
            op=PredicateOp.EQ,
            value="doc-1",
        ),
        "wrong_filter_type": FilterPredicate(
            field="external_id",
            op=PredicateOp.EQ,
            value=7,
        ),
    }
    if tamper in filter_by_tamper:
        queries[0] = queries[0].model_copy(update={"filters": filter_by_tamper[tamper]})
    suite = _seed_live_suite(
        repository,
        judged_queries=queries,
        namespace="invalid namespace" if tamper == "namespace" else "pufferlab-live-runtime-test",
    )
    probe = _DiagnosticRuntimeProbe(suite.manifest)

    def authenticate(*_values: object) -> None:
        probe.events.append("auth")
        if tamper == "source":
            raise PersistenceValidationError("PRIVATE_SOURCE_AUTH_MARKER")

    runtime = _runtime(
        settings,
        database,
        probe,
        diagnostic_credential_getter=probe.diagnostic_credential,
        diagnostic_provider_factory=probe.make_provider,
        diagnostic_embedder_factory=probe.make_embedder,
        query_set_authenticator=authenticate,
    )
    await runtime.start()
    run = _create_replay_run(repository, suite, f"tamper-{tamper}")
    config_id = (
        _id("foreign-diagnostic-config") if tamper == "foreign_config" else suite.configs[0].id
    )
    request = ExpectedDocumentDiagnosticRequest(
        config_id=config_id,
        include_no_filter_counterfactual=tamper == "ineligible_no_filter",
    )
    before = database.path.read_bytes()

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            request,
        )

    assert raised.value.http_status == 422
    assert probe.events == (["auth"] if tamper == "source" else ["auth", "manifest"])
    assert probe.calls["credential"] == 0
    assert probe.calls["catalog"] == 0
    assert probe.providers == probe.embedders == []
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential",
    [None, "", "contains space", "line\nbreak"],
)
async def test_diagnostic_runtime_rejects_missing_or_malformed_credentials_before_catalog(
    tmp_path: Path,
    credential: str | None,
) -> None:
    settings = Settings(
        pufferlab_data_dir=tmp_path,
        turbopuffer_api_key=credential,
        turbopuffer_region="gcp-us-west1",
    )
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"credential-{credential!r}")

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    assert raised.value.http_status == 503
    assert probe.events == ["auth", "manifest", "credential"]
    assert probe.calls["catalog"] == 0
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_kind",
    ["none", "wrong_type", "empty", "space", "control", "oversize", "forged_subclass"],
)
async def test_diagnostic_runtime_independently_validates_injected_credential_getter(
    tmp_path: Path,
    credential_kind: str,
) -> None:
    marker = "PRIVATE_FORGED_CREDENTIAL_MARKER"

    class _ForgedCredentialString(str):
        def __len__(self) -> int:
            raise AssertionError(marker)

        def __iter__(self) -> Iterator[str]:
            raise AssertionError(marker)

    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    credential: object = {
        "none": None,
        "wrong_type": "test-only-secret",
        "empty": SecretStr(""),
        "space": SecretStr("contains space"),
        "control": SecretStr("line\nbreak"),
        "oversize": SecretStr("x" * 4097),
        "forged_subclass": SecretStr("temporary"),
    }[credential_kind]
    if credential_kind == "forged_subclass":
        object.__setattr__(credential, "_secret_value", _ForgedCredentialString(marker))

    def injected_getter(_settings: Settings) -> SecretStr:
        probe.events.append("credential")
        return cast(SecretStr, credential)

    def authenticate(*_values: object) -> None:
        probe.events.append("auth")

    runtime = _runtime(
        settings,
        database,
        probe,
        diagnostic_credential_getter=injected_getter,
        diagnostic_provider_factory=probe.make_provider,
        diagnostic_embedder_factory=probe.make_embedder,
        query_set_authenticator=authenticate,
    )
    await runtime.start()
    run = _create_replay_run(repository, suite, f"credential-getter-{credential_kind}")

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    _assert_detached_error(raised.value, marker=marker, http_status=503)
    assert probe.events == ["auth", "manifest", "credential"]
    assert probe.calls["catalog"] == 0
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_uses_fresh_credential_copy_after_catalog_callback_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    original = settings.turbopuffer_api_key
    assert original is not None

    def mutating_catalog(
        dataset: DatasetVersion,
        manifest: DatasetManifest,
        configs: tuple[RetrievalConfig, ...],
    ) -> BoundSearchCatalog:
        bound = probe.make_catalog(dataset, manifest, configs)
        object.__setattr__(original, "_secret_value", "changed-after-validation")
        return bound

    def authenticate(*_values: object) -> None:
        probe.events.append("auth")

    runtime = _runtime(
        settings,
        database,
        probe,
        diagnostic_credential_getter=probe.diagnostic_credential,
        diagnostic_provider_factory=probe.make_provider,
        diagnostic_embedder_factory=probe.make_embedder,
        query_set_authenticator=authenticate,
    )
    runtime._bound_catalog_factory = mutating_catalog
    await runtime.start()
    run = _create_replay_run(repository, suite, "credential-copy")

    response = await runtime.diagnose_expected_document(
        run.id,
        queries[0].id,
        document_ids[0],
        ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
    )

    assert response.target_document_id == document_ids[0]
    assert probe.events == ["auth", "manifest", "credential", "catalog", "provider"]
    assert probe.providers[0].close_calls == 1
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_rejects_synthetic_before_every_sensitive_factory(
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
    probe = _DiagnosticRuntimeProbe(AUTHORED_SYNTHETIC_DEMO.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    query = AUTHORED_SYNTHETIC_DEMO.queries[0].judged_query

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            synthetic.queued_run.id,
            query.id,
            query.qrels[0].document_id,
            ExpectedDocumentDiagnosticRequest(config_id=synthetic.configs[0].id),
        )

    assert raised.value.http_status == 409
    assert raised.value.operation == "diagnose_expected_document"
    assert probe.events == []
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["run", "query"])
async def test_diagnostic_runtime_missing_run_or_query_is_fixed_404_and_zero_sensitive_work(
    tmp_path: Path,
    missing: str,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"missing-{missing}")

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            _id("absent-diagnostic-run") if missing == "run" else run.id,
            _id("missing-query") if missing == "query" else queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    assert raised.value.http_status == 404
    assert probe.events == ([] if missing == "run" else ["auth", "manifest"])
    assert probe.providers == probe.embedders == []
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["factory", "query", "close"])
async def test_diagnostic_runtime_late_failures_are_fixed_detached_and_read_only(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    marker = f"PRIVATE_DIAGNOSTIC_{failure_stage.upper()}_MARKER"
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    queries[0] = queries[0].model_copy(update={"text": f"query-{marker}"})
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    error = RuntimeError(marker)
    if failure_stage == "factory":
        probe.provider_factory_error = error
    elif failure_stage == "query":
        probe.provider_query_error = error
    else:
        probe.provider_close_error = error
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"late-{failure_stage}")
    before = database.path.read_bytes()

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    _assert_detached_error(raised.value, marker=marker, http_status=503)
    assert raised.value.operation == "diagnose_expected_document"
    assert len(probe.providers) == (0 if failure_stage == "factory" else 1)
    if probe.providers:
        assert probe.providers[0].close_calls == 1
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["namespace", "role", "limit"])
async def test_diagnostic_runtime_maps_forged_provider_result_echoes_to_fixed_failure(
    tmp_path: Path,
    mutation: str,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.provider_result_mutation = mutation
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"forged-provider-result-{mutation}")
    before = database.path.read_bytes()

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    assert raised.value.http_status == 503
    assert raised.value.operation == "diagnose_expected_document"
    assert len(probe.providers) == 1
    assert probe.providers[0].close_calls == 1
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_embedding_failure_is_fixed_before_provider_construction(
    tmp_path: Path,
) -> None:
    marker = "PRIVATE_DIAGNOSTIC_EMBEDDING_MARKER"
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.embedder_error = RuntimeError(marker)
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "embedding-failure")
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)
    before = database.path.read_bytes()

    with pytest.raises(EvaluationViewError) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=vector.id),
        )

    _assert_detached_error(raised.value, marker=marker, http_status=503)
    assert probe.events == ["auth", "manifest", "credential", "catalog", "embedder"]
    assert len(probe.embedders) == 1
    assert probe.embedders[0].calls == 1
    assert probe.providers == []
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original", "expected_type"),
    [
        (KeyboardInterrupt("PRIVATE_DIAGNOSTIC_EMBED_CONTROL"), KeyboardInterrupt),
        (SystemExit("PRIVATE_DIAGNOSTIC_EMBED_CONTROL"), SystemExit),
    ],
)
async def test_diagnostic_runtime_embedding_process_control_is_fresh_and_provider_free(
    tmp_path: Path,
    original: BaseException,
    expected_type: type[BaseException],
) -> None:
    marker = "PRIVATE_DIAGNOSTIC_EMBED_CONTROL"
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.embedder_error = original
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"embed-control-{expected_type.__name__}")
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)
    before = database.path.read_bytes()

    with pytest.raises(expected_type) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=vector.id),
        )

    _assert_fresh_process_control(raised.value, original=original, marker=marker)
    assert len(probe.embedders) == 1
    assert probe.embedders[0].calls == 1
    assert probe.providers == []
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_cancellation_during_embedding_constructs_no_provider(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.block_embedder = True
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "blocked-embedding")
    vector = _config_for_mode(suite, RetrievalMode.VECTOR)
    before = database.path.read_bytes()
    task = asyncio.create_task(
        runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=vector.id),
        )
    )
    for _ in range(500):
        if probe.embedders and probe.embedders[0].started.is_set():
            break
        await asyncio.sleep(0.002)
    assert probe.embedders and probe.embedders[0].started.is_set()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert probe.providers == []
    probe.embedders[0].release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert probe.providers == []
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
async def test_diagnostic_runtime_cancellation_drains_provider_close_without_partial_response(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.block_provider_query = True
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, "cancelled-diagnostic")
    before = database.path.read_bytes()
    task = asyncio.create_task(
        runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )
    )
    for _ in range(500):
        if probe.providers and probe.providers[0].query_started.is_set():
            break
        await asyncio.sleep(0.002)
    assert probe.providers and probe.providers[0].query_started.is_set()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert probe.providers[0].close_calls == 1
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original", "expected_type"),
    [
        (KeyboardInterrupt("PRIVATE_DIAGNOSTIC_CONTROL_MARKER"), KeyboardInterrupt),
        (SystemExit("PRIVATE_DIAGNOSTIC_CONTROL_MARKER"), SystemExit),
    ],
)
async def test_diagnostic_runtime_original_process_control_wins_after_one_close(
    tmp_path: Path,
    original: BaseException,
    expected_type: type[BaseException],
) -> None:
    marker = "PRIVATE_DIAGNOSTIC_CONTROL_MARKER"
    settings = _settings(tmp_path)
    database = Database.from_settings(settings)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    queries, document_ids = _replay_queries()
    suite = _seed_live_suite(repository, judged_queries=queries)
    probe = _DiagnosticRuntimeProbe(suite.manifest)
    probe.provider_query_error = original
    probe.provider_close_error = RuntimeError("PRIVATE_DIAGNOSTIC_CLOSE_MARKER")
    runtime = _diagnostic_runtime(settings, database, probe)
    await runtime.start()
    run = _create_replay_run(repository, suite, f"control-{expected_type.__name__}")
    before = database.path.read_bytes()

    with pytest.raises(expected_type) as raised:
        await runtime.diagnose_expected_document(
            run.id,
            queries[0].id,
            document_ids[0],
            ExpectedDocumentDiagnosticRequest(config_id=suite.configs[0].id),
        )

    _assert_fresh_process_control(raised.value, original=original, marker=marker)
    assert probe.providers[0].close_calls == 1
    assert database.path.read_bytes() == before
    await runtime.shutdown_execution()
    runtime.dispose()
