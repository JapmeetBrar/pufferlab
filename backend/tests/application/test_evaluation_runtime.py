from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from fastapi.testclient import TestClient
from pufferlab.application.evaluation_runtime import EvaluationApiRuntime
from pufferlab.application.evaluations import create_evaluation_run
from pufferlab.application.view_errors import EvaluationViewError
from pufferlab.config import Settings
from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion, FtsProfile, IndexProfile
from pufferlab.contracts.evals import (
    CreateEvalRunRequest,
    EvalRun,
    EvalRunStatus,
    QuerySet,
    RunEnvironment,
)
from pufferlab.contracts.retrieval import RetrievalConfig, RetrievalConfigSummary
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.main import create_app
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.retrieval.config import BoundSearchCatalog, derive_bound_retrieval_configs
from pufferlab.retrieval.errors import provider_failed
from pufferlab.retrieval.types import SearchExecuteRequest, SearchExecuteResult
from pufferlab.synthetic_demo import AUTHORED_SYNTHETIC_DEMO
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo

_TEST_NAMESPACE = UUID("cc1bc5f7-0f4e-4b99-a8ad-8cc647027700")
_NOW = datetime(2026, 8, 23, 0, 0, tzinfo=UTC)


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
    ) -> _BlockingBackend:
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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        pufferlab_data_dir=tmp_path,
        turbopuffer_api_key="test-only-secret",
        turbopuffer_region="gcp-us-west1",
    )


def _seed_live_suite(repository: PufferLabRepository) -> _LiveSuite:
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
    query_set = QuerySet(
        id=_id("live-query-set"),
        name="PufferLab-authored runtime test queries",
        version="v1",
        dataset_version_id=dataset.id,
        query_count=50,
        content_hash="pufferlab-authored-runtime-query-hash",
        created_at=_NOW,
    )
    repository.put_dataset_version(dataset)
    for config in configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(
        query_set,
        [item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries],
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
) -> EvaluationApiRuntime:
    return EvaluationApiRuntime(
        settings,
        database=database,
        manifest_loader=probe.load_manifest,
        credential_check=probe.check_credential,
        bound_catalog_factory=probe.make_catalog,
        search_backend_factory=probe.make_runtime,
        provider_factory=probe.provider,
        embedder_factory=probe.embedder,
        reranker_factory=probe.reranker,
        git_revision_factory=lambda: "a" * 40,
        now=lambda: datetime.now(UTC),
    )


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
