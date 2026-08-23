"""Single-worker API ownership for durable, dataset-bound evaluation jobs."""

from __future__ import annotations

import asyncio
import fcntl
import os
import platform as runtime_platform
import stat
import subprocess
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pufferlab.application.evaluation_controls import ProviderFreeEvaluationControls
from pufferlab.application.evaluation_views import EvaluationViewService
from pufferlab.application.evaluations import (
    EvaluationApplicationService,
    create_evaluation_run,
)
from pufferlab.application.view_errors import (
    EvaluationViewError,
    evaluation_conflict,
    evaluation_invalid,
    evaluation_not_found,
    evaluation_unavailable,
)
from pufferlab.config import Settings
from pufferlab.contracts.datasets import DataOrigin, DatasetVersion
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    EvalRun,
    EvalRunStatus,
    RunEnvironment,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
)
from pufferlab.contracts.retrieval import RetrievalConfig
from pufferlab.datasets import load_unix_dataset_manifest
from pufferlab.datasets.models import DatasetManifest
from pufferlab.jobs import RunJobManager
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.errors import (
    PersistenceError,
    PersistenceValidationError,
    RecordNotFoundError,
)
from pufferlab.providers.rerankers import Reranker, SentenceTransformersReranker
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.retrieval.config import (
    BoundSearchCatalog,
    derive_bound_retrieval_configs,
)
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from pufferlab.retrieval.types import QueryEmbedder, RetrievalProvider, SearchBackend

_DEFAULT_UNIX_MANIFEST = Path("datasets/cqadupstack-unix/dataset-manifest.json")
_MAX_ACTIVE_RUNS = 1
_MAX_PERSISTED_ACTIVE_ROWS = 100


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


class _BoundCatalogFactory(Protocol):
    def __call__(
        self,
        dataset: DatasetVersion,
        manifest: DatasetManifest,
        configs: tuple[RetrievalConfig, ...],
    ) -> BoundSearchCatalog: ...


class _SearchBackendFactory(Protocol):
    def __call__(
        self,
        settings: Settings,
        manifest: DatasetManifest,
        bound: BoundSearchCatalog,
        provider_factory: _ProviderFactory,
        embedder_factory: _EmbedderFactory,
        reranker_factory: _RerankerFactory,
    ) -> SearchBackend: ...


class _WorkerGuard(Protocol):
    def acquire(self) -> None: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _RunBinding:
    dataset: DatasetVersion
    manifest: DatasetManifest
    configs: tuple[RetrievalConfig, ...]


class EvaluationApiRuntime:
    """Compose SQLite views and one bounded local evaluation execution owner."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        unix_manifest_path: Path = _DEFAULT_UNIX_MANIFEST,
        manifest_loader: Callable[[Path], DatasetManifest] = load_unix_dataset_manifest,
        credential_check: Callable[[Settings], None] | None = None,
        bound_catalog_factory: _BoundCatalogFactory | None = None,
        search_backend_factory: _SearchBackendFactory | None = None,
        provider_factory: _ProviderFactory = TurbopufferProvider,
        embedder_factory: _EmbedderFactory = SentenceTransformerQueryEmbedder,
        reranker_factory: _RerankerFactory = SentenceTransformersReranker,
        worker_guard_factory: Callable[[Path], _WorkerGuard] | None = None,
        git_revision_factory: Callable[[], str] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._database = database or Database.from_settings(settings)
        self._repository = PufferLabRepository(self._database.session_factory)
        self._views = EvaluationViewService(self._repository)
        self._provider_free_controls = ProviderFreeEvaluationControls(self._views)
        self._job_manager = RunJobManager(self._repository)
        self._unix_manifest_path = unix_manifest_path
        self._manifest_loader = manifest_loader
        self._credential_check = credential_check or _require_live_credential
        self._bound_catalog_factory = bound_catalog_factory or _make_bound_catalog
        self._search_backend_factory = search_backend_factory or _make_search_backend
        self._provider_factory = provider_factory
        self._embedder_factory = embedder_factory
        self._reranker_factory = reranker_factory
        self._worker_guard = (worker_guard_factory or _FileWorkerGuard)(
            settings.pufferlab_data_dir.resolve() / ".pufferlab-api.lock"
        )
        self._git_revision_factory = git_revision_factory or _git_revision
        self._now = now
        self._control_lock = asyncio.Lock()
        self._drivers: dict[UUID, asyncio.Task[None]] = {}
        self._services: dict[UUID, EvaluationApplicationService] = {}
        self._backends: dict[UUID, SearchBackend] = {}
        self._started = False
        self._closing = False
        self._disposed = False

    @property
    def views(self) -> EvaluationViewService:
        return self._views

    @property
    def database(self) -> Database:
        return self._database

    async def start(self) -> None:
        """Acquire exclusive ownership, migrate, interrupt stale work, and reclaim queued work."""
        async with self._control_lock:
            if self._started:
                return
            if self._closing or self._disposed:
                raise evaluation_unavailable(operation="start_evaluation_runtime")
            try:
                self._worker_guard.acquire()
                self._database.migrate()
                self._job_manager.recover_startup()
                self._started = True
                self._pump_queued_locked()
            except EvaluationViewError:
                self._worker_guard.release()
                raise
            except BaseException:
                self._worker_guard.release()
                raise evaluation_unavailable(operation="start_evaluation_runtime") from None

    async def create_eval_run(self, request: CreateEvalRunRequest) -> CreateEvalRunResponse:
        async with self._control_lock:
            self._require_started("create_eval_run")
            try:
                binding = self._resolve_request_binding(request)
                active = self._repository.list_active_runs(limit=_MAX_PERSISTED_ACTIVE_ROWS)
            except EvaluationViewError:
                raise
            except RecordNotFoundError:
                raise evaluation_invalid(
                    message="evaluation request references an unknown immutable revision",
                    operation="create_eval_run",
                ) from None
            except PersistenceValidationError:
                raise evaluation_invalid(
                    message="evaluation request does not match the canonical stored suite",
                    operation="create_eval_run",
                ) from None
            except BaseException:
                raise evaluation_unavailable(operation="create_eval_run") from None

            suite = (
                request.query_set_id,
                request.baseline_config_id,
                *request.candidate_config_ids,
            )
            if any(self._suite_identity(run) == suite for run in active):
                raise evaluation_conflict(
                    message="an equivalent evaluation run is already active",
                    operation="create_eval_run",
                )
            if len(active) >= _MAX_ACTIVE_RUNS:
                raise evaluation_conflict(
                    message="the single-worker evaluation runtime is at capacity",
                    operation="create_eval_run",
                )

            environment = RunEnvironment(
                pufferlab_git_revision=self._git_revision_factory(),
                turbopuffer_region=self._settings.turbopuffer_region,
                python_version=runtime_platform.python_version(),
                platform=runtime_platform.platform(),
                max_concurrency=request.max_concurrency,
                warmup_query_count=request.warmup_query_count,
                query_embedding_cache_enabled=False,
            )
            try:
                run = create_evaluation_run(
                    self._repository,
                    request,
                    environment,
                    now=self._now,
                )
            except PersistenceValidationError:
                raise evaluation_invalid(
                    message="evaluation request does not match the canonical stored suite",
                    operation="create_eval_run",
                ) from None
            self._schedule_driver_locked(run.id, binding=binding)
            # No await occurs after durable persistence and scheduling, so request cancellation
            # cannot own or cancel the background driver.
            return CreateEvalRunResponse(result=self._views.get_eval_run(run.id).result)

    async def cancel_eval_run(self, run_id: UUID) -> CancelEvalRunResponse:
        service: EvaluationApplicationService | None = None
        async with self._control_lock:
            self._require_started("cancel_eval_run")
            try:
                run = self._repository.get_run(run_id)
                dataset = self._dataset_for_run(run)
            except RecordNotFoundError:
                raise evaluation_not_found(
                    message="evaluation run was not found",
                    operation="cancel_eval_run",
                ) from None
            except PersistenceError:
                raise evaluation_unavailable(operation="cancel_eval_run") from None
            if dataset.data_origin is DataOrigin.SYNTHETIC_DEMO:
                return await self._provider_free_controls.cancel_eval_run(run_id)
            if run.status is EvalRunStatus.QUEUED:
                self._repository.transition_run(run.id, EvalRunStatus.CANCELLED, at=self._now())
            elif run.status is EvalRunStatus.RUNNING:
                service = self._services.get(run.id)
                if service is None:
                    raise evaluation_unavailable(operation="cancel_eval_run")

        if service is not None:
            # The cooperative cancellation remains process-owned if the HTTP request disconnects.
            await asyncio.shield(service.cancel(run_id))
        return CancelEvalRunResponse(result=self._views.get_eval_run(run_id).result)

    async def replay_eval_query(
        self,
        run_id: UUID,
        query_id: UUID,
        request: EvalRunQueryReplayRequest,
    ) -> EvalRunQueryReplayResponse:
        return await self._provider_free_controls.replay_eval_query(run_id, query_id, request)

    async def shutdown_execution(self) -> None:
        """Stop admission, cooperatively drain jobs, and close every eval search runtime."""
        async with self._control_lock:
            if self._closing:
                drivers = tuple(self._drivers.values())
            else:
                self._closing = True
                drivers = tuple(self._drivers.values())
        await self._job_manager.close()
        if drivers:
            await asyncio.gather(*drivers, return_exceptions=True)
        for backend in tuple(self._backends.values()):
            with suppress(Exception):
                await backend.close()
        self._backends.clear()
        self._services.clear()

    def dispose(self) -> None:
        """Dispose SQLite and release the process guard after all search runtimes are closed."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self._database.dispose()
        finally:
            self._worker_guard.release()

    def _resolve_request_binding(self, request: CreateEvalRunRequest) -> _RunBinding:
        query_set = self._repository.get_query_set_revision(request.query_set_id)
        dataset = self._repository.get_dataset_version(query_set.dataset_version_id)
        self._reject_non_live(dataset, operation="create_eval_run")
        manifest = self._manifest_loader(self._unix_manifest_path)
        expected = derive_bound_retrieval_configs(
            dataset,
            manifest,
            namespace=dataset.namespace,
        )
        requested_ids = (request.baseline_config_id, *request.candidate_config_ids)
        configs = tuple(self._repository.get_retrieval_config(value) for value in requested_ids)
        if configs != expected:
            raise PersistenceValidationError(
                "requested configs differ from the exact dataset-bound immutable suite"
            )
        return _RunBinding(dataset=dataset, manifest=manifest, configs=tuple(configs))

    def _resolve_run_binding(self, run: EvalRun) -> _RunBinding:
        dataset = self._dataset_for_run(run)
        self._reject_non_live(dataset, operation="recover_evaluation_run")
        manifest = self._manifest_loader(self._unix_manifest_path)
        expected = derive_bound_retrieval_configs(
            dataset,
            manifest,
            namespace=dataset.namespace,
        )
        configs = tuple(self._repository.list_run_configs(run.id))
        if configs != expected:
            raise PersistenceValidationError(
                "durable run configs differ from the exact dataset-bound immutable suite"
            )
        return _RunBinding(dataset=dataset, manifest=manifest, configs=tuple(configs))

    def _dataset_for_run(self, run: EvalRun) -> DatasetVersion:
        query_set = self._repository.get_query_set_revision(run.query_set.id)
        return self._repository.get_dataset_version(query_set.dataset_version_id)

    def _schedule_driver_locked(
        self,
        run_id: UUID,
        *,
        binding: _RunBinding | None = None,
    ) -> None:
        if run_id in self._drivers:
            return
        task = asyncio.get_running_loop().create_task(
            self._drive_run(run_id, binding=binding),
            name=f"pufferlab-api-evaluation-{run_id}",
        )
        self._drivers[run_id] = task

    def _pump_queued_locked(self) -> None:
        if self._closing:
            return
        active = self._repository.list_active_runs(limit=_MAX_PERSISTED_ACTIVE_ROWS)
        running_count = sum(run.status is EvalRunStatus.RUNNING for run in active)
        scheduled_count = sum(
            run.status is EvalRunStatus.QUEUED and run.id in self._drivers for run in active
        )
        available = max(0, _MAX_ACTIVE_RUNS - running_count - scheduled_count)
        if available == 0:
            return
        for run in active:
            if run.status is EvalRunStatus.QUEUED and run.id not in self._drivers:
                self._schedule_driver_locked(run.id)
                available -= 1
                if available == 0:
                    return

    async def _drive_run(self, run_id: UUID, *, binding: _RunBinding | None) -> None:
        backend: SearchBackend | None = None
        try:
            async with self._control_lock:
                if self._closing:
                    return
                run = self._repository.get_run(run_id)
                if run.status is not EvalRunStatus.QUEUED:
                    return
                try:
                    resolved = binding or self._resolve_run_binding(run)
                    claimed = self._repository.claim_queued_run(run.id, at=self._now())
                    self._credential_check(self._settings)
                    bound = self._bound_catalog_factory(
                        resolved.dataset,
                        resolved.manifest,
                        resolved.configs,
                    )
                    runtime_settings = self._settings.model_copy(
                        update={"pufferlab_search_namespace": resolved.dataset.namespace}
                    )
                    backend = self._search_backend_factory(
                        runtime_settings,
                        resolved.manifest,
                        bound,
                        self._provider_factory,
                        self._embedder_factory,
                        self._reranker_factory,
                    )
                    service = EvaluationApplicationService(
                        repository=self._repository,
                        job_manager=self._job_manager,
                        search_backend=backend,
                        now=self._now,
                    )
                    self._backends[run.id] = backend
                    self._services[run.id] = service
                    service.start_claimed_run(claimed.id)
                except BaseException:
                    self._fail_reconstruction(run.id)
                    return
            await service.drain(run_id)
        except BaseException:
            self._fail_reconstruction(run_id)
        finally:
            if backend is not None:
                with suppress(Exception):
                    await backend.close()
            async with self._control_lock:
                self._backends.pop(run_id, None)
                self._services.pop(run_id, None)
                self._drivers.pop(run_id, None)
                self._pump_queued_locked()

    def _fail_reconstruction(self, run_id: UUID) -> None:
        with suppress(PersistenceError):
            run = self._repository.get_run(run_id)
            if run.status in {EvalRunStatus.QUEUED, EvalRunStatus.RUNNING}:
                self._repository.transition_run(
                    run.id,
                    EvalRunStatus.FAILED,
                    at=self._now(),
                    error=_safe_runtime_error(),
                )

    def _require_started(self, operation: str) -> None:
        if not self._started or self._closing:
            raise evaluation_unavailable(operation=operation)

    @staticmethod
    def _suite_identity(run: EvalRun) -> tuple[UUID, ...]:
        return (run.query_set.id, run.baseline_config_id, *run.candidate_config_ids)

    @staticmethod
    def _reject_non_live(dataset: DatasetVersion, *, operation: str) -> None:
        if dataset.data_origin is DataOrigin.SYNTHETIC_DEMO:
            raise evaluation_conflict(
                message="synthetic demo evaluations are read/export-only",
                operation=operation,
            )


class _FileWorkerGuard:
    """A non-blocking POSIX advisory lock for the documented one-worker P0 runtime."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: int | None = None
        try:
            fd = os.open(self._path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("evaluation worker guard is not a regular file")
            os.fchmod(fd, 0o600)
        except OSError:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            raise evaluation_unavailable(operation="start_evaluation_runtime") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with suppress(OSError):
                os.close(fd)
            raise EvaluationViewError(
                code=ApiErrorCode.RUN_CONFLICT,
                message="another PufferLab API worker owns evaluation execution",
                http_status=503,
                operation="start_evaluation_runtime",
            ) from None
        except OSError:
            with suppress(OSError):
                os.close(fd)
            raise evaluation_unavailable(operation="start_evaluation_runtime") from None
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(fd)


def _require_live_credential(settings: Settings) -> None:
    api_key = settings.turbopuffer_api_key
    if api_key is None or not api_key.get_secret_value():
        raise RuntimeError("live evaluation credentials are unavailable")


def _make_bound_catalog(
    dataset: DatasetVersion,
    manifest: DatasetManifest,
    configs: tuple[RetrievalConfig, ...],
) -> BoundSearchCatalog:
    validated = tuple(RetrievalConfig.model_validate(config) for config in configs)
    return BoundSearchCatalog(dataset_version=dataset, manifest=manifest, configs=validated)


def _make_search_backend(
    settings: Settings,
    manifest: DatasetManifest,
    bound: BoundSearchCatalog,
    provider_factory: _ProviderFactory,
    embedder_factory: _EmbedderFactory,
    reranker_factory: _RerankerFactory,
) -> SearchBackend:
    return RuntimeSearchBackend(
        settings=settings,
        manifest=manifest,
        bound_catalog=bound,
        provider_factory=provider_factory,
        embedder_factory=embedder_factory,
        reranker_factory=reranker_factory,
    )


def _safe_runtime_error() -> ApiErrorDetail:
    return ApiErrorDetail(
        code=ApiErrorCode.INTERNAL_ERROR,
        message="evaluation runtime could not execute the durable run",
        retryable=False,
        trace_id=uuid4(),
        details={"operation": "execute_evaluation"},
    )


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        return "unknown"
    return revision
