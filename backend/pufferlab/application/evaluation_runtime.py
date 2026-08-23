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
from pufferlab.application.evaluation_forensics import (
    CounterfactualProbeAnalysis,
    analyze_counterfactual_probe,
    annotate_primary_with_exact_grades,
    build_forensic_observations,
    exact_qrel_grades,
    failed_counterfactual_probe,
)
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
    JudgedQuery,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
    ReplayFailedCounterfactualProbe,
)
from pufferlab.contracts.retrieval import (
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
)
from pufferlab.contracts.search import SearchCompareRequest
from pufferlab.datasets import load_unix_dataset_manifest
from pufferlab.datasets.cqadupstack import (
    CuratedQueryManifest,
    DatasetPreparationError,
    SourceLock,
    load_curated_query_manifest,
    load_source_lock,
)
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.unix_application import authenticate_persisted_unix_query_set
from pufferlab.jobs import RunJobManager
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.errors import (
    PersistenceError,
    PersistenceValidationError,
    RecordNotFoundError,
)
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.rerankers import Reranker, SentenceTransformersReranker
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.retrieval.config import (
    BoundSearchCatalog,
    derive_bound_retrieval_configs,
)
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from pufferlab.retrieval.types import (
    HybridProbeExecuteRequest,
    QueryEmbedder,
    ReplaySearchBackend,
    RetrievalProvider,
    SearchBackend,
)

_DEFAULT_UNIX_MANIFEST = Path("datasets/cqadupstack-unix/dataset-manifest.json")
_DEFAULT_UNIX_CURATED_MANIFEST = Path("datasets/cqadupstack-unix/curated-50.json")
_DEFAULT_UNIX_SOURCE_LOCK = Path("datasets/cqadupstack-unix/source-lock.json")
_MAX_ACTIVE_RUNS = 1
_MAX_PERSISTED_ACTIVE_ROWS = 100
_REPLAY_NOTICE = (
    "Recorded outcomes contain no original stage evidence. Primary results are a new live replay; "
    "raw candidate probes are separate counterfactual requests and never establish causality."
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
    ) -> ReplaySearchBackend: ...


class _WorkerGuard(Protocol):
    def acquire(self) -> None: ...

    def release(self) -> None: ...


class _QuerySetAuthenticator(Protocol):
    def __call__(
        self,
        dataset: DatasetVersion,
        query_set: QuerySet,
        judged_queries: tuple[JudgedQuery, ...],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _RunBinding:
    dataset: DatasetVersion
    manifest: DatasetManifest
    configs: tuple[RetrievalConfig, ...]


@dataclass(frozen=True, slots=True)
class _ReplayBinding:
    dataset: DatasetVersion
    manifest: DatasetManifest
    configs: tuple[RetrievalConfig, ...]
    selected_configs: tuple[RetrievalConfig, RetrievalConfig]
    query: JudgedQuery
    grades: dict[UUID, int]


class EvaluationApiRuntime:
    """Compose SQLite views and one bounded local evaluation execution owner."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        unix_manifest_path: Path = _DEFAULT_UNIX_MANIFEST,
        unix_curated_manifest_path: Path = _DEFAULT_UNIX_CURATED_MANIFEST,
        unix_source_lock_path: Path = _DEFAULT_UNIX_SOURCE_LOCK,
        manifest_loader: Callable[[Path], DatasetManifest] = load_unix_dataset_manifest,
        curated_manifest_loader: Callable[[Path], CuratedQueryManifest] = (
            load_curated_query_manifest
        ),
        source_lock_loader: Callable[[Path], SourceLock] = load_source_lock,
        query_set_authenticator: _QuerySetAuthenticator | None = None,
        credential_check: Callable[[Settings], None] | None = None,
        bound_catalog_factory: _BoundCatalogFactory | None = None,
        search_backend_factory: _SearchBackendFactory | None = None,
        provider_factory: _ProviderFactory = TurbopufferProvider,
        embedder_factory: _EmbedderFactory = SentenceTransformerQueryEmbedder,
        reranker_factory: _RerankerFactory = SentenceTransformersReranker,
        worker_guard_factory: Callable[[Path], _WorkerGuard] | None = None,
        git_revision_factory: Callable[[], str] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        trace_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._settings = settings
        self._database = database or Database.from_settings(settings)
        self._repository = PufferLabRepository(self._database.session_factory)
        self._views = EvaluationViewService(self._repository)
        self._provider_free_controls = ProviderFreeEvaluationControls(self._views)
        self._job_manager = RunJobManager(self._repository)
        self._unix_manifest_path = unix_manifest_path
        self._unix_curated_manifest_path = unix_curated_manifest_path
        self._unix_source_lock_path = unix_source_lock_path
        self._manifest_loader = manifest_loader
        self._curated_manifest_loader = curated_manifest_loader
        self._source_lock_loader = source_lock_loader
        self._query_set_authenticator = (
            query_set_authenticator or self._authenticate_persisted_query_set
        )
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
        self._trace_id_factory = trace_id_factory
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
            error: EvaluationViewError | None = None
            try:
                self._worker_guard.acquire()
                self._database.migrate()
                self._job_manager.recover_startup()
                self._started = True
                self._pump_queued_locked()
            except EvaluationViewError as caught:
                error = _copy_view_error(caught)
            except BaseException:
                error = evaluation_unavailable(operation="start_evaluation_runtime")
            if error is not None:
                self._started = False
                self._worker_guard.release()
                raise error

    async def create_eval_run(self, request: CreateEvalRunRequest) -> CreateEvalRunResponse:
        async with self._control_lock:
            self._require_started("create_eval_run")
            error: EvaluationViewError | None = None
            try:
                binding = self._resolve_request_binding(request)
                active = self._repository.list_active_runs(limit=_MAX_PERSISTED_ACTIVE_ROWS)
            except EvaluationViewError as caught:
                error = _copy_view_error(caught)
            except RecordNotFoundError:
                error = evaluation_invalid(
                    message="evaluation request references an unknown immutable revision",
                    operation="create_eval_run",
                )
            except PersistenceValidationError:
                error = evaluation_invalid(
                    message="evaluation request does not match the canonical stored suite",
                    operation="create_eval_run",
                )
            except BaseException:
                error = evaluation_unavailable(operation="create_eval_run")
            if error is not None:
                raise error

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
            error = None
            try:
                run = create_evaluation_run(
                    self._repository,
                    request,
                    environment,
                    now=self._now,
                )
            except PersistenceValidationError:
                error = evaluation_invalid(
                    message="evaluation request does not match the canonical stored suite",
                    operation="create_eval_run",
                )
            if error is not None:
                raise error
            self._schedule_driver_locked(run.id, binding=binding)
            # No await occurs after durable persistence and scheduling, so request cancellation
            # cannot own or cancel the background driver.
            return CreateEvalRunResponse(result=self._views.get_eval_run(run.id).result)

    async def cancel_eval_run(self, run_id: UUID) -> CancelEvalRunResponse:
        service: EvaluationApplicationService | None = None
        async with self._control_lock:
            self._require_started("cancel_eval_run")
            error: EvaluationViewError | None = None
            try:
                run = self._repository.get_run(run_id)
                dataset = self._dataset_for_run(run)
            except RecordNotFoundError:
                error = evaluation_not_found(
                    message="evaluation run was not found",
                    operation="cancel_eval_run",
                )
            except PersistenceError:
                error = evaluation_unavailable(operation="cancel_eval_run")
            if error is not None:
                raise error
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
        async with self._control_lock:
            self._require_started("replay_eval_query")
            binding_error: EvaluationViewError | None = None
            binding: _ReplayBinding | None = None
            try:
                binding = self._resolve_replay_binding(run_id, query_id, request)
            except EvaluationViewError as caught:
                binding_error = _copy_view_error(caught)
            except RecordNotFoundError:
                binding_error = evaluation_not_found(
                    message="evaluation run or query was not found",
                    operation="replay_eval_query",
                )
            except (DatasetPreparationError, PersistenceValidationError, ValueError):
                binding_error = evaluation_invalid(
                    message="replay request does not match the immutable stored run",
                    operation="replay_eval_query",
                )
            except BaseException:
                binding_error = evaluation_unavailable(operation="replay_eval_query")
            if binding_error is not None:
                raise binding_error
            assert binding is not None

            backend_error: EvaluationViewError | None = None
            backend: ReplaySearchBackend | None = None
            try:
                self._credential_check(self._settings)
                bound = self._bound_catalog_factory(
                    binding.dataset,
                    binding.manifest,
                    binding.configs,
                )
                runtime_settings = self._settings.model_copy(
                    update={"pufferlab_search_namespace": binding.dataset.namespace}
                )
                backend = self._search_backend_factory(
                    runtime_settings,
                    binding.manifest,
                    bound,
                    self._provider_factory,
                    self._embedder_factory,
                    self._reranker_factory,
                )
            except BaseException:
                backend_error = evaluation_unavailable(operation="replay_eval_query")
            if backend_error is not None:
                raise backend_error
            assert backend is not None

        return await self._execute_replay(
            backend,
            run_id=run_id,
            request=request,
            binding=binding,
        )

    async def _execute_replay(
        self,
        backend: ReplaySearchBackend,
        *,
        run_id: UUID,
        request: EvalRunQueryReplayRequest,
        binding: _ReplayBinding,
    ) -> EvalRunQueryReplayResponse:
        response: EvalRunQueryReplayResponse | None = None
        failure: EvaluationViewError | ProviderError | SearchError | None = None
        cancelled: asyncio.CancelledError | None = None
        try:
            raw_primary = await backend.compare(
                SearchCompareRequest(
                    query_text=binding.query.text,
                    config_ids=list(request.config_ids),
                    query_id=binding.query.id,
                    filter_override=binding.query.filters,
                    expected_document_ids=[],
                    debug_provenance=False,
                )
            )
            expected_summaries = tuple(
                _retrieval_config_summary(config) for config in binding.selected_configs
            )
            if (
                raw_primary.query_text != binding.query.text
                or raw_primary.query_id != binding.query.id
                or tuple(result.config for result in raw_primary.results) != expected_summaries
            ):
                raise ValueError("primary replay response does not match its exact binding")
            primary_observed_at = self._now()
            primary = annotate_primary_with_exact_grades(raw_primary, binding.grades)
            used_trace_ids = {result.trace_id for result in primary.results}
            if len(used_trace_ids) != len(primary.results):
                raise ValueError("primary replay traces are not distinct")

            analyses: dict[UUID, CounterfactualProbeAnalysis] = {}
            failed: dict[UUID, ReplayFailedCounterfactualProbe] = {}
            if request.include_counterfactual_probe:
                for config in binding.selected_configs:
                    if config.mode not in {
                        RetrievalMode.HYBRID_RRF,
                        RetrievalMode.HYBRID_RERANK,
                    }:
                        continue
                    trace_id = self._trace_id_factory()
                    if trace_id in used_trace_ids:
                        raise ValueError("primary and counterfactual traces must be disjoint")
                    used_trace_ids.add(trace_id)
                    try:
                        execution = await backend.probe_hybrid_candidates(
                            HybridProbeExecuteRequest(
                                namespace=binding.dataset.namespace,
                                query_text=binding.query.text,
                                config_id=config.id,
                                trace_id=trace_id,
                                query_id=binding.query.id,
                                filter_override=binding.query.filters,
                            )
                        )
                        if (
                            execution.config_id != config.id
                            or execution.query_id != binding.query.id
                            or execution.trace_id != trace_id
                        ):
                            raise ValueError(
                                "counterfactual response does not match its exact binding"
                            )
                        observed_at = self._now()
                        primary_result = next(
                            result for result in primary.results if result.config.id == config.id
                        )
                        analyses[config.id] = analyze_counterfactual_probe(
                            execution,
                            observed_at=observed_at,
                            config=config,
                            primary=primary_result,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        observed_at = self._now()
                        failed[config.id] = failed_counterfactual_probe(
                            config_id=config.id,
                            observed_at=observed_at,
                            trace_id=trace_id,
                        )

            selected_by_id = {config.id: config for config in binding.selected_configs}
            observations = build_forensic_observations(
                primary=primary,
                primary_observed_at=primary_observed_at,
                config_ids=(request.config_ids[0], request.config_ids[1]),
                configs=selected_by_id,
                target_document_ids=tuple(binding.grades),
                probe_analyses=analyses,
                failed_probes=failed,
            )
            response = EvalRunQueryReplayResponse(
                run_id=run_id,
                query_id=binding.query.id,
                config_ids=list(request.config_ids),
                primary_observed_at=primary_observed_at,
                primary=primary,
                counterfactual_probes=[
                    analyses[config_id].probe
                    for config_id in request.config_ids
                    if config_id in analyses
                ],
                failed_counterfactual_probes=[
                    failed[config_id] for config_id in request.config_ids if config_id in failed
                ],
                observations=observations,
                observability_notice=_REPLAY_NOTICE,
            )
        except asyncio.CancelledError as caught:
            cancelled = caught
        except EvaluationViewError as caught:
            failure = _copy_view_error(caught)
        except ProviderError as caught:
            failure = _copy_provider_error(caught)
        except SearchError as caught:
            failure = _copy_search_error(caught)
        except BaseException:
            failure = evaluation_unavailable(operation="replay_eval_query")

        close_task = asyncio.create_task(
            backend.close(),
            name=f"pufferlab-replay-close-{run_id}",
        )
        close_failed = False
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as caught:
                if cancelled is None:
                    cancelled = caught
            except BaseException:
                close_failed = True
        if close_task.cancelled():
            if cancelled is None:
                cancelled = asyncio.CancelledError()
        elif close_task.exception() is not None:
            close_failed = True
        if close_failed and failure is None and cancelled is None:
            failure = evaluation_unavailable(operation="replay_eval_query")

        if cancelled is not None:
            raise cancelled from None
        if failure is not None:
            raise failure from None
        if response is None:  # pragma: no cover - every path selects response or failure
            raise AssertionError("replay execution ended without a response")
        return response

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

    def _resolve_replay_binding(
        self,
        run_id: UUID,
        query_id: UUID,
        request: EvalRunQueryReplayRequest,
    ) -> _ReplayBinding:
        run = self._repository.get_run(run_id)
        query_set, judged_queries = self._repository.get_query_set(run.query_set.id)
        expected_summary = QuerySetSummary(
            id=query_set.id,
            name=query_set.name,
            version=query_set.version,
            query_count=query_set.query_count,
            content_hash=query_set.content_hash,
        )
        if run.query_set != expected_summary or run.total_queries != query_set.query_count:
            raise PersistenceValidationError(
                "durable run does not match its immutable query-set revision"
            )
        dataset = self._repository.get_dataset_version(query_set.dataset_version_id)
        self._reject_non_live(dataset, operation="replay_eval_query")
        self._query_set_authenticator(dataset, query_set, tuple(judged_queries))
        configs = tuple(self._repository.list_run_configs(run.id))
        by_id = {config.id: config for config in configs}
        if len(by_id) != len(configs) or any(
            config_id not in by_id for config_id in request.config_ids
        ):
            raise PersistenceValidationError(
                "replay config IDs must belong to the immutable stored run"
            )
        query_by_id = {query.id: query for query in judged_queries}
        try:
            query = query_by_id[query_id]
        except KeyError:
            raise RecordNotFoundError(
                "judged query was not found in the requested query set"
            ) from None
        grades = exact_qrel_grades(query.qrels)

        # Only immutable, provider-free identities are accepted before loading executable assets.
        manifest = self._manifest_loader(self._unix_manifest_path)
        expected = derive_bound_retrieval_configs(
            dataset,
            manifest,
            namespace=dataset.namespace,
        )
        if configs != expected:
            raise PersistenceValidationError(
                "durable replay configs differ from the exact dataset-bound immutable suite"
            )
        selected = (by_id[request.config_ids[0]], by_id[request.config_ids[1]])
        return _ReplayBinding(
            dataset=dataset,
            manifest=manifest,
            configs=configs,
            selected_configs=selected,
            query=query,
            grades=grades,
        )

    def _authenticate_persisted_query_set(
        self,
        dataset: DatasetVersion,
        query_set: QuerySet,
        judged_queries: tuple[JudgedQuery, ...],
    ) -> None:
        curated_manifest = self._curated_manifest_loader(self._unix_curated_manifest_path)
        source_lock = self._source_lock_loader(self._unix_source_lock_path)
        authenticate_persisted_unix_query_set(
            dataset,
            query_set,
            judged_queries,
            curated_manifest=curated_manifest,
            checked_source_lock=source_lock,
        )

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
        fd = _open_worker_guard(self._path, flags)
        if fd is None:
            raise evaluation_unavailable(operation="start_evaluation_runtime")
        lock_result = _try_lock_worker_guard(fd)
        if lock_result != "acquired":
            with suppress(OSError):
                os.close(fd)
        if lock_result == "blocked":
            raise EvaluationViewError(
                code=ApiErrorCode.RUN_CONFLICT,
                message="another PufferLab API worker owns evaluation execution",
                http_status=503,
                operation="start_evaluation_runtime",
            )
        if lock_result == "failed":
            raise evaluation_unavailable(operation="start_evaluation_runtime")
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


def _copy_view_error(error: EvaluationViewError) -> EvaluationViewError:
    return EvaluationViewError(
        code=error.code,
        message=str(error),
        http_status=error.http_status,
        operation=error.operation,
        retryable=error.retryable,
    )


def _copy_provider_error(error: ProviderError) -> ProviderError:
    return ProviderError(str(error), error.details)


def _copy_search_error(error: SearchError) -> SearchError:
    return SearchError(str(error), error.details)


def _open_worker_guard(path: Path, flags: int) -> int | None:
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("evaluation worker guard is not a regular file")
        os.fchmod(fd, 0o600)
        return fd
    except OSError:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        return None


def _try_lock_worker_guard(fd: int) -> str:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "blocked"
    except OSError:
        return "failed"
    return "acquired"


def _make_bound_catalog(
    dataset: DatasetVersion,
    manifest: DatasetManifest,
    configs: tuple[RetrievalConfig, ...],
) -> BoundSearchCatalog:
    validated = tuple(RetrievalConfig.model_validate(config) for config in configs)
    return BoundSearchCatalog(dataset_version=dataset, manifest=manifest, configs=validated)


def _retrieval_config_summary(config: RetrievalConfig) -> RetrievalConfigSummary:
    return RetrievalConfigSummary(
        id=config.id,
        revision=config.revision,
        name=config.name,
        mode=config.mode,
        config_hash=config.config_hash,
    )


def _make_search_backend(
    settings: Settings,
    manifest: DatasetManifest,
    bound: BoundSearchCatalog,
    provider_factory: _ProviderFactory,
    embedder_factory: _EmbedderFactory,
    reranker_factory: _RerankerFactory,
) -> ReplaySearchBackend:
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
