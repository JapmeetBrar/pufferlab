"""Production composition for the durable Unix evaluation CLI."""

from __future__ import annotations

import importlib.util
import platform as runtime_platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pufferlab.application import EvaluationApplicationService, EvaluationSeedResult
from pufferlab.cli.evaluation import (
    ConfigSeedOptions,
    EvalRunOptions,
    EvaluationCommandError,
    ProgressCallback,
    SeedResult,
    UnixIngestOptions,
)
from pufferlab.config import Settings
from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion
from pufferlab.contracts.evals import (
    CreateEvalRunRequest,
    EvalRun,
    EvalRunExport,
    QuerySet,
    RunEnvironment,
)
from pufferlab.datasets import (
    IngestionCheckpointStore,
    UnixDatasetApplicationService,
    UnixIngestionResult,
    load_unix_dataset_manifest,
)
from pufferlab.datasets.embeddings import SentenceTransformerDocumentEmbedder
from pufferlab.datasets.ingestion import (
    Embedder,
    IngestionProgress,
    IngestionService,
)
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.turbopuffer_writer import TurbopufferNamespaceWriter
from pufferlab.jobs import RunJobManager
from pufferlab.jobs.eval_runner import export_outcome_record
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.retrieval.config import BoundSearchCatalog, bind_retrieval_catalog
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from pufferlab.retrieval.types import SearchBackend

_DEFAULT_UNIX_MANIFEST = Path("datasets/cqadupstack-unix/dataset-manifest.json")


class _IngestionProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> TurbopufferProvider: ...


class _DocumentEmbedderFactory(Protocol):
    def __call__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        batch_size: int,
    ) -> Embedder: ...


class _UnixService(Protocol):
    async def ingest(
        self,
        *,
        namespace: str,
        on_progress: Callable[[IngestionProgress], None] | None = None,
    ) -> UnixIngestionResult: ...


type UnixServiceFactory = Callable[
    [IngestionService, IngestionCheckpointStore, UnixIngestOptions],
    _UnixService,
]
type SearchBackendFactory = Callable[
    [Settings, DatasetManifest, BoundSearchCatalog],
    SearchBackend,
]


class RuntimeCliApplication:
    """Own SQLite, provider/model clients, and one command-scoped evaluation lifecycle."""

    def __init__(
        self,
        settings: Settings,
        *,
        unix_manifest_path: Path = _DEFAULT_UNIX_MANIFEST,
        ingestion_provider_factory: _IngestionProviderFactory = TurbopufferProvider,
        document_embedder_factory: _DocumentEmbedderFactory = (SentenceTransformerDocumentEmbedder),
        unix_service_factory: UnixServiceFactory | None = None,
        search_backend_factory: SearchBackendFactory | None = None,
        git_revision_factory: Callable[[], str] | None = None,
        optional_runtime_available: Callable[[], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._unix_manifest_path = unix_manifest_path
        self._ingestion_provider_factory = ingestion_provider_factory
        self._document_embedder_factory = document_embedder_factory
        self._unix_service_factory = unix_service_factory or _make_unix_service
        self._search_backend_factory = search_backend_factory or _make_search_backend
        self._git_revision_factory = git_revision_factory or _git_revision
        self._optional_runtime_available = (
            optional_runtime_available or _sentence_transformers_available
        )
        self._database = Database.from_settings(settings)
        self._database.migrate()
        self._repository = PufferLabRepository(self._database.session_factory)
        self._job_manager = RunJobManager(self._repository)
        self._job_manager.recover_startup()
        self._evaluation_service: EvaluationApplicationService | None = None
        self._closed = False

    async def ingest_unix(
        self,
        options: UnixIngestOptions,
        *,
        emit: Callable[[str], None],
    ) -> SeedResult:
        self._require_open()
        self._require_optional_runtime()
        manifest = load_unix_dataset_manifest(options.dataset_manifest_path)
        provider = self._ingestion_provider_factory(
            api_key=self._required_api_key(),
            region=self._settings.turbopuffer_region,
        )
        failure: BaseException | None = None
        result: UnixIngestionResult | None = None
        try:
            embedder = self._document_embedder_factory(
                model=manifest.embedding.model,
                revision=manifest.embedding.revision,
                dimensions=manifest.embedding.dimensions,
                batch_size=options.batch_size,
            )
            ingestion = IngestionService(
                embedder,
                TurbopufferNamespaceWriter.from_provider(provider),
                batch_size=options.batch_size,
                max_concurrency=options.max_concurrency,
                readiness_attempts=options.readiness_attempts,
            )
            unix_service = self._unix_service_factory(
                ingestion,
                IngestionCheckpointStore(self._settings.pufferlab_data_dir.resolve()),
                options,
            )
            result = await unix_service.ingest(
                namespace=options.namespace,
                on_progress=_CompactUnixProgress(emit),
            )
        except BaseException as error:
            failure = error
        try:
            await provider.close()
        except Exception:
            if failure is None:
                raise EvaluationCommandError(
                    "Unix ingestion runtime did not close cleanly"
                ) from None
        if failure is not None:
            raise failure
        assert result is not None

        bound = bind_retrieval_catalog(
            result.evaluation_seed.dataset_version,
            manifest,
            namespace=options.namespace,
        )
        runtime_settings = self._settings.model_copy(
            update={"pufferlab_search_namespace": options.namespace}
        )
        service = EvaluationApplicationService(
            repository=self._repository,
            job_manager=self._job_manager,
            search_backend=self._search_backend_factory(runtime_settings, manifest, bound),
        )
        self._evaluation_service = service
        return service.seed(result.evaluation_seed, bound.configs)

    def seed(self, options: ConfigSeedOptions) -> SeedResult:
        self._require_open()
        manifest = load_unix_dataset_manifest(self._unix_manifest_path)
        dataset, query_set = self._select_seed(options.dataset_version_id)
        bound = bind_retrieval_catalog(
            dataset,
            manifest,
            namespace=dataset.namespace,
        )
        for config in bound.configs:
            self._repository.put_retrieval_config(config)
        return EvaluationSeedResult(
            dataset_version=dataset,
            query_set=query_set,
            configs=bound.configs,
        )

    async def run(
        self,
        options: EvalRunOptions,
        *,
        run_id: UUID,
        on_progress: ProgressCallback,
    ) -> EvalRun:
        self._require_open()
        self._require_optional_runtime()
        manifest = load_unix_dataset_manifest(self._unix_manifest_path)
        dataset, query_set, bound = self._select_run_suite(options, manifest=manifest)
        runtime_settings = self._settings.model_copy(
            update={"pufferlab_search_namespace": dataset.namespace}
        )
        search_backend = self._search_backend_factory(runtime_settings, manifest, bound)
        service = EvaluationApplicationService(
            repository=self._repository,
            job_manager=self._job_manager,
            search_backend=search_backend,
        )
        self._evaluation_service = service
        request = CreateEvalRunRequest(
            query_set_id=query_set.id,
            baseline_config_id=bound.configs[0].id,
            candidate_config_ids=[config.id for config in bound.configs[1:]],
            random_seed=options.random_seed,
            max_concurrency=options.max_concurrency,
            warmup_query_count=options.warmup_query_count,
        )
        environment = RunEnvironment(
            pufferlab_git_revision=self._git_revision_factory(),
            turbopuffer_region=self._settings.turbopuffer_region,
            python_version=runtime_platform.python_version(),
            platform=runtime_platform.platform(),
            max_concurrency=options.max_concurrency,
            warmup_query_count=options.warmup_query_count,
            query_embedding_cache_enabled=False,
        )
        return await service.run(
            request,
            environment,
            run_id=run_id,
            on_progress=on_progress,
        )

    async def cancel_and_drain(self, run_id: UUID) -> EvalRun:
        self._require_open()
        if self._evaluation_service is not None:
            return await self._evaluation_service.cancel(run_id)
        return await self._job_manager.cancel(run_id)

    def export(self, run_id: UUID) -> EvalRunExport:
        self._require_open()
        run = self._repository.get_run(run_id)
        outcomes = sorted(
            self._repository.list_outcomes(run_id),
            key=lambda outcome: (str(outcome.config_id), str(outcome.query_id)),
        )
        return EvalRunExport(
            run=run,
            outcomes=[export_outcome_record(outcome) for outcome in outcomes],
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._evaluation_service is not None:
                await self._evaluation_service.close()
            else:
                await self._job_manager.close()
        finally:
            self._database.dispose()

    def _select_seed(self, dataset_version_id: UUID | None) -> tuple[DatasetVersion, QuerySet]:
        if dataset_version_id is not None:
            dataset = self._repository.get_dataset_version(dataset_version_id)
            query_sets = self._eligible_query_sets(dataset)
        else:
            candidates: list[tuple[DatasetVersion, QuerySet]] = []
            for candidate in self._repository.list_dataset_versions():
                if candidate.status is not DatasetStatus.READY:
                    continue
                candidate_query_sets = self._eligible_query_sets(candidate)
                if len(candidate_query_sets) == 1:
                    candidates.append((candidate, candidate_query_sets[0]))
            if len(candidates) != 1:
                raise EvaluationCommandError(
                    "configuration seed requires one unambiguous READY curated-50 dataset; "
                    "pass --dataset-version",
                    exit_code=2,
                )
            return candidates[0]

        if dataset.status is not DatasetStatus.READY:
            raise EvaluationCommandError("dataset revision is not READY", exit_code=2)
        if len(query_sets) != 1:
            raise EvaluationCommandError(
                "dataset revision must have one curated 50-query set",
                exit_code=2,
            )
        return dataset, query_sets[0]

    def _select_run_suite(
        self,
        options: EvalRunOptions,
        *,
        manifest: DatasetManifest,
    ) -> tuple[DatasetVersion, QuerySet, BoundSearchCatalog]:
        if options.seeded_defaults:
            suites: list[tuple[DatasetVersion, QuerySet, BoundSearchCatalog]] = []
            for dataset in self._repository.list_dataset_versions():
                if dataset.status is not DatasetStatus.READY:
                    continue
                query_sets = self._eligible_query_sets(dataset)
                if len(query_sets) != 1:
                    continue
                try:
                    bound = bind_retrieval_catalog(
                        dataset,
                        manifest,
                        namespace=dataset.namespace,
                    )
                except ValueError:
                    continue
                if self._persisted_suite_matches(bound):
                    suites.append((dataset, query_sets[0], bound))
            if len(suites) != 1:
                raise EvaluationCommandError(
                    "--seeded-defaults requires one unambiguous persisted Unix suite",
                    exit_code=2,
                )
            return suites[0]

        if options.query_set_id is None or options.baseline_config_id is None:
            raise EvaluationCommandError("explicit evaluation IDs are incomplete", exit_code=2)
        query_set, _ = self._repository.get_query_set(options.query_set_id)
        dataset = self._repository.get_dataset_version(query_set.dataset_version_id)
        bound = bind_retrieval_catalog(dataset, manifest, namespace=dataset.namespace)
        expected_ids = tuple(config.id for config in bound.configs)
        actual_ids = (options.baseline_config_id, *options.candidate_config_ids)
        if actual_ids != expected_ids:
            raise EvaluationCommandError(
                "explicit evaluation IDs must name the canonical BM25 baseline and three "
                "ordered candidates",
                exit_code=2,
            )
        if not self._persisted_suite_matches(bound):
            raise EvaluationCommandError(
                "evaluation configurations are absent or differ from their immutable seed",
                exit_code=2,
            )
        return dataset, query_set, bound

    def _persisted_suite_matches(self, bound: BoundSearchCatalog) -> bool:
        persisted = {
            config.id: config
            for config in self._repository.list_retrieval_configs(
                dataset_version_id=bound.dataset_version.id
            )
        }
        return all(persisted.get(config.id) == config for config in bound.configs)

    def _eligible_query_sets(self, dataset: DatasetVersion) -> list[QuerySet]:
        return [
            query_set
            for query_set in self._repository.list_query_sets(dataset_version_id=dataset.id)
            if query_set.query_count == 50
        ]

    def _required_api_key(self) -> str:
        value = self._settings.turbopuffer_api_key
        if value is None or not value.get_secret_value():
            raise EvaluationCommandError(
                "TURBOPUFFER_API_KEY is required for this command",
                exit_code=2,
            )
        return value.get_secret_value()

    def _require_optional_runtime(self) -> None:
        if not self._optional_runtime_available():
            raise EvaluationCommandError(
                "sentence-transformers is not installed; run `uv sync --extra live-search`",
                exit_code=2,
            )
        self._required_api_key()

    def _require_open(self) -> None:
        if self._closed:
            raise EvaluationCommandError("evaluation application is closed")


class _CompactUnixProgress:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._last: tuple[object, ...] | None = None

    def __call__(self, progress: IngestionProgress) -> None:
        current = (
            progress.state,
            progress.batches_completed,
            progress.documents_completed,
        )
        if current == self._last:
            return
        self._last = current
        self._emit(
            f"progress state={progress.state.value} "
            f"batches={progress.batches_completed}/{progress.batches_total} "
            f"documents={progress.documents_completed}/{progress.documents_total}"
        )


def _make_unix_service(
    ingestion: IngestionService,
    checkpoints: IngestionCheckpointStore,
    options: UnixIngestOptions,
) -> UnixDatasetApplicationService:
    return UnixDatasetApplicationService.from_paths(
        ingestion,
        checkpoints,
        processed_path=options.processed_pack_path,
        source_lock_path=options.source_lock_path,
        processed_pack_lock_path=options.processed_pack_lock_path,
        dataset_manifest_path=options.dataset_manifest_path,
        curated_manifest_path=options.curated_manifest_path,
    )


def _make_search_backend(
    settings: Settings,
    manifest: DatasetManifest,
    bound: BoundSearchCatalog,
) -> SearchBackend:
    return RuntimeSearchBackend(
        settings=settings,
        manifest=manifest,
        bound_catalog=bound,
    )


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


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
