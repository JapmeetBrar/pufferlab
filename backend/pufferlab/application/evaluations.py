"""Durable orchestration for a four-configuration judged evaluation."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    CreateEvalRunRequest,
    EvalRun,
    EvalRunExport,
    EvalRunStatus,
    JudgedQuery,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.datasets.unix_application import UnixEvaluationSeed
from pufferlab.jobs.eval_runner import (
    EvaluationOutcomeExecutor,
    export_outcome_record,
    finalize_durable_outcomes,
)
from pufferlab.jobs.manager import ProgressCallback, QueryWorkItem, RunJobManager
from pufferlab.persistence.errors import PersistenceValidationError
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.persistence.types import QueryOutcome
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.rerankers import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.types import SearchBackend, SearchExecuteRequest

_CONFIG_MODE_ORDER = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)


class EvaluationRunError(RuntimeError):
    """A public-safe application failure that never retains the internal exception."""


@dataclass(frozen=True, slots=True)
class EvaluationSeedResult:
    dataset_version: DatasetVersion
    query_set: QuerySet
    configs: tuple[RetrievalConfig, ...]


class EvaluationApplicationService:
    """Seed immutable revisions, schedule durable work, and export canonical run state."""

    def __init__(
        self,
        *,
        repository: PufferLabRepository,
        job_manager: RunJobManager,
        search_backend: SearchBackend,
        run_id_factory: Callable[[], UUID] = uuid4,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._job_manager = job_manager
        self._search_backend = search_backend
        self._run_id_factory = run_id_factory
        self._now = now
        self._tasks: dict[UUID, asyncio.Task[EvalRun]] = {}

    def seed(
        self,
        seed: UnixEvaluationSeed,
        configs: Sequence[RetrievalConfig],
    ) -> EvaluationSeedResult:
        """Persist the dataset-bound query set and one exact four-mode config catalog."""
        by_mode = {config.mode: config for config in configs}
        if len(configs) != len(by_mode) or set(by_mode) != set(_CONFIG_MODE_ORDER):
            raise PersistenceValidationError(
                "evaluation seeding requires exactly one config for each supported mode"
            )
        ordered = tuple(by_mode[mode] for mode in _CONFIG_MODE_ORDER)
        _validate_canonical_seed(seed, ordered)

        self._repository.put_dataset_version(seed.dataset_version)
        for config in ordered:
            self._repository.put_retrieval_config(config)
        query_set, _ = self._repository.put_query_set(seed.query_set, seed.judged_queries)
        return EvaluationSeedResult(
            dataset_version=seed.dataset_version,
            query_set=query_set,
            configs=ordered,
        )

    def create_run(
        self,
        request: CreateEvalRunRequest,
        environment: RunEnvironment,
        *,
        run_id: UUID | None = None,
    ) -> EvalRun:
        """Create one queued, immutable run after validating its exact 50-by-four shape."""
        if len(request.candidate_config_ids) != 3:
            raise PersistenceValidationError(
                "evaluation runs require one baseline and exactly three candidates"
            )
        config_ids = [request.baseline_config_id, *request.candidate_config_ids]
        if len(set(config_ids)) != 4:
            raise PersistenceValidationError("evaluation run config IDs must be distinct")
        query_set, queries = self._repository.get_query_set(request.query_set_id)
        if query_set.query_count != 50 or len(queries) != 50:
            raise PersistenceValidationError("evaluation runs require the curated 50-query set")
        configs = [self._repository.get_retrieval_config(config_id) for config_id in config_ids]
        if {config.mode for config in configs} != set(_CONFIG_MODE_ORDER):
            raise PersistenceValidationError(
                "evaluation runs require BM25, vector, server RRF, and local reranker configs"
            )
        if environment.max_concurrency != request.max_concurrency:
            raise PersistenceValidationError(
                "run environment concurrency must match the create-run request"
            )
        if environment.warmup_query_count != request.warmup_query_count:
            raise PersistenceValidationError(
                "run environment warmup count must match the create-run request"
            )
        if request.warmup_query_count > query_set.query_count:
            raise PersistenceValidationError("warmup query count cannot exceed query-set size")

        run = EvalRun(
            id=run_id or self._run_id_factory(),
            status=EvalRunStatus.QUEUED,
            query_set=QuerySetSummary(
                id=query_set.id,
                name=query_set.name,
                version=query_set.version,
                query_count=query_set.query_count,
                content_hash=query_set.content_hash,
            ),
            baseline_config_id=request.baseline_config_id,
            candidate_config_ids=list(request.candidate_config_ids),
            summaries=[],
            completed_queries=0,
            total_queries=query_set.query_count,
            random_seed=request.random_seed,
            environment=environment,
            created_at=self._now(),
            started_at=None,
            completed_at=None,
            error=None,
        )
        return self._repository.create_run(run)

    def start_run(
        self,
        run_id: UUID,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> asyncio.Task[EvalRun]:
        """Start a queued run; terminal revisions are never reopened or mutated."""
        if run_id in self._tasks:
            raise PersistenceValidationError(f"run {run_id} already has a local task")
        run = self._repository.get_run(run_id)
        if run.status is not EvalRunStatus.QUEUED:
            raise PersistenceValidationError("only a queued run may start")
        task = asyncio.create_task(
            self._execute_run(run, on_progress=on_progress),
            name=f"pufferlab-evaluation-application-{run_id}",
        )
        self._tasks[run_id] = task
        return task

    async def run(
        self,
        request: CreateEvalRunRequest,
        environment: RunEnvironment,
        *,
        run_id: UUID | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> EvalRun:
        created = self.create_run(request, environment, run_id=run_id)
        self.start_run(created.id, on_progress=on_progress)
        return await self.drain(created.id)

    async def cancel(self, run_id: UUID) -> EvalRun:
        """Request cooperative manager cancellation and drain already-started work."""
        await self._job_manager.cancel(run_id)
        task = self._tasks.get(run_id)
        if task is None:
            return self._repository.get_run(run_id)
        try:
            return await task
        finally:
            self._tasks.pop(run_id, None)

    async def drain(self, run_id: UUID) -> EvalRun:
        task = self._tasks.get(run_id)
        if task is None:
            return self._repository.get_run(run_id)
        try:
            return await task
        finally:
            self._tasks.pop(run_id, None)

    async def close(self) -> None:
        """Cooperatively cancel and drain owned work before closing runtime resources."""
        results = await asyncio.gather(
            *(self.cancel(run_id) for run_id in tuple(self._tasks)),
            return_exceptions=True,
        )
        await self._job_manager.close()
        await self._search_backend.close()
        if any(isinstance(result, BaseException) for result in results):
            raise EvaluationRunError("evaluation shutdown encountered a failed run")

    def export(self, run_id: UUID) -> EvalRunExport:
        """Export any durable lifecycle state without depending on repository row ordering."""
        run = self._repository.get_run(run_id)
        outcomes = sorted(
            self._repository.list_outcomes(run_id),
            key=lambda outcome: (str(outcome.config_id), str(outcome.query_id)),
        )
        return EvalRunExport(
            run=run,
            outcomes=[export_outcome_record(outcome) for outcome in outcomes],
        )

    async def _execute_run(
        self,
        run: EvalRun,
        *,
        on_progress: ProgressCallback | None,
    ) -> EvalRun:
        query_set, queries = self._repository.get_query_set(run.query_set.id)
        config_ids = [run.baseline_config_id, *run.candidate_config_ids]
        ordered_queries = _randomized_queries(queries, seed=run.random_seed)
        work_items = _interleaved_work_items(ordered_queries, config_ids)
        try:
            await self._warm_up(
                run_id=run.id,
                query_set=query_set,
                queries=ordered_queries[: run.environment.warmup_query_count],
                config_ids=config_ids,
            )
            current = self._repository.get_run(run.id)
            if current.status is not EvalRunStatus.QUEUED:
                return current
            executor = EvaluationOutcomeExecutor(
                run_id=run.id,
                namespace=self._dataset_namespace(query_set),
                queries={query.id: query for query in queries},
                search_backend=self._search_backend,
                now=self._now,
            )

            async def finalize(
                durable_run: EvalRun,
                durable_outcomes: Sequence[QueryOutcome],
            ) -> Sequence[ConfigRunSummary]:
                return finalize_durable_outcomes(
                    durable_run,
                    durable_outcomes,
                    query_ids=[query.id for query in queries],
                )

            return await self._job_manager.start(
                run.id,
                work_items,
                executor,
                max_concurrency=run.environment.max_concurrency,
                finalize=finalize,
                on_progress=on_progress,
            )
        except (SearchError, ProviderError):
            return self._fail_queued_warmup(run.id)
        except Exception:
            current = self._repository.get_run(run.id)
            if current.status is EvalRunStatus.QUEUED:
                self._fail_queued_warmup(run.id)
            raise EvaluationRunError(
                "evaluation run failed; inspect the durable run status"
            ) from None

    async def _warm_up(
        self,
        *,
        run_id: UUID,
        query_set: QuerySet,
        queries: Sequence[JudgedQuery],
        config_ids: Sequence[UUID],
    ) -> None:
        namespace = self._dataset_namespace(query_set)
        for raw_query in queries:
            query = raw_query
            for config_id in config_ids:
                if self._repository.get_run(run_id).status is not EvalRunStatus.QUEUED:
                    return
                try:
                    await self._search_backend.search_one(
                        SearchExecuteRequest(
                            namespace=namespace,
                            query_text=query.text,
                            config_id=config_id,
                            query_id=query.id,
                            filter_override=query.filters,
                            debug_provenance=False,
                        )
                    )
                except (SearchError, ProviderError):
                    # Warm hints are unmeasured and never determine evaluation quality.
                    continue

    def _dataset_namespace(self, query_set: QuerySet) -> str:
        dataset = self._repository.get_dataset_version(query_set.dataset_version_id)
        return dataset.namespace

    def _fail_queued_warmup(self, run_id: UUID) -> EvalRun:
        current = self._repository.get_run(run_id)
        if current.status is EvalRunStatus.QUEUED:
            current = self._repository.transition_run(run_id, EvalRunStatus.RUNNING, at=self._now())
        if current.status is EvalRunStatus.RUNNING:
            return self._repository.transition_run(
                run_id,
                EvalRunStatus.FAILED,
                at=self._now(),
                error=ApiErrorDetail(
                    code=ApiErrorCode.INTERNAL_ERROR,
                    message="evaluation execution failed",
                    retryable=False,
                    trace_id=uuid4(),
                ),
            )
        return current


def _randomized_queries(queries: Sequence[JudgedQuery], *, seed: int) -> list[JudgedQuery]:
    ordered = list(queries)
    random.Random(seed).shuffle(ordered)
    return ordered


def _interleaved_work_items(
    queries: Sequence[JudgedQuery],
    config_ids: Sequence[UUID],
) -> list[QueryWorkItem]:
    items: list[QueryWorkItem] = []
    for query_index, query in enumerate(queries):
        rotation = query_index % len(config_ids)
        rotated = [*config_ids[rotation:], *config_ids[:rotation]]
        items.extend(QueryWorkItem(config_id=config_id, query_id=query.id) for config_id in rotated)
    return items


def _validate_canonical_seed(
    seed: UnixEvaluationSeed,
    configs: tuple[RetrievalConfig, ...],
) -> None:
    if seed.dataset_version.status is not DatasetStatus.READY:
        raise PersistenceValidationError("evaluation dataset revision must be ready")
    if seed.query_set.query_count != 50 or len(seed.judged_queries) != 50:
        raise PersistenceValidationError("evaluation seeding requires the curated 50-query set")
    if any(config.dataset_version_id != seed.dataset_version.id for config in configs):
        raise PersistenceValidationError(
            "every seeded retrieval config must bind to the evaluation dataset revision"
        )
    if any(
        config.result_k != 50
        or config.candidate_k != 100
        or config.consistency != "strong"
        or config.filters is not None
        for config in configs
    ):
        raise PersistenceValidationError(
            "evaluation configs require result_k=50, candidate_k=100, strong consistency, "
            "and no implicit filters"
        )

    lexical = LexicalSpec()
    vector = VectorSpec(
        attribute=seed.dataset_version.index_profile.vector_attribute,
        embedding_model=seed.dataset_version.index_profile.embedding_model,
    )
    rrf = RrfSpec()
    reranker = RerankerSpec(
        provider="sentence_transformers",
        model=DEFAULT_RERANKER_MODEL,
        revision=DEFAULT_RERANKER_REVISION,
        depth=50,
    )
    expected_specs = (
        (lexical, None, None, None),
        (None, vector, None, None),
        (lexical, vector, rrf, None),
        (lexical, vector, rrf, reranker),
    )
    actual_specs = tuple(
        (config.lexical, config.vector, config.rrf, config.reranker) for config in configs
    )
    if actual_specs != expected_specs:
        raise PersistenceValidationError(
            "evaluation configs do not match the canonical weighted lexical, vector, server RRF, "
            "and pinned local-reranker suite"
        )
