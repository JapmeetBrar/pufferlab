"""Recoverable in-process scheduling over the durable run repository."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from pufferlab.contracts.evals import ConfigRunSummary, EvalRun, EvalRunStatus
from pufferlab.persistence.errors import PersistenceValidationError
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.persistence.types import QueryOutcome


@dataclass(frozen=True, slots=True)
class QueryWorkItem:
    config_id: UUID
    query_id: UUID


type OutcomeExecutor = Callable[[QueryWorkItem], Awaitable[QueryOutcome]]
type ProgressCallback = Callable[[EvalRun], Awaitable[None]]
type SummaryFinalizer = Callable[
    [EvalRun, Sequence[QueryOutcome]], Awaitable[Sequence[ConfigRunSummary]]
]


@dataclass(slots=True)
class _ActiveJob:
    cancellation_requested: asyncio.Event
    task: asyncio.Task[EvalRun]


class RunJobManager:
    """Run bounded work locally while making SQLite the source of truth."""

    def __init__(self, repository: PufferLabRepository) -> None:
        self._repository = repository
        self._active: dict[UUID, _ActiveJob] = {}

    def recover_startup(self) -> list[UUID]:
        """Mark process-owned running rows interrupted after a restart."""
        return self._repository.interrupt_stale_runs()

    def start(
        self,
        run_id: UUID,
        work_items: Sequence[QueryWorkItem],
        executor: OutcomeExecutor,
        *,
        max_concurrency: int,
        finalize: SummaryFinalizer,
        on_progress: ProgressCallback | None = None,
    ) -> asyncio.Task[EvalRun]:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if run_id in self._active:
            raise PersistenceValidationError(f"run {run_id} already has an active local job")
        if len(set(work_items)) != len(work_items):
            raise PersistenceValidationError("work items must be unique by config and query")
        run = self._repository.get_run(run_id)
        if run.status is not EvalRunStatus.QUEUED:
            raise PersistenceValidationError("only a queued run may start a local job")
        if run.environment.max_concurrency != max_concurrency:
            raise PersistenceValidationError(
                "scheduler concurrency must match the persisted run environment"
            )

        cancellation_requested = asyncio.Event()
        task = asyncio.create_task(
            self._run(
                run_id,
                list(work_items),
                executor,
                max_concurrency=max_concurrency,
                cancellation_requested=cancellation_requested,
                finalize=finalize,
                on_progress=on_progress,
            ),
            name=f"pufferlab-eval-{run_id}",
        )
        self._active[run_id] = _ActiveJob(cancellation_requested, task)
        return task

    async def cancel(self, run_id: UUID) -> EvalRun:
        active = self._active.get(run_id)
        if active is not None:
            active.cancellation_requested.set()
            return await active.task

        run = self._repository.get_run(run_id)
        if run.status in {EvalRunStatus.QUEUED, EvalRunStatus.RUNNING}:
            return self._repository.transition_run(run_id, EvalRunStatus.CANCELLED)
        return run

    async def close(self) -> None:
        active = list(self._active.values())
        for job in active:
            job.cancellation_requested.set()
        if active:
            await asyncio.gather(*(job.task for job in active), return_exceptions=True)

    async def _run(
        self,
        run_id: UUID,
        work_items: list[QueryWorkItem],
        executor: OutcomeExecutor,
        *,
        max_concurrency: int,
        cancellation_requested: asyncio.Event,
        finalize: SummaryFinalizer,
        on_progress: ProgressCallback | None,
    ) -> EvalRun:
        inflight: dict[asyncio.Task[QueryOutcome], tuple[int, QueryWorkItem]] = {}
        next_index = 0
        first_failure: BaseException | None = None
        try:
            self._repository.transition_run(run_id, EvalRunStatus.RUNNING)
            while True:
                while (
                    first_failure is None
                    and not cancellation_requested.is_set()
                    and len(inflight) < max_concurrency
                    and next_index < len(work_items)
                ):
                    item_index = next_index
                    item = work_items[next_index]
                    next_index += 1
                    inflight[asyncio.create_task(_execute(executor, item))] = (item_index, item)

                if not inflight:
                    break

                done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda completed: inflight[completed][0]):
                    _, item = inflight.pop(task)
                    try:
                        outcome = task.result()
                        if outcome.run_id != run_id:
                            raise PersistenceValidationError(
                                "executor returned an outcome for another run"
                            )
                        if (outcome.config_id, outcome.query_id) != (
                            item.config_id,
                            item.query_id,
                        ):
                            raise PersistenceValidationError(
                                "executor outcome identity does not match its work item"
                            )
                        persisted_run = self._repository.record_outcome(outcome)
                        if on_progress is not None:
                            await on_progress(persisted_run)
                    except BaseException as error:
                        if first_failure is None:
                            first_failure = error

            if first_failure is not None:
                raise first_failure
            if cancellation_requested.is_set():
                return self._repository.transition_run(run_id, EvalRunStatus.CANCELLED)

            durable_run = self._repository.get_run(run_id)
            durable_outcomes = self._repository.list_outcomes(run_id)
            summaries = await finalize(durable_run, durable_outcomes)
            return self._repository.complete_run(run_id, summaries)
        except BaseException:
            for task in inflight:
                task.cancel()
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
            current = self._repository.get_run(run_id)
            if current.status is EvalRunStatus.RUNNING:
                self._repository.transition_run(run_id, EvalRunStatus.FAILED)
            raise
        finally:
            self._active.pop(run_id, None)


async def _execute(executor: OutcomeExecutor, item: QueryWorkItem) -> QueryOutcome:
    return await executor(item)
