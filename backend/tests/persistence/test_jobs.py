import asyncio
from collections.abc import Sequence

import pytest
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import ConfigRunSummary, EvalRun, EvalRunStatus
from pufferlab.jobs import QueryWorkItem, RunJobManager
from pufferlab.persistence import PufferLabRepository, QueryOutcome
from pufferlab.persistence.errors import PersistenceValidationError

from .helpers import SampleGraph, make_outcome, persist_graph, summarize_outcomes


@pytest.mark.asyncio
async def test_job_persists_outcomes_before_publishing_progress(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("successful-job", max_concurrency=1)
    repository.create_run(run)
    work = [
        QueryWorkItem(config.id, query.id)
        for query in sample_graph.queries
        for config in sample_graph.configs
    ]
    observed_progress: list[int] = []

    async def execute(item: QueryWorkItem) -> QueryOutcome:
        return make_outcome(run, item.config_id, item.query_id, value=len(observed_progress) + 1)

    async def observe(snapshot: EvalRun) -> None:
        outcomes = repository.list_outcomes(run.id)
        persisted = repository.get_run(run.id)
        assert persisted.completed_queries == snapshot.completed_queries
        assert outcomes
        observed_progress.append(snapshot.completed_queries)

    async def finalize(
        snapshot: EvalRun,
        outcomes: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        assert snapshot.status is EvalRunStatus.RUNNING
        assert list(outcomes) == repository.list_outcomes(run.id)
        return summarize_outcomes(snapshot, outcomes)

    manager = RunJobManager(repository)
    completed = await manager.start(
        run.id,
        work,
        execute,
        max_concurrency=1,
        finalize=finalize,
        on_progress=observe,
    )

    assert completed.status is EvalRunStatus.COMPLETED
    assert completed.completed_queries == 2
    assert completed.summaries == summarize_outcomes(run, repository.list_outcomes(run.id))
    assert observed_progress == [0, 1, 1, 2]
    assert len(repository.list_outcomes(run.id)) == 4


@pytest.mark.asyncio
async def test_claim_rejection_schedules_zero_executor_work(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("claim-rejected", max_concurrency=1)
    repository.create_run(run)
    work = [QueryWorkItem(sample_graph.configs[0].id, sample_graph.queries[0].id)]
    executed: list[QueryWorkItem] = []

    async def execute(item: QueryWorkItem) -> QueryOutcome:
        executed.append(item)
        return make_outcome(run, item.config_id, item.query_id)

    async def must_not_finalize(
        _: EvalRun,
        __: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        raise AssertionError("a rejected claim cannot finalize")

    def reject_claim(*_: object, **__: object) -> EvalRun:
        raise PersistenceValidationError("claim rejected")

    monkeypatch.setattr(repository, "claim_queued_run", reject_claim)
    manager = RunJobManager(repository)
    with pytest.raises(PersistenceValidationError, match="claim rejected"):
        manager.start(
            run.id,
            work,
            execute,
            max_concurrency=1,
            finalize=must_not_finalize,
        )
    await asyncio.sleep(0)

    assert executed == []
    assert repository.get_run(run.id).status is EvalRunStatus.QUEUED


@pytest.mark.asyncio
async def test_start_claimed_executes_an_already_running_job(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("claimed-job", max_concurrency=1)
    repository.create_run(run)
    claimed = repository.claim_queued_run(run.id)
    work = [
        QueryWorkItem(config.id, query.id)
        for query in sample_graph.queries
        for config in sample_graph.configs
    ]

    async def execute(item: QueryWorkItem) -> QueryOutcome:
        assert repository.get_run(run.id).status is EvalRunStatus.RUNNING
        return make_outcome(run, item.config_id, item.query_id)

    async def finalize(
        snapshot: EvalRun,
        outcomes: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        return summarize_outcomes(snapshot, outcomes)

    manager = RunJobManager(repository)
    completed = await manager.start_claimed(
        claimed.id,
        work,
        execute,
        max_concurrency=1,
        finalize=finalize,
    )

    assert completed.status is EvalRunStatus.COMPLETED
    assert len(repository.list_outcomes(run.id)) == 4


@pytest.mark.asyncio
async def test_duplicate_claimed_local_job_is_rejected_and_cancellation_is_idempotent(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("duplicate-claimed-job", max_concurrency=1)
    repository.create_run(run)
    repository.claim_queued_run(run.id)
    item = QueryWorkItem(sample_graph.configs[0].id, sample_graph.queries[0].id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(work_item: QueryWorkItem) -> QueryOutcome:
        started.set()
        await release.wait()
        return make_outcome(run, work_item.config_id, work_item.query_id)

    async def must_not_finalize(
        _: EvalRun,
        __: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        raise AssertionError("cancelled work cannot finalize")

    manager = RunJobManager(repository)
    manager.start_claimed(
        run.id,
        [item],
        execute,
        max_concurrency=1,
        finalize=must_not_finalize,
    )
    with pytest.raises(PersistenceValidationError, match="active local job"):
        manager.start_claimed(
            run.id,
            [item],
            execute,
            max_concurrency=1,
            finalize=must_not_finalize,
        )

    await started.wait()
    cancellation = asyncio.create_task(manager.cancel(run.id))
    await asyncio.sleep(0)
    release.set()
    cancelled = await cancellation
    cancelled_again = await manager.cancel(run.id)

    assert cancelled.status is EvalRunStatus.CANCELLED
    assert cancelled_again == cancelled
    assert repository.list_outcomes(run.id) == [make_outcome(run, item.config_id, item.query_id)]


@pytest.mark.asyncio
async def test_cancellation_stops_new_scheduling_and_keeps_started_outcome(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("cancel-job", max_concurrency=1)
    repository.create_run(run)
    work = [
        QueryWorkItem(config.id, query.id)
        for query in sample_graph.queries
        for config in sample_graph.configs
    ]
    started: list[QueryWorkItem] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def execute(item: QueryWorkItem) -> QueryOutcome:
        started.append(item)
        first_started.set()
        await release_first.wait()
        return make_outcome(run, item.config_id, item.query_id)

    async def must_not_finalize(
        _: EvalRun,
        __: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        raise AssertionError("cancelled jobs must not finalize summaries")

    manager = RunJobManager(repository)
    manager.start(
        run.id,
        work,
        execute,
        max_concurrency=1,
        finalize=must_not_finalize,
    )
    await first_started.wait()
    cancellation = asyncio.create_task(manager.cancel(run.id))
    await asyncio.sleep(0)

    assert started == [work[0]]
    release_first.set()
    cancelled = await cancellation

    assert cancelled.status is EvalRunStatus.CANCELLED
    assert started == [work[0]]
    assert repository.list_outcomes(run.id) == [
        make_outcome(run, work[0].config_id, work[0].query_id)
    ]


@pytest.mark.asyncio
async def test_simultaneous_failure_drains_every_successful_started_peer(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)

    async def must_not_finalize(
        _: EvalRun,
        __: Sequence[QueryOutcome],
    ) -> Sequence[ConfigRunSummary]:
        raise AssertionError("failed jobs must not finalize summaries")

    async def exercise_iteration(iteration: int) -> None:
        run = sample_graph.make_run(f"simultaneous-peer-{iteration}")
        repository.create_run(run)
        successful = QueryWorkItem(sample_graph.configs[0].id, sample_graph.queries[0].id)
        failing = QueryWorkItem(sample_graph.configs[1].id, sample_graph.queries[0].id)
        work = [successful, failing] if iteration % 2 == 0 else [failing, successful]
        release = asyncio.Event()
        started: list[QueryWorkItem] = []

        async def execute(item: QueryWorkItem) -> QueryOutcome:
            started.append(item)
            if len(started) == 2:
                release.set()
            await release.wait()
            if item == failing:
                raise RuntimeError("synthetic simultaneous failure")
            return make_outcome(run, item.config_id, item.query_id, value=iteration)

        manager = RunJobManager(repository)
        with pytest.raises(RuntimeError, match="synthetic simultaneous failure"):
            await manager.start(
                run.id,
                work,
                execute,
                max_concurrency=2,
                finalize=must_not_finalize,
            )

        assert started == work
        assert repository.list_outcomes(run.id) == [
            make_outcome(run, successful.config_id, successful.query_id, value=iteration)
        ]
        failed = repository.get_run(run.id)
        assert failed.status is EvalRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code is ApiErrorCode.INTERNAL_ERROR
        assert failed.error.message == "evaluation execution failed"
        assert failed.error.retryable is False
        assert failed.error.details == {}
        assert "synthetic simultaneous failure" not in failed.model_dump_json()

    for iteration in range(40):
        await exercise_iteration(iteration)
