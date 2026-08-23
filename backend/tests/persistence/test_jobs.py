import asyncio

import pytest
from pufferlab.contracts.evals import EvalRunStatus
from pufferlab.jobs import QueryWorkItem, RunJobManager
from pufferlab.persistence import PufferLabRepository

from .helpers import SampleGraph, make_outcome, persist_graph


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

    async def execute(item: QueryWorkItem):  # type: ignore[no-untyped-def]
        return make_outcome(run, item.config_id, item.query_id, value=len(observed_progress) + 1)

    async def observe(snapshot):  # type: ignore[no-untyped-def]
        outcomes = repository.list_outcomes(run.id)
        persisted = repository.get_run(run.id)
        assert persisted.completed_queries == snapshot.completed_queries
        assert outcomes
        observed_progress.append(snapshot.completed_queries)

    manager = RunJobManager(repository)
    completed = await manager.start(
        run.id,
        work,
        execute,
        max_concurrency=1,
        on_progress=observe,
    )

    assert completed.status is EvalRunStatus.COMPLETED
    assert completed.completed_queries == 2
    assert observed_progress == [0, 1, 1, 2]
    assert len(repository.list_outcomes(run.id)) == 4


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

    async def execute(item: QueryWorkItem):  # type: ignore[no-untyped-def]
        started.append(item)
        first_started.set()
        await release_first.wait()
        return make_outcome(run, item.config_id, item.query_id)

    manager = RunJobManager(repository)
    manager.start(run.id, work, execute, max_concurrency=1)
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
