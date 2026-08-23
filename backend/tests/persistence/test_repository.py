import pytest
from pufferlab.contracts.evals import EvalRunStatus
from pufferlab.persistence import (
    Database,
    ImmutableRecordError,
    InvalidRunTransitionError,
    PersistenceValidationError,
    PufferLabRepository,
    RecordNotFoundError,
)
from pufferlab.persistence.canonical import canonical_json
from pufferlab.persistence.models import DatasetVersionRow
from sqlalchemy import select

from .helpers import SampleGraph, make_outcome, persist_graph


def test_immutable_revision_graph_round_trips_exactly(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)

    assert repository.get_dataset_version(sample_graph.dataset.id) == sample_graph.dataset
    assert repository.get_retrieval_config(sample_graph.configs[0].id) == sample_graph.configs[0]
    assert repository.get_query_set(sample_graph.query_set.id) == (
        sample_graph.query_set,
        list(sample_graph.queries),
    )
    with database.session_factory() as session:
        stored_json = session.scalar(
            select(DatasetVersionRow.payload_json).where(
                DatasetVersionRow.id == str(sample_graph.dataset.id)
            )
        )
    assert stored_json == canonical_json(sample_graph.dataset)

    # Exact replays are idempotent; a changed payload under the same identity is not.
    persist_graph(repository, sample_graph)
    with pytest.raises(ImmutableRecordError, match="immutable revision"):
        repository.put_dataset_version(
            sample_graph.dataset.model_copy(update={"corpus_hash": "different"})
        )
    with pytest.raises(ImmutableRecordError, match="immutable revision"):
        repository.put_retrieval_config(
            sample_graph.configs[0].model_copy(update={"config_hash": "different"})
        )
    changed_queries = list(sample_graph.queries)
    changed_queries[0] = changed_queries[0].model_copy(update={"text": "Different text"})
    with pytest.raises(ImmutableRecordError, match="immutable revision"):
        repository.put_query_set(sample_graph.query_set, changed_queries)


def test_query_set_validation_rolls_back_the_whole_revision(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    repository.put_dataset_version(sample_graph.dataset)
    invalid_query_set = sample_graph.query_set.model_copy(update={"query_count": 3})

    with pytest.raises(PersistenceValidationError, match="query_count"):
        repository.put_query_set(invalid_query_set, sample_graph.queries)

    with pytest.raises(RecordNotFoundError, match="was not found"):
        repository.get_query_set(invalid_query_set.id)


def test_incremental_outcome_survives_a_fresh_engine(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("restart-durability")
    repository.create_run(run)
    repository.transition_run(run.id, EvalRunStatus.RUNNING)
    outcome = make_outcome(run, sample_graph.configs[0].id, sample_graph.queries[0].id)
    repository.record_outcome(outcome)

    with Database(database.path) as reopened:
        fresh_repository = PufferLabRepository(reopened.session_factory)
        assert fresh_repository.get_outcome(run.id, outcome.config_id, outcome.query_id) == outcome
        assert fresh_repository.get_run(run.id).completed_queries == 0


def test_outcomes_drive_progress_and_completed_runs_are_immutable(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run()
    repository.create_run(run)
    repository.transition_run(run.id, EvalRunStatus.RUNNING)

    first_query = sample_graph.queries[0].id
    first = make_outcome(run, sample_graph.configs[0].id, first_query)
    assert repository.record_outcome(first).completed_queries == 0
    assert repository.get_outcome(run.id, first.config_id, first.query_id) == first

    second = make_outcome(run, sample_graph.configs[1].id, first_query, value=2)
    assert repository.record_outcome(second).completed_queries == 1
    for query in sample_graph.queries[1:]:
        for index, config in enumerate(sample_graph.configs, start=3):
            repository.record_outcome(make_outcome(run, config.id, query.id, value=index))

    completed = repository.transition_run(run.id, EvalRunStatus.COMPLETED)
    assert completed.completed_queries == completed.total_queries == 2
    assert completed.completed_at is not None
    assert len(repository.list_outcomes(run.id)) == 4

    with pytest.raises(ImmutableRecordError, match="outcomes require a running run"):
        repository.record_outcome(first.model_copy(update={"payload": {"value": 999}}))
    with pytest.raises(InvalidRunTransitionError, match="not allowed"):
        repository.transition_run(run.id, EvalRunStatus.FAILED)


@pytest.mark.parametrize(
    ("start", "target", "allowed"),
    [
        (EvalRunStatus.QUEUED, EvalRunStatus.RUNNING, True),
        (EvalRunStatus.QUEUED, EvalRunStatus.CANCELLED, True),
        (EvalRunStatus.QUEUED, EvalRunStatus.INTERRUPTED, True),
        (EvalRunStatus.QUEUED, EvalRunStatus.COMPLETED, False),
        (EvalRunStatus.QUEUED, EvalRunStatus.FAILED, False),
        (EvalRunStatus.RUNNING, EvalRunStatus.CANCELLED, True),
        (EvalRunStatus.RUNNING, EvalRunStatus.INTERRUPTED, True),
        (EvalRunStatus.RUNNING, EvalRunStatus.FAILED, True),
        (EvalRunStatus.RUNNING, EvalRunStatus.RUNNING, False),
    ],
)
def test_run_transition_graph(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
    start: EvalRunStatus,
    target: EvalRunStatus,
    allowed: bool,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run(f"transition-{start}-{target}")
    repository.create_run(run)
    if start is EvalRunStatus.RUNNING:
        repository.transition_run(run.id, EvalRunStatus.RUNNING)

    if allowed:
        assert repository.transition_run(run.id, target).status is target
    else:
        with pytest.raises(InvalidRunTransitionError):
            repository.transition_run(run.id, target)


def test_startup_recovery_interrupts_queued_and_running_only(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    queued = sample_graph.make_run("queued-stale")
    running = sample_graph.make_run("running-stale")
    cancelled = sample_graph.make_run("already-cancelled")
    for run in (queued, running, cancelled):
        repository.create_run(run)
    repository.transition_run(running.id, EvalRunStatus.RUNNING)
    durable_outcome = make_outcome(
        running,
        sample_graph.configs[0].id,
        sample_graph.queries[0].id,
    )
    repository.record_outcome(durable_outcome)
    repository.transition_run(cancelled.id, EvalRunStatus.CANCELLED)

    interrupted = repository.interrupt_stale_runs()

    assert set(interrupted) == {queued.id, running.id}
    assert repository.get_run(queued.id).status is EvalRunStatus.INTERRUPTED
    assert repository.get_run(running.id).status is EvalRunStatus.INTERRUPTED
    assert repository.list_outcomes(running.id) == [durable_outcome]
    assert repository.get_run(cancelled.id).status is EvalRunStatus.CANCELLED
