import json
from datetime import timedelta, timezone

import pytest
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import ConfigRunSummary, EvalRunStatus
from pufferlab.persistence import (
    Database,
    ImmutableRecordError,
    InvalidRunTransitionError,
    PersistenceValidationError,
    PufferLabRepository,
    QueryOutcomeStatus,
    RecordNotFoundError,
)
from pufferlab.persistence.canonical import canonical_json, canonical_utc
from pufferlab.persistence.models import DatasetVersionRow, EvalRunRow, QueryOutcomeRow
from sqlalchemy import select

from .helpers import (
    FIXED_TIME,
    SampleGraph,
    make_outcome,
    persist_graph,
    stable_uuid,
    summarize_outcomes,
)


def _safe_run_error(name: str = "run-error") -> ApiErrorDetail:
    return ApiErrorDetail(
        code=ApiErrorCode.INTERNAL_ERROR,
        message="evaluation execution failed safely",
        retryable=False,
        trace_id=stable_uuid(name),
        details={"operation": "execute_evaluation"},
    )


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

    # Same-instant offsets and exact replays are idempotent; changed payloads are not.
    pacific_time = FIXED_TIME.astimezone(timezone(-timedelta(hours=7)))
    repository.put_dataset_version(
        sample_graph.dataset.model_copy(update={"created_at": pacific_time})
    )
    for config in sample_graph.configs:
        repository.put_retrieval_config(config.model_copy(update={"created_at": pacific_time}))
    repository.put_query_set(
        sample_graph.query_set.model_copy(update={"created_at": pacific_time}),
        sample_graph.queries,
    )
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


def test_revision_selectors_are_deterministic_and_dataset_scoped(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)

    other_dataset = sample_graph.dataset.model_copy(
        update={
            "id": stable_uuid("other-dataset"),
            "namespace": "pufferlab-test-other",
        }
    )
    other_config = sample_graph.configs[0].model_copy(
        update={
            "id": stable_uuid("other-config"),
            "dataset_version_id": other_dataset.id,
            "name": "other-bm25",
            "config_hash": "other-config-hash",
        }
    )
    other_query_set = sample_graph.query_set.model_copy(
        update={
            "id": stable_uuid("other-query-set"),
            "dataset_version_id": other_dataset.id,
            "name": "other queries",
            "content_hash": "other-query-set-hash",
        }
    )
    repository.put_dataset_version(other_dataset)
    repository.put_retrieval_config(other_config)
    repository.put_query_set(other_query_set, sample_graph.queries)

    assert repository.list_dataset_versions() == sorted(
        [sample_graph.dataset, other_dataset],
        key=lambda value: (value.created_at, str(value.id)),
    )
    assert repository.list_retrieval_configs(dataset_version_id=other_dataset.id) == [other_config]
    assert repository.list_query_sets(dataset_version_id=other_dataset.id) == [other_query_set]
    assert {value.id for value in repository.list_retrieval_configs()} == {
        *(value.id for value in sample_graph.configs),
        other_config.id,
    }
    assert {value.id for value in repository.list_query_sets()} == {
        sample_graph.query_set.id,
        other_query_set.id,
    }


def test_read_selectors_are_bounded_ordered_and_run_scoped(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    first = sample_graph.make_run("read-first")
    second = sample_graph.make_run("read-second")
    repository.create_run(first)
    repository.create_run(second)

    assert repository.list_runs(limit=1) == [min((first, second), key=lambda run: str(run.id))]
    assert repository.list_run_configs(first.id) == list(sample_graph.configs)
    assert repository.list_query_ids(sample_graph.query_set.id) == [
        query.id for query in sample_graph.queries
    ]
    assert (
        repository.get_judged_query(
            sample_graph.query_set.id,
            sample_graph.queries[0].id,
        )
        == sample_graph.queries[0]
    )
    with pytest.raises(RecordNotFoundError, match="requested query set"):
        repository.get_judged_query(
            sample_graph.query_set.id,
            stable_uuid("foreign-query"),
        )
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_runs(limit=0)
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_runs(limit=101)
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_query_ids(sample_graph.query_set.id, limit=101)
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_dataset_versions(limit=True)


def test_active_run_selector_is_oldest_first_bounded_and_read_only(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    oldest = sample_graph.make_run("active-oldest").model_copy(
        update={"created_at": FIXED_TIME - timedelta(minutes=2)}
    )
    tied_queued = sample_graph.make_run("active-tied-queued").model_copy(
        update={"created_at": FIXED_TIME - timedelta(minutes=1)}
    )
    tied_running = sample_graph.make_run("active-tied-running").model_copy(
        update={"created_at": FIXED_TIME - timedelta(minutes=1)}
    )
    terminal = sample_graph.make_run("inactive-terminal").model_copy(
        update={"created_at": FIXED_TIME - timedelta(minutes=3)}
    )
    for run in (oldest, tied_queued, tied_running, terminal):
        repository.create_run(run)
    repository.claim_queued_run(tied_running.id, at=FIXED_TIME)
    repository.transition_run(terminal.id, EvalRunStatus.CANCELLED, at=FIXED_TIME)

    expected = [
        oldest,
        *sorted((tied_queued, repository.get_run(tied_running.id)), key=lambda run: str(run.id)),
    ]
    selected = repository.list_active_runs(limit=3)

    assert selected == expected
    assert [run.status for run in selected] == [run.status for run in expected]
    assert repository.get_run(terminal.id).status is EvalRunStatus.CANCELLED
    with pytest.raises(PersistenceValidationError, match="active run selection exceeds its bound"):
        repository.list_active_runs(limit=2)
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_active_runs(limit=0)
    with pytest.raises(PersistenceValidationError, match="between 1 and 100"):
        repository.list_active_runs(limit=101)


def test_active_run_selector_strictly_decodes_the_overflow_probe(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    runs = [
        sample_graph.make_run(f"active-overflow-{index}").model_copy(
            update={"created_at": FIXED_TIME + timedelta(minutes=index)}
        )
        for index in range(3)
    ]
    for run in runs:
        repository.create_run(run)
    with database.session_factory.begin() as session:
        row = session.get(EvalRunRow, str(runs[-1].id))
        assert row is not None
        row.status = EvalRunStatus.RUNNING.value
    with database.session_factory() as session:
        corrupted = session.get(EvalRunRow, str(runs[-1].id))
        assert corrupted is not None
        before = (corrupted.status, corrupted.started_at, corrupted.payload_json)

    with pytest.raises(PersistenceValidationError, match="indexed lifecycle state"):
        repository.list_active_runs(limit=2)

    with database.session_factory() as session:
        unchanged = session.get(EvalRunRow, str(runs[-1].id))
        assert unchanged is not None
        assert (unchanged.status, unchanged.started_at, unchanged.payload_json) == before


def test_strict_read_decoding_rejects_index_payload_divergence(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("corrupt-read")
    repository.create_run(run)
    with database.session_factory.begin() as session:
        row = session.get(EvalRunRow, str(run.id))
        assert row is not None
        row.status = EvalRunStatus.RUNNING.value

    with pytest.raises(PersistenceValidationError, match="indexed lifecycle state"):
        repository.get_run(run.id)


def test_revision_run_and_outcome_payload_times_are_canonical_utc(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    pacific_time = FIXED_TIME.astimezone(timezone(-timedelta(hours=7)))
    offset_dataset = sample_graph.dataset.model_copy(update={"created_at": pacific_time})
    repository.put_dataset_version(offset_dataset)
    for config in sample_graph.configs:
        repository.put_retrieval_config(config.model_copy(update={"created_at": pacific_time}))
    repository.put_query_set(
        sample_graph.query_set.model_copy(update={"created_at": pacific_time}),
        sample_graph.queries,
    )

    run = sample_graph.make_run("offset-time").model_copy(update={"created_at": pacific_time})
    repository.create_run(run)
    repository.create_run(run.model_copy(update={"created_at": FIXED_TIME}))
    repository.transition_run(run.id, EvalRunStatus.RUNNING, at=pacific_time)
    offset_outcome = make_outcome(
        run,
        sample_graph.configs[0].id,
        sample_graph.queries[0].id,
    ).model_copy(update={"created_at": pacific_time})
    repository.record_outcome(offset_outcome)
    repository.record_outcome(offset_outcome.model_copy(update={"created_at": FIXED_TIME}))

    expected = canonical_utc(FIXED_TIME, field_name="test")
    with database.session_factory() as session:
        dataset_row = session.get(DatasetVersionRow, str(sample_graph.dataset.id))
        run_row = session.get(EvalRunRow, str(run.id))
        outcome_row = session.get(
            QueryOutcomeRow,
            (str(run.id), str(offset_outcome.config_id), str(offset_outcome.query_id)),
        )
    assert dataset_row is not None
    assert run_row is not None
    assert outcome_row is not None
    assert dataset_row.created_at == json.loads(dataset_row.payload_json)["created_at"] == expected
    assert json.loads(run_row.payload_json)["created_at"] == expected
    assert json.loads(run_row.payload_json)["started_at"] == expected
    assert outcome_row.created_at == json.loads(outcome_row.payload_json)["created_at"] == expected

    with pytest.raises(PersistenceValidationError, match="timezone-aware"):
        repository.put_dataset_version(
            sample_graph.dataset.model_copy(update={"created_at": FIXED_TIME.replace(tzinfo=None)})
        )


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

    outcomes = repository.list_outcomes(run.id)
    completed = repository.complete_run(run.id, summarize_outcomes(run, outcomes))
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
        (EvalRunStatus.QUEUED, EvalRunStatus.INTERRUPTED, False),
        (EvalRunStatus.QUEUED, EvalRunStatus.COMPLETED, False),
        (EvalRunStatus.QUEUED, EvalRunStatus.FAILED, True),
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
        error = _safe_run_error(f"transition-{start}-{target}")
        assert (
            repository.transition_run(
                run.id,
                target,
                error=error if target is EvalRunStatus.FAILED else None,
            ).status
            is target
        )
    else:
        with pytest.raises(InvalidRunTransitionError):
            repository.transition_run(run.id, target)


def test_queued_and_running_failures_require_contract_valid_safe_errors(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    queued = sample_graph.make_run("queued-reconstruction-failure")
    running = sample_graph.make_run("running-execution-failure")
    repository.create_run(queued)
    repository.create_run(running)
    started = repository.claim_queued_run(running.id, at=FIXED_TIME + timedelta(seconds=1))

    for run_id in (queued.id, running.id):
        before = repository.get_run(run_id)
        with pytest.raises(PersistenceValidationError, match="safe run-level error"):
            repository.transition_run(run_id, EvalRunStatus.FAILED)
        assert repository.get_run(run_id) == before

    with pytest.raises(PersistenceValidationError, match="contract-valid safe run-level error"):
        repository.transition_run(
            queued.id,
            EvalRunStatus.FAILED,
            error=object(),  # type: ignore[arg-type]
        )
    assert repository.get_run(queued.id) == queued

    queued_error = _safe_run_error("queued-failure")
    queued_failed = repository.transition_run(
        queued.id,
        EvalRunStatus.FAILED,
        at=FIXED_TIME + timedelta(seconds=2),
        error=queued_error,
    )
    running_error = _safe_run_error("running-failure")
    running_failed = repository.transition_run(
        running.id,
        EvalRunStatus.FAILED,
        at=FIXED_TIME + timedelta(seconds=3),
        error=running_error,
    )

    assert queued_failed.started_at is None
    assert queued_failed.completed_at == FIXED_TIME + timedelta(seconds=2)
    assert queued_failed.error == queued_error
    assert running_failed.started_at == started.started_at
    assert running_failed.completed_at == FIXED_TIME + timedelta(seconds=3)
    assert running_failed.error == running_error
    assert repository.list_active_runs() == []


def test_claim_queued_run_is_exact_and_rejects_nonqueued_without_mutation(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    claimed_source = sample_graph.make_run("claim-source")
    untouched = sample_graph.make_run("claim-untouched")
    terminal = sample_graph.make_run("claim-terminal")
    stale = sample_graph.make_run("claim-stale")
    for run in (claimed_source, untouched, terminal, stale):
        repository.create_run(run)
    repository.transition_run(terminal.id, EvalRunStatus.CANCELLED, at=FIXED_TIME)
    repository.claim_queued_run(stale.id, at=FIXED_TIME)
    repository.transition_run(stale.id, EvalRunStatus.INTERRUPTED, at=FIXED_TIME)

    claim_time = FIXED_TIME + timedelta(seconds=1)
    claimed = repository.claim_queued_run(claimed_source.id, at=claim_time)

    assert claimed.status is EvalRunStatus.RUNNING
    assert claimed.started_at == claim_time
    assert claimed.completed_at is None
    assert claimed.error is None
    assert repository.get_run(claimed.id) == claimed
    assert repository.get_run(untouched.id) == untouched

    for run_id in (claimed.id, terminal.id, stale.id):
        before = repository.get_run(run_id)
        with pytest.raises(InvalidRunTransitionError, match="only queued runs may be claimed"):
            repository.claim_queued_run(run_id, at=claim_time + timedelta(seconds=1))
        assert repository.get_run(run_id) == before
    with pytest.raises(RecordNotFoundError, match="was not found"):
        repository.claim_queued_run(stable_uuid("missing-claim"))
    assert repository.get_run(untouched.id) == untouched


def test_claim_queued_run_rejects_invalid_index_payload_without_rewriting_it(
    database: Database,
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("invalid-claim-payload")
    repository.create_run(run)
    with database.session_factory.begin() as session:
        row = session.get(EvalRunRow, str(run.id))
        assert row is not None
        row.status = EvalRunStatus.RUNNING.value
    with database.session_factory() as session:
        corrupted = session.get(EvalRunRow, str(run.id))
        assert corrupted is not None
        before = (corrupted.status, corrupted.started_at, corrupted.payload_json)

    with pytest.raises(PersistenceValidationError, match="indexed lifecycle state"):
        repository.claim_queued_run(run.id, at=FIXED_TIME + timedelta(seconds=1))

    with database.session_factory() as session:
        unchanged = session.get(EvalRunRow, str(run.id))
        assert unchanged is not None
        assert (unchanged.status, unchanged.started_at, unchanged.payload_json) == before


def test_startup_recovery_interrupts_running_and_preserves_queued(
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

    assert interrupted == [running.id]
    queued_after_recovery = repository.get_run(queued.id)
    assert queued_after_recovery.status is EvalRunStatus.QUEUED
    assert queued_after_recovery.completed_at is None
    assert repository.get_run(running.id).status is EvalRunStatus.INTERRUPTED
    assert repository.list_outcomes(running.id) == [durable_outcome]
    assert repository.get_run(cancelled.id).status is EvalRunStatus.CANCELLED


def test_completion_requires_exact_summaries_matching_durable_outcomes(
    repository: PufferLabRepository,
    sample_graph: SampleGraph,
) -> None:
    persist_graph(repository, sample_graph)
    run = sample_graph.make_run("summary-validation")
    repository.create_run(run)
    repository.transition_run(run.id, EvalRunStatus.RUNNING)
    for query_index, query in enumerate(sample_graph.queries):
        for config_index, config in enumerate(sample_graph.configs):
            outcome = make_outcome(run, config.id, query.id)
            if query_index == config_index == 0:
                outcome = outcome.model_copy(update={"status": QueryOutcomeStatus.FAILED})
            repository.record_outcome(outcome)
    summaries = summarize_outcomes(run, repository.list_outcomes(run.id))

    with pytest.raises(InvalidRunTransitionError, match="require final config summaries"):
        repository.transition_run(run.id, EvalRunStatus.COMPLETED)
    with pytest.raises(PersistenceValidationError, match="exactly one ordered summary"):
        repository.complete_run(run.id, summaries[:1])
    with pytest.raises(PersistenceValidationError, match="duplicate"):
        repository.complete_run(run.id, [summaries[0], summaries[0]])
    foreign_summary = ConfigRunSummary(
        config_id=stable_uuid("foreign-config"),
        metrics=[],
        completed_queries=2,
        failed_queries=0,
    )
    with pytest.raises(PersistenceValidationError, match="exactly one ordered summary"):
        repository.complete_run(run.id, [summaries[0], foreign_summary])
    with pytest.raises(PersistenceValidationError, match="exactly one ordered summary"):
        repository.complete_run(run.id, list(reversed(summaries)))
    bad_count = summaries[0].model_copy(
        update={"completed_queries": summaries[0].completed_queries + 1}
    )
    with pytest.raises(PersistenceValidationError, match="counts do not match"):
        repository.complete_run(run.id, [bad_count, summaries[1]])

    completed = repository.complete_run(run.id, summaries)
    assert completed.status is EvalRunStatus.COMPLETED
    assert completed.summaries == summaries
    assert completed.summaries[0].failed_queries == 1
