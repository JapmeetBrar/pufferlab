from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid5

import pytest
from fastapi.testclient import TestClient
from pufferlab.application.evaluation_controls import ProviderFreeEvaluationControls
from pufferlab.application.evaluation_views import EvaluationViewService
from pufferlab.application.view_errors import EvaluationViewError, evaluation_invalid
from pufferlab.contracts.datasets import (
    DataOrigin,
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    CreateEvalRunRequest,
    EvalFailurePayload,
    EvalOutcomeWarning,
    EvalRun,
    EvalRunDetailResponse,
    EvalRunListQuery,
    EvalRunStatus,
    EvalSuccessPayload,
    JudgedQuery,
    PerQueryMetrics,
    Qrel,
    QuerySet,
    QuerySetSummary,
    RegressionPairStatus,
    RegressionQuery,
    RunEnvironment,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    ExpectedDocumentDiagnosticRequest,
)
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import Judgment
from pufferlab.jobs.eval_runner import (
    decode_outcome_payload,
    encode_outcome_payload,
    finalize_durable_outcomes,
)
from pufferlab.main import create_app
from pufferlab.persistence import (
    Database,
    PersistenceValidationError,
    PufferLabRepository,
    QueryOutcome,
    QueryOutcomeStatus,
    RecordNotFoundError,
)
from pufferlab.persistence.canonical import canonical_json, canonical_utc
from pufferlab.persistence.models import EvalRunRow, QueryOutcomeRow
from pufferlab.retrieval.types import SearchExecuteRequest, SearchExecuteResult
from sqlalchemy import select

_NAMESPACE = UUID("6c8d76a2-495f-4c12-a2cc-9a632ddff602")
_NOW = datetime(2026, 8, 23, 16, 0, tzinfo=UTC)


def _id(name: str) -> UUID:
    return uuid5(_NAMESPACE, name)


class _NoopSearchBackend:
    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return ()

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise AssertionError(f"unexpected compare call for {request.query_text}")

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        raise AssertionError(f"unexpected search call for {request.query_text}")

    async def close(self) -> None:
        return None


def _rewrite_run(
    database: Database,
    run_id: UUID,
    *,
    updates: dict[str, object],
) -> EvalRun:
    with database.session_factory.begin() as session:
        row = session.get(EvalRunRow, str(run_id))
        assert row is not None
        current = EvalRun.model_validate_json(row.payload_json)
        rewritten = current.model_copy(update=updates)
        row.status = rewritten.status.value
        row.completed_queries = rewritten.completed_queries
        row.started_at = (
            canonical_utc(rewritten.started_at, field_name="test.started_at")
            if rewritten.started_at is not None
            else None
        )
        row.completed_at = (
            canonical_utc(rewritten.completed_at, field_name="test.completed_at")
            if rewritten.completed_at is not None
            else None
        )
        row.payload_json = canonical_json(rewritten)
    return rewritten


def _assert_detached_failure(
    loader: Callable[[], object],
    *,
    marker: str,
    http_status: int,
) -> EvaluationViewError:
    with pytest.raises(EvaluationViewError) as raised:
        loader()
    error = raised.value
    assert error.http_status == http_status
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert marker not in rendered
    assert marker not in repr(error)
    return error


@dataclass(frozen=True, slots=True)
class CanonicalGraph:
    dataset: DatasetVersion
    configs: tuple[RetrievalConfig, ...]
    query_set: QuerySet
    queries: tuple[JudgedQuery, ...]

    def make_run(self, name: str, *, offset: int = 0) -> EvalRun:
        return EvalRun(
            id=_id(f"run-{name}"),
            status=EvalRunStatus.QUEUED,
            query_set=QuerySetSummary(
                id=self.query_set.id,
                name=self.query_set.name,
                version=self.query_set.version,
                query_count=self.query_set.query_count,
                content_hash=self.query_set.content_hash,
            ),
            baseline_config_id=self.configs[0].id,
            candidate_config_ids=[config.id for config in self.configs[1:]],
            summaries=[],
            completed_queries=0,
            total_queries=50,
            random_seed=20260822,
            environment=RunEnvironment(
                pufferlab_git_revision="test-revision",
                turbopuffer_region="gcp-us-west1",
                python_version="3.12",
                platform="test",
                max_concurrency=4,
                warmup_query_count=5,
                query_embedding_cache_enabled=False,
            ),
            created_at=_NOW + timedelta(seconds=offset),
            started_at=None,
            completed_at=None,
            error=None,
        )


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    value = Database(tmp_path / "evaluation-views.sqlite3")
    value.migrate()
    try:
        yield value
    finally:
        value.dispose()


@pytest.fixture
def repository(database: Database) -> PufferLabRepository:
    return PufferLabRepository(database.session_factory)


@pytest.fixture
def graph(repository: PufferLabRepository) -> CanonicalGraph:
    value = _make_graph()
    repository.put_dataset_version(value.dataset)
    for config in value.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(value.query_set, value.queries)
    return value


def _make_graph(*, no_positive_index: int | None = None) -> CanonicalGraph:
    dataset_id = _id("dataset")
    dataset = DatasetVersion(
        id=dataset_id,
        slug="unix",
        version="v1",
        namespace="pufferlab-live-test",
        index_profile=IndexProfile(
            id="bge384-bm25v4",
            embedding_provider="sentence_transformers",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_revision="test-revision",
            vector_dimensions=384,
            vector_dtype="f16",
            distance_metric="cosine_distance",
            fts_profile=FtsProfile(),
            schema_hash="schema-hash",
        ),
        document_count=500,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=_NOW,
    )
    lexical = LexicalSpec(title_weight=2.0, body_weight=1.0)
    vector = VectorSpec(
        attribute="vector",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    rrf = RrfSpec()
    configs = (
        RetrievalConfig(
            id=_id("config-bm25"),
            revision=1,
            name="BM25",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.BM25,
            result_k=50,
            candidate_k=100,
            lexical=lexical,
            config_hash="bm25-hash",
            created_at=_NOW,
        ),
        RetrievalConfig(
            id=_id("config-vector"),
            revision=1,
            name="Vector",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.VECTOR,
            result_k=50,
            candidate_k=100,
            vector=vector,
            config_hash="vector-hash",
            created_at=_NOW,
        ),
        RetrievalConfig(
            id=_id("config-rrf"),
            revision=1,
            name="Hybrid RRF",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.HYBRID_RRF,
            result_k=50,
            candidate_k=100,
            lexical=lexical,
            vector=vector,
            rrf=rrf,
            config_hash="rrf-hash",
            created_at=_NOW,
        ),
        RetrievalConfig(
            id=_id("config-rerank"),
            revision=1,
            name="Hybrid rerank",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.HYBRID_RERANK,
            result_k=50,
            candidate_k=100,
            lexical=lexical,
            vector=vector,
            rrf=rrf,
            reranker=RerankerSpec(
                provider="sentence_transformers",
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                revision="test-revision",
                depth=50,
            ),
            config_hash="rerank-hash",
            created_at=_NOW,
        ),
    )
    queries = tuple(
        JudgedQuery(
            id=_id(f"query-{index:02d}"),
            external_id=f"unix-{index:02d}",
            text=f"Safe test query {index}",
            tags=["test"],
            qrels=(
                []
                if index == no_positive_index
                else [Qrel(document_id=_id(f"relevant-{index:02d}"), relevance_grade=2)]
            ),
        )
        for index in range(50)
    )
    query_set = QuerySet(
        id=_id("query-set"),
        name="Unix judged queries",
        version="v1",
        dataset_version_id=dataset_id,
        query_count=50,
        content_hash="query-set-hash",
        created_at=_NOW,
    )
    return CanonicalGraph(dataset, configs, query_set, queries)


def _ranked_ids(query_index: int, config_index: int) -> list[UUID]:
    relevant = _id(f"relevant-{query_index:02d}")
    relevant_rank = (50, 11, 1, 25)[config_index]
    fillers = [_id(f"filler-{query_index:02d}-{config_index}-{rank}") for rank in range(49)]
    return [*fillers[: relevant_rank - 1], relevant, *fillers[relevant_rank - 1 :]]


def _outcome(
    run: EvalRun,
    graph: CanonicalGraph,
    query_index: int,
    config_index: int,
) -> QueryOutcome:
    ranked_document_ids = _ranked_ids(query_index, config_index)
    evaluated = evaluate_ranking(
        ranked_document_ids,
        [
            Judgment(
                document_id=qrel.document_id,
                relevance_grade=qrel.relevance_grade,
            )
            for qrel in graph.queries[query_index].qrels
        ],
    )
    payload = EvalSuccessPayload(
        ranked_document_ids=ranked_document_ids,
        metrics=PerQueryMetrics(
            ndcg_at_10=evaluated.ndcg_at_10,
            recall_at_50=evaluated.recall_at_50,
            mrr_at_10=evaluated.mrr_at_10,
        ),
        total_client_wall_latency_ms=10.0 + config_index,
        stage_timings=[],
        candidate_counts={"retrieved": 50},
        warnings=[
            EvalOutcomeWarning(code=warning.code.value, message=warning.message)
            for warning in evaluated.warnings
        ],
        trace_id=_id(f"trace-{run.id}-{query_index}-{config_index}"),
    )
    return QueryOutcome(
        run_id=run.id,
        config_id=graph.configs[config_index].id,
        query_id=graph.queries[query_index].id,
        status=QueryOutcomeStatus.SUCCEEDED,
        payload=encode_outcome_payload(payload),
        created_at=run.created_at + timedelta(seconds=1),
    )


def _failure(
    run: EvalRun,
    graph: CanonicalGraph,
    query_index: int,
    config_index: int,
) -> QueryOutcome:
    payload = EvalFailurePayload(
        code=ApiErrorCode.NAMESPACE_NOT_READY,
        message="retrieval was unavailable",
        retryable=True,
        operation="search_one",
        trace_id=_id(f"failure-trace-{run.id}-{query_index}-{config_index}"),
        total_client_wall_latency_ms=5.0,
    )
    return QueryOutcome(
        run_id=run.id,
        config_id=graph.configs[config_index].id,
        query_id=graph.queries[query_index].id,
        status=QueryOutcomeStatus.FAILED,
        payload=encode_outcome_payload(payload),
        created_at=run.created_at + timedelta(seconds=1),
    )


def _persist_six_status_runs(
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> dict[EvalRunStatus, EvalRun]:
    runs = {
        status: graph.make_run(status.value, offset=index)
        for index, status in enumerate(EvalRunStatus)
    }
    for run in runs.values():
        repository.create_run(run)

    repository.transition_run(
        runs[EvalRunStatus.RUNNING].id,
        EvalRunStatus.RUNNING,
        at=runs[EvalRunStatus.RUNNING].created_at + timedelta(seconds=1),
    )
    repository.transition_run(
        runs[EvalRunStatus.CANCELLED].id,
        EvalRunStatus.CANCELLED,
        at=runs[EvalRunStatus.CANCELLED].created_at + timedelta(seconds=1),
    )
    repository.transition_run(
        runs[EvalRunStatus.INTERRUPTED].id,
        EvalRunStatus.RUNNING,
        at=runs[EvalRunStatus.INTERRUPTED].created_at + timedelta(seconds=1),
    )
    repository.transition_run(
        runs[EvalRunStatus.INTERRUPTED].id,
        EvalRunStatus.INTERRUPTED,
        at=runs[EvalRunStatus.INTERRUPTED].created_at + timedelta(seconds=2),
    )
    repository.transition_run(
        runs[EvalRunStatus.FAILED].id,
        EvalRunStatus.RUNNING,
        at=runs[EvalRunStatus.FAILED].created_at + timedelta(seconds=1),
    )
    repository.transition_run(
        runs[EvalRunStatus.FAILED].id,
        EvalRunStatus.FAILED,
        at=runs[EvalRunStatus.FAILED].created_at + timedelta(seconds=2),
        error=ApiErrorDetail(
            code=ApiErrorCode.INTERNAL_ERROR,
            message="evaluation failed safely",
            retryable=False,
            trace_id=_id("failed-run-trace"),
            details={"operation": "evaluate_run"},
        ),
    )

    completed = runs[EvalRunStatus.COMPLETED]
    repository.transition_run(
        completed.id,
        EvalRunStatus.RUNNING,
        at=completed.created_at + timedelta(seconds=1),
    )
    for query_index in range(50):
        for config_index in range(4):
            repository.record_outcome(_outcome(completed, graph, query_index, config_index))
    outcomes = repository.list_outcomes(completed.id, limit=200)
    summaries = finalize_durable_outcomes(
        repository.get_run(completed.id),
        outcomes,
        query_ids=[query.id for query in graph.queries],
    )
    repository.complete_run(
        completed.id,
        summaries,
        at=completed.created_at + timedelta(seconds=2),
    )
    return {status: repository.get_run(run.id) for status, run in runs.items()}


def test_catalogs_and_all_six_run_states_are_provider_free(
    repository: PufferLabRepository,
    graph: CanonicalGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _persist_six_status_runs(repository, graph)
    service = EvaluationViewService(repository)
    original_list_query_ids = repository.list_query_ids
    query_id_reads = 0

    def counted_list_query_ids(query_set_id: UUID, *, limit: int = 100) -> list[UUID]:
        nonlocal query_id_reads
        query_id_reads += 1
        return original_list_query_ids(query_set_id, limit=limit)

    monkeypatch.setattr(repository, "list_query_ids", counted_list_query_ids)

    assert service.list_datasets().datasets[0].dataset == graph.dataset
    assert service.get_dataset(graph.dataset.id).data_origin is DataOrigin.LIVE
    assert service.list_query_sets(graph.dataset.id).query_sets[0].query_set == graph.query_set
    assert [config.mode for config in service.list_dataset_configs(graph.dataset.id).configs] == [
        RetrievalMode.BM25,
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    ]

    listed = service.list_eval_runs(EvalRunListQuery(limit=6))
    assert query_id_reads == 1
    assert {item.run.status for item in listed.runs} == set(EvalRunStatus)
    for status, run in runs.items():
        detail = service.get_eval_run(run.id).result
        assert detail.run.status is status
        assert service.export_eval_run(run.id).export.run.status is status
        assert service.get_query_detail(run.id, graph.queries[0].id).run_id == run.id
        regression = service.get_regressions(
            run.id,
            RegressionQuery(candidate_config_id=graph.configs[1].id),
        )
        assert regression.coverage.total_queries == 50

    queued_regression = service.get_regressions(
        runs[EvalRunStatus.QUEUED].id,
        RegressionQuery(candidate_config_id=graph.configs[1].id),
    )
    assert queued_regression.coverage.paired_queries == 0
    assert queued_regression.coverage.excluded[0].status is RegressionPairStatus.BASELINE_MISSING
    assert queued_regression.coverage.excluded[0].count == 50


def test_regressions_use_durable_pairing_and_exact_qrels_through_rank_50(
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    completed = _persist_six_status_runs(repository, graph)[EvalRunStatus.COMPLETED]
    service = EvaluationViewService(repository)

    regression = service.get_regressions(
        completed.id,
        RegressionQuery(candidate_config_id=graph.configs[1].id, limit=50),
    )
    assert regression.coverage.paired_queries == 50
    assert sum(item.count for item in regression.coverage.excluded) == 0
    row = next(item for item in regression.rows if item.query_id == graph.queries[0].id)
    assert row.ndcg_delta == 0.0
    assert row.relevant_rank_changes[0].baseline_rank == 50
    assert row.relevant_rank_changes[0].candidate_rank == 11
    assert "Safe test query" not in row.playground_url

    judged_document_id = graph.queries[0].qrels[0].document_id
    repository.put_judged_document_titles(
        graph.query_set.id,
        {judged_document_id: "Readable judged document title"},
    )
    detail = service.get_query_detail(completed.id, graph.queries[0].id)
    assert [record.config_id for record in detail.outcomes] == [
        config.id for config in graph.configs
    ]
    assert detail.rank_changes[0].changes[0].baseline_rank == 50
    assert detail.rank_changes[0].changes[0].candidate_rank == 11
    assert detail.judged_documents[0].document_id == judged_document_id
    assert detail.judged_documents[0].title == "Readable judged document title"
    with pytest.raises(EvaluationViewError) as not_found:
        service.get_query_detail(completed.id, _id("foreign-query"))
    assert not_found.value.http_status == 404


def test_partial_regression_coverage_uses_all_frozen_exclusions_and_exact_qrels(
    repository: PufferLabRepository,
) -> None:
    graph = _make_graph(no_positive_index=5)
    repository.put_dataset_version(graph.dataset)
    for config in graph.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(graph.query_set, graph.queries)
    run = graph.make_run("partial-coverage")
    repository.create_run(run)
    repository.transition_run(
        run.id,
        EvalRunStatus.RUNNING,
        at=run.created_at + timedelta(seconds=1),
    )

    # q0 lacks baseline; q1 lacks candidate; q2/q3 fail one side; q4 fails both; q5 has
    # no positive qrels. The remaining exact query IDs are observed successful pairs.
    repository.record_outcome(_outcome(run, graph, 0, 1))
    repository.record_outcome(_outcome(run, graph, 1, 0))
    repository.record_outcome(_failure(run, graph, 2, 0))
    repository.record_outcome(_outcome(run, graph, 2, 1))
    repository.record_outcome(_outcome(run, graph, 3, 0))
    repository.record_outcome(_failure(run, graph, 3, 1))
    repository.record_outcome(_failure(run, graph, 4, 0))
    repository.record_outcome(_failure(run, graph, 4, 1))
    for query_index in range(5, 50):
        repository.record_outcome(_outcome(run, graph, query_index, 0))
        repository.record_outcome(_outcome(run, graph, query_index, 1))

    response = EvaluationViewService(repository).get_regressions(
        run.id,
        RegressionQuery(candidate_config_id=graph.configs[1].id, limit=50),
    )

    assert response.coverage.paired_queries == 44
    assert [item.status for item in response.coverage.excluded] == [
        RegressionPairStatus.BASELINE_MISSING,
        RegressionPairStatus.CANDIDATE_MISSING,
        RegressionPairStatus.BASELINE_FAILED,
        RegressionPairStatus.CANDIDATE_FAILED,
        RegressionPairStatus.BOTH_FAILED,
        RegressionPairStatus.NO_POSITIVE_QRELS,
    ]
    assert [item.count for item in response.coverage.excluded] == [1, 1, 1, 1, 1, 1]
    assert len(response.rows) == 44


def test_corrupt_durable_outcome_is_a_redacted_unavailable_error(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    run = graph.make_run("corrupt")
    repository.create_run(run)
    repository.transition_run(run.id, EvalRunStatus.RUNNING)
    outcome = _outcome(run, graph, 0, 0)
    repository.record_outcome(outcome)
    with database.session_factory.begin() as session:
        row = session.get(
            QueryOutcomeRow,
            (str(run.id), str(outcome.config_id), str(outcome.query_id)),
        )
        assert row is not None
        row.status = QueryOutcomeStatus.FAILED.value

    with pytest.raises(EvaluationViewError) as unavailable:
        EvaluationViewService(repository).export_eval_run(run.id)
    assert unavailable.value.http_status == 503
    assert str(unavailable.value) == "stored evaluation data is temporarily unavailable"


def test_completed_run_rewritten_queued_fails_real_http_projections(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    completed = _persist_six_status_runs(repository, graph)[EvalRunStatus.COMPLETED]
    rewritten = _rewrite_run(
        database,
        completed.id,
        updates={"status": EvalRunStatus.QUEUED},
    )
    assert rewritten.started_at is not None
    assert rewritten.completed_at is not None
    assert rewritten.completed_queries == 50
    assert len(rewritten.summaries) == 4
    assert len(repository.list_outcomes(rewritten.id)) == 200

    app = create_app(
        search_backend=_NoopSearchBackend(),
        evaluation_views=EvaluationViewService(repository),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        responses = (
            client.get("/api/v1/eval-runs?limit=6"),
            client.get(f"/api/v1/eval-runs/{rewritten.id}"),
            client.get(f"/api/v1/eval-runs/{rewritten.id}/export"),
        )

    for response in responses:
        assert response.status_code == 503
        assert response.json()["code"] == "internal_error"
        assert response.json()["message"] == ("stored evaluation data is temporarily unavailable")
        assert "detail" not in response.json()
        UUID(response.json()["trace_id"])


@pytest.mark.parametrize(
    "attack",
    [
        "start_before_creation",
        "completion_before_start",
        "nonterminal_summaries",
        "progress_mismatch",
    ],
)
def test_impossible_lifecycle_variants_fail_closed(
    attack: str,
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    run = graph.make_run(f"lifecycle-{attack}")
    repository.create_run(run)
    if attack == "start_before_creation":
        repository.transition_run(
            run.id,
            EvalRunStatus.RUNNING,
            at=run.created_at + timedelta(seconds=1),
        )
        _rewrite_run(
            database,
            run.id,
            updates={"started_at": run.created_at - timedelta(seconds=1)},
        )
    elif attack == "completion_before_start":
        repository.transition_run(
            run.id,
            EvalRunStatus.RUNNING,
            at=run.created_at + timedelta(seconds=2),
        )
        repository.transition_run(
            run.id,
            EvalRunStatus.INTERRUPTED,
            at=run.created_at + timedelta(seconds=3),
        )
        _rewrite_run(
            database,
            run.id,
            updates={"completed_at": run.created_at + timedelta(seconds=1)},
        )
    elif attack == "nonterminal_summaries":
        repository.transition_run(
            run.id,
            EvalRunStatus.RUNNING,
            at=run.created_at + timedelta(seconds=1),
        )
        _rewrite_run(
            database,
            run.id,
            updates={
                "summaries": [
                    ConfigRunSummary(
                        config_id=graph.configs[0].id,
                        metrics=[],
                        completed_queries=0,
                        failed_queries=0,
                    )
                ]
            },
        )
    else:
        repository.transition_run(
            run.id,
            EvalRunStatus.RUNNING,
            at=run.created_at + timedelta(seconds=1),
        )
        for config_index in range(4):
            repository.record_outcome(_outcome(run, graph, 0, config_index))
        assert repository.get_run(run.id).completed_queries == 1
        _rewrite_run(database, run.id, updates={"completed_queries": 0})

    error = _assert_detached_failure(
        lambda: EvaluationViewService(repository).get_eval_run(run.id),
        marker="lifecycle-marker-not-public",
        http_status=503,
    )
    assert str(error) == "stored evaluation data is temporarily unavailable"


def test_completed_summary_mismatch_fails_durable_recomputation(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    completed = _persist_six_status_runs(repository, graph)[EvalRunStatus.COMPLETED]
    first_summary = completed.summaries[0]
    first_metric = first_summary.metrics[0]
    corrupted_metric = first_metric.model_copy(
        update={"value": 0.123 if first_metric.value != 0.123 else 0.456}
    )
    corrupted_summary = first_summary.model_copy(
        update={"metrics": [corrupted_metric, *first_summary.metrics[1:]]}
    )
    _rewrite_run(
        database,
        completed.id,
        updates={"summaries": [corrupted_summary, *completed.summaries[1:]]},
    )

    error = _assert_detached_failure(
        lambda: EvaluationViewService(repository).get_eval_run(completed.id),
        marker="summary-marker-not-public",
        http_status=503,
    )
    assert str(error) == "stored evaluation data is temporarily unavailable"


def test_queued_recovery_failure_and_running_full_coverage_are_valid_states(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    recovery_failed = graph.make_run("queued-recovery-failed")
    repository.create_run(recovery_failed)
    _rewrite_run(
        database,
        recovery_failed.id,
        updates={
            "status": EvalRunStatus.FAILED,
            "completed_at": recovery_failed.created_at + timedelta(seconds=1),
            "error": ApiErrorDetail(
                code=ApiErrorCode.INTERNAL_ERROR,
                message="queued run could not be reconstructed",
                retryable=False,
                trace_id=_id("queued-recovery-error"),
                details={"operation": "recover_queued_run"},
            ),
        },
    )

    running = graph.make_run("running-full-coverage", offset=1)
    repository.create_run(running)
    repository.transition_run(
        running.id,
        EvalRunStatus.RUNNING,
        at=running.created_at + timedelta(seconds=1),
    )
    for query_index in range(50):
        for config_index in range(4):
            repository.record_outcome(_outcome(running, graph, query_index, config_index))

    service = EvaluationViewService(repository)
    failed_view = service.get_eval_run(recovery_failed.id).result
    running_view = service.get_eval_run(running.id).result
    assert failed_view.run.status is EvalRunStatus.FAILED
    assert failed_view.run.started_at is None
    assert failed_view.completed_attempts == 0
    assert running_view.run.status is EvalRunStatus.RUNNING
    assert running_view.run.completed_queries == 50
    assert running_view.completed_attempts == 200


def test_completed_foreign_query_identity_attack_fails_every_run_projection(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    completed = _persist_six_status_runs(repository, graph)[EvalRunStatus.COMPLETED]
    original_query_id = graph.queries[0].id
    foreign_query_id = _id("foreign-durable-query")
    with database.session_factory.begin() as session:
        rows = session.scalars(
            select(QueryOutcomeRow).where(
                QueryOutcomeRow.run_id == str(completed.id),
                QueryOutcomeRow.query_id == str(original_query_id),
            )
        ).all()
        assert len(rows) == 4
        for row in rows:
            durable = QueryOutcome.model_validate_json(row.payload_json).model_copy(
                update={"query_id": foreign_query_id}
            )
            row.query_id = str(foreign_query_id)
            row.payload_json = canonical_json(durable)

    service = EvaluationViewService(repository)
    loaders = (
        lambda: service.list_eval_runs(EvalRunListQuery(limit=6)),
        lambda: service.get_eval_run(completed.id),
        lambda: service.export_eval_run(completed.id),
    )
    for loader in loaders:
        with pytest.raises(EvaluationViewError) as unavailable:
            loader()
        assert unavailable.value.http_status == 503
        assert str(unavailable.value) == "stored evaluation data is temporarily unavailable"


def test_valid_shaped_metric_corruption_fails_qrel_bearing_views(
    database: Database,
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    completed = _persist_six_status_runs(repository, graph)[EvalRunStatus.COMPLETED]
    query_id = graph.queries[0].id
    with database.session_factory.begin() as session:
        row = session.get(
            QueryOutcomeRow,
            (str(completed.id), str(graph.configs[0].id), str(query_id)),
        )
        assert row is not None
        durable = QueryOutcome.model_validate_json(row.payload_json)
        payload = decode_outcome_payload(durable)
        assert isinstance(payload, EvalSuccessPayload)
        corrupted_payload = payload.model_copy(
            update={
                "metrics": PerQueryMetrics(
                    ndcg_at_10=1.0,
                    recall_at_50=1.0,
                    mrr_at_10=1.0,
                )
            }
        )
        row.payload_json = canonical_json(
            durable.model_copy(update={"payload": encode_outcome_payload(corrupted_payload)})
        )

    service = EvaluationViewService(repository)
    loaders = (
        lambda: service.get_regressions(
            completed.id,
            RegressionQuery(candidate_config_id=graph.configs[1].id, limit=50),
        ),
        lambda: service.get_query_detail(completed.id, query_id),
    )
    for loader in loaders:
        with pytest.raises(EvaluationViewError) as unavailable:
            loader()
        assert unavailable.value.http_status == 503
        assert str(unavailable.value) == "stored evaluation data is temporarily unavailable"


def test_safe_error_translation_detaches_all_internal_exception_frames(
    repository: PufferLabRepository,
    graph: CanonicalGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EvaluationViewService(repository)
    app = create_app(
        search_backend=_NoopSearchBackend(),
        evaluation_views=service,
    )

    def assert_http_error(
        response_status: int,
        response_text: str,
        marker: str,
        expected_status: int,
    ) -> None:
        assert response_status == expected_status
        assert marker not in response_text
        assert '"detail"' not in response_text

    with TestClient(app, raise_server_exceptions=False) as client:
        value_marker = "VALUE_ERROR_PRIVATE_MARKER"

        def fail_value(*_args: object, **_kwargs: object) -> None:
            raise ValueError(value_marker)

        with monkeypatch.context() as patch:
            patch.setattr(repository, "list_dataset_versions", fail_value)
            _assert_detached_failure(
                service.list_datasets,
                marker=value_marker,
                http_status=503,
            )
            response = client.get("/api/v1/datasets")
            assert_http_error(response.status_code, response.text, value_marker, 503)

        persistence_marker = "PERSISTENCE_PRIVATE_MARKER"

        def fail_persistence(*_args: object, **_kwargs: object) -> None:
            raise PersistenceValidationError(persistence_marker)

        with monkeypatch.context() as patch:
            patch.setattr(repository, "list_runs", fail_persistence)
            _assert_detached_failure(
                lambda: service.list_eval_runs(EvalRunListQuery(limit=1)),
                marker=persistence_marker,
                http_status=503,
            )
            response = client.get("/api/v1/eval-runs?limit=1")
            assert_http_error(response.status_code, response.text, persistence_marker, 503)

        missing_marker = "NOT_FOUND_PRIVATE_MARKER"

        def fail_not_found(*_args: object, **_kwargs: object) -> None:
            raise RecordNotFoundError(missing_marker)

        with monkeypatch.context() as patch:
            patch.setattr(repository, "get_dataset_version", fail_not_found)
            _assert_detached_failure(
                lambda: service.get_dataset(graph.dataset.id),
                marker=missing_marker,
                http_status=404,
            )
            response = client.get(f"/api/v1/datasets/{graph.dataset.id}")
            assert_http_error(response.status_code, response.text, missing_marker, 404)

        catalog_marker = "CONFIG_CATALOG_PRIVATE_MARKER"

        def fail_catalog(*_args: object, **_kwargs: object) -> None:
            raise ValueError(catalog_marker)

        with monkeypatch.context() as patch:
            patch.setattr(service, "_canonical_configs", fail_catalog)
            _assert_detached_failure(
                lambda: service.list_dataset_configs(graph.dataset.id),
                marker=catalog_marker,
                http_status=409,
            )
            response = client.get(f"/api/v1/datasets/{graph.dataset.id}/configs")
            assert_http_error(response.status_code, response.text, catalog_marker, 409)

        propagated_marker = "PROPAGATED_ERROR_PRIVATE_MARKER"

        def fail_with_context(*_args: object, **_kwargs: object) -> None:
            try:
                raise ValueError(propagated_marker)
            except ValueError:
                raise evaluation_invalid(
                    message="safe propagated validation failure",
                    operation="list_datasets",
                ) from None

        with monkeypatch.context() as patch:
            patch.setattr(repository, "list_dataset_versions", fail_with_context)
            error = _assert_detached_failure(
                service.list_datasets,
                marker=propagated_marker,
                http_status=422,
            )
            assert str(error) == "safe propagated validation failure"
            response = client.get("/api/v1/datasets")
            assert_http_error(response.status_code, response.text, propagated_marker, 422)


class _OriginViews:
    def __init__(self, origin: DataOrigin) -> None:
        self.origin = origin

    def query_set_data_origin(self, query_set_id: UUID) -> DataOrigin:
        del query_set_id
        return self.origin

    def get_eval_run(self, run_id: UUID) -> EvalRunDetailResponse:
        del run_id
        return cast(
            EvalRunDetailResponse,
            SimpleNamespace(result=SimpleNamespace(data_origin=self.origin)),
        )


@pytest.mark.asyncio
async def test_provider_free_controls_reject_synthetic_and_defer_live_cost_paths(
    graph: CanonicalGraph,
) -> None:
    request = CreateEvalRunRequest(
        query_set_id=graph.query_set.id,
        baseline_config_id=graph.configs[0].id,
        candidate_config_ids=[config.id for config in graph.configs[1:]],
    )
    replay = EvalRunQueryReplayRequest(config_ids=[graph.configs[0].id, graph.configs[1].id])
    diagnostic = ExpectedDocumentDiagnosticRequest(config_id=graph.configs[0].id)

    with pytest.raises(EvaluationViewError) as synthetic_create:
        synthetic_controls = ProviderFreeEvaluationControls(_OriginViews(DataOrigin.SYNTHETIC_DEMO))
        await synthetic_controls.create_eval_run(request)
    assert synthetic_create.value.http_status == 409
    with pytest.raises(EvaluationViewError) as synthetic_replay:
        await ProviderFreeEvaluationControls(
            _OriginViews(DataOrigin.SYNTHETIC_DEMO)
        ).replay_eval_query(_id("run"), graph.queries[0].id, replay)
    assert synthetic_replay.value.http_status == 409
    with pytest.raises(EvaluationViewError) as synthetic_cancel:
        await synthetic_controls.cancel_eval_run(_id("run"))
    assert synthetic_cancel.value.http_status == 409
    with pytest.raises(EvaluationViewError) as synthetic_diagnostic:
        await synthetic_controls.diagnose_expected_document(
            _id("run"),
            graph.queries[0].id,
            graph.queries[0].qrels[0].document_id,
            diagnostic,
        )
    assert synthetic_diagnostic.value.http_status == 409
    assert synthetic_diagnostic.value.operation == "diagnose_expected_document"
    with pytest.raises(EvaluationViewError) as live_create:
        live_controls = ProviderFreeEvaluationControls(_OriginViews(DataOrigin.LIVE))
        await live_controls.create_eval_run(request)
    assert live_create.value.http_status == 503
    with pytest.raises(EvaluationViewError) as live_diagnostic:
        await live_controls.diagnose_expected_document(
            _id("run"),
            graph.queries[0].id,
            graph.queries[0].qrels[0].document_id,
            diagnostic,
        )
    assert live_diagnostic.value.http_status == 503
    assert live_diagnostic.value.operation == "diagnose_expected_document"
