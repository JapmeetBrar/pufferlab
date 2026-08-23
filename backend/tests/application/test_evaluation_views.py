from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid5

import pytest
from pufferlab.application.evaluation_controls import ProviderFreeEvaluationControls
from pufferlab.application.evaluation_views import EvaluationViewService
from pufferlab.application.view_errors import EvaluationViewError
from pufferlab.contracts.datasets import (
    DataOrigin,
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import (
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
from pufferlab.contracts.forensics import EvalRunQueryReplayRequest
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.jobs.eval_runner import encode_outcome_payload, finalize_durable_outcomes
from pufferlab.persistence import Database, PufferLabRepository, QueryOutcome, QueryOutcomeStatus
from pufferlab.persistence.models import QueryOutcomeRow

_NAMESPACE = UUID("6c8d76a2-495f-4c12-a2cc-9a632ddff602")
_NOW = datetime(2026, 8, 23, 16, 0, tzinfo=UTC)


def _id(name: str) -> UUID:
    return uuid5(_NAMESPACE, name)


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
    has_positive_qrels = any(qrel.relevance_grade > 0 for qrel in graph.queries[query_index].qrels)
    payload = EvalSuccessPayload(
        ranked_document_ids=_ranked_ids(query_index, config_index),
        metrics=PerQueryMetrics(
            ndcg_at_10=(0.6, 0.4, 0.8, 0.7)[config_index] if has_positive_qrels else None,
            recall_at_50=1.0 if has_positive_qrels else None,
            mrr_at_10=(0.5, 0.3, 1.0, 0.7)[config_index] if has_positive_qrels else None,
        ),
        total_client_wall_latency_ms=10.0 + config_index,
        stage_timings=[],
        candidate_counts={"retrieved": 50},
        warnings=(
            []
            if has_positive_qrels
            else [
                EvalOutcomeWarning(
                    code="no_positive_qrels",
                    message="quality metrics are undefined because there are no positive qrels",
                )
            ]
        ),
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

    repository.transition_run(runs[EvalRunStatus.RUNNING].id, EvalRunStatus.RUNNING)
    repository.transition_run(runs[EvalRunStatus.CANCELLED].id, EvalRunStatus.CANCELLED)
    repository.transition_run(runs[EvalRunStatus.INTERRUPTED].id, EvalRunStatus.RUNNING)
    repository.transition_run(runs[EvalRunStatus.INTERRUPTED].id, EvalRunStatus.INTERRUPTED)
    repository.transition_run(runs[EvalRunStatus.FAILED].id, EvalRunStatus.RUNNING)
    repository.transition_run(
        runs[EvalRunStatus.FAILED].id,
        EvalRunStatus.FAILED,
        error=ApiErrorDetail(
            code=ApiErrorCode.INTERNAL_ERROR,
            message="evaluation failed safely",
            retryable=False,
            trace_id=_id("failed-run-trace"),
            details={"operation": "evaluate_run"},
        ),
    )

    completed = runs[EvalRunStatus.COMPLETED]
    repository.transition_run(completed.id, EvalRunStatus.RUNNING)
    for query_index in range(50):
        for config_index in range(4):
            repository.record_outcome(_outcome(completed, graph, query_index, config_index))
    outcomes = repository.list_outcomes(completed.id, limit=200)
    summaries = finalize_durable_outcomes(
        repository.get_run(completed.id),
        outcomes,
        query_ids=[query.id for query in graph.queries],
    )
    repository.complete_run(completed.id, summaries)
    return {status: repository.get_run(run.id) for status, run in runs.items()}


def test_catalogs_and_all_six_run_states_are_provider_free(
    repository: PufferLabRepository,
    graph: CanonicalGraph,
) -> None:
    runs = _persist_six_status_runs(repository, graph)
    service = EvaluationViewService(repository)

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
    assert row.ndcg_delta == pytest.approx(-0.2)
    assert row.relevant_rank_changes[0].baseline_rank == 50
    assert row.relevant_rank_changes[0].candidate_rank == 11
    assert "Safe test query" not in row.playground_url

    detail = service.get_query_detail(completed.id, graph.queries[0].id)
    assert [record.config_id for record in detail.outcomes] == [
        config.id for config in graph.configs
    ]
    assert detail.rank_changes[0].changes[0].baseline_rank == 50
    assert detail.rank_changes[0].changes[0].candidate_rank == 11
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
    repository.transition_run(run.id, EvalRunStatus.RUNNING)

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
    with pytest.raises(EvaluationViewError) as live_create:
        await ProviderFreeEvaluationControls(_OriginViews(DataOrigin.LIVE)).create_eval_run(request)
    assert live_create.value.http_status == 503
