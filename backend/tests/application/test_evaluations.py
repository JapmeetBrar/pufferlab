import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pufferlab.application import EvaluationApplicationService, EvaluationRunError
from pufferlab.contracts.common import ObservedScore, ScoreDirection, ScoreKind, ScoreSource
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import (
    CreateEvalRunRequest,
    EvalRun,
    EvalRunExport,
    EvalRunStatus,
    JudgedQuery,
    MetricName,
    Qrel,
    QuerySet,
    RunEnvironment,
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
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareRequest,
    SearchCompareResponse,
    SearchHit,
    StageMembership,
    StageTiming,
    TimingStage,
)
from pufferlab.datasets.unix_application import (
    CuratedJudgedQuerySeed,
    UnixEvaluationSeed,
)
from pufferlab.jobs import (
    RunJobManager,
    decode_outcome_payload,
    finalize_durable_outcomes,
)
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.errors import PersistenceValidationError
from pufferlab.providers.rerankers import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION
from pufferlab.retrieval.errors import provider_failed
from pufferlab.retrieval.types import (
    SearchExecuteRequest,
    SearchExecuteResult,
)

_TEST_NAMESPACE = UUID("274cd26a-8291-4da8-ac21-991f4a68b431")
_FIXED_TIME = datetime(2026, 8, 22, 17, 0, tzinfo=UTC)
_METRIC_ORDER = [
    MetricName.NDCG_AT_10,
    MetricName.RECALL_AT_50,
    MetricName.MRR_AT_10,
    MetricName.LATENCY_P50_MS,
    MetricName.LATENCY_P95_MS,
    MetricName.ERROR_RATE,
]


def _id(name: str) -> UUID:
    return uuid5(_TEST_NAMESPACE, name)


def _documents(query_id: UUID) -> list[UUID]:
    return [_id(f"{query_id}:document:{rank}") for rank in range(1, 51)]


class FakeSearchBackend:
    def __init__(self, configs: tuple[RetrievalConfig, ...]) -> None:
        self._configs = {config.id: config for config in configs}
        self.calls: list[SearchExecuteRequest] = []
        self.active = 0
        self.max_active = 0
        self.operational_failure: tuple[UUID, UUID] | None = None
        self.fatal_after: int | None = None
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return tuple(self._summary(config) for config in self._configs.values())

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise NotImplementedError

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        self.calls.append(request)
        call_number = len(self.calls)
        if self.fatal_after == call_number:
            raise RuntimeError("private provider response must not escape")
        if request.query_id is not None and self.operational_failure == (
            request.config_id,
            request.query_id,
        ):
            raise provider_failed("fake_search")

        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            else:
                await asyncio.sleep(0.001)
            assert request.query_id is not None
            config = self._configs[request.config_id]
            hits = [
                SearchHit(
                    document_id=document_id,
                    external_id=f"result-{rank}",
                    title="not persisted",
                    body_excerpt="not persisted",
                    final_rank=rank,
                    stage_membership=[
                        StageMembership(
                            stage=RetrievalStage.FINAL,
                            rank=rank,
                            score=ObservedScore(
                                kind=ScoreKind.RRF,
                                value=1.0 / rank,
                                direction=ScoreDirection.HIGHER_IS_BETTER,
                                source=ScoreSource.CLIENT_COMPUTED,
                            ),
                        )
                    ],
                )
                for rank, document_id in enumerate(_documents(request.query_id), start=1)
            ]
            result = ConfigSearchResult(
                config=self._summary(config),
                hits=hits,
                timings=[StageTiming(stage=TimingStage.TURBOPUFFER, duration_ms=2.5)],
                candidate_counts={RetrievalStage.FINAL.value: len(hits)},
                warnings=[],
                trace_id=_id(f"trace:{request.config_id}:{request.query_id}"),
            )
            return SearchExecuteResult(
                config_id=request.config_id,
                query_id=request.query_id,
                result=result,
            )
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed = True

    @staticmethod
    def _summary(config: RetrievalConfig) -> RetrievalConfigSummary:
        return RetrievalConfigSummary(
            id=config.id,
            revision=config.revision,
            name=config.name,
            mode=config.mode,
            config_hash=config.config_hash,
        )


@pytest.fixture
def repository(tmp_path: Path) -> Iterator[PufferLabRepository]:
    database = Database(tmp_path / "application.sqlite3")
    database.migrate()
    try:
        yield PufferLabRepository(database.session_factory)
    finally:
        database.dispose()


def _seed() -> tuple[UnixEvaluationSeed, tuple[RetrievalConfig, ...]]:
    dataset_id = _id("dataset")
    dataset = DatasetVersion(
        id=dataset_id,
        slug="synthetic-unix",
        version="v1",
        namespace="pufferlab-test-eval",
        index_profile=IndexProfile(
            id="test-profile",
            embedding_provider="sentence_transformers",
            embedding_model="test-embedding",
            embedding_revision="revision",
            vector_dimensions=3,
            vector_dtype="f16",
            distance_metric="cosine_distance",
            fts_profile=FtsProfile(),
            schema_hash="schema-hash",
        ),
        document_count=100,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=_FIXED_TIME,
    )
    curated: list[CuratedJudgedQuerySeed] = []
    for index in range(50):
        query_id = _id(f"query:{index:02d}")
        documents = _documents(query_id)
        query = JudgedQuery(
            id=query_id,
            external_id=f"query-{index:02d}",
            text=f"synthetic query {index:02d}",
            tags=["hybrid"],
            qrels=[
                Qrel(document_id=documents[10], relevance_grade=1),
                Qrel(document_id=documents[49], relevance_grade=2),
            ],
        )
        curated.append(
            CuratedJudgedQuerySeed(
                judged_query=query,
                primary_tag="hybrid",
                tags=("hybrid",),
                reason="synthetic test reason",
            )
        )
    query_set = QuerySet(
        id=_id("query-set"),
        name="synthetic curated 50",
        version="v1",
        dataset_version_id=dataset_id,
        query_count=50,
        content_hash="query-set-hash",
        created_at=_FIXED_TIME,
    )
    configs = (
        RetrievalConfig(
            id=_id("config:bm25"),
            revision=1,
            name="BM25",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.BM25,
            result_k=50,
            candidate_k=100,
            lexical=LexicalSpec(),
            config_hash="bm25-hash",
            created_at=_FIXED_TIME,
        ),
        RetrievalConfig(
            id=_id("config:vector"),
            revision=1,
            name="Vector",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.VECTOR,
            result_k=50,
            candidate_k=100,
            vector=VectorSpec(attribute="vector", embedding_model="test-embedding"),
            config_hash="vector-hash",
            created_at=_FIXED_TIME,
        ),
        RetrievalConfig(
            id=_id("config:rrf"),
            revision=1,
            name="RRF",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.HYBRID_RRF,
            result_k=50,
            candidate_k=100,
            lexical=LexicalSpec(),
            vector=VectorSpec(attribute="vector", embedding_model="test-embedding"),
            rrf=RrfSpec(),
            config_hash="rrf-hash",
            created_at=_FIXED_TIME,
        ),
        RetrievalConfig(
            id=_id("config:rerank"),
            revision=1,
            name="Rerank",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.HYBRID_RERANK,
            result_k=50,
            candidate_k=100,
            lexical=LexicalSpec(),
            vector=VectorSpec(attribute="vector", embedding_model="test-embedding"),
            rrf=RrfSpec(),
            reranker=RerankerSpec(
                provider="sentence_transformers",
                model=DEFAULT_RERANKER_MODEL,
                revision=DEFAULT_RERANKER_REVISION,
                depth=50,
            ),
            config_hash="rerank-hash",
            created_at=_FIXED_TIME,
        ),
    )
    return (
        UnixEvaluationSeed(
            dataset_version=dataset,
            query_set=query_set,
            curated_queries=tuple(curated),
        ),
        configs,
    )


def _request(
    seed: UnixEvaluationSeed,
    configs: tuple[RetrievalConfig, ...],
    **updates: int,
) -> CreateEvalRunRequest:
    values = {
        "query_set_id": seed.query_set.id,
        "baseline_config_id": configs[0].id,
        "candidate_config_ids": [config.id for config in configs[1:]],
        "random_seed": 20260822,
        "max_concurrency": 3,
        "warmup_query_count": 2,
    }
    values.update(updates)
    return CreateEvalRunRequest.model_validate(values)


def _environment(request: CreateEvalRunRequest) -> RunEnvironment:
    return RunEnvironment(
        pufferlab_git_revision="test-revision",
        turbopuffer_region="gcp-us-west1",
        python_version="3.12",
        platform="test",
        max_concurrency=request.max_concurrency,
        warmup_query_count=request.warmup_query_count,
        query_embedding_cache_enabled=True,
    )


def _service(
    repository: PufferLabRepository,
    backend: FakeSearchBackend,
) -> EvaluationApplicationService:
    return EvaluationApplicationService(
        repository=repository,
        job_manager=RunJobManager(repository),
        search_backend=backend,
        now=lambda: _FIXED_TIME,
    )


@pytest.mark.asyncio
async def test_run_persists_200_outcomes_after_unmeasured_warmups_and_exports_typed_state(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    service = _service(repository, backend)
    service.seed(seed, configs)
    request = _request(seed, configs)
    progress: list[tuple[int, int]] = []

    async def observe(snapshot: EvalRun) -> None:
        durable = repository.get_run(snapshot.id)
        outcomes = repository.list_outcomes(snapshot.id)
        assert durable.completed_queries == snapshot.completed_queries
        assert outcomes
        progress.append((snapshot.completed_queries, len(outcomes)))

    completed = await service.run(request, _environment(request), on_progress=observe)

    outcomes = repository.list_outcomes(completed.id)
    assert completed.status is EvalRunStatus.COMPLETED
    assert completed.completed_queries == 50
    assert len(outcomes) == 200
    assert len(backend.calls) == 208
    assert backend.max_active == request.max_concurrency
    assert progress[-1] == (50, 200)
    assert all(persisted_count >= completed_count for completed_count, persisted_count in progress)
    assert [summary.config_id for summary in completed.summaries] == [
        request.baseline_config_id,
        *request.candidate_config_ids,
    ]
    for summary in completed.summaries:
        assert summary.completed_queries == 50
        assert summary.failed_queries == 0
        assert [metric.name for metric in summary.metrics] == _METRIC_ORDER
        assert [metric.sample_count for metric in summary.metrics] == [50] * 6
    query_ids = [query.id for query in seed.judged_queries]
    assert (
        finalize_durable_outcomes(completed, list(reversed(outcomes)), query_ids=query_ids)
        == completed.summaries
    )
    with pytest.raises(ValueError, match="exact config/query outcome coverage"):
        finalize_durable_outcomes(completed, outcomes[:-1], query_ids=query_ids)

    ranked = decode_outcome_payload(outcomes[0])
    assert ranked.kind == "success"
    assert len(ranked.ranked_document_ids) == 50
    assert ranked.metrics.recall_at_50 == 1.0
    assert ranked.metrics.ndcg_at_10 == 0.0
    assert ranked.metrics.mrr_at_10 == 0.0

    exported = service.export(completed.id)
    restored = EvalRunExport.model_validate_json(exported.model_dump_json())
    assert restored == exported
    assert [(item.config_id, item.query_id) for item in exported.outcomes] == sorted(
        ((item.config_id, item.query_id) for item in exported.outcomes),
        key=lambda identity: (str(identity[0]), str(identity[1])),
    )
    _assert_no_exposed_fields(exported.model_dump(mode="json"))

    before = exported.model_dump_json()
    with pytest.raises(PersistenceValidationError, match="only a queued run"):
        service.start_run(completed.id)
    assert service.export(completed.id).model_dump_json() == before


@pytest.mark.asyncio
async def test_operational_failure_is_durable_and_excluded_from_quality_means(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    backend.operational_failure = (configs[2].id, seed.judged_queries[7].id)
    service = _service(repository, backend)
    service.seed(seed, configs)
    request = _request(seed, configs, warmup_query_count=0)

    completed = await service.run(request, _environment(request))

    summary = next(item for item in completed.summaries if item.config_id == configs[2].id)
    assert completed.status is EvalRunStatus.COMPLETED
    assert summary.completed_queries == 49
    assert summary.failed_queries == 1
    metrics = {metric.name: metric for metric in summary.metrics}
    assert metrics[MetricName.NDCG_AT_10].sample_count == 49
    assert metrics[MetricName.RECALL_AT_50].sample_count == 49
    assert metrics[MetricName.MRR_AT_10].sample_count == 49
    assert metrics[MetricName.ERROR_RATE].value == pytest.approx(1 / 50)
    assert metrics[MetricName.ERROR_RATE].sample_count == 50
    assert (
        sum(outcome.outcome.kind == "failure" for outcome in service.export(completed.id).outcomes)
        == 1
    )


@pytest.mark.asyncio
async def test_systemic_failure_is_redacted_and_preserves_prior_durable_outcomes(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    backend.fatal_after = 5
    service = _service(repository, backend)
    service.seed(seed, configs)
    request = _request(seed, configs, warmup_query_count=0, max_concurrency=1)
    run = service.create_run(request, _environment(request), run_id=_id("fatal-run"))
    service.start_run(run.id)

    with pytest.raises(EvaluationRunError) as raised:
        await service.drain(run.id)

    assert "private provider response" not in str(raised.value)
    assert repository.get_run(run.id).status is EvalRunStatus.FAILED
    assert len(repository.list_outcomes(run.id)) == 4
    assert len(service.export(run.id).outcomes) == 4


@pytest.mark.asyncio
async def test_cancellation_uses_manager_and_drains_started_outcomes(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    backend.block = True
    service = _service(repository, backend)
    service.seed(seed, configs)
    request = _request(seed, configs, warmup_query_count=0, max_concurrency=2)
    run = service.create_run(request, _environment(request), run_id=_id("cancel-run"))
    service.start_run(run.id)
    await backend.started.wait()

    cancellation = asyncio.create_task(service.cancel(run.id))
    await asyncio.sleep(0)
    backend.release.set()
    cancelled = await cancellation

    assert cancelled.status is EvalRunStatus.CANCELLED
    assert len(backend.calls) == 2
    assert len(repository.list_outcomes(run.id)) == 2
    assert len(service.export(run.id).outcomes) == 2


@pytest.mark.asyncio
async def test_close_cooperatively_cancels_a_blocked_warmup_without_outcomes(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    backend.block = True
    service = _service(repository, backend)
    service.seed(seed, configs)
    request = _request(seed, configs, warmup_query_count=1, max_concurrency=2)
    run = service.create_run(request, _environment(request), run_id=_id("warmup-cancel-run"))
    service.start_run(run.id)
    await backend.started.wait()

    closing = asyncio.create_task(service.close())
    for _ in range(20):
        await asyncio.sleep(0)
        if repository.get_run(run.id).status is EvalRunStatus.CANCELLED:
            break
    assert repository.get_run(run.id).status is EvalRunStatus.CANCELLED
    assert len(backend.calls) == 1
    backend.release.set()
    await closing

    assert backend.closed is True
    assert repository.get_run(run.id).status is EvalRunStatus.CANCELLED
    assert repository.list_outcomes(run.id) == []


def test_seed_rejects_noncanonical_candidate_depth_and_reranker_depth(
    repository: PufferLabRepository,
) -> None:
    seed, configs = _seed()
    backend = FakeSearchBackend(configs)
    service = _service(repository, backend)
    wrong_candidate_depth = configs[0].model_copy(update={"candidate_k": 99})
    assert configs[3].reranker is not None
    wrong_reranker_depth = configs[3].model_copy(
        update={"reranker": configs[3].reranker.model_copy(update={"depth": 51})}
    )

    with pytest.raises(PersistenceValidationError, match="candidate_k=100"):
        service.seed(seed, (wrong_candidate_depth, *configs[1:]))
    with pytest.raises(PersistenceValidationError, match="pinned local-reranker suite"):
        service.seed(seed, (*configs[:3], wrong_reranker_depth))

    repository.put_dataset_version(seed.dataset_version)
    for config in (wrong_candidate_depth, *configs[1:]):
        repository.put_retrieval_config(config)
    repository.put_query_set(seed.query_set, seed.judged_queries)
    request = _request(seed, configs, warmup_query_count=0)
    with pytest.raises(PersistenceValidationError, match="candidate_k=100"):
        service.create_run(request, _environment(request))


def _assert_no_exposed_fields(value: object) -> None:
    forbidden = {
        "api_key",
        "body",
        "body_excerpt",
        "credentials",
        "document_text",
        "headers",
        "query_text",
        "raw_response",
        "request_body",
        "title",
        "vector",
        "vectors",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for nested in value.values():
            _assert_no_exposed_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_exposed_fields(nested)
