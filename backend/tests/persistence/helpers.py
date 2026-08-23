from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import (
    EvalRun,
    EvalRunStatus,
    JudgedQuery,
    Qrel,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.contracts.retrieval import LexicalSpec, RetrievalConfig, RetrievalMode
from pufferlab.persistence import PufferLabRepository, QueryOutcome, QueryOutcomeStatus

TEST_NAMESPACE = UUID("1fe66f83-6eb6-44cf-a4a4-1e62cf682844")
FIXED_TIME = datetime(2026, 8, 22, 12, 34, 56, 789012, tzinfo=UTC)


def stable_uuid(name: str) -> UUID:
    return uuid5(TEST_NAMESPACE, name)


@dataclass(frozen=True)
class SampleGraph:
    dataset: DatasetVersion
    configs: tuple[RetrievalConfig, RetrievalConfig]
    query_set: QuerySet
    queries: tuple[JudgedQuery, JudgedQuery]

    def make_run(self, name: str = "run", *, max_concurrency: int = 2) -> EvalRun:
        return EvalRun(
            id=stable_uuid(name),
            status=EvalRunStatus.QUEUED,
            query_set=QuerySetSummary(
                id=self.query_set.id,
                name=self.query_set.name,
                version=self.query_set.version,
                query_count=self.query_set.query_count,
                content_hash=self.query_set.content_hash,
            ),
            baseline_config_id=self.configs[0].id,
            candidate_config_ids=[self.configs[1].id],
            summaries=[],
            completed_queries=0,
            total_queries=self.query_set.query_count,
            random_seed=20260822,
            environment=RunEnvironment(
                pufferlab_git_revision="test-revision",
                turbopuffer_region="gcp-us-west1",
                python_version="3.12",
                platform="test",
                max_concurrency=max_concurrency,
                query_embedding_cache_enabled=False,
            ),
            created_at=FIXED_TIME,
            started_at=None,
            completed_at=None,
            error=None,
        )


def make_sample_graph() -> SampleGraph:
    dataset_id = stable_uuid("dataset")
    dataset = DatasetVersion(
        id=dataset_id,
        slug="synthetic-unix",
        version="v1",
        namespace="pufferlab-test-owned",
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
        document_count=2,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=FIXED_TIME,
    )
    configs = tuple(
        RetrievalConfig(
            id=stable_uuid(f"config-{index}"),
            revision=index,
            name=f"bm25-{index}",
            dataset_version_id=dataset_id,
            mode=RetrievalMode.BM25,
            lexical=LexicalSpec(title_weight=float(index), body_weight=1.0),
            config_hash=f"config-hash-{index}",
            created_at=FIXED_TIME,
        )
        for index in (1, 2)
    )
    queries = (
        JudgedQuery(
            id=stable_uuid("query-1"),
            external_id="synthetic-query-1",
            text="Which process owns a port?",
            tags=["exact-token"],
            qrels=[Qrel(document_id=stable_uuid("document-1"), relevance_grade=2)],
        ),
        JudgedQuery(
            id=stable_uuid("query-2"),
            external_id="synthetic-query-2",
            text="Find disk usage",
            tags=["semantic"],
            qrels=[
                Qrel(document_id=stable_uuid("document-2"), relevance_grade=1),
                Qrel(document_id=stable_uuid("document-1"), relevance_grade=0),
            ],
        ),
    )
    query_set = QuerySet(
        id=stable_uuid("query-set"),
        name="synthetic queries",
        version="v1",
        dataset_version_id=dataset_id,
        query_count=len(queries),
        content_hash="query-set-hash",
        created_at=FIXED_TIME,
    )
    return SampleGraph(dataset, configs, query_set, queries)


def persist_graph(repository: PufferLabRepository, graph: SampleGraph) -> None:
    repository.put_dataset_version(graph.dataset)
    for config in graph.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(graph.query_set, graph.queries)


def make_outcome(
    run: EvalRun,
    config_id: UUID,
    query_id: UUID,
    *,
    value: int = 1,
) -> QueryOutcome:
    return QueryOutcome(
        run_id=run.id,
        config_id=config_id,
        query_id=query_id,
        status=QueryOutcomeStatus.SUCCEEDED,
        payload={"ranked_document_ids": [str(stable_uuid(f"result-{value}"))], "value": value},
        created_at=FIXED_TIME,
    )
