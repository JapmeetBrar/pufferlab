from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid5

from fastapi.testclient import TestClient
from pufferlab.application.view_errors import (
    EvaluationViewError,
    evaluation_conflict,
    evaluation_not_found,
    evaluation_unavailable,
)
from pufferlab.contracts.catalog import (
    DatasetCatalogItem,
    DatasetDetailResponse,
    DatasetListResponse,
    QuerySetCatalogItem,
    QuerySetListResponse,
    RetrievalConfigCatalogResponse,
)
from pufferlab.contracts.datasets import (
    DataOrigin,
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    CandidateRelevantRankChanges,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    DatasetAttribution,
    EvalRun,
    EvalRunDetailResponse,
    EvalRunExport,
    EvalRunExportResponse,
    EvalRunListQuery,
    EvalRunListResponse,
    EvalRunQueryDetailResponse,
    EvalRunStatus,
    EvalRunView,
    ExcludedPairCount,
    JudgedQuery,
    Qrel,
    QuerySet,
    QuerySetSummary,
    RegressionCoverage,
    RegressionPairStatus,
    RegressionQuery,
    RegressionResponse,
    RunEnvironment,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
)
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.main import create_app
from pufferlab.retrieval.types import SearchExecuteRequest, SearchExecuteResult

_NAMESPACE = UUID("f14476f1-a5bc-4565-83d2-d712220332d9")
_NOW = datetime(2026, 8, 23, 17, 0, tzinfo=UTC)


def _id(name: str) -> UUID:
    return uuid5(_NAMESPACE, name)


class _SearchBackend:
    def __init__(self) -> None:
        self.closed = False
        self.summary = RetrievalConfigSummary(
            id=_id("playground-config"),
            revision=1,
            name="Playground fixture",
            mode=RetrievalMode.BM25,
            config_hash="playground-hash",
        )

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return (self.summary,)

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        raise AssertionError(f"unexpected compare request: {request.query_text}")

    async def search_one(self, request: SearchExecuteRequest) -> SearchExecuteResult:
        raise AssertionError(f"unexpected single search request: {request.query_text}")

    async def close(self) -> None:
        self.closed = True


class _EvaluationViews:
    def __init__(self) -> None:
        self.dataset = DatasetVersion(
            id=_id("dataset"),
            slug="unix",
            version="v1",
            namespace="pufferlab-route-test",
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
            document_count=50,
            corpus_hash="corpus-hash",
            status=DatasetStatus.READY,
            created_at=_NOW,
        )
        self.query_set = QuerySet(
            id=_id("query-set"),
            name="Unix queries",
            version="v1",
            dataset_version_id=self.dataset.id,
            query_count=50,
            content_hash="query-set-hash",
            created_at=_NOW,
        )
        self.configs = [
            RetrievalConfigSummary(
                id=_id(f"eval-config-{mode.value}"),
                revision=1,
                name=mode.value,
                mode=mode,
                config_hash=f"{mode.value}-hash",
            )
            for mode in (
                RetrievalMode.BM25,
                RetrievalMode.VECTOR,
                RetrievalMode.HYBRID_RRF,
                RetrievalMode.HYBRID_RERANK,
            )
        ]
        run = EvalRun(
            id=_id("run"),
            status=EvalRunStatus.QUEUED,
            query_set=QuerySetSummary(
                id=self.query_set.id,
                name=self.query_set.name,
                version=self.query_set.version,
                query_count=50,
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
            created_at=_NOW,
            started_at=None,
            completed_at=None,
            error=None,
        )
        self.view = EvalRunView(
            run=run,
            dataset_version_id=self.dataset.id,
            data_origin=DataOrigin.LIVE,
            configs=self.configs,
            completed_attempts=0,
            original_stage_evidence_available=False,
            live_replay_policy_permitted=True,
        )
        self.query = JudgedQuery(
            id=_id("query"),
            external_id="unix-1",
            text="Safe route test query",
            qrels=[Qrel(document_id=_id("document"), relevance_grade=2)],
        )
        self.calls: list[tuple[str, object]] = []
        self.failure: Exception | None = None

    def _called(self, operation: str, value: object = None) -> None:
        self.calls.append((operation, value))
        if self.failure is not None:
            raise self.failure

    def list_datasets(self) -> DatasetListResponse:
        self._called("list_datasets")
        return DatasetListResponse(
            datasets=[DatasetCatalogItem(dataset=self.dataset, data_origin=DataOrigin.LIVE)]
        )

    def get_dataset(self, dataset_version_id: UUID) -> DatasetDetailResponse:
        self._called("get_dataset", dataset_version_id)
        return DatasetDetailResponse(dataset=self.dataset, data_origin=DataOrigin.LIVE)

    def list_query_sets(self, dataset_version_id: UUID) -> QuerySetListResponse:
        self._called("list_query_sets", dataset_version_id)
        return QuerySetListResponse(
            dataset_version_id=self.dataset.id,
            query_sets=[QuerySetCatalogItem(query_set=self.query_set, data_origin=DataOrigin.LIVE)],
        )

    def list_dataset_configs(
        self,
        dataset_version_id: UUID,
    ) -> RetrievalConfigCatalogResponse:
        self._called("list_dataset_configs", dataset_version_id)
        return RetrievalConfigCatalogResponse(
            dataset_version_id=self.dataset.id,
            data_origin=DataOrigin.LIVE,
            configs=self.configs,
        )

    def list_eval_runs(self, query: EvalRunListQuery) -> EvalRunListResponse:
        self._called("list_eval_runs", query)
        return EvalRunListResponse(runs=[self.view])

    def get_eval_run(self, run_id: UUID) -> EvalRunDetailResponse:
        self._called("get_eval_run", run_id)
        return EvalRunDetailResponse(result=self.view)

    def get_regressions(self, run_id: UUID, query: RegressionQuery) -> RegressionResponse:
        self._called("get_regressions", (run_id, query))
        return RegressionResponse(
            run_id=run_id,
            data_origin=DataOrigin.LIVE,
            baseline_config_id=self.configs[0].id,
            candidate_config_id=query.candidate_config_id,
            order=query.order,
            limit=query.limit,
            rows=[],
            coverage=RegressionCoverage(
                paired_queries=0,
                excluded=[
                    ExcludedPairCount(
                        status=status,
                        count=50 if status is RegressionPairStatus.BASELINE_MISSING else 0,
                    )
                    for status in (
                        RegressionPairStatus.BASELINE_MISSING,
                        RegressionPairStatus.CANDIDATE_MISSING,
                        RegressionPairStatus.BASELINE_FAILED,
                        RegressionPairStatus.CANDIDATE_FAILED,
                        RegressionPairStatus.BOTH_FAILED,
                        RegressionPairStatus.NO_POSITIVE_QRELS,
                    )
                ],
            ),
        )

    def get_query_detail(self, run_id: UUID, query_id: UUID) -> EvalRunQueryDetailResponse:
        self._called("get_query_detail", (run_id, query_id))
        return EvalRunQueryDetailResponse(
            run_id=run_id,
            data_origin=DataOrigin.LIVE,
            query=self.query,
            baseline_config_id=self.configs[0].id,
            candidate_config_ids=[config.id for config in self.configs[1:]],
            configs=self.configs,
            outcomes=[],
            rank_changes=[
                CandidateRelevantRankChanges(candidate_config_id=config.id, changes=[])
                for config in self.configs[1:]
            ],
            attribution=DatasetAttribution(source_name="Route test"),
            original_stage_evidence_available=False,
            live_replay_policy_permitted=True,
        )

    def export_eval_run(self, run_id: UUID) -> EvalRunExportResponse:
        self._called("export_eval_run", run_id)
        return EvalRunExportResponse(
            data_origin=DataOrigin.LIVE,
            export=EvalRunExport(run=self.view.run, outcomes=[]),
        )

    def query_set_data_origin(self, query_set_id: UUID) -> DataOrigin:
        self._called("query_set_data_origin", query_set_id)
        return DataOrigin.LIVE


class _EvaluationControls:
    def __init__(self, views: _EvaluationViews) -> None:
        self.views = views
        self.calls: list[tuple[str, object]] = []

    async def create_eval_run(self, request: CreateEvalRunRequest) -> CreateEvalRunResponse:
        self.calls.append(("create_eval_run", request))
        return CreateEvalRunResponse(result=self.views.view)

    async def cancel_eval_run(self, run_id: UUID) -> CancelEvalRunResponse:
        self.calls.append(("cancel_eval_run", run_id))
        return CancelEvalRunResponse(result=self.views.view)

    async def replay_eval_query(
        self,
        run_id: UUID,
        query_id: UUID,
        request: EvalRunQueryReplayRequest,
    ) -> EvalRunQueryReplayResponse:
        self.calls.append(("replay_eval_query", (run_id, query_id, request)))
        raise EvaluationViewError(
            code=ApiErrorCode.INTERNAL_ERROR,
            message="evaluation control runtime is not available",
            http_status=503,
            operation="replay_eval_query",
        )


def _create_body(views: _EvaluationViews) -> dict[str, object]:
    return {
        "contract_version": 1,
        "query_set_id": str(views.query_set.id),
        "baseline_config_id": str(views.configs[0].id),
        "candidate_config_ids": [str(config.id) for config in views.configs[1:]],
        "random_seed": 20260822,
        "max_concurrency": 4,
        "warmup_query_count": 5,
    }


def test_evaluation_catalog_and_read_routes_delegate_to_injected_facades() -> None:
    search = _SearchBackend()
    views = _EvaluationViews()
    controls = _EvaluationControls(views)
    with TestClient(
        create_app(
            search_backend=search,
            evaluation_views=views,
            evaluation_controls=controls,
        )
    ) as client:
        assert client.get("/api/v1/datasets").status_code == 200
        assert client.get(f"/api/v1/datasets/{views.dataset.id}").status_code == 200
        assert (
            client.get(f"/api/v1/query-sets?dataset_version_id={views.dataset.id}").status_code
            == 200
        )
        scoped = client.get(f"/api/v1/datasets/{views.dataset.id}/configs")
        unscoped = client.get("/api/v1/configs")
        listed = client.get("/api/v1/eval-runs?limit=7")
        detail = client.get(f"/api/v1/eval-runs/{views.view.run.id}")
        regression = client.get(
            f"/api/v1/eval-runs/{views.view.run.id}/regressions",
            params={"candidate_config_id": str(views.configs[1].id)},
        )
        query = client.get(f"/api/v1/eval-runs/{views.view.run.id}/queries/{views.query.id}")
        exported = client.get(f"/api/v1/eval-runs/{views.view.run.id}/export")
        created = client.post("/api/v1/eval-runs", json=_create_body(views))
        cancelled = client.post(f"/api/v1/eval-runs/{views.view.run.id}/cancel")
        replayed = client.post(
            f"/api/v1/eval-runs/{views.view.run.id}/queries/{views.query.id}/replay",
            json={"config_ids": [str(views.configs[0].id), str(views.configs[1].id)]},
        )

    assert scoped.status_code == unscoped.status_code == 200
    assert [item["mode"] for item in scoped.json()["configs"]] == [
        "bm25",
        "vector",
        "hybrid_rrf",
        "hybrid_rerank",
    ]
    assert unscoped.json()["configs"][0]["id"] == str(search.summary.id)
    assert listed.status_code == detail.status_code == regression.status_code == 200
    assert query.status_code == exported.status_code == 200
    assert created.status_code == 202
    assert cancelled.status_code == 200
    assert replayed.status_code == 503
    assert replayed.json()["details"] == {"operation": "replay_eval_query"}
    assert ("list_eval_runs", EvalRunListQuery(limit=7)) in views.calls
    assert [name for name, _value in controls.calls] == [
        "create_eval_run",
        "cancel_eval_run",
        "replay_eval_query",
    ]
    assert search.closed


def test_evaluation_errors_are_direct_redacted_and_validation_precedes_facade() -> None:
    views = _EvaluationViews()
    controls = _EvaluationControls(views)
    try:
        raise RuntimeError("sqlite-provider-secret")
    except RuntimeError:
        views.failure = evaluation_not_found(
            message="evaluation run was not found",
            operation="get_eval_run",
        )
    app = create_app(
        search_backend=_SearchBackend(),
        evaluation_views=views,
        evaluation_controls=controls,
    )
    client = TestClient(app, raise_server_exceptions=False)

    missing = client.get(f"/api/v1/eval-runs/{_id('missing')}")
    views.failure = evaluation_conflict(
        message="dataset does not have one canonical four-config evaluation catalog",
        operation="list_dataset_configs",
    )
    conflict = client.get(f"/api/v1/datasets/{views.dataset.id}/configs")
    views.failure = evaluation_unavailable(operation="export_eval_run")
    unavailable = client.get(f"/api/v1/eval-runs/{views.view.run.id}/export")
    calls_before_invalid = list(views.calls)
    invalid = client.get("/api/v1/eval-runs/not-a-uuid")
    invalid_limit = client.get("/api/v1/eval-runs?limit=101")

    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    assert missing.json()["message"] == "evaluation run was not found"
    assert missing.json()["details"] == {"operation": "get_eval_run"}
    UUID(missing.json()["trace_id"])
    assert "detail" not in missing.json()
    assert "sqlite-provider-secret" not in missing.text
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "run_conflict"
    assert conflict.json()["details"] == {"operation": "list_dataset_configs"}
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "internal_error"
    assert unavailable.json()["message"] == "stored evaluation data is temporarily unavailable"
    assert unavailable.json()["details"] == {"operation": "export_eval_run"}
    assert "sqlite-provider-secret" not in unavailable.text
    assert invalid.status_code == 422
    assert invalid.json()["message"] == "request validation failed"
    assert invalid_limit.status_code == 422
    assert invalid_limit.json()["message"] == "request validation failed"
    assert views.calls == calls_before_invalid


def test_openapi_freezes_all_evaluation_routes_and_contract_schemas() -> None:
    schema = create_app(
        search_backend=_SearchBackend(),
        evaluation_views=_EvaluationViews(),
    ).openapi()
    expected_paths = {
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_version_id}",
        "/api/v1/query-sets",
        "/api/v1/datasets/{dataset_version_id}/configs",
        "/api/v1/eval-runs",
        "/api/v1/eval-runs/{run_id}",
        "/api/v1/eval-runs/{run_id}/cancel",
        "/api/v1/eval-runs/{run_id}/regressions",
        "/api/v1/eval-runs/{run_id}/queries/{query_id}",
        "/api/v1/eval-runs/{run_id}/export",
        "/api/v1/eval-runs/{run_id}/queries/{query_id}/replay",
    }

    assert expected_paths <= schema["paths"].keys()
    for component in (
        "DatasetListResponse",
        "RetrievalConfigCatalogResponse",
        "CreateEvalRunRequest",
        "EvalRunListResponse",
        "EvalRunDetailResponse",
        "RegressionResponse",
        "EvalRunQueryDetailResponse",
        "EvalRunExportResponse",
        "EvalRunQueryReplayRequest",
        "EvalRunQueryReplayResponse",
    ):
        assert component in schema["components"]["schemas"]
