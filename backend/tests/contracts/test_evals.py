from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pufferlab.contracts.catalog import QuerySetCatalogItem, QuerySetListResponse
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    ConfigRunSummary,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    EvalOutcomeRecord,
    EvalRun,
    EvalRunExport,
    EvalRunExportResponse,
    EvalRunListResponse,
    EvalRunStatus,
    EvalRunView,
    EvalSuccessPayload,
    ExcludedPairCount,
    MetricAggregate,
    MetricName,
    PerQueryMetrics,
    QuerySet,
    QuerySetSummary,
    RegressionCoverage,
    RegressionOrder,
    RegressionPairStatus,
    RegressionResponse,
    RegressionRow,
    RunEnvironment,
    TimingSource,
)
from pydantic import ValidationError

_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
_CONFIG_IDS = [UUID(int=index) for index in range(1, 5)]


def test_eval_run_serializes_contract_version() -> None:
    eval_run = EvalRun(
        id=uuid4(),
        status=EvalRunStatus.QUEUED,
        query_set=QuerySetSummary(
            id=uuid4(),
            name="tiny queries",
            version="v1",
            query_count=1,
            content_hash="content-hash",
        ),
        baseline_config_id=uuid4(),
        candidate_config_ids=[uuid4()],
        summaries=[],
        completed_queries=0,
        total_queries=1,
        random_seed=20260822,
        environment=RunEnvironment(
            pufferlab_git_revision="test-revision",
            turbopuffer_region="gcp-us-central1",
            python_version="3.12",
            platform="test",
            max_concurrency=1,
            warmup_query_count=3,
            query_embedding_cache_enabled=False,
        ),
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=None,
        error=None,
    )

    payload = eval_run.model_dump(mode="json")
    assert payload["contract_version"] == 1
    assert payload["environment"]["warmup_query_count"] == 3


def _environment(*, synthetic: bool = False) -> RunEnvironment:
    return RunEnvironment(
        pufferlab_git_revision="test-revision",
        turbopuffer_region="gcp-us-west1" if not synthetic else "unavailable",
        python_version="3.12",
        platform="test",
        max_concurrency=4,
        warmup_query_count=5,
        timing_source=(
            TimingSource.SYNTHETIC_UNAVAILABLE if synthetic else TimingSource.PERF_COUNTER
        ),
        query_embedding_cache_enabled=False,
    )


def _summary(config_id: UUID, *, synthetic: bool) -> ConfigRunSummary:
    return ConfigRunSummary(
        config_id=config_id,
        metrics=[
            MetricAggregate(name=MetricName.NDCG_AT_10, value=0.5, sample_count=50),
            MetricAggregate(name=MetricName.RECALL_AT_50, value=0.75, sample_count=50),
            MetricAggregate(name=MetricName.MRR_AT_10, value=0.6, sample_count=50),
            MetricAggregate(
                name=MetricName.LATENCY_P50_MS,
                value=None if synthetic else 10.0,
                sample_count=0 if synthetic else 50,
            ),
            MetricAggregate(
                name=MetricName.LATENCY_P95_MS,
                value=None if synthetic else 20.0,
                sample_count=0 if synthetic else 50,
            ),
            MetricAggregate(name=MetricName.ERROR_RATE, value=0.0, sample_count=50),
        ],
        completed_queries=50,
        failed_queries=0,
    )


def _run(
    status: EvalRunStatus,
    *,
    synthetic: bool = False,
    run_id: UUID | None = None,
) -> EvalRun:
    terminal = status in {
        EvalRunStatus.COMPLETED,
        EvalRunStatus.FAILED,
        EvalRunStatus.CANCELLED,
        EvalRunStatus.INTERRUPTED,
    }
    completed = status is EvalRunStatus.COMPLETED
    error = (
        ApiErrorDetail(
            code=ApiErrorCode.INTERNAL_ERROR,
            message="queued run binding could not be reconstructed",
            retryable=False,
            trace_id=UUID(int=999),
        )
        if status is EvalRunStatus.FAILED
        else None
    )
    return EvalRun(
        id=run_id or UUID(int=100),
        status=status,
        query_set=QuerySetSummary(
            id=UUID(int=101),
            name="canonical queries",
            version="v1",
            query_count=50,
            content_hash="content-hash",
        ),
        baseline_config_id=_CONFIG_IDS[0],
        candidate_config_ids=_CONFIG_IDS[1:],
        summaries=[_summary(config_id, synthetic=synthetic) for config_id in _CONFIG_IDS]
        if completed
        else [],
        completed_queries=50 if completed else 0,
        total_queries=50,
        random_seed=20260822,
        environment=_environment(synthetic=synthetic),
        created_at=_NOW,
        started_at=(
            _NOW
            if status in {EvalRunStatus.RUNNING, EvalRunStatus.COMPLETED, EvalRunStatus.CANCELLED}
            else None
        ),
        completed_at=_NOW if terminal else None,
        error=error,
    )


def _view(status: EvalRunStatus, *, synthetic: bool = False) -> EvalRunView:
    completed = status is EvalRunStatus.COMPLETED
    return EvalRunView(
        run=_run(status, synthetic=synthetic),
        data_origin=DataOrigin.SYNTHETIC_DEMO if synthetic else DataOrigin.LIVE,
        completed_attempts=200 if completed else 0,
        live_replay_allowed=not synthetic,
    )


def test_create_request_freezes_exact_distinct_canonical_suite_and_bounds() -> None:
    request = CreateEvalRunRequest(
        query_set_id=uuid4(),
        baseline_config_id=_CONFIG_IDS[0],
        candidate_config_ids=_CONFIG_IDS[1:],
    )

    assert request.max_concurrency == 4
    assert request.warmup_query_count == 5
    with pytest.raises(ValidationError, match="at least 3 items"):
        CreateEvalRunRequest(
            query_set_id=uuid4(),
            baseline_config_id=_CONFIG_IDS[0],
            candidate_config_ids=_CONFIG_IDS[1:3],
        )
    with pytest.raises(ValidationError, match="must be distinct"):
        CreateEvalRunRequest(
            query_set_id=uuid4(),
            baseline_config_id=_CONFIG_IDS[0],
            candidate_config_ids=[_CONFIG_IDS[0], *_CONFIG_IDS[2:]],
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CreateEvalRunRequest.model_validate(
            {
                "query_set_id": str(uuid4()),
                "baseline_config_id": str(_CONFIG_IDS[0]),
                "candidate_config_ids": [str(value) for value in _CONFIG_IDS[1:]],
                "data_origin": "synthetic_demo",
            }
        )


@pytest.mark.parametrize(
    ("value", "sample_count", "valid"),
    [(None, 0, True), (0.0, 50, True), (None, 50, False), (1.0, 0, False)],
)
def test_metric_aggregate_freezes_null_sample_pairing(
    value: float | None,
    sample_count: int,
    valid: bool,
) -> None:
    if valid:
        MetricAggregate(name=MetricName.LATENCY_P50_MS, value=value, sample_count=sample_count)
    else:
        with pytest.raises(ValidationError, match="if and only if"):
            MetricAggregate(name=MetricName.LATENCY_P50_MS, value=value, sample_count=sample_count)


def test_old_m2_success_payload_decodes_and_round_trips_without_new_default_field() -> None:
    old_payload = {
        "contract_version": 1,
        "kind": "success",
        "ranked_document_ids": [str(UUID(int=200))],
        "metrics": {"ndcg_at_10": 1.0, "recall_at_50": 1.0, "mrr_at_10": 1.0},
        "total_client_wall_latency_ms": 12.5,
        "stage_timings": [],
        "candidate_counts": {"final": 1},
        "warnings": [],
        "trace_id": str(UUID(int=201)),
    }

    restored = EvalSuccessPayload.model_validate(old_payload)

    assert restored.timing_source is TimingSource.PERF_COUNTER
    assert restored.model_dump(mode="json") == old_payload


def test_synthetic_success_has_unavailable_latency_and_no_provider_evidence() -> None:
    synthetic = EvalSuccessPayload(
        ranked_document_ids=[UUID(int=200)],
        metrics=PerQueryMetrics(ndcg_at_10=1.0, recall_at_50=1.0, mrr_at_10=1.0),
        timing_source=TimingSource.SYNTHETIC_UNAVAILABLE,
        total_client_wall_latency_ms=None,
        stage_timings=[],
        candidate_counts={},
        warnings=[],
        trace_id=None,
    )

    payload = synthetic.model_dump(mode="json")
    assert payload["timing_source"] == "synthetic_unavailable"
    assert payload["total_client_wall_latency_ms"] is None
    assert payload["trace_id"] is None
    with pytest.raises(ValidationError, match="cannot claim total"):
        EvalSuccessPayload.model_validate({**payload, "total_client_wall_latency_ms": 0.0})
    with pytest.raises(ValidationError, match="cannot claim a provider trace"):
        EvalSuccessPayload.model_validate({**payload, "trace_id": str(UUID(int=201))})


@pytest.mark.parametrize("status", list(EvalRunStatus))
def test_run_views_cover_all_six_durable_statuses(status: EvalRunStatus) -> None:
    view = _view(status)

    assert view.run.status is status
    assert view.original_stage_evidence_available is False
    if status is EvalRunStatus.QUEUED:
        assert CreateEvalRunResponse(result=view).result.run.status is EvalRunStatus.QUEUED
    assert CancelEvalRunResponse(result=view).result.run.status is status


def test_failed_run_view_requires_a_direct_redacted_error() -> None:
    failed = _run(EvalRunStatus.FAILED).model_copy(update={"error": None})

    with pytest.raises(ValidationError, match="direct redacted error"):
        EvalRunView(
            run=failed,
            data_origin=DataOrigin.LIVE,
            completed_attempts=0,
            live_replay_allowed=True,
        )


def test_completed_run_view_requires_six_ordered_metric_summaries() -> None:
    completed = _run(EvalRunStatus.COMPLETED)
    first_summary = completed.summaries[0]
    completed.summaries[0] = first_summary.model_copy(
        update={"metrics": list(reversed(first_summary.metrics))}
    )

    with pytest.raises(ValidationError, match="six metrics in contract order"):
        EvalRunView(
            run=completed,
            data_origin=DataOrigin.LIVE,
            completed_attempts=200,
            live_replay_allowed=True,
        )


def test_run_list_is_versioned_newest_first_and_propagates_origin() -> None:
    newest = _view(EvalRunStatus.QUEUED)
    older = EvalRunView(
        run=_run(
            EvalRunStatus.QUEUED,
            run_id=UUID(int=99),
        ).model_copy(update={"created_at": datetime(2026, 8, 22, tzinfo=UTC)}),
        data_origin=DataOrigin.LIVE,
        completed_attempts=0,
        live_replay_allowed=True,
    )

    response = EvalRunListResponse(runs=[newest, older])

    assert response.model_dump(mode="json")["runs"][0]["data_origin"] == "live"
    with pytest.raises(ValidationError, match="newest first"):
        EvalRunListResponse(runs=[older, newest])


def test_query_set_catalog_propagates_synthetic_origin() -> None:
    query_set = QuerySet(
        id=UUID(int=300),
        name="PufferLab-authored demo queries",
        version="v1",
        dataset_version_id=UUID(int=301),
        query_count=50,
        content_hash="query-hash",
        created_at=_NOW,
    )
    response = QuerySetListResponse(
        dataset_version_id=query_set.dataset_version_id,
        query_sets=[
            QuerySetCatalogItem(query_set=query_set, data_origin=DataOrigin.SYNTHETIC_DEMO)
        ],
    )

    assert response.model_dump(mode="json")["query_sets"][0]["data_origin"] == "synthetic_demo"


def test_regression_response_has_paired_rows_and_exact_excluded_coverage() -> None:
    excluded = [
        ExcludedPairCount(status=status, count=0)
        for status in (
            RegressionPairStatus.BASELINE_MISSING,
            RegressionPairStatus.CANDIDATE_MISSING,
            RegressionPairStatus.BASELINE_FAILED,
            RegressionPairStatus.CANDIDATE_FAILED,
            RegressionPairStatus.BOTH_FAILED,
            RegressionPairStatus.NO_POSITIVE_QRELS,
        )
    ]
    row = RegressionRow(
        query_id=UUID(int=400),
        query_text="How do pipes work?",
        baseline_config_id=_CONFIG_IDS[0],
        candidate_config_id=_CONFIG_IDS[1],
        baseline_ndcg_at_10=0.8,
        candidate_ndcg_at_10=0.2,
        ndcg_delta=-0.6,
        recall_delta=-0.5,
        mrr_delta=-0.4,
        relevant_rank_changes=[],
        playground_url=("/playground?run=00000000-0000-0000-0000-000000000100&query=one"),
    )

    response = RegressionResponse(
        run_id=UUID(int=100),
        data_origin=DataOrigin.LIVE,
        baseline_config_id=_CONFIG_IDS[0],
        candidate_config_id=_CONFIG_IDS[1],
        order=RegressionOrder.REGRESSIONS,
        limit=10,
        rows=[row],
        coverage=RegressionCoverage(
            paired_queries=50,
            excluded=excluded,
        ),
    )

    assert response.coverage.paired_queries == 50
    with pytest.raises(ValidationError, match="cover all 50"):
        RegressionCoverage(paired_queries=49, excluded=excluded)


def test_gain_order_preserves_m2_full_inverse_uuid_tie_breaker() -> None:
    excluded = [
        ExcludedPairCount(status=status, count=0)
        for status in (
            RegressionPairStatus.BASELINE_MISSING,
            RegressionPairStatus.CANDIDATE_MISSING,
            RegressionPairStatus.BASELINE_FAILED,
            RegressionPairStatus.CANDIDATE_FAILED,
            RegressionPairStatus.BOTH_FAILED,
            RegressionPairStatus.NO_POSITIVE_QRELS,
        )
    ]

    def row(query_id: UUID) -> RegressionRow:
        return RegressionRow(
            query_id=query_id,
            query_text=f"query {query_id.int}",
            baseline_config_id=_CONFIG_IDS[0],
            candidate_config_id=_CONFIG_IDS[1],
            baseline_ndcg_at_10=0.5,
            candidate_ndcg_at_10=0.6,
            ndcg_delta=0.1,
            recall_delta=0.1,
            mrr_delta=0.1,
            relevant_rank_changes=[],
            playground_url="/playground?run=one",
        )

    response = RegressionResponse(
        run_id=UUID(int=100),
        data_origin=DataOrigin.LIVE,
        baseline_config_id=_CONFIG_IDS[0],
        candidate_config_id=_CONFIG_IDS[1],
        order=RegressionOrder.GAINS,
        limit=10,
        rows=[row(UUID(int=402)), row(UUID(int=401))],
        coverage=RegressionCoverage(paired_queries=50, excluded=excluded),
    )

    assert [item.query_id.int for item in response.rows] == [402, 401]
    with pytest.raises(ValidationError, match="deterministic quality ordering"):
        RegressionResponse.model_validate(
            {
                **response.model_dump(),
                "rows": list(reversed(response.model_dump()["rows"])),
            }
        )


def test_synthetic_export_requires_200_successes_with_null_timing() -> None:
    run = _run(EvalRunStatus.COMPLETED, synthetic=True)
    records = []
    for config_id in sorted(_CONFIG_IDS, key=str):
        for query_number in range(1, 51):
            query_id = UUID(int=1_000 + query_number)
            records.append(
                EvalOutcomeRecord(
                    run_id=run.id,
                    config_id=config_id,
                    query_id=query_id,
                    created_at=_NOW,
                    outcome=EvalSuccessPayload(
                        ranked_document_ids=[UUID(int=2_000 + query_number)],
                        metrics=PerQueryMetrics(
                            ndcg_at_10=1.0,
                            recall_at_50=1.0,
                            mrr_at_10=1.0,
                        ),
                        timing_source=TimingSource.SYNTHETIC_UNAVAILABLE,
                        total_client_wall_latency_ms=None,
                        stage_timings=[],
                        candidate_counts={},
                        warnings=[],
                        trace_id=None,
                    ),
                )
            )
    export = EvalRunExport(run=run, outcomes=records)

    response = EvalRunExportResponse(data_origin=DataOrigin.SYNTHETIC_DEMO, export=export)

    assert response.export.outcomes[0].outcome.total_client_wall_latency_ms is None
    with pytest.raises(ValidationError, match="exact 50-by-four"):
        EvalRunExportResponse(
            data_origin=DataOrigin.SYNTHETIC_DEMO,
            export=export.model_copy(update={"outcomes": records[:-1]}),
        )
