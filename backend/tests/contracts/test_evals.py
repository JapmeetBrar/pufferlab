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
    EvalRunQueryDetailResponse,
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
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pydantic import ValidationError

_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
_CONFIG_IDS = [UUID(int=index) for index in range(1, 5)]
_DATASET_ID = UUID(int=50)


def _configs() -> list[RetrievalConfigSummary]:
    return [
        RetrievalConfigSummary(
            id=config_id,
            revision=1,
            name=name,
            mode=mode,
            config_hash=f"hash-{config_id}",
        )
        for config_id, name, mode in zip(
            _CONFIG_IDS,
            ("BM25", "ANN", "Server RRF", "Local reranker"),
            (
                RetrievalMode.BM25,
                RetrievalMode.VECTOR,
                RetrievalMode.HYBRID_RRF,
                RetrievalMode.HYBRID_RERANK,
            ),
            strict=True,
        )
    ]


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
        dataset_version_id=_DATASET_ID,
        data_origin=DataOrigin.SYNTHETIC_DEMO if synthetic else DataOrigin.LIVE,
        configs=_configs(),
        completed_attempts=200 if completed else 0,
        live_replay_policy_permitted=not synthetic,
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
    assert view.dataset_version_id == _DATASET_ID
    assert [config.id for config in view.configs] == _CONFIG_IDS
    assert view.original_stage_evidence_available is False
    payload = view.model_dump(mode="json")
    assert payload["live_replay_policy_permitted"] is True
    assert "live_replay_allowed" not in payload
    if status is EvalRunStatus.QUEUED:
        assert CreateEvalRunResponse(result=view).result.run.status is EvalRunStatus.QUEUED
    assert CancelEvalRunResponse(result=view).result.run.status is status

    with pytest.raises(ValidationError, match="contract order"):
        EvalRunView.model_validate(
            {
                **view.model_dump(),
                "configs": list(reversed(view.model_dump()["configs"])),
            }
        )


def test_failed_run_view_requires_a_direct_redacted_error() -> None:
    failed = _run(EvalRunStatus.FAILED).model_copy(update={"error": None})

    with pytest.raises(ValidationError, match="direct redacted error"):
        EvalRunView(
            run=failed,
            dataset_version_id=_DATASET_ID,
            data_origin=DataOrigin.LIVE,
            configs=_configs(),
            completed_attempts=0,
            live_replay_policy_permitted=True,
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
            dataset_version_id=_DATASET_ID,
            data_origin=DataOrigin.LIVE,
            configs=_configs(),
            completed_attempts=200,
            live_replay_policy_permitted=True,
        )


def test_synthetic_run_view_is_policy_ineligible_without_claiming_namespace_state() -> None:
    view = _view(EvalRunStatus.COMPLETED, synthetic=True)

    payload = view.model_dump(mode="json")
    assert payload["data_origin"] == "synthetic_demo"
    assert payload["live_replay_policy_permitted"] is False
    assert "namespace_available" not in payload
    with pytest.raises(ValidationError, match="read/export-only"):
        EvalRunView.model_validate({**payload, "live_replay_policy_permitted": True})


def test_query_detail_schema_uses_policy_permission_not_namespace_availability() -> None:
    properties = EvalRunQueryDetailResponse.model_json_schema()["properties"]

    assert "live_replay_policy_permitted" in properties
    assert "maxItems" not in properties["judged_documents"]
    assert "live_replay_allowed" not in properties
    assert "namespace_available" not in properties


def test_run_list_is_versioned_newest_first_and_propagates_origin() -> None:
    newest = _view(EvalRunStatus.QUEUED)
    older = EvalRunView(
        run=_run(
            EvalRunStatus.QUEUED,
            run_id=UUID(int=99),
        ).model_copy(update={"created_at": datetime(2026, 8, 22, tzinfo=UTC)}),
        dataset_version_id=_DATASET_ID,
        data_origin=DataOrigin.LIVE,
        configs=_configs(),
        completed_attempts=0,
        live_replay_policy_permitted=True,
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
        baseline_latency_ms=10.0,
        candidate_latency_ms=20.0,
        relevant_rank_changes=[],
        playground_url=(
            "/playground?run=00000000-0000-0000-0000-000000000064"
            "&query=00000000-0000-0000-0000-000000000190"
            "&left=00000000-0000-0000-0000-000000000001"
            "&right=00000000-0000-0000-0000-000000000002"
        ),
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
    with pytest.raises(ValidationError, match="synthetic regressions cannot claim"):
        RegressionResponse.model_validate(
            {**response.model_dump(), "data_origin": DataOrigin.SYNTHETIC_DEMO}
        )
    with pytest.raises(ValidationError, match="live paired regressions require"):
        RegressionResponse.model_validate(
            {
                **response.model_dump(),
                "rows": [
                    {
                        **response.model_dump()["rows"][0],
                        "baseline_latency_ms": None,
                        "candidate_latency_ms": None,
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="run/query/left/right"):
        RegressionRow.model_validate(
            {
                **row.model_dump(),
                "playground_url": (
                    "/playground?run=00000000-0000-0000-0000-000000000064"
                    "&query_text=licensed-query-text"
                ),
            }
        )


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
            baseline_latency_ms=10.0,
            candidate_latency_ms=11.0,
            relevant_rank_changes=[],
            playground_url=(
                "/playground?run=00000000-0000-0000-0000-000000000064"
                f"&query={query_id}&left={_CONFIG_IDS[0]}&right={_CONFIG_IDS[1]}"
            ),
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


def test_export_origin_binds_environment_outcomes_and_latency_summaries() -> None:
    synthetic_run = _run(EvalRunStatus.COMPLETED, synthetic=True)
    synthetic_records = []
    for config_id in sorted(_CONFIG_IDS, key=str):
        for query_number in range(1, 51):
            synthetic_records.append(
                EvalOutcomeRecord(
                    run_id=synthetic_run.id,
                    config_id=config_id,
                    query_id=UUID(int=3_000 + query_number),
                    created_at=_NOW,
                    outcome=EvalSuccessPayload(
                        ranked_document_ids=[],
                        metrics=PerQueryMetrics(
                            ndcg_at_10=0.0,
                            recall_at_50=0.0,
                            mrr_at_10=0.0,
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

    wrong_summary_run = synthetic_run.model_copy(deep=True)
    for summary in wrong_summary_run.summaries:
        summary.metrics[3] = MetricAggregate(
            name=MetricName.LATENCY_P50_MS,
            value=12.0,
            sample_count=50,
        )
        summary.metrics[4] = MetricAggregate(
            name=MetricName.LATENCY_P95_MS,
            value=20.0,
            sample_count=50,
        )
    with pytest.raises(ValidationError, match="null with zero samples"):
        EvalRunExportResponse(
            data_origin=DataOrigin.SYNTHETIC_DEMO,
            export=EvalRunExport(run=wrong_summary_run, outcomes=synthetic_records),
        )

    live_run = _run(EvalRunStatus.COMPLETED)
    with pytest.raises(ValidationError, match="match the export timing origin"):
        EvalRunExportResponse(
            data_origin=DataOrigin.LIVE,
            export=EvalRunExport(run=live_run, outcomes=synthetic_records),
        )

    wrong_environment = synthetic_run.model_copy(
        update={"environment": _environment(synthetic=False)}
    )
    with pytest.raises(ValidationError, match="origin and run timing source"):
        EvalRunExportResponse(
            data_origin=DataOrigin.SYNTHETIC_DEMO,
            export=EvalRunExport(run=wrong_environment, outcomes=synthetic_records),
        )
