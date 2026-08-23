from datetime import UTC, datetime
from uuid import UUID

import pytest
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    EvalFailurePayload,
    EvalSuccessPayload,
    PerQueryMetrics,
)
from pufferlab.jobs import decode_outcome_payload, encode_outcome_payload
from pufferlab.persistence import QueryOutcome, QueryOutcomeStatus
from pydantic import ValidationError

_RUN_ID = UUID("c0a62f7d-a1bb-48dc-a9d6-9200ea14b525")
_CONFIG_ID = UUID("36d99904-8a46-4fe8-97fb-a027ab353c43")
_QUERY_ID = UUID("8d167380-4d61-456f-b28f-343b61994b28")
_TRACE_ID = UUID("efee7df1-b5e6-4897-98dd-e4e4941dd355")
_CREATED_AT = datetime(2026, 8, 22, 18, 0, tzinfo=UTC)


def _outcome(
    payload: EvalSuccessPayload | EvalFailurePayload,
    *,
    status: QueryOutcomeStatus,
) -> QueryOutcome:
    return QueryOutcome(
        run_id=_RUN_ID,
        config_id=_CONFIG_ID,
        query_id=_QUERY_ID,
        status=status,
        payload=encode_outcome_payload(payload),
        created_at=_CREATED_AT,
    )


def test_codec_round_trips_versioned_success_and_failure_variants() -> None:
    success = EvalSuccessPayload(
        ranked_document_ids=[],
        metrics=PerQueryMetrics(ndcg_at_10=0.0, recall_at_50=0.0, mrr_at_10=0.0),
        total_client_wall_latency_ms=1.25,
        stage_timings=[],
        candidate_counts={"final": 0},
        warnings=[],
        trace_id=_TRACE_ID,
    )
    failure = EvalFailurePayload(
        code=ApiErrorCode.RATE_LIMITED,
        message="retrieval provider request failed",
        retryable=True,
        operation="query_ann",
        trace_id=_TRACE_ID,
        total_client_wall_latency_ms=2.5,
    )

    assert decode_outcome_payload(_outcome(success, status=QueryOutcomeStatus.SUCCEEDED)) == success
    assert decode_outcome_payload(_outcome(failure, status=QueryOutcomeStatus.FAILED)) == failure


def test_codec_rejects_extra_fields_and_status_disagreement() -> None:
    failure = EvalFailurePayload(
        code=ApiErrorCode.PROVIDER_ERROR,
        message="retrieval provider request failed",
        retryable=False,
        operation="query_bm25",
        trace_id=_TRACE_ID,
        total_client_wall_latency_ms=1.0,
    )
    malformed = _outcome(failure, status=QueryOutcomeStatus.FAILED)
    malformed.payload["raw_response"] = "must not be accepted"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        decode_outcome_payload(malformed)
    with pytest.raises(ValueError, match="does not match"):
        decode_outcome_payload(_outcome(failure, status=QueryOutcomeStatus.SUCCEEDED))


def test_null_quality_requires_explicit_no_positive_qrels_warning() -> None:
    with pytest.raises(ValidationError, match="no_positive_qrels"):
        EvalSuccessPayload(
            ranked_document_ids=[],
            metrics=PerQueryMetrics(),
            total_client_wall_latency_ms=1.0,
            stage_timings=[],
            candidate_counts={},
            warnings=[],
            trace_id=_TRACE_ID,
        )
