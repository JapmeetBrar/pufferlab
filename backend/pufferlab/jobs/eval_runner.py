"""Strict durable codecs and execution helpers for judged evaluation jobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import cast
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from pufferlab.contracts.common import JsonValue
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    EvalFailurePayload,
    EvalOutcomePayload,
    EvalOutcomeRecord,
    EvalOutcomeWarning,
    EvalRun,
    EvalSuccessPayload,
    JudgedQuery,
    MetricAggregate,
    MetricName,
    PerQueryMetrics,
)
from pufferlab.evals.aggregation import aggregate_outcomes
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import EvaluationWarning, EvaluationWarningCode, Judgment, QueryMetrics
from pufferlab.evals.models import QueryOutcome as EvaluatedQueryOutcome
from pufferlab.jobs.manager import QueryWorkItem
from pufferlab.persistence.types import QueryOutcome, QueryOutcomeStatus
from pufferlab.providers.errors import ProviderError
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.types import SearchBackend, SearchExecuteRequest

_PAYLOAD_ADAPTER: TypeAdapter[EvalOutcomePayload] = TypeAdapter(EvalOutcomePayload)
_SUMMARY_METRIC_ORDER = (
    MetricName.NDCG_AT_10,
    MetricName.RECALL_AT_50,
    MetricName.MRR_AT_10,
    MetricName.LATENCY_P50_MS,
    MetricName.LATENCY_P95_MS,
    MetricName.ERROR_RATE,
)


def encode_outcome_payload(payload: EvalOutcomePayload) -> dict[str, JsonValue]:
    """Encode only a contract-validated, JSON-safe durable payload."""
    encoded = _PAYLOAD_ADAPTER.dump_python(payload, mode="json")
    return cast(dict[str, JsonValue], encoded)


def decode_outcome_payload(outcome: QueryOutcome) -> EvalOutcomePayload:
    """Decode and cross-check the persistence status against the versioned discriminator."""
    payload = _PAYLOAD_ADAPTER.validate_python(outcome.payload)
    expected = (
        QueryOutcomeStatus.SUCCEEDED
        if isinstance(payload, EvalSuccessPayload)
        else QueryOutcomeStatus.FAILED
    )
    if outcome.status is not expected:
        raise ValueError("durable outcome status does not match its typed payload")
    return payload


def export_outcome_record(outcome: QueryOutcome) -> EvalOutcomeRecord:
    return EvalOutcomeRecord(
        run_id=outcome.run_id,
        config_id=outcome.config_id,
        query_id=outcome.query_id,
        created_at=outcome.created_at,
        outcome=decode_outcome_payload(outcome),
    )


class EvaluationOutcomeExecutor:
    """Execute one config/query retrieval and produce a redacted durable outcome."""

    def __init__(
        self,
        *,
        run_id: UUID,
        namespace: str,
        queries: Mapping[UUID, JudgedQuery],
        search_backend: SearchBackend,
        clock: Callable[[], float] = perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        trace_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._run_id = run_id
        self._namespace = namespace
        self._queries = dict(queries)
        self._search_backend = search_backend
        self._clock = clock
        self._now = now
        self._trace_id_factory = trace_id_factory

    async def __call__(self, item: QueryWorkItem) -> QueryOutcome:
        query = self._queries.get(item.query_id)
        if query is None:
            raise ValueError("work item references a query outside the bound query set")

        started = self._clock()
        try:
            execution = await self._search_backend.search_one(
                SearchExecuteRequest(
                    namespace=self._namespace,
                    query_text=query.text,
                    config_id=item.config_id,
                    query_id=query.id,
                    filter_override=query.filters,
                    debug_provenance=False,
                )
            )
        except SearchError as error:
            return self._failed_outcome(
                item,
                code=error.details.code,
                message=str(error),
                retryable=error.details.retryable,
                operation=error.details.operation,
                started=started,
            )
        except ProviderError as error:
            return self._failed_outcome(
                item,
                code=error.details.code,
                message=str(error),
                retryable=error.details.retryable,
                operation=error.details.operation,
                started=started,
            )

        if execution.config_id != item.config_id or execution.query_id != item.query_id:
            raise ValueError("search execution identity does not match its work item")
        result = execution.result
        if result.config.id != item.config_id:
            raise ValueError("search result configuration does not match its work item")
        ranked_document_ids = [hit.document_id for hit in result.hits[:50]]

        judged = evaluate_ranking(
            ranked_document_ids,
            [
                Judgment(document_id=qrel.document_id, relevance_grade=qrel.relevance_grade)
                for qrel in query.qrels
            ],
        )
        warnings = [
            EvalOutcomeWarning(code=warning.code.value, message=warning.message)
            for warning in judged.warnings
        ]
        warnings.extend(
            EvalOutcomeWarning(code=warning.code, message=warning.message)
            for warning in result.warnings
        )
        payload = EvalSuccessPayload(
            ranked_document_ids=ranked_document_ids,
            metrics=_contract_metrics(judged),
            total_client_wall_latency_ms=_elapsed_ms(self._clock, started),
            stage_timings=result.timings,
            candidate_counts=result.candidate_counts,
            warnings=warnings,
            trace_id=result.trace_id,
        )
        return QueryOutcome(
            run_id=self._run_id,
            config_id=item.config_id,
            query_id=item.query_id,
            status=QueryOutcomeStatus.SUCCEEDED,
            payload=encode_outcome_payload(payload),
            created_at=self._now(),
        )

    def _failed_outcome(
        self,
        item: QueryWorkItem,
        *,
        code: ApiErrorCode,
        message: str,
        retryable: bool,
        operation: str,
        started: float,
        trace_id: UUID | None = None,
    ) -> QueryOutcome:
        payload = EvalFailurePayload(
            code=code,
            message=message,
            retryable=retryable,
            operation=operation,
            trace_id=trace_id or self._trace_id_factory(),
            total_client_wall_latency_ms=_elapsed_ms(self._clock, started),
        )
        return QueryOutcome(
            run_id=self._run_id,
            config_id=item.config_id,
            query_id=item.query_id,
            status=QueryOutcomeStatus.FAILED,
            payload=encode_outcome_payload(payload),
            created_at=self._now(),
        )


def finalize_durable_outcomes(
    run: EvalRun,
    outcomes: Sequence[QueryOutcome],
    *,
    query_ids: Sequence[UUID],
) -> list[ConfigRunSummary]:
    """Require exact run coverage and aggregate exclusively from typed durable outcomes."""
    config_ids = [run.baseline_config_id, *run.candidate_config_ids]
    expected = {(config_id, query_id) for config_id in config_ids for query_id in query_ids}
    if any(outcome.run_id != run.id for outcome in outcomes):
        raise ValueError("durable outcome belongs to another run")
    actual = {(outcome.config_id, outcome.query_id) for outcome in outcomes}
    if len(outcomes) != len(actual):
        raise ValueError("durable outcomes contain duplicate config/query identities")
    if actual != expected:
        raise ValueError("finalization requires exact config/query outcome coverage")

    summaries: list[ConfigRunSummary] = []
    for config_id in config_ids:
        evaluated: list[EvaluatedQueryOutcome] = []
        for outcome in sorted(
            (item for item in outcomes if item.config_id == config_id),
            key=lambda item: str(item.query_id),
        ):
            payload = decode_outcome_payload(outcome)
            if isinstance(payload, EvalSuccessPayload):
                evaluated.append(
                    EvaluatedQueryOutcome.succeeded(
                        query_id=outcome.query_id,
                        metrics=_engine_metrics(payload),
                        latency_ms=payload.total_client_wall_latency_ms,
                    )
                )
            else:
                evaluated.append(
                    EvaluatedQueryOutcome.failed(
                        query_id=outcome.query_id,
                        error_code=payload.code.value,
                        latency_ms=payload.total_client_wall_latency_ms,
                    )
                )
        aggregate = aggregate_outcomes(evaluated)
        metric_values = {
            MetricName.NDCG_AT_10: aggregate.ndcg_at_10,
            MetricName.RECALL_AT_50: aggregate.recall_at_50,
            MetricName.MRR_AT_10: aggregate.mrr_at_10,
            MetricName.LATENCY_P50_MS: aggregate.latency_p50_ms,
            MetricName.LATENCY_P95_MS: aggregate.latency_p95_ms,
            MetricName.ERROR_RATE: aggregate.error_rate,
        }
        summaries.append(
            ConfigRunSummary(
                config_id=config_id,
                metrics=[
                    MetricAggregate(
                        name=name,
                        value=metric_values[name].value,
                        sample_count=metric_values[name].sample_count,
                    )
                    for name in _SUMMARY_METRIC_ORDER
                ],
                completed_queries=aggregate.completed_queries,
                failed_queries=aggregate.failed_queries,
            )
        )
    return summaries


def _contract_metrics(metrics: QueryMetrics) -> PerQueryMetrics:
    return PerQueryMetrics(
        ndcg_at_10=metrics.ndcg_at_10,
        recall_at_50=metrics.recall_at_50,
        mrr_at_10=metrics.mrr_at_10,
    )


def _engine_metrics(payload: EvalSuccessPayload) -> QueryMetrics:
    warnings = (
        (
            EvaluationWarning(
                code=EvaluationWarningCode.NO_POSITIVE_QRELS,
                message="quality metrics are undefined because the query has no positive qrels",
            ),
        )
        if payload.metrics.ndcg_at_10 is None
        else ()
    )
    return QueryMetrics(
        ndcg_at_10=payload.metrics.ndcg_at_10,
        recall_at_50=payload.metrics.recall_at_50,
        mrr_at_10=payload.metrics.mrr_at_10,
        warnings=warnings,
    )


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return max(0.0, (clock() - started) * 1000.0)
