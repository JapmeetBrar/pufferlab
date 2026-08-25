"""Provider-free catalog, run, regression, query-detail, and export projections."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import urlencode
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from pufferlab.application.view_errors import (
    EvaluationViewError,
    evaluation_conflict,
    evaluation_invalid,
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
from pufferlab.contracts.datasets import DataOrigin, DatasetVersion
from pufferlab.contracts.evals import (
    CandidateRelevantRankChanges,
    DatasetAttribution,
    EvalFailurePayload,
    EvalOutcomeRecord,
    EvalRun,
    EvalRunDetailResponse,
    EvalRunExport,
    EvalRunExportResponse,
    EvalRunListQuery,
    EvalRunListResponse,
    EvalRunQueryDetailResponse,
    EvalRunStatus,
    EvalRunView,
    EvalSuccessPayload,
    ExcludedPairCount,
    JudgedDocumentSummary,
    JudgedQuery,
    QuerySet,
    QuerySetSummary,
    RegressionCoverage,
    RegressionPairStatus,
    RegressionQuery,
    RegressionResponse,
    RegressionRow,
    RelevantRankChange,
)
from pufferlab.contracts.retrieval import (
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
)
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import (
    EvaluationWarning,
    EvaluationWarningCode,
    Judgment,
    PairStatus,
    QueryMetrics,
)
from pufferlab.evals.models import QueryOutcome as EvaluatedQueryOutcome
from pufferlab.evals.pairing import order_quality_deltas, paired_deltas
from pufferlab.jobs.eval_runner import export_outcome_record, finalize_durable_outcomes
from pufferlab.persistence.errors import PersistenceError, RecordNotFoundError
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.persistence.types import QueryOutcome

_CANONICAL_QUERY_COUNT = 50
_CANONICAL_CONFIG_COUNT = 4
_QUALITY_METRIC_TOLERANCE = 1e-12
_CANONICAL_CONFIG_MODES = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)
_EXCLUDED_PAIR_STATUS_ORDER = (
    RegressionPairStatus.BASELINE_MISSING,
    RegressionPairStatus.CANDIDATE_MISSING,
    RegressionPairStatus.BASELINE_FAILED,
    RegressionPairStatus.CANDIDATE_FAILED,
    RegressionPairStatus.BOTH_FAILED,
    RegressionPairStatus.NO_POSITIVE_QRELS,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _RunContext:
    run: EvalRun
    query_set: QuerySet
    dataset: DatasetVersion
    configs: tuple[RetrievalConfig, ...]
    outcomes: tuple[EvalOutcomeRecord, ...]

    @property
    def config_ids(self) -> tuple[UUID, ...]:
        return (self.run.baseline_config_id, *self.run.candidate_config_ids)


class EvaluationViewService:
    """Read immutable SQLite evidence without constructing provider-capable dependencies."""

    def __init__(self, repository: PufferLabRepository) -> None:
        self._repository = repository

    def list_datasets(self) -> DatasetListResponse:
        return self._read(
            "list_datasets",
            lambda: DatasetListResponse(
                datasets=[
                    DatasetCatalogItem(dataset=dataset, data_origin=dataset.data_origin)
                    for dataset in self._repository.list_dataset_versions(limit=100)
                ]
            ),
        )

    def get_dataset(self, dataset_version_id: UUID) -> DatasetDetailResponse:
        return self._read(
            "get_dataset",
            lambda: self._dataset_detail(dataset_version_id),
            not_found_message="dataset revision was not found",
        )

    def list_query_sets(self, dataset_version_id: UUID) -> QuerySetListResponse:
        return self._read(
            "list_query_sets",
            lambda: self._query_set_catalog(dataset_version_id),
            not_found_message="dataset revision was not found",
        )

    def list_dataset_configs(self, dataset_version_id: UUID) -> RetrievalConfigCatalogResponse:
        return self._read(
            "list_dataset_configs",
            lambda: self._config_catalog(dataset_version_id),
            not_found_message="dataset revision was not found",
        )

    def list_eval_runs(self, query: EvalRunListQuery) -> EvalRunListResponse:
        return self._read(
            "list_eval_runs",
            lambda: self._run_list(query),
        )

    def get_eval_run(self, run_id: UUID) -> EvalRunDetailResponse:
        return self._read(
            "get_eval_run",
            lambda: EvalRunDetailResponse(
                result=self._run_view(self._load_run_context(self._repository.get_run(run_id)))
            ),
            not_found_message="evaluation run was not found",
        )

    def get_regressions(self, run_id: UUID, query: RegressionQuery) -> RegressionResponse:
        return self._read(
            "get_regressions",
            lambda: self._regression_response(run_id, query),
            not_found_message="evaluation run was not found",
        )

    def get_query_detail(self, run_id: UUID, query_id: UUID) -> EvalRunQueryDetailResponse:
        return self._read(
            "get_query_detail",
            lambda: self._query_detail(run_id, query_id),
            not_found_message="evaluation run or query was not found",
        )

    def export_eval_run(self, run_id: UUID) -> EvalRunExportResponse:
        return self._read(
            "export_eval_run",
            lambda: self._export_response(run_id),
            not_found_message="evaluation run was not found",
        )

    def query_set_data_origin(self, query_set_id: UUID) -> DataOrigin:
        """Resolve origin for the provider-free control guard without loading query text."""
        return self._read(
            "resolve_query_set_origin",
            lambda: self._query_set_origin(query_set_id),
            not_found_message="query set was not found",
        )

    def _dataset_detail(self, dataset_version_id: UUID) -> DatasetDetailResponse:
        dataset = self._repository.get_dataset_version(dataset_version_id)
        return DatasetDetailResponse(dataset=dataset, data_origin=dataset.data_origin)

    def _query_set_catalog(self, dataset_version_id: UUID) -> QuerySetListResponse:
        dataset = self._repository.get_dataset_version(dataset_version_id)
        query_sets = sorted(
            (
                query_set
                for query_set in self._repository.list_query_sets(
                    dataset_version_id=dataset.id,
                    limit=100,
                )
                if query_set.query_count == _CANONICAL_QUERY_COUNT
            ),
            key=lambda value: (value.created_at, str(value.id)),
        )
        return QuerySetListResponse(
            dataset_version_id=dataset.id,
            query_sets=[
                QuerySetCatalogItem(query_set=query_set, data_origin=dataset.data_origin)
                for query_set in query_sets
            ],
        )

    def _config_catalog(self, dataset_version_id: UUID) -> RetrievalConfigCatalogResponse:
        dataset = self._repository.get_dataset_version(dataset_version_id)
        configs = self._repository.list_retrieval_configs(
            dataset_version_id=dataset.id,
            limit=100,
        )
        try:
            ordered = self._canonical_configs(configs, dataset_id=dataset.id)
        except ValueError:
            error = evaluation_conflict(
                message="dataset does not have one canonical four-config evaluation catalog",
                operation="list_dataset_configs",
            )
        else:
            return RetrievalConfigCatalogResponse(
                dataset_version_id=dataset.id,
                data_origin=dataset.data_origin,
                configs=[self._config_summary(config) for config in ordered],
            )
        raise error

    def _run_list(self, query: EvalRunListQuery) -> EvalRunListResponse:
        query_ids_by_set: dict[UUID, frozenset[UUID]] = {}
        return EvalRunListResponse(
            runs=[
                self._run_view(
                    self._load_run_context(
                        run,
                        query_ids_by_set=query_ids_by_set,
                    )
                )
                for run in self._repository.list_runs(limit=query.limit)
            ]
        )

    def _load_run_context(
        self,
        run: EvalRun,
        *,
        query_ids_by_set: dict[UUID, frozenset[UUID]] | None = None,
    ) -> _RunContext:
        query_set = self._repository.get_query_set_revision(run.query_set.id)
        expected_summary = QuerySetSummary(
            id=query_set.id,
            name=query_set.name,
            version=query_set.version,
            query_count=query_set.query_count,
            content_hash=query_set.content_hash,
        )
        if run.query_set != expected_summary or run.total_queries != query_set.query_count:
            raise ValueError("durable run and query-set revision do not agree")
        dataset = self._repository.get_dataset_version(query_set.dataset_version_id)
        configs = tuple(self._repository.list_run_configs(run.id))
        if configs != self._canonical_configs(configs, dataset_id=dataset.id):
            raise ValueError("durable run configs are not in canonical contract order")
        durable_outcomes = tuple(self._repository.list_outcomes(run.id, limit=200))
        outcomes = tuple(export_outcome_record(outcome) for outcome in durable_outcomes)
        query_ids = None if query_ids_by_set is None else query_ids_by_set.get(query_set.id)
        if query_ids is None:
            selected_query_ids = self._repository.list_query_ids(query_set.id, limit=50)
            query_ids = frozenset(selected_query_ids)
            if len(selected_query_ids) != _CANONICAL_QUERY_COUNT or len(query_ids) != len(
                selected_query_ids
            ):
                raise ValueError("P0 run views require exactly 50 unique judged-query identities")
            if query_ids_by_set is not None:
                query_ids_by_set[query_set.id] = query_ids
        self._validate_outcome_scope(run, query_set, configs, outcomes, query_ids=query_ids)
        self._validate_run_lifecycle(
            run,
            configs=configs,
            durable_outcomes=durable_outcomes,
            outcomes=outcomes,
            query_ids=query_ids,
        )
        return _RunContext(
            run=run,
            query_set=query_set,
            dataset=dataset,
            configs=configs,
            outcomes=outcomes,
        )

    @staticmethod
    def _validate_outcome_scope(
        run: EvalRun,
        query_set: QuerySet,
        configs: Sequence[RetrievalConfig],
        outcomes: Sequence[EvalOutcomeRecord],
        *,
        query_ids: frozenset[UUID],
    ) -> None:
        config_ids = {config.id for config in configs}
        identities = [(outcome.config_id, outcome.query_id) for outcome in outcomes]
        if len(identities) != len(set(identities)):
            raise ValueError("durable run outcomes contain duplicate identities")
        if any(
            outcome.run_id != run.id
            or outcome.config_id not in config_ids
            or outcome.query_id not in query_ids
            for outcome in outcomes
        ):
            raise ValueError("durable outcome is outside its run/config/query-set scope")
        if query_set.query_count != _CANONICAL_QUERY_COUNT:
            raise ValueError("P0 run views require exactly 50 judged queries")

    @staticmethod
    def _validate_run_lifecycle(
        run: EvalRun,
        *,
        configs: Sequence[RetrievalConfig],
        durable_outcomes: Sequence[QueryOutcome],
        outcomes: Sequence[EvalOutcomeRecord],
        query_ids: frozenset[UUID],
    ) -> None:
        config_ids = {config.id for config in configs}
        configs_by_query: dict[UUID, set[UUID]] = {query_id: set() for query_id in query_ids}
        for outcome in outcomes:
            configs_by_query[outcome.query_id].add(outcome.config_id)
        completed_groups = sum(
            observed_config_ids == config_ids for observed_config_ids in configs_by_query.values()
        )
        if run.completed_queries != completed_groups:
            raise ValueError("run progress does not match complete durable query groups")

        if run.started_at is not None and run.started_at < run.created_at:
            raise ValueError("run start timestamp precedes creation")
        if run.completed_at is not None and run.completed_at < run.created_at:
            raise ValueError("run completion timestamp precedes creation")
        if (
            run.started_at is not None
            and run.completed_at is not None
            and run.completed_at < run.started_at
        ):
            raise ValueError("run completion timestamp precedes its start")

        if run.status is EvalRunStatus.QUEUED:
            if (
                run.started_at is not None
                or run.completed_at is not None
                or outcomes
                or completed_groups != 0
                or run.summaries
                or run.error is not None
            ):
                raise ValueError("queued run lifecycle is inconsistent with durable evidence")
            return

        if run.status is EvalRunStatus.RUNNING:
            if (
                run.started_at is None
                or run.completed_at is not None
                or run.summaries
                or run.error is not None
            ):
                raise ValueError("running run lifecycle is inconsistent with durable evidence")
            return

        if run.status is EvalRunStatus.COMPLETED:
            if (
                run.started_at is None
                or run.completed_at is None
                or run.error is not None
                or completed_groups != _CANONICAL_QUERY_COUNT
                or len(outcomes) != _CANONICAL_QUERY_COUNT * _CANONICAL_CONFIG_COUNT
            ):
                raise ValueError("completed run lifecycle is inconsistent with durable evidence")
            recomputed_summaries = finalize_durable_outcomes(
                run,
                durable_outcomes,
                query_ids=sorted(query_ids, key=str),
            )
            if run.summaries != recomputed_summaries:
                raise ValueError("completed run summaries do not match durable outcomes")
            return

        if run.status is EvalRunStatus.FAILED:
            if run.completed_at is None or run.error is None or run.summaries:
                raise ValueError("failed run lifecycle is inconsistent with durable evidence")
            if run.started_at is None and (outcomes or completed_groups != 0):
                raise ValueError("unstarted failed run cannot contain durable outcomes")
            return

        if run.status is EvalRunStatus.CANCELLED:
            if run.completed_at is None or run.error is not None or run.summaries:
                raise ValueError("cancelled run lifecycle is inconsistent with durable evidence")
            if run.started_at is None and (outcomes or completed_groups != 0):
                raise ValueError("unstarted cancelled run cannot contain durable outcomes")
            return

        if run.status is EvalRunStatus.INTERRUPTED:
            if (
                run.started_at is None
                or run.completed_at is None
                or run.error is not None
                or run.summaries
            ):
                raise ValueError("interrupted run lifecycle is inconsistent with durable evidence")
            return

        raise ValueError("run status is outside the frozen lifecycle")

    def _run_view(self, context: _RunContext) -> EvalRunView:
        return EvalRunView(
            run=context.run,
            dataset_version_id=context.dataset.id,
            data_origin=context.dataset.data_origin,
            configs=[self._config_summary(config) for config in context.configs],
            completed_attempts=len(context.outcomes),
            original_stage_evidence_available=False,
            live_replay_policy_permitted=context.dataset.data_origin is DataOrigin.LIVE,
        )

    def _regression_response(self, run_id: UUID, query: RegressionQuery) -> RegressionResponse:
        context = self._load_run_context(self._repository.get_run(run_id))
        if query.candidate_config_id not in context.run.candidate_config_ids:
            raise evaluation_invalid(
                message="candidate config is not part of the requested evaluation run",
                operation="get_regressions",
            )
        query_set, judged_queries = self._repository.get_query_set(context.query_set.id)
        if query_set != context.query_set:
            raise ValueError("query-set metadata changed while building regressions")
        queries = self._exact_query_map(judged_queries)
        records = {(record.config_id, record.query_id): record for record in context.outcomes}
        baseline = self._engine_outcomes(
            records,
            config_id=context.run.baseline_config_id,
            query_ids=queries,
        )
        candidate = self._engine_outcomes(
            records,
            config_id=query.candidate_config_id,
            query_ids=queries,
        )
        deltas = paired_deltas(baseline, candidate)
        observed_query_ids = {outcome.query_id for outcome in baseline} | {
            outcome.query_id for outcome in candidate
        }
        both_absent_count = len(set(queries) - observed_query_ids)
        status_counts = Counter(delta.status.value for delta in deltas)
        # The frozen enum has no both-missing case. Match the existing engine's baseline-first
        # precedence for exact query-set IDs absent on both sides.
        status_counts[RegressionPairStatus.BASELINE_MISSING.value] += both_absent_count
        ordered = order_quality_deltas(
            deltas,
            order=query.order.value,
            limit=query.limit,
        )
        rows = [
            self._regression_row(
                context=context,
                query=queries[delta.query_id],
                candidate_config_id=query.candidate_config_id,
                baseline_record=records[(context.run.baseline_config_id, delta.query_id)],
                candidate_record=records[(query.candidate_config_id, delta.query_id)],
                ndcg_delta=delta.ndcg_delta,
                recall_delta=delta.recall_delta,
                mrr_delta=delta.mrr_delta,
            )
            for delta in ordered
        ]
        return RegressionResponse(
            run_id=context.run.id,
            data_origin=context.dataset.data_origin,
            baseline_config_id=context.run.baseline_config_id,
            candidate_config_id=query.candidate_config_id,
            order=query.order,
            limit=query.limit,
            rows=rows,
            coverage=RegressionCoverage(
                paired_queries=status_counts[PairStatus.PAIRED.value],
                excluded=[
                    ExcludedPairCount(status=status, count=status_counts[status.value])
                    for status in _EXCLUDED_PAIR_STATUS_ORDER
                ],
            ),
        )

    def _regression_row(
        self,
        *,
        context: _RunContext,
        query: JudgedQuery,
        candidate_config_id: UUID,
        baseline_record: EvalOutcomeRecord,
        candidate_record: EvalOutcomeRecord,
        ndcg_delta: float | None,
        recall_delta: float | None,
        mrr_delta: float | None,
    ) -> RegressionRow:
        baseline = baseline_record.outcome
        candidate = candidate_record.outcome
        if (
            not isinstance(baseline, EvalSuccessPayload)
            or not isinstance(candidate, EvalSuccessPayload)
            or baseline.metrics.ndcg_at_10 is None
            or candidate.metrics.ndcg_at_10 is None
            or ndcg_delta is None
            or recall_delta is None
            or mrr_delta is None
        ):
            raise ValueError("paired regression row lacks successful defined quality evidence")
        return RegressionRow(
            query_id=query.id,
            query_text=query.text,
            baseline_config_id=context.run.baseline_config_id,
            candidate_config_id=candidate_config_id,
            baseline_ndcg_at_10=baseline.metrics.ndcg_at_10,
            candidate_ndcg_at_10=candidate.metrics.ndcg_at_10,
            ndcg_delta=ndcg_delta,
            recall_delta=recall_delta,
            mrr_delta=mrr_delta,
            baseline_latency_ms=baseline.total_client_wall_latency_ms,
            candidate_latency_ms=candidate.total_client_wall_latency_ms,
            relevant_rank_changes=self._rank_changes(query, baseline, candidate),
            playground_url=self._playground_url(
                run_id=context.run.id,
                query_id=query.id,
                baseline_config_id=context.run.baseline_config_id,
                candidate_config_id=candidate_config_id,
            ),
        )

    def _query_detail(self, run_id: UUID, query_id: UUID) -> EvalRunQueryDetailResponse:
        context = self._load_run_context(self._repository.get_run(run_id))
        query_set, judged_queries = self._repository.get_query_set(context.query_set.id)
        if query_set != context.query_set:
            raise ValueError("query-set metadata changed while building query detail")
        queries = self._exact_query_map(judged_queries)
        query = queries.get(query_id)
        if query is None:
            raise evaluation_not_found(
                message="query was not found in the requested evaluation run",
                operation="get_query_detail",
            )
        judged_document_titles = self._repository.get_judged_document_titles(
            query_set.id,
            [qrel.document_id for qrel in query.qrels],
        )
        records_by_config = {
            record.config_id: record for record in context.outcomes if record.query_id == query.id
        }
        for record in records_by_config.values():
            self._validate_success_judgment(query, record)
        outcomes = [
            records_by_config[config_id]
            for config_id in context.config_ids
            if config_id in records_by_config
        ]
        baseline_payload = self._success_payload(
            records_by_config.get(context.run.baseline_config_id)
        )
        return EvalRunQueryDetailResponse(
            run_id=context.run.id,
            data_origin=context.dataset.data_origin,
            query=query,
            judged_documents=[
                JudgedDocumentSummary(
                    document_id=qrel.document_id,
                    title=judged_document_titles.get(qrel.document_id),
                )
                for qrel in query.qrels
            ],
            baseline_config_id=context.run.baseline_config_id,
            candidate_config_ids=list(context.run.candidate_config_ids),
            configs=[self._config_summary(config) for config in context.configs],
            outcomes=outcomes,
            rank_changes=[
                CandidateRelevantRankChanges(
                    candidate_config_id=candidate_config_id,
                    changes=self._rank_changes(
                        query,
                        baseline_payload,
                        self._success_payload(records_by_config.get(candidate_config_id)),
                    ),
                )
                for candidate_config_id in context.run.candidate_config_ids
            ],
            attribution=self._attribution(context.dataset.data_origin),
            original_stage_evidence_available=False,
            live_replay_policy_permitted=context.dataset.data_origin is DataOrigin.LIVE,
        )

    def _export_response(self, run_id: UUID) -> EvalRunExportResponse:
        context = self._load_run_context(self._repository.get_run(run_id))
        outcomes = sorted(
            context.outcomes,
            key=lambda record: (str(record.config_id), str(record.query_id)),
        )
        return EvalRunExportResponse(
            data_origin=context.dataset.data_origin,
            export=EvalRunExport(run=context.run, outcomes=outcomes),
        )

    def _query_set_origin(self, query_set_id: UUID) -> DataOrigin:
        query_set = self._repository.get_query_set_revision(query_set_id)
        return self._repository.get_dataset_version(query_set.dataset_version_id).data_origin

    @staticmethod
    def _exact_query_map(queries: Sequence[JudgedQuery]) -> dict[UUID, JudgedQuery]:
        query_map = {query.id: query for query in queries}
        if len(queries) != _CANONICAL_QUERY_COUNT or len(query_map) != len(queries):
            raise ValueError("P0 query reads require 50 unique judged query identities")
        return query_map

    @staticmethod
    def _engine_outcomes(
        records: dict[tuple[UUID, UUID], EvalOutcomeRecord],
        *,
        config_id: UUID,
        query_ids: dict[UUID, JudgedQuery],
    ) -> list[EvaluatedQueryOutcome]:
        outcomes: list[EvaluatedQueryOutcome] = []
        for query_id in sorted(query_ids, key=str):
            record = records.get((config_id, query_id))
            if record is None:
                continue
            payload = record.outcome
            if isinstance(payload, EvalFailurePayload):
                outcomes.append(
                    EvaluatedQueryOutcome.failed(
                        query_id=query_id,
                        error_code=payload.code.value,
                        latency_ms=payload.total_client_wall_latency_ms,
                    )
                )
                continue
            EvaluationViewService._validate_success_judgment(query_ids[query_id], record)
            warnings = (
                (
                    EvaluationWarning(
                        code=EvaluationWarningCode.NO_POSITIVE_QRELS,
                        message=(
                            "quality metrics are undefined because the query has no positive qrels"
                        ),
                    ),
                )
                if payload.metrics.ndcg_at_10 is None
                else ()
            )
            outcomes.append(
                EvaluatedQueryOutcome.succeeded(
                    query_id=query_id,
                    metrics=QueryMetrics(
                        ndcg_at_10=payload.metrics.ndcg_at_10,
                        recall_at_50=payload.metrics.recall_at_50,
                        mrr_at_10=payload.metrics.mrr_at_10,
                        warnings=warnings,
                    ),
                    latency_ms=payload.total_client_wall_latency_ms,
                )
            )
        return outcomes

    @staticmethod
    def _validate_success_judgment(query: JudgedQuery, record: EvalOutcomeRecord) -> None:
        payload = record.outcome
        if isinstance(payload, EvalFailurePayload):
            return
        recomputed = evaluate_ranking(
            payload.ranked_document_ids,
            [
                Judgment(
                    document_id=qrel.document_id,
                    relevance_grade=qrel.relevance_grade,
                )
                for qrel in query.qrels
            ],
        )
        stored_values = (
            payload.metrics.ndcg_at_10,
            payload.metrics.recall_at_50,
            payload.metrics.mrr_at_10,
        )
        recomputed_values = (
            recomputed.ndcg_at_10,
            recomputed.recall_at_50,
            recomputed.mrr_at_10,
        )
        if any(
            not EvaluationViewService._quality_metric_matches(stored, expected)
            for stored, expected in zip(stored_values, recomputed_values, strict=True)
        ):
            raise ValueError(
                "durable quality metrics do not match the ranked IDs and exact stored qrels"
            )

    @staticmethod
    def _quality_metric_matches(stored: float | None, expected: float | None) -> bool:
        if stored is None or expected is None:
            return stored is expected
        return math.isclose(
            stored,
            expected,
            rel_tol=_QUALITY_METRIC_TOLERANCE,
            abs_tol=_QUALITY_METRIC_TOLERANCE,
        )

    @staticmethod
    def _success_payload(record: EvalOutcomeRecord | None) -> EvalSuccessPayload | None:
        if record is None or isinstance(record.outcome, EvalFailurePayload):
            return None
        return record.outcome

    @staticmethod
    def _rank_changes(
        query: JudgedQuery,
        baseline: EvalSuccessPayload | None,
        candidate: EvalSuccessPayload | None,
    ) -> list[RelevantRankChange]:
        baseline_ranks = {
            document_id: rank
            for rank, document_id in enumerate(
                () if baseline is None else baseline.ranked_document_ids[:50],
                start=1,
            )
        }
        candidate_ranks = {
            document_id: rank
            for rank, document_id in enumerate(
                () if candidate is None else candidate.ranked_document_ids[:50],
                start=1,
            )
        }
        positive_qrels = [qrel for qrel in query.qrels if qrel.relevance_grade > 0]
        qrel_ids = [qrel.document_id for qrel in positive_qrels]
        if len(qrel_ids) != len(set(qrel_ids)):
            raise ValueError("judged query contains duplicate positive document identities")
        return [
            RelevantRankChange(
                document_id=qrel.document_id,
                relevance_grade=qrel.relevance_grade,
                baseline_rank=baseline_ranks.get(qrel.document_id),
                candidate_rank=candidate_ranks.get(qrel.document_id),
            )
            for qrel in positive_qrels
        ]

    @staticmethod
    def _playground_url(
        *,
        run_id: UUID,
        query_id: UUID,
        baseline_config_id: UUID,
        candidate_config_id: UUID,
    ) -> str:
        return "/playground?" + urlencode(
            {
                "run": str(run_id),
                "query": str(query_id),
                "left": str(baseline_config_id),
                "right": str(candidate_config_id),
            }
        )

    @staticmethod
    def _canonical_configs(
        configs: Sequence[RetrievalConfig],
        *,
        dataset_id: UUID,
    ) -> tuple[RetrievalConfig, ...]:
        by_mode = {config.mode: config for config in configs}
        if (
            len(configs) != _CANONICAL_CONFIG_COUNT
            or len(by_mode) != _CANONICAL_CONFIG_COUNT
            or any(config.dataset_version_id != dataset_id for config in configs)
        ):
            raise ValueError("evaluation configs are not one canonical four-mode catalog")
        ordered = tuple(by_mode[mode] for mode in _CANONICAL_CONFIG_MODES)
        if any(
            config.result_k != 50
            or config.candidate_k != 100
            or config.consistency != "strong"
            or config.filters is not None
            for config in ordered
        ):
            raise ValueError("evaluation configs do not have canonical retrieval bounds")
        reranker = ordered[-1].reranker
        if reranker is None or reranker.depth != 50:
            raise ValueError("evaluation reranker depth must be 50")
        return ordered

    @staticmethod
    def _config_summary(config: RetrievalConfig) -> RetrievalConfigSummary:
        return RetrievalConfigSummary(
            id=config.id,
            revision=config.revision,
            name=config.name,
            mode=config.mode,
            config_hash=config.config_hash,
        )

    @staticmethod
    def _attribution(origin: DataOrigin) -> DatasetAttribution:
        if origin is DataOrigin.SYNTHETIC_DEMO:
            return DatasetAttribution(source_name="PufferLab authored synthetic demo")
        return DatasetAttribution(
            source_name="CQADupStack Unix · Unix & Linux Stack Exchange",
            source_url="https://unix.stackexchange.com/",
            license_name="CC BY-SA 2.5 OR CC BY-SA 3.0",
            license_url="https://stackoverflow.com/help/licensing",
        )

    @staticmethod
    def _read(
        operation: str,
        loader: Callable[[], _T],
        *,
        not_found_message: str = "stored evaluation record was not found",
    ) -> _T:
        try:
            return loader()
        except EvaluationViewError as caught:
            error = EvaluationViewError(
                code=caught.code,
                message=str(caught),
                http_status=caught.http_status,
                operation=caught.operation,
                retryable=caught.retryable,
            )
        except RecordNotFoundError:
            error = evaluation_not_found(
                message=not_found_message,
                operation=operation,
            )
        except (PersistenceError, ValidationError, SQLAlchemyError, TypeError, ValueError):
            error = evaluation_unavailable(operation=operation)
        raise error
