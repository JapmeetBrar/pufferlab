"""Provider-free evaluation gates over one authenticated read-only catalog snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from pufferlab.application.evaluation_views import EvaluationViewService
from pufferlab.application.view_errors import EvaluationViewError
from pufferlab.contracts.datasets import DataOrigin, DatasetVersion
from pufferlab.contracts.evals import (
    EvalFailurePayload,
    EvalRun,
    EvalRunStatus,
    EvalSuccessPayload,
    JudgedQuery,
    MetricName,
    QuerySet,
)
from pufferlab.contracts.gates import GatePolicy, GateReport
from pufferlab.contracts.retrieval import RetrievalConfig
from pufferlab.datasets.cqadupstack import (
    DatasetPreparationError,
    load_curated_query_manifest,
    load_source_lock,
    load_unix_dataset_manifest,
)
from pufferlab.datasets.unix_application import authenticate_persisted_unix_query_set
from pufferlab.evals.aggregation import aggregate_outcomes
from pufferlab.evals.gates import GateEvaluationError, evaluate_gate
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import Judgment, QueryOutcome
from pufferlab.jobs.eval_runner import decode_outcome_payload
from pufferlab.persistence.errors import PersistenceError
from pufferlab.persistence.read_only import (
    ExistingReadOnlyCatalog,
    ReadOnlyCatalogError,
    open_existing_read_only_catalog,
)
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.persistence.types import QueryOutcome as DurableQueryOutcome
from pufferlab.retrieval.config import derive_bound_retrieval_configs
from pufferlab.synthetic_demo.authored import AUTHORED_SYNTHETIC_DEMO
from pufferlab.synthetic_demo.seeder import SyntheticDemoSeedError, materialize_synthetic_demo

_ROOT = Path(__file__).resolve().parents[3]
_UNIX_DATASET_ROOT = _ROOT / "datasets" / "cqadupstack-unix"
_UNIX_MANIFEST = _UNIX_DATASET_ROOT / "dataset-manifest.json"
_UNIX_CURATED_MANIFEST = _UNIX_DATASET_ROOT / "curated-50.json"
_UNIX_SOURCE_LOCK = _UNIX_DATASET_ROOT / "source-lock.json"
_CANONICAL_QUERY_COUNT = 50
_CANONICAL_CONFIG_COUNT = 4
_CANONICAL_OUTCOME_COUNT = _CANONICAL_QUERY_COUNT * _CANONICAL_CONFIG_COUNT


class GateApplicationStatus(StrEnum):
    """Finite application outcomes that carry no evidence or exception detail."""

    REPORT = "report"
    INVALID = "invalid"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class GateApplicationResult:
    """A verdict only after the source catalog has closed and retained its identity."""

    status: GateApplicationStatus
    report: GateReport | None = None

    def __post_init__(self) -> None:
        if (self.status is GateApplicationStatus.REPORT) != (self.report is not None):
            raise ValueError("only a successful application evaluation may carry a report")


_INVALID_EVIDENCE_ERRORS = (
    DatasetPreparationError,
    EvaluationViewError,
    GateEvaluationError,
    PersistenceError,
    ReadOnlyCatalogError,
    SQLAlchemyError,
    SyntheticDemoSeedError,
    TypeError,
    ValidationError,
    ValueError,
)


def evaluate_durable_gate(
    *,
    database_path: Path,
    run_id: UUID,
    candidate_config_id: UUID,
    policy: GatePolicy,
) -> GateApplicationResult:
    """Evaluate one completed durable run without migration, recovery, or live dependencies.

    The existing database is copied through the bounded portable read-only catalog. A verdict is
    returned only after catalog close repeats the exact file/parent/sidecar identity checks.
    """

    catalog: ExistingReadOnlyCatalog | None = None
    result = GateApplicationResult(status=GateApplicationStatus.INTERNAL)
    try:
        catalog = open_existing_read_only_catalog(database_path)
        result = _evaluate_catalog(
            catalog.repository,
            run_id=run_id,
            candidate_config_id=candidate_config_id,
            policy=policy,
        )
    except _INVALID_EVIDENCE_ERRORS as caught:
        _detach_exception(caught)
        del caught
        result = GateApplicationResult(status=GateApplicationStatus.INVALID)
    except BaseException as caught:
        _detach_exception(caught)
        del caught
        result = GateApplicationResult(status=GateApplicationStatus.INTERNAL)

    if catalog is not None:
        try:
            catalog.close()
        except ReadOnlyCatalogError as caught:
            _detach_exception(caught)
            del caught
            if result.status is not GateApplicationStatus.INTERNAL:
                result = GateApplicationResult(status=GateApplicationStatus.INVALID)
        except BaseException as caught:
            _detach_exception(caught)
            del caught
            result = GateApplicationResult(status=GateApplicationStatus.INTERNAL)
    catalog = None
    database_path = Path()
    return result


def _evaluate_catalog(
    repository: PufferLabRepository,
    *,
    run_id: UUID,
    candidate_config_id: UUID,
    policy: GatePolicy,
) -> GateApplicationResult:
    # This public projection validates indexed/payload identities, exact completed coverage,
    # config order, lifecycle timestamps, origin/timing rules, and stored summary consistency.
    exported = EvaluationViewService(repository).export_eval_run(run_id)
    run = exported.export.run
    if run.status is not EvalRunStatus.COMPLETED:
        raise ValueError("evaluation gates require completed durable evidence")
    if candidate_config_id not in run.candidate_config_ids:
        raise ValueError("candidate is outside the immutable run binding")

    query_set, query_values = repository.get_query_set(run.query_set.id)
    queries = tuple(query_values)
    dataset = repository.get_dataset_version(query_set.dataset_version_id)
    configs = tuple(repository.list_run_configs(run.id))
    catalog_configs = tuple(
        repository.list_retrieval_configs(
            dataset_version_id=dataset.id,
            limit=100,
        )
    )
    outcomes = tuple(repository.list_outcomes(run.id, limit=_CANONICAL_OUTCOME_COUNT))
    _authenticate_source_and_configs(
        dataset=dataset,
        query_set=query_set,
        queries=queries,
        configs=configs,
    )
    if len(catalog_configs) != _CANONICAL_CONFIG_COUNT or {
        config.id: config for config in catalog_configs
    } != {config.id: config for config in configs}:
        raise ValueError("dataset config catalog contains a hidden or foreign revision")
    expected_query_ids = tuple(query.id for query in queries)
    _validate_complete_identity_matrix(
        run=run,
        configs=configs,
        expected_query_ids=expected_query_ids,
        outcomes=outcomes,
    )

    query_by_id = {query.id: query for query in queries}
    evaluated: dict[tuple[UUID, UUID], QueryOutcome] = {}
    for outcome in outcomes:
        evaluated[(outcome.config_id, outcome.query_id)] = _recomputed_outcome(
            outcome,
            query=query_by_id[outcome.query_id],
        )
    _validate_recomputed_summaries(
        run=run,
        configs=configs,
        expected_query_ids=expected_query_ids,
        evaluated=evaluated,
    )

    baseline = tuple(
        _without_latency(evaluated[(run.baseline_config_id, query_id)])
        for query_id in expected_query_ids
    )
    candidate = tuple(
        _without_latency(evaluated[(candidate_config_id, query_id)])
        for query_id in expected_query_ids
    )
    report = evaluate_gate(
        run_id=run.id,
        baseline_config_id=run.baseline_config_id,
        candidate_config_id=candidate_config_id,
        expected_query_ids=expected_query_ids,
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=policy,
    )

    # Do not retain query text, qrels, ranked IDs, failure messages, or unselected outcomes in the
    # returned application object. GateReport contains only the frozen bounded safe fields.
    del exported
    run = cast(EvalRun, None)
    query_set = cast(QuerySet, None)
    query_values = cast(list[JudgedQuery], None)
    queries = ()
    dataset = cast(DatasetVersion, None)
    configs = ()
    catalog_configs = ()
    outcomes = ()
    query_by_id = {}
    evaluated = {}
    baseline = ()
    candidate = ()
    return GateApplicationResult(status=GateApplicationStatus.REPORT, report=report)


def _authenticate_source_and_configs(
    *,
    dataset: DatasetVersion,
    query_set: QuerySet,
    queries: tuple[JudgedQuery, ...],
    configs: tuple[RetrievalConfig, ...],
) -> None:
    if len(queries) != _CANONICAL_QUERY_COUNT or len(configs) != _CANONICAL_CONFIG_COUNT:
        raise ValueError("evaluation gate requires the canonical query and config catalogs")

    if dataset.data_origin is DataOrigin.SYNTHETIC_DEMO:
        expected = materialize_synthetic_demo()
        expected_queries = tuple(item.judged_query for item in AUTHORED_SYNTHETIC_DEMO.queries)
        if (
            dataset != expected.dataset_version
            or query_set != expected.query_set
            or queries != expected_queries
            or configs != expected.configs
        ):
            raise ValueError("synthetic sources or configs differ from authored materialization")
        return

    if dataset.data_origin is not DataOrigin.LIVE:
        raise ValueError("evaluation origin is outside the frozen gate domain")
    authenticate_persisted_unix_query_set(
        dataset,
        query_set,
        queries,
        curated_manifest=load_curated_query_manifest(_UNIX_CURATED_MANIFEST),
        checked_source_lock=load_source_lock(_UNIX_SOURCE_LOCK),
    )
    expected_configs = derive_bound_retrieval_configs(
        dataset,
        load_unix_dataset_manifest(_UNIX_MANIFEST),
        namespace=dataset.namespace,
    )
    if configs != expected_configs:
        raise ValueError("live configs differ from the exact dataset-bound catalog")


def _validate_complete_identity_matrix(
    *,
    run: EvalRun,
    configs: tuple[RetrievalConfig, ...],
    expected_query_ids: tuple[UUID, ...],
    outcomes: tuple[DurableQueryOutcome, ...],
) -> None:
    config_ids = tuple(config.id for config in configs)
    if config_ids != (run.baseline_config_id, *run.candidate_config_ids):
        raise ValueError("run config identities differ from the authenticated catalog")
    if len(expected_query_ids) != _CANONICAL_QUERY_COUNT or len(set(expected_query_ids)) != len(
        expected_query_ids
    ):
        raise ValueError("authenticated query identities are not one ordered canonical suite")
    expected = {
        (config_id, query_id) for config_id in config_ids for query_id in expected_query_ids
    }
    actual = {(outcome.config_id, outcome.query_id) for outcome in outcomes}
    if (
        len(outcomes) != _CANONICAL_OUTCOME_COUNT
        or len(actual) != len(outcomes)
        or actual != expected
    ):
        raise ValueError("completed evidence does not contain one exact 50-by-four outcome matrix")
    if (
        run.started_at is None
        or run.completed_at is None
        or any(
            outcome.created_at < run.started_at or outcome.created_at > run.completed_at
            for outcome in outcomes
        )
    ):
        raise ValueError("durable outcome timestamps are outside the completed run lifecycle")


def _recomputed_outcome(
    outcome: DurableQueryOutcome,
    *,
    query: JudgedQuery,
) -> QueryOutcome:
    payload = decode_outcome_payload(outcome)
    if isinstance(payload, EvalFailurePayload):
        return QueryOutcome.failed(
            query_id=query.id,
            error_code=payload.code.value,
            latency_ms=payload.total_client_wall_latency_ms,
        )
    if not isinstance(payload, EvalSuccessPayload):  # pragma: no cover - discriminated contract
        raise ValueError("durable outcome has an unknown payload kind")

    metrics = evaluate_ranking(
        payload.ranked_document_ids,
        tuple(
            Judgment(
                document_id=qrel.document_id,
                relevance_grade=qrel.relevance_grade,
            )
            for qrel in query.qrels
        ),
    )
    stored = (
        payload.metrics.ndcg_at_10,
        payload.metrics.recall_at_50,
        payload.metrics.mrr_at_10,
    )
    recomputed = (metrics.ndcg_at_10, metrics.recall_at_50, metrics.mrr_at_10)
    if stored != recomputed:
        raise ValueError("stored quality values differ from authenticated recomputation")
    return QueryOutcome.succeeded(
        query_id=query.id,
        metrics=metrics,
        latency_ms=payload.total_client_wall_latency_ms,
    )


def _validate_recomputed_summaries(
    *,
    run: EvalRun,
    configs: tuple[RetrievalConfig, ...],
    expected_query_ids: tuple[UUID, ...],
    evaluated: dict[tuple[UUID, UUID], QueryOutcome],
) -> None:
    summary_by_config = {summary.config_id: summary for summary in run.summaries}
    if len(summary_by_config) != _CANONICAL_CONFIG_COUNT:
        raise ValueError("completed run does not have four unique summaries")
    metric_order = (
        MetricName.NDCG_AT_10,
        MetricName.RECALL_AT_50,
        MetricName.MRR_AT_10,
        MetricName.LATENCY_P50_MS,
        MetricName.LATENCY_P95_MS,
        MetricName.ERROR_RATE,
    )
    for config in configs:
        outcomes = tuple(evaluated[(config.id, query_id)] for query_id in expected_query_ids)
        aggregate = aggregate_outcomes(outcomes)
        expected_metrics = (
            aggregate.ndcg_at_10,
            aggregate.recall_at_50,
            aggregate.mrr_at_10,
            aggregate.latency_p50_ms,
            aggregate.latency_p95_ms,
            aggregate.error_rate,
        )
        summary = summary_by_config[config.id]
        if (
            tuple(metric.name for metric in summary.metrics) != metric_order
            or tuple((metric.value, metric.sample_count) for metric in summary.metrics)
            != tuple((metric.value, metric.sample_count) for metric in expected_metrics)
            or summary.completed_queries != aggregate.completed_queries
            or summary.failed_queries != aggregate.failed_queries
        ):
            raise ValueError("stored summaries differ from full authenticated recomputation")


def _without_latency(outcome: QueryOutcome) -> QueryOutcome:
    if outcome.metrics is None:
        assert outcome.error_code is not None
        return QueryOutcome.failed(query_id=outcome.query_id, error_code=outcome.error_code)
    return QueryOutcome.succeeded(query_id=outcome.query_id, metrics=outcome.metrics)


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
