"""Independently verify one completed M2 evaluation from its owned local state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from pufferlab.config import Settings
from pufferlab.contracts.datasets import DatasetVersion
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    EvalRunExport,
    EvalRunStatus,
    EvalSuccessPayload,
    JudgedQuery,
    MetricAggregate,
    MetricName,
    QuerySet,
    QuerySetSummary,
)
from pufferlab.contracts.retrieval import RetrievalConfig, RetrievalMode
from pufferlab.datasets.cqadupstack import load_processed_pack_lock, load_source_lock
from pufferlab.datasets.unix_application import (
    build_ready_unix_evaluation_seed,
    load_curated_unix_local_pack,
)
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import Judgment
from pufferlab.jobs import decode_outcome_payload
from pufferlab.jobs.eval_runner import export_outcome_record
from pufferlab.persistence import PufferLabRepository, QueryOutcome
from pufferlab.persistence.canonical import canonical_json
from pufferlab.retrieval.config import bind_retrieval_catalog
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

if __package__:
    from scripts import m2_live_namespace_session as _packaged_session

    m2_live_namespace_session = _packaged_session
else:
    import m2_live_namespace_session as _direct_session  # type: ignore[import-not-found]

    m2_live_namespace_session = _direct_session

_ROOT = Path(__file__).parents[1]
_DATASET_DIR = _ROOT / "datasets" / "cqadupstack-unix"
_PACK_CONTENT_SHA256 = "6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8"
_PROCESSED_PACK = (
    _ROOT / "data" / "cqadupstack-unix" / "processed" / f"cqadupstack-unix-{_PACK_CONTENT_SHA256}"
)
_QUERY_COUNT = 50
_CONFIG_MODES = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)
_OUTCOME_COUNT = _QUERY_COUNT * len(_CONFIG_MODES)
_METRIC_ORDER = (
    MetricName.NDCG_AT_10,
    MetricName.RECALL_AT_50,
    MetricName.MRR_AT_10,
    MetricName.LATENCY_P50_MS,
    MetricName.LATENCY_P95_MS,
    MetricName.ERROR_RATE,
)
_FORBIDDEN_EXPORT_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "body",
        "body_excerpt",
        "credentials",
        "document_text",
        "embedding",
        "embeddings",
        "headers",
        "query_text",
        "raw_provider_response",
        "raw_response",
        "request_headers",
        "secret",
        "text",
        "title",
        "vector",
        "vectors",
    }
)


class EvaluationVerificationError(RuntimeError):
    """A safe internal classification that the CLI reports without details."""


@dataclass(frozen=True, slots=True)
class _ExpectedSuite:
    dataset: DatasetVersion
    query_set: QuerySet
    queries: tuple[JudgedQuery, ...]
    configs: tuple[RetrievalConfig, ...]


@dataclass(frozen=True, slots=True)
class VerifiedEvaluation:
    run_id: UUID
    dataset_version_id: UUID
    query_set_id: UUID
    query_count: int
    outcome_count: int
    summaries: tuple[ConfigRunSummary, ...]
    export_sha256: str


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        f"{path.as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )


@contextmanager
def _read_only_repository(path: Path) -> Iterator[PufferLabRepository]:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise EvaluationVerificationError("configured evaluation database does not exist") from None
    if not resolved.is_file():
        raise EvaluationVerificationError("configured evaluation database is not a file")

    engine: Engine = create_engine(
        "sqlite+pysqlite://",
        creator=lambda: _read_only_connection(resolved),
    )
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    try:
        yield PufferLabRepository(factory)
    finally:
        engine.dispose()


def _load_expected_suite(namespace: str) -> _ExpectedSuite:
    source_lock = load_source_lock(_DATASET_DIR / "source-lock.json")
    processed_lock = load_processed_pack_lock(_DATASET_DIR / "processed-pack-lock.json")
    local_pack = load_curated_unix_local_pack(
        _PROCESSED_PACK,
        source_lock=source_lock,
        processed_pack_lock=processed_lock,
        dataset_manifest_path=_DATASET_DIR / "dataset-manifest.json",
        curated_manifest_path=_DATASET_DIR / "curated-50.json",
    )
    seed = build_ready_unix_evaluation_seed(local_pack, namespace=namespace)
    bound = bind_retrieval_catalog(
        seed.dataset_version,
        local_pack.corpus.manifest,
        namespace=namespace,
    )
    return _ExpectedSuite(
        dataset=seed.dataset_version,
        query_set=seed.query_set,
        queries=seed.judged_queries,
        configs=bound.configs,
    )


def _reject_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvaluationVerificationError("evaluation export has a non-string field")
            if key.casefold() in _FORBIDDEN_EXPORT_FIELDS:
                raise EvaluationVerificationError("evaluation export contains a forbidden field")
            _reject_forbidden_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_fields(item)


def _mean(values: Sequence[float]) -> float:
    if len(values) != _QUERY_COUNT:
        raise EvaluationVerificationError("evaluation metric sample coverage is invalid")
    return math.fsum(values) / len(values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if len(values) != _QUERY_COUNT:
        raise EvaluationVerificationError("evaluation latency sample coverage is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _recompute_summaries(
    outcomes: Sequence[QueryOutcome],
    *,
    configs: tuple[RetrievalConfig, ...],
    queries: tuple[JudgedQuery, ...],
) -> tuple[ConfigRunSummary, ...]:
    expected_identities = {(config.id, query.id) for config in configs for query in queries}
    actual_identities = {(outcome.config_id, outcome.query_id) for outcome in outcomes}
    if len(outcomes) != len(actual_identities) or actual_identities != expected_identities:
        raise EvaluationVerificationError("evaluation outcome identity coverage is invalid")
    query_by_id = {query.id: query for query in queries}
    outcome_by_identity = {(outcome.config_id, outcome.query_id): outcome for outcome in outcomes}

    summaries: list[ConfigRunSummary] = []
    for config in configs:
        ndcg_values: list[float] = []
        recall_values: list[float] = []
        mrr_values: list[float] = []
        latency_values: list[float] = []
        for query_id in sorted(query_by_id, key=str):
            query = query_by_id[query_id]
            outcome = outcome_by_identity[(config.id, query_id)]
            try:
                payload = decode_outcome_payload(outcome)
            except (TypeError, ValueError):
                raise EvaluationVerificationError("evaluation outcome payload is invalid") from None
            if not isinstance(payload, EvalSuccessPayload):
                raise EvaluationVerificationError("evaluation must have 200 successful outcomes")
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
            quality = (
                recomputed.ndcg_at_10,
                recomputed.recall_at_50,
                recomputed.mrr_at_10,
            )
            stored_quality = (
                payload.metrics.ndcg_at_10,
                payload.metrics.recall_at_50,
                payload.metrics.mrr_at_10,
            )
            if quality != stored_quality:
                raise EvaluationVerificationError(
                    "stored per-query metrics do not match ranking and qrels"
                )
            if any(value is None for value in quality):
                raise EvaluationVerificationError("curated query quality metrics must be defined")
            ndcg, recall, mrr = quality
            assert ndcg is not None and recall is not None and mrr is not None
            ndcg_values.append(ndcg)
            recall_values.append(recall)
            mrr_values.append(mrr)
            latency_values.append(payload.total_client_wall_latency_ms)

        summaries.append(
            ConfigRunSummary(
                config_id=config.id,
                metrics=[
                    MetricAggregate(
                        name=MetricName.NDCG_AT_10,
                        value=_mean(ndcg_values),
                        sample_count=_QUERY_COUNT,
                    ),
                    MetricAggregate(
                        name=MetricName.RECALL_AT_50,
                        value=_mean(recall_values),
                        sample_count=_QUERY_COUNT,
                    ),
                    MetricAggregate(
                        name=MetricName.MRR_AT_10,
                        value=_mean(mrr_values),
                        sample_count=_QUERY_COUNT,
                    ),
                    MetricAggregate(
                        name=MetricName.LATENCY_P50_MS,
                        value=_percentile(latency_values, 0.50),
                        sample_count=_QUERY_COUNT,
                    ),
                    MetricAggregate(
                        name=MetricName.LATENCY_P95_MS,
                        value=_percentile(latency_values, 0.95),
                        sample_count=_QUERY_COUNT,
                    ),
                    MetricAggregate(
                        name=MetricName.ERROR_RATE,
                        value=0.0,
                        sample_count=_QUERY_COUNT,
                    ),
                ],
                completed_queries=_QUERY_COUNT,
                failed_queries=0,
            )
        )
    return tuple(summaries)


def _verify_repository(
    repository: PufferLabRepository,
    run_id: UUID,
    *,
    expected: _ExpectedSuite,
) -> VerifiedEvaluation:
    run = repository.get_run(run_id)
    if run.status is not EvalRunStatus.COMPLETED:
        raise EvaluationVerificationError("evaluation run is not completed")
    if (
        run.total_queries != _QUERY_COUNT
        or run.completed_queries != _QUERY_COUNT
        or run.started_at is None
        or run.completed_at is None
        or run.error is not None
    ):
        raise EvaluationVerificationError("evaluation run lifecycle or coverage is invalid")

    config_ids = (run.baseline_config_id, *run.candidate_config_ids)
    expected_config_ids = tuple(config.id for config in expected.configs)
    if config_ids != expected_config_ids:
        raise EvaluationVerificationError("evaluation run does not use the exact canonical configs")
    configs = tuple(repository.get_retrieval_config(config_id) for config_id in config_ids)
    if configs != expected.configs or tuple(config.mode for config in configs) != _CONFIG_MODES:
        raise EvaluationVerificationError("persisted evaluation configs are not canonical")

    query_set, queries = repository.get_query_set(run.query_set.id)
    if query_set != expected.query_set or tuple(queries) != expected.queries:
        raise EvaluationVerificationError("persisted curated query set is not canonical")
    dataset = repository.get_dataset_version(query_set.dataset_version_id)
    if dataset != expected.dataset:
        raise EvaluationVerificationError("persisted Unix dataset revision is not canonical")
    expected_query_summary = QuerySetSummary(
        id=expected.query_set.id,
        name=expected.query_set.name,
        version=expected.query_set.version,
        query_count=expected.query_set.query_count,
        content_hash=expected.query_set.content_hash,
    )
    if run.query_set != expected_query_summary:
        raise EvaluationVerificationError("evaluation run query-set summary is not canonical")

    outcomes = repository.list_outcomes(run_id)
    if len(outcomes) != _OUTCOME_COUNT:
        raise EvaluationVerificationError("evaluation must contain exactly 200 outcomes")
    recomputed = _recompute_summaries(
        outcomes,
        configs=expected.configs,
        queries=expected.queries,
    )
    if tuple(run.summaries) != recomputed:
        raise EvaluationVerificationError("persisted summaries do not match independent results")
    if any(
        tuple(metric.name for metric in summary.metrics) != _METRIC_ORDER for summary in recomputed
    ):
        raise EvaluationVerificationError("evaluation summary metric order is invalid")

    exported = EvalRunExport(
        run=run,
        outcomes=[
            export_outcome_record(outcome)
            for outcome in sorted(
                outcomes,
                key=lambda outcome: (str(outcome.config_id), str(outcome.query_id)),
            )
        ],
    )
    canonical = canonical_json(exported)
    try:
        restored = EvalRunExport.model_validate_json(canonical)
    except ValueError:
        raise EvaluationVerificationError("canonical evaluation export is invalid") from None
    if restored != exported or canonical_json(restored) != canonical:
        raise EvaluationVerificationError("canonical evaluation export does not round-trip")
    _reject_forbidden_fields(json.loads(canonical))

    return VerifiedEvaluation(
        run_id=run.id,
        dataset_version_id=dataset.id,
        query_set_id=query_set.id,
        query_count=len(queries),
        outcome_count=len(outcomes),
        summaries=recomputed,
        export_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _verify_evaluation_at(
    run_id: UUID,
    *,
    database_path: Path,
    expected: _ExpectedSuite,
) -> VerifiedEvaluation:
    with _read_only_repository(database_path) as repository:
        return _verify_repository(repository, run_id, expected=expected)


def verify_evaluation(run_id: UUID) -> VerifiedEvaluation:
    """Verify the UUID against only fixed owned session, pack, locks, and database paths."""
    session = m2_live_namespace_session.load_session()
    expected = _load_expected_suite(session.namespace)
    return _verify_evaluation_at(
        run_id,
        database_path=Settings().database_path,
        expected=expected,
    )


def _metric_value(summary: ConfigRunSummary, name: str) -> str:
    metric = next(metric for metric in summary.metrics if metric.name.value == name)
    value = "null" if metric.value is None else format(metric.value, ".12g")
    return f"{value}[n={metric.sample_count}]"


def render_report(report: VerifiedEvaluation) -> tuple[str, ...]:
    lines = [
        f"run_id={report.run_id} status=completed",
        f"dataset_version_id={report.dataset_version_id} query_set_id={report.query_set_id} "
        f"query_count={report.query_count} config_count={len(report.summaries)} "
        f"outcome_count={report.outcome_count}",
    ]
    for summary in report.summaries:
        metrics = " ".join(
            f"{name}={_metric_value(summary, name)}"
            for name in (
                "ndcg@10",
                "recall@50",
                "mrr@10",
                "latency_p50_ms",
                "latency_p95_ms",
                "error_rate",
            )
        )
        lines.append(
            f"config_id={summary.config_id} completed={summary.completed_queries} "
            f"failed={summary.failed_queries} {metrics}"
        )
    lines.append(f"export_sha256={report.export_sha256}")
    lines.append("verification=passed")
    return tuple(lines)


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Accept only one run UUID and never print stored payloads or exception details."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID, help="completed evaluation run UUID")
    arguments = parser.parse_args(argv)
    try:
        report = verify_evaluation(arguments.run_id)
    except Exception:
        print("verification=failed", file=sys.stderr)
        return 1
    for line in render_report(report):
        print(line)
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
