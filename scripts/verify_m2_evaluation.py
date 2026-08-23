"""Verify one completed Milestone 2 evaluation from ignored SQLite state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from pufferlab.config import Settings
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    EvalRunExport,
    EvalRunStatus,
    MetricName,
    QuerySetSummary,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.jobs import finalize_durable_outcomes
from pufferlab.jobs.eval_runner import export_outcome_record
from pufferlab.persistence import PufferLabRepository
from pufferlab.persistence.canonical import canonical_json
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

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
class VerifiedEvaluation:
    run_id: UUID
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


def _verify_repository(
    repository: PufferLabRepository,
    run_id: UUID,
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
    if len(config_ids) != len(_CONFIG_MODES) or len(set(config_ids)) != len(_CONFIG_MODES):
        raise EvaluationVerificationError("evaluation run must contain four unique configs")
    configs = tuple(repository.get_retrieval_config(config_id) for config_id in config_ids)
    if tuple(config.mode for config in configs) != _CONFIG_MODES:
        raise EvaluationVerificationError("evaluation configs are not the canonical ordered modes")

    query_set, queries = repository.get_query_set(run.query_set.id)
    query_ids = tuple(query.id for query in queries)
    expected_query_set = QuerySetSummary(
        id=query_set.id,
        name=query_set.name,
        version=query_set.version,
        query_count=query_set.query_count,
        content_hash=query_set.content_hash,
    )
    if (
        run.query_set != expected_query_set
        or run.total_queries != query_set.query_count
        or query_set.dataset_version_id != configs[0].dataset_version_id
        or any(config.dataset_version_id != query_set.dataset_version_id for config in configs)
        or query_set.query_count != _QUERY_COUNT
        or len(query_ids) != _QUERY_COUNT
        or len(set(query_ids)) != _QUERY_COUNT
    ):
        raise EvaluationVerificationError("evaluation query-set binding or coverage is invalid")

    outcomes = repository.list_outcomes(run_id)
    if len(outcomes) != _OUTCOME_COUNT:
        raise EvaluationVerificationError("evaluation must contain exactly 200 outcomes")
    try:
        recomputed = finalize_durable_outcomes(run, outcomes, query_ids=query_ids)
    except (TypeError, ValueError):
        raise EvaluationVerificationError(
            "durable outcome coverage or payload is invalid"
        ) from None
    if run.summaries != recomputed:
        raise EvaluationVerificationError("persisted summaries do not match durable outcomes")
    for summary in recomputed:
        if summary.completed_queries != _QUERY_COUNT or summary.failed_queries != 0:
            raise EvaluationVerificationError("evaluation must have 200 successful outcomes")
        if tuple(metric.name for metric in summary.metrics) != _METRIC_ORDER:
            raise EvaluationVerificationError("evaluation summary metric order is invalid")
        if any(metric.sample_count != _QUERY_COUNT for metric in summary.metrics):
            raise EvaluationVerificationError("evaluation summary sample coverage is invalid")
        error_rate = summary.metrics[-1]
        if error_rate.name is not MetricName.ERROR_RATE or error_rate.value != 0.0:
            raise EvaluationVerificationError("evaluation error rate must be zero")

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
        query_set_id=query_set.id,
        query_count=len(query_ids),
        outcome_count=len(outcomes),
        summaries=tuple(recomputed),
        export_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def verify_evaluation(
    run_id: UUID,
    *,
    settings: Settings | None = None,
) -> VerifiedEvaluation:
    """Read and independently verify one run without migrating or writing its database."""
    resolved_settings = settings or Settings()
    with _read_only_repository(resolved_settings.database_path) as repository:
        return _verify_repository(repository, run_id)


def _metric_value(summary: ConfigRunSummary, name: str) -> str:
    metric = next(metric for metric in summary.metrics if metric.name.value == name)
    value = "null" if metric.value is None else format(metric.value, ".12g")
    return f"{value}[n={metric.sample_count}]"


def render_report(report: VerifiedEvaluation) -> tuple[str, ...]:
    lines = [
        f"run_id={report.run_id} status=completed",
        f"query_set_id={report.query_set_id} query_count={report.query_count} "
        f"config_count={len(report.summaries)} outcome_count={report.outcome_count}",
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


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
) -> int:
    """Accept only one run UUID and never print stored payloads or exception details."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID, help="completed evaluation run UUID")
    arguments = parser.parse_args(argv)
    try:
        report = verify_evaluation(arguments.run_id, settings=settings)
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
