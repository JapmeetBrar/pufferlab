"""Idempotently materialize the authored synthetic demo through durable repository writes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid5

from pufferlab.contracts.datasets import (
    DataOrigin,
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import (
    EvalOutcomeWarning,
    EvalRun,
    EvalRunExport,
    EvalRunStatus,
    EvalSuccessPayload,
    PerQueryMetrics,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
    TimingSource,
)
from pufferlab.contracts.retrieval import RetrievalConfig, RetrievalMode
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID, corpus_hash
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import Judgment
from pufferlab.jobs.eval_runner import (
    encode_outcome_payload,
    export_outcome_record,
    finalize_durable_outcomes,
)
from pufferlab.persistence.canonical import canonical_json
from pufferlab.persistence.errors import RecordNotFoundError
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.persistence.types import QueryOutcome, QueryOutcomeStatus
from pufferlab.retrieval.config import bind_retrieval_catalog
from pufferlab.synthetic_demo.authored import (
    AUTHORED_SYNTHETIC_DEMO,
    SYNTHETIC_DEMO_CREATED_AT,
    AuthoredSyntheticDemo,
)

_RUN_STARTED_AT = SYNTHETIC_DEMO_CREATED_AT + timedelta(seconds=1)
_RUN_COMPLETED_AT = SYNTHETIC_DEMO_CREATED_AT + timedelta(seconds=2)
_CONFIG_MODE_ORDER = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)


class SyntheticDemoSeedError(RuntimeError):
    """A safe failure when existing state differs from the immutable authored demo."""


@dataclass(frozen=True, slots=True)
class SyntheticDemoMaterialization:
    """Pure deterministic contracts derived from the checked-in authored inputs."""

    dataset_version: DatasetVersion
    query_set: QuerySet
    configs: tuple[RetrievalConfig, ...]
    queued_run: EvalRun
    completed_run: EvalRun
    outcomes: tuple[QueryOutcome, ...]
    export: EvalRunExport


@dataclass(frozen=True, slots=True)
class SyntheticDemoSeedResult:
    """Exact durable identities and canonical export representation after seeding."""

    dataset_version: DatasetVersion
    query_set: QuerySet
    configs: tuple[RetrievalConfig, ...]
    run: EvalRun
    export: EvalRunExport

    @property
    def canonical_export_bytes(self) -> bytes:
        return f"{canonical_json(self.export)}\n".encode()


def materialize_synthetic_demo(
    authored: AuthoredSyntheticDemo = AUTHORED_SYNTHETIC_DEMO,
) -> SyntheticDemoMaterialization:
    """Derive identities, metrics, summaries, and export without persistence or provider work."""
    dataset = _dataset_version(authored)
    bound = bind_retrieval_catalog(dataset, authored.manifest)
    configs = bound.configs
    if tuple(config.mode for config in configs) != _CONFIG_MODE_ORDER:
        raise SyntheticDemoSeedError("synthetic config compiler returned an unexpected order")

    query_set = _query_set(authored, dataset_id=dataset.id)
    run_id = _run_id(authored, dataset=dataset, query_set=query_set, configs=configs)
    queued_run = EvalRun(
        id=run_id,
        status=EvalRunStatus.QUEUED,
        query_set=QuerySetSummary(
            id=query_set.id,
            name=query_set.name,
            version=query_set.version,
            query_count=query_set.query_count,
            content_hash=query_set.content_hash,
        ),
        baseline_config_id=configs[0].id,
        candidate_config_ids=[config.id for config in configs[1:]],
        summaries=[],
        completed_queries=0,
        total_queries=50,
        random_seed=20260822,
        environment=RunEnvironment(
            pufferlab_git_revision="synthetic-demo-v1",
            turbopuffer_region="not_applicable",
            python_version="not_applicable",
            platform="not_applicable",
            max_concurrency=1,
            warmup_query_count=0,
            timing_source=TimingSource.SYNTHETIC_UNAVAILABLE,
            query_embedding_cache_enabled=False,
        ),
        created_at=SYNTHETIC_DEMO_CREATED_AT,
        started_at=None,
        completed_at=None,
        error=None,
    )
    outcomes = _outcomes(authored, run_id=run_id, configs=configs)
    summaries = finalize_durable_outcomes(
        queued_run,
        outcomes,
        query_ids=[item.judged_query.id for item in authored.queries],
    )
    completed_run = EvalRun.model_validate(
        queued_run.model_copy(
            update={
                "status": EvalRunStatus.COMPLETED,
                "summaries": summaries,
                "completed_queries": 50,
                "started_at": _RUN_STARTED_AT,
                "completed_at": _RUN_COMPLETED_AT,
            }
        ).model_dump(mode="python")
    )
    ordered_outcomes = tuple(
        sorted(outcomes, key=lambda item: (str(item.config_id), str(item.query_id)))
    )
    export = EvalRunExport(
        run=completed_run,
        outcomes=[export_outcome_record(outcome) for outcome in ordered_outcomes],
    )
    return SyntheticDemoMaterialization(
        dataset_version=dataset,
        query_set=query_set,
        configs=configs,
        queued_run=queued_run,
        completed_run=completed_run,
        outcomes=ordered_outcomes,
        export=export,
    )


def seed_synthetic_demo(
    repository: PufferLabRepository,
    *,
    authored: AuthoredSyntheticDemo = AUTHORED_SYNTHETIC_DEMO,
) -> SyntheticDemoSeedResult:
    """Create or verify the one immutable offline demo without constructing runtime clients."""
    expected = materialize_synthetic_demo(authored)
    repository.put_dataset_version(expected.dataset_version)
    for config in expected.configs:
        repository.put_retrieval_config(config)
    repository.put_query_set(
        expected.query_set,
        [item.judged_query for item in authored.queries],
    )

    try:
        current = repository.get_run(expected.queued_run.id)
    except RecordNotFoundError:
        current = repository.create_run(expected.queued_run)

    _validate_run_identity(current, expected.queued_run)
    actual_outcomes = repository.list_outcomes(current.id)
    expected_by_identity = {
        (outcome.config_id, outcome.query_id): outcome for outcome in expected.outcomes
    }
    actual_by_identity = {
        (outcome.config_id, outcome.query_id): outcome for outcome in actual_outcomes
    }
    if len(actual_by_identity) != len(actual_outcomes):
        raise SyntheticDemoSeedError("synthetic demo contains duplicate durable outcomes")
    for identity, outcome in actual_by_identity.items():
        if expected_by_identity.get(identity) != outcome:
            raise SyntheticDemoSeedError("existing synthetic outcome differs from authored input")

    if current.status is EvalRunStatus.COMPLETED:
        return _verified_result(repository, expected)
    if current.status is EvalRunStatus.QUEUED:
        if actual_outcomes:
            raise SyntheticDemoSeedError("queued synthetic demo cannot contain outcomes")
        current = repository.transition_run(
            current.id,
            EvalRunStatus.RUNNING,
            at=_RUN_STARTED_AT,
        )
    if current.status is not EvalRunStatus.RUNNING:
        raise SyntheticDemoSeedError("existing synthetic demo is not safely resumable")
    if current.started_at != _RUN_STARTED_AT or current.completed_at is not None:
        raise SyntheticDemoSeedError("existing synthetic run lifecycle differs from authored input")

    for outcome in expected.outcomes:
        identity = (outcome.config_id, outcome.query_id)
        if identity not in actual_by_identity:
            repository.record_outcome(outcome)

    durable = repository.get_run(current.id)
    durable_outcomes = repository.list_outcomes(current.id)
    summaries = finalize_durable_outcomes(
        durable,
        durable_outcomes,
        query_ids=[item.judged_query.id for item in authored.queries],
    )
    repository.complete_run(current.id, summaries, at=_RUN_COMPLETED_AT)
    return _verified_result(repository, expected)


def _verified_result(
    repository: PufferLabRepository,
    expected: SyntheticDemoMaterialization,
) -> SyntheticDemoSeedResult:
    run = repository.get_run(expected.completed_run.id)
    outcomes = sorted(
        repository.list_outcomes(run.id),
        key=lambda item: (str(item.config_id), str(item.query_id)),
    )
    export = EvalRunExport(
        run=run,
        outcomes=[export_outcome_record(outcome) for outcome in outcomes],
    )
    if export != expected.export:
        raise SyntheticDemoSeedError("existing synthetic demo differs from authored input")
    return SyntheticDemoSeedResult(
        dataset_version=expected.dataset_version,
        query_set=expected.query_set,
        configs=expected.configs,
        run=run,
        export=export,
    )


def _dataset_version(authored: AuthoredSyntheticDemo) -> DatasetVersion:
    manifest = authored.manifest
    write_spec = compile_namespace_write_spec(manifest)
    content_hash = corpus_hash(authored.documents)
    identity_hash = _canonical_hash(
        {
            "data_origin": DataOrigin.SYNTHETIC_DEMO.value,
            "manifest": manifest.model_dump(mode="json"),
            "corpus_hash": content_hash,
            "document_count": len(authored.documents),
            "schema_hash": write_spec.schema_hash,
        }
    )
    dataset_id = uuid5(
        PUFFERLAB_NAMESPACE_UUID,
        f"synthetic-demo-dataset:{identity_hash}",
    )
    return DatasetVersion(
        id=dataset_id,
        slug=manifest.slug,
        version=manifest.version,
        data_origin=DataOrigin.SYNTHETIC_DEMO,
        namespace="",
        index_profile=IndexProfile(
            id=f"{manifest.slug}-{write_spec.schema_hash[:16]}",
            embedding_provider=manifest.embedding.provider,
            embedding_model=manifest.embedding.model,
            embedding_revision=manifest.embedding.revision,
            vector_attribute=manifest.vector.attribute,
            vector_dimensions=manifest.embedding.dimensions,
            vector_dtype=manifest.vector.dtype,
            distance_metric=manifest.vector.distance_metric,
            fts_profile=FtsProfile(
                tokenizer=manifest.fts.tokenizer,
                case_sensitive=manifest.fts.case_sensitive,
                language=manifest.fts.language,
                stemming=manifest.fts.stemming,
                remove_stopwords=manifest.fts.remove_stopwords,
                ascii_folding=manifest.fts.ascii_folding,
                max_token_length=manifest.fts.max_token_length,
                k1=manifest.fts.k1,
                b=manifest.fts.b,
                k3=manifest.fts.k3,
            ),
            schema_hash=write_spec.schema_hash,
        ),
        document_count=len(authored.documents),
        corpus_hash=content_hash,
        status=DatasetStatus.READY,
        created_at=SYNTHETIC_DEMO_CREATED_AT,
    )


def _query_set(authored: AuthoredSyntheticDemo, *, dataset_id: UUID) -> QuerySet:
    content_hash = _canonical_hash(
        {
            "format_version": 1,
            "queries": [item.judged_query.model_dump(mode="json") for item in authored.queries],
        }
    )
    return QuerySet(
        id=uuid5(
            PUFFERLAB_NAMESPACE_UUID,
            f"synthetic-demo-query-set:{dataset_id}:{content_hash}",
        ),
        name="PufferLab offline synthetic 50",
        version="synthetic-demo-query-set-v1",
        dataset_version_id=dataset_id,
        query_count=len(authored.queries),
        content_hash=content_hash,
        created_at=SYNTHETIC_DEMO_CREATED_AT,
    )


def _run_id(
    authored: AuthoredSyntheticDemo,
    *,
    dataset: DatasetVersion,
    query_set: QuerySet,
    configs: tuple[RetrievalConfig, ...],
) -> UUID:
    outcome_input_hash = _canonical_hash(
        {
            "dataset_id": str(dataset.id),
            "query_set_id": str(query_set.id),
            "configs": [
                {"id": str(config.id), "config_hash": config.config_hash} for config in configs
            ],
            "rankings": [
                {
                    "query_id": str(item.judged_query.id),
                    "by_mode": {
                        mode.value: [str(document_id) for document_id in ranking]
                        for mode, ranking in item.rankings
                    },
                }
                for item in authored.queries
            ],
        }
    )
    return uuid5(PUFFERLAB_NAMESPACE_UUID, f"synthetic-demo-run:{outcome_input_hash}")


def _outcomes(
    authored: AuthoredSyntheticDemo,
    *,
    run_id: UUID,
    configs: tuple[RetrievalConfig, ...],
) -> tuple[QueryOutcome, ...]:
    outcomes: list[QueryOutcome] = []
    for config in configs:
        for item in authored.queries:
            ranking = item.ranking_for(config.mode)
            evaluated = evaluate_ranking(
                ranking,
                [
                    Judgment(
                        document_id=qrel.document_id,
                        relevance_grade=qrel.relevance_grade,
                    )
                    for qrel in item.judged_query.qrels
                ],
            )
            payload = EvalSuccessPayload(
                ranked_document_ids=list(ranking),
                metrics=PerQueryMetrics(
                    ndcg_at_10=evaluated.ndcg_at_10,
                    recall_at_50=evaluated.recall_at_50,
                    mrr_at_10=evaluated.mrr_at_10,
                ),
                timing_source=TimingSource.SYNTHETIC_UNAVAILABLE,
                total_client_wall_latency_ms=None,
                stage_timings=[],
                candidate_counts={},
                warnings=[
                    EvalOutcomeWarning(code=warning.code.value, message=warning.message)
                    for warning in evaluated.warnings
                ],
                trace_id=None,
            )
            outcomes.append(
                QueryOutcome(
                    run_id=run_id,
                    config_id=config.id,
                    query_id=item.judged_query.id,
                    status=QueryOutcomeStatus.SUCCEEDED,
                    payload=encode_outcome_payload(payload),
                    created_at=_RUN_STARTED_AT,
                )
            )
    return tuple(outcomes)


def _validate_run_identity(actual: EvalRun, expected: EvalRun) -> None:
    ignored = {
        "status",
        "summaries",
        "completed_queries",
        "started_at",
        "completed_at",
        "error",
    }
    if actual.model_dump(mode="json", exclude=ignored) != expected.model_dump(
        mode="json", exclude=ignored
    ):
        raise SyntheticDemoSeedError("existing synthetic run identity differs from authored input")
    if actual.error is not None:
        raise SyntheticDemoSeedError("existing synthetic run contains an unexpected error")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
