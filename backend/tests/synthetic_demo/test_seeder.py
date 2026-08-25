from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import fields, replace
from pathlib import Path
from uuid import UUID

from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.evals import EvalRunStatus, EvalSuccessPayload, MetricName, TimingSource
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.jobs import decode_outcome_payload
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.canonical import canonical_json
from pufferlab.synthetic_demo import AUTHORED_SYNTHETIC_DEMO, AuthoredSyntheticQuery
from pufferlab.synthetic_demo.seeder import (
    SyntheticDemoSeedResult,
    materialize_synthetic_demo,
    seed_synthetic_demo,
)

_DATASET_ID = UUID("0dc8c34d-7298-54da-ae8f-248394cd1cf4")
_QUERY_SET_ID = UUID("2a0dc5ae-7f0e-50ff-a915-f052df276dab")
_RUN_ID = UUID("063e0537-615c-59c5-a970-4c3b20459e17")
_CONFIG_IDS = (
    UUID("1419db8e-2635-5d27-87b8-6ec8b556573d"),
    UUID("96c96db7-2bb9-5655-98ec-900ca7f44c45"),
    UUID("270c24f1-10b7-55a2-85d1-eeb7c51e2d09"),
    UUID("3a92ded8-f266-5d5e-a8d4-945afae84328"),
)


def _seed(path: Path) -> SyntheticDemoSeedResult:
    database = Database(path)
    database.migrate()
    try:
        result = seed_synthetic_demo(PufferLabRepository(database.session_factory))
    finally:
        database.dispose()
    return result


def test_clean_database_seeds_exact_provider_free_completed_shape(tmp_path: Path) -> None:
    database_path = tmp_path / "new-data" / "pufferlab.sqlite3"
    assert not database_path.parent.exists()

    result = _seed(database_path)

    assert result.dataset_version.data_origin is DataOrigin.SYNTHETIC_DEMO
    assert result.dataset_version.namespace == ""
    assert result.dataset_version.document_count == 60
    assert result.query_set.query_count == 50
    assert tuple(config.mode for config in result.configs) == (
        RetrievalMode.BM25,
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    )
    assert result.run.status is EvalRunStatus.COMPLETED
    assert result.run.completed_queries == 50
    assert result.run.environment.timing_source is TimingSource.SYNTHETIC_UNAVAILABLE
    assert len(result.export.outcomes) == 200
    assert all(record.outcome.kind == "success" for record in result.export.outcomes)
    assert [summary.config_id for summary in result.run.summaries] == [
        config.id for config in result.configs
    ]
    for summary in result.run.summaries:
        assert [metric.name for metric in summary.metrics] == list(MetricName)
        assert summary.completed_queries == 50
        assert summary.failed_queries == 0
        for metric in summary.metrics:
            if metric.name in {MetricName.LATENCY_P50_MS, MetricName.LATENCY_P95_MS}:
                assert metric.value is None
                assert metric.sample_count == 0
            elif metric.name is MetricName.ERROR_RATE:
                assert metric.value == 0.0
                assert metric.sample_count == 50
            else:
                assert metric.value is not None
                assert metric.sample_count == 49

    for record in result.export.outcomes:
        payload = record.outcome
        assert isinstance(payload, EvalSuccessPayload)
        assert payload.timing_source is TimingSource.SYNTHETIC_UNAVAILABLE
        assert payload.total_client_wall_latency_ms is None
        assert payload.stage_timings == []
        assert payload.trace_id is None
        assert payload.candidate_counts == {}


def test_content_addressed_identities_match_reviewed_golden_values() -> None:
    materialized = materialize_synthetic_demo()

    assert materialized.dataset_version.id == _DATASET_ID
    assert materialized.dataset_version.corpus_hash == (
        "bf4b3da16ee2c5ebefc3e11e983b10b2533d3788f995f54f60ab4b5f05686038"
    )
    assert materialized.dataset_version.index_profile.schema_hash == (
        "0251f57f6166bf8f1ab8351ae0a4a797cfcf691fb0699bcfc59a4083945eea1d"
    )
    assert materialized.query_set.id == _QUERY_SET_ID
    assert materialized.query_set.content_hash == (
        "a35834dab656d0a74ac0b711500480115949b7e1ff43d8eecdc8c7823c259127"
    )
    assert materialized.completed_run.id == _RUN_ID
    assert tuple(config.id for config in materialized.configs) == _CONFIG_IDS
    assert hashlib.sha256(f"{canonical_json(materialized.export)}\n".encode()).hexdigest() == (
        "b3bff5d09321ba535dd5bb09a95174a7d89e37a3a43720eabcdde8a33ac2b02e"
    )


def test_rerun_is_byte_identical_and_does_not_duplicate_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "pufferlab.sqlite3"
    first = _seed(database_path)
    first_database_bytes = database_path.read_bytes()
    second = _seed(database_path)

    assert second.canonical_export_bytes == first.canonical_export_bytes
    # SQLite may update internal bookkeeping during idempotent reads, so table identities—not raw
    # database bytes—are the durable idempotence boundary. The first bytes remain useful evidence
    # that the initial seed produced a real database rather than a tracked export fixture.
    assert first_database_bytes
    with sqlite3.connect(database_path) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "dataset_versions",
                "retrieval_configs",
                "query_sets",
                "judged_queries",
                "judged_document_titles",
                "qrels",
                "eval_runs",
                "run_configs",
                "query_outcomes",
            )
        }
        stored_titles = {
            row[0]
            for row in connection.execute("SELECT title FROM judged_document_titles").fetchall()
        }
    assert counts == {
        "dataset_versions": 1,
        "retrieval_configs": 4,
        "query_sets": 1,
        "judged_queries": 50,
        "judged_document_titles": 60,
        "qrels": 100,
        "eval_runs": 1,
        "run_configs": 4,
        "query_outcomes": 200,
    }
    assert "Synthetic troubleshooting note 001" in stored_titles
    assert "Synthetic troubleshooting note 060" in stored_titles


def test_quality_is_recomputed_from_authored_ranks_and_qrels() -> None:
    original = materialize_synthetic_demo()
    authored_query = AUTHORED_SYNTHETIC_DEMO.queries[0]
    original_ranking = authored_query.ranking_for(RetrievalMode.BM25)
    reordered_ranking = (
        original_ranking[-1],
        *original_ranking[1:-1],
        original_ranking[0],
    )
    reordered_query = replace(
        authored_query,
        rankings=tuple(
            (mode, reordered_ranking if mode is RetrievalMode.BM25 else ranking)
            for mode, ranking in authored_query.rankings
        ),
    )
    reordered_authored = replace(
        AUTHORED_SYNTHETIC_DEMO,
        queries=(reordered_query, *AUTHORED_SYNTHETIC_DEMO.queries[1:]),
    )
    reordered = materialize_synthetic_demo(reordered_authored)

    original_config_id = original.configs[0].id
    reordered_config_id = reordered.configs[0].id
    query_id = authored_query.judged_query.id
    original_payload = decode_outcome_payload(
        next(
            outcome
            for outcome in original.outcomes
            if outcome.config_id == original_config_id and outcome.query_id == query_id
        )
    )
    reordered_payload = decode_outcome_payload(
        next(
            outcome
            for outcome in reordered.outcomes
            if outcome.config_id == reordered_config_id and outcome.query_id == query_id
        )
    )
    assert isinstance(original_payload, EvalSuccessPayload)
    assert isinstance(reordered_payload, EvalSuccessPayload)
    assert reordered_payload.metrics.ndcg_at_10 != original_payload.metrics.ndcg_at_10
    assert reordered.completed_run.id != original.completed_run.id


def test_no_positive_qrels_uses_normal_null_metric_warning() -> None:
    materialized = materialize_synthetic_demo()
    query_id = AUTHORED_SYNTHETIC_DEMO.queries[-1].judged_query.id
    payloads = [
        decode_outcome_payload(outcome)
        for outcome in materialized.outcomes
        if outcome.query_id == query_id
    ]

    assert len(payloads) == 4
    for payload in payloads:
        assert isinstance(payload, EvalSuccessPayload)
        assert payload.metrics.ndcg_at_10 is None
        assert payload.metrics.recall_at_50 is None
        assert payload.metrics.mrr_at_10 is None
        assert [warning.code for warning in payload.warnings] == ["no_positive_qrels"]


def test_authored_input_has_only_judgments_and_rankings_not_results() -> None:
    field_names = {field.name for field in fields(AuthoredSyntheticQuery)}
    assert field_names == {"judged_query", "rankings"}
    forbidden = {"metrics", "summaries", "latency", "timings", "trace_id", "candidate_counts"}
    assert field_names.isdisjoint(forbidden)
    assert all(len(item.judged_query.qrels) == 2 for item in AUTHORED_SYNTHETIC_DEMO.queries)
    assert all(
        len(ranking) == 50
        for item in AUTHORED_SYNTHETIC_DEMO.queries
        for _mode, ranking in item.rankings
    )
