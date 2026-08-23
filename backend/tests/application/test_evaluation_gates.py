from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NoReturn
from uuid import UUID, uuid4

import pufferlab.application.evaluation_gates as gate_module
import pytest
from pufferlab.application.evaluation_gates import (
    GateApplicationStatus,
    evaluate_durable_gate,
)
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    EvalFailurePayload,
    EvalOutcomeWarning,
    EvalRun,
    EvalRunStatus,
    EvalSuccessPayload,
    PerQueryMetrics,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.contracts.gates import GatePolicy, GateVerdict
from pufferlab.contracts.retrieval import RetrievalConfig
from pufferlab.datasets.cqadupstack import CuratedQueryManifest, SourceLock
from pufferlab.datasets.unix_application import (
    build_ready_unix_evaluation_seed,
    load_curated_unix_local_pack,
)
from pufferlab.evals.metrics import evaluate_ranking
from pufferlab.evals.models import Judgment
from pufferlab.jobs.eval_runner import (
    encode_outcome_payload,
    finalize_durable_outcomes,
)
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.canonical import canonical_json
from pufferlab.persistence.read_only import ReadOnlyCatalogError
from pufferlab.persistence.types import QueryOutcome, QueryOutcomeStatus
from pufferlab.retrieval.config import derive_bound_retrieval_configs
from pufferlab.synthetic_demo.seeder import SyntheticDemoSeedResult, seed_synthetic_demo
from tests.datasets.test_unix_application import (
    DATASET_MANIFEST_PATH,
    _prepared_curated_pack,
)


def _seed(path: Path) -> SyntheticDemoSeedResult:
    with Database(path) as database:
        database.migrate()
        return seed_synthetic_demo(PufferLabRepository(database.session_factory))


def _snapshot(path: Path) -> tuple[str, tuple[int, int, int, int], tuple[str, ...]]:
    metadata = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),
        tuple(sorted(item.name for item in path.parent.iterdir())),
    )


def _passing_policy() -> GatePolicy:
    return GatePolicy(
        min_delta=-1.0,
        max_query_drop=1.0,
        max_error_rate=1.0,
        min_paired_queries=49,
    )


def _refresh_summaries(path: Path, run_id: UUID) -> EvalRun:
    with Database(path) as database:
        repository = PufferLabRepository(database.session_factory)
        run = repository.get_run(run_id)
        summaries = finalize_durable_outcomes(
            run,
            repository.list_outcomes(run.id),
            query_ids=repository.list_query_ids(run.query_set.id, limit=50),
        )
    forged = run.model_copy(update={"summaries": summaries})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE eval_runs SET payload_json = ? WHERE id = ?",
            (canonical_json(forged), str(run.id)),
        )
    return forged


@dataclass(frozen=True, slots=True)
class _LiveSeed:
    run: EvalRun
    configs: tuple[RetrievalConfig, ...]
    curated_manifest: CuratedQueryManifest
    source_lock: SourceLock


def _seed_live(path: Path, assets: Path) -> _LiveSeed:
    assets.mkdir(parents=True)
    processed, curated_path, source_lock, processed_pack_lock = _prepared_curated_pack(assets)
    local_pack = load_curated_unix_local_pack(
        processed,
        source_lock=source_lock,
        processed_pack_lock=processed_pack_lock,
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        curated_manifest_path=curated_path,
    )
    seed = build_ready_unix_evaluation_seed(local_pack, namespace="test-owned-live-unix")
    curated_manifest = local_pack.curated_manifest.model_copy(
        update={"query_set_content_sha256": seed.query_set.content_hash}
    )
    manifest = gate_module.load_unix_dataset_manifest(DATASET_MANIFEST_PATH)
    configs = derive_bound_retrieval_configs(
        seed.dataset_version,
        manifest,
        namespace=seed.dataset_version.namespace,
    )
    created_at = seed.dataset_version.created_at
    started_at = created_at + timedelta(seconds=1)
    completed_at = created_at + timedelta(seconds=2)
    queued = EvalRun(
        id=uuid4(),
        status=EvalRunStatus.QUEUED,
        query_set=QuerySetSummary(
            id=seed.query_set.id,
            name=seed.query_set.name,
            version=seed.query_set.version,
            query_count=seed.query_set.query_count,
            content_hash=seed.query_set.content_hash,
        ),
        baseline_config_id=configs[0].id,
        candidate_config_ids=[config.id for config in configs[1:]],
        summaries=[],
        completed_queries=0,
        total_queries=50,
        random_seed=20260822,
        environment=RunEnvironment(
            pufferlab_git_revision="test-live-v1",
            turbopuffer_region="test-region",
            python_version="test",
            platform="test",
            max_concurrency=1,
            warmup_query_count=0,
            query_embedding_cache_enabled=False,
        ),
        created_at=created_at,
        started_at=None,
        completed_at=None,
        error=None,
    )
    with Database(path) as database:
        database.migrate()
        repository = PufferLabRepository(database.session_factory)
        repository.put_dataset_version(seed.dataset_version)
        repository.put_query_set(seed.query_set, seed.judged_queries)
        for config in configs:
            repository.put_retrieval_config(config)
        repository.create_run(queued)
        repository.transition_run(queued.id, EvalRunStatus.RUNNING, at=started_at)
        for config in configs:
            for query in seed.judged_queries:
                ranking = [qrel.document_id for qrel in query.qrels]
                metrics = evaluate_ranking(
                    ranking,
                    tuple(
                        Judgment(
                            document_id=qrel.document_id,
                            relevance_grade=qrel.relevance_grade,
                        )
                        for qrel in query.qrels
                    ),
                )
                payload = EvalSuccessPayload(
                    ranked_document_ids=ranking,
                    metrics=PerQueryMetrics(
                        ndcg_at_10=metrics.ndcg_at_10,
                        recall_at_50=metrics.recall_at_50,
                        mrr_at_10=metrics.mrr_at_10,
                    ),
                    total_client_wall_latency_ms=1.0,
                    stage_timings=[],
                    candidate_counts={},
                    warnings=[
                        EvalOutcomeWarning(code=warning.code.value, message=warning.message)
                        for warning in metrics.warnings
                    ],
                    trace_id=uuid4(),
                )
                repository.record_outcome(
                    QueryOutcome(
                        run_id=queued.id,
                        config_id=config.id,
                        query_id=query.id,
                        status=QueryOutcomeStatus.SUCCEEDED,
                        payload=encode_outcome_payload(payload),
                        created_at=started_at,
                    )
                )
        run = repository.get_run(queued.id)
        summaries = finalize_durable_outcomes(
            run,
            repository.list_outcomes(run.id),
            query_ids=[query.id for query in seed.judged_queries],
        )
        run = repository.complete_run(run.id, summaries, at=completed_at)
    return _LiveSeed(
        run=run,
        configs=configs,
        curated_manifest=curated_manifest,
        source_lock=source_lock,
    )


def test_synthetic_gate_passes_only_after_full_read_only_close(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    before = _snapshot(path)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[-1].id,
        policy=_passing_policy(),
    )

    assert result.status is GateApplicationStatus.REPORT
    assert result.report is not None
    assert result.report.verdict is GateVerdict.PASSED
    assert result.report.run_id == seeded.run.id
    assert _snapshot(path) == before


def test_live_stored_gate_uses_exact_authenticator_with_every_runtime_boundary_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed_live(path, tmp_path / "assets")
    manifest = gate_module.load_unix_dataset_manifest(DATASET_MANIFEST_PATH)
    calls = {"authenticator": 0}
    real_authenticator = gate_module.authenticate_persisted_unix_query_set

    def authenticate(*args: object, **kwargs: object) -> None:
        calls["authenticator"] += 1
        real_authenticator(*args, **kwargs)

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError(
            "Database RuntimeCliApplication EvaluationApiRuntime job provider model search"
        )

    import pufferlab.application.evaluation_runtime as api_runtime
    import pufferlab.cli.runtime as cli_runtime
    import pufferlab.jobs.manager as job_manager
    import pufferlab.persistence.database as database_module
    import pufferlab.providers.rerankers as reranker_module
    import pufferlab.providers.turbopuffer as provider_module
    import pufferlab.retrieval.embeddings as embedding_module
    import pufferlab.retrieval.runtime as search_module

    monkeypatch.setattr(gate_module, "authenticate_persisted_unix_query_set", authenticate)
    monkeypatch.setattr(
        gate_module,
        "load_curated_query_manifest",
        lambda _path: seeded.curated_manifest,
    )
    monkeypatch.setattr(gate_module, "load_source_lock", lambda _path: seeded.source_lock)
    monkeypatch.setattr(gate_module, "load_unix_dataset_manifest", lambda _path: manifest)
    monkeypatch.setattr(database_module, "Database", forbidden)
    monkeypatch.setattr(cli_runtime, "RuntimeCliApplication", forbidden)
    monkeypatch.setattr(api_runtime, "EvaluationApiRuntime", forbidden)
    monkeypatch.setattr(job_manager, "RunJobManager", forbidden)
    monkeypatch.setattr(provider_module, "TurbopufferProvider", forbidden)
    monkeypatch.setattr(reranker_module, "SentenceTransformersReranker", forbidden)
    monkeypatch.setattr(embedding_module, "SentenceTransformerQueryEmbedder", forbidden)
    monkeypatch.setattr(search_module, "RuntimeSearchBackend", forbidden)
    before = _snapshot(path)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=GatePolicy(min_paired_queries=50),
    )

    assert result.status is GateApplicationStatus.REPORT
    assert result.report is not None
    assert result.report.verdict is GateVerdict.PASSED
    assert calls == {"authenticator": 1}
    assert _snapshot(path) == before

    # The same fixed source authentication must reject direct qrel corruption before any verdict.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE qrels SET relevance_grade = relevance_grade + 1 WHERE rowid = ("
            "SELECT rowid FROM qrels LIMIT 1)"
        )
    corrupted = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=GatePolicy(min_paired_queries=50),
    )
    assert corrupted.status is GateApplicationStatus.INVALID
    assert corrupted.report is None
    assert calls == {"authenticator": 2}


def test_live_typed_failure_is_valid_evidence_and_reduces_paired_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed_live(path, tmp_path / "assets")
    manifest = gate_module.load_unix_dataset_manifest(DATASET_MANIFEST_PATH)
    monkeypatch.setattr(
        gate_module,
        "load_curated_query_manifest",
        lambda _path: seeded.curated_manifest,
    )
    monkeypatch.setattr(gate_module, "load_source_lock", lambda _path: seeded.source_lock)
    monkeypatch.setattr(gate_module, "load_unix_dataset_manifest", lambda _path: manifest)
    candidate_id = seeded.configs[1].id
    failure = EvalFailurePayload(
        code=ApiErrorCode.NAMESPACE_NOT_READY,
        message="retrieval was unavailable",
        retryable=True,
        operation="search_one",
        trace_id=uuid4(),
        total_client_wall_latency_ms=1.0,
    )
    with sqlite3.connect(path) as connection:
        rowid, encoded = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes "
            "WHERE run_id = ? AND config_id = ? ORDER BY query_id LIMIT 1",
            (str(seeded.run.id), str(candidate_id)),
        ).fetchone()
        value = json.loads(encoded)
        value["status"] = "failed"
        value["payload"] = encode_outcome_payload(failure)
        connection.execute(
            "UPDATE query_outcomes SET status = 'failed', payload_json = ? WHERE rowid = ?",
            (json.dumps(value, sort_keys=True, separators=(",", ":")), rowid),
        )
    _refresh_summaries(path, seeded.run.id)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=candidate_id,
        policy=GatePolicy(
            min_delta=-1.0,
            max_query_drop=1.0,
            max_error_rate=0.02,
            min_paired_queries=49,
        ),
    )

    assert result.status is GateApplicationStatus.REPORT
    assert result.report is not None
    assert result.report.verdict is GateVerdict.PASSED
    error_rate, paired_coverage = result.report.checks[:2]
    assert error_rate.failed_candidate_queries == 1
    assert error_rate.sample_count == 50
    assert paired_coverage.paired_query_count == 49
    assert paired_coverage.excluded_query_count == 1


def test_missing_symlink_and_fifo_catalogs_fail_without_creation_or_blocking(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-parent-marker" / "catalog.sqlite3"
    real = tmp_path / "real.sqlite3"
    _seed(real)
    symlink = tmp_path / "linked.sqlite3"
    symlink.symlink_to(real)
    fifo = tmp_path / "catalog.fifo"
    os.mkfifo(fifo)

    for path in (missing, symlink, fifo):
        result = evaluate_durable_gate(
            database_path=path,
            run_id=uuid4(),
            candidate_config_id=uuid4(),
            policy=GatePolicy(),
        )
        assert result.status is GateApplicationStatus.INVALID
        assert result.report is None
    assert not missing.parent.exists()
    assert symlink.is_symlink()
    assert fifo.exists()


def test_unknown_candidate_is_invalid_evidence_not_policy_failure(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=uuid4(),
        policy=_passing_policy(),
    )

    assert result == gate_module.GateApplicationResult(status=GateApplicationStatus.INVALID)


def test_compensating_metric_tamper_and_recomputed_summary_still_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    config_id = seeded.configs[1].id
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT query_id, payload_json FROM query_outcomes "
            "WHERE run_id = ? AND config_id = ? ORDER BY query_id LIMIT 2",
            (str(seeded.run.id), str(config_id)),
        ).fetchall()
        for index, (query_id, encoded) in enumerate(rows):
            value = json.loads(encoded)
            current = float(value["payload"]["metrics"]["ndcg_at_10"])
            replacement = current + (0.001 if index == 0 else -0.001)
            assert 0.0 <= replacement <= 1.0
            value["payload"]["metrics"]["ndcg_at_10"] = replacement
            connection.execute(
                "UPDATE query_outcomes SET payload_json = ? "
                "WHERE run_id = ? AND config_id = ? AND query_id = ?",
                (
                    json.dumps(value, sort_keys=True, separators=(",", ":")),
                    str(seeded.run.id),
                    str(config_id),
                    query_id,
                ),
            )

    # Make the aggregate summary self-consistent with the forged per-query values. The adapter
    # must still reject because each success is independently recomputed from rankings and qrels.
    run = _refresh_summaries(path, seeded.run.id)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=run.id,
        candidate_config_id=config_id,
        policy=_passing_policy(),
    )
    assert result.status is GateApplicationStatus.INVALID


def test_self_consistent_ranked_id_change_is_explicitly_trusted_local_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    config_id = seeded.configs[1].id
    with sqlite3.connect(path) as connection:
        query_id, encoded = connection.execute(
            "SELECT query_id, payload_json FROM query_outcomes "
            "WHERE run_id = ? AND config_id = ? ORDER BY query_id LIMIT 1",
            (str(seeded.run.id), str(config_id)),
        ).fetchone()
        value = json.loads(encoded)
        ranking = value["payload"]["ranked_document_ids"]
        ranking[0], ranking[1] = ranking[1], ranking[0]

    with Database(path) as database:
        query = PufferLabRepository(database.session_factory).get_judged_query(
            seeded.query_set.id,
            uuid4() if query_id is None else UUID(query_id),
        )
    metrics = evaluate_ranking(
        [UUID(document_id) for document_id in ranking],
        tuple(
            Judgment(document_id=qrel.document_id, relevance_grade=qrel.relevance_grade)
            for qrel in query.qrels
        ),
    )
    value["payload"]["metrics"] = {
        "ndcg_at_10": metrics.ndcg_at_10,
        "recall_at_50": metrics.recall_at_50,
        "mrr_at_10": metrics.mrr_at_10,
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE query_outcomes SET payload_json = ? "
            "WHERE run_id = ? AND config_id = ? AND query_id = ?",
            (
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                str(seeded.run.id),
                str(config_id),
                query_id,
            ),
        )
    _refresh_summaries(path, seeded.run.id)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=config_id,
        policy=_passing_policy(),
    )
    assert result.status is GateApplicationStatus.REPORT
    assert result.report is not None


@pytest.mark.parametrize(
    "attack",
    [
        "false_summary",
        "query_text",
        "qrel_grade",
        "config_payload",
        "run_config_role",
        "run_config_ordinal",
        "outcome_timestamp",
        "nonfinite_metric",
        "extra_query",
        "extra_outcome",
    ],
)
def test_structural_source_config_lifecycle_and_outcome_corruption_is_invalid(
    tmp_path: Path,
    attack: str,
) -> None:
    path = tmp_path / attack / "catalog.sqlite3"
    seeded = _seed(path)
    with sqlite3.connect(path) as connection:
        if attack == "false_summary":
            value = json.loads(
                connection.execute(
                    "SELECT payload_json FROM eval_runs WHERE id = ?",
                    (str(seeded.run.id),),
                ).fetchone()[0]
            )
            value["summaries"][0]["metrics"][0]["value"] = 0.123456789
            connection.execute(
                "UPDATE eval_runs SET payload_json = ? WHERE id = ?",
                (json.dumps(value), str(seeded.run.id)),
            )
        elif attack == "query_text":
            rowid, encoded = connection.execute(
                "SELECT rowid, payload_json FROM judged_queries LIMIT 1"
            ).fetchone()
            value = json.loads(encoded)
            value["text"] = "mutated-source-query-marker"
            connection.execute(
                "UPDATE judged_queries SET payload_json = ? WHERE rowid = ?",
                (json.dumps(value), rowid),
            )
        elif attack == "qrel_grade":
            connection.execute(
                "UPDATE qrels SET relevance_grade = relevance_grade + 1 WHERE rowid = ("
                "SELECT rowid FROM qrels LIMIT 1)"
            )
        elif attack == "config_payload":
            rowid, encoded = connection.execute(
                "SELECT rowid, payload_json FROM retrieval_configs LIMIT 1"
            ).fetchone()
            value = json.loads(encoded)
            value["name"] = "mutated config"
            connection.execute(
                "UPDATE retrieval_configs SET name = ?, payload_json = ? WHERE rowid = ?",
                (value["name"], json.dumps(value), rowid),
            )
        elif attack == "run_config_role":
            connection.execute(
                "UPDATE run_configs SET role = 'candidate' WHERE run_id = ? AND ordinal = 0",
                (str(seeded.run.id),),
            )
        elif attack == "run_config_ordinal":
            connection.execute(
                "UPDATE run_configs SET ordinal = 9 WHERE run_id = ? AND ordinal = 0",
                (str(seeded.run.id),),
            )
        elif attack == "outcome_timestamp":
            rowid, encoded = connection.execute(
                "SELECT rowid, payload_json FROM query_outcomes LIMIT 1"
            ).fetchone()
            value = json.loads(encoded)
            value["created_at"] = "2000-01-01T00:00:00Z"
            connection.execute(
                "UPDATE query_outcomes SET created_at = ?, payload_json = ? WHERE rowid = ?",
                (value["created_at"], json.dumps(value), rowid),
            )
        elif attack == "nonfinite_metric":
            rowid, encoded = connection.execute(
                "SELECT rowid, payload_json FROM query_outcomes LIMIT 1"
            ).fetchone()
            value = json.loads(encoded)
            value["payload"]["metrics"]["ndcg_at_10"] = float("nan")
            connection.execute(
                "UPDATE query_outcomes SET payload_json = ? WHERE rowid = ?",
                (json.dumps(value), rowid),
            )
        elif attack == "extra_query":
            encoded = connection.execute(
                "SELECT payload_json FROM judged_queries LIMIT 1"
            ).fetchone()[0]
            value = json.loads(encoded)
            foreign_id = uuid4()
            value["id"] = str(foreign_id)
            value["external_id"] = "foreign-query-marker"
            connection.execute(
                "INSERT INTO judged_queries (query_set_id, query_id, ordinal, payload_json) "
                "VALUES (?, ?, 50, ?)",
                (str(seeded.query_set.id), str(foreign_id), json.dumps(value)),
            )
        else:
            encoded, created_at = connection.execute(
                "SELECT payload_json, created_at FROM query_outcomes LIMIT 1"
            ).fetchone()
            value = json.loads(encoded)
            foreign_id = uuid4()
            value["query_id"] = str(foreign_id)
            connection.execute(
                "INSERT INTO query_outcomes "
                "(run_id, config_id, query_id, status, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    value["run_id"],
                    value["config_id"],
                    str(foreign_id),
                    value["status"],
                    created_at,
                    json.dumps(value),
                ),
            )

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=_passing_policy(),
    )
    assert result.status is GateApplicationStatus.INVALID
    assert result.report is None


def test_unselected_candidate_payload_corruption_invalidates_selected_verdict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    selected = seeded.configs[1].id
    unselected = seeded.configs[2].id
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes "
            "WHERE run_id = ? AND config_id = ? ORDER BY query_id",
            (str(seeded.run.id), str(unselected)),
        ).fetchall()
        rowid, encoded = next(
            (rowid, encoded)
            for rowid, encoded in rows
            if json.loads(encoded)["payload"]["metrics"]["ndcg_at_10"] is not None
        )
        value = json.loads(encoded)
        value["payload"]["ranked_document_ids"].reverse()
        connection.execute(
            "UPDATE query_outcomes SET payload_json = ? WHERE rowid = ?",
            (json.dumps(value), rowid),
        )

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=selected,
        policy=_passing_policy(),
    )
    assert result.status is GateApplicationStatus.INVALID


def test_noncompleted_durable_run_is_invalid_evidence(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    with sqlite3.connect(path) as connection:
        value = json.loads(
            connection.execute(
                "SELECT payload_json FROM eval_runs WHERE id = ?",
                (str(seeded.run.id),),
            ).fetchone()[0]
        )
        value["status"] = "running"
        value["completed_at"] = None
        value["summaries"] = []
        connection.execute(
            "UPDATE eval_runs SET status = 'running', completed_at = NULL, payload_json = ? "
            "WHERE id = ?",
            (json.dumps(value), str(seeded.run.id)),
        )

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=_passing_policy(),
    )
    assert result.status is GateApplicationStatus.INVALID
    assert result.report is None


def test_missing_outcome_and_hidden_same_dataset_config_are_invalid(tmp_path: Path) -> None:
    for attack in ("missing_outcome", "extra_config"):
        path = tmp_path / attack / "catalog.sqlite3"
        seeded = _seed(path)
        with sqlite3.connect(path) as connection:
            if attack == "missing_outcome":
                connection.execute(
                    "DELETE FROM query_outcomes WHERE rowid = ("
                    "SELECT rowid FROM query_outcomes WHERE run_id = ? LIMIT 1)",
                    (str(seeded.run.id),),
                )
            else:
                source = seeded.configs[0]
                foreign_id = uuid4()
                payload = source.model_copy(update={"id": foreign_id})
                connection.execute(
                    "INSERT INTO retrieval_configs "
                    "(id, revision, dataset_version_id, name, config_hash, created_at, "
                    "payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(foreign_id),
                        payload.revision,
                        str(payload.dataset_version_id),
                        payload.name,
                        payload.config_hash,
                        payload.created_at.isoformat().replace("+00:00", "Z"),
                        canonical_json(payload),
                    ),
                )

        result = evaluate_durable_gate(
            database_path=path,
            run_id=seeded.run.id,
            candidate_config_id=seeded.configs[1].id,
            policy=_passing_policy(),
        )
        assert result.status is GateApplicationStatus.INVALID


def test_close_failure_discards_computed_report_and_closes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    real_open = gate_module.open_existing_read_only_catalog
    close_calls = 0

    class FailingCloseCatalog:
        def __init__(self) -> None:
            self._inner = real_open(path)
            self.repository = self._inner.repository

        def close(self) -> NoReturn:
            nonlocal close_calls
            close_calls += 1
            self._inner.close()
            raise ReadOnlyCatalogError()

    monkeypatch.setattr(
        gate_module,
        "open_existing_read_only_catalog",
        lambda _path: FailingCloseCatalog(),
    )
    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[-1].id,
        policy=_passing_policy(),
    )

    assert close_calls == 1
    assert result.status is GateApplicationStatus.INVALID
    assert result.report is None


@pytest.mark.parametrize("operation_control", [KeyboardInterrupt, SystemExit])
def test_operation_control_is_not_downgraded_by_close_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_control: type[BaseException],
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    real_open = gate_module.open_existing_read_only_catalog
    close_calls = 0

    class FailingCloseCatalog:
        def __init__(self) -> None:
            self._inner = real_open(path)
            self.repository = self._inner.repository

        def close(self) -> NoReturn:
            nonlocal close_calls
            close_calls += 1
            self._inner.close()
            raise ReadOnlyCatalogError()

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise operation_control("query-provider-secret-marker")

    monkeypatch.setattr(
        gate_module,
        "open_existing_read_only_catalog",
        lambda _path: FailingCloseCatalog(),
    )
    monkeypatch.setattr(gate_module, "_recomputed_outcome", fail)
    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=_passing_policy(),
    )

    assert close_calls == 1
    assert result.status is GateApplicationStatus.INTERNAL
    assert result.report is None


@pytest.mark.parametrize("close_control", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("operation", ["report", "invalid", "internal"])
def test_close_control_always_upgrades_to_internal_and_closes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_control: type[BaseException],
    operation: str,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    real_open = gate_module.open_existing_read_only_catalog
    close_calls = 0
    close_error = close_control("close-query-provider-secret-marker")

    class ControlledCloseCatalog:
        def __init__(self) -> None:
            self._inner = real_open(path)
            self.repository = self._inner.repository

        def close(self) -> NoReturn:
            nonlocal close_calls
            close_calls += 1
            self._inner.close()
            raise close_error

    monkeypatch.setattr(
        gate_module,
        "open_existing_read_only_catalog",
        lambda _path: ControlledCloseCatalog(),
    )
    candidate = seeded.configs[-1].id
    if operation == "invalid":
        candidate = uuid4()
    elif operation == "internal":

        def fail(*_args: object, **_kwargs: object) -> NoReturn:
            raise KeyboardInterrupt("operation-query-provider-secret-marker")

        monkeypatch.setattr(gate_module, "_recomputed_outcome", fail)

    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=candidate,
        policy=_passing_policy(),
    )

    assert close_calls == 1
    assert result.status is GateApplicationStatus.INTERNAL
    assert result.report is None
    assert close_error.__traceback__ is None
    assert close_error.__context__ is None
    assert close_error.__cause__ is None


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_sensitive_process_control_is_detached_and_catalog_still_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    path = tmp_path / "catalog.sqlite3"
    seeded = _seed(path)
    marker = "query-and-provider-secret-marker"
    caught = control(marker)

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise caught

    monkeypatch.setattr(gate_module, "_recomputed_outcome", fail)
    before = _snapshot(path)
    result = evaluate_durable_gate(
        database_path=path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[1].id,
        policy=_passing_policy(),
    )

    assert result.status is GateApplicationStatus.INTERNAL
    assert marker not in repr(result)
    assert caught.__traceback__ is None
    assert caught.__context__ is None
    assert caught.__cause__ is None
    assert _snapshot(path) == before
