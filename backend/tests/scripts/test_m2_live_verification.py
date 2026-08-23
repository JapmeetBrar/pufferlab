from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pufferlab.config import Settings
from pufferlab.contracts.common import JsonValue
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    EvalFailurePayload,
    EvalRun,
    EvalRunStatus,
    EvalSuccessPayload,
    JudgedQuery,
    PerQueryMetrics,
    Qrel,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.jobs import encode_outcome_payload, finalize_durable_outcomes
from pufferlab.persistence import Database, PufferLabRepository, QueryOutcome, QueryOutcomeStatus
from pufferlab.persistence.canonical import canonical_json
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.providers.rerankers import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION
from pufferlab.providers.types import (
    ProviderDeleteResult,
    ProviderNamespaceMetadata,
)

from scripts import m2_live_namespace_session, verify_m2_evaluation

_TEST_NAMESPACE = UUID("147c12c2-7938-4711-8d40-d4659dc92767")
_FIXED_TIME = datetime(2026, 8, 22, 19, 0, tzinfo=UTC)
_SAFE_SESSION_NAMESPACE = "pufferlab-unix-live-" + "a" * 24


def _id(name: str) -> UUID:
    return uuid5(_TEST_NAMESPACE, name)


def _provider_error(code: ApiErrorCode, operation: str) -> ProviderError:
    return ProviderError(
        "redacted provider failure",
        ProviderErrorDetails(
            code=code,
            retryable=False,
            operation=operation,
            status_code=404 if code is ApiErrorCode.NOT_FOUND else 500,
        ),
    )


class _CleanupProvider:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.metadata_calls: list[str] = []
        self.closed = False
        self.delete_failure: Exception | None = None
        self.metadata_failures: list[Exception | None] = [
            None,
            _provider_error(ApiErrorCode.NOT_FOUND, "metadata"),
        ]
        self.close_failure: Exception | None = None

    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
        self.deleted.append(namespace)
        if self.delete_failure is not None:
            raise self.delete_failure
        return ProviderDeleteResult(client_duration_ms=1.0)

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
        self.metadata_calls.append(namespace)
        failure = self.metadata_failures.pop(0)
        if failure is not None:
            raise failure
        return ProviderNamespaceMetadata(
            approx_row_count=1,
            index_status="up-to-date",
            unindexed_bytes=0,
            schema={},
            client_duration_ms=1.0,
        )

    async def close(self) -> None:
        self.closed = True
        if self.close_failure is not None:
            raise self.close_failure


def _settings(data_dir: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        pufferlab_data_dir=data_dir,
        turbopuffer_api_key=api_key,
        turbopuffer_region="gcp-us-west1",
    )


def _session_factory(provider: _CleanupProvider) -> object:
    def factory(*, api_key: str, region: str) -> _CleanupProvider:
        assert api_key == "test-secret"
        assert region == "gcp-us-west1"
        return provider

    return factory


@pytest.mark.asyncio
async def test_m2_session_owns_exact_target_and_unlinks_only_after_confirmation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m2-live-session.json"
    session = m2_live_namespace_session.create_session(
        path,
        token_factory=lambda size: "a" * (size * 2),
    )
    assert session.namespace == _SAFE_SESSION_NAMESPACE
    assert m2_live_namespace_session.load_session(path) == session
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        m2_live_namespace_session.create_session(path)

    provider = _CleanupProvider()
    cleaned = await m2_live_namespace_session.cleanup_session(
        path,
        settings=_settings(tmp_path, api_key="test-secret"),
        provider_factory=_session_factory(provider),  # type: ignore[arg-type]
        poll_interval=0,
    )

    assert cleaned == session
    assert provider.deleted == [_SAFE_SESSION_NAMESPACE]
    assert provider.metadata_calls == [_SAFE_SESSION_NAMESPACE, _SAFE_SESSION_NAMESPACE]
    assert provider.closed
    assert not path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"format_version": 1, "namespace": "production"},
        {"format_version": 1, "namespace": _SAFE_SESSION_NAMESPACE, "cleanup": "production"},
        {"format_version": True, "namespace": _SAFE_SESSION_NAMESPACE},
        {"format_version": 1, "namespace": "pufferlab-unix-live-" + "A" * 24},
    ],
)
def test_m2_session_refuses_tampered_targets_and_shapes(
    tmp_path: Path,
    payload: dict[str, JsonValue],
) -> None:
    path = tmp_path / "m2-live-session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(RuntimeError):
        m2_live_namespace_session.load_session(path)


def test_m2_session_refuses_unsafe_permissions_and_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "m2-live-session.json"
    path.write_text(
        json.dumps({"format_version": 1, "namespace": _SAFE_SESSION_NAMESPACE}),
        encoding="utf-8",
    )
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        m2_live_namespace_session.load_session(path)

    target = tmp_path / "target.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        m2_live_namespace_session.load_session(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["delete", "poll", "close"])
async def test_m2_session_retains_record_on_every_cleanup_failure(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    path = tmp_path / "m2-live-session.json"
    m2_live_namespace_session.create_session(path, token_factory=lambda _: "b" * 24)
    provider = _CleanupProvider()
    if failure_stage == "delete":
        provider.delete_failure = _provider_error(ApiErrorCode.PROVIDER_ERROR, "delete")
    elif failure_stage == "poll":
        provider.metadata_failures = [_provider_error(ApiErrorCode.PROVIDER_ERROR, "metadata")]
    else:
        provider.metadata_failures = [_provider_error(ApiErrorCode.NOT_FOUND, "metadata")]
        provider.close_failure = RuntimeError("credential=test-secret")

    with pytest.raises((ProviderError, RuntimeError)):
        await m2_live_namespace_session.cleanup_session(
            path,
            settings=_settings(tmp_path, api_key="test-secret"),
            provider_factory=_session_factory(provider),  # type: ignore[arg-type]
            poll_interval=0,
        )

    assert provider.deleted == ["pufferlab-unix-live-" + "b" * 24]
    assert provider.closed
    assert path.exists()


@pytest.mark.asyncio
async def test_m2_session_retains_record_if_it_changes_before_unlink(tmp_path: Path) -> None:
    path = tmp_path / "m2-live-session.json"
    m2_live_namespace_session.create_session(path, token_factory=lambda _: "c" * 24)

    class MutatingProvider(_CleanupProvider):
        async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
            path.unlink()
            m2_live_namespace_session.create_session(path, token_factory=lambda _: "d" * 24)
            raise _provider_error(ApiErrorCode.NOT_FOUND, "metadata")

    provider = MutatingProvider()
    with pytest.raises(RuntimeError, match="changed during cleanup"):
        await m2_live_namespace_session.cleanup_session(
            path,
            settings=_settings(tmp_path, api_key="test-secret"),
            provider_factory=_session_factory(provider),  # type: ignore[arg-type]
            poll_interval=0,
        )
    assert provider.deleted == ["pufferlab-unix-live-" + "c" * 24]
    assert m2_live_namespace_session.load_session(path).namespace == (
        "pufferlab-unix-live-" + "d" * 24
    )


def test_m2_session_cli_redacts_cleanup_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "m2-live-session.json"
    m2_live_namespace_session.create_session(path, token_factory=lambda _: "e" * 24)
    provider = _CleanupProvider()
    provider.delete_failure = RuntimeError("credential=test-secret query=private text")

    exit_code = m2_live_namespace_session.run_cli(
        ["cleanup"],
        path=path,
        settings=_settings(tmp_path, api_key="test-secret"),
        provider_factory=_session_factory(provider),  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "m2_namespace_session command=cleanup status=failed\n"
    assert "test-secret" not in captured.err
    assert path.exists()


def _retrieval_configs(dataset_id: UUID) -> tuple[RetrievalConfig, ...]:
    common = {
        "revision": 1,
        "dataset_version_id": dataset_id,
        "result_k": 50,
        "candidate_k": 100,
        "created_at": _FIXED_TIME,
    }
    lexical = LexicalSpec()
    vector = VectorSpec(attribute="vector", embedding_model="test-embedding")
    rrf = RrfSpec()
    return (
        RetrievalConfig(
            id=_id("config:bm25"),
            name="BM25",
            mode=RetrievalMode.BM25,
            lexical=lexical,
            config_hash="bm25-hash",
            **common,
        ),
        RetrievalConfig(
            id=_id("config:vector"),
            name="ANN",
            mode=RetrievalMode.VECTOR,
            vector=vector,
            config_hash="vector-hash",
            **common,
        ),
        RetrievalConfig(
            id=_id("config:rrf"),
            name="Server RRF",
            mode=RetrievalMode.HYBRID_RRF,
            lexical=lexical,
            vector=vector,
            rrf=rrf,
            config_hash="rrf-hash",
            **common,
        ),
        RetrievalConfig(
            id=_id("config:rerank"),
            name="Local reranker",
            mode=RetrievalMode.HYBRID_RERANK,
            lexical=lexical,
            vector=vector,
            rrf=rrf,
            reranker=RerankerSpec(
                provider="sentence_transformers",
                model=DEFAULT_RERANKER_MODEL,
                revision=DEFAULT_RERANKER_REVISION,
                depth=50,
            ),
            config_hash="rerank-hash",
            **common,
        ),
    )


def _build_completed_database(path: Path) -> UUID:
    database = Database(path)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    dataset_id = _id("dataset")
    dataset = DatasetVersion(
        id=dataset_id,
        slug="synthetic-unix",
        version="v1",
        namespace=_SAFE_SESSION_NAMESPACE,
        index_profile=IndexProfile(
            id="test-profile",
            embedding_provider="sentence_transformers",
            embedding_model="test-embedding",
            embedding_revision="test-revision",
            vector_dimensions=3,
            vector_dtype="f16",
            distance_metric="cosine_distance",
            fts_profile=FtsProfile(),
            schema_hash="schema-hash",
        ),
        document_count=50,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=_FIXED_TIME,
    )
    repository.put_dataset_version(dataset)
    configs = _retrieval_configs(dataset_id)
    for config in configs:
        repository.put_retrieval_config(config)

    queries = tuple(
        JudgedQuery(
            id=_id(f"query:{index:02d}"),
            external_id=f"query-{index:02d}",
            text=f"PRIVATE QUERY TEXT {index:02d}",
            tags=["test"],
            qrels=[Qrel(document_id=_id(f"document:{index:02d}"), relevance_grade=2)],
        )
        for index in range(50)
    )
    query_set = QuerySet(
        id=_id("query-set"),
        name="synthetic curated 50",
        version="v1",
        dataset_version_id=dataset_id,
        query_count=50,
        content_hash="query-set-hash",
        created_at=_FIXED_TIME,
    )
    repository.put_query_set(query_set, queries)
    run = EvalRun(
        id=_id("run"),
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
            pufferlab_git_revision="test-revision",
            turbopuffer_region="gcp-us-west1",
            python_version="3.12",
            platform="test",
            max_concurrency=4,
            warmup_query_count=5,
            query_embedding_cache_enabled=False,
        ),
        created_at=_FIXED_TIME,
        started_at=None,
        completed_at=None,
        error=None,
    )
    repository.create_run(run)
    running = repository.transition_run(
        run.id,
        EvalRunStatus.RUNNING,
        at=_FIXED_TIME + timedelta(seconds=1),
    )
    for query_index, query in enumerate(queries):
        for config_index, config in enumerate(configs):
            payload = EvalSuccessPayload(
                ranked_document_ids=[query.qrels[0].document_id],
                metrics=PerQueryMetrics(
                    ndcg_at_10=1.0,
                    recall_at_50=1.0,
                    mrr_at_10=1.0,
                ),
                total_client_wall_latency_ms=float(config_index + query_index + 1),
                stage_timings=[],
                candidate_counts={"final": 1},
                warnings=[],
                trace_id=_id(f"trace:{config_index}:{query_index}"),
            )
            repository.record_outcome(
                QueryOutcome(
                    run_id=run.id,
                    config_id=config.id,
                    query_id=query.id,
                    status=QueryOutcomeStatus.SUCCEEDED,
                    payload=encode_outcome_payload(payload),
                    created_at=_FIXED_TIME + timedelta(seconds=2),
                )
            )
    running = repository.get_run(running.id)
    outcomes = repository.list_outcomes(running.id)
    summaries = finalize_durable_outcomes(
        running,
        outcomes,
        query_ids=[query.id for query in queries],
    )
    repository.complete_run(
        running.id,
        summaries,
        at=_FIXED_TIME + timedelta(seconds=3),
    )
    database.dispose()
    return running.id


@pytest.fixture(scope="module")
def completed_database(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, UUID]:
    data_dir = tmp_path_factory.mktemp("m2-completed")
    database_path = data_dir / "pufferlab.sqlite3"
    return database_path, _build_completed_database(database_path)


def _copy_database(completed_database: tuple[Path, UUID], target_dir: Path) -> tuple[Path, UUID]:
    source, run_id = completed_database
    target = target_dir / "pufferlab.sqlite3"
    shutil.copy2(source, target)
    return target, run_id


def test_verifier_recomputes_exact_coverage_read_only_and_prints_safe_report(
    completed_database: tuple[Path, UUID],
) -> None:
    path, run_id = completed_database
    before = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns

    report = verify_m2_evaluation.verify_evaluation(run_id, settings=_settings(path.parent))
    rendered = "\n".join(verify_m2_evaluation.render_report(report))

    after = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    assert before == after
    assert report.query_count == 50
    assert report.outcome_count == 200
    assert len(report.summaries) == 4
    assert len(report.export_sha256) == 64
    assert all(summary.completed_queries == 50 for summary in report.summaries)
    assert "PRIVATE QUERY TEXT" not in rendered
    assert "vector" not in rendered
    assert "credential" not in rendered
    assert rendered.endswith("verification=passed")


def test_verifier_rejects_persisted_summary_mismatch(
    completed_database: tuple[Path, UUID],
    tmp_path: Path,
) -> None:
    path, run_id = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM eval_runs WHERE id = ?", (str(run_id),)
            ).fetchone()[0]
        )
        payload["summaries"][0]["metrics"][0]["value"] = 0.125
        connection.execute(
            "UPDATE eval_runs SET payload_json = ? WHERE id = ?",
            (json.dumps(payload), str(run_id)),
        )

    with pytest.raises(
        verify_m2_evaluation.EvaluationVerificationError,
        match="summaries do not match",
    ):
        verify_m2_evaluation.verify_evaluation(run_id, settings=_settings(path.parent))


def test_verifier_rejects_incomplete_outcome_coverage(
    completed_database: tuple[Path, UUID],
    tmp_path: Path,
) -> None:
    path, run_id = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM query_outcomes WHERE rowid = "
            "(SELECT rowid FROM query_outcomes WHERE run_id = ? LIMIT 1)",
            (str(run_id),),
        )

    with pytest.raises(
        verify_m2_evaluation.EvaluationVerificationError,
        match="exactly 200 outcomes",
    ):
        verify_m2_evaluation.verify_evaluation(run_id, settings=_settings(path.parent))


def test_verifier_rejects_self_consistent_completed_run_with_a_failed_outcome(
    completed_database: tuple[Path, UUID],
    tmp_path: Path,
) -> None:
    path, run_id = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        row_id, payload_json = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes WHERE run_id = ? LIMIT 1",
            (str(run_id),),
        ).fetchone()
        outcome = QueryOutcome.model_validate_json(payload_json)
        failure = EvalFailurePayload(
            code=ApiErrorCode.PROVIDER_ERROR,
            message="redacted provider failure",
            retryable=False,
            operation="query",
            trace_id=_id("failed-outcome-trace"),
            total_client_wall_latency_ms=1.0,
        )
        failed = outcome.model_copy(
            update={
                "status": QueryOutcomeStatus.FAILED,
                "payload": encode_outcome_payload(failure),
            }
        )
        connection.execute(
            "UPDATE query_outcomes SET status = ?, payload_json = ? WHERE rowid = ?",
            (QueryOutcomeStatus.FAILED.value, canonical_json(failed), row_id),
        )

    database = Database(path)
    repository = PufferLabRepository(database.session_factory)
    run = repository.get_run(run_id)
    _, queries = repository.get_query_set(run.query_set.id)
    recomputed = finalize_durable_outcomes(
        run,
        repository.list_outcomes(run_id),
        query_ids=[query.id for query in queries],
    )
    database.dispose()
    with sqlite3.connect(path) as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT payload_json FROM eval_runs WHERE id = ?", (str(run_id),)
            ).fetchone()[0]
        )
        persisted["summaries"] = [summary.model_dump(mode="json") for summary in recomputed]
        connection.execute(
            "UPDATE eval_runs SET payload_json = ? WHERE id = ?",
            (json.dumps(persisted), str(run_id)),
        )

    with pytest.raises(
        verify_m2_evaluation.EvaluationVerificationError,
        match="200 successful outcomes",
    ):
        verify_m2_evaluation.verify_evaluation(run_id, settings=_settings(path.parent))


def test_verifier_cli_never_exposes_tampered_sensitive_payload(
    completed_database: tuple[Path, UUID],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, run_id = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        row_id, payload_json = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes WHERE run_id = ? LIMIT 1",
            (str(run_id),),
        ).fetchone()
        payload = json.loads(payload_json)
        payload["query_text"] = "TOP_SECRET_QUERY credential=test-secret vector=[0.1]"
        connection.execute(
            "UPDATE query_outcomes SET payload_json = ? WHERE rowid = ?",
            (json.dumps(payload), row_id),
        )

    exit_code = verify_m2_evaluation.run_cli(
        [str(run_id)],
        settings=_settings(path.parent),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "verification=failed\n"
    assert "TOP_SECRET_QUERY" not in captured.err
    assert "test-secret" not in captured.err


def test_verifier_forbidden_field_scan_is_recursive() -> None:
    with pytest.raises(
        verify_m2_evaluation.EvaluationVerificationError,
        match="forbidden field",
    ):
        verify_m2_evaluation._reject_forbidden_fields(
            {"outcomes": [{"outcome": {"vector": [0.1, 0.2]}}]}
        )
