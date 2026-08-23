from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import sqlite3
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pufferlab.config import Settings
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
from pufferlab.providers.types import ProviderDeleteResult, ProviderNamespaceMetadata

from scripts import m2_live_namespace_session, verify_m2_evaluation

_TEST_NAMESPACE = UUID("147c12c2-7938-4711-8d40-d4659dc92767")
_FIXED_TIME = datetime(2026, 8, 22, 19, 0, tzinfo=UTC)
_SAFE_NAMESPACE = "pufferlab-unix-live-" + "a" * 24


def _id(name: str) -> UUID:
    return uuid5(_TEST_NAMESPACE, name)


def _paths(tmp_path: Path, name: str = "owned") -> m2_live_namespace_session._SessionPaths:
    directory = tmp_path / name
    return m2_live_namespace_session._SessionPaths(
        session=directory / "m2-live-session.json",
        owner_key=directory / "m2-live-owner.key",
    )


def _deterministic_random(*values: bytes) -> Callable[[int], bytes]:
    remaining = iter(values)

    def generate(size: int) -> bytes:
        value = next(remaining)
        assert len(value) == size
        return value

    return generate


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


def _create_owned_session(
    paths: m2_live_namespace_session._SessionPaths,
    *,
    key_byte: bytes = b"k",
    nonce_byte: bytes = b"n",
) -> m2_live_namespace_session.M2LiveNamespaceSession:
    return m2_live_namespace_session._create_session_at(
        paths,
        random_bytes=_deterministic_random(key_byte * 32, nonce_byte * 32),
    )


@pytest.mark.asyncio
async def test_session_owns_authenticated_fixed_target_and_cleans_after_confirmation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    session = _create_owned_session(paths)
    assert re.fullmatch(r"pufferlab-unix-live-[0-9a-f]{24}", session.namespace)
    assert m2_live_namespace_session._load_owned_session_at(paths).session == session
    assert stat.S_IMODE(paths.session.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.owner_key.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _create_owned_session(paths, nonce_byte=b"o")

    provider = _CleanupProvider()
    cleaned = await m2_live_namespace_session._cleanup_session_at(
        paths,
        settings=_settings(tmp_path, api_key="test-secret"),
        provider_factory=_session_factory(provider),  # type: ignore[arg-type]
        poll_interval=0,
    )

    assert cleaned == session
    assert provider.deleted == [session.namespace]
    assert provider.metadata_calls == [session.namespace, session.namespace]
    assert provider.closed
    assert not paths.session.exists()
    assert paths.owner_key.exists()


@pytest.mark.asyncio
async def test_foreign_valid_prefix_record_cannot_authorize_cleanup(tmp_path: Path) -> None:
    owned_paths = _paths(tmp_path, "owned")
    foreign_paths = _paths(tmp_path, "foreign")
    owned = _create_owned_session(owned_paths, key_byte=b"a", nonce_byte=b"b")
    foreign = _create_owned_session(foreign_paths, key_byte=b"c", nonce_byte=b"d")
    assert owned.namespace != foreign.namespace
    shutil.copy2(foreign_paths.session, owned_paths.session)
    provider = _CleanupProvider()

    with pytest.raises(RuntimeError, match="not locally owned"):
        await m2_live_namespace_session._cleanup_session_at(
            owned_paths,
            settings=_settings(tmp_path, api_key="test-secret"),
            provider_factory=_session_factory(provider),  # type: ignore[arg-type]
            poll_interval=0,
        )

    assert provider.deleted == []
    assert owned_paths.session.exists()


def test_hand_authored_valid_prefix_and_shape_is_not_owned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _create_owned_session(paths)
    payload = {
        "format_version": 1,
        "nonce": "f" * 64,
        "namespace": "pufferlab-unix-live-" + "f" * 24,
        "ownership_tag": "f" * 64,
    }
    paths.session.write_text(json.dumps(payload), encoding="utf-8")
    paths.session.chmod(0o600)

    with pytest.raises(RuntimeError, match="not locally owned"):
        m2_live_namespace_session._load_owned_session_at(paths)


@pytest.mark.parametrize("target", ["session", "owner_key"])
def test_owned_files_reject_unsafe_permissions_and_symlinks(
    tmp_path: Path,
    target: str,
) -> None:
    paths = _paths(tmp_path)
    _create_owned_session(paths)
    path = getattr(paths, target)
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        m2_live_namespace_session._load_owned_session_at(paths)

    path.chmod(0o600)
    moved = path.with_suffix(".target")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        m2_live_namespace_session._load_owned_session_at(paths)


def test_public_namespace_apis_have_no_target_or_randomness_injection() -> None:
    assert list(inspect.signature(m2_live_namespace_session.create_session).parameters) == []
    assert list(inspect.signature(m2_live_namespace_session.load_session).parameters) == []
    assert list(inspect.signature(m2_live_namespace_session.cleanup_session).parameters) == []
    assert list(inspect.signature(m2_live_namespace_session.run_cli).parameters) == ["argv"]
    with pytest.raises(SystemExit):
        m2_live_namespace_session.run_cli(["cleanup", "pufferlab-unix-live-" + "f" * 24])


def test_start_fsyncs_owner_session_and_directory_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    observed_types: list[str] = []
    real_fsync = os.fsync

    def observed_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        observed_types.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(m2_live_namespace_session.os, "fsync", observed_fsync)
    _create_owned_session(paths)

    assert observed_types == ["file", "directory", "file", "directory"]


def test_start_refuses_to_return_if_session_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    regular_file_calls = 0
    real_fsync = os.fsync

    def failing_fsync(descriptor: int) -> None:
        nonlocal regular_file_calls
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            regular_file_calls += 1
            if regular_file_calls == 2:
                raise OSError("simulated durability failure")
        real_fsync(descriptor)

    monkeypatch.setattr(m2_live_namespace_session.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="durability"):
        _create_owned_session(paths)
    assert not paths.session.exists()
    assert paths.owner_key.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["delete", "poll", "close"])
async def test_session_retains_record_on_every_cleanup_failure(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    paths = _paths(tmp_path)
    session = _create_owned_session(paths)
    provider = _CleanupProvider()
    if failure_stage == "delete":
        provider.delete_failure = _provider_error(ApiErrorCode.PROVIDER_ERROR, "delete")
    elif failure_stage == "poll":
        provider.metadata_failures = [_provider_error(ApiErrorCode.PROVIDER_ERROR, "metadata")]
    else:
        provider.metadata_failures = [_provider_error(ApiErrorCode.NOT_FOUND, "metadata")]
        provider.close_failure = RuntimeError("credential=test-secret")

    with pytest.raises((ProviderError, RuntimeError)):
        await m2_live_namespace_session._cleanup_session_at(
            paths,
            settings=_settings(tmp_path, api_key="test-secret"),
            provider_factory=_session_factory(provider),  # type: ignore[arg-type]
            poll_interval=0,
        )

    assert provider.deleted == [session.namespace]
    assert provider.closed
    assert paths.session.exists()


@pytest.mark.asyncio
async def test_session_retains_replacement_record_before_unlink(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    session = _create_owned_session(paths)

    class MutatingProvider(_CleanupProvider):
        async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
            paths.session.unlink()
            replacement = {
                "format_version": 1,
                "nonce": "f" * 64,
                "namespace": "pufferlab-unix-live-" + "f" * 24,
                "ownership_tag": "f" * 64,
            }
            paths.session.write_text(json.dumps(replacement), encoding="utf-8")
            paths.session.chmod(0o600)
            raise _provider_error(ApiErrorCode.NOT_FOUND, "metadata")

    provider = MutatingProvider()
    with pytest.raises(RuntimeError):
        await m2_live_namespace_session._cleanup_session_at(
            paths,
            settings=_settings(tmp_path, api_key="test-secret"),
            provider_factory=_session_factory(provider),  # type: ignore[arg-type]
            poll_interval=0,
        )
    assert provider.deleted == [session.namespace]
    assert paths.session.exists()


def test_namespace_cli_redacts_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail() -> m2_live_namespace_session.M2LiveNamespaceSession:
        raise RuntimeError("credential=test-secret query=private text")

    monkeypatch.setattr(m2_live_namespace_session, "cleanup_session", fail)
    assert m2_live_namespace_session.run_cli(["cleanup"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "m2_namespace_session command=cleanup status=failed\n"
    assert "test-secret" not in captured.err


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


def _build_completed_database(
    path: Path,
) -> tuple[UUID, verify_m2_evaluation._ExpectedSuite]:
    database = Database(path)
    database.migrate()
    repository = PufferLabRepository(database.session_factory)
    dataset = DatasetVersion(
        id=_id("dataset"),
        slug="synthetic-unix",
        version="v1",
        namespace=_SAFE_NAMESPACE,
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
    configs = _retrieval_configs(dataset.id)
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
        dataset_version_id=dataset.id,
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
    repository.transition_run(run.id, EvalRunStatus.RUNNING, at=_FIXED_TIME + timedelta(seconds=1))
    for query_index, query in enumerate(queries):
        for config_index, config in enumerate(configs):
            payload = EvalSuccessPayload(
                ranked_document_ids=[query.qrels[0].document_id],
                metrics=PerQueryMetrics(ndcg_at_10=1.0, recall_at_50=1.0, mrr_at_10=1.0),
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
    running = repository.get_run(run.id)
    summaries = finalize_durable_outcomes(
        running, repository.list_outcomes(run.id), query_ids=[query.id for query in queries]
    )
    repository.complete_run(run.id, summaries, at=_FIXED_TIME + timedelta(seconds=3))
    database.dispose()
    expected = verify_m2_evaluation._ExpectedSuite(
        dataset=dataset, query_set=query_set, queries=queries, configs=configs
    )
    return run.id, expected


@pytest.fixture(scope="module")
def completed_database(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite]:
    path = tmp_path_factory.mktemp("m2-completed") / "pufferlab.sqlite3"
    run_id, expected = _build_completed_database(path)
    return path, run_id, expected


def _copy_database(
    completed: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite], target_dir: Path
) -> tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite]:
    source, run_id, expected = completed
    target = target_dir / "pufferlab.sqlite3"
    shutil.copy2(source, target)
    return target, run_id, expected


def _verify(
    path: Path, run_id: UUID, expected: verify_m2_evaluation._ExpectedSuite
) -> verify_m2_evaluation.VerifiedEvaluation:
    return verify_m2_evaluation._verify_evaluation_at(run_id, database_path=path, expected=expected)


def test_verifier_recomputes_read_only_and_prints_safe_bound_report(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
) -> None:
    path, run_id, expected = completed_database
    before = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    sidecars = tuple(path.parent.glob(f"{path.name}-*"))
    report = _verify(path, run_id, expected)
    rendered = "\n".join(verify_m2_evaluation.render_report(report))
    after = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    assert before == after
    assert tuple(path.parent.glob(f"{path.name}-*")) == sidecars
    assert report.dataset_version_id == expected.dataset.id
    assert report.query_count == 50 and report.outcome_count == 200
    assert len(report.summaries) == 4 and len(report.export_sha256) == 64
    assert "PRIVATE QUERY TEXT" not in rendered
    assert "vector" not in rendered and "credential" not in rendered
    assert "[n=50]" in rendered and rendered.endswith("verification=passed")


@pytest.mark.parametrize(
    ("table", "mutation"),
    [
        ("dataset_versions", {"namespace": "pufferlab-unix-live-" + "f" * 24}),
        ("dataset_versions", {"status": "failed", "document_count": 1}),
        ("retrieval_configs", {"result_k": 49, "candidate_k": 99, "consistency": "eventual"}),
    ],
)
def test_verifier_rejects_other_dataset_or_noncanonical_config(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
    tmp_path: Path,
    table: str,
    mutation: dict[str, object],
) -> None:
    path, run_id, expected = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        if table == "retrieval_configs":
            row_id, payload_json = connection.execute(
                "SELECT rowid, payload_json FROM retrieval_configs WHERE id = ?",
                (str(expected.configs[0].id),),
            ).fetchone()
        else:
            row_id, payload_json = connection.execute(
                "SELECT rowid, payload_json FROM dataset_versions WHERE id = ?",
                (str(expected.dataset.id),),
            ).fetchone()
        payload = json.loads(payload_json)
        payload.update(mutation)
        connection.execute(
            f"UPDATE {table} SET payload_json = ? WHERE rowid = ?",
            (json.dumps(payload), row_id),
        )
    with pytest.raises(verify_m2_evaluation.EvaluationVerificationError, match="canonical"):
        _verify(path, run_id, expected)


def test_verifier_rejects_persisted_summary_mismatch(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
    tmp_path: Path,
) -> None:
    path, run_id, expected = _copy_database(completed_database, tmp_path)
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
    with pytest.raises(verify_m2_evaluation.EvaluationVerificationError, match="independent"):
        _verify(path, run_id, expected)


def test_verifier_rejects_incomplete_outcome_coverage(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
    tmp_path: Path,
) -> None:
    path, run_id, expected = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM query_outcomes WHERE rowid IN (SELECT rowid FROM query_outcomes LIMIT 1)"
        )
    with pytest.raises(verify_m2_evaluation.EvaluationVerificationError, match="exactly 200"):
        _verify(path, run_id, expected)


def test_verifier_rejects_self_consistent_failed_outcome(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
    tmp_path: Path,
) -> None:
    path, run_id, expected = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        row_id, payload_json = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes LIMIT 1"
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
            update={"status": QueryOutcomeStatus.FAILED, "payload": encode_outcome_payload(failure)}
        )
        connection.execute(
            "UPDATE query_outcomes SET status = ?, payload_json = ? WHERE rowid = ?",
            (QueryOutcomeStatus.FAILED.value, canonical_json(failed), row_id),
        )
    with pytest.raises(verify_m2_evaluation.EvaluationVerificationError, match="200 successful"):
        _verify(path, run_id, expected)


def test_verifier_recomputes_query_metrics_instead_of_trusting_self_consistent_storage(
    completed_database: tuple[Path, UUID, verify_m2_evaluation._ExpectedSuite],
    tmp_path: Path,
) -> None:
    path, run_id, expected = _copy_database(completed_database, tmp_path)
    with sqlite3.connect(path) as connection:
        row_id, payload_json = connection.execute(
            "SELECT rowid, payload_json FROM query_outcomes WHERE config_id = ? LIMIT 1",
            (str(expected.configs[0].id),),
        ).fetchone()
        outcome = json.loads(payload_json)
        outcome["payload"]["metrics"] = {
            "ndcg_at_10": 0.25,
            "recall_at_50": 0.25,
            "mrr_at_10": 0.25,
        }
        connection.execute(
            "UPDATE query_outcomes SET payload_json = ? WHERE rowid = ?",
            (json.dumps(outcome), row_id),
        )
        run_payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM eval_runs WHERE id = ?", (str(run_id),)
            ).fetchone()[0]
        )
        for metric in run_payload["summaries"][0]["metrics"][:3]:
            metric["value"] = 0.985
        connection.execute(
            "UPDATE eval_runs SET payload_json = ? WHERE id = ?",
            (json.dumps(run_payload), str(run_id)),
        )
    with pytest.raises(
        verify_m2_evaluation.EvaluationVerificationError,
        match="stored per-query metrics do not match ranking and qrels",
    ):
        _verify(path, run_id, expected)


def test_verifier_cli_and_public_api_accept_only_run_uuid_and_redact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert list(inspect.signature(verify_m2_evaluation.verify_evaluation).parameters) == ["run_id"]
    assert list(inspect.signature(verify_m2_evaluation.run_cli).parameters) == ["argv"]

    def fail(_: UUID) -> verify_m2_evaluation.VerifiedEvaluation:
        raise RuntimeError("TOP_SECRET_QUERY credential=test-secret vector=[0.1]")

    monkeypatch.setattr(verify_m2_evaluation, "verify_evaluation", fail)
    assert verify_m2_evaluation.run_cli([str(_id("run"))]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "verification=failed\n"
    assert "TOP_SECRET_QUERY" not in captured.err and "test-secret" not in captured.err


def test_verifier_forbidden_field_scan_is_recursive() -> None:
    with pytest.raises(verify_m2_evaluation.EvaluationVerificationError, match="forbidden field"):
        verify_m2_evaluation._reject_forbidden_fields(
            {"outcomes": [{"outcome": {"vector": [0.1, 0.2]}}]}
        )
