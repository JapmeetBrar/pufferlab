from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from pufferlab.cli.evaluation import (
    ConfigSeedOptions,
    EvalRunOptions,
    ProgressCallback,
    UnixIngestOptions,
)
from pufferlab.cli.main import main
from pufferlab.config import Settings
from pufferlab.contracts.common import ContractModel, JsonValue
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    EvalRun,
    EvalRunStatus,
    MetricAggregate,
    MetricName,
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

NOW = datetime(2026, 8, 22, tzinfo=UTC)
RUN_ID = UUID("10000000-0000-0000-0000-000000000001")
DATASET_ID = UUID("20000000-0000-0000-0000-000000000001")
QUERY_SET_ID = UUID("30000000-0000-0000-0000-000000000001")
CONFIG_IDS = tuple(UUID(f"40000000-0000-0000-0000-00000000000{index}") for index in range(1, 5))
MODES = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)


class FakeExport(ContractModel):
    contract_version: Literal[1] = 1
    run: EvalRun
    outcomes: list[dict[str, JsonValue]]


class FakeApplication:
    def __init__(
        self,
        *,
        run: EvalRun | None = None,
        export: ContractModel | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.seed_result = _seed_result()
        self.result_run = run or _run()
        self.export_result = export or FakeExport(
            run=self.result_run,
            outcomes=[{"config_id": str(CONFIG_IDS[0]), "status": "succeeded"}],
        )
        self.failure = failure
        self.ingest_options: UnixIngestOptions | None = None
        self.seed_options: ConfigSeedOptions | None = None
        self.run_options: EvalRunOptions | None = None
        self.run_ids: list[UUID] = []
        self.cancelled: list[UUID] = []
        self.exported: list[UUID] = []
        self.close_calls = 0

    async def ingest_unix(
        self,
        options: UnixIngestOptions,
        *,
        emit: Callable[[str], None],
    ) -> SeedResultValue:
        self.ingest_options = options
        emit("progress state=ingesting batches=1/2 documents=64/100")
        emit("progress state=ready batches=2/2 documents=100/100")
        if self.failure is not None:
            raise self.failure
        return self.seed_result

    def seed(self, options: ConfigSeedOptions) -> SeedResultValue:
        self.seed_options = options
        if self.failure is not None:
            raise self.failure
        return self.seed_result

    async def run(
        self,
        options: EvalRunOptions,
        *,
        run_id: UUID,
        on_progress: ProgressCallback,
    ) -> EvalRun:
        self.run_options = options
        self.run_ids.append(run_id)
        if self.failure is not None:
            raise self.failure
        await on_progress(
            self.result_run.model_copy(
                update={
                    "status": EvalRunStatus.RUNNING,
                    "completed_queries": 1,
                    "summaries": [],
                }
            )
        )
        await on_progress(
            self.result_run.model_copy(
                update={
                    "status": EvalRunStatus.RUNNING,
                    "completed_queries": 1,
                    "summaries": [],
                }
            )
        )
        await on_progress(self.result_run)
        return self.result_run

    async def cancel_and_drain(self, run_id: UUID) -> EvalRun:
        self.cancelled.append(run_id)
        return self.result_run.model_copy(update={"status": EvalRunStatus.CANCELLED})

    def export(self, run_id: UUID) -> ContractModel:
        self.exported.append(run_id)
        if self.failure is not None:
            raise self.failure
        return self.export_result

    async def close(self) -> None:
        self.close_calls += 1


class SeedResultValue:
    def __init__(
        self,
        dataset_version: DatasetVersion,
        query_set: QuerySet,
        configs: tuple[RetrievalConfig, ...],
    ) -> None:
        self.dataset_version = dataset_version
        self.query_set = query_set
        self.configs = configs


def _settings(data_dir: Path) -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_data_dir": data_dir,
            "turbopuffer_api_key": "server-only-secret",
            "turbopuffer_region": "gcp-us-west1",
        }
    )


def _seed_result() -> SeedResultValue:
    dataset = DatasetVersion(
        id=DATASET_ID,
        slug="cqadupstack-unix",
        version="unix-v1",
        namespace="pufferlab-unix-test",
        index_profile=IndexProfile(
            id="unix-index",
            embedding_provider="sentence_transformers",
            embedding_model="model",
            embedding_revision="revision",
            vector_dimensions=384,
            vector_dtype="f16",
            distance_metric="cosine_distance",
            fts_profile=FtsProfile(),
            schema_hash="schema-hash",
        ),
        document_count=47_382,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=NOW,
    )
    query_set = QuerySet(
        id=QUERY_SET_ID,
        name="Unix curated 50",
        version="curated-v1",
        dataset_version_id=dataset.id,
        query_count=50,
        content_hash="query-set-hash",
        created_at=NOW,
    )
    configs = tuple(
        _config(config_id, mode) for config_id, mode in zip(CONFIG_IDS, MODES, strict=True)
    )
    return SeedResultValue(dataset, query_set, configs)


def _config(config_id: UUID, mode: RetrievalMode) -> RetrievalConfig:
    lexical = LexicalSpec() if mode is not RetrievalMode.VECTOR else None
    vector = VectorSpec(embedding_model="model") if mode is not RetrievalMode.BM25 else None
    rrf = RrfSpec() if mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK} else None
    reranker = (
        RerankerSpec(provider="sentence_transformers", model="reranker", revision="revision")
        if mode is RetrievalMode.HYBRID_RERANK
        else None
    )
    return RetrievalConfig(
        id=config_id,
        revision=1,
        name=mode.value,
        dataset_version_id=DATASET_ID,
        mode=mode,
        result_k=50,
        candidate_k=100,
        lexical=lexical,
        vector=vector,
        rrf=rrf,
        reranker=reranker,
        config_hash=f"hash-{mode.value}",
        created_at=NOW,
    )


def _run(*, failed_queries: int = 0, status: EvalRunStatus = EvalRunStatus.COMPLETED) -> EvalRun:
    summaries = [
        ConfigRunSummary(
            config_id=config_id,
            metrics=[
                MetricAggregate(
                    name=name, value=0.0 if name is MetricName.ERROR_RATE else 0.5, sample_count=50
                )
                for name in MetricName
            ],
            completed_queries=50 - failed_queries,
            failed_queries=failed_queries,
        )
        for config_id in CONFIG_IDS
    ]
    return EvalRun(
        id=RUN_ID,
        status=status,
        query_set=QuerySetSummary(
            id=QUERY_SET_ID,
            name="Unix curated 50",
            version="curated-v1",
            query_count=50,
            content_hash="query-set-hash",
        ),
        baseline_config_id=CONFIG_IDS[0],
        candidate_config_ids=list(CONFIG_IDS[1:]),
        summaries=summaries if status is EvalRunStatus.COMPLETED else [],
        completed_queries=50,
        total_queries=50,
        random_seed=20260822,
        environment=RunEnvironment(
            pufferlab_git_revision="revision",
            turbopuffer_region="gcp-us-west1",
            python_version="3.12",
            platform="test",
            max_concurrency=4,
            query_embedding_cache_enabled=True,
        ),
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
        error=None,
    )


def _invoke(
    arguments: list[str],
    application: FakeApplication,
    tmp_path: Path,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        arguments,
        settings_factory=lambda: _settings(tmp_path),
        cli_application_factory=lambda settings: application,
        run_id_factory=lambda: RUN_ID,
        stdout=stdout,
        stderr=stderr,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize(
    ("arguments", "required"),
    [
        (
            ["dataset", "ingest-unix", "--help"],
            (
                "TURBOPUFFER_API_KEY",
                "TURBOPUFFER_REGION",
                "PUFFERLAB_DATA_DIR",
                "cost-bearing turbopuffer writes",
                "--processed-pack",
                "--namespace",
            ),
        ),
        (
            ["config", "seed", "--help"],
            ("four canonical retrieval configurations", "PUFFERLAB_DATA_DIR"),
        ),
        (
            ["eval", "run", "--help"],
            (
                "TURBOPUFFER_API_KEY",
                "TURBOPUFFER_REGION",
                "cost-bearing provider target",
                "--seeded-defaults",
            ),
        ),
        (
            ["eval", "export", "--help"],
            ("cancelled", "partial", "PUFFERLAB_DATA_DIR", "--output"),
        ),
    ],
)
def test_help_names_required_environment_data_namespace_and_cost(
    arguments: list[str],
    required: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(arguments)

    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for fragment in required:
        assert fragment in help_text


def test_unix_ingest_routes_options_progress_and_seed_without_sensitive_values(
    tmp_path: Path,
) -> None:
    application = FakeApplication()
    processed = tmp_path / "processed-secret-name"
    processed.mkdir()

    exit_code, stdout, stderr = _invoke(
        [
            "dataset",
            "ingest-unix",
            "--processed-pack",
            str(processed),
            "--namespace",
            "pufferlab-unix-evaluation",
            "--batch-size",
            "32",
            "--max-concurrency",
            "3",
        ],
        application,
        tmp_path,
    )

    assert exit_code == 0
    assert stderr == ""
    assert application.ingest_options is not None
    assert application.ingest_options.processed_pack_path == processed
    assert application.ingest_options.namespace == "pufferlab-unix-evaluation"
    assert application.ingest_options.batch_size == 32
    assert application.ingest_options.max_concurrency == 3
    assert "unix ingestion plan" in stdout
    assert "progress state=ready" in stdout
    assert f"dataset id={DATASET_ID} revision=unix-v1" in stdout
    assert stdout.count("config ordinal=") == 4
    assert "server-only-secret" not in stdout
    assert "processed-secret-name" not in stdout
    assert application.close_calls == 1


@pytest.mark.parametrize("path_kind", ["missing", "outside", "symlink"])
def test_unix_ingest_refuses_unsafe_or_missing_processed_pack(
    tmp_path: Path,
    path_kind: str,
) -> None:
    application = FakeApplication()
    if path_kind == "missing":
        processed = tmp_path / "missing"
    elif path_kind == "outside":
        processed = tmp_path.parent / f"{tmp_path.name}-outside-pack"
        processed.mkdir()
    else:
        real = tmp_path / "real-pack"
        real.mkdir()
        processed = tmp_path / "pack-link"
        processed.symlink_to(real, target_is_directory=True)

    exit_code, stdout, stderr = _invoke(
        ["dataset", "ingest-unix", "--processed-pack", str(processed)],
        application,
        tmp_path,
    )

    assert exit_code == 2
    assert stdout == ""
    assert "processed pack" in stderr
    assert application.ingest_options is None


def test_config_seed_reports_dataset_revision_and_four_ordered_immutable_ids(
    tmp_path: Path,
) -> None:
    application = FakeApplication()

    exit_code, stdout, stderr = _invoke(
        ["config", "seed", "--dataset-version", str(DATASET_ID)],
        application,
        tmp_path,
    )

    assert exit_code == 0
    assert stderr == ""
    assert application.seed_options == ConfigSeedOptions(dataset_version_id=DATASET_ID)
    assert f"dataset id={DATASET_ID} revision=unix-v1" in stdout
    positions = [stdout.index(f"id={config_id}") for config_id in CONFIG_IDS]
    assert positions == sorted(positions)
    assert "server-only-secret" not in stdout
    assert application.close_calls == 1


def test_successful_seeded_run_emits_compact_progress_six_metrics_and_zero_exit(
    tmp_path: Path,
) -> None:
    application = FakeApplication()

    exit_code, stdout, stderr = _invoke(
        [
            "eval",
            "run",
            "--seeded-defaults",
            "--random-seed",
            "7",
            "--max-concurrency",
            "2",
            "--warmup-count",
            "0",
        ],
        application,
        tmp_path,
    )

    assert exit_code == 0
    assert stderr == ""
    assert application.run_options == EvalRunOptions(
        query_set_id=None,
        baseline_config_id=None,
        candidate_config_ids=(),
        seeded_defaults=True,
        random_seed=7,
        max_concurrency=2,
        warmup_query_count=0,
    )
    assert application.run_ids == [RUN_ID]
    assert stdout.count("status=running queries=1/50") == 1
    assert f"run_id={RUN_ID} status=completed queries=50/50" in stdout
    for metric in MetricName:
        assert stdout.count(f"{metric.value}=") == 4
    assert stdout.count("[n=50]") == 24
    assert application.close_calls == 1


def test_explicit_run_routes_query_set_baseline_and_candidates(tmp_path: Path) -> None:
    application = FakeApplication()
    arguments = [
        "eval",
        "run",
        "--query-set",
        str(QUERY_SET_ID),
        "--baseline",
        str(CONFIG_IDS[0]),
    ]
    for config_id in CONFIG_IDS[1:]:
        arguments.extend(("--candidate", str(config_id)))

    exit_code, _, stderr = _invoke(arguments, application, tmp_path)

    assert exit_code == 0
    assert stderr == ""
    assert application.run_options == EvalRunOptions(
        query_set_id=QUERY_SET_ID,
        baseline_config_id=CONFIG_IDS[0],
        candidate_config_ids=CONFIG_IDS[1:],
        seeded_defaults=False,
    )


def test_completed_run_with_failed_outcomes_has_nonzero_coverage_exit(tmp_path: Path) -> None:
    application = FakeApplication(run=_run(failed_queries=1))

    exit_code, stdout, stderr = _invoke(
        ["eval", "run", "--seeded-defaults"],
        application,
        tmp_path,
    )

    assert exit_code == 3
    assert stderr == ""
    assert "completed=49 failed=1" in stdout


def test_run_system_error_is_redacted_and_nonzero(tmp_path: Path) -> None:
    secret = "provider-error-with-secret"
    application = FakeApplication(failure=RuntimeError(secret))

    exit_code, stdout, stderr = _invoke(
        ["eval", "run", "--seeded-defaults"],
        application,
        tmp_path,
    )

    assert exit_code == 1
    assert stdout == ""
    assert stderr == "error: evaluation run failed\n"
    assert secret not in stderr
    assert application.close_calls == 1


def test_keyboard_interrupt_cancels_drains_and_returns_130(tmp_path: Path) -> None:
    application = FakeApplication(failure=KeyboardInterrupt())

    exit_code, stdout, stderr = _invoke(
        ["eval", "run", "--seeded-defaults"],
        application,
        tmp_path,
    )

    assert exit_code == 130
    assert stdout == ""
    assert stderr == f"run_id={RUN_ID} status=cancelled\n"
    assert application.cancelled == [RUN_ID]
    assert application.close_calls == 1


def test_export_writes_canonical_contract_json_and_round_trips(tmp_path: Path) -> None:
    application = FakeApplication()

    exit_code, stdout, stderr = _invoke(
        ["eval", "export", str(RUN_ID), "--output", "exports/run.json"],
        application,
        tmp_path,
    )

    output_path = tmp_path / "exports" / "run.json"
    assert exit_code == 0
    assert stderr == ""
    assert stdout == f"exported run_id={RUN_ID} path={output_path}\n"
    encoded = output_path.read_text(encoding="utf-8")
    assert encoded.endswith("\n")
    assert (
        encoded
        == json.dumps(
            application.export_result.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert FakeExport.model_validate_json(encoded) == application.export_result
    assert application.exported == [RUN_ID]
    assert application.close_calls == 1


def test_export_refuses_existing_file_escape_and_symbolic_link(tmp_path: Path) -> None:
    application = FakeApplication()
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")

    exit_code, _, stderr = _invoke(
        ["eval", "export", str(RUN_ID), "--output", "existing.json"],
        application,
        tmp_path,
    )
    assert exit_code == 2
    assert "--overwrite" in stderr
    assert existing.read_text(encoding="utf-8") == "keep"
    assert application.exported == []

    exit_code, _, stderr = _invoke(
        ["eval", "export", str(RUN_ID), "--output", str(tmp_path.parent / "escape.json")],
        FakeApplication(),
        tmp_path,
    )
    assert exit_code == 2
    assert "inside PUFFERLAB_DATA_DIR" in stderr

    target = tmp_path / "target.json"
    target.write_text("keep", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    exit_code, _, stderr = _invoke(
        ["eval", "export", str(RUN_ID), "--output", "link.json", "--overwrite"],
        FakeApplication(),
        tmp_path,
    )
    assert exit_code == 2
    assert "symbolic link" in stderr
    assert target.read_text(encoding="utf-8") == "keep"

    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    exit_code, _, stderr = _invoke(
        ["eval", "export", str(RUN_ID), "--output", "pipe.json", "--overwrite"],
        FakeApplication(),
        tmp_path,
    )
    assert exit_code == 2
    assert "regular file" in stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval", "run"],
        ["eval", "run", "--seeded-defaults", "--candidate", str(CONFIG_IDS[1])],
        [
            "eval",
            "run",
            "--query-set",
            str(QUERY_SET_ID),
            "--baseline",
            str(CONFIG_IDS[0]),
        ],
        ["eval", "run", "--seeded-defaults", "--max-concurrency", "0"],
        ["eval", "export", "not-a-uuid", "--output", "run.json"],
        ["dataset", "ingest-unix"],
    ],
)
def test_invalid_arguments_exit_two(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(arguments)

    assert caught.value.code == 2
