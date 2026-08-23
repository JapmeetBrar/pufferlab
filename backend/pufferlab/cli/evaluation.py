"""Typed command boundary for durable evaluation workflows."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pufferlab.config import Settings
from pufferlab.contracts.common import ContractModel
from pufferlab.contracts.datasets import DatasetVersion
from pufferlab.contracts.evals import EvalRun, EvalRunStatus, MetricName, QuerySet
from pufferlab.contracts.retrieval import RetrievalConfig


class EvaluationCommandError(RuntimeError):
    """A safe user-facing command failure with a stable exit code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class UnixIngestOptions:
    namespace: str
    processed_pack_path: Path
    source_lock_path: Path
    processed_pack_lock_path: Path
    dataset_manifest_path: Path
    curated_manifest_path: Path
    batch_size: int = 64
    max_concurrency: int = 2
    readiness_attempts: int = 180


@dataclass(frozen=True, slots=True)
class ConfigSeedOptions:
    dataset_version_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class EvalRunOptions:
    query_set_id: UUID | None
    baseline_config_id: UUID | None
    candidate_config_ids: tuple[UUID, ...]
    seeded_defaults: bool
    random_seed: int = 20260822
    max_concurrency: int = 4
    warmup_query_count: int = 5


@dataclass(frozen=True, slots=True)
class EvalExportOptions:
    run_id: UUID
    output_path: Path
    overwrite: bool = False


class SeedResult(Protocol):
    @property
    def dataset_version(self) -> DatasetVersion: ...

    @property
    def query_set(self) -> QuerySet: ...

    @property
    def configs(self) -> Sequence[RetrievalConfig]: ...


type ProgressCallback = Callable[[EvalRun], Awaitable[None]]


class CliApplication(Protocol):
    """Composition facade injected into the command layer."""

    async def ingest_unix(
        self,
        options: UnixIngestOptions,
        *,
        emit: Callable[[str], None],
    ) -> SeedResult: ...

    def seed(self, options: ConfigSeedOptions) -> SeedResult: ...

    async def run(
        self,
        options: EvalRunOptions,
        *,
        run_id: UUID,
        on_progress: ProgressCallback,
    ) -> EvalRun: ...

    async def cancel_and_drain(self, run_id: UUID) -> EvalRun: ...

    def export(self, run_id: UUID) -> ContractModel: ...

    async def close(self) -> None: ...


type CliApplicationFactory = Callable[[Settings], CliApplication]


def render_seed(result: SeedResult, *, emit: Callable[[str], None]) -> None:
    """Print immutable identities without licensed query text or secret-bearing settings."""
    expected_modes = ("bm25", "vector", "hybrid_rrf", "hybrid_rerank")
    if tuple(config.mode.value for config in result.configs) != expected_modes:
        raise EvaluationCommandError(
            "configuration seed did not return the four canonical ordered modes"
        )
    emit(
        f"dataset id={result.dataset_version.id} revision={result.dataset_version.version} "
        f"namespace={result.dataset_version.namespace}"
    )
    emit(
        f"query_set id={result.query_set.id} version={result.query_set.version} "
        f"queries={result.query_set.query_count}"
    )
    for ordinal, config in enumerate(result.configs, start=1):
        emit(
            f"config ordinal={ordinal} id={config.id} revision={config.revision} "
            f"mode={config.mode.value} hash={config.config_hash}"
        )


class CompactEvalProgress:
    """Suppress duplicate snapshots while preserving post-commit progress."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._last: tuple[object, ...] | None = None

    async def __call__(self, run: EvalRun) -> None:
        current = (run.status, run.completed_queries, run.total_queries)
        if current == self._last:
            return
        self._last = current
        self._emit(
            f"progress run_id={run.id} status={run.status.value} "
            f"queries={run.completed_queries}/{run.total_queries}"
        )


def render_run(run: EvalRun, *, emit: Callable[[str], None]) -> None:
    if run.status is EvalRunStatus.COMPLETED:
        if len(run.summaries) != 4:
            raise EvaluationCommandError("completed evaluation is missing canonical summaries")
        expected_metrics = tuple(MetricName)
        if any(
            tuple(metric.name for metric in summary.metrics) != expected_metrics
            for summary in run.summaries
        ):
            raise EvaluationCommandError("completed evaluation is missing canonical metrics")
    emit(
        f"run_id={run.id} status={run.status.value} "
        f"queries={run.completed_queries}/{run.total_queries}"
    )
    for summary in run.summaries:
        metrics = " ".join(
            f"{metric.name.value}={_metric_value(metric.value)}[n={metric.sample_count}]"
            for metric in summary.metrics
        )
        emit(
            f"config id={summary.config_id} completed={summary.completed_queries} "
            f"failed={summary.failed_queries} {metrics}"
        )


def run_exit_code(run: EvalRun) -> int:
    if run.status is not EvalRunStatus.COMPLETED:
        return 1
    if any(summary.failed_queries for summary in run.summaries):
        return 3
    return 0


def write_canonical_export(
    export: ContractModel,
    options: EvalExportOptions,
    *,
    settings: Settings,
) -> Path:
    """Atomically write canonical contract JSON only inside the ignored data directory."""
    target = validate_export_destination(options, settings=settings)
    parent = target.parent.resolve()

    parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        export.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded = f"{payload}\n".encode()
    if not options.overwrite:
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise EvaluationCommandError(
                "export output already exists; pass --overwrite to replace it",
                exit_code=2,
            ) from None
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return target

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            os.chmod(temporary_name, 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def validate_export_destination(
    options: EvalExportOptions,
    *,
    settings: Settings,
) -> Path:
    """Resolve and reject unsafe export targets before application work begins."""
    data_dir = settings.pufferlab_data_dir.resolve()
    target = options.output_path.expanduser()
    if not target.is_absolute():
        target = data_dir / target
    target = target.absolute()
    parent = target.parent.resolve()
    try:
        parent.relative_to(data_dir)
    except ValueError:
        raise EvaluationCommandError(
            "export output must be inside PUFFERLAB_DATA_DIR",
            exit_code=2,
        ) from None
    target = parent / target.name
    if target.is_symlink():
        raise EvaluationCommandError("export output must not be a symbolic link", exit_code=2)
    if target.exists() and target.is_dir():
        raise EvaluationCommandError("export output must be a file path", exit_code=2)
    if target.exists() and not target.is_file():
        raise EvaluationCommandError(
            "existing export output must be a regular file",
            exit_code=2,
        )
    if target.exists() and not options.overwrite:
        raise EvaluationCommandError(
            "export output already exists; pass --overwrite to replace it",
            exit_code=2,
        )
    return target


def _metric_value(value: float | None) -> str:
    return "null" if value is None else format(value, ".6g")
