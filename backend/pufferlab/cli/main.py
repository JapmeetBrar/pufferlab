"""Argparse entrypoint for focused PufferLab workflows."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO
from uuid import UUID, uuid4

from pufferlab.cli.evaluation import (
    CliApplication,
    CliApplicationFactory,
    CompactEvalProgress,
    ConfigSeedOptions,
    EvalExportOptions,
    EvalRunOptions,
    EvaluationCommandError,
    SeedResult,
    UnixIngestOptions,
    render_run,
    render_seed,
    run_exit_code,
    validate_export_destination,
    write_canonical_export,
)
from pufferlab.cli.ingest import (
    IngestTinyOptions,
    TinyFixtureIngestor,
    TinyIngestionCommandError,
    resolve_owned_namespace,
)
from pufferlab.cli.synthetic_demo import (
    render_synthetic_demo,
    seed_synthetic_demo_database,
)
from pufferlab.config import Settings
from pufferlab.contracts.common import ContractModel
from pufferlab.contracts.evals import EvalRun, EvalRunStatus
from pufferlab.datasets.ingestion import IngestionReport
from pufferlab.synthetic_demo import SyntheticDemoSeedResult

if TYPE_CHECKING:
    from pufferlab.cli.doctor import DoctorDependencies
    from pufferlab.cli.serve import ServeDependencies

_UNIX_PREFIX = "pufferlab-unix-"
_UNIX_DATASET_DIR = Path("datasets/cqadupstack-unix")


class _IngestRunner(Protocol):
    async def __call__(
        self,
        settings: Settings,
        options: IngestTinyOptions,
        *,
        emit: Callable[[str], None],
    ) -> IngestionReport: ...


class _SyntheticDemoSeedRunner(Protocol):
    def __call__(self, settings: Settings) -> SyntheticDemoSeedResult: ...


class _EvaluationInterrupted(Exception):
    """Carry only safe, durable interrupt-cleanup state across ``asyncio.run``."""

    def __init__(
        self,
        *,
        durable_status: EvalRunStatus | None,
        cleanup_failed: bool,
    ) -> None:
        super().__init__("evaluation interrupted")
        self.durable_status = durable_status
        self.cleanup_failed = cleanup_failed


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_factory: Callable[[], Settings] = Settings,
    ingest_runner: _IngestRunner | None = None,
    synthetic_demo_seed_runner: _SyntheticDemoSeedRunner | None = None,
    doctor_dependencies: DoctorDependencies | None = None,
    serve_dependencies: ServeDependencies | None = None,
    cli_application_factory: CliApplicationFactory | None = None,
    run_id_factory: Callable[[], UUID] = uuid4,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    parser = _parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "serve":
        return _run_serve_command(
            arguments,
            dependencies=serve_dependencies,
            output=output,
            error_output=error_output,
        )

    if arguments.command == "doctor":
        if arguments.dataset_version is not None and arguments.mode not in {"evaluation", "all"}:
            parser.error("--dataset-version is accepted only for doctor evaluation or all")
        if arguments.live and arguments.mode == "demo":
            parser.error("--live is accepted only for doctor live-tiny, evaluation, or all")
        return _run_doctor_command(
            arguments,
            settings_factory=settings_factory,
            dependencies=doctor_dependencies,
            output=output,
            error_output=error_output,
        )

    if arguments.command == "dataset" and arguments.dataset_command == "ingest-tiny":
        return _run_tiny_ingest(
            arguments,
            settings_factory=settings_factory,
            ingest_runner=ingest_runner,
            output=output,
            error_output=error_output,
        )

    if arguments.command == "namespace":
        return _run_namespace_command(
            arguments,
            settings_factory=settings_factory,
            output=output,
            error_output=error_output,
        )

    if arguments.command == "demo" and arguments.demo_command == "seed":
        return _run_synthetic_demo_seed(
            settings_factory=settings_factory,
            seed_runner=synthetic_demo_seed_runner,
            output=output,
            error_output=error_output,
        )

    factory = cli_application_factory or _default_application_factory
    try:
        settings = settings_factory()
        if arguments.command == "dataset" and arguments.dataset_command == "ingest-unix":
            namespace = resolve_owned_namespace(
                arguments.namespace,
                generated_prefix=_UNIX_PREFIX,
            )
            unix_options = UnixIngestOptions(
                namespace=namespace,
                processed_pack_path=arguments.processed_pack,
                source_lock_path=arguments.source_lock,
                processed_pack_lock_path=arguments.processed_pack_lock,
                dataset_manifest_path=arguments.dataset_manifest,
                curated_manifest_path=arguments.curated_manifest,
                batch_size=arguments.batch_size,
                max_concurrency=arguments.max_concurrency,
                readiness_attempts=arguments.readiness_attempts,
            )
            _validate_unix_paths(unix_options, settings=settings)
            print(
                "unix ingestion plan (local model execution and remote writes follow)", file=output
            )
            print(f"region={settings.turbopuffer_region}", file=output)
            print(f"namespace={namespace}", file=output)
            print("processed_pack=explicit verified local path", file=output)
            result = asyncio.run(
                _ingest_unix(
                    factory(settings),
                    unix_options,
                    emit=lambda message: print(message, file=output),
                )
            )
            render_seed(result, emit=lambda message: print(message, file=output))
            return 0

        if arguments.command == "config" and arguments.config_command == "seed":
            seed_options = ConfigSeedOptions(dataset_version_id=arguments.dataset_version)
            result = asyncio.run(_seed_configs(factory(settings), seed_options))
            render_seed(result, emit=lambda message: print(message, file=output))
            return 0

        if arguments.command == "eval" and arguments.eval_command == "run":
            run_options = _eval_run_options(arguments, parser)
            run_id = run_id_factory()
            application = factory(settings)
            progress = CompactEvalProgress(lambda message: print(message, file=output))
            try:
                run = asyncio.run(
                    _run_evaluation(
                        application,
                        run_options,
                        run_id=run_id,
                        on_progress=progress,
                    )
                )
            except _EvaluationInterrupted as interrupted:
                _render_evaluation_interrupt(run_id, interrupted, output=error_output)
                return 130
            except KeyboardInterrupt:
                print(
                    f"run_id={run_id} status=interrupt_requested cleanup=unknown",
                    file=error_output,
                )
                return 130
            render_run(run, emit=lambda message: print(message, file=output))
            return run_exit_code(run)

        if arguments.command == "eval" and arguments.eval_command == "export":
            export_options = EvalExportOptions(
                run_id=arguments.run_id,
                output_path=arguments.output,
                overwrite=arguments.overwrite,
            )
            validate_export_destination(export_options, settings=settings)
            export = asyncio.run(_load_export(factory(settings), export_options.run_id))
            path = write_canonical_export(export, export_options, settings=settings)
            print(f"exported run_id={export_options.run_id} path={path}", file=output)
            return 0
    except (EvaluationCommandError, TinyIngestionCommandError) as error:
        print(f"error: {error}", file=error_output)
        return error.exit_code
    except KeyboardInterrupt:
        print("error: command cancelled", file=error_output)
        return 130
    except Exception:
        print(f"error: {_failure_message(arguments)}", file=error_output)
        return 1

    raise AssertionError("argparse returned an unknown command")


def _run_synthetic_demo_seed(
    *,
    settings_factory: Callable[[], Settings],
    seed_runner: _SyntheticDemoSeedRunner | None,
    output: TextIO,
    error_output: TextIO,
) -> int:
    try:
        result = (seed_runner or seed_synthetic_demo_database)(settings_factory())
        render_synthetic_demo(result, emit=lambda message: print(message, file=output))
    except KeyboardInterrupt:
        print("error: command cancelled", file=error_output)
        return 130
    except Exception:
        print("error: synthetic demo seed failed", file=error_output)
        return 1
    return 0


def _run_doctor_command(
    arguments: argparse.Namespace,
    *,
    settings_factory: Callable[[], Settings],
    dependencies: DoctorDependencies | None,
    output: TextIO,
    error_output: TextIO,
) -> int:
    from pufferlab.cli.doctor import DoctorMode, render_doctor, run_doctor

    try:
        execution = asyncio.run(
            run_doctor(
                settings_factory(),
                mode=DoctorMode(arguments.mode),
                dataset_version_id=arguments.dataset_version,
                live=arguments.live,
                dependencies=dependencies,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("error: doctor cancelled", file=error_output)
        return 130
    except Exception:
        print("error: doctor failed", file=error_output)
        return 1
    if execution.exit_code == 130:
        print("error: doctor cancelled", file=error_output)
        return 130
    for line in render_doctor(execution.report):
        print(line, file=output)
    return execution.exit_code


def _run_serve_command(
    arguments: argparse.Namespace,
    *,
    dependencies: ServeDependencies | None,
    output: TextIO,
    error_output: TextIO,
) -> int:
    from pufferlab.cli.serve import (
        ServeOptions,
        render_serve_finish,
        render_serve_start,
        run_serve,
    )

    options = ServeOptions(host=arguments.host, port=arguments.port)
    print(render_serve_start(options), file=output, flush=True)
    execution = run_serve(options, dependencies=dependencies)
    output_lines, error_lines = render_serve_finish(execution)
    for line in output_lines:
        print(line, file=output, flush=True)
    for line in error_lines:
        print(line, file=error_output, flush=True)
    return execution.exit_code


def _run_tiny_ingest(
    arguments: argparse.Namespace,
    *,
    settings_factory: Callable[[], Settings],
    ingest_runner: _IngestRunner | None,
    output: TextIO,
    error_output: TextIO,
) -> int:
    options = IngestTinyOptions(
        namespace=arguments.namespace,
        batch_size=arguments.batch_size,
        max_concurrency=arguments.max_concurrency,
        readiness_attempts=arguments.readiness_attempts,
    )
    runner = ingest_runner or TinyFixtureIngestor().run
    try:
        asyncio.run(
            runner(
                settings_factory(),
                options,
                emit=lambda message: print(message, file=output),
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("error: tiny fixture ingestion cancelled", file=error_output)
        return 130
    except SystemExit:
        print("error: tiny fixture ingestion failed", file=error_output)
        return 1
    except TinyIngestionCommandError as error:
        print(f"error: {error}", file=error_output)
        return error.exit_code
    except Exception:
        print("error: tiny fixture ingestion failed", file=error_output)
        return 1
    return 0


def _run_namespace_command(
    arguments: argparse.Namespace,
    *,
    settings_factory: Callable[[], Settings],
    output: TextIO,
    error_output: TextIO,
) -> int:
    from pufferlab.cli.namespace import (
        NamespaceCommandError,
        cleanup_owned_tiny,
        show_owned_tiny,
    )

    try:
        if arguments.namespace_command == "show-tiny":
            show_owned_tiny(emit=lambda message: print(message, file=output))
        elif arguments.namespace_command == "cleanup-tiny":
            asyncio.run(
                cleanup_owned_tiny(
                    settings_factory(),
                    emit=lambda message: print(message, file=output),
                )
            )
        else:  # pragma: no cover - argparse freezes the subcommand domain
            raise AssertionError("unknown namespace command")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("error: namespace command cancelled", file=error_output)
        return 130
    except SystemExit:
        print("error: namespace command failed", file=error_output)
        return 1
    except NamespaceCommandError as error:
        print(f"error: {error}", file=error_output)
        return error.exit_code
    except Exception:
        print("error: namespace command failed", file=error_output)
        return 1
    return 0


async def _ingest_unix(
    application: CliApplication,
    options: UnixIngestOptions,
    *,
    emit: Callable[[str], None],
) -> SeedResult:
    try:
        return await application.ingest_unix(options, emit=emit)
    finally:
        await application.close()


async def _seed_configs(
    application: CliApplication,
    options: ConfigSeedOptions,
) -> SeedResult:
    try:
        return application.seed(options)
    finally:
        await application.close()


async def _run_evaluation(
    application: CliApplication,
    options: EvalRunOptions,
    *,
    run_id: UUID,
    on_progress: CompactEvalProgress,
) -> EvalRun:
    interrupted = False
    try:
        return await application.run(
            options,
            run_id=run_id,
            on_progress=on_progress,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        durable_status: EvalRunStatus | None = None
        cleanup_failed = False
        try:
            durable = await asyncio.shield(application.cancel_and_drain(run_id))
            durable_status = durable.status
        except (Exception, asyncio.CancelledError, KeyboardInterrupt):
            cleanup_failed = True
        try:
            await asyncio.shield(application.close())
        except (Exception, asyncio.CancelledError, KeyboardInterrupt):
            cleanup_failed = True
        raise _EvaluationInterrupted(
            durable_status=durable_status,
            cleanup_failed=cleanup_failed,
        ) from None
    finally:
        if not interrupted:
            await application.close()


def _render_evaluation_interrupt(
    run_id: UUID,
    interrupted: _EvaluationInterrupted,
    *,
    output: TextIO,
) -> None:
    fields = [f"run_id={run_id}"]
    if interrupted.durable_status is None:
        fields.append("status=interrupt_requested")
    else:
        fields.extend(
            (
                f"status={interrupted.durable_status.value}",
                "interrupt_requested=true",
            )
        )
    if interrupted.cleanup_failed:
        fields.append("cleanup=failed")
    print(" ".join(fields), file=output)


async def _load_export(application: CliApplication, run_id: UUID) -> ContractModel:
    try:
        return application.export(run_id)
    finally:
        await application.close()


def _eval_run_options(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> EvalRunOptions:
    explicit_values = (
        arguments.query_set,
        arguments.baseline,
        arguments.candidates,
    )
    if arguments.seeded_defaults:
        if any(value for value in explicit_values):
            parser.error(
                "--seeded-defaults cannot be combined with --query-set, --baseline, or --candidate"
            )
    elif arguments.query_set is None or arguments.baseline is None or not arguments.candidates:
        parser.error(
            "eval run requires --seeded-defaults or all of --query-set, --baseline, and --candidate"
        )
    candidate_ids = tuple(arguments.candidates)
    if len(candidate_ids) > 3:
        parser.error("--candidate may be repeated at most three times")
    if len(set(candidate_ids)) != len(candidate_ids):
        parser.error("--candidate values must be unique")
    if arguments.baseline in candidate_ids:
        parser.error("the baseline cannot also be a candidate")
    return EvalRunOptions(
        query_set_id=arguments.query_set,
        baseline_config_id=arguments.baseline,
        candidate_config_ids=candidate_ids,
        seeded_defaults=arguments.seeded_defaults,
        random_seed=arguments.random_seed,
        max_concurrency=arguments.max_concurrency,
        warmup_query_count=arguments.warmup_count,
    )


def _validate_unix_paths(options: UnixIngestOptions, *, settings: Settings) -> None:
    data_dir = settings.pufferlab_data_dir.resolve()
    processed = options.processed_pack_path.expanduser().resolve()
    try:
        processed.relative_to(data_dir)
    except ValueError:
        raise EvaluationCommandError(
            "processed pack must be inside PUFFERLAB_DATA_DIR",
            exit_code=2,
        ) from None
    if options.processed_pack_path.is_symlink() or not processed.is_dir():
        raise EvaluationCommandError(
            "processed pack must be an existing non-symbolic-link directory",
            exit_code=2,
        )
    checked_paths = (
        options.source_lock_path,
        options.processed_pack_lock_path,
        options.dataset_manifest_path,
        options.curated_manifest_path,
    )
    if any(path.is_symlink() or not path.is_file() for path in checked_paths):
        raise EvaluationCommandError(
            "Unix source locks and manifests must be existing non-symbolic-link files",
            exit_code=2,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pufferlab",
        description="PufferLab local dataset and judged-evaluation workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    _add_doctor_parser(commands)
    _add_serve_parser(commands)

    dataset = commands.add_parser("dataset", help="Prepare and ingest evaluation datasets.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    _add_tiny_ingest_parser(dataset_commands)
    _add_unix_ingest_parser(dataset_commands)

    namespace = commands.add_parser(
        "namespace",
        help="Inspect or clean the one authenticated generated tiny namespace.",
    )
    namespace_commands = namespace.add_subparsers(dest="namespace_command", required=True)
    namespace_commands.add_parser(
        "show-tiny",
        help="Print region and namespace assignments from the authenticated fixed receipt.",
        description=(
            "Print TURBOPUFFER_REGION and PUFFERLAB_SEARCH_NAMESPACE assignments from the one "
            "authenticated generated-tiny receipt. No provider request is made."
        ),
    )
    namespace_commands.add_parser(
        "cleanup-tiny",
        help="Delete only the exact authenticated generated tiny namespace.",
        description=(
            "Delete only the target in the fixed authenticated generated-tiny receipt, then "
            "perform bounded not-found verification. Verification metadata requests may be "
            "billed as zero-row queries. No target, path, token, or ownership input is accepted."
        ),
    )

    config = commands.add_parser("config", help="Manage immutable retrieval configurations.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    seed = config_commands.add_parser(
        "seed",
        help="Persist the four canonical retrieval configurations.",
        description=(
            "Persist the four canonical retrieval configurations—BM25, ANN, server RRF, and "
            "local reranker—for one READY dataset. PUFFERLAB_DATA_DIR selects the local SQLite "
            "state. No provider write is performed."
        ),
    )
    seed.add_argument(
        "--dataset-version",
        type=_uuid,
        help="READY dataset revision UUID; omit only when one seeded Unix default is unambiguous.",
    )

    evaluation = commands.add_parser("eval", help="Run and export durable judged evaluations.")
    evaluation_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    _add_eval_run_parser(evaluation_commands)
    _add_eval_export_parser(evaluation_commands)

    demo = commands.add_parser("demo", help="Manage the provider-free offline dashboard demo.")
    demo_commands = demo.add_subparsers(dest="demo_command", required=True)
    demo_commands.add_parser(
        "seed",
        help="Seed one deterministic synthetic 50-query evaluation into local SQLite.",
        description=(
            "Seed one deterministic, read/export-only synthetic evaluation into the configured "
            "PUFFERLAB_DATA_DIR. This command requires no API key, model, provider, or network "
            "access and writes no export artifact. Re-running it verifies the same durable state."
        ),
    )
    return parser


def _add_doctor_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    doctor = commands.add_parser(
        "doctor",
        help="Inspect local PufferLab readiness without changing durable state.",
        description=(
            "Inspect provider-free local readiness. --live explicitly adds at most one "
            "metadata-only turbopuffer request; no search, write, create, or delete is issued."
        ),
    )
    doctor.add_argument(
        "--mode",
        choices=("demo", "live-tiny", "evaluation", "all"),
        required=True,
    )
    doctor.add_argument(
        "--dataset-version",
        type=_uuid,
        help="Exact persisted evaluation dataset UUID (evaluation/all only).",
    )
    doctor.add_argument(
        "--live",
        action="store_true",
        help="Explicitly perform one metadata-only check for the resolved live target.",
    )


def _add_serve_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    serve = commands.add_parser(
        "serve",
        help="Serve the local API with one loopback-only worker.",
        description=(
            "Serve the local API with one worker on an exact loopback address. Worker count, "
            "reload, proxy headers, application target, logging, and shutdown bounds are fixed."
        ),
    )
    serve.add_argument(
        "--host",
        type=_serve_host,
        default="127.0.0.1",
        help="Exact loopback host: 127.0.0.1, ::1, or localhost (default: 127.0.0.1).",
    )
    serve.add_argument(
        "--port",
        type=_serve_port,
        default=8000,
        help="Loopback TCP port from 1 through 65535 (default: 8000).",
    )


def _add_tiny_ingest_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ingest = commands.add_parser(
        "ingest-tiny",
        help="Embed and upsert the checked-in 20-document fixture.",
        description=(
            "Embed and upsert the checked-in tiny fixture. This performs local model execution "
            "and cost-bearing turbopuffer writes. TURBOPUFFER_API_KEY is required; "
            "TURBOPUFFER_REGION selects the creating region. Without --namespace, the command "
            "durably creates or resumes the one authenticated generated-tiny receipt."
        ),
    )
    _add_ingest_arguments(ingest, default_batch_size=20)


def _add_unix_ingest_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ingest = commands.add_parser(
        "ingest-unix",
        help="Resume the verified CQADupStack Unix ingestion and persist its eval seed.",
        description=(
            "Verify an ignored CQADupStack Unix processed pack, run the pinned local embedding "
            "model, perform cost-bearing turbopuffer writes, and persist the READY dataset and "
            "50-query seed. TURBOPUFFER_API_KEY, TURBOPUFFER_REGION, PUFFERLAB_DATA_DIR, and an "
            "explicit --processed-pack are required inputs."
        ),
    )
    _add_ingest_arguments(ingest, default_batch_size=64)
    ingest.add_argument(
        "--processed-pack",
        type=Path,
        required=True,
        help="Ignored content-addressed processed-pack directory under PUFFERLAB_DATA_DIR.",
    )
    ingest.add_argument(
        "--source-lock",
        type=Path,
        default=_UNIX_DATASET_DIR / "source-lock.json",
        help="Checked source lock (default: datasets/cqadupstack-unix/source-lock.json).",
    )
    ingest.add_argument(
        "--processed-pack-lock",
        type=Path,
        default=_UNIX_DATASET_DIR / "processed-pack-lock.json",
        help="Checked processed-pack lock.",
    )
    ingest.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_UNIX_DATASET_DIR / "dataset-manifest.json",
        help="Checked Unix dataset/index manifest.",
    )
    ingest.add_argument(
        "--curated-manifest",
        type=Path,
        default=_UNIX_DATASET_DIR / "curated-50.json",
        help="Checked ID-only curated 50-query manifest.",
    )


def _add_ingest_arguments(parser: argparse.ArgumentParser, *, default_batch_size: int) -> None:
    parser.add_argument(
        "--namespace",
        help=(
            "Caller-managed explicit pufferlab-* target for an idempotent rerun. Explicit "
            "targets never receive generated-tiny cleanup authority. Omit to use the "
            "command-specific generated target."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=default_batch_size,
        help=f"Documents per embedding/upsert batch (default: {default_batch_size}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=2,
        help="Maximum concurrent ingestion batches (default: 2).",
    )
    parser.add_argument(
        "--readiness-attempts",
        type=_positive_int,
        default=180,
        help="Bounded metadata/index readiness checks (default: 180 at 0.5s intervals).",
    )


def _add_eval_run_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = commands.add_parser(
        "run",
        help="Execute and incrementally persist one judged evaluation.",
        description=(
            "Execute a judged query set through turbopuffer and pinned local embedding/reranking "
            "models. TURBOPUFFER_API_KEY and TURBOPUFFER_REGION select the cost-bearing provider "
            "target; PUFFERLAB_DATA_DIR stores durable SQLite state. Progress is emitted only "
            "after outcomes commit."
        ),
    )
    run.add_argument(
        "--seeded-defaults",
        action="store_true",
        help="Use the canonical 50-query set, BM25 baseline, and other three seeded candidates.",
    )
    run.add_argument("--query-set", type=_uuid, help="Persisted judged query-set UUID.")
    run.add_argument("--baseline", type=_uuid, help="Persisted baseline config UUID.")
    run.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        type=_uuid,
        default=[],
        help="Persisted candidate config UUID; repeat one to three times.",
    )
    run.add_argument("--random-seed", type=int, default=20260822)
    run.add_argument("--max-concurrency", type=_positive_int, default=4)
    run.add_argument("--warmup-count", type=_nonnegative_int, default=5)


def _add_eval_export_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    export = commands.add_parser(
        "export",
        help="Write one durable run and its outcomes as canonical JSON.",
        description=(
            "Export completed, failed, cancelled, interrupted, or partial durable state without "
            "inventing outcomes. The explicit output must stay under ignored PUFFERLAB_DATA_DIR; "
            "credential values, raw vectors, and provider request data are never printed."
        ),
    )
    export.add_argument("run_id", type=_uuid, help="Durable evaluation run UUID.")
    export.add_argument("--format", choices=("json",), default="json")
    export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit file path inside PUFFERLAB_DATA_DIR (relative paths resolve beneath it).",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing regular export file; symbolic links are refused.",
    )


def _failure_message(arguments: argparse.Namespace) -> str:
    if arguments.command == "dataset":
        return "Unix dataset ingestion failed"
    if arguments.command == "config":
        return "configuration seed failed"
    if arguments.eval_command == "run":
        return "evaluation run failed"
    return "evaluation export failed"


def _default_application_factory(settings: Settings) -> CliApplication:
    from pufferlab.cli.runtime import RuntimeCliApplication

    return RuntimeCliApplication(settings)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a UUID") from None


def _serve_host(value: str) -> str:
    if value not in {"127.0.0.1", "::1", "localhost"}:
        raise argparse.ArgumentTypeError("must be an exact loopback host")
    return value


def _serve_port(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("must be an integer from 1 through 65535")
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be an integer from 1 through 65535")
    return port
