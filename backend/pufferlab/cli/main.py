"""Argparse entrypoint for focused PufferLab workflows."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from typing import Protocol, TextIO

from pufferlab.cli.ingest import (
    IngestTinyOptions,
    TinyFixtureIngestor,
    TinyIngestionCommandError,
)
from pufferlab.config import Settings
from pufferlab.datasets.ingestion import IngestionReport


class _IngestRunner(Protocol):
    async def __call__(
        self,
        settings: Settings,
        options: IngestTinyOptions,
        *,
        emit: Callable[[str], None],
    ) -> IngestionReport: ...


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_factory: Callable[[], Settings] = Settings,
    ingest_runner: _IngestRunner | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    arguments = _parser().parse_args(argv)
    if arguments.command != "dataset" or arguments.dataset_command != "ingest-tiny":
        raise AssertionError("argparse returned an unknown command")

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
    except TinyIngestionCommandError as error:
        print(f"error: {error}", file=error_output)
        return error.exit_code
    except Exception:
        print("error: tiny fixture ingestion failed", file=error_output)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pufferlab",
        description="PufferLab local workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Manage fixture datasets.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    ingest = dataset_commands.add_parser(
        "ingest-tiny",
        help="Embed and upsert the checked-in 20-document fixture.",
        description=(
            "Embed and upsert the checked-in tiny fixture. This performs local model execution "
            "and cost-bearing turbopuffer writes. TURBOPUFFER_API_KEY is required; "
            "TURBOPUFFER_REGION selects the target region."
        ),
    )
    ingest.add_argument(
        "--namespace",
        help=(
            "Explicit owned pufferlab-* target for an idempotent rerun. "
            "Omit to generate a unique random pufferlab-tiny-* namespace."
        ),
    )
    ingest.add_argument(
        "--batch-size",
        type=_positive_int,
        default=20,
        help="Documents per embedding/upsert batch (default: 20).",
    )
    ingest.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=2,
        help="Maximum concurrent ingestion batches (default: 2).",
    )
    ingest.add_argument(
        "--readiness-attempts",
        type=_positive_int,
        default=180,
        help="Bounded metadata/index readiness checks (default: 180 at 0.5s intervals).",
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed
