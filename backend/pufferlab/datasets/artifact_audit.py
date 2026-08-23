"""Repository-wide enforcement of the licensed dataset artifact boundary."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pufferlab.datasets.cqadupstack import ForbiddenTokenWindow, SourceLock

_FORBIDDEN_SUFFIXES = {
    ".7z",
    ".arrow",
    ".bin",
    ".bz2",
    ".csv",
    ".db",
    ".feather",
    ".gz",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".partial",
    ".pickle",
    ".pkl",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tsv",
    ".tgz",
    ".xz",
    ".zip",
    ".zst",
}
_FORBIDDEN_PATH_PARTS = {"cache", "evidence", "exports", "processed", "raw"}
_ALLOWED_JSONL = {
    "fixtures/tiny-corpus/documents.jsonl",
    "fixtures/tiny-corpus/queries.jsonl",
}
_EXPECTED_IGNORED = (
    "data/cqadupstack.zip",
    "data/cqadupstack.zip.part",
    "data/cqadupstack.partial",
    "data/datasets/cqadupstack-unix/raw/corpus.jsonl",
    "data/datasets/cqadupstack-unix/raw/queries.jsonl",
    "data/datasets/cqadupstack-unix/raw/qrels/test.tsv",
    "data/datasets/cqadupstack-unix/processed/documents.jsonl",
    "data/datasets/cqadupstack-unix/processed/attribution.jsonl",
    "data/datasets/cqadupstack-unix/cache/model.bin",
    "data/datasets/cqadupstack-unix/embeddings.npy",
    "data/datasets/cqadupstack-unix/vectors.jsonl",
    "data/pufferlab.sqlite3",
    "data/exports/run.json",
    "data/logs/evaluation.log",
    "datasets/cqadupstack-unix/raw/corpus.jsonl",
    "datasets/cqadupstack-unix/processed/documents.jsonl",
    "datasets/cqadupstack-unix/cache/model.bin",
    "datasets/cqadupstack-unix/exports/run.json",
    "datasets/cqadupstack-unix/evidence/live.json",
)


@dataclass(frozen=True, slots=True)
class ArtifactAuditReport:
    current_files_scanned: int
    historical_blobs_scanned: int
    ignored_paths_verified: int


class DatasetArtifactAuditError(RuntimeError):
    """A tracked/current or historical blob crosses the licensed artifact boundary."""


def audit_repository(root: Path, source_lock: SourceLock) -> ArtifactAuditReport:
    root = root.resolve()
    _assert_full_history(root)
    current_paths = _current_candidate_paths(root)
    violations: list[str] = []
    current_files_scanned = 0
    for relative in current_paths:
        violation = _path_violation(relative)
        if violation is not None:
            violations.append(f"current path {relative}: {violation}")
            continue
        data = (root / relative).read_bytes()
        current_files_scanned += 1
        content_violation = _content_violation(
            data,
            source_lock.forbidden_source_token_windows,
        )
        if content_violation is not None:
            violations.append(f"current path {relative}: {content_violation}")

    historical_blobs_scanned = 0
    for object_id, historical_path in _historical_blobs(root):
        violation = _path_violation(historical_path)
        if violation is not None:
            violations.append(f"historical path {historical_path}: {violation}")
            continue
        data = _git_bytes(root, "cat-file", "blob", object_id)
        historical_blobs_scanned += 1
        content_violation = _content_violation(
            data,
            source_lock.forbidden_source_token_windows,
        )
        if content_violation is not None:
            violations.append(
                f"historical blob {object_id} ({historical_path}): {content_violation}"
            )

    for relative in _EXPECTED_IGNORED:
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            violations.append(f"ignored inventory path is exposed: {relative}")

    if violations:
        detail = "\n".join(sorted(set(violations)))
        raise DatasetArtifactAuditError(f"dataset artifact audit failed:\n{detail}")
    return ArtifactAuditReport(
        current_files_scanned=current_files_scanned,
        historical_blobs_scanned=historical_blobs_scanned,
        ignored_paths_verified=len(_EXPECTED_IGNORED),
    )


def _assert_full_history(root: Path) -> None:
    shallow = _git_text(root, "rev-parse", "--is-shallow-repository").strip()
    if shallow not in {"true", "false"}:
        raise DatasetArtifactAuditError("git returned an invalid shallow-repository state")
    if shallow == "true":
        raise DatasetArtifactAuditError(
            "repository history is shallow; whole-history dataset audit cannot run"
        )


def _current_candidate_paths(root: Path) -> tuple[str, ...]:
    output = _git_text(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _historical_blobs(root: Path) -> tuple[tuple[str, str], ...]:
    output = _git_text(root, "rev-list", "--objects", "--all")
    candidates: list[tuple[str, str]] = []
    for line in output.splitlines():
        object_id, separator, path = line.partition(" ")
        if not separator or not path:
            continue
        kind = _git_text(root, "cat-file", "-t", object_id)
        if kind.strip() == "blob":
            candidates.append((object_id, path))
    return tuple(candidates)


def _path_violation(relative: str) -> str | None:
    path = PurePosixPath(relative)
    suffix = path.suffix.lower()
    normalized = path.as_posix()
    if suffix == ".jsonl" and normalized not in _ALLOWED_JSONL:
        return "non-synthetic JSONL must remain outside Git"
    if suffix in _FORBIDDEN_SUFFIXES:
        return f"{suffix} artifacts must remain outside Git"
    if re.search(r"(?:\.db|\.sqlite3?)(?:-(?:journal|shm|wal))$", path.name.lower()):
        return "database journal artifacts must remain outside Git"
    if re.search(r"\.log(?:\.\d+)?$", path.name.lower()):
        return "log artifacts must remain outside Git"
    if _FORBIDDEN_PATH_PARTS.intersection(part.lower() for part in path.parts):
        return "raw/processed/cache/export/evidence directories must remain outside Git"
    return None


def _content_violation(
    data: bytes,
    forbidden_windows: tuple[ForbiddenTokenWindow, ...],
) -> str | None:
    if b"PK\x03\x04" in data or b"PK\x05\x06" in data or b"PK\x06\x06" in data:
        return "ZIP archive signature is forbidden"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    tokens = re.findall(r"\S+", text)
    grouped: dict[int, set[str]] = {}
    for forbidden in forbidden_windows:
        grouped.setdefault(forbidden.token_count, set()).add(forbidden.sha256)
    for token_count, forbidden_hashes in grouped.items():
        for offset in range(0, len(tokens) - token_count + 1):
            window = " ".join(tokens[offset : offset + token_count]).casefold().encode()
            if hashlib.sha256(window).hexdigest() in forbidden_hashes:
                return "known upstream source-text token window is forbidden"
    return None


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DatasetArtifactAuditError(
            f"git {' '.join(arguments)} failed during dataset audit: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace")
        raise DatasetArtifactAuditError(
            f"git {' '.join(arguments)} failed during dataset audit: {stderr.strip()}"
        )
    return completed.stdout
