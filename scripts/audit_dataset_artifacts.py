"""Fail CI when licensed/raw dataset artifacts cross the Git boundary."""

from pathlib import Path

from pufferlab.datasets.artifact_audit import audit_repository
from pufferlab.datasets.cqadupstack import load_source_lock


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source_lock = load_source_lock(root / "datasets/cqadupstack-unix/source-lock.json")
    report = audit_repository(root, source_lock)
    print(
        "dataset artifact audit passed "
        f"current_files={report.current_files_scanned} "
        f"historical_blobs={report.historical_blobs_scanned} "
        f"ignored_paths={report.ignored_paths_verified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
