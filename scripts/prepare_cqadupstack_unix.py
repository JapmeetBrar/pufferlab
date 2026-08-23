"""Verify and deterministically prepare the ignored CQADupStack Unix pack."""

import argparse
from pathlib import Path

from pufferlab.datasets.cqadupstack import (
    load_source_lock,
    prepare_unix_pack,
    source_lock_sha256,
    verify_curated_query_manifest,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Verify the complete pinned CQADupStack archive, then prepare only its Unix subset."
        )
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    archive_path = arguments.archive.resolve()
    output_dir = arguments.output_dir.resolve()
    ignored_data_dir = (root / "data").resolve()
    for label, path in (("archive", archive_path), ("output directory", output_dir)):
        if path.is_relative_to(root) and not path.is_relative_to(ignored_data_dir):
            parser.error(f"{label} inside the repository must be under the ignored data/ directory")
    source_lock = load_source_lock(root / "datasets/cqadupstack-unix/source-lock.json")
    output = prepare_unix_pack(
        archive_path,
        output_dir,
        source_lock,
    )
    verify_curated_query_manifest(
        output,
        root / "datasets/cqadupstack-unix/curated-50.json",
    )
    print(f"ready processed_pack={output} source_lock_sha256={source_lock_sha256(source_lock)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
