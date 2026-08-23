import hashlib
from collections import Counter
from pathlib import Path

from pufferlab.datasets.artifact_audit import (
    _content_violation,
    _path_violation,
    audit_repository,
)
from pufferlab.datasets.cqadupstack import (
    ForbiddenTokenWindow,
    load_curated_query_manifest,
    load_source_lock,
    source_lock_sha256,
)

REPOSITORY_ROOT = Path(__file__).parents[3]


def test_path_inventory_rejects_real_dataset_artifact_shapes() -> None:
    assert _path_violation("data/archive.zip") == ".zip artifacts must remain outside Git"
    assert _path_violation("outside/corpus.jsonl") == (
        "non-synthetic JSONL must remain outside Git"
    )
    assert _path_violation("datasets/example/processed/manifest.json") == (
        "raw/processed/cache/export/evidence directories must remain outside Git"
    )
    assert _path_violation("fixtures/tiny-corpus/documents.jsonl") is None
    assert _path_violation("datasets/cqadupstack-unix/source-lock.json") is None
    assert _path_violation("outside/embeddings.bin") == (".bin artifacts must remain outside Git")
    assert _path_violation("outside/archive.tar.gz") == (".gz artifacts must remain outside Git")
    assert _path_violation("outside/pufferlab.sqlite3-wal") == (
        "database journal artifacts must remain outside Git"
    )
    assert _path_violation("outside/evaluation.log.1") == ("log artifacts must remain outside Git")


def test_content_inventory_rejects_archives_and_casefolded_hashed_source_windows() -> None:
    assert _content_violation(b"prefix PK\x03\x04 suffix", ()) == (
        "ZIP archive signature is forbidden"
    )
    synthetic = "synthetic forbidden window"
    lock = (
        ForbiddenTokenWindow(
            token_count=3,
            sha256=hashlib.sha256(synthetic.encode()).hexdigest(),
        ),
    )

    assert _content_violation(b"prefix SYNTHETIC FORBIDDEN WINDOW suffix", lock) == (
        "known upstream source-text token window is forbidden"
    )
    assert _content_violation(b"ordinary tracked source code", lock) is None


def test_checked_manifests_and_entire_repository_pass_artifact_audit() -> None:
    source_lock = load_source_lock(REPOSITORY_ROOT / "datasets/cqadupstack-unix/source-lock.json")
    curated = load_curated_query_manifest(
        REPOSITORY_ROOT / "datasets/cqadupstack-unix/curated-50.json"
    )

    assert curated.source_lock_sha256 == source_lock_sha256(source_lock)
    assert Counter(entry.primary_tag for entry in curated.entries) == {
        "exact_token": 13,
        "semantic": 13,
        "hybrid": 12,
        "reranker": 12,
    }
    report = audit_repository(REPOSITORY_ROOT, source_lock)
    assert report.current_files_scanned > 0
    assert report.historical_blobs_scanned > 0
    assert report.ignored_paths_verified == 19
