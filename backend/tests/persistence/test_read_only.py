from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pufferlab.persistence.read_only as read_only_module
import pytest
from pufferlab.contracts.datasets import DatasetVersion
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.read_only import (
    ReadOnlyCatalogError,
    open_existing_read_only_catalog,
)
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo

_READ_ONLY_MODULE_FILE = Path(read_only_module.__file__).resolve()


def _database(path: Path) -> Path:
    with Database(path) as database:
        database.migrate()
    return path


def _database_with_dataset(
    path: Path,
    *,
    replacement: bool = False,
) -> tuple[Path, DatasetVersion]:
    dataset = materialize_synthetic_demo().dataset_version
    if replacement:
        dataset = dataset.model_copy(update={"id": uuid4()})
    with Database(path) as database:
        database.migrate()
        PufferLabRepository(database.session_factory).put_dataset_version(dataset)
    return path, dataset


def _snapshot(path: Path) -> tuple[str, tuple[int, int, int, int], tuple[str, ...]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = path.stat()
    siblings = tuple(sorted(item.name for item in path.parent.iterdir()))
    return (
        digest,
        (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),
        siblings,
    )


def test_existing_catalog_is_byte_mtime_and_sidecar_stable_for_uri_metacharacters(
    tmp_path: Path,
) -> None:
    source = _database(tmp_path / "source.sqlite3")
    path = tmp_path / "uri ?#% space" / "catalog ?#%.sqlite3"
    path.parent.mkdir()
    source.replace(path)
    before = _snapshot(path)

    with open_existing_read_only_catalog(path) as catalog:
        assert catalog.repository.list_dataset_versions(limit=2) == []
        assert repr(catalog) == "ExistingReadOnlyCatalog(state=open)"

    after = _snapshot(path)
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert before[2] == after[2] == (path.name,)
    assert repr(catalog) == "ExistingReadOnlyCatalog(state=closed)"


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_existing_catalog_rejects_every_preexisting_hot_sidecar(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = _database(tmp_path / "catalog.sqlite3")
    Path(f"{path}{suffix}").write_bytes(b"hostile")

    with pytest.raises(ReadOnlyCatalogError) as error:
        open_existing_read_only_catalog(path)

    assert str(error.value) == "existing diagnostic catalog is unavailable or invalid"
    assert str(path) not in repr(error.value)


def test_existing_catalog_rejects_absent_symlink_and_non_regular_targets(tmp_path: Path) -> None:
    missing = tmp_path / "absent-hostile-marker" / "catalog.sqlite3"
    target = _database(tmp_path / "real.sqlite3")
    symlink = tmp_path / "linked.sqlite3"
    symlink.symlink_to(target)
    directory = tmp_path / "directory.sqlite3"
    directory.mkdir()

    for path in (missing, symlink, directory):
        with pytest.raises(ReadOnlyCatalogError) as error:
            open_existing_read_only_catalog(path)
        assert "hostile-marker" not in _traceback_locals(error.value)
    assert not missing.parent.exists()


def test_existing_catalog_resolves_parent_symlink_to_one_fixed_regular_file(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    path = _database(real_parent / "catalog.sqlite3")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    before = _snapshot(path)

    with open_existing_read_only_catalog(linked_parent / path.name):
        pass

    assert _snapshot(path) == before


@pytest.mark.parametrize("kind", ["corrupt", "truncated", "missing_schema", "future_schema"])
def test_existing_catalog_rejects_corrupt_or_wrong_schema(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"{kind}.sqlite3"
    if kind == "corrupt":
        path.write_bytes(b"not sqlite hostile marker")
    elif kind == "truncated":
        valid = _database(tmp_path / "source.sqlite3").read_bytes()
        path.write_bytes(valid[: max(1, len(valid) // 3)])
    else:
        sqlite3.connect(path).close()
        with sqlite3.connect(path) as connection:
            if kind == "future_schema":
                connection.execute("CREATE TABLE alembic_version (version_num TEXT NOT NULL)")
                connection.execute("INSERT INTO alembic_version VALUES ('20990101_future')")

    before = _snapshot(path)
    with pytest.raises(ReadOnlyCatalogError):
        open_existing_read_only_catalog(path)
    assert _snapshot(path) == before


def test_existing_catalog_rejects_foreign_key_violation_without_writing(tmp_path: Path) -> None:
    path = _database(tmp_path / "foreign.sqlite3")
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            """INSERT INTO retrieval_configs
               (id, revision, dataset_version_id, name, config_hash, created_at, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("a" * 36, 1, "missing", "bad", "bad", "2026-01-01", "{}"),
        )
    before = _snapshot(path)

    with pytest.raises(ReadOnlyCatalogError):
        open_existing_read_only_catalog(path)

    assert _snapshot(path) == before


def test_existing_catalog_detects_identity_substitution_before_close(tmp_path: Path) -> None:
    path = _database(tmp_path / "catalog.sqlite3")
    replacement = _database(tmp_path / "replacement.sqlite3")
    catalog = open_existing_read_only_catalog(path)
    path.replace(tmp_path / "old.sqlite3")
    replacement.replace(path)

    with pytest.raises(ReadOnlyCatalogError):
        catalog.close()


def test_existing_catalog_repository_reads_stay_on_validated_handle_after_path_swap(
    tmp_path: Path,
) -> None:
    path, original_dataset = _database_with_dataset(tmp_path / "catalog.sqlite3")
    replacement, replacement_dataset = _database_with_dataset(
        tmp_path / "replacement.sqlite3",
        replacement=True,
    )
    original_before = _snapshot(path)
    replacement_before = _snapshot(replacement)
    displaced = tmp_path / "displaced.sqlite3"
    catalog = open_existing_read_only_catalog(path)
    path.replace(displaced)
    replacement.replace(path)
    try:
        assert catalog.repository.list_dataset_versions(limit=2) == [original_dataset]
        assert catalog.repository.list_dataset_versions(limit=2) != [replacement_dataset]
    finally:
        path.replace(replacement)
        displaced.replace(path)
        catalog.close()

    assert _snapshot(path) == original_before
    assert _snapshot(replacement) == replacement_before


def test_existing_catalog_refuses_reopen_after_pinned_connection_invalidation(
    tmp_path: Path,
) -> None:
    path, _original_dataset = _database_with_dataset(tmp_path / "catalog.sqlite3")
    replacement, _replacement_dataset = _database_with_dataset(
        tmp_path / "replacement.sqlite3",
        replacement=True,
    )
    original_before = _snapshot(path)
    replacement_before = _snapshot(replacement)
    displaced = tmp_path / "displaced.sqlite3"
    catalog = open_existing_read_only_catalog(path)
    path.replace(displaced)
    replacement.replace(path)
    try:
        catalog._engine.dispose()
        with pytest.raises(ReadOnlyCatalogError):
            catalog.repository.list_dataset_versions(limit=2)
    finally:
        path.replace(replacement)
        displaced.replace(path)
        catalog.close()

    assert _snapshot(path) == original_before
    assert _snapshot(replacement) == replacement_before


def test_open_does_not_convert_process_control_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _database(tmp_path / "catalog.sqlite3")

    def interrupt(_engine: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("pufferlab.persistence.read_only._validate_catalog", interrupt)
    with pytest.raises(KeyboardInterrupt) as error:
        open_existing_read_only_catalog(path)
    assert str(path) not in _traceback_locals(error.value)


def test_close_dispose_failure_is_value_free_and_still_checks_identity(
    tmp_path: Path,
) -> None:
    path = _database(tmp_path / "catalog.sqlite3")
    catalog = open_existing_read_only_catalog(path)

    class FailingEngine:
        def dispose(self) -> None:
            raise RuntimeError(f"hostile path {path}")

    catalog._engine = cast(Any, FailingEngine())
    with pytest.raises(ReadOnlyCatalogError) as error:
        catalog.close()
    assert str(path) not in str(error.value)
    assert str(path) not in _traceback_locals(error.value)


def test_close_dispose_interrupt_is_fresh_and_has_no_catalog_path(tmp_path: Path) -> None:
    path = _database(tmp_path / "interrupt-hostile-marker.sqlite3")
    catalog = open_existing_read_only_catalog(path)

    class InterruptingEngine:
        def dispose(self) -> None:
            raise KeyboardInterrupt

    catalog._engine = cast(Any, InterruptingEngine())
    with pytest.raises(KeyboardInterrupt) as error:
        catalog.close()
    assert "interrupt-hostile-marker" not in _traceback_locals(error.value)


def test_traceback_filter_excludes_checkout_named_pufferlab_caller_frames(
    tmp_path: Path,
) -> None:
    marker = "caller-frame-hostile-marker"

    def caller() -> None:
        caller_marker = marker
        open_existing_read_only_catalog(tmp_path / "absent.sqlite3")
        raise AssertionError(caller_marker)

    try:
        caller()
    except ReadOnlyCatalogError as error:
        assert marker in _all_traceback_locals(error)
        assert marker not in _traceback_locals(error)
    else:
        raise AssertionError("read-only catalog unexpectedly opened")


def _traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if Path(traceback.tb_frame.f_code.co_filename).resolve() == _READ_ONLY_MODULE_FILE:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def _all_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)
