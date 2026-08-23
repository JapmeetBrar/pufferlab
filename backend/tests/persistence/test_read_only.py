from __future__ import annotations

import hashlib
import os
import socket
import sqlite3
import stat
import time
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


def test_existing_catalog_never_resolves_transiently_substituted_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original_dataset = _database_with_dataset(tmp_path / "catalog.sqlite3")
    replacement, replacement_dataset = _database_with_dataset(
        tmp_path / "replacement.sqlite3",
        replacement=True,
    )
    original_before = _snapshot(path)
    replacement_before = _snapshot(replacement)
    displaced = tmp_path / "displaced.sqlite3"
    real_resolve = Path.resolve
    substituted = False
    resolved_inputs: list[Path] = []

    def substituting_resolve(candidate: Path, strict: bool = False) -> Path:
        nonlocal substituted
        if not substituted and candidate in {path, path.parent}:
            resolved_inputs.append(candidate)
            path.replace(displaced)
            path.symlink_to(replacement)
            substituted = True
            try:
                return real_resolve(candidate, strict=strict)
            finally:
                path.unlink()
                displaced.replace(path)
        return real_resolve(candidate, strict=strict)

    monkeypatch.setattr(Path, "resolve", substituting_resolve)
    with open_existing_read_only_catalog(path) as catalog:
        assert catalog.repository.list_dataset_versions(limit=2) == [original_dataset]
        assert catalog.repository.list_dataset_versions(limit=2) != [replacement_dataset]

    assert substituted
    assert resolved_inputs == [path.parent]
    assert _snapshot(path) == original_before
    assert _snapshot(replacement) == replacement_before


def test_existing_catalog_fails_closed_when_leaf_becomes_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _original_dataset = _database_with_dataset(tmp_path / "catalog.sqlite3")
    replacement, _replacement_dataset = _database_with_dataset(
        tmp_path / "replacement.sqlite3",
        replacement=True,
    )
    original_before = _snapshot(path)
    replacement_before = _snapshot(replacement)
    displaced = tmp_path / "displaced.sqlite3"
    real_open = os.open
    substituted = False

    def substituting_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal substituted
        fd = real_open(target, flags, *args, **kwargs)
        if not substituted and Path(target) == path.parent and kwargs.get("dir_fd") is None:
            path.replace(displaced)
            path.symlink_to(replacement)
            substituted = True
        return fd

    monkeypatch.setattr(read_only_module.os, "open", substituting_open)
    try:
        with pytest.raises(ReadOnlyCatalogError):
            open_existing_read_only_catalog(path)
    finally:
        if path.is_symlink():
            path.unlink()
        if displaced.exists():
            displaced.replace(path)

    assert substituted
    assert _snapshot(path) == original_before
    assert _snapshot(replacement) == replacement_before


def test_existing_catalog_fifo_without_writer_fails_promptly_and_stays_intact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hostile-marker.fifo"
    os.mkfifo(path)
    before = path.stat(follow_symlinks=False)

    started = time.monotonic()
    with pytest.raises(ReadOnlyCatalogError) as error:
        open_existing_read_only_catalog(path)
    elapsed = time.monotonic() - started

    after = path.stat(follow_symlinks=False)
    assert elapsed < 1.0
    assert stat.S_ISFIFO(after.st_mode)
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert str(error.value) == "existing diagnostic catalog is unavailable or invalid"
    assert "hostile-marker" not in _traceback_locals(error.value)


def test_existing_catalog_rejects_socket_and_device_without_removing_them(
    tmp_path: Path,
) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"pufferlab-ro-{uuid4().hex}.socket"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix_socket:
            unix_socket.bind(str(socket_path))
            with pytest.raises(ReadOnlyCatalogError):
                open_existing_read_only_catalog(socket_path)
            assert stat.S_ISSOCK(socket_path.stat(follow_symlinks=False).st_mode)
    finally:
        socket_path.unlink(missing_ok=True)

    device_path = Path("/dev/null")
    with pytest.raises(ReadOnlyCatalogError):
        open_existing_read_only_catalog(device_path)
    assert stat.S_ISCHR(device_path.stat(follow_symlinks=False).st_mode)


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


def test_existing_catalog_connect_boundary_cannot_substitute_valid_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original_dataset = _database_with_dataset(tmp_path / "catalog.sqlite3")
    replacement, replacement_dataset = _database_with_dataset(
        tmp_path / "replacement.sqlite3",
        replacement=True,
    )
    original_before = _snapshot(path)
    replacement_before = _snapshot(replacement)
    displaced = tmp_path / "displaced.sqlite3"
    real_connect = sqlite3.connect
    connect_targets: list[str] = []
    substitution_count = 0

    def substituting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal substitution_count
        connect_targets.append(str(args[0]))
        path.replace(displaced)
        replacement.replace(path)
        substitution_count += 1
        try:
            return real_connect(*args, **kwargs)
        finally:
            path.replace(replacement)
            displaced.replace(path)

    monkeypatch.setattr(read_only_module.sqlite3, "connect", substituting_connect)
    with open_existing_read_only_catalog(path) as catalog:
        assert catalog.repository.list_dataset_versions(limit=2) == [original_dataset]
        assert catalog.repository.list_dataset_versions(limit=2) != [replacement_dataset]

    assert substitution_count == 1
    assert len(connect_targets) == 1
    assert "/dev/fd/" in connect_targets[0]
    assert str(path) not in connect_targets[0]
    assert _snapshot(path) == original_before
    assert _snapshot(replacement) == replacement_before


def test_existing_catalog_accepts_only_identity_verified_same_descriptor_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _database(tmp_path / "catalog.sqlite3")
    catalog = open_existing_read_only_catalog(path)
    guard = catalog._guard
    guard_fd = guard._fd
    assert guard_fd is not None
    real_open = read_only_module.os.open
    linux_alias = f"/proc/self/fd/{guard_fd}"
    macos_alias = f"/dev/fd/{guard_fd}"
    opened_aliases: list[str] = []

    def evidenced_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if str(target) == linux_alias:
            opened_aliases.append(linux_alias)
            return real_open(macos_alias, flags, *args, **kwargs)
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(read_only_module.os, "open", evidenced_open)
    guard.validate_descriptor_alias(macos_alias)
    guard.validate_descriptor_alias(linux_alias)
    with pytest.raises(ReadOnlyCatalogError):
        guard.validate_descriptor_alias(str(path))
    with pytest.raises(ReadOnlyCatalogError):
        guard.validate_descriptor_alias(f"/dev/fd/{guard_fd + 1}")
    catalog.close()

    assert opened_aliases == [linux_alias]


def test_existing_catalog_rejects_allowed_descriptor_alias_with_wrong_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _database(tmp_path / "catalog.sqlite3")
    replacement = _database(tmp_path / "replacement.sqlite3")
    catalog = open_existing_read_only_catalog(path)
    guard = catalog._guard
    guard_fd = guard._fd
    assert guard_fd is not None
    alias = f"/dev/fd/{guard_fd}"
    real_open = os.open

    def substituted_alias_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if str(target) == alias:
            return real_open(replacement, flags)
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(read_only_module.os, "open", substituted_alias_open)
    with pytest.raises(ReadOnlyCatalogError):
        guard.validate_descriptor_alias(alias)
    monkeypatch.setattr(read_only_module.os, "open", real_open)
    catalog.close()


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


def test_existing_catalog_fails_closed_without_descriptor_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _database(tmp_path / "descriptor-support-hostile-marker.sqlite3")
    before = _snapshot(path)
    missing_descriptor_root = tmp_path / "missing-descriptor-root"
    monkeypatch.setattr(read_only_module, "_DESCRIPTOR_ROOT", missing_descriptor_root)

    with pytest.raises(ReadOnlyCatalogError) as error:
        open_existing_read_only_catalog(path)

    assert _snapshot(path) == before
    assert not missing_descriptor_root.exists()
    assert "descriptor-support-hostile-marker" not in _traceback_locals(error.value)


@pytest.mark.parametrize(
    "outcome",
    [
        "normal",
        "acquire_cancel",
        "acquire_system_exit",
        "connect_error",
        "connect_cancel",
        "validate_error",
        "validate_cancel",
    ],
)
def test_existing_catalog_closes_exact_guard_fd_on_every_open_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    path = _database(tmp_path / "guard-close-hostile-marker.sqlite3")
    resolved = path.resolve(strict=True)
    real_open = read_only_module.os.open
    real_close = read_only_module.os.close
    real_connect = read_only_module.sqlite3.connect
    parent_fd: int | None = None
    guard_fds: list[int] = []
    guard_close_counts: dict[int, int] = {}
    injected: BaseException | None = None

    def tracked_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal parent_fd
        fd = real_open(target, flags, *args, **kwargs)
        if Path(target) == resolved.parent and kwargs.get("dir_fd") is None:
            parent_fd = fd
            guard_fds.append(fd)
            guard_close_counts[fd] = 0
        elif target == resolved.name and kwargs.get("dir_fd") == parent_fd:
            guard_fds.append(fd)
            guard_close_counts[fd] = 0
        return fd

    def tracked_close(fd: int) -> None:
        if fd in guard_close_counts:
            guard_close_counts[fd] += 1
        real_close(fd)

    monkeypatch.setattr(read_only_module.os, "open", tracked_open)
    monkeypatch.setattr(read_only_module.os, "close", tracked_close)
    if outcome.startswith("acquire"):
        injected = (
            KeyboardInterrupt("acquire-cancel-hostile-marker")
            if outcome == "acquire_cancel"
            else SystemExit("acquire-exit-hostile-marker")
        )

        def failing_clear(_fd: int, *, nonblocking: int) -> None:
            del nonblocking
            assert injected is not None
            raise injected

        monkeypatch.setattr(read_only_module, "_clear_nonblocking", failing_clear)
    elif outcome.startswith("connect"):
        injected = (
            RuntimeError("connect-error-hostile-marker")
            if outcome == "connect_error"
            else KeyboardInterrupt("connect-cancel-hostile-marker")
        )

        def failing_connect(*_args: Any, **_kwargs: Any) -> sqlite3.Connection:
            assert injected is not None
            raise injected

        monkeypatch.setattr(read_only_module.sqlite3, "connect", failing_connect)
    elif outcome.startswith("validate"):
        injected = (
            RuntimeError("validate-error-hostile-marker")
            if outcome == "validate_error"
            else KeyboardInterrupt("validate-cancel-hostile-marker")
        )

        def failing_validation(_engine: object) -> None:
            assert injected is not None
            raise injected

        monkeypatch.setattr(read_only_module, "_validate_catalog", failing_validation)

    if outcome == "normal":
        open_existing_read_only_catalog(path).close()
    elif outcome == "acquire_system_exit":
        with pytest.raises(SystemExit):
            open_existing_read_only_catalog(path)
    elif outcome.endswith("cancel"):
        with pytest.raises(KeyboardInterrupt):
            open_existing_read_only_catalog(path)
    else:
        with pytest.raises(ReadOnlyCatalogError):
            open_existing_read_only_catalog(path)

    assert len(guard_fds) == 2
    assert guard_close_counts == {guard_fds[0]: 1, guard_fds[1]: 1}
    for guard_fd in guard_fds:
        with pytest.raises(OSError):
            read_only_module.os.fstat(guard_fd)
    if injected is not None:
        assert injected.__traceback__ is None
        assert "hostile-marker" not in _traceback_locals(injected)
    monkeypatch.setattr(read_only_module.sqlite3, "connect", real_connect)


def test_existing_catalog_closes_parent_and_fifo_descriptors_after_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.fifo"
    os.mkfifo(path)
    real_open = os.open
    real_close = os.close
    parent_fd: int | None = None
    owned_fds: list[int] = []
    close_counts: dict[int, int] = {}

    def tracked_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal parent_fd
        fd = real_open(target, flags, *args, **kwargs)
        if Path(target) == path.parent and kwargs.get("dir_fd") is None:
            parent_fd = fd
            owned_fds.append(fd)
            close_counts[fd] = 0
        elif target == path.name and kwargs.get("dir_fd") == parent_fd:
            owned_fds.append(fd)
            close_counts[fd] = 0
        return fd

    def tracked_close(fd: int) -> None:
        if fd in close_counts:
            close_counts[fd] += 1
        real_close(fd)

    monkeypatch.setattr(read_only_module.os, "open", tracked_open)
    monkeypatch.setattr(read_only_module.os, "close", tracked_close)
    with pytest.raises(ReadOnlyCatalogError):
        open_existing_read_only_catalog(path)

    assert len(owned_fds) == 2
    assert close_counts == {owned_fds[0]: 1, owned_fds[1]: 1}
    for owned_fd in owned_fds:
        with pytest.raises(OSError):
            os.fstat(owned_fd)
    assert stat.S_ISFIFO(path.stat(follow_symlinks=False).st_mode)


def test_existing_catalog_clears_nonblocking_before_sqlite_use(tmp_path: Path) -> None:
    import fcntl

    path = _database(tmp_path / "catalog.sqlite3")
    with open_existing_read_only_catalog(path) as catalog:
        guard_fd = catalog._guard._fd
        assert guard_fd is not None
        assert fcntl.fcntl(guard_fd, fcntl.F_GETFL) & os.O_NONBLOCK == 0


@pytest.mark.parametrize(
    "outcome",
    [
        "normal",
        "configuration_error",
        "configuration_cancel",
        "validation_error",
        "validation_cancel",
    ],
)
def test_existing_catalog_closes_exact_sqlite_connection_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    path = _database(tmp_path / "connection-close.sqlite3")
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    connections: list[TrackingConnection] = []

    def tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = real_connect(*args, factory=TrackingConnection, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(read_only_module.sqlite3, "connect", tracked_connect)
    if outcome != "normal":
        injected: BaseException = (
            RuntimeError("connection setup failure")
            if outcome.endswith("error")
            else KeyboardInterrupt()
        )

    if outcome.startswith("configuration"):

        def failing_configuration(_connection: sqlite3.Connection) -> None:
            raise injected

        monkeypatch.setattr(read_only_module, "_configure_connection", failing_configuration)
    elif outcome.startswith("validation"):

        def failing_validation(_engine: object) -> None:
            raise injected

        monkeypatch.setattr(read_only_module, "_validate_catalog", failing_validation)

    if outcome == "normal":
        open_existing_read_only_catalog(path).close()
    elif outcome.endswith("cancel"):
        with pytest.raises(KeyboardInterrupt):
            open_existing_read_only_catalog(path)
    else:
        with pytest.raises(ReadOnlyCatalogError):
            open_existing_read_only_catalog(path)

    assert len(connections) == 1
    assert connections[0].close_calls == 1


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
