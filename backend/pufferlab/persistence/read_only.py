"""Existing-file, immutable SQLite composition for diagnostic reads."""

from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import NoReturn, cast

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from pufferlab.persistence.repository import PufferLabRepository

_CURRENT_REVISION = "20260822_0001"
_DESCRIPTOR_ROOT = Path("/dev/fd")
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD = os.stat in os.supports_dir_fd


class ReadOnlyCatalogError(RuntimeError):
    """Value-free local failure opening or validating the diagnostic catalog."""

    def __init__(self) -> None:
        super().__init__("existing diagnostic catalog is unavailable or invalid")


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int


class _OpenControl(StrEnum):
    NONE = "none"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    SYSTEM_EXIT = "system_exit"


@dataclass(frozen=True, slots=True)
class _OpenOutcome:
    catalog: ExistingReadOnlyCatalog | None = None
    control: _OpenControl = _OpenControl.NONE


class _GuardedDatabaseFile:
    """Own exact parent and leaf descriptions selected without following the leaf."""

    __slots__ = (
        "_fd",
        "_leaf_name",
        "_parent_fd",
        "_parent_identity",
        "_parent_path",
        "identity",
    )

    def __init__(
        self,
        *,
        fd: int,
        identity: _FileIdentity,
        parent_fd: int,
        parent_identity: _DirectoryIdentity,
        parent_path: Path,
        leaf_name: str,
    ) -> None:
        self._fd: int | None = fd
        self._parent_fd: int | None = parent_fd
        self._parent_identity = parent_identity
        self._parent_path = parent_path
        self._leaf_name = leaf_name
        self.identity = identity

    @classmethod
    def acquire(cls, path: Path) -> _GuardedDatabaseFile:
        if os.name != "posix":
            raise ReadOnlyCatalogError()
        no_follow = getattr(os, "O_NOFOLLOW", None)
        close_on_exec = getattr(os, "O_CLOEXEC", None)
        nonblocking = getattr(os, "O_NONBLOCK", None)
        directory = getattr(os, "O_DIRECTORY", None)
        if (
            not isinstance(no_follow, int)
            or no_follow == 0
            or not isinstance(close_on_exec, int)
            or close_on_exec == 0
            or not isinstance(nonblocking, int)
            or nonblocking == 0
            or not isinstance(directory, int)
            or directory == 0
            or not _OPEN_SUPPORTS_DIR_FD
            or not _STAT_SUPPORTS_DIR_FD
        ):
            raise ReadOnlyCatalogError()
        leaf_name = path.name
        if not leaf_name or leaf_name in {".", ".."}:
            raise ReadOnlyCatalogError()
        # Parent symlinks are deliberately supported, but the caller's leaf is never resolved.
        # Holding the concrete parent directory open lets the leaf acquisition and every later
        # location check use one directory identity even if an attacker renames a path component.
        parent_path = path.parent.resolve(strict=True)
        parent_fd: int | None = None
        fd: int | None = None
        try:
            parent_fd = os.open(
                parent_path,
                os.O_RDONLY | directory | no_follow | close_on_exec,
            )
            parent_identity = _directory_identity_from_metadata(os.fstat(parent_fd))
            if _directory_identity(parent_path) != parent_identity:
                raise ReadOnlyCatalogError()
            # O_NONBLOCK makes FIFO/device rejection bounded. It is removed only after fstat has
            # established that the exact no-follow leaf description is a regular file.
            fd = os.open(
                leaf_name,
                os.O_RDONLY | nonblocking | no_follow | close_on_exec,
                dir_fd=parent_fd,
            )
            identity = _identity_from_metadata(os.fstat(fd))
            _clear_nonblocking(fd, nonblocking=nonblocking)
            guard = cls(
                fd=fd,
                identity=identity,
                parent_fd=parent_fd,
                parent_identity=parent_identity,
                parent_path=parent_path,
                leaf_name=leaf_name,
            )
            guard.verify()
            return guard
        except BaseException:
            if fd is not None:
                with suppress(BaseException):
                    os.close(fd)
            if parent_fd is not None:
                with suppress(BaseException):
                    os.close(parent_fd)
            raise

    @property
    def path(self) -> Path:
        return self._parent_path / self._leaf_name

    @property
    def descriptor_path(self) -> Path:
        fd = self._fd
        if fd is None or not _DESCRIPTOR_ROOT.is_absolute():
            raise ReadOnlyCatalogError()
        descriptor = _DESCRIPTOR_ROOT / str(fd)
        try:
            close_on_exec = getattr(os, "O_CLOEXEC", None)
            if not isinstance(close_on_exec, int) or close_on_exec == 0:
                raise ReadOnlyCatalogError()
            descriptor_fd = os.open(descriptor, os.O_RDONLY | close_on_exec)
            try:
                descriptor_identity = _identity_from_metadata(os.fstat(descriptor_fd))
            finally:
                os.close(descriptor_fd)
        except Exception as error:
            _detach_exception(error)
            raise ReadOnlyCatalogError() from None
        if descriptor_identity != self.identity:
            raise ReadOnlyCatalogError()
        return descriptor

    def validate_descriptor_alias(self, alias: str) -> None:
        """Accept only SQLite's evidenced names for this exact process-local descriptor."""

        fd = self._fd
        if fd is None or alias not in {f"/dev/fd/{fd}", f"/proc/self/fd/{fd}"}:
            raise ReadOnlyCatalogError()
        try:
            close_on_exec = getattr(os, "O_CLOEXEC", None)
            if not isinstance(close_on_exec, int) or close_on_exec == 0:
                raise ReadOnlyCatalogError()
            descriptor_fd = os.open(alias, os.O_RDONLY | close_on_exec)
            try:
                descriptor_identity = _identity_from_metadata(os.fstat(descriptor_fd))
            finally:
                os.close(descriptor_fd)
        except Exception as error:
            _detach_exception(error)
            raise ReadOnlyCatalogError() from None
        if descriptor_identity != self.identity:
            raise ReadOnlyCatalogError()

    def sqlite_uri(self) -> str:
        return f"{self.descriptor_path.as_uri()}?mode=ro&immutable=1"

    def verify(self) -> None:
        fd = self._fd
        parent_fd = self._parent_fd
        if (
            fd is None
            or parent_fd is None
            or _identity_from_metadata(os.fstat(fd)) != self.identity
            or _directory_identity_from_metadata(os.fstat(parent_fd)) != self._parent_identity
            or _directory_identity(self._parent_path) != self._parent_identity
            or _file_identity_at(parent_fd, self._leaf_name) != self.identity
            or _sidecars_exist_at(parent_fd, self._leaf_name)
        ):
            raise ReadOnlyCatalogError()

    def close(self) -> None:
        fd = self._fd
        parent_fd = self._parent_fd
        self._fd = None
        self._parent_fd = None
        failure: BaseException | None = None
        for owned_fd in (fd, parent_fd):
            if owned_fd is not None:
                try:
                    os.close(owned_fd)
                except BaseException as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure


class _PinnedConnectionCreator:
    """Give SQLAlchemy one already-open handle and refuse every attempted reopen."""

    __slots__ = ("_claimed", "_connection")

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection: sqlite3.Connection | None = connection
        self._claimed = False

    def __call__(self) -> sqlite3.Connection:
        connection = self._connection
        if self._claimed or connection is None:
            raise ReadOnlyCatalogError()
        self._claimed = True
        return connection

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    @property
    def claimed(self) -> bool:
        return self._claimed

    def release_after_engine_dispose(self) -> None:
        self._connection = None


class ExistingReadOnlyCatalog:
    """Own a validated immutable engine and its normal read repositories."""

    __slots__ = (
        "_closed",
        "_creator",
        "_engine",
        "_guard",
        "_identity",
        "_path",
        "repository",
    )

    def __init__(
        self,
        *,
        path: Path,
        identity: _FileIdentity,
        guard: _GuardedDatabaseFile,
        creator: _PinnedConnectionCreator,
        engine: Engine,
        repository: PufferLabRepository,
    ) -> None:
        self._path = path
        self._identity = identity
        self._guard = guard
        self._creator = creator
        self._engine = engine
        self.repository = repository
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self._engine
        creator = self._creator
        guard = self._guard
        path = self._path
        identity = self._identity
        failed, control = _dispose_and_validate(
            engine,
            creator=creator,
            guard=guard,
            path=path,
            identity=identity,
        )

        # A raised exception retains this frame's locals, including ``self``. The catalog is
        # terminal after close, so erase every path/creator-bearing reference before a separate
        # value-free helper raises.
        self._engine = cast(Engine, None)
        self._creator = cast(_PinnedConnectionCreator, None)
        self._guard = cast(_GuardedDatabaseFile, None)
        self._identity = _FileIdentity(0, 0, 0, 0)
        self._path = Path()
        self.repository = cast(PufferLabRepository, None)
        engine = cast(Engine, None)
        creator = cast(_PinnedConnectionCreator, None)
        guard = cast(_GuardedDatabaseFile, None)
        path = Path()
        identity = _FileIdentity(0, 0, 0, 0)
        if control is _OpenControl.KEYBOARD_INTERRUPT:
            _raise_keyboard_interrupt()
        if control is _OpenControl.SYSTEM_EXIT:
            _raise_system_exit()
        if failed:
            _raise_read_only_catalog_error()

    def __enter__(self) -> ExistingReadOnlyCatalog:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def __repr__(self) -> str:
        return (
            "ExistingReadOnlyCatalog(state=open)"
            if not self._closed
            else ("ExistingReadOnlyCatalog(state=closed)")
        )


def open_existing_read_only_catalog(path: Path) -> ExistingReadOnlyCatalog:
    """Open one exact regular database without creation, migration, or recovery."""

    outcome = _open_existing_read_only_catalog_inner(path)
    path = Path()
    if outcome.control is _OpenControl.KEYBOARD_INTERRUPT:
        _raise_keyboard_interrupt()
    if outcome.control is _OpenControl.SYSTEM_EXIT:
        _raise_system_exit()
    if outcome.catalog is None:
        _raise_read_only_catalog_error()
    return outcome.catalog


def _open_existing_read_only_catalog_inner(path: Path) -> _OpenOutcome:
    """Consume all value-bearing failures before the public boundary can raise."""

    engine: Engine | None = None
    creator: _PinnedConnectionCreator | None = None
    guard: _GuardedDatabaseFile | None = None
    resolved: Path | None = None
    identity: _FileIdentity | None = None
    try:
        guard = _GuardedDatabaseFile.acquire(path)
        resolved = guard.path
        identity = guard.identity
        connection = sqlite3.connect(
            guard.sqlite_uri(),
            uri=True,
            check_same_thread=False,
        )
        creator = _PinnedConnectionCreator(connection)
        _configure_connection(connection)

        engine = create_engine(
            "sqlite+pysqlite://",
            creator=creator,
            poolclass=StaticPool,
        )
        session_factory = sessionmaker(
            bind=engine,
            class_=Session,
            expire_on_commit=False,
        )
        _validate_database_binding(engine, guard=guard)
        guard.verify()
        _validate_catalog(engine)
        guard.verify()
        if _sidecars_exist(resolved) or _file_identity(resolved) != identity:
            raise ReadOnlyCatalogError()
        return _OpenOutcome(
            catalog=ExistingReadOnlyCatalog(
                path=resolved,
                identity=identity,
                guard=guard,
                creator=creator,
                engine=engine,
                repository=PufferLabRepository(session_factory),
            )
        )
    except KeyboardInterrupt as error:
        _detach_exception(error)
        _dispose_failed_open(
            engine,
            creator=creator,
            guard=guard,
            path=resolved,
            identity=identity,
        )
        return _OpenOutcome(control=_OpenControl.KEYBOARD_INTERRUPT)
    except SystemExit as error:
        _detach_exception(error)
        _dispose_failed_open(
            engine,
            creator=creator,
            guard=guard,
            path=resolved,
            identity=identity,
        )
        return _OpenOutcome(control=_OpenControl.SYSTEM_EXIT)
    except Exception as error:
        _detach_exception(error)
        control = _dispose_failed_open(
            engine,
            creator=creator,
            guard=guard,
            path=resolved,
            identity=identity,
        )
        return _OpenOutcome(control=control)


def _dispose_and_validate(
    engine: Engine,
    *,
    creator: _PinnedConnectionCreator,
    guard: _GuardedDatabaseFile,
    path: Path,
    identity: _FileIdentity,
) -> tuple[bool, _OpenControl]:
    failed = False
    control = _OpenControl.NONE
    try:
        guard.verify()
    except KeyboardInterrupt as error:
        _detach_exception(error)
        control = _OpenControl.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        control = _OpenControl.SYSTEM_EXIT
    except Exception:
        failed = True
    dispose_failed, dispose_control = _dispose_engine_and_connection(engine, creator=creator)
    failed = failed or dispose_failed
    if control is _OpenControl.NONE:
        control = dispose_control
    try:
        guard.verify()
    except KeyboardInterrupt as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.SYSTEM_EXIT
    except Exception:
        failed = True
    try:
        guard.close()
    except KeyboardInterrupt as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.SYSTEM_EXIT
    except Exception:
        failed = True
    try:
        sidecars_exist = _sidecars_exist(path)
        current_identity = _file_identity(path)
        failed = failed or sidecars_exist or current_identity != identity
    except KeyboardInterrupt as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        if control is _OpenControl.NONE:
            control = _OpenControl.SYSTEM_EXIT
    except Exception:
        failed = True
    return failed, control


def _dispose_failed_open(
    engine: Engine | None,
    *,
    creator: _PinnedConnectionCreator | None,
    guard: _GuardedDatabaseFile | None,
    path: Path | None,
    identity: _FileIdentity | None,
) -> _OpenControl:
    control = _OpenControl.NONE
    if guard is not None:
        try:
            guard.verify()
        except KeyboardInterrupt as error:
            _detach_exception(error)
            control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
            control = _OpenControl.SYSTEM_EXIT
        except Exception:
            pass
    _dispose_failed, dispose_control = _dispose_engine_and_connection(engine, creator=creator)
    if control is _OpenControl.NONE:
        control = dispose_control
    if guard is not None:
        try:
            guard.verify()
        except KeyboardInterrupt as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.SYSTEM_EXIT
        except Exception:
            pass
        try:
            guard.close()
        except KeyboardInterrupt as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.SYSTEM_EXIT
        except Exception:
            pass
    if path is not None and identity is not None:
        try:
            _sidecars_exist(path)
            _file_identity(path)
        except KeyboardInterrupt as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
            if control is _OpenControl.NONE:
                control = _OpenControl.SYSTEM_EXIT
        except Exception:
            pass
    return control


def _dispose_engine_and_connection(
    engine: Engine | None,
    *,
    creator: _PinnedConnectionCreator | None,
) -> tuple[bool, _OpenControl]:
    failed = False
    control = _OpenControl.NONE
    engine_disposed = False
    if engine is not None:
        try:
            engine.dispose()
            engine_disposed = True
        except KeyboardInterrupt as error:
            _detach_exception(error)
            control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
            control = _OpenControl.SYSTEM_EXIT
        except Exception:
            failed = True
    if creator is not None:
        if engine_disposed and creator.claimed:
            creator.release_after_engine_dispose()
        else:
            try:
                creator.close()
            except KeyboardInterrupt as error:
                _detach_exception(error)
                if control is _OpenControl.NONE:
                    control = _OpenControl.KEYBOARD_INTERRUPT
            except SystemExit as error:
                _detach_exception(error)
                if control is _OpenControl.NONE:
                    control = _OpenControl.SYSTEM_EXIT
            except Exception:
                failed = True
    return failed, control


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None


def _raise_read_only_catalog_error() -> NoReturn:
    raise ReadOnlyCatalogError()


def _raise_keyboard_interrupt() -> NoReturn:
    raise KeyboardInterrupt()


def _raise_system_exit() -> NoReturn:
    raise SystemExit()


def _validate_catalog(engine: Engine) -> None:
    with engine.connect() as connection:
        quick_check = connection.exec_driver_sql("PRAGMA quick_check").scalars().all()
        if list(quick_check) != ["ok"]:
            raise ReadOnlyCatalogError()
        if connection.exec_driver_sql("PRAGMA foreign_key_check").first() is not None:
            raise ReadOnlyCatalogError()
        revision = (
            connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalars().all()
        )
        if list(revision) != [_CURRENT_REVISION]:
            raise ReadOnlyCatalogError()


def _validate_database_binding(engine: Engine, *, guard: _GuardedDatabaseFile) -> None:
    with engine.connect() as connection:
        databases = connection.exec_driver_sql("PRAGMA database_list").all()
        if len(databases) != 1 or databases[0][1] != "main":
            raise ReadOnlyCatalogError()
        alias = databases[0][2]
        if not isinstance(alias, str):
            raise ReadOnlyCatalogError()
        guard.validate_descriptor_alias(alias)


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")


def _file_identity(path: Path) -> _FileIdentity:
    return _identity_from_metadata(path.stat(follow_symlinks=False))


def _file_identity_at(parent_fd: int, leaf_name: str) -> _FileIdentity:
    return _identity_from_metadata(os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False))


def _identity_from_metadata(metadata: os.stat_result) -> _FileIdentity:
    if not stat.S_ISREG(metadata.st_mode):
        raise ReadOnlyCatalogError()
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
    )


def _directory_identity(path: Path) -> _DirectoryIdentity:
    return _directory_identity_from_metadata(path.stat(follow_symlinks=False))


def _directory_identity_from_metadata(metadata: os.stat_result) -> _DirectoryIdentity:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReadOnlyCatalogError()
    return _DirectoryIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _clear_nonblocking(fd: int, *, nonblocking: int) -> None:
    try:
        import fcntl

        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~nonblocking)
        if fcntl.fcntl(fd, fcntl.F_GETFL) & nonblocking:
            raise ReadOnlyCatalogError()
    except Exception as error:
        _detach_exception(error)
        raise ReadOnlyCatalogError() from None


def _sidecars_exist(path: Path) -> bool:
    return any(os.path.lexists(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm"))


def _sidecars_exist_at(parent_fd: int, leaf_name: str) -> bool:
    for suffix in ("-journal", "-wal", "-shm"):
        try:
            os.stat(f"{leaf_name}{suffix}", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        return True
    return False
