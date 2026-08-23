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
from sqlalchemy.pool import NullPool

from pufferlab.persistence.repository import PufferLabRepository

_CURRENT_REVISION = "20260822_0001"


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


class _OpenControl(StrEnum):
    NONE = "none"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    SYSTEM_EXIT = "system_exit"


@dataclass(frozen=True, slots=True)
class _OpenOutcome:
    catalog: ExistingReadOnlyCatalog | None = None
    control: _OpenControl = _OpenControl.NONE


class ExistingReadOnlyCatalog:
    """Own a validated immutable engine and its normal read repositories."""

    __slots__ = ("_closed", "_engine", "_identity", "_path", "repository")

    def __init__(
        self,
        *,
        path: Path,
        identity: _FileIdentity,
        engine: Engine,
        repository: PufferLabRepository,
    ) -> None:
        self._path = path
        self._identity = identity
        self._engine = engine
        self.repository = repository
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self._engine
        path = self._path
        identity = self._identity
        failed, control = _dispose_and_validate(engine, path=path, identity=identity)

        # A raised exception retains this frame's locals, including ``self``. The catalog is
        # terminal after close, so erase every path/creator-bearing reference before a separate
        # value-free helper raises.
        self._engine = cast(Engine, None)
        self._identity = _FileIdentity(0, 0, 0, 0)
        self._path = Path()
        self.repository = cast(PufferLabRepository, None)
        engine = cast(Engine, None)
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
    resolved: Path | None = None
    identity: _FileIdentity | None = None
    try:
        if path.is_symlink() or _sidecars_exist(path):
            raise ReadOnlyCatalogError()
        # Parent symlinks are resolved once to a concrete absolute file. The exact file is
        # rejected when symbolic, and device/inode/size/mtime plus sidecars are rechecked after
        # validation and disposal to detect replacement around the SQLite open.
        resolved = path.resolve(strict=True)
        identity = _file_identity(resolved)
        uri = f"{resolved.as_uri()}?mode=ro&immutable=1"

        def connect() -> sqlite3.Connection:
            connection = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
            )
            try:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA temp_store=MEMORY")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
            except BaseException:
                with suppress(BaseException):
                    connection.close()
                raise
            return connection

        engine = create_engine(
            "sqlite+pysqlite://",
            creator=connect,
            poolclass=NullPool,
        )
        session_factory = sessionmaker(
            bind=engine,
            class_=Session,
            expire_on_commit=False,
        )
        _validate_catalog(engine)
        if _sidecars_exist(resolved) or _file_identity(resolved) != identity:
            raise ReadOnlyCatalogError()
        return _OpenOutcome(
            catalog=ExistingReadOnlyCatalog(
                path=resolved,
                identity=identity,
                engine=engine,
                repository=PufferLabRepository(session_factory),
            )
        )
    except KeyboardInterrupt as error:
        _detach_exception(error)
        _dispose_failed_open(engine, path=resolved, identity=identity)
        return _OpenOutcome(control=_OpenControl.KEYBOARD_INTERRUPT)
    except SystemExit as error:
        _detach_exception(error)
        _dispose_failed_open(engine, path=resolved, identity=identity)
        return _OpenOutcome(control=_OpenControl.SYSTEM_EXIT)
    except Exception as error:
        _detach_exception(error)
        control = _dispose_failed_open(engine, path=resolved, identity=identity)
        return _OpenOutcome(control=control)


def _dispose_and_validate(
    engine: Engine,
    *,
    path: Path,
    identity: _FileIdentity,
) -> tuple[bool, _OpenControl]:
    failed = False
    control = _OpenControl.NONE
    try:
        engine.dispose()
    except KeyboardInterrupt as error:
        _detach_exception(error)
        control = _OpenControl.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
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
    path: Path | None,
    identity: _FileIdentity | None,
) -> _OpenControl:
    control = _OpenControl.NONE
    if engine is not None:
        try:
            engine.dispose()
        except KeyboardInterrupt as error:
            _detach_exception(error)
            control = _OpenControl.KEYBOARD_INTERRUPT
        except SystemExit as error:
            _detach_exception(error)
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


def _file_identity(path: Path) -> _FileIdentity:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ReadOnlyCatalogError()
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
    )


def _sidecars_exist(path: Path) -> bool:
    return any(os.path.lexists(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm"))
