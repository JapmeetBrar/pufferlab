"""Authenticated, installation-local ownership for one generated tiny namespace.

The production locator and child names in this module are deliberately fixed.  Public callers can
inspect or operate the capability, but cannot supply a state path, namespace, nonce, or ownership
token.  Tests isolate the filesystem by monkeypatching the private ``_production_state_path``
helper.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import hmac
import json
import os
import pwd
import secrets
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, cast

from pufferlab.config import Settings
from pufferlab.contracts.capabilities import CapabilityRequirementCode
from pufferlab.providers.metadata_probe import is_valid_metadata_probe_region

_STATE_COMPONENTS = (".pufferlab", "state", "owned-tiny-v1")
_OWNER_KEY_NAME = "owner.key"
_RECEIPT_NAME = "receipt.json"
_LOCK_NAME = "operation.lock"
_PURPOSE = "pufferlab-owned-tiny"
_FORMAT_VERSION = 1
_OWNER_KEY_BYTES = 32
_NONCE_BYTES = 32
_MAX_RECEIPT_BYTES = 4096
_NAMESPACE_PREFIX = "pufferlab-tiny-"
_LINUX_RENAME_NOREPLACE = 1
_LINUX_RENAME_EXCHANGE = 2
_DARWIN_RENAME_SWAP = 0x00000002
_DARWIN_RENAME_EXCL = 0x00000004
_RECEIPT_FIELDS = frozenset(
    {
        "format_version",
        "purpose",
        "creating_region",
        "nonce",
        "namespace",
        "state",
        "credential_tag",
        "authentication_tag",
    }
)


class OwnedTinyState(StrEnum):
    INTENT = "intent"
    CREATED = "created"
    READY = "ready"
    CLEANUP_REQUESTED = "cleanup_requested"
    NOT_FOUND_VERIFIED = "not_found_verified"


class OwnedTinyStateError(RuntimeError):
    """A fixed, value-free local ownership failure."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class OwnedTinyBusyError(OwnedTinyStateError):
    def __init__(self) -> None:
        super().__init__("another owned tiny operation is already running", exit_code=2)


class OwnedTinyReceiptMissingError(OwnedTinyStateError):
    def __init__(self) -> None:
        super().__init__("no owned tiny receipt is available", exit_code=2)


class OwnedTinyCredentialMismatchError(OwnedTinyStateError):
    def __init__(self) -> None:
        super().__init__(
            "the configured API key does not match the owned tiny receipt",
            exit_code=2,
        )


class OwnedTinyCleanupRequiredError(OwnedTinyStateError):
    def __init__(self) -> None:
        super().__init__(
            "owned tiny cleanup is already in progress; run `pufferlab namespace cleanup-tiny`",
            exit_code=2,
        )


class OwnedTinyTerminalReceiptError(OwnedTinyStateError):
    def __init__(self) -> None:
        super().__init__(
            "owned tiny cleanup is complete; run cleanup-tiny once more before a new ingestion",
            exit_code=2,
        )


class _FailureKind(StrEnum):
    INVALID = "invalid"
    BUSY = "busy"
    MISSING = "missing"
    CREDENTIAL = "credential"
    CLEANUP = "cleanup"
    TERMINAL = "terminal"


class _StateFailure(Exception):
    def __init__(self, kind: _FailureKind = _FailureKind.INVALID) -> None:
        super().__init__(kind.value)
        self.kind = kind


@dataclass(frozen=True, slots=True, repr=False)
class OwnedTinyReceipt:
    format_version: int
    purpose: str
    creating_region: str
    nonce: str
    namespace: str
    state: OwnedTinyState
    credential_tag: str
    authentication_tag: str


@dataclass(frozen=True, slots=True, repr=False)
class OwnedTinyTarget:
    namespace: str
    region: str


@dataclass(frozen=True, slots=True, repr=False)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, repr=False)
class _ReceiptSnapshot:
    receipt: OwnedTinyReceipt
    raw: bytes
    identity: _FileIdentity


@dataclass(frozen=True, slots=True, repr=False)
class _RawFileSnapshot:
    raw: bytes
    identity: _FileIdentity


@dataclass(slots=True, repr=False)
class _StateDirectory:
    fd: int
    identity: tuple[int, int]
    chain_identities: tuple[tuple[int, int], ...]


@dataclass(slots=True, repr=False)
class _CoordinationAnchor:
    fd: int
    identity: tuple[int, int]


def _production_account_home() -> Path:
    if (
        os.name != "posix"
        or os.getuid() != os.geteuid()
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(fcntl, "LOCK_EX")
    ):
        raise _StateFailure()
    failed = False
    account_home = ""
    try:
        account_home = pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError):
        failed = True
    if failed:
        raise _StateFailure()
    home = Path(account_home)
    if not home.is_absolute() or any(part in {"", ".", ".."} for part in home.parts[1:]):
        raise _StateFailure()
    return home


def _production_state_path() -> Path:
    """Resolve the frozen state location from the POSIX account database, never the environment."""

    home = _production_account_home()
    return home.joinpath(*_STATE_COMPONENTS)


def _production_anchor_path() -> Path:
    """Resolve the stable coordination anchor from the same OS account identity."""

    return _production_account_home()


def owned_tiny_requirements(settings: Settings) -> tuple[CapabilityRequirementCode, ...]:
    """Return only frozen, value-free requirements without creating or mutating local state."""

    inspection = _inspect_receipt_read_only()
    if inspection is _Inspection.INVALID:
        return (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)
    if inspection is _Inspection.ABSENT:
        return ()
    if not isinstance(inspection, _ReadOnlySnapshot):
        return (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)
    receipt = inspection.snapshot.receipt
    configured_namespace = settings.pufferlab_search_namespace
    if configured_namespace != receipt.namespace:
        return ()
    if receipt.state is not OwnedTinyState.READY:
        return (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)

    requirements: list[CapabilityRequirementCode] = []
    secret = settings.turbopuffer_api_key
    credential_matches = False
    api_key = ""
    if secret is not None:
        failed = False
        try:
            api_key = secret.get_secret_value()
            credential_matches = _credential_matches(receipt, api_key, inspection.owner_key)
        except Exception:
            failed = True
        api_key = ""
        if failed:
            credential_matches = False
    if not credential_matches:
        requirements.append(CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH)
    if settings.turbopuffer_region != receipt.creating_region:
        requirements.append(CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH)
    return tuple(requirements)


def resolve_owned_tiny_target(settings: Settings) -> OwnedTinyTarget | None:
    """Resolve a ready exact target only when local namespace, key, and region all match."""

    inspection = _inspect_receipt_read_only()
    if not isinstance(inspection, _ReadOnlySnapshot):
        return None
    receipt = inspection.snapshot.receipt
    if (
        receipt.state is not OwnedTinyState.READY
        or settings.pufferlab_search_namespace != receipt.namespace
        or settings.turbopuffer_region != receipt.creating_region
    ):
        return None
    secret = settings.turbopuffer_api_key
    if secret is None:
        return None
    api_key = ""
    matches = False
    try:
        api_key = secret.get_secret_value()
        matches = _credential_matches(receipt, api_key, inspection.owner_key)
    except Exception:
        matches = False
    api_key = ""
    if not matches:
        return None
    return OwnedTinyTarget(namespace=receipt.namespace, region=receipt.creating_region)


@contextmanager
def owned_tiny_ingest_operation() -> Iterator[_OwnedTinyOperation]:
    """Create if needed, then lock the fixed capability for generated ingestion."""

    with _owned_tiny_operation(create=True) as operation:
        yield operation


@contextmanager
def owned_tiny_existing_operation() -> Iterator[_OwnedTinyOperation]:
    """Lock an existing fixed capability without creating any state."""

    with _owned_tiny_operation(create=False) as operation:
        yield operation


@contextmanager
def _owned_tiny_operation(*, create: bool) -> Iterator[_OwnedTinyOperation]:
    """Acquire the one fixed process lock for an ingest, show, or cleanup operation."""

    operation: _OwnedTinyOperation | None = None
    kind: _FailureKind | None = None
    try:
        operation = _begin_operation(
            _production_state_path(),
            anchor_path=_production_anchor_path(),
            create=create,
        )
    except _StateFailure as error:
        kind = error.kind
    if operation is None:
        _raise_public_failure(kind or _FailureKind.INVALID)
    body_failed = False
    try:
        yield operation
    except BaseException:
        body_failed = True
        raise
    finally:
        close_failed = not operation.close()
        if close_failed and not body_failed:
            raise OwnedTinyStateError("owned tiny state did not close cleanly") from None


class _Inspection(StrEnum):
    ABSENT = "absent"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True, repr=False)
class _ReadOnlySnapshot:
    snapshot: _ReceiptSnapshot
    owner_key: bytes

    @property
    def receipt(self) -> OwnedTinyReceipt:
        return self.snapshot.receipt


def _inspect_receipt_read_only() -> _Inspection | _ReadOnlySnapshot:
    directory: _StateDirectory | None = None
    try:
        directory = _open_state_directory(_production_state_path(), create=False)
        if directory is None:
            return _Inspection.ABSENT
        lock_present = _validate_existing_lock(directory.fd)
        owner = _read_owner_key(directory.fd, required=False)
        snapshot = _read_receipt(directory.fd, owner.key if owner is not None else None)
        if snapshot is None:
            return _Inspection.ABSENT
        if owner is None or not lock_present:
            return _Inspection.INVALID
        reloaded = _read_owner_key(directory.fd, required=True)
        if reloaded is None or reloaded.identity != owner.identity:
            return _Inspection.INVALID
        return _ReadOnlySnapshot(snapshot=snapshot, owner_key=owner.key)
    except Exception:
        return _Inspection.INVALID
    finally:
        if directory is not None:
            _close_quietly(directory.fd)


@dataclass(frozen=True, slots=True, repr=False)
class _OwnerKey:
    key: bytes
    identity: _FileIdentity


class _OwnedTinyOperation:
    def __init__(
        self,
        anchor: _CoordinationAnchor,
        directory: _StateDirectory,
        *,
        anchor_path: Path,
        state_path: Path,
        lock_fd: int,
        lock_identity: _FileIdentity,
        owner: _OwnerKey,
    ) -> None:
        self._anchor = anchor
        self._directory = directory
        self._anchor_path = anchor_path
        self._state_path = state_path
        self._lock_fd = lock_fd
        self._lock_identity = lock_identity
        self._owner = owner
        self._closed = False

    def load(self, *, required: bool = True) -> _ReceiptSnapshot | None:
        invalid = False
        snapshot: _ReceiptSnapshot | None = None
        try:
            self._verify_continuity()
            _verify_owner_identity(self._directory.fd, self._owner)
            snapshot = _read_receipt(self._directory.fd, self._owner.key)
        except _StateFailure:
            invalid = True
        if invalid:
            raise OwnedTinyStateError("owned tiny receipt is invalid", exit_code=2) from None
        if required and snapshot is None:
            raise OwnedTinyReceiptMissingError() from None
        return snapshot

    def create_intent(self, *, api_key: str, region: str) -> _ReceiptSnapshot:
        configuration_invalid = not api_key or not is_valid_metadata_probe_region(region)
        if configuration_invalid:
            api_key = ""
            raise OwnedTinyStateError("owned tiny configuration is invalid", exit_code=2)
        existing = self.load(required=False)
        if existing is not None:
            return existing
        nonce = secrets.token_bytes(_NONCE_BYTES).hex()
        credential_failed = False
        credential_tag = ""
        try:
            credential_tag = _credential_tag(self._owner.key, api_key)
        except UnicodeEncodeError:
            credential_failed = True
        api_key = ""
        if credential_failed:
            raise OwnedTinyStateError("owned tiny configuration is invalid", exit_code=2)
        namespace = _derive_namespace(
            self._owner.key,
            nonce=nonce,
            region=region,
            credential_tag=credential_tag,
        )
        unsigned = OwnedTinyReceipt(
            format_version=_FORMAT_VERSION,
            purpose=_PURPOSE,
            creating_region=region,
            nonce=nonce,
            namespace=namespace,
            state=OwnedTinyState.INTENT,
            credential_tag=credential_tag,
            authentication_tag="",
        )
        receipt = replace(unsigned, authentication_tag=_receipt_tag(self._owner.key, unsigned))
        raw = _encode_receipt(receipt)
        persistence_failed = False
        snapshot = None
        persisted_identity: _FileIdentity | None = None
        try:
            self._verify_continuity()
            _verify_owner_identity(self._directory.fd, self._owner)
            persisted_identity = _atomic_create_receipt(
                self._directory.fd,
                raw,
                continuity=self._verify_continuity,
            )
            snapshot = _read_receipt(self._directory.fd, self._owner.key)
        except _StateFailure:
            persistence_failed = True
        if (
            persistence_failed
            or snapshot is None
            or persisted_identity is None
            or not _same_file_object(snapshot.identity, persisted_identity)
            or not hmac.compare_digest(snapshot.raw, raw)
            or snapshot.receipt != receipt
        ):
            raise OwnedTinyStateError("owned tiny intent could not be persisted")
        return snapshot

    def require_credential(self, snapshot: _ReceiptSnapshot, api_key: str) -> None:
        matches = _credential_matches(snapshot.receipt, api_key, self._owner.key)
        api_key = ""
        if not matches:
            raise OwnedTinyCredentialMismatchError() from None

    def authenticate_current(self, snapshot: _ReceiptSnapshot) -> None:
        """Re-authenticate the exact owner key and receipt identity at a live boundary."""

        invalid = False
        try:
            self._verify_continuity()
            _verify_owner_identity(self._directory.fd, self._owner)
            _verify_prior_receipt(
                self._directory.fd,
                prior=snapshot,
                owner_key=self._owner.key,
            )
        except _StateFailure:
            invalid = True
        if invalid:
            raise OwnedTinyStateError("owned tiny receipt changed during the operation") from None

    def transition(
        self,
        snapshot: _ReceiptSnapshot,
        state: OwnedTinyState,
    ) -> _ReceiptSnapshot:
        if state is snapshot.receipt.state:
            return snapshot
        allowed = {
            OwnedTinyState.INTENT: {OwnedTinyState.CREATED, OwnedTinyState.CLEANUP_REQUESTED},
            OwnedTinyState.CREATED: {OwnedTinyState.READY, OwnedTinyState.CLEANUP_REQUESTED},
            OwnedTinyState.READY: {OwnedTinyState.CLEANUP_REQUESTED},
            OwnedTinyState.CLEANUP_REQUESTED: {OwnedTinyState.NOT_FOUND_VERIFIED},
            OwnedTinyState.NOT_FOUND_VERIFIED: set(),
        }
        if state not in allowed[snapshot.receipt.state]:
            raise OwnedTinyStateError("owned tiny lifecycle transition was rejected")
        unsigned = replace(snapshot.receipt, state=state, authentication_tag="")
        receipt = replace(unsigned, authentication_tag=_receipt_tag(self._owner.key, unsigned))
        replacement_raw = _encode_receipt(receipt)
        persistence_failed = False
        updated = None
        persisted_identity: _FileIdentity | None = None
        try:
            self._verify_continuity()
            _verify_owner_identity(self._directory.fd, self._owner)
            persisted_identity = _atomic_replace_receipt(
                self._directory.fd,
                prior=snapshot,
                replacement=replacement_raw,
                owner=self._owner,
                continuity=self._verify_continuity,
            )
            updated = _read_receipt(self._directory.fd, self._owner.key)
        except _StateFailure:
            persistence_failed = True
        if (
            persistence_failed
            or updated is None
            or persisted_identity is None
            or not _same_file_object(updated.identity, persisted_identity)
            or not hmac.compare_digest(updated.raw, replacement_raw)
            or updated.receipt != receipt
        ):
            raise OwnedTinyStateError("owned tiny lifecycle transition could not be persisted")
        return updated

    def remove_terminal(self, snapshot: _ReceiptSnapshot) -> None:
        if snapshot.receipt.state is not OwnedTinyState.NOT_FOUND_VERIFIED:
            raise OwnedTinyStateError("owned tiny terminal receipt removal was rejected")
        removal_failed = False
        try:
            self._verify_continuity()
            _verify_owner_identity(self._directory.fd, self._owner)
            _remove_receipt_cas(
                self._directory.fd,
                prior=snapshot,
                owner=self._owner,
                continuity=self._verify_continuity,
            )
        except _StateFailure:
            removal_failed = True
        if removal_failed:
            raise OwnedTinyStateError("owned tiny terminal receipt could not be removed") from None

    def _verify_continuity(self) -> None:
        _verify_operation_continuity(
            anchor=self._anchor,
            anchor_path=self._anchor_path,
            directory=self._directory,
            state_path=self._state_path,
            lock_fd=self._lock_fd,
            lock_identity=self._lock_identity,
        )

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        lock_ok = _close_quietly(self._lock_fd)
        directory_ok = _close_quietly(self._directory.fd)
        anchor_ok = _close_quietly(self._anchor.fd)
        return lock_ok and directory_ok and anchor_ok


def _begin_operation(
    path: Path,
    *,
    anchor_path: Path,
    create: bool,
) -> _OwnedTinyOperation:
    anchor = _open_coordination_anchor(anchor_path)
    try:
        fcntl.flock(anchor.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        busy = error.errno in {errno.EACCES, errno.EAGAIN}
        _close_quietly(anchor.fd)
        raise _StateFailure(_FailureKind.BUSY if busy else _FailureKind.INVALID) from None
    try:
        directory = _open_state_directory(path, create=create)
    except BaseException:
        # The anchor flock is still owned locally until the operation object exists.
        # Release it on every state-tree open failure, including process-control exits.
        _close_quietly(anchor.fd)
        raise
    if directory is None:
        _close_quietly(anchor.fd)
        raise _StateFailure(_FailureKind.MISSING)
    lock_fd = -1
    try:
        lock_fd = _open_fixed_file(directory.fd, _LOCK_NAME, create=create, writable=True)
        try:
            os.fsync(directory.fd)
        except OSError:
            raise _StateFailure() from None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            busy = error.errno in {errno.EACCES, errno.EAGAIN}
            raise _StateFailure(_FailureKind.BUSY if busy else _FailureKind.INVALID) from None
        lock_identity = _identity(_validate_regular_file(lock_fd))
        owner = _read_owner_key(directory.fd, required=False)
        if owner is None:
            if not create:
                raise _StateFailure(_FailureKind.MISSING)
            owner = _create_owner_key(directory.fd)
        operation = _OwnedTinyOperation(
            anchor,
            directory,
            anchor_path=anchor_path,
            state_path=path,
            lock_fd=lock_fd,
            lock_identity=lock_identity,
            owner=owner,
        )
        operation._verify_continuity()
        return operation
    except _StateFailure:
        if lock_fd >= 0:
            _close_quietly(lock_fd)
        _close_quietly(directory.fd)
        _close_quietly(anchor.fd)
        raise


def _open_state_directory(path: Path, *, create: bool) -> _StateDirectory | None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise _StateFailure()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    current = -1
    chain_identities: list[tuple[int, int]] = []
    try:
        current = os.open("/", flags)
        chain_identities.append(_directory_identity(os.fstat(current)))
        parts = path.parts[1:]
        for index, component in enumerate(parts):
            child = -1
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    closing = current
                    current = -1
                    if not _close_quietly(closing):
                        raise _StateFailure() from None
                    return None
                child = _install_private_directory(current, component, flags=flags)
            except OSError:
                raise _StateFailure() from None
            if child < 0:
                raise _StateFailure()
            parent = current
            current = -1
            if not _close_quietly(parent):
                _close_quietly(child)
                raise _StateFailure()
            current = child
            try:
                info = os.fstat(current)
            except OSError:
                raise _StateFailure() from None
            if not stat.S_ISDIR(info.st_mode):
                raise _StateFailure()
            chain_identities.append(_directory_identity(info))
            if index == len(parts) - 1 and (
                info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise _StateFailure()
        _clear_nonblocking(current)
        return _StateDirectory(
            fd=current,
            identity=_directory_identity(info),
            chain_identities=tuple(chain_identities),
        )
    except _StateFailure:
        if current >= 0:
            _close_quietly(current)
        raise


def _install_private_directory(parent_fd: int, component: str, *, flags: int) -> int:
    # POSIX mkdir does not return a descriptor. Stage under a 128-bit random internal name,
    # validate that private inode, then publish with native no-replace rename. Fixed-path
    # occupants are never chmodded, replaced, or populated after a stale lookup.
    staging = _directory_temporary_name()
    staged_fd = -1
    installed_fd = -1
    staged_identity: tuple[int, int] | None = None
    try:
        os.mkdir(staging, mode=0o700, dir_fd=parent_fd)
        staged_fd = os.open(staging, flags, dir_fd=parent_fd)
        staged = os.fstat(staged_fd)
        if (
            not stat.S_ISDIR(staged.st_mode)
            or staged.st_uid != os.geteuid()
            or stat.S_IMODE(staged.st_mode) != 0o700
        ):
            raise _StateFailure()
        staged_identity = _directory_identity(staged)
        _rename_noreplace(parent_fd, staging, component)
        installed_fd = os.open(component, flags, dir_fd=parent_fd)
        installed = os.fstat(installed_fd)
        if (
            not stat.S_ISDIR(installed.st_mode)
            or installed.st_uid != os.geteuid()
            or stat.S_IMODE(installed.st_mode) != 0o700
            or _directory_identity(installed) != staged_identity
        ):
            raise _StateFailure()
        os.fsync(parent_fd)
        staged_closed = _close_quietly(staged_fd)
        staged_fd = -1
        if not staged_closed:
            raise _StateFailure()
        return installed_fd
    except (OSError, _StateFailure):
        if installed_fd >= 0:
            _close_quietly(installed_fd)
        if staged_fd >= 0:
            _close_quietly(staged_fd)
        # Never touch the published fixed component here. A random staging directory left by
        # ambiguity is inert and is never scanned as authority.
        del staged_identity
        raise _StateFailure() from None


def _open_coordination_anchor(path: Path) -> _CoordinationAnchor:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise _StateFailure()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    current = -1
    try:
        current = os.open("/", flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=current)
            parent = current
            current = -1
            if not _close_quietly(parent):
                _close_quietly(child)
                raise _StateFailure()
            current = child
        info = os.fstat(current)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise _StateFailure()
        _clear_nonblocking(current)
        return _CoordinationAnchor(fd=current, identity=_directory_identity(info))
    except (OSError, _StateFailure):
        if current >= 0:
            _close_quietly(current)
        raise _StateFailure() from None


def _verify_operation_continuity(
    *,
    anchor: _CoordinationAnchor,
    anchor_path: Path,
    directory: _StateDirectory,
    state_path: Path,
    lock_fd: int,
    lock_identity: _FileIdentity,
) -> None:
    try:
        held_anchor = os.fstat(anchor.fd)
        held_directory = os.fstat(directory.fd)
        held_lock = _validate_regular_file(lock_fd)
    except (OSError, _StateFailure):
        raise _StateFailure() from None
    if (
        not stat.S_ISDIR(held_anchor.st_mode)
        or held_anchor.st_uid != os.geteuid()
        or _directory_identity(held_anchor) != anchor.identity
        or not stat.S_ISDIR(held_directory.st_mode)
        or held_directory.st_uid != os.geteuid()
        or stat.S_IMODE(held_directory.st_mode) != 0o700
        or _directory_identity(held_directory) != directory.identity
        or _identity(held_lock) != lock_identity
    ):
        raise _StateFailure()

    current_anchor: _CoordinationAnchor | None = None
    current_directory: _StateDirectory | None = None
    current_lock = -1
    try:
        current_anchor = _open_coordination_anchor(anchor_path)
        current_directory = _open_state_directory(state_path, create=False)
        if current_directory is None:
            raise _StateFailure()
        current_lock = _open_fixed_file(
            current_directory.fd,
            _LOCK_NAME,
            create=False,
            writable=True,
        )
        if (
            current_anchor.identity != anchor.identity
            or current_directory.identity != directory.identity
            or current_directory.chain_identities != directory.chain_identities
            or _identity(_validate_regular_file(current_lock)) != lock_identity
        ):
            raise _StateFailure()
    finally:
        if current_lock >= 0:
            _close_quietly(current_lock)
        if current_directory is not None:
            _close_quietly(current_directory.fd)
        if current_anchor is not None:
            _close_quietly(current_anchor.fd)


def _open_fixed_file(directory_fd: int, name: str, *, create: bool, writable: bool) -> int:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
    created = False
    fd = -1
    try:
        if create:
            try:
                fd = os.open(
                    name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                created = True
            except FileExistsError:
                fd = os.open(name, flags, dir_fd=directory_fd)
        else:
            fd = os.open(name, flags, dir_fd=directory_fd)
        if created:
            os.fchmod(fd, 0o600)
        _validate_regular_file(fd)
        _clear_nonblocking(fd)
        return fd
    except OSError:
        if fd >= 0:
            _close_quietly(fd)
        raise _StateFailure() from None
    except _StateFailure:
        if fd >= 0:
            _close_quietly(fd)
        raise


def _validate_regular_file(fd: int) -> os.stat_result:
    try:
        info = os.fstat(fd)
    except OSError:
        raise _StateFailure() from None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
    ):
        raise _StateFailure()
    return info


def _validate_existing_lock(directory_fd: int) -> bool:
    try:
        fd = _open_fixed_file(directory_fd, _LOCK_NAME, create=False, writable=False)
    except _StateFailure:
        return False
    return _close_quietly(fd)


def _create_owner_key(directory_fd: int) -> _OwnerKey:
    key = secrets.token_bytes(_OWNER_KEY_BYTES)
    temporary = _owner_temporary_name()
    identity = _prepare_temporary(directory_fd, temporary, key)
    installed = False
    try:
        _rename_noreplace(directory_fd, temporary, _OWNER_KEY_NAME)
        installed = True
        if not _named_regular_file_matches(
            directory_fd,
            _OWNER_KEY_NAME,
            identity=identity,
            raw=key,
        ):
            raise _StateFailure()
        os.fsync(directory_fd)
    except (OSError, _StateFailure):
        # A no-replace collision leaves the fixed key untouched. Once published, never remove or
        # overwrite that fixed pathname after a userspace identity check; a post-publication
        # substitute and the staged inode are preserved for fail-closed recovery.
        if not installed:
            with suppress(_StateFailure):
                _remove_known_regular_file(
                    directory_fd,
                    temporary,
                    identity=identity,
                    raw=key,
                )
        raise _StateFailure() from None
    owner = _read_owner_key(directory_fd, required=True)
    if (
        owner is None
        or not _same_file_object(owner.identity, identity)
        or not hmac.compare_digest(owner.key, key)
    ):
        raise _StateFailure()
    return owner


def _read_owner_key(directory_fd: int, *, required: bool) -> _OwnerKey | None:
    try:
        fd = os.open(
            _OWNER_KEY_NAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        if required:
            raise _StateFailure() from None
        return None
    except OSError:
        raise _StateFailure() from None
    try:
        before = _validate_regular_file(fd)
        _clear_nonblocking(fd)
        raw = _read_bounded(fd, _OWNER_KEY_BYTES + 1)
        after = _validate_regular_file(fd)
    finally:
        closed = _close_quietly(fd)
    if not closed or _identity(before) != _identity(after) or len(raw) != _OWNER_KEY_BYTES:
        raise _StateFailure()
    return _OwnerKey(key=raw, identity=_identity(after))


def _verify_owner_identity(directory_fd: int, owner: _OwnerKey) -> None:
    current = _read_owner_key(directory_fd, required=True)
    if (
        current is None
        or current.identity != owner.identity
        or not hmac.compare_digest(current.key, owner.key)
    ):
        raise _StateFailure()


def _read_receipt(directory_fd: int, owner_key: bytes | None) -> _ReceiptSnapshot | None:
    return _read_named_receipt(directory_fd, _RECEIPT_NAME, owner_key)


def _read_named_receipt(
    directory_fd: int,
    name: str,
    owner_key: bytes | None,
) -> _ReceiptSnapshot | None:
    snapshot = _read_named_regular_file(directory_fd, name, maximum=_MAX_RECEIPT_BYTES)
    if snapshot is None:
        return None
    if owner_key is None or not snapshot.raw:
        raise _StateFailure()
    receipt = _decode_receipt(snapshot.raw, owner_key)
    return _ReceiptSnapshot(
        receipt=receipt,
        raw=snapshot.raw,
        identity=snapshot.identity,
    )


def _read_named_regular_file(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
) -> _RawFileSnapshot | None:
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return None
    except OSError:
        raise _StateFailure() from None
    try:
        before = _validate_regular_file(fd)
        _clear_nonblocking(fd)
        raw = _read_bounded(fd, maximum + 1)
        after = _validate_regular_file(fd)
    finally:
        closed = _close_quietly(fd)
    if not closed or len(raw) > maximum or _identity(before) != _identity(after):
        raise _StateFailure()
    return _RawFileSnapshot(raw=raw, identity=_identity(after))


def _atomic_create_receipt(
    directory_fd: int,
    raw: bytes,
    *,
    continuity: Callable[[], None] | None = None,
) -> _FileIdentity:
    temporary = _temporary_name()
    identity = _prepare_temporary(directory_fd, temporary, raw)
    installed = False
    try:
        if continuity is not None:
            continuity()
        _rename_noreplace(directory_fd, temporary, _RECEIPT_NAME)
        installed = True
        if not _named_regular_file_matches(
            directory_fd,
            _RECEIPT_NAME,
            identity=identity,
            raw=raw,
        ):
            raise _StateFailure()
        os.fsync(directory_fd)
        return identity
    except (OSError, _StateFailure):
        if not installed:
            with suppress(_StateFailure):
                _remove_known_regular_file(
                    directory_fd,
                    temporary,
                    identity=identity,
                    raw=raw,
                )
        raise _StateFailure() from None


def _atomic_replace_receipt(
    directory_fd: int,
    *,
    prior: _ReceiptSnapshot,
    replacement: bytes,
    owner: _OwnerKey,
    continuity: Callable[[], None],
) -> _FileIdentity:
    temporary = _temporary_name()
    replacement_identity = _prepare_temporary(directory_fd, temporary, replacement)
    try:
        continuity()
        _verify_owner_identity(directory_fd, owner)
        _exchange_paths(directory_fd, temporary, _RECEIPT_NAME)
    except _StateFailure:
        _remove_known_regular_file(
            directory_fd,
            temporary,
            identity=replacement_identity,
            raw=replacement,
        )
        raise

    displaced_matches = _named_receipt_matches(
        directory_fd,
        temporary,
        expected=prior,
        owner_key=owner.key,
    )
    replacement_matches = _named_regular_file_matches(
        directory_fd,
        _RECEIPT_NAME,
        identity=replacement_identity,
        raw=replacement,
    )
    if not displaced_matches or not replacement_matches:
        _restore_exchange(
            directory_fd,
            temporary,
            replacement_identity=replacement_identity,
            replacement=replacement,
        )
        raise _StateFailure()

    if not _named_regular_file_matches(
        directory_fd,
        _RECEIPT_NAME,
        identity=replacement_identity,
        raw=replacement,
    ):
        _restore_exchange(
            directory_fd,
            temporary,
            replacement_identity=replacement_identity,
            replacement=replacement,
        )
        raise _StateFailure()

    try:
        _remove_known_regular_file(
            directory_fd,
            temporary,
            identity=prior.identity,
            raw=prior.raw,
        )
    except (OSError, _StateFailure):
        _restore_exchange(
            directory_fd,
            temporary,
            replacement_identity=replacement_identity,
            replacement=replacement,
        )
        raise _StateFailure() from None
    try:
        os.fsync(directory_fd)
    except OSError:
        raise _StateFailure() from None
    return replacement_identity


def _remove_receipt_cas(
    directory_fd: int,
    *,
    prior: _ReceiptSnapshot,
    owner: _OwnerKey,
    continuity: Callable[[], None],
) -> None:
    # The fixed-locator mutation is one exclusive move, never verify-then-unlink. The namespace
    # is already durably not_found_verified, so a crash after this linearization cannot orphan a
    # live namespace. The moved inode stays bound to one held descriptor through authentication,
    # wipe, and fsync; a replacement of the random quarantine name is never truncated.
    quarantine = _temporary_name()
    fd = -1
    validated_prior = False
    try:
        continuity()
        _verify_owner_identity(directory_fd, owner)
        _rename_noreplace(directory_fd, _RECEIPT_NAME, quarantine)
    except _StateFailure:
        raise

    try:
        fd = os.open(
            quarantine,
            os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        before = _validate_regular_file(fd)
        _clear_nonblocking(fd)
        raw = _read_bounded(fd, _MAX_RECEIPT_BYTES + 1)
        after = _validate_regular_file(fd)
        current = _decode_receipt(raw, owner.key)
        if (
            not _same_file_object(_identity(before), _identity(after))
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or not _same_file_object(_identity(after), prior.identity)
            or not hmac.compare_digest(raw, prior.raw)
            or current != prior.receipt
            or _read_named_regular_file(
                directory_fd,
                _RECEIPT_NAME,
                maximum=_MAX_RECEIPT_BYTES,
            )
            is not None
        ):
            raise _StateFailure()
        # Make fixed-locator absence durable while the authenticated moved inode still contains
        # its complete terminal receipt. Only then may a power loss recover either the old fixed
        # entry or the full quarantine, never a fixed entry that points at a wiped inode.
        os.fsync(directory_fd)
        validated_prior = True
        os.ftruncate(fd, 0)
        os.fsync(fd)
        wiped = _validate_regular_file(fd)
        if not _same_file_object(_identity(wiped), prior.identity) or wiped.st_size != 0:
            raise _StateFailure()
        os.fsync(directory_fd)
    except (OSError, _StateFailure):
        if fd >= 0:
            _close_quietly(fd)
            fd = -1
        # Before exact held-FD validation, restore only by no-replace move. After validation,
        # terminal removal is commit-like: never install whatever may now occupy the random
        # quarantine as fixed authority. The remote namespace was already absence-verified, and
        # fixed absence makes every retry/provider path fail closed with zero remote action.
        if not validated_prior:
            with suppress(_StateFailure, OSError):
                _rename_noreplace(directory_fd, quarantine, _RECEIPT_NAME)
                os.fsync(directory_fd)
        raise _StateFailure() from None
    except BaseException:
        if fd >= 0:
            _close_quietly(fd)
            fd = -1
        if not validated_prior:
            with suppress(_StateFailure, OSError):
                _rename_noreplace(directory_fd, quarantine, _RECEIPT_NAME)
                os.fsync(directory_fd)
        raise

    if not _close_quietly(fd):
        raise _StateFailure()
    fd = -1

    # Random staging names are not authority locators. Compliant commands cannot race this
    # cleanup because they share the stable account anchor; an uncooperative same-UID directory
    # watcher is outside this local capability boundary (and can already read owner.key). If the
    # name changed, leave it intact instead of deleting it. Crashes may leave a zero-byte 0600
    # tombstone; no code scans or restores such names as authority.
    if _named_regular_file_matches(
        directory_fd,
        quarantine,
        identity=prior.identity,
        raw=b"",
    ):
        try:
            os.unlink(quarantine, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except OSError:
            raise _StateFailure() from None


def _prepare_temporary(directory_fd: int, name: str, raw: bytes) -> _FileIdentity:
    fd = -1
    created_identity: _FileIdentity | None = None
    try:
        fd = _create_temporary(directory_fd, name)
        created_identity = _identity(_validate_regular_file(fd))
        _write_all(fd, raw)
        os.fsync(fd)
        identity = _identity(_validate_regular_file(fd))
        closed = _close_quietly(fd)
        fd = -1
        if not closed:
            raise _StateFailure()
        return identity
    except (OSError, _StateFailure):
        if fd >= 0:
            _close_quietly(fd)
        if created_identity is not None:
            with suppress(_StateFailure):
                _remove_known_random_file(
                    directory_fd,
                    name,
                    identity=created_identity,
                )
        raise _StateFailure() from None


def _restore_exchange(
    directory_fd: int,
    temporary: str,
    *,
    replacement_identity: _FileIdentity,
    replacement: bytes,
) -> None:
    replacement_quarantine = _temporary_name()
    _rename_noreplace(directory_fd, _RECEIPT_NAME, replacement_quarantine)
    moved_replacement_matches = _named_regular_file_matches(
        directory_fd,
        replacement_quarantine,
        identity=replacement_identity,
        raw=replacement,
    )
    if not moved_replacement_matches:
        with suppress(_StateFailure):
            _rename_noreplace(
                directory_fd,
                replacement_quarantine,
                _RECEIPT_NAME,
            )
        raise _StateFailure()
    try:
        _rename_noreplace(directory_fd, temporary, _RECEIPT_NAME)
        _remove_known_regular_file(
            directory_fd,
            replacement_quarantine,
            identity=replacement_identity,
            raw=replacement,
        )
        os.fsync(directory_fd)
    except (OSError, _StateFailure):
        raise _StateFailure() from None


def _remove_known_regular_file(
    directory_fd: int,
    name: str,
    *,
    identity: _FileIdentity,
    raw: bytes,
) -> None:
    if not _named_regular_file_matches(
        directory_fd,
        name,
        identity=identity,
        raw=raw,
    ):
        raise _StateFailure()
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        raise _StateFailure() from None


def _remove_known_random_file(
    directory_fd: int,
    name: str,
    *,
    identity: _FileIdentity,
) -> None:
    # Internal 128-bit O_EXCL staging names are serialized by the stable account lock. They are
    # deliberately not authority locators; preserve any collider or replacement whose inode does
    # not match the object created by this invocation.
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError:
        raise _StateFailure() from None
    try:
        current = _identity(_validate_regular_file(fd))
    finally:
        closed = _close_quietly(fd)
    if not closed or not _same_file_object(current, identity):
        raise _StateFailure()
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        raise _StateFailure() from None


def _named_receipt_matches(
    directory_fd: int,
    name: str,
    *,
    expected: _ReceiptSnapshot,
    owner_key: bytes,
) -> bool:
    try:
        current = _read_named_receipt(directory_fd, name, owner_key)
    except _StateFailure:
        return False
    return bool(
        current is not None
        and _same_file_object(current.identity, expected.identity)
        and hmac.compare_digest(current.raw, expected.raw)
        and current.receipt == expected.receipt
    )


def _named_regular_file_matches(
    directory_fd: int,
    name: str,
    *,
    identity: _FileIdentity,
    raw: bytes,
) -> bool:
    try:
        current = _read_named_regular_file(directory_fd, name, maximum=max(len(raw), 1))
    except _StateFailure:
        return False
    return bool(
        current is not None
        and _same_file_object(current.identity, identity)
        and hmac.compare_digest(current.raw, raw)
    )


def _exchange_paths(directory_fd: int, first: str, second: str) -> None:
    _platform_rename(
        directory_fd,
        first,
        second,
        linux_flags=_LINUX_RENAME_EXCHANGE,
        darwin_flags=_DARWIN_RENAME_SWAP,
    )


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    _platform_rename(
        directory_fd,
        source,
        destination,
        linux_flags=_LINUX_RENAME_NOREPLACE,
        darwin_flags=_DARWIN_RENAME_EXCL,
    )


def _platform_rename(
    directory_fd: int,
    source: str,
    destination: str,
    *,
    linux_flags: int,
    darwin_flags: int,
) -> None:
    try:
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            function = library.renameat2
            flags = linux_flags
        elif sys.platform == "darwin":
            function = library.renameatx_np
            flags = darwin_flags
        else:
            raise _StateFailure()
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            directory_fd,
            os.fsencode(source),
            directory_fd,
            os.fsencode(destination),
            flags,
        )
    except (AttributeError, OSError, _StateFailure):
        raise _StateFailure() from None
    if result != 0:
        raise _StateFailure()


def _verify_prior_receipt(
    directory_fd: int,
    *,
    prior: _ReceiptSnapshot,
    owner_key: bytes,
) -> None:
    current = _read_receipt(directory_fd, owner_key)
    if (
        current is None
        or current.identity != prior.identity
        or not hmac.compare_digest(current.raw, prior.raw)
        or current.receipt != prior.receipt
    ):
        raise _StateFailure()


def _create_temporary(directory_fd: int, name: str) -> int:
    fd = -1
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(fd, 0o600)
        _validate_regular_file(fd)
        _clear_nonblocking(fd)
        return fd
    except OSError:
        if fd >= 0:
            _close_quietly(fd)
        raise _StateFailure() from None
    except _StateFailure:
        if fd >= 0:
            _close_quietly(fd)
        raise


def _temporary_name() -> str:
    return f".receipt-{secrets.token_hex(16)}.tmp"


def _directory_temporary_name() -> str:
    return f".owned-tiny-directory-{secrets.token_hex(16)}.tmp"


def _owner_temporary_name() -> str:
    return f".owner-{secrets.token_hex(16)}.tmp"


def _credential_tag(owner_key: bytes, api_key: str) -> str:
    return hmac.new(
        owner_key,
        b"pufferlab-owned-tiny-credential-v1\0" + api_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _credential_matches(receipt: OwnedTinyReceipt, api_key: str, owner_key: bytes) -> bool:
    try:
        candidate = _credential_tag(owner_key, api_key)
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(candidate, receipt.credential_tag)


def _derive_namespace(
    owner_key: bytes,
    *,
    nonce: str,
    region: str,
    credential_tag: str,
) -> str:
    binding = _canonical_json(
        {
            "credential_tag": credential_tag,
            "nonce": nonce,
            "purpose": _PURPOSE,
            "region": region,
        }
    )
    digest = hmac.new(
        owner_key,
        b"pufferlab-owned-tiny-namespace-v1\0" + binding,
        hashlib.sha256,
    ).hexdigest()
    return f"{_NAMESPACE_PREFIX}{digest}"


def _receipt_tag(owner_key: bytes, receipt: OwnedTinyReceipt) -> str:
    return hmac.new(
        owner_key,
        b"pufferlab-owned-tiny-receipt-v1\0" + _canonical_json(_unsigned_payload(receipt)),
        hashlib.sha256,
    ).hexdigest()


def _unsigned_payload(receipt: OwnedTinyReceipt) -> dict[str, object]:
    return {
        "format_version": receipt.format_version,
        "purpose": receipt.purpose,
        "creating_region": receipt.creating_region,
        "nonce": receipt.nonce,
        "namespace": receipt.namespace,
        "state": receipt.state.value,
        "credential_tag": receipt.credential_tag,
    }


def _encode_receipt(receipt: OwnedTinyReceipt) -> bytes:
    payload = _unsigned_payload(receipt)
    payload["authentication_tag"] = receipt.authentication_tag
    return _canonical_json(payload)


def _decode_receipt(raw: bytes, owner_key: bytes) -> OwnedTinyReceipt:
    duplicate = False

    def pairs_hook(pairs: list[tuple[str, object]]) -> Mapping[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _StateFailure() from None
    if duplicate or not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise _StateFailure()
    try:
        state = OwnedTinyState(value["state"])
    except (ValueError, TypeError):
        raise _StateFailure() from None
    receipt = OwnedTinyReceipt(
        format_version=cast(int, value["format_version"]),
        purpose=cast(str, value["purpose"]),
        creating_region=cast(str, value["creating_region"]),
        nonce=cast(str, value["nonce"]),
        namespace=cast(str, value["namespace"]),
        state=state,
        credential_tag=cast(str, value["credential_tag"]),
        authentication_tag=cast(str, value["authentication_tag"]),
    )
    if (
        type(receipt.format_version) is not int
        or receipt.format_version != _FORMAT_VERSION
        or receipt.purpose != _PURPOSE
        or not is_valid_metadata_probe_region(receipt.creating_region)
        or not _valid_hex(receipt.nonce)
        or not _valid_hex(receipt.credential_tag)
        or not _valid_hex(receipt.authentication_tag)
        or receipt.namespace
        != _derive_namespace(
            owner_key,
            nonce=receipt.nonce,
            region=receipt.creating_region,
            credential_tag=receipt.credential_tag,
        )
        or not hmac.compare_digest(_receipt_tag(owner_key, receipt), receipt.authentication_tag)
        or raw != _encode_receipt(receipt)
    ):
        raise _StateFailure()
    return receipt


def _valid_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _identity(info: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _same_file_object(first: _FileIdentity, second: _FileIdentity) -> bool:
    return first.device == second.device and first.inode == second.inode


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _clear_nonblocking(fd: int) -> None:
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if flags & os.O_NONBLOCK:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    except OSError:
        raise _StateFailure() from None


def _read_bounded(fd: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum
    try:
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError:
        raise _StateFailure() from None
    return b"".join(chunks)


def _write_all(fd: int, value: bytes) -> None:
    offset = 0
    try:
        while offset < len(value):
            written = os.write(fd, value[offset:])
            if written <= 0:
                raise _StateFailure()
            offset += written
    except OSError:
        raise _StateFailure() from None


def _close_quietly(fd: int) -> bool:
    try:
        os.close(fd)
    except OSError:
        return False
    return True


def _raise_public_failure(kind: _FailureKind) -> NoReturn:
    if kind is _FailureKind.BUSY:
        raise OwnedTinyBusyError() from None
    if kind is _FailureKind.MISSING:
        raise OwnedTinyReceiptMissingError() from None
    if kind is _FailureKind.CREDENTIAL:
        raise OwnedTinyCredentialMismatchError() from None
    if kind is _FailureKind.CLEANUP:
        raise OwnedTinyCleanupRequiredError() from None
    if kind is _FailureKind.TERMINAL:
        raise OwnedTinyTerminalReceiptError() from None
    raise OwnedTinyStateError("owned tiny state is unavailable", exit_code=2) from None
