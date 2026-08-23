"""Authenticated, installation-local ownership for one generated tiny namespace.

The production locator and child names in this module are deliberately fixed.  Public callers can
inspect or operate the capability, but cannot supply a state path, namespace, nonce, or ownership
token.  Tests isolate the filesystem by monkeypatching the private ``_production_state_path``
helper.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import hmac
import json
import os
import pwd
import secrets
import stat
from collections.abc import Iterator, Mapping
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


@dataclass(slots=True, repr=False)
class _StateDirectory:
    fd: int


def _production_state_path() -> Path:
    """Resolve the frozen state location from the POSIX account database, never the environment."""

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
    return home.joinpath(*_STATE_COMPONENTS)


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
        operation = _begin_operation(_production_state_path(), create=create)
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
        directory: _StateDirectory,
        *,
        lock_fd: int,
        owner: _OwnerKey,
    ) -> None:
        self._directory = directory
        self._lock_fd = lock_fd
        self._owner = owner
        self._closed = False

    def load(self, *, required: bool = True) -> _ReceiptSnapshot | None:
        invalid = False
        snapshot: _ReceiptSnapshot | None = None
        try:
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
        try:
            _atomic_create_receipt(self._directory.fd, raw)
            snapshot = _read_receipt(self._directory.fd, self._owner.key)
        except _StateFailure:
            persistence_failed = True
        if (
            persistence_failed
            or snapshot is None
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
        try:
            _verify_owner_identity(self._directory.fd, self._owner)
            _atomic_replace_receipt(
                self._directory.fd,
                prior=snapshot,
                replacement=replacement_raw,
                owner=self._owner,
            )
            updated = _read_receipt(self._directory.fd, self._owner.key)
        except _StateFailure:
            persistence_failed = True
        if (
            persistence_failed
            or updated is None
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
            _verify_owner_identity(self._directory.fd, self._owner)
            _remove_receipt_cas(
                self._directory.fd,
                prior=snapshot,
                owner=self._owner,
            )
        except _StateFailure:
            removal_failed = True
        if removal_failed:
            raise OwnedTinyStateError("owned tiny terminal receipt could not be removed") from None

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        lock_ok = _close_quietly(self._lock_fd)
        directory_ok = _close_quietly(self._directory.fd)
        return lock_ok and directory_ok


def _begin_operation(path: Path, *, create: bool) -> _OwnedTinyOperation:
    directory = _open_state_directory(path, create=create)
    if directory is None:
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
            _close_quietly(lock_fd)
            _close_quietly(directory.fd)
            raise _StateFailure(_FailureKind.BUSY if busy else _FailureKind.INVALID) from None
        owner = _read_owner_key(directory.fd, required=False)
        if owner is None:
            if not create:
                raise _StateFailure(_FailureKind.MISSING)
            owner = _create_owner_key(directory.fd)
        return _OwnedTinyOperation(directory, lock_fd=lock_fd, owner=owner)
    except _StateFailure:
        if lock_fd >= 0:
            _close_quietly(lock_fd)
        _close_quietly(directory.fd)
        raise


def _open_state_directory(path: Path, *, create: bool) -> _StateDirectory | None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise _StateFailure()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = -1
    try:
        current = os.open("/", flags)
        parts = path.parts[1:]
        for index, component in enumerate(parts):
            child = -1
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    _close_quietly(current)
                    return None
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current)
                    child = os.open(component, flags, dir_fd=current)
                    os.fchmod(child, 0o700)
                    os.fsync(current)
                except OSError:
                    _close_quietly(current)
                    raise _StateFailure() from None
            except OSError:
                _close_quietly(current)
                raise _StateFailure() from None
            if child < 0 or not _close_quietly(current):
                if child >= 0:
                    _close_quietly(child)
                raise _StateFailure()
            current = child
            try:
                info = os.fstat(current)
            except OSError:
                raise _StateFailure() from None
            if not stat.S_ISDIR(info.st_mode):
                raise _StateFailure()
            if index == len(parts) - 1 and (
                info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise _StateFailure()
        return _StateDirectory(fd=current)
    except _StateFailure:
        if current >= 0:
            _close_quietly(current)
        raise


def _open_fixed_file(directory_fd: int, name: str, *, create: bool, writable: bool) -> int:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW
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
    fd = -1
    try:
        fd = os.open(
            _OWNER_KEY_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(fd, 0o600)
        _write_all(fd, key)
        os.fsync(fd)
        if not _close_quietly(fd):
            raise _StateFailure()
        fd = -1
        os.fsync(directory_fd)
    except (OSError, _StateFailure):
        if fd >= 0:
            _close_quietly(fd)
        _unlink_quietly(directory_fd, _OWNER_KEY_NAME)
        with suppress(OSError):
            os.fsync(directory_fd)
        raise _StateFailure() from None
    owner = _read_owner_key(directory_fd, required=True)
    if owner is None:
        raise _StateFailure()
    return owner


def _read_owner_key(directory_fd: int, *, required: bool) -> _OwnerKey | None:
    try:
        fd = os.open(_OWNER_KEY_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        if required:
            raise _StateFailure() from None
        return None
    except OSError:
        raise _StateFailure() from None
    try:
        before = _validate_regular_file(fd)
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
    try:
        fd = os.open(_RECEIPT_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError:
        raise _StateFailure() from None
    try:
        before = _validate_regular_file(fd)
        raw = _read_bounded(fd, _MAX_RECEIPT_BYTES + 1)
        after = _validate_regular_file(fd)
    finally:
        closed = _close_quietly(fd)
    if (
        not closed
        or owner_key is None
        or not raw
        or len(raw) > _MAX_RECEIPT_BYTES
        or _identity(before) != _identity(after)
    ):
        raise _StateFailure()
    receipt = _decode_receipt(raw, owner_key)
    return _ReceiptSnapshot(receipt=receipt, raw=raw, identity=_identity(after))


def _atomic_create_receipt(directory_fd: int, raw: bytes) -> None:
    temporary = _temporary_name()
    fd = -1
    try:
        fd = _create_temporary(directory_fd, temporary)
        _write_all(fd, raw)
        os.fsync(fd)
        if not _close_quietly(fd):
            raise _StateFailure()
        fd = -1
        os.link(
            temporary,
            _RECEIPT_NAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError:
        if fd >= 0:
            _close_quietly(fd)
        _unlink_quietly(directory_fd, temporary)
        raise _StateFailure() from None
    except _StateFailure:
        if fd >= 0:
            _close_quietly(fd)
        _unlink_quietly(directory_fd, temporary)
        raise


def _atomic_replace_receipt(
    directory_fd: int,
    *,
    prior: _ReceiptSnapshot,
    replacement: bytes,
    owner: _OwnerKey,
) -> None:
    temporary = _temporary_name()
    fd = -1
    try:
        fd = _create_temporary(directory_fd, temporary)
        _write_all(fd, replacement)
        os.fsync(fd)
        if not _close_quietly(fd):
            raise _StateFailure()
        fd = -1
        _verify_owner_identity(directory_fd, owner)
        _verify_prior_receipt(directory_fd, prior=prior, owner_key=owner.key)
        os.replace(
            temporary,
            _RECEIPT_NAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError:
        if fd >= 0:
            _close_quietly(fd)
        _unlink_quietly(directory_fd, temporary)
        raise _StateFailure() from None
    except _StateFailure:
        if fd >= 0:
            _close_quietly(fd)
        _unlink_quietly(directory_fd, temporary)
        raise


def _remove_receipt_cas(
    directory_fd: int,
    *,
    prior: _ReceiptSnapshot,
    owner: _OwnerKey,
) -> None:
    _verify_owner_identity(directory_fd, owner)
    _verify_prior_receipt(directory_fd, prior=prior, owner_key=owner.key)
    try:
        os.unlink(_RECEIPT_NAME, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError:
        # If unlink succeeded but the directory sync failed, best-effort restoration preserves the
        # authenticated terminal receipt for an explicit retry.
        try:
            present = _read_receipt(directory_fd, owner.key)
        except _StateFailure:
            present = None
        if present is None:
            with suppress(_StateFailure):
                _atomic_create_receipt(directory_fd, prior.raw)
        raise _StateFailure() from None


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
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(fd, 0o600)
        _validate_regular_file(fd)
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


def _unlink_quietly(directory_fd: int, name: str) -> None:
    with suppress(OSError):
        os.unlink(name, dir_fd=directory_fd)


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
