"""Own and clean one authenticated Milestone 2 live-evaluation namespace."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderDeleteResult, ProviderNamespaceMetadata

_ROOT = Path(__file__).parents[1]
_DATA_DIR = _ROOT / "data"
_SESSION_PATH = _DATA_DIR / "m2-live-session.json"
_OWNER_KEY_PATH = _DATA_DIR / "m2-live-owner.key"
_NAMESPACE_PREFIX = "pufferlab-unix-live-"
_NAMESPACE_PATTERN = re.compile(r"pufferlab-unix-live-[0-9a-f]{24}")
_HEX_64_PATTERN = re.compile(r"[0-9a-f]{64}")
_SESSION_FIELDS = frozenset({"format_version", "nonce", "namespace", "ownership_tag"})
_OWNER_KEY_BYTES = 32
_NONCE_BYTES = 32


@dataclass(frozen=True, slots=True)
class M2LiveNamespaceSession:
    format_version: int
    nonce: str
    namespace: str
    ownership_tag: str


@dataclass(frozen=True, slots=True)
class _SessionPaths:
    session: Path
    owner_key: Path


@dataclass(frozen=True, slots=True)
class _LoadedFile:
    content: bytes
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _LoadedOwnedSession:
    session: M2LiveNamespaceSession
    session_file: _LoadedFile
    owner_key_file: _LoadedFile


_PRODUCTION_PATHS = _SessionPaths(session=_SESSION_PATH, owner_key=_OWNER_KEY_PATH)


class _CleanupProvider(Protocol):
    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult: ...

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata: ...

    async def close(self) -> None: ...


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> _CleanupProvider: ...


def _open_flags(*, write: bool, directory: bool = False) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    return flags | getattr(os, "O_NOFOLLOW", 0)


def _require_safe_paths(paths: _SessionPaths) -> Path:
    session_parent = paths.session.parent
    if session_parent != paths.owner_key.parent:
        raise RuntimeError("M2 live ownership files must share one fixed directory")
    session_parent.mkdir(parents=True, exist_ok=True)
    if session_parent.is_symlink() or not session_parent.is_dir():
        raise RuntimeError("M2 live ownership directory is invalid")
    return session_parent


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _open_flags(write=False, directory=True))
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeError("M2 live ownership directory is invalid")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, content: bytes) -> None:
    parent = path.parent
    descriptor = os.open(path, _open_flags(write=True), 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink(missing_ok=True)
            _fsync_directory(parent)
        except OSError:
            pass
        raise


def _read_secure_file(path: Path, *, label: str) -> _LoadedFile:
    try:
        descriptor = os.open(path, _open_flags(write=False))
    except OSError:
        raise RuntimeError(f"M2 live {label} is missing or unreadable") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError(f"M2 live {label} must be one regular 0600 file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read()
    except OSError:
        raise RuntimeError(f"M2 live {label} is missing or unreadable") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _LoadedFile(content=content, device=metadata.st_dev, inode=metadata.st_ino)


def _load_or_create_owner_key(
    paths: _SessionPaths,
    *,
    random_bytes: Callable[[int], bytes],
) -> _LoadedFile:
    _require_safe_paths(paths)
    with suppress(FileExistsError):
        _write_exclusive(paths.owner_key, random_bytes(_OWNER_KEY_BYTES))
    loaded = _read_secure_file(paths.owner_key, label="owner key")
    if len(loaded.content) != _OWNER_KEY_BYTES:
        raise RuntimeError("M2 live owner key is invalid")
    return loaded


def _namespace_for(owner_key: bytes, nonce: str) -> str:
    digest = hmac.new(
        owner_key,
        b"pufferlab-m2-namespace-v1\0" + bytes.fromhex(nonce),
        hashlib.sha256,
    ).hexdigest()
    return _NAMESPACE_PREFIX + digest[:24]


def _ownership_tag(
    owner_key: bytes,
    *,
    format_version: int,
    nonce: str,
    namespace: str,
) -> str:
    canonical = json.dumps(
        {
            "format_version": format_version,
            "namespace": namespace,
            "nonce": nonce,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hmac.new(
        owner_key,
        b"pufferlab-m2-session-v1\0" + canonical,
        hashlib.sha256,
    ).hexdigest()


def _validate_owned_session(session: M2LiveNamespaceSession, owner_key: bytes) -> None:
    if (
        session.format_version != 1
        or _HEX_64_PATTERN.fullmatch(session.nonce) is None
        or _NAMESPACE_PATTERN.fullmatch(session.namespace) is None
        or _HEX_64_PATTERN.fullmatch(session.ownership_tag) is None
    ):
        raise RuntimeError("M2 live namespace session is invalid; refusing cleanup")
    expected_namespace = _namespace_for(owner_key, session.nonce)
    expected_tag = _ownership_tag(
        owner_key,
        format_version=session.format_version,
        nonce=session.nonce,
        namespace=session.namespace,
    )
    if not hmac.compare_digest(session.namespace, expected_namespace) or not hmac.compare_digest(
        session.ownership_tag,
        expected_tag,
    ):
        raise RuntimeError("M2 live namespace session is not locally owned; refusing cleanup")


def _parse_session(content: bytes, owner_key: bytes) -> M2LiveNamespaceSession:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("M2 live namespace session is unreadable") from None
    if not isinstance(payload, dict) or frozenset(payload) != _SESSION_FIELDS:
        raise RuntimeError("M2 live namespace session has an invalid shape")
    if (
        not isinstance(payload["format_version"], int)
        or isinstance(payload["format_version"], bool)
        or not isinstance(payload["nonce"], str)
        or not isinstance(payload["namespace"], str)
        or not isinstance(payload["ownership_tag"], str)
    ):
        raise RuntimeError("M2 live namespace session has invalid values")
    session = M2LiveNamespaceSession(**payload)
    _validate_owned_session(session, owner_key)
    return session


def _load_owned_session_at(paths: _SessionPaths) -> _LoadedOwnedSession:
    _require_safe_paths(paths)
    owner_key_file = _read_secure_file(paths.owner_key, label="owner key")
    if len(owner_key_file.content) != _OWNER_KEY_BYTES:
        raise RuntimeError("M2 live owner key is invalid")
    session_file = _read_secure_file(paths.session, label="namespace session")
    session = _parse_session(session_file.content, owner_key_file.content)
    return _LoadedOwnedSession(
        session=session,
        session_file=session_file,
        owner_key_file=owner_key_file,
    )


def _create_session_at(
    paths: _SessionPaths,
    *,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> M2LiveNamespaceSession:
    owner_key_file = _load_or_create_owner_key(paths, random_bytes=random_bytes)
    nonce = random_bytes(_NONCE_BYTES).hex()
    namespace = _namespace_for(owner_key_file.content, nonce)
    session = M2LiveNamespaceSession(
        format_version=1,
        nonce=nonce,
        namespace=namespace,
        ownership_tag=_ownership_tag(
            owner_key_file.content,
            format_version=1,
            nonce=nonce,
            namespace=namespace,
        ),
    )
    _validate_owned_session(session, owner_key_file.content)
    encoded = json.dumps(asdict(session), sort_keys=True).encode("utf-8") + b"\n"
    _write_exclusive(paths.session, encoded)
    return session


def create_session() -> M2LiveNamespaceSession:
    """Create only the fixed production record with OS-generated ownership material."""
    return _create_session_at(_PRODUCTION_PATHS)


def load_session() -> M2LiveNamespaceSession:
    """Load only the fixed, authenticated production session record."""
    return _load_owned_session_at(_PRODUCTION_PATHS).session


def _require_unchanged_record(paths: _SessionPaths, expected: _LoadedOwnedSession) -> None:
    current = _load_owned_session_at(paths)
    if current != expected:
        raise RuntimeError("M2 live ownership capability changed during cleanup; retaining record")


async def _cleanup_session_at(
    paths: _SessionPaths,
    *,
    settings: Settings,
    provider_factory: _ProviderFactory,
    attempts: int = 30,
    poll_interval: float = 0.5,
) -> M2LiveNamespaceSession:
    loaded = _load_owned_session_at(paths)
    if attempts < 1 or poll_interval < 0:
        raise ValueError("cleanup polling bounds are invalid")

    secret = settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        raise RuntimeError("TURBOPUFFER_API_KEY is required for M2 live namespace cleanup")
    provider = provider_factory(
        api_key=secret.get_secret_value(),
        region=settings.turbopuffer_region,
    )
    not_found_confirmed = False
    try:
        try:
            await provider.delete_namespace(loaded.session.namespace)
        except ProviderError as error:
            if error.details.code is not ApiErrorCode.NOT_FOUND:
                raise

        for attempt in range(attempts):
            try:
                await provider.namespace_metadata(loaded.session.namespace)
            except ProviderError as error:
                if error.details.code is ApiErrorCode.NOT_FOUND:
                    not_found_confirmed = True
                    break
                raise
            if attempt + 1 < attempts:
                await asyncio.sleep(poll_interval)
        if not not_found_confirmed:
            raise RuntimeError("M2 live namespace deletion was not confirmed as not found")
    finally:
        await provider.close()

    _require_unchanged_record(paths, loaded)
    paths.session.unlink()
    _fsync_directory(paths.session.parent)
    return loaded.session


async def cleanup_session() -> M2LiveNamespaceSession:
    """Clean only the authenticated fixed-path production capability."""
    return await _cleanup_session_at(
        _PRODUCTION_PATHS,
        settings=Settings(),
        provider_factory=TurbopufferProvider,
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Expose fixed production start/show/cleanup commands without target injection."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="create one immutable ignored session record")
    subparsers.add_parser("show", help="print the authenticated owned namespace")
    subparsers.add_parser("cleanup", help="delete and verify only the authenticated record")
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "start":
            session = create_session()
            print(f"session_file={_SESSION_PATH.relative_to(_ROOT)}")
            print(f"PUFFERLAB_SEARCH_NAMESPACE={session.namespace}")
        elif arguments.command == "show":
            session = load_session()
            print(f"PUFFERLAB_SEARCH_NAMESPACE={session.namespace}")
        else:
            session = asyncio.run(cleanup_session())
            print(f"cleanup namespace={session.namespace} status=not_found_verified")
    except Exception:
        print(f"m2_namespace_session command={arguments.command} status=failed", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
