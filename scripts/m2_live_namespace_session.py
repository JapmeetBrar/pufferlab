"""Own and clean one immutable Milestone 2 live-evaluation namespace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderDeleteResult, ProviderNamespaceMetadata

_ROOT = Path(__file__).parents[1]
_SESSION_PATH = _ROOT / "data" / "m2-live-session.json"
_NAMESPACE_PATTERN = re.compile(r"pufferlab-unix-live-[0-9a-f]{24}")
_SESSION_FIELDS = frozenset({"format_version", "namespace"})


@dataclass(frozen=True, slots=True)
class M2LiveNamespaceSession:
    format_version: int
    namespace: str


@dataclass(frozen=True, slots=True)
class _LoadedSession:
    session: M2LiveNamespaceSession
    device: int
    inode: int


class _CleanupProvider(Protocol):
    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult: ...

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata: ...

    async def close(self) -> None: ...


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> _CleanupProvider: ...


def _validate_session(session: M2LiveNamespaceSession) -> None:
    if session.format_version != 1 or _NAMESPACE_PATTERN.fullmatch(session.namespace) is None:
        raise RuntimeError("M2 live namespace session is invalid; refusing cleanup")


def _open_flags(*, write: bool) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
    return flags | getattr(os, "O_NOFOLLOW", 0)


def create_session(
    path: Path = _SESSION_PATH,
    *,
    token_factory: Callable[[int], str] = secrets.token_hex,
) -> M2LiveNamespaceSession:
    """Create exactly one exclusive record for an internally generated namespace."""
    session = M2LiveNamespaceSession(
        format_version=1,
        namespace=f"pufferlab-unix-live-{token_factory(12)}",
    )
    _validate_session(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, _open_flags(write=True), 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(session), handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return session


def _load_session_record(path: Path) -> _LoadedSession:
    try:
        descriptor = os.open(path, _open_flags(write=False))
    except OSError:
        raise RuntimeError("M2 live namespace session is missing or unreadable") from None

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError("M2 live namespace session must be one regular 0600 file")
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("M2 live namespace session is missing or unreadable") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if not isinstance(payload, dict) or frozenset(payload) != _SESSION_FIELDS:
        raise RuntimeError("M2 live namespace session has an invalid shape")
    format_version = payload["format_version"]
    namespace = payload["namespace"]
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise RuntimeError("M2 live namespace session has invalid values")
    if not isinstance(namespace, str):
        raise RuntimeError("M2 live namespace session has invalid values")
    session = M2LiveNamespaceSession(format_version=format_version, namespace=namespace)
    _validate_session(session)
    return _LoadedSession(session=session, device=metadata.st_dev, inode=metadata.st_ino)


def load_session(path: Path = _SESSION_PATH) -> M2LiveNamespaceSession:
    """Load a record only after validating its exact shape, target, type, and mode."""
    return _load_session_record(path).session


def _require_unchanged_record(path: Path, expected: _LoadedSession) -> None:
    current = _load_session_record(path)
    if (
        current.session != expected.session
        or current.device != expected.device
        or current.inode != expected.inode
    ):
        raise RuntimeError("M2 live namespace session changed during cleanup; retaining record")


async def cleanup_session(
    path: Path = _SESSION_PATH,
    *,
    settings: Settings | None = None,
    provider_factory: _ProviderFactory = TurbopufferProvider,
    attempts: int = 30,
    poll_interval: float = 0.5,
) -> M2LiveNamespaceSession:
    """Delete only the retained namespace and forget it only after confirmed clean closure."""
    loaded = _load_session_record(path)
    if attempts < 1 or poll_interval < 0:
        raise ValueError("cleanup polling bounds are invalid")

    resolved_settings = settings or Settings()
    secret = resolved_settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        raise RuntimeError("TURBOPUFFER_API_KEY is required for M2 live namespace cleanup")

    provider = provider_factory(
        api_key=secret.get_secret_value(),
        region=resolved_settings.turbopuffer_region,
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

    _require_unchanged_record(path, loaded)
    path.unlink()
    return loaded.session


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    path: Path = _SESSION_PATH,
    settings: Settings | None = None,
    provider_factory: _ProviderFactory = TurbopufferProvider,
) -> int:
    """Run the redacted command surface; cleanup has no namespace argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="create one immutable ignored session record")
    subparsers.add_parser("show", help="print the owned namespace without credentials")
    subparsers.add_parser("cleanup", help="delete and verify only the recorded namespace")
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "start":
            session = create_session(path)
            try:
                session_path = str(path.relative_to(_ROOT))
            except ValueError:
                session_path = path.name
            print(f"session_file={session_path}")
            print(f"PUFFERLAB_SEARCH_NAMESPACE={session.namespace}")
        elif arguments.command == "show":
            session = load_session(path)
            print(f"PUFFERLAB_SEARCH_NAMESPACE={session.namespace}")
        else:
            session = asyncio.run(
                cleanup_session(
                    path,
                    settings=settings,
                    provider_factory=provider_factory,
                )
            )
            print(f"cleanup namespace={session.namespace} status=not_found_verified")
    except Exception:
        print(f"m2_namespace_session command={arguments.command} status=failed", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
