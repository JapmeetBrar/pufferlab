"""Own and clean one immutable Milestone 1 live-verification namespace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderDeleteResult, ProviderNamespaceMetadata

_ROOT = Path(__file__).parents[1]
_SESSION_PATH = _ROOT / "data" / "m1-live-session.json"
_NAMESPACE_PATTERN = re.compile(r"pufferlab-tiny-[0-9a-f]{24}")


@dataclass(frozen=True, slots=True)
class LiveNamespaceSession:
    format_version: int
    namespace: str


class _CleanupProvider(Protocol):
    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult: ...

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata: ...

    async def close(self) -> None: ...


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> _CleanupProvider: ...


def _validate_session(session: LiveNamespaceSession) -> None:
    if session.format_version != 1 or _NAMESPACE_PATTERN.fullmatch(session.namespace) is None:
        raise RuntimeError("live namespace session is invalid; refusing cleanup")


def create_session(
    path: Path = _SESSION_PATH,
    *,
    token_factory: Callable[[int], str] = secrets.token_hex,
) -> LiveNamespaceSession:
    session = LiveNamespaceSession(
        format_version=1,
        namespace=f"pufferlab-tiny-{token_factory(12)}",
    )
    _validate_session(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(session), handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return session


def load_session(path: Path = _SESSION_PATH) -> LiveNamespaceSession:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("live namespace session is missing or unreadable") from None
    if not isinstance(payload, dict) or set(payload) != {"format_version", "namespace"}:
        raise RuntimeError("live namespace session has an invalid shape")
    format_version = payload["format_version"]
    namespace = payload["namespace"]
    if not isinstance(format_version, int) or not isinstance(namespace, str):
        raise RuntimeError("live namespace session has invalid values")
    session = LiveNamespaceSession(format_version=format_version, namespace=namespace)
    _validate_session(session)
    return session


async def cleanup_session(
    path: Path = _SESSION_PATH,
    *,
    settings: Settings | None = None,
    provider_factory: _ProviderFactory = TurbopufferProvider,
    attempts: int = 30,
    poll_interval: float = 0.5,
) -> LiveNamespaceSession:
    session = load_session(path)
    resolved_settings = settings or Settings()
    secret = resolved_settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        raise RuntimeError("TURBOPUFFER_API_KEY is required for live namespace cleanup")
    if attempts < 1 or poll_interval < 0:
        raise ValueError("cleanup polling bounds are invalid")

    provider = provider_factory(
        api_key=secret.get_secret_value(),
        region=resolved_settings.turbopuffer_region,
    )
    not_found_confirmed = False
    try:
        try:
            await provider.delete_namespace(session.namespace)
        except ProviderError as error:
            if error.details.code is not ApiErrorCode.NOT_FOUND:
                raise

        for attempt in range(attempts):
            try:
                await provider.namespace_metadata(session.namespace)
            except ProviderError as error:
                if error.details.code is ApiErrorCode.NOT_FOUND:
                    not_found_confirmed = True
                    break
                raise
            if attempt + 1 < attempts:
                await asyncio.sleep(poll_interval)
        if not not_found_confirmed:
            raise RuntimeError("live namespace deletion was not confirmed as not found")
    finally:
        await provider.close()
    path.unlink()
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="create one immutable ignored session record")
    subparsers.add_parser("show", help="print the owned namespace without credentials")
    subparsers.add_parser("cleanup", help="delete and verify only the recorded namespace")
    arguments = parser.parse_args()

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


if __name__ == "__main__":
    main()
