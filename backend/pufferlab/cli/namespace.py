"""Receipt-bound show and cleanup commands for the one generated tiny namespace."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.owned_tiny import (
    OwnedTinyCredentialMismatchError,
    OwnedTinyReceiptMissingError,
    OwnedTinyState,
    OwnedTinyStateError,
    owned_tiny_existing_operation,
)
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderDeleteResult, ProviderNamespaceMetadata

_NOT_FOUND_ATTEMPTS = 20
_NOT_FOUND_POLL_INTERVAL = 0.25


class NamespaceCommandError(RuntimeError):
    """A value-free namespace command failure with a stable process exit code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _CleanupProvider(Protocol):
    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult: ...

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata: ...

    async def close(self) -> None: ...


class _ProviderFactory(Protocol):
    def __call__(self, *, api_key: str, region: str) -> _CleanupProvider: ...


_PROVIDER_FACTORY: _ProviderFactory = TurbopufferProvider


def show_owned_tiny(*, emit: Callable[[str], None]) -> None:
    """Print the sole intentional configured-target output from an authenticated receipt."""

    state_failure: tuple[str, int] | None = None
    try:
        with owned_tiny_existing_operation() as operation:
            snapshot = operation.load(required=True)
            assert snapshot is not None
            emit(f"TURBOPUFFER_REGION={snapshot.receipt.creating_region}")
            emit(f"PUFFERLAB_SEARCH_NAMESPACE={snapshot.receipt.namespace}")
    except OwnedTinyStateError as error:
        state_failure = (str(error), error.exit_code)
    if state_failure is not None:
        raise NamespaceCommandError(state_failure[0], exit_code=state_failure[1]) from None


async def cleanup_owned_tiny(
    settings: Settings,
    *,
    emit: Callable[[str], None],
) -> None:
    """Delete and verify absence of only the exact authenticated fixed receipt target."""

    state_failure: tuple[str, int] | None = None
    try:
        with owned_tiny_existing_operation() as operation:
            snapshot = operation.load(required=True)
            assert snapshot is not None
            if snapshot.receipt.state is OwnedTinyState.NOT_FOUND_VERIFIED:
                operation.remove_terminal(snapshot)
                _emit_cleanup_complete(emit)
                return

            secret = settings.turbopuffer_api_key
            if secret is None:
                raise NamespaceCommandError(
                    "TURBOPUFFER_API_KEY is required for owned tiny cleanup",
                    exit_code=2,
                )
            api_key = ""
            key_failed = False
            try:
                api_key = secret.get_secret_value()
            except Exception:
                key_failed = True
            secret = None
            settings = settings.model_copy(update={"turbopuffer_api_key": None})
            if key_failed or not api_key:
                api_key = ""
                raise NamespaceCommandError(
                    "TURBOPUFFER_API_KEY is required for owned tiny cleanup",
                    exit_code=2,
                )
            operation.require_credential(snapshot, api_key)

            if snapshot.receipt.state in {
                OwnedTinyState.INTENT,
                OwnedTinyState.CREATED,
                OwnedTinyState.READY,
            }:
                snapshot = operation.transition(snapshot, OwnedTinyState.CLEANUP_REQUESTED)
            elif snapshot.receipt.state is not OwnedTinyState.CLEANUP_REQUESTED:
                api_key = ""
                raise NamespaceCommandError("owned tiny cleanup state is invalid")

            operation.authenticate_current(snapshot)
            provider: _CleanupProvider | None = None
            factory_failed = False
            try:
                provider = _PROVIDER_FACTORY(
                    api_key=api_key,
                    region=snapshot.receipt.creating_region,
                )
            except Exception:
                factory_failed = True
            api_key = ""
            if factory_failed or provider is None:
                raise NamespaceCommandError("owned tiny cleanup provider could not start")

            cleanup_snapshot = snapshot
            control = await _delete_verify_and_close(
                provider,
                namespace=snapshot.receipt.namespace,
                before_provider_action=lambda: operation.authenticate_current(cleanup_snapshot),
            )
            provider = None
            if control.cancelled:
                _raise_cancelled()
            if not control.succeeded:
                raise NamespaceCommandError("owned tiny cleanup was not verified")

            snapshot = operation.transition(snapshot, OwnedTinyState.NOT_FOUND_VERIFIED)
            operation.remove_terminal(snapshot)
            _emit_cleanup_complete(emit)
    except NamespaceCommandError:
        raise
    except OwnedTinyCredentialMismatchError as error:
        api_key = ""
        state_failure = (str(error), error.exit_code)
    except OwnedTinyReceiptMissingError as error:
        api_key = ""
        state_failure = (str(error), error.exit_code)
    except OwnedTinyStateError as error:
        api_key = ""
        state_failure = (str(error), error.exit_code)
    if state_failure is not None:
        raise NamespaceCommandError(state_failure[0], exit_code=state_failure[1]) from None


@dataclass(frozen=True, slots=True)
class _CleanupControl:
    succeeded: bool
    cancelled: bool = False


async def _delete_verify_and_close(
    provider: _CleanupProvider,
    *,
    namespace: str,
    before_provider_action: Callable[[], None],
) -> _CleanupControl:
    succeeded = False
    cancelled = False
    operation_failed = False
    try:
        try:
            before_provider_action()
            await provider.delete_namespace(namespace)
        except ProviderError as error:
            if error.details.code is not ApiErrorCode.NOT_FOUND:
                operation_failed = True
        if not operation_failed:
            succeeded, verification_cancelled = await _verify_not_found(
                provider,
                namespace=namespace,
                before_provider_action=before_provider_action,
            )
            cancelled = cancelled or verification_cancelled
    except asyncio.CancelledError:
        cancelled = True
    except Exception:
        operation_failed = True

    close_failed, close_cancelled = await _drain_close(provider)
    cancelled = cancelled or close_cancelled
    return _CleanupControl(
        succeeded=succeeded and not operation_failed and not close_failed and not cancelled,
        cancelled=cancelled,
    )


async def _verify_not_found(
    provider: _CleanupProvider,
    *,
    namespace: str,
    before_provider_action: Callable[[], None],
) -> tuple[bool, bool]:
    for attempt in range(_NOT_FOUND_ATTEMPTS):
        try:
            before_provider_action()
            await provider.namespace_metadata(namespace)
        except ProviderError as error:
            if error.details.code is ApiErrorCode.NOT_FOUND:
                return True, False
            return False, False
        except asyncio.CancelledError:
            return False, True
        except Exception:
            return False, False
        if attempt + 1 < _NOT_FOUND_ATTEMPTS:
            try:
                await asyncio.sleep(_NOT_FOUND_POLL_INTERVAL)
            except asyncio.CancelledError:
                return False, True
    return False, False


async def _drain_close(provider: _CleanupProvider) -> tuple[bool, bool]:
    try:
        close_task = asyncio.create_task(provider.close())
    except Exception:
        return True, False
    cancelled = False
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    close_failed = False
    try:
        close_task.result()
    except asyncio.CancelledError:
        cancelled = True
        close_failed = True
    except Exception:
        close_failed = True
    return close_failed, cancelled


def _emit_cleanup_complete(emit: Callable[[str], None]) -> None:
    emit("cleanup verified; clear PUFFERLAB_SEARCH_NAMESPACE and restart the API")


def _raise_cancelled() -> None:
    raise asyncio.CancelledError() from None
