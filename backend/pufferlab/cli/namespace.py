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

    outcome = await _execute_cleanup_owned_tiny(settings, emit=emit)
    settings = settings.model_copy(update={"turbopuffer_api_key": None})
    if outcome.cancelled:
        _raise_cancelled()
    if outcome.message is not None:
        _raise_namespace_outcome(outcome)


@dataclass(frozen=True, slots=True)
class _NamespaceOutcome:
    message: str | None = None
    exit_code: int = 0
    cancelled: bool = False


async def _execute_cleanup_owned_tiny(
    settings: Settings,
    *,
    emit: Callable[[str], None],
) -> _NamespaceOutcome:
    """Return only a value-free result after all authority and provider frames unwind."""

    api_key = ""
    try:
        with owned_tiny_existing_operation() as operation:
            snapshot = operation.load(required=True)
            assert snapshot is not None
            if snapshot.receipt.state is OwnedTinyState.NOT_FOUND_VERIFIED:
                operation.remove_terminal(snapshot)
                _emit_cleanup_complete(emit)
                return _NamespaceOutcome()

            secret = settings.turbopuffer_api_key
            if secret is None:
                return _NamespaceOutcome(
                    message="TURBOPUFFER_API_KEY is required for owned tiny cleanup",
                    exit_code=2,
                )
            key_failed = False
            key_cancelled = False
            try:
                api_key = secret.get_secret_value()
            except (KeyboardInterrupt, asyncio.CancelledError):
                key_cancelled = True
            except BaseException:
                key_failed = True
            secret = None
            settings = settings.model_copy(update={"turbopuffer_api_key": None})
            if key_cancelled:
                api_key = ""
                return _NamespaceOutcome(cancelled=True)
            if key_failed or not api_key:
                api_key = ""
                return _NamespaceOutcome(
                    message="TURBOPUFFER_API_KEY is required for owned tiny cleanup",
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
                return _NamespaceOutcome(message="owned tiny cleanup state is invalid", exit_code=1)

            operation.authenticate_current(snapshot)
            provider: _CleanupProvider | None = None
            provider_started = False
            factory_failed = False
            factory_cancelled = False
            control = _CleanupControl(succeeded=False)
            close_control = _CloseControl(failed=False)
            try:
                try:
                    provider = _PROVIDER_FACTORY(
                        api_key=api_key,
                        region=snapshot.receipt.creating_region,
                    )
                    provider_started = provider is not None
                except (KeyboardInterrupt, asyncio.CancelledError):
                    factory_cancelled = True
                except BaseException:
                    factory_failed = True
                api_key = ""
                if provider is not None:
                    cleanup_snapshot = snapshot
                    try:
                        control = await _delete_and_verify(
                            provider,
                            namespace=snapshot.receipt.namespace,
                            before_provider_action=lambda: operation.authenticate_current(
                                cleanup_snapshot
                            ),
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        control = _CleanupControl(succeeded=False, cancelled=True)
                    except BaseException:
                        control = _CleanupControl(succeeded=False, internal_failure=True)
            finally:
                api_key = ""
                if provider is not None:
                    close_control = await _drain_close(provider)
                    provider = None

            if factory_cancelled:
                return _NamespaceOutcome(cancelled=True)
            if factory_failed or not provider_started:
                if close_control.cancelled:
                    return _NamespaceOutcome(cancelled=True)
                return _NamespaceOutcome(
                    message="owned tiny cleanup provider could not start",
                    exit_code=1,
                )
            if control.cancelled:
                return _NamespaceOutcome(cancelled=True)
            if control.internal_failure:
                return _NamespaceOutcome(
                    message="owned tiny cleanup was not verified",
                    exit_code=1,
                )
            if close_control.cancelled:
                return _NamespaceOutcome(cancelled=True)
            if close_control.internal_failure or close_control.failed or not control.succeeded:
                return _NamespaceOutcome(
                    message="owned tiny cleanup was not verified",
                    exit_code=1,
                )

            snapshot = operation.transition(snapshot, OwnedTinyState.NOT_FOUND_VERIFIED)
            operation.remove_terminal(snapshot)
            _emit_cleanup_complete(emit)
            return _NamespaceOutcome()
    except (KeyboardInterrupt, asyncio.CancelledError):
        api_key = ""
        return _NamespaceOutcome(cancelled=True)
    except OwnedTinyCredentialMismatchError as error:
        api_key = ""
        return _NamespaceOutcome(message=str(error), exit_code=error.exit_code)
    except OwnedTinyReceiptMissingError as error:
        api_key = ""
        return _NamespaceOutcome(message=str(error), exit_code=error.exit_code)
    except OwnedTinyStateError as error:
        api_key = ""
        return _NamespaceOutcome(message=str(error), exit_code=error.exit_code)
    except BaseException:
        api_key = ""
        return _NamespaceOutcome(message="owned tiny cleanup failed", exit_code=1)


@dataclass(frozen=True, slots=True)
class _CleanupControl:
    succeeded: bool
    cancelled: bool = False
    internal_failure: bool = False


async def _delete_and_verify(
    provider: _CleanupProvider,
    *,
    namespace: str,
    before_provider_action: Callable[[], None],
) -> _CleanupControl:
    succeeded = False
    cancelled = False
    operation_failed = False
    internal_failure = False
    try:
        try:
            before_provider_action()
            await provider.delete_namespace(namespace)
        except ProviderError as error:
            if error.details.code is not ApiErrorCode.NOT_FOUND:
                operation_failed = True
        if not operation_failed:
            verification = await _verify_not_found(
                provider,
                namespace=namespace,
                before_provider_action=before_provider_action,
            )
            succeeded = verification.succeeded
            cancelled = cancelled or verification.cancelled
            internal_failure = internal_failure or verification.internal_failure
    except (KeyboardInterrupt, asyncio.CancelledError):
        cancelled = True
    except BaseException:
        operation_failed = True
        internal_failure = True

    return _CleanupControl(
        succeeded=(succeeded and not operation_failed and not cancelled and not internal_failure),
        cancelled=cancelled,
        internal_failure=internal_failure,
    )


async def _verify_not_found(
    provider: _CleanupProvider,
    *,
    namespace: str,
    before_provider_action: Callable[[], None],
) -> _CleanupControl:
    for attempt in range(_NOT_FOUND_ATTEMPTS):
        try:
            before_provider_action()
            await provider.namespace_metadata(namespace)
        except ProviderError as error:
            if error.details.code is ApiErrorCode.NOT_FOUND:
                return _CleanupControl(succeeded=True)
            return _CleanupControl(succeeded=False)
        except (KeyboardInterrupt, asyncio.CancelledError):
            return _CleanupControl(succeeded=False, cancelled=True)
        except BaseException:
            return _CleanupControl(succeeded=False, internal_failure=True)
        if attempt + 1 < _NOT_FOUND_ATTEMPTS:
            try:
                await asyncio.sleep(_NOT_FOUND_POLL_INTERVAL)
            except (KeyboardInterrupt, asyncio.CancelledError):
                return _CleanupControl(succeeded=False, cancelled=True)
            except BaseException:
                return _CleanupControl(succeeded=False, internal_failure=True)
    return _CleanupControl(succeeded=False)


@dataclass(frozen=True, slots=True)
class _CloseControl:
    failed: bool
    cancelled: bool = False
    internal_failure: bool = False


async def _capture_close(provider: _CleanupProvider) -> _CloseControl:
    try:
        await provider.close()
    except (KeyboardInterrupt, asyncio.CancelledError):
        return _CloseControl(failed=True, cancelled=True)
    except BaseException:
        return _CloseControl(failed=True, internal_failure=True)
    return _CloseControl(failed=False)


async def _drain_close(provider: _CleanupProvider) -> _CloseControl:
    close_coroutine = _capture_close(provider)
    try:
        close_task = asyncio.create_task(close_coroutine)
    except (KeyboardInterrupt, asyncio.CancelledError):
        close_coroutine.close()
        return _CloseControl(failed=True, cancelled=True)
    except BaseException:
        close_coroutine.close()
        return _CloseControl(failed=True, internal_failure=True)
    cancelled = False
    internal_failure = False
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            cancelled = True
        except BaseException:
            internal_failure = True
    try:
        result = close_task.result()
    except (KeyboardInterrupt, asyncio.CancelledError):
        cancelled = True
        result = _CloseControl(failed=True, cancelled=True)
    except BaseException:
        internal_failure = True
        result = _CloseControl(failed=True, internal_failure=True)
    return _CloseControl(
        failed=result.failed or cancelled or internal_failure,
        cancelled=result.cancelled or cancelled,
        internal_failure=result.internal_failure or internal_failure,
    )


def _emit_cleanup_complete(emit: Callable[[str], None]) -> None:
    emit("cleanup verified; clear PUFFERLAB_SEARCH_NAMESPACE and restart the API")


def _raise_cancelled() -> None:
    raise asyncio.CancelledError() from None


def _raise_namespace_outcome(outcome: _NamespaceOutcome) -> None:
    assert outcome.message is not None
    raise NamespaceCommandError(outcome.message, exit_code=outcome.exit_code) from None
