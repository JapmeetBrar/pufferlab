from __future__ import annotations

import asyncio
import io
import json
import os
import traceback
from pathlib import Path

import pytest
from pufferlab.cli.main import main
from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.owned_tiny import (
    OwnedTinyState,
    owned_tiny_existing_operation,
    owned_tiny_ingest_operation,
)
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.providers.types import (
    ProviderDeleteResult,
    ProviderNamespaceMetadata,
)

_KEY = "fake-cleanup-key"
_REGION = "aws-us-east-1"


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path.resolve() / ".pufferlab" / "state" / "owned-tiny-v1"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())
    monkeypatch.setattr("pufferlab.cli.namespace._NOT_FOUND_ATTEMPTS", 3)
    monkeypatch.setattr("pufferlab.cli.namespace._NOT_FOUND_POLL_INTERVAL", 0)
    return state


def _settings(*, key: str | None = _KEY, region: str = "unrelated-current-region") -> Settings:
    return Settings.model_validate(
        {
            "turbopuffer_api_key": key,
            "turbopuffer_region": region,
        }
    )


def _create_receipt(state: OwnedTinyState = OwnedTinyState.READY):
    with owned_tiny_ingest_operation() as operation:
        snapshot = operation.create_intent(api_key=_KEY, region=_REGION)
        if state in {OwnedTinyState.CREATED, OwnedTinyState.READY}:
            snapshot = operation.transition(snapshot, OwnedTinyState.CREATED)
        if state is OwnedTinyState.READY:
            snapshot = operation.transition(snapshot, OwnedTinyState.READY)
        if state in {OwnedTinyState.CLEANUP_REQUESTED, OwnedTinyState.NOT_FOUND_VERIFIED}:
            snapshot = operation.transition(snapshot, OwnedTinyState.CLEANUP_REQUESTED)
        if state is OwnedTinyState.NOT_FOUND_VERIFIED:
            snapshot = operation.transition(snapshot, OwnedTinyState.NOT_FOUND_VERIFIED)
        return snapshot


def _provider_error(code: ApiErrorCode) -> ProviderError:
    return ProviderError(
        "safe fake provider failure",
        ProviderErrorDetails(
            code=code,
            retryable=False,
            operation="fake",
            status_code=404 if code is ApiErrorCode.NOT_FOUND else 503,
        ),
    )


class FakeCleanupProvider:
    def __init__(self) -> None:
        self.delete_calls: list[str] = []
        self.metadata_calls: list[str] = []
        self.close_calls = 0
        self.delete_error: BaseException | None = None
        self.metadata_outcomes: list[str] = ["not_found"]
        self.metadata_error: BaseException | None = None
        self.close_error: BaseException | None = None

    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
        self.delete_calls.append(namespace)
        if self.delete_error is not None:
            raise self.delete_error
        return ProviderDeleteResult(client_duration_ms=1.0)

    async def namespace_metadata(self, namespace: str) -> ProviderNamespaceMetadata:
        self.metadata_calls.append(namespace)
        if self.metadata_error is not None:
            raise self.metadata_error
        outcome = self.metadata_outcomes.pop(0) if self.metadata_outcomes else "present"
        if outcome == "not_found":
            raise _provider_error(ApiErrorCode.NOT_FOUND)
        if outcome == "error":
            raise _provider_error(ApiErrorCode.PROVIDER_ERROR)
        return ProviderNamespaceMetadata(
            approx_row_count=0,
            index_status="up-to-date",
            unindexed_bytes=0,
            schema={},
            client_duration_ms=1.0,
        )

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeProviderFactory:
    def __init__(self, provider: FakeCleanupProvider) -> None:
        self.provider = provider
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, api_key: str, region: str) -> FakeCleanupProvider:
        self.calls.append((api_key, region))
        return self.provider


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: FakeCleanupProvider,
) -> FakeProviderFactory:
    factory = FakeProviderFactory(provider)
    monkeypatch.setattr("pufferlab.cli.namespace._PROVIDER_FACTORY", factory)
    return factory


def _receipt_state() -> OwnedTinyState:
    with owned_tiny_existing_operation() as operation:
        snapshot = operation.load(required=True)
        assert snapshot is not None
        return snapshot.receipt.state


def test_namespace_help_exposes_no_target_path_or_token_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["namespace", "cleanup-tiny", "--help"])

    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "zero-row queries" in help_text
    assert "No target, path, token, or ownership input is accepted" in help_text
    assert "--namespace" not in help_text


@pytest.mark.parametrize("command", ["show-tiny", "cleanup-tiny"])
def test_missing_receipt_commands_do_not_create_state(
    isolated_state: Path,
    command: str,
) -> None:
    assert (
        main(
            ["namespace", command],
            settings_factory=_settings,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 2
    )
    assert not isolated_state.exists()


def test_show_prints_only_exact_authenticated_assignments(isolated_state: Path) -> None:
    del isolated_state
    snapshot = _create_receipt()
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["namespace", "show-tiny"],
        settings_factory=lambda: (_ for _ in ()).throw(AssertionError("settings accessed")),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        f"TURBOPUFFER_REGION={_REGION}",
        f"PUFFERLAB_SEARCH_NAMESPACE={snapshot.receipt.namespace}",
    ]
    assert stderr.getvalue() == ""


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("boundary", ["anchor", "state"])
def test_show_process_control_at_directory_open_is_fixed_and_leak_free(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
    boundary: str,
) -> None:
    from pufferlab import owned_tiny

    snapshot = _create_receipt()
    target = isolated_state.parents[2] if boundary == "anchor" else isolated_state
    target_inode = target.stat().st_ino
    real_clear_nonblocking = owned_tiny._clear_nonblocking
    marker = f"private-show-{boundary}-{control.__name__}-marker"
    attacked = False

    def interrupt_directory_open(fd: int) -> None:
        nonlocal attacked
        if not attacked and os.fstat(fd).st_ino == target_inode:
            attacked = True
            raise control(marker)
        real_clear_nonblocking(fd)

    monkeypatch.setattr(owned_tiny, "_clear_nonblocking", interrupt_directory_open)
    stdout = io.StringIO()
    stderr = io.StringIO()
    descriptor_count = _open_descriptor_count()

    exit_code = main(
        ["namespace", "show-tiny"],
        stdout=stdout,
        stderr=stderr,
    )

    assert attacked
    assert exit_code == (130 if control is KeyboardInterrupt else 1)
    assert stdout.getvalue() == ""
    assert marker not in stderr.getvalue()
    assert snapshot.receipt.namespace not in stderr.getvalue()
    assert _open_descriptor_count() == descriptor_count


def test_cleanup_uses_receipt_region_exact_target_and_retains_owner_key(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_receipt()
    provider = FakeCleanupProvider()
    factory = _install_provider(monkeypatch, provider)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["namespace", "cleanup-tiny"],
        settings_factory=_settings,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert factory.calls == [(_KEY, _REGION)]
    assert provider.delete_calls == [snapshot.receipt.namespace]
    assert provider.metadata_calls == [snapshot.receipt.namespace]
    assert provider.close_calls == 1
    assert not (isolated_state / "receipt.json").exists()
    assert (isolated_state / "owner.key").is_file()
    assert (isolated_state / "operation.lock").is_file()
    assert stdout.getvalue() == (
        "cleanup verified; clear PUFFERLAB_SEARCH_NAMESPACE and restart the API\n"
    )
    assert snapshot.receipt.namespace not in stdout.getvalue()
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    "starting_state",
    [
        OwnedTinyState.INTENT,
        OwnedTinyState.CREATED,
        OwnedTinyState.READY,
        OwnedTinyState.CLEANUP_REQUESTED,
    ],
)
def test_cleanup_reconciles_every_active_state(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    starting_state: OwnedTinyState,
) -> None:
    snapshot = _create_receipt(starting_state)
    provider = FakeCleanupProvider()
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 0
    assert provider.delete_calls == [snapshot.receipt.namespace]
    assert not (isolated_state / "receipt.json").exists()


def test_cleanup_rerun_after_crash_immediately_after_cleanup_request(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    snapshot = _create_receipt()
    provider = FakeCleanupProvider()
    factory = _install_provider(monkeypatch, provider)
    real_transition = owned_tiny._OwnedTinyOperation.transition
    interrupted = False

    def interrupt_after_cleanup_request(
        operation: owned_tiny._OwnedTinyOperation,
        current: object,
        state: OwnedTinyState,
    ) -> object:
        nonlocal interrupted
        transitioned = real_transition(operation, current, state)  # type: ignore[arg-type]
        if state is OwnedTinyState.CLEANUP_REQUESTED and not interrupted:
            interrupted = True
            raise SystemExit("private-after-cleanup-request-marker")
        return transitioned

    monkeypatch.setattr(
        owned_tiny._OwnedTinyOperation,
        "transition",
        interrupt_after_cleanup_request,
    )

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert interrupted
    assert factory.calls == []
    assert provider.delete_calls == []
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 0
    assert factory.calls == [(_KEY, _REGION)]
    assert provider.delete_calls == [snapshot.receipt.namespace]
    assert provider.metadata_calls == [snapshot.receipt.namespace]
    assert provider.close_calls == 1
    assert not (isolated_state / "receipt.json").exists()


def test_cleanup_rerun_after_crash_between_delete_and_verification(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_receipt()
    first_provider = FakeCleanupProvider()
    first_factory = _install_provider(monkeypatch, first_provider)

    async def interrupt_before_verification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SystemExit("private-after-delete-marker")

    with monkeypatch.context() as crash:
        crash.setattr("pufferlab.cli.namespace._verify_not_found", interrupt_before_verification)
        assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1

    assert first_factory.calls == [(_KEY, _REGION)]
    assert first_provider.delete_calls == [snapshot.receipt.namespace]
    assert first_provider.metadata_calls == []
    assert first_provider.close_calls == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED

    second_provider = FakeCleanupProvider()
    second_provider.delete_error = _provider_error(ApiErrorCode.NOT_FOUND)
    second_factory = _install_provider(monkeypatch, second_provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 0
    assert second_factory.calls == [(_KEY, _REGION)]
    assert second_provider.delete_calls == [snapshot.receipt.namespace]
    assert second_provider.metadata_calls == [snapshot.receipt.namespace]
    assert second_provider.close_calls == 1
    assert not (isolated_state / "receipt.json").exists()


def test_cleanup_rerun_after_crash_during_not_found_polling(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_receipt()
    first_provider = FakeCleanupProvider()
    first_provider.metadata_outcomes = ["present"]
    first_factory = _install_provider(monkeypatch, first_provider)

    async def interrupt_polling(delay: float) -> None:
        del delay
        raise SystemExit("private-not-found-poll-marker")

    with monkeypatch.context() as crash:
        crash.setattr("pufferlab.cli.namespace.asyncio.sleep", interrupt_polling)
        assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1

    assert first_factory.calls == [(_KEY, _REGION)]
    assert first_provider.delete_calls == [snapshot.receipt.namespace]
    assert first_provider.metadata_calls == [snapshot.receipt.namespace]
    assert first_provider.close_calls == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED

    second_provider = FakeCleanupProvider()
    second_provider.delete_error = _provider_error(ApiErrorCode.NOT_FOUND)
    second_factory = _install_provider(monkeypatch, second_provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 0
    assert second_factory.calls == [(_KEY, _REGION)]
    assert second_provider.delete_calls == [snapshot.receipt.namespace]
    assert second_provider.metadata_calls == [snapshot.receipt.namespace]
    assert second_provider.close_calls == 1
    assert not (isolated_state / "receipt.json").exists()


def test_terminal_cleanup_rerun_after_crash_uses_no_provider(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    snapshot = _create_receipt()
    provider = FakeCleanupProvider()
    factory = _install_provider(monkeypatch, provider)
    real_remove_terminal = owned_tiny._OwnedTinyOperation.remove_terminal
    interrupted = False

    def interrupt_once_before_terminal_removal(
        operation: owned_tiny._OwnedTinyOperation,
        current: object,
    ) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SystemExit("private-before-terminal-removal-marker")
        real_remove_terminal(operation, current)  # type: ignore[arg-type]

    monkeypatch.setattr(
        owned_tiny._OwnedTinyOperation,
        "remove_terminal",
        interrupt_once_before_terminal_removal,
    )

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert interrupted
    assert factory.calls == [(_KEY, _REGION)]
    assert provider.delete_calls == [snapshot.receipt.namespace]
    assert provider.metadata_calls == [snapshot.receipt.namespace]
    assert provider.close_calls == 1
    assert _receipt_state() is OwnedTinyState.NOT_FOUND_VERIFIED

    provider_calls = (
        tuple(factory.calls),
        tuple(provider.delete_calls),
        tuple(provider.metadata_calls),
        provider.close_calls,
    )
    assert main(["namespace", "cleanup-tiny"], settings_factory=lambda: _settings(key=None)) == 0
    assert (
        tuple(factory.calls),
        tuple(provider.delete_calls),
        tuple(provider.metadata_calls),
        provider.close_calls,
    ) == provider_calls
    assert not (isolated_state / "receipt.json").exists()


def test_terminal_rerun_unlinks_without_key_or_provider_call(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)

    def fail_provider(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("terminal cleanup constructed a provider")

    monkeypatch.setattr("pufferlab.cli.namespace._PROVIDER_FACTORY", fail_provider)

    assert (
        main(
            ["namespace", "cleanup-tiny"],
            settings_factory=lambda: _settings(key=None),
        )
        == 0
    )
    assert not (isolated_state / "receipt.json").exists()


def test_rotated_key_fails_before_provider_and_retains_ready_receipt(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_receipt()
    provider = FakeCleanupProvider()
    factory = _install_provider(monkeypatch, provider)
    stderr = io.StringIO()

    exit_code = main(
        ["namespace", "cleanup-tiny"],
        settings_factory=lambda: _settings(key="rotated-key"),
        stderr=stderr,
    )

    assert exit_code == 2
    assert factory.calls == []
    assert _receipt_state() is OwnedTinyState.READY
    assert (isolated_state / "receipt.json").is_file()
    assert "rotated-key" not in stderr.getvalue()


def test_rotated_key_is_absent_from_retained_namespace_traceback_locals(
    isolated_state: Path,
) -> None:
    from pufferlab.cli.namespace import NamespaceCommandError, cleanup_owned_tiny

    del isolated_state
    _create_receipt()
    secret = "rotated-key-retained-frame-marker"

    with pytest.raises(NamespaceCommandError) as caught:
        asyncio.run(
            cleanup_owned_tiny(
                _settings(key=secret),
                emit=lambda message: None,
            )
        )

    traceback_value = caught.value.__traceback__
    production_locals: list[str] = []
    while traceback_value is not None:
        if traceback_value.tb_frame.f_code.co_filename.endswith("/pufferlab/cli/namespace.py"):
            production_locals.append(repr(traceback_value.tb_frame.f_locals))
        traceback_value = traceback_value.tb_next
    assert secret not in "".join(production_locals)


def test_provider_factory_failure_is_detached_and_retains_cleanup_request(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_state
    marker = "private-provider-factory-marker"
    _create_receipt()

    def fail_provider(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(marker)

    monkeypatch.setattr("pufferlab.cli.namespace._PROVIDER_FACTORY", fail_provider)
    stderr = io.StringIO()

    exit_code = main(
        ["namespace", "cleanup-tiny"],
        settings_factory=_settings,
        stderr=stderr,
    )

    assert exit_code == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED
    assert marker not in stderr.getvalue()


@pytest.mark.parametrize("failure", ["delete", "verify", "close"])
def test_remote_or_close_failure_retains_cleanup_requested_receipt(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    del isolated_state
    _create_receipt()
    provider = FakeCleanupProvider()
    if failure == "delete":
        provider.delete_error = _provider_error(ApiErrorCode.PROVIDER_ERROR)
    elif failure == "verify":
        provider.metadata_outcomes = ["error"]
    else:
        provider.close_error = RuntimeError("private-close-marker")
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED
    assert provider.close_calls == 1


def test_bounded_present_verification_retains_receipt_after_exact_attempt_count(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_state
    _create_receipt()
    provider = FakeCleanupProvider()
    provider.metadata_outcomes = ["present", "present", "present"]
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert len(provider.metadata_calls) == 3
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED


def test_delete_not_found_still_performs_independent_not_found_verification(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_receipt()
    provider = FakeCleanupProvider()
    provider.delete_error = _provider_error(ApiErrorCode.NOT_FOUND)
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 0
    assert provider.delete_calls == [snapshot.receipt.namespace]
    assert provider.metadata_calls == [snapshot.receipt.namespace]
    assert not (isolated_state / "receipt.json").exists()


@pytest.mark.parametrize("attack", ["replace-lock", "relocate-state"])
def test_cleanup_rechecks_anchor_continuity_before_next_provider_action(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _create_receipt()

    class AttackingProvider(FakeCleanupProvider):
        async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
            result = await super().delete_namespace(namespace)
            if attack == "replace-lock":
                replacement = isolated_state / "replacement-lock"
                replacement.write_bytes(b"")
                replacement.chmod(0o600)
                replacement.replace(isolated_state / "operation.lock")
            else:
                isolated_state.replace(isolated_state.parent / "relocated-state")
            return result

    provider = AttackingProvider()
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert len(provider.delete_calls) == 1
    assert provider.metadata_calls == []
    assert provider.close_calls == 1
    receipt_root = (
        isolated_state if attack == "replace-lock" else isolated_state.parent / "relocated-state"
    )
    receipt = json.loads((receipt_root / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == OwnedTinyState.CLEANUP_REQUESTED.value


def test_cleanup_rejects_account_home_anchor_swap_before_next_provider_action(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_receipt()
    anchor = isolated_state.parents[2]
    relocated_anchor = anchor.parent / f"{anchor.name}-relocated"

    class AnchorSwappingProvider(FakeCleanupProvider):
        async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
            result = await super().delete_namespace(namespace)
            anchor.replace(relocated_anchor)
            anchor.mkdir(mode=0o700)
            anchor.chmod(0o700)
            return result

    provider = AnchorSwappingProvider()
    _install_provider(monkeypatch, provider)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert len(provider.delete_calls) == 1
    assert provider.metadata_calls == []
    assert provider.close_calls == 1
    assert list(anchor.iterdir()) == []

    anchor.rmdir()
    relocated_anchor.replace(anchor)
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED


def test_cancellation_closes_provider_and_retains_cleanup_requested(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_state
    _create_receipt()
    provider = FakeCleanupProvider()
    provider.delete_error = asyncio.CancelledError()
    _install_provider(monkeypatch, provider)
    stderr = io.StringIO()

    assert (
        main(
            ["namespace", "cleanup-tiny"],
            settings_factory=_settings,
            stderr=stderr,
        )
        == 130
    )
    assert provider.close_calls == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED
    assert stderr.getvalue() == "error: namespace command cancelled\n"


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "phase",
    ["factory", "handoff", "continuity", "delete", "metadata", "sleep", "close"],
)
def test_process_control_drains_cleanup_provider_once_and_returns_fixed_output(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
    phase: str,
) -> None:
    del isolated_state
    _create_receipt()
    provider = FakeCleanupProvider()
    marker = f"private-{phase}-{control.__name__}-marker"

    if phase == "factory":

        def controlled_factory(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise control(marker)

        monkeypatch.setattr("pufferlab.cli.namespace._PROVIDER_FACTORY", controlled_factory)
    else:
        factory = _install_provider(monkeypatch, provider)
        if phase == "handoff":

            async def controlled_handoff(*args: object, **kwargs: object) -> None:
                del args, kwargs
                raise control(marker)

            monkeypatch.setattr("pufferlab.cli.namespace._delete_and_verify", controlled_handoff)
        elif phase == "continuity":
            from pufferlab import owned_tiny

            authenticate_current = owned_tiny._OwnedTinyOperation.authenticate_current

            def controlled_continuity(
                operation: owned_tiny._OwnedTinyOperation,
                snapshot: object,
            ) -> None:
                if factory.calls:
                    raise control(marker)
                authenticate_current(operation, snapshot)  # type: ignore[arg-type]

            monkeypatch.setattr(
                owned_tiny._OwnedTinyOperation,
                "authenticate_current",
                controlled_continuity,
            )
        elif phase == "delete":
            provider.delete_error = control(marker)
        elif phase == "metadata":
            provider.metadata_error = control(marker)
        elif phase == "sleep":
            provider.metadata_outcomes = ["present"]

            async def controlled_sleep(delay: float) -> None:
                del delay
                raise control(marker)

            monkeypatch.setattr("pufferlab.cli.namespace.asyncio.sleep", controlled_sleep)
        else:
            provider.close_error = control(marker)

    stderr = io.StringIO()
    exit_code = main(
        ["namespace", "cleanup-tiny"],
        settings_factory=_settings,
        stderr=stderr,
    )

    assert exit_code == (130 if control is KeyboardInterrupt else 1)
    assert provider.close_calls == (0 if phase == "factory" else 1)
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED
    assert marker not in stderr.getvalue()
    assert _KEY not in stderr.getvalue()


@pytest.mark.parametrize(
    ("operation_control", "close_control", "expected_exit"),
    [
        (KeyboardInterrupt, SystemExit, 130),
        (SystemExit, KeyboardInterrupt, 1),
    ],
)
def test_cleanup_operation_control_precedes_close_control(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_control: type[BaseException],
    close_control: type[BaseException],
    expected_exit: int,
) -> None:
    del isolated_state
    _create_receipt()
    marker = "private-cleanup-control-precedence-marker"
    provider = FakeCleanupProvider()
    provider.close_error = close_control(marker)
    _install_provider(monkeypatch, provider)

    async def controlled_handoff(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise operation_control(marker)

    monkeypatch.setattr("pufferlab.cli.namespace._delete_and_verify", controlled_handoff)
    stderr = io.StringIO()

    assert (
        main(
            ["namespace", "cleanup-tiny"],
            settings_factory=_settings,
            stderr=stderr,
        )
        == expected_exit
    )
    assert provider.close_calls == 1
    assert _receipt_state() is OwnedTinyState.CLEANUP_REQUESTED
    assert marker not in stderr.getvalue()
    assert _KEY not in stderr.getvalue()


def test_cleanup_process_control_error_trace_retains_no_key_target_provider_or_marker(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab.cli.namespace import NamespaceCommandError, cleanup_owned_tiny

    del isolated_state
    snapshot = _create_receipt()
    marker = "private-cleanup-control-exception-marker"

    class MarkedProvider(FakeCleanupProvider):
        def __repr__(self) -> str:
            return marker

    provider = MarkedProvider()
    provider.delete_error = SystemExit(marker)
    _install_provider(monkeypatch, provider)

    with pytest.raises(NamespaceCommandError) as caught:
        asyncio.run(cleanup_owned_tiny(_settings(), emit=lambda message: None))

    production_locals: list[str] = []
    traceback_value = caught.value.__traceback__
    while traceback_value is not None:
        if traceback_value.tb_frame.f_code.co_filename.endswith("/pufferlab/cli/namespace.py"):
            production_locals.append(repr(traceback_value.tb_frame.f_locals))
        traceback_value = traceback_value.tb_next
    rendered = "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in rendered
    assert _KEY not in rendered
    assert snapshot.receipt.namespace not in rendered
    assert provider.close_calls == 1


def test_terminal_fixed_move_failure_retains_authenticated_terminal_receipt(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt()
    provider = FakeCleanupProvider()
    _install_provider(monkeypatch, provider)
    real_rename_noreplace = owned_tiny._rename_noreplace

    def fail_receipt_move(directory_fd: int, source: str, destination: str) -> None:
        if source == "receipt.json":
            raise owned_tiny._StateFailure()
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", fail_receipt_move)

    assert main(["namespace", "cleanup-tiny"], settings_factory=_settings) == 1
    assert (isolated_state / "receipt.json").is_file()
    assert _receipt_state() is OwnedTinyState.NOT_FOUND_VERIFIED


def test_cleanup_command_rejects_every_caller_target() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["namespace", "cleanup-tiny", "pufferlab-foreign"])

    assert caught.value.code == 2


def test_show_corruption_output_contains_no_receipt_path_or_payload(
    isolated_state: Path,
) -> None:
    snapshot = _create_receipt()
    marker = "private-receipt-marker"
    receipt_path = isolated_state / "receipt.json"
    receipt_path.write_text(marker, encoding="utf-8")
    receipt_path.chmod(0o600)
    stderr = io.StringIO()

    exit_code = main(["namespace", "show-tiny"], stderr=stderr)

    assert exit_code == 2
    rendered = stderr.getvalue()
    assert marker not in rendered
    assert str(isolated_state) not in rendered
    assert snapshot.receipt.namespace not in rendered


def test_namespace_error_trace_does_not_retain_provider_factory_marker(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab.cli.namespace import NamespaceCommandError, cleanup_owned_tiny

    del isolated_state
    marker = "provider-exception-graph-marker"
    _create_receipt()

    def fail_provider(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(marker)

    monkeypatch.setattr("pufferlab.cli.namespace._PROVIDER_FACTORY", fail_provider)

    with pytest.raises(NamespaceCommandError) as caught:
        asyncio.run(cleanup_owned_tiny(_settings(), emit=lambda message: None))

    rendered = "".join(traceback.format_exception(caught.value, chain=True))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in rendered


def _open_descriptor_count() -> int:
    descriptor_directory = Path("/proc/self/fd")
    if not descriptor_directory.is_dir():
        descriptor_directory = Path("/dev/fd")
    return len(tuple(descriptor_directory.iterdir()))
