from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pufferlab.cli.doctor import _default_owned_tiny_target_resolver
from pufferlab.config import Settings
from pufferlab.contracts.capabilities import CapabilityRequirementCode
from pufferlab.owned_tiny import (
    OwnedTinyBusyError,
    OwnedTinyCredentialMismatchError,
    OwnedTinyState,
    OwnedTinyStateError,
    _derive_namespace,
    owned_tiny_ingest_operation,
    owned_tiny_requirements,
    resolve_owned_tiny_target,
)

_KEY = "fake-owned-tiny-key"
_REGION = "gcp-us-west1"


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path.resolve() / ".pufferlab" / "state" / "owned-tiny-v1"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())
    return state


def _settings(
    *,
    key: str = _KEY,
    region: str = _REGION,
    namespace: str | None = None,
) -> Settings:
    return Settings.model_validate(
        {
            "turbopuffer_api_key": key,
            "turbopuffer_region": region,
            "pufferlab_search_namespace": namespace,
        }
    )


def _create_receipt(state: OwnedTinyState = OwnedTinyState.INTENT):
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


def test_production_locator_uses_account_record_not_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    monkeypatch.setenv("HOME", "/attacker/home")
    monkeypatch.setenv("XDG_STATE_HOME", "/attacker/xdg")
    monkeypatch.setenv("PUFFERLAB_DATA_DIR", "/attacker/data")
    monkeypatch.setattr(
        owned_tiny.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_dir=f"/Users/account-{uid}"),
    )

    assert owned_tiny._production_state_path() == (
        Path(f"/Users/account-{os.getuid()}") / ".pufferlab/state/owned-tiny-v1"
    )
    assert owned_tiny._production_anchor_path() == Path(f"/Users/account-{os.getuid()}")


def test_intent_is_authenticated_derived_and_durable_with_fixed_modes(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    fsync_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        fsync_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(owned_tiny.os, "fsync", recording_fsync)
    snapshot = _create_receipt()
    receipt_payload = json.loads((isolated_state / "receipt.json").read_text(encoding="utf-8"))
    owner_key = (isolated_state / "owner.key").read_bytes()

    assert snapshot.receipt.state is OwnedTinyState.INTENT
    assert snapshot.receipt.namespace == _derive_namespace(
        owner_key,
        nonce=receipt_payload["nonce"],
        region=_REGION,
        credential_tag=receipt_payload["credential_tag"],
    )
    assert snapshot.receipt.namespace.startswith("pufferlab-tiny-")
    assert set(receipt_payload) == {
        "format_version",
        "purpose",
        "creating_region",
        "nonce",
        "namespace",
        "state",
        "credential_tag",
        "authentication_tag",
    }
    assert len(owner_key) == 32
    assert stat_mode(isolated_state) == 0o700
    assert stat_mode(isolated_state / "owner.key") == 0o600
    assert stat_mode(isolated_state / "receipt.json") == 0o600
    assert stat_mode(isolated_state / "operation.lock") == 0o600
    assert any(stat.S_ISDIR(mode) for mode in fsync_modes)
    assert any(stat.S_ISREG(mode) for mode in fsync_modes)
    rendered = repr(snapshot.receipt)
    assert receipt_payload["credential_tag"] not in rendered
    assert receipt_payload["authentication_tag"] not in rendered
    assert _KEY not in rendered


def test_intent_resume_keeps_exact_namespace_and_binds_credential(
    isolated_state: Path,
) -> None:
    first = _create_receipt()
    original_bytes = (isolated_state / "receipt.json").read_bytes()

    with owned_tiny_ingest_operation() as operation:
        resumed = operation.load(required=True)
        assert resumed is not None
        operation.require_credential(resumed, _KEY)
        with pytest.raises(OwnedTinyCredentialMismatchError):
            operation.require_credential(resumed, "rotated-key")

    assert resumed.receipt.namespace == first.receipt.namespace
    assert (isolated_state / "receipt.json").read_bytes() == original_bytes


def test_authenticated_cas_rejects_same_byte_replacement_without_clobbering(
    isolated_state: Path,
) -> None:
    first = _create_receipt()
    raw = (isolated_state / "receipt.json").read_bytes()
    replacement = isolated_state / "replacement"
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    os.replace(replacement, isolated_state / "receipt.json")

    with (
        owned_tiny_ingest_operation() as operation,
        pytest.raises(OwnedTinyStateError, match="transition"),
    ):
        operation.transition(first, OwnedTinyState.CREATED)

    assert (isolated_state / "receipt.json").read_bytes() == raw


@pytest.mark.parametrize("terminal", [False, True], ids=["transition", "terminal-removal"])
def test_atomic_exchange_restores_last_moment_receipt_substitute(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: bool,
) -> None:
    from pufferlab import owned_tiny

    starting_state = OwnedTinyState.NOT_FOUND_VERIFIED if terminal else OwnedTinyState.INTENT
    _create_receipt(starting_state)
    expected_path = isolated_state / "receipt.json"
    expected_raw = expected_path.read_bytes()
    expected_inode = expected_path.stat().st_ino
    preserved_expected = isolated_state / "expected-before-swap"
    real_exchange = owned_tiny._exchange_paths
    attacked = False

    def exchange_after_substitution(directory_fd: int, first: str, second: str) -> None:
        nonlocal attacked
        if not attacked:
            attacked = True
            expected_path.replace(preserved_expected)
            substitute = isolated_state / "manual-substitute"
            substitute.write_bytes(expected_raw)
            substitute.chmod(0o600)
            substitute.replace(expected_path)
        real_exchange(directory_fd, first, second)

    monkeypatch.setattr(owned_tiny, "_exchange_paths", exchange_after_substitution)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        if terminal:
            with pytest.raises(OwnedTinyStateError, match="could not be removed"):
                operation.remove_terminal(current)
        else:
            with pytest.raises(OwnedTinyStateError, match="transition"):
                operation.transition(current, OwnedTinyState.CREATED)

    assert attacked
    assert expected_path.read_bytes() == expected_raw
    assert expected_path.stat().st_ino != expected_inode
    assert preserved_expected.read_bytes() == expected_raw
    assert preserved_expected.stat().st_ino == expected_inode


def test_atomic_cas_rechecks_owner_identity_after_temporary_file_sync(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt()
    original = (isolated_state / "receipt.json").read_bytes()
    real_create_temporary = owned_tiny._create_temporary

    def replace_owner_after_temporary_open(directory_fd: int, name: str) -> int:
        fd = real_create_temporary(directory_fd, name)
        replacement = isolated_state / "replacement-owner"
        replacement.write_bytes(b"x" * 32)
        replacement.chmod(0o600)
        replacement.replace(isolated_state / "owner.key")
        return fd

    monkeypatch.setattr(owned_tiny, "_create_temporary", replace_owner_after_temporary_open)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="transition"):
            operation.transition(current, OwnedTinyState.CREATED)

    assert (isolated_state / "receipt.json").read_bytes() == original


@pytest.mark.parametrize("child", ["owner.key", "receipt.json", "operation.lock"])
def test_fixed_children_reject_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child: str,
) -> None:
    state = tmp_path.resolve() / child.replace(".", "-")
    state.mkdir(mode=0o700)
    target = tmp_path.resolve() / f"{child}.target"
    target.write_bytes(b"x" * 32)
    target.chmod(0o600)
    (state / child).symlink_to(target)
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())

    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation() as operation:
        operation.load(required=False)


def test_state_path_component_symlink_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path.resolve() / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path.resolve() / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        "pufferlab.owned_tiny._production_state_path",
        lambda: linked / "owned",
    )
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())

    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass


def test_nonblocking_process_lock_rejects_concurrent_operation(isolated_state: Path) -> None:
    del isolated_state
    with (
        owned_tiny_ingest_operation(),
        pytest.raises(OwnedTinyBusyError),
        owned_tiny_ingest_operation(),
    ):
        pass


def test_replaced_named_lock_cannot_split_coordination(isolated_state: Path) -> None:
    with owned_tiny_ingest_operation() as first_operation:
        snapshot = first_operation.create_intent(api_key=_KEY, region=_REGION)
        replacement = isolated_state / "replacement-lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o600)
        replacement.replace(isolated_state / "operation.lock")

        with pytest.raises(OwnedTinyBusyError), owned_tiny_ingest_operation():
            pass
        with pytest.raises(OwnedTinyStateError, match="changed"):
            first_operation.authenticate_current(snapshot)


def test_relocated_state_directory_cannot_split_coordination(isolated_state: Path) -> None:
    relocated = isolated_state.parent / "relocated-owned-state"
    with owned_tiny_ingest_operation() as first_operation:
        snapshot = first_operation.create_intent(api_key=_KEY, region=_REGION)
        isolated_state.replace(relocated)

        with pytest.raises(OwnedTinyBusyError), owned_tiny_ingest_operation():
            pass
        with pytest.raises(OwnedTinyStateError, match="changed"):
            first_operation.authenticate_current(snapshot)
        assert not isolated_state.exists()
        assert (relocated / "receipt.json").read_bytes() == snapshot.raw


def test_replaced_intermediate_chain_fails_even_when_final_state_inode_returns(
    isolated_state: Path,
) -> None:
    pufferlab_directory = isolated_state.parents[1]
    relocated_pufferlab = pufferlab_directory.parent / ".pufferlab-relocated"
    with owned_tiny_ingest_operation() as operation:
        snapshot = operation.create_intent(api_key=_KEY, region=_REGION)
        pufferlab_directory.replace(relocated_pufferlab)
        isolated_state.parent.mkdir(parents=True, mode=0o700)
        relocated_state = relocated_pufferlab / "state" / "owned-tiny-v1"
        relocated_state.replace(isolated_state)

        with pytest.raises(OwnedTinyStateError, match="changed"):
            operation.authenticate_current(snapshot)
        assert (isolated_state / "receipt.json").read_bytes() == snapshot.raw


@pytest.mark.parametrize("child", ["owner.key", "receipt.json", "operation.lock"])
def test_writerless_fifo_fixed_children_fail_promptly_across_local_readers(
    isolated_state: Path,
    child: str,
) -> None:
    snapshot = _create_receipt(OwnedTinyState.READY)
    child_path = isolated_state / child
    child_path.unlink()
    os.mkfifo(child_path, mode=0o600)
    child_path.chmod(0o600)
    settings = _settings(namespace=snapshot.receipt.namespace)

    descriptor_count = _open_descriptor_count()
    started = time.monotonic()
    requirements = owned_tiny_requirements(settings)
    doctor_target = _default_owned_tiny_target_resolver(settings)
    from pufferlab.cli.main import main

    exit_code = main(["namespace", "show-tiny"])
    elapsed = time.monotonic() - started

    assert requirements == (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)
    assert doctor_target is None
    assert exit_code == 2
    assert elapsed < 1.0
    assert _open_descriptor_count() == descriptor_count
    assert stat.S_ISFIFO(child_path.stat().st_mode)
    assert stat_mode(child_path) == 0o600


def test_read_only_resolver_reports_corruption_and_never_repairs(
    isolated_state: Path,
) -> None:
    snapshot = _create_receipt(OwnedTinyState.READY)
    receipt_path = isolated_state / "receipt.json"
    receipt_path.write_bytes(b'{"forged":true}')
    receipt_path.chmod(0o600)
    before = receipt_path.stat()

    requirements = owned_tiny_requirements(_settings(namespace=snapshot.receipt.namespace))

    assert requirements == (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)
    after = receipt_path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_ready_exact_target_reports_credential_then_region_mismatch_in_frozen_order(
    isolated_state: Path,
) -> None:
    del isolated_state
    snapshot = _create_receipt(OwnedTinyState.READY)
    settings = _settings(
        key="rotated-key",
        region="aws-us-east-1",
        namespace=snapshot.receipt.namespace,
    )

    assert owned_tiny_requirements(settings) == (
        CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
        CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
    )
    assert resolve_owned_tiny_target(settings) is None


def test_ready_exact_target_resolves_only_with_exact_key_region_and_namespace(
    isolated_state: Path,
) -> None:
    del isolated_state
    snapshot = _create_receipt(OwnedTinyState.READY)
    target = resolve_owned_tiny_target(_settings(namespace=snapshot.receipt.namespace))

    assert target is not None
    assert target.namespace == snapshot.receipt.namespace
    assert target.region == _REGION
    assert resolve_owned_tiny_target(_settings(namespace="pufferlab-explicit")) is None


def test_active_nonready_exact_target_fails_closed(isolated_state: Path) -> None:
    del isolated_state
    snapshot = _create_receipt(OwnedTinyState.CREATED)

    assert owned_tiny_requirements(_settings(namespace=snapshot.receipt.namespace)) == (
        CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
    )


def test_retained_owner_key_without_receipt_is_valid_absent_state(
    isolated_state: Path,
) -> None:
    snapshot = _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        operation.remove_terminal(current)

    assert (isolated_state / "owner.key").is_file()
    assert not (isolated_state / "receipt.json").exists()
    assert owned_tiny_requirements(_settings(namespace=snapshot.receipt.namespace)) == ()
    assert resolve_owned_tiny_target(_settings(namespace=snapshot.receipt.namespace)) is None


def test_atomic_transition_failure_retains_prior_authenticated_receipt(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    snapshot = _create_receipt()
    original = (isolated_state / "receipt.json").read_bytes()

    def fail_exchange(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise owned_tiny._StateFailure()

    monkeypatch.setattr(owned_tiny, "_exchange_paths", fail_exchange)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError):
            operation.transition(current, OwnedTinyState.CREATED)

    assert snapshot.receipt.state is OwnedTinyState.INTENT
    assert (isolated_state / "receipt.json").read_bytes() == original


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _open_descriptor_count() -> int:
    descriptor_directory = Path("/proc/self/fd")
    if not descriptor_directory.is_dir():
        descriptor_directory = Path("/dev/fd")
    return len(tuple(descriptor_directory.iterdir()))
