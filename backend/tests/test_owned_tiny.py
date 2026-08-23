from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    state = tmp_path.resolve() / "owned-tiny-state"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
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

    def fail_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(owned_tiny.os, "replace", fail_replace)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError):
            operation.transition(current, OwnedTinyState.CREATED)

    assert snapshot.receipt.state is OwnedTinyState.INTENT
    assert (isolated_state / "receipt.json").read_bytes() == original


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
