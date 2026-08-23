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
    _FileIdentity,
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


@pytest.mark.parametrize(
    ("mode", "accepted"),
    [(0o700, True), (0o750, True), (0o770, False), (0o707, False)],
)
def test_account_home_anchor_rejects_group_or_world_write_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    accepted: bool,
) -> None:
    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: anchor)
    anchor.chmod(mode)
    try:
        if accepted:
            snapshot = _create_receipt()
            assert snapshot.receipt.state is OwnedTinyState.INTENT
        else:
            with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
                pass
            assert not (anchor / ".pufferlab").exists()
    finally:
        anchor.chmod(0o700)


@pytest.mark.parametrize("component_index", range(3))
def test_existing_nonprivate_application_component_is_rejected_before_authority_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component_index: int,
) -> None:
    anchor = tmp_path.resolve()
    components = [anchor / ".pufferlab"]
    components.append(components[-1] / "state")
    components.append(components[-1] / "owned-tiny-v1")
    for component in components:
        component.mkdir(mode=0o700)
        component.chmod(0o700)
    components[component_index].chmod(0o777)
    state = components[-1]
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: anchor)

    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert not (state / "owner.key").exists()
    assert not (state / "receipt.json").exists()


def test_foreign_owned_application_component_is_rejected_before_authority_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    for component in (anchor / ".pufferlab", anchor / ".pufferlab/state", state):
        component.mkdir(mode=0o700)
        component.chmod(0o700)
    foreign_inode = (anchor / ".pufferlab").stat().st_ino
    real_fstat = owned_tiny.os.fstat

    def report_foreign_owner(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if info.st_ino != foreign_inode:
            return info
        fields = list(info)
        fields[4] = os.geteuid() + 1
        return os.stat_result(fields)

    monkeypatch.setattr(owned_tiny.os, "fstat", report_foreign_owner)
    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: anchor)

    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert not (state / "owner.key").exists()
    assert not (state / "receipt.json").exists()


def test_foreign_owned_account_home_anchor_is_rejected_before_application_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    anchor_inode = anchor.stat().st_ino
    real_fstat = owned_tiny.os.fstat

    def report_foreign_anchor(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if info.st_ino != anchor_inode:
            return info
        fields = list(info)
        fields[4] = os.geteuid() + 1
        return os.stat_result(fields)

    monkeypatch.setattr(owned_tiny.os, "fstat", report_foreign_anchor)
    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: anchor)

    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert not (anchor / ".pufferlab").exists()


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


def test_private_directory_install_rejects_post_publish_fixed_substitute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    state = tmp_path.resolve() / ".pufferlab" / "state" / "owned-tiny-v1"
    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: tmp_path.resolve())
    real_rename_noreplace = owned_tiny._rename_noreplace
    preserved = tmp_path / "preserved-staged-pufferlab"
    attacked = False

    def replace_after_publish(directory_fd: int, source: str, destination: str) -> None:
        nonlocal attacked
        real_rename_noreplace(directory_fd, source, destination)
        if destination == ".pufferlab" and not attacked:
            attacked = True
            (tmp_path / ".pufferlab").replace(preserved)
            (tmp_path / ".pufferlab").mkdir(mode=0o755)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", replace_after_publish)
    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert stat_mode(tmp_path / ".pufferlab") == 0o755
    assert list((tmp_path / ".pufferlab").iterdir()) == []
    assert stat_mode(preserved) == 0o700
    assert list(preserved.iterdir()) == []


def test_private_directory_install_rejects_staging_source_substitute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    state = tmp_path.resolve() / ".pufferlab" / "state" / "owned-tiny-v1"
    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: tmp_path.resolve())
    real_rename_noreplace = owned_tiny._rename_noreplace
    preserved = tmp_path / "preserved-private-staging"
    attacked = False

    def replace_staging_before_publish(
        directory_fd: int,
        source: str,
        destination: str,
    ) -> None:
        nonlocal attacked
        if destination == ".pufferlab" and not attacked:
            attacked = True
            (tmp_path / source).replace(preserved)
            (tmp_path / source).mkdir(mode=0o700)
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", replace_staging_before_publish)
    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert stat_mode(tmp_path / ".pufferlab") == 0o700
    assert list((tmp_path / ".pufferlab").iterdir()) == []
    assert stat_mode(preserved) == 0o700
    assert list(preserved.iterdir()) == []


def test_owner_key_no_replace_collision_is_preserved(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    real_rename_noreplace = owned_tiny._rename_noreplace
    collision = b"c" * 32
    attacked = False

    def collide_before_owner_publish(directory_fd: int, source: str, destination: str) -> None:
        nonlocal attacked
        if destination == "owner.key" and not attacked:
            attacked = True
            owner_path = isolated_state / "owner.key"
            owner_path.write_bytes(collision)
            owner_path.chmod(0o600)
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", collide_before_owner_publish)
    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert (isolated_state / "owner.key").read_bytes() == collision


def test_owner_key_post_publish_substitute_is_not_removed(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    real_matches = owned_tiny._named_regular_file_matches
    preserved = isolated_state / "preserved-created-owner"
    substitute = b"s" * 32
    attacked = False

    def substitute_after_owner_verification(
        directory_fd: int,
        name: str,
        *,
        identity: _FileIdentity,
        raw: bytes,
    ) -> bool:
        nonlocal attacked
        matches = real_matches(directory_fd, name, identity=identity, raw=raw)
        if name == "owner.key" and matches and not attacked:
            attacked = True
            (isolated_state / "owner.key").replace(preserved)
            (isolated_state / "owner.key").write_bytes(substitute)
            (isolated_state / "owner.key").chmod(0o600)
        return matches

    monkeypatch.setattr(
        owned_tiny,
        "_named_regular_file_matches",
        substitute_after_owner_verification,
    )
    with pytest.raises(OwnedTinyStateError), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert (isolated_state / "owner.key").read_bytes() == substitute
    assert len(preserved.read_bytes()) == 32


@pytest.mark.parametrize("stage", ["write", "file-fsync", "close", "directory-fsync"])
def test_owner_key_staging_failures_never_publish_partial_fixed_key_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    from pufferlab import owned_tiny

    state = tmp_path / "owner-key-staging"
    state.mkdir(mode=0o700)
    directory_fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
    real_write_all = owned_tiny._write_all
    real_fsync = owned_tiny.os.fsync
    real_close = owned_tiny._close_quietly

    def fail_key_write(fd: int, value: bytes) -> None:
        if stage == "write" and len(value) == 32:
            raise owned_tiny._StateFailure()
        real_write_all(fd, value)

    def fail_selected_fsync(fd: int) -> None:
        info = os.fstat(fd)
        if stage == "file-fsync" and stat.S_ISREG(info.st_mode) and info.st_size == 32:
            raise OSError("synthetic owner file fsync failure")
        if stage == "directory-fsync" and stat.S_ISDIR(info.st_mode):
            raise OSError("synthetic owner directory fsync failure")
        real_fsync(fd)

    def fail_key_close(fd: int) -> bool:
        info = os.fstat(fd)
        closed = real_close(fd)
        if stage == "close" and stat.S_ISREG(info.st_mode) and info.st_size == 32:
            return False
        return closed

    try:
        with monkeypatch.context() as attack:
            attack.setattr(owned_tiny, "_write_all", fail_key_write)
            attack.setattr(owned_tiny.os, "fsync", fail_selected_fsync)
            attack.setattr(owned_tiny, "_close_quietly", fail_key_close)
            with pytest.raises(owned_tiny._StateFailure):
                owned_tiny._create_owner_key(directory_fd)

        fixed = state / "owner.key"
        if stage == "directory-fsync":
            resumed = owned_tiny._read_owner_key(directory_fd, required=True)
            assert resumed is not None
            assert len(resumed.key) == 32
        else:
            assert not fixed.exists()
            resumed = owned_tiny._create_owner_key(directory_fd)
            assert len(resumed.key) == 32
        assert fixed.is_file()
        assert stat_mode(fixed) == 0o600
    finally:
        os.close(directory_fd)


def test_initial_receipt_post_publish_same_byte_substitute_fails_closed(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    real_matches = owned_tiny._named_regular_file_matches
    preserved = isolated_state / "preserved-created-receipt"
    substitute_inode: int | None = None
    attacked = False

    def substitute_after_receipt_verification(
        directory_fd: int,
        name: str,
        *,
        identity: _FileIdentity,
        raw: bytes,
    ) -> bool:
        nonlocal attacked, substitute_inode
        matches = real_matches(directory_fd, name, identity=identity, raw=raw)
        if name == "receipt.json" and matches and not attacked:
            attacked = True
            (isolated_state / "receipt.json").replace(preserved)
            substitute = isolated_state / "same-byte-created-receipt-substitute"
            substitute.write_bytes(raw)
            substitute.chmod(0o600)
            substitute_inode = substitute.stat().st_ino
            substitute.replace(isolated_state / "receipt.json")
        return matches

    monkeypatch.setattr(
        owned_tiny,
        "_named_regular_file_matches",
        substitute_after_receipt_verification,
    )
    with (
        owned_tiny_ingest_operation() as operation,
        pytest.raises(OwnedTinyStateError, match="could not be persisted"),
    ):
        operation.create_intent(api_key=_KEY, region=_REGION)

    assert attacked
    assert substitute_inode is not None
    assert (isolated_state / "receipt.json").stat().st_ino == substitute_inode
    assert preserved.read_bytes() == (isolated_state / "receipt.json").read_bytes()


def test_initial_receipt_no_replace_collision_is_preserved(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    real_rename_noreplace = owned_tiny._rename_noreplace
    collision_inode: int | None = None
    attacked = False

    def collide_before_receipt_publish(
        directory_fd: int,
        source: str,
        destination: str,
    ) -> None:
        nonlocal attacked, collision_inode
        if destination == "receipt.json" and not attacked:
            attacked = True
            staged_raw = (isolated_state / source).read_bytes()
            collision = isolated_state / "same-byte-receipt-collider"
            collision.write_bytes(staged_raw)
            collision.chmod(0o600)
            collision_inode = collision.stat().st_ino
            collision.replace(isolated_state / "receipt.json")
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", collide_before_receipt_publish)
    with (
        owned_tiny_ingest_operation() as operation,
        pytest.raises(OwnedTinyStateError, match="could not be persisted"),
    ):
        operation.create_intent(api_key=_KEY, region=_REGION)

    assert attacked
    assert collision_inode is not None
    assert (isolated_state / "receipt.json").stat().st_ino == collision_inode


def test_forced_random_temporary_collision_is_never_deleted(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    current = _create_receipt()
    collision = isolated_state / ".receipt-forced-collision.tmp"
    collision.write_bytes(b"random-collider")
    collision.chmod(0o600)
    monkeypatch.setattr(owned_tiny, "_temporary_name", lambda: collision.name)

    with owned_tiny_ingest_operation() as operation:
        loaded = operation.load(required=True)
        assert loaded is not None
        with pytest.raises(OwnedTinyStateError, match="transition"):
            operation.transition(loaded, OwnedTinyState.CREATED)

    assert collision.read_bytes() == b"random-collider"
    assert (isolated_state / "receipt.json").read_bytes() == current.raw


def test_transition_restore_preserves_fixed_occupant_and_both_staged_receipts(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    prior = _create_receipt()
    fixed_occupant = b"manual-fixed-occupant"
    real_receipt_matches = owned_tiny._named_receipt_matches
    real_regular_matches = owned_tiny._named_regular_file_matches
    forced_mismatch = False
    inserted_occupant = False

    def reject_displaced_once(*args: object, **kwargs: object) -> bool:
        nonlocal forced_mismatch
        matches = real_receipt_matches(*args, **kwargs)
        if matches and not forced_mismatch:
            forced_mismatch = True
            return False
        return matches

    def occupy_fixed_after_replacement_was_moved(
        directory_fd: int,
        name: str,
        *,
        identity: _FileIdentity,
        raw: bytes,
    ) -> bool:
        nonlocal inserted_occupant
        matches = real_regular_matches(
            directory_fd,
            name,
            identity=identity,
            raw=raw,
        )
        if matches and name.startswith(".receipt-") and raw != prior.raw and not inserted_occupant:
            inserted_occupant = True
            fixed = isolated_state / "receipt.json"
            assert not fixed.exists()
            fixed.write_bytes(fixed_occupant)
            fixed.chmod(0o600)
        return matches

    monkeypatch.setattr(owned_tiny, "_named_receipt_matches", reject_displaced_once)
    monkeypatch.setattr(
        owned_tiny,
        "_named_regular_file_matches",
        occupy_fixed_after_replacement_was_moved,
    )
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="transition"):
            operation.transition(current, OwnedTinyState.CREATED)

    assert forced_mismatch
    assert inserted_occupant
    assert (isolated_state / "receipt.json").read_bytes() == fixed_occupant
    staged_receipts = [child.read_bytes() for child in isolated_state.glob(".receipt-*.tmp")]
    assert prior.raw in staged_receipts
    assert any(raw != prior.raw and raw.startswith(b"{") for raw in staged_receipts)


def test_transition_post_persist_reload_rejects_same_byte_substitute(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt()
    real_remove_known = owned_tiny._remove_known_regular_file
    preserved_replacement = isolated_state / "preserved-installed-transition"
    substitute_inode: int | None = None
    attacked = False

    def substitute_before_old_staging_cleanup(
        directory_fd: int,
        name: str,
        *,
        identity: _FileIdentity,
        raw: bytes,
    ) -> None:
        nonlocal attacked, substitute_inode
        if name.startswith(".receipt-") and not attacked:
            attacked = True
            fixed = isolated_state / "receipt.json"
            replacement_raw = fixed.read_bytes()
            fixed.replace(preserved_replacement)
            substitute = isolated_state / "same-byte-transition-substitute"
            substitute.write_bytes(replacement_raw)
            substitute.chmod(0o600)
            substitute_inode = substitute.stat().st_ino
            substitute.replace(fixed)
        real_remove_known(
            directory_fd,
            name,
            identity=identity,
            raw=raw,
        )

    monkeypatch.setattr(
        owned_tiny,
        "_remove_known_regular_file",
        substitute_before_old_staging_cleanup,
    )
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="transition"):
            operation.transition(current, OwnedTinyState.CREATED)

    assert attacked
    assert substitute_inode is not None
    assert (isolated_state / "receipt.json").stat().st_ino == substitute_inode
    assert (isolated_state / "receipt.json").read_bytes() == preserved_replacement.read_bytes()


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


def test_atomic_exchange_restores_last_moment_receipt_substitute(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.INTENT)
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
        with pytest.raises(OwnedTinyStateError, match="transition"):
            operation.transition(current, OwnedTinyState.CREATED)

    assert attacked
    assert expected_path.read_bytes() == expected_raw
    assert expected_path.stat().st_ino != expected_inode
    assert preserved_expected.read_bytes() == expected_raw
    assert preserved_expected.stat().st_ino == expected_inode


def test_terminal_removal_preserves_substitute_at_atomic_move_boundary(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    receipt_path = isolated_state / "receipt.json"
    original_raw = receipt_path.read_bytes()
    original_inode = receipt_path.stat().st_ino
    preserved_expected = isolated_state / "expected-before-substitution"
    real_rename_noreplace = owned_tiny._rename_noreplace
    substitute_inode: int | None = None
    attacked = False

    def rename_after_substitution(
        directory_fd: int,
        source: str,
        destination: str,
    ) -> None:
        nonlocal attacked, substitute_inode
        if source == "receipt.json" and not attacked:
            attacked = True
            receipt_path.replace(preserved_expected)
            substitute = isolated_state / "manual-move-boundary-substitute"
            substitute.write_bytes(original_raw)
            substitute.chmod(0o600)
            substitute_inode = substitute.stat().st_ino
            substitute.replace(receipt_path)
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(
        owned_tiny,
        "_rename_noreplace",
        rename_after_substitution,
    )
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="could not be removed"):
            operation.remove_terminal(current)

    assert attacked
    assert substitute_inode is not None
    assert receipt_path.read_bytes() == original_raw
    assert receipt_path.stat().st_ino == substitute_inode
    assert preserved_expected.read_bytes() == original_raw
    assert preserved_expected.stat().st_ino == original_inode


def test_terminal_removal_wipes_only_held_inode_when_quarantine_path_is_replaced(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    receipt_path = isolated_state / "receipt.json"
    original_raw = receipt_path.read_bytes()
    original_inode = receipt_path.stat().st_ino
    preserved_expected = isolated_state / "held-expected-after-substitution"
    substitute_inode: int | None = None
    real_read_bounded = owned_tiny._read_bounded
    attacked = False

    def substitute_after_held_read(fd: int, maximum: int) -> bytes:
        nonlocal attacked, substitute_inode
        raw = real_read_bounded(fd, maximum)
        quarantines = list(isolated_state.glob(".receipt-*.tmp"))
        if not attacked and raw == original_raw and not receipt_path.exists() and quarantines:
            attacked = True
            quarantine = quarantines[0]
            quarantine.replace(preserved_expected)
            substitute = isolated_state / "manual-quarantine-substitute"
            substitute.write_bytes(original_raw)
            substitute.chmod(0o600)
            substitute_inode = substitute.stat().st_ino
            substitute.replace(quarantine)
        return raw

    monkeypatch.setattr(owned_tiny, "_read_bounded", substitute_after_held_read)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        operation.remove_terminal(current)

    assert attacked
    assert substitute_inode is not None
    assert not receipt_path.exists()
    assert preserved_expected.stat().st_ino == original_inode
    assert preserved_expected.read_bytes() == b""
    substitutes = [
        child
        for child in isolated_state.glob(".receipt-*.tmp")
        if child.stat().st_ino == substitute_inode
    ]
    assert len(substitutes) == 1
    assert substitutes[0].read_bytes() == original_raw


def test_terminal_removal_durably_moves_fixed_receipt_before_wiping_inode(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    real_rename_noreplace = owned_tiny._rename_noreplace
    real_fsync = owned_tiny.os.fsync
    real_ftruncate = owned_tiny.os.ftruncate
    events: list[str] = []

    def record_rename(directory_fd: int, source: str, destination: str) -> None:
        real_rename_noreplace(directory_fd, source, destination)
        if source == "receipt.json":
            events.append("rename")

    def record_fsync(fd: int) -> None:
        info = os.fstat(fd)
        events.append("directory-fsync" if stat.S_ISDIR(info.st_mode) else "file-fsync")
        real_fsync(fd)

    def record_ftruncate(fd: int, length: int) -> None:
        events.append("ftruncate")
        real_ftruncate(fd, length)

    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        monkeypatch.setattr(owned_tiny, "_rename_noreplace", record_rename)
        monkeypatch.setattr(owned_tiny.os, "fsync", record_fsync)
        monkeypatch.setattr(owned_tiny.os, "ftruncate", record_ftruncate)
        operation.remove_terminal(current)

    assert events[:4] == ["rename", "directory-fsync", "ftruncate", "file-fsync"]
    assert events.count("directory-fsync") >= 2
    assert not (isolated_state / "receipt.json").exists()


def test_terminal_quarantine_collision_preserves_fixed_receipt(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    prior = _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    collision = isolated_state / ".receipt-terminal-collision.tmp"
    collision.write_bytes(b"terminal-collider")
    collision.chmod(0o600)
    monkeypatch.setattr(owned_tiny, "_temporary_name", lambda: collision.name)

    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="could not be removed"):
            operation.remove_terminal(current)

    assert (isolated_state / "receipt.json").read_bytes() == prior.raw
    assert collision.read_bytes() == b"terminal-collider"


def test_terminal_prevalidation_fixed_occupant_preserves_both_objects(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    prior = _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    real_rename_noreplace = owned_tiny._rename_noreplace
    fixed_occupant = b"post-move-fixed-occupant"
    quarantine_name: str | None = None

    def occupy_fixed_after_terminal_move(
        directory_fd: int,
        source: str,
        destination: str,
    ) -> None:
        nonlocal quarantine_name
        real_rename_noreplace(directory_fd, source, destination)
        if source == "receipt.json" and quarantine_name is None:
            quarantine_name = destination
            fixed = isolated_state / "receipt.json"
            fixed.write_bytes(fixed_occupant)
            fixed.chmod(0o600)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", occupy_fixed_after_terminal_move)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="could not be removed"):
            operation.remove_terminal(current)

    assert quarantine_name is not None
    assert (isolated_state / "receipt.json").read_bytes() == fixed_occupant
    assert (isolated_state / quarantine_name).read_bytes() == prior.raw


def test_terminal_validated_fsync_failure_never_restores_quarantine_as_authority(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    real_fsync = owned_tiny.os.fsync
    attacked = False

    def fail_zero_inode_fsync(fd: int) -> None:
        nonlocal attacked
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode) and info.st_size == 0 and not attacked:
            attacked = True
            real_fsync(fd)
            raise OSError("synthetic post-truncate fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(owned_tiny.os, "fsync", fail_zero_inode_fsync)
    with owned_tiny_ingest_operation() as operation:
        current = operation.load(required=True)
        assert current is not None
        with pytest.raises(OwnedTinyStateError, match="could not be removed"):
            operation.remove_terminal(current)

    assert attacked
    assert not (isolated_state / "receipt.json").exists()
    tombstones = list(isolated_state.glob(".receipt-*.tmp"))
    assert len(tombstones) == 1
    assert tombstones[0].read_bytes() == b""
    with owned_tiny_ingest_operation() as operation:
        assert operation.load(required=False) is None


def test_terminal_interruption_after_validation_closes_fd_without_restoring_authority(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    descriptor_count = _open_descriptor_count()

    def interrupt_before_truncate(fd: int, length: int) -> None:
        del fd, length
        raise KeyboardInterrupt()

    monkeypatch.setattr(owned_tiny.os, "ftruncate", interrupt_before_truncate)
    with (
        pytest.raises(KeyboardInterrupt),
        owned_tiny_ingest_operation() as operation,
    ):
        current = operation.load(required=True)
        assert current is not None
        operation.remove_terminal(current)

    assert not (isolated_state / "receipt.json").exists()
    quarantines = list(isolated_state.glob(".receipt-*.tmp"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes()
    assert _open_descriptor_count() == descriptor_count


def test_terminal_interruption_after_file_fsync_closes_fd_and_keeps_fixed_absent(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    real_fsync = owned_tiny.os.fsync
    file_synced = False
    descriptor_count = _open_descriptor_count()

    def interrupt_before_directory_fsync(fd: int) -> None:
        nonlocal file_synced
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode) and info.st_size == 0:
            real_fsync(fd)
            file_synced = True
            return
        if stat.S_ISDIR(info.st_mode) and file_synced:
            raise KeyboardInterrupt()
        real_fsync(fd)

    monkeypatch.setattr(owned_tiny.os, "fsync", interrupt_before_directory_fsync)
    with (
        pytest.raises(KeyboardInterrupt),
        owned_tiny_ingest_operation() as operation,
    ):
        current = operation.load(required=True)
        assert current is not None
        operation.remove_terminal(current)

    assert file_synced
    assert not (isolated_state / "receipt.json").exists()
    tombstones = list(isolated_state.glob(".receipt-*.tmp"))
    assert len(tombstones) == 1
    assert tombstones[0].read_bytes() == b""
    assert _open_descriptor_count() == descriptor_count


def test_crash_moved_terminal_quarantine_is_never_restored_as_authority(
    isolated_state: Path,
) -> None:
    prior = _create_receipt(OwnedTinyState.NOT_FOUND_VERIFIED)
    quarantine = isolated_state / ".receipt-simulated-crash.tmp"
    (isolated_state / "receipt.json").replace(quarantine)

    with owned_tiny_ingest_operation() as operation:
        assert operation.load(required=False) is None
        replacement = operation.create_intent(api_key=_KEY, region=_REGION)

    assert replacement.receipt.namespace != prior.receipt.namespace
    assert quarantine.read_bytes() == prior.raw


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
    state = tmp_path.resolve() / ".pufferlab/state/owned-tiny-v1"
    for component in (state.parents[1], state.parent, state):
        component.mkdir(mode=0o700)
        component.chmod(0o700)
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
    real = tmp_path.resolve() / "real-pufferlab"
    real.mkdir(mode=0o700)
    linked = tmp_path.resolve() / ".pufferlab"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        "pufferlab.owned_tiny._production_state_path",
        lambda: linked / "state/owned-tiny-v1",
    )
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())

    descriptor_count = _open_descriptor_count()
    for _ in range(2):
        with pytest.raises(OwnedTinyStateError) as caught, owned_tiny_ingest_operation():
            pass
        assert type(caught.value) is OwnedTinyStateError
    assert _open_descriptor_count() == descriptor_count


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_state_open_process_control_closes_every_owned_descriptor_and_releases_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    from pufferlab import owned_tiny

    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    for component in (state.parents[1], state.parent, state):
        component.mkdir(mode=0o700)
        component.chmod(0o700)
    state_inode = state.stat().st_ino
    real_clear_nonblocking = owned_tiny._clear_nonblocking
    attacked = False

    def interrupt_after_final_state_open(fd: int) -> None:
        nonlocal attacked
        if not attacked and os.fstat(fd).st_ino == state_inode:
            attacked = True
            raise control("private-state-open-control-marker")
        real_clear_nonblocking(fd)

    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: anchor)
    monkeypatch.setattr(owned_tiny, "_clear_nonblocking", interrupt_after_final_state_open)
    descriptor_count = _open_descriptor_count()

    with pytest.raises(control), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert _open_descriptor_count() == descriptor_count
    monkeypatch.setattr(owned_tiny, "_clear_nonblocking", real_clear_nonblocking)
    with owned_tiny_ingest_operation():
        pass


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_anchor_validation_process_control_closes_descriptor_and_allows_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    from pufferlab import owned_tiny

    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    anchor_inode = anchor.stat().st_ino
    real_validate = owned_tiny._validate_anchor_directory
    attacked = False

    def interrupt_final_anchor_validation(
        info: os.stat_result,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal attacked
        if not attacked and info.st_ino == anchor_inode:
            attacked = True
            raise control("private-anchor-control-marker")
        real_validate(info, expected_identity=expected_identity)

    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: anchor)
    monkeypatch.setattr(owned_tiny, "_validate_anchor_directory", interrupt_final_anchor_validation)
    descriptor_count = _open_descriptor_count()

    with pytest.raises(control), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert _open_descriptor_count() == descriptor_count
    monkeypatch.setattr(owned_tiny, "_validate_anchor_directory", real_validate)
    with owned_tiny_ingest_operation():
        pass


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_fixed_lock_post_open_process_control_closes_all_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    from pufferlab import owned_tiny

    anchor = tmp_path.resolve()
    state = anchor / ".pufferlab/state/owned-tiny-v1"
    for component in (state.parents[1], state.parent, state):
        component.mkdir(mode=0o700)
        component.chmod(0o700)
    real_validate = owned_tiny._validate_regular_file
    attacked = False

    def interrupt_lock_validation(fd: int) -> os.stat_result:
        nonlocal attacked
        info = os.fstat(fd)
        if not attacked and stat.S_ISREG(info.st_mode) and info.st_size == 0:
            attacked = True
            raise control("private-lock-control-marker")
        return real_validate(fd)

    monkeypatch.setattr(owned_tiny, "_production_state_path", lambda: state)
    monkeypatch.setattr(owned_tiny, "_production_anchor_path", lambda: anchor)
    monkeypatch.setattr(owned_tiny, "_validate_regular_file", interrupt_lock_validation)
    descriptor_count = _open_descriptor_count()

    with pytest.raises(control), owned_tiny_ingest_operation():
        pass

    assert attacked
    assert _open_descriptor_count() == descriptor_count
    monkeypatch.setattr(owned_tiny, "_validate_regular_file", real_validate)
    with owned_tiny_ingest_operation():
        pass


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_private_directory_staging_process_control_closes_staged_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    from pufferlab import owned_tiny

    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_validate = owned_tiny._validate_private_directory
    attacked = False

    def interrupt_staged_validation(info: os.stat_result) -> None:
        nonlocal attacked
        if not attacked:
            attacked = True
            raise control("private-directory-stage-control-marker")
        real_validate(info)

    monkeypatch.setattr(owned_tiny, "_validate_private_directory", interrupt_staged_validation)
    descriptor_count = _open_descriptor_count()
    try:
        with pytest.raises(control):
            owned_tiny._install_private_directory(
                parent_fd,
                ".pufferlab",
                flags=os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        assert attacked
        assert _open_descriptor_count() == descriptor_count
        assert not (tmp_path / ".pufferlab").exists()
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_private_file_staging_process_control_closes_created_descriptor(
    isolated_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: type[BaseException],
) -> None:
    from pufferlab import owned_tiny

    _create_receipt()
    directory_fd = os.open(isolated_state, os.O_RDONLY | os.O_DIRECTORY)
    attacked = False

    def interrupt_staged_write(fd: int, value: bytes) -> None:
        nonlocal attacked
        del fd, value
        attacked = True
        raise control("private-file-stage-control-marker")

    monkeypatch.setattr(owned_tiny, "_write_all", interrupt_staged_write)
    descriptor_count = _open_descriptor_count()
    try:
        with pytest.raises(control):
            owned_tiny._prepare_temporary(directory_fd, ".receipt-control-stage.tmp", b"value")
        assert attacked
        assert _open_descriptor_count() == descriptor_count
        assert not (isolated_state / ".receipt-control-stage.tmp").exists()
    finally:
        os.close(directory_fd)


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
