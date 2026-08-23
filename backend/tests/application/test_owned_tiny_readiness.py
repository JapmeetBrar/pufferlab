from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pufferlab.application.readiness import LocalCapabilityInspector
from pufferlab.cli.doctor import (
    DoctorCheckState,
    DoctorMode,
    default_doctor_dependencies,
    run_doctor,
)
from pufferlab.config import Settings
from pufferlab.contracts.capabilities import (
    CapabilityRequirementCode,
    CapabilityState,
)
from pufferlab.owned_tiny import OwnedTinyState, owned_tiny_ingest_operation
from pufferlab.providers.metadata_probe import MetadataProbeResult, MetadataProbeState

_KEY = "fake-readiness-key"
_REGION = "gcp-us-central1"


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path.resolve() / "owned-tiny-state"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    return state


def _ready_receipt() -> str:
    with owned_tiny_ingest_operation() as operation:
        snapshot = operation.create_intent(api_key=_KEY, region=_REGION)
        snapshot = operation.transition(snapshot, OwnedTinyState.CREATED)
        snapshot = operation.transition(snapshot, OwnedTinyState.READY)
        return snapshot.receipt.namespace


def _settings(namespace: str, *, key: str = _KEY, region: str = _REGION) -> Settings:
    return Settings.model_validate(
        {
            "turbopuffer_api_key": key,
            "turbopuffer_region": region,
            "pufferlab_search_namespace": namespace,
        }
    )


def test_default_capability_inspector_integrates_authenticated_receipt(
    isolated_state: Path,
) -> None:
    del isolated_state
    namespace = _ready_receipt()

    response = LocalCapabilityInspector(
        _settings(namespace),
        runtime_available=lambda: True,
    ).inspect()

    assert response.live_playground.state is CapabilityState.LOCALLY_CONFIGURED
    assert response.live_playground.requirements == ()
    assert response.live_playground.next_action is None


def test_default_capability_inspector_reports_exact_receipt_mismatch_codes(
    isolated_state: Path,
) -> None:
    del isolated_state
    namespace = _ready_receipt()

    response = LocalCapabilityInspector(
        _settings(namespace, key="rotated-key", region="aws-us-east-2"),
        runtime_available=lambda: True,
    ).inspect()

    assert response.live_playground.state is CapabilityState.ACTION_REQUIRED
    assert response.live_playground.requirements == (
        CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
        CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
    )


@pytest.mark.asyncio
async def test_default_doctor_resolves_ready_receipt_without_provider_work(
    isolated_state: Path,
) -> None:
    del isolated_state
    namespace = _ready_receipt()
    dependencies = replace(
        default_doctor_dependencies(),
        capability_inspector_factory=lambda settings: LocalCapabilityInspector(
            settings,
            runtime_available=lambda: True,
        ),
    )

    execution = await run_doctor(
        _settings(namespace),
        mode=DoctorMode.LIVE_TINY,
        dataset_version_id=None,
        live=False,
        dependencies=dependencies,
    )

    assert execution.exit_code == 0
    assert len(execution.report.checks) == 1
    assert execution.report.checks[0].state is DoctorCheckState.READY


@pytest.mark.asyncio
async def test_default_doctor_live_probe_uses_exact_authenticated_target(
    isolated_state: Path,
) -> None:
    del isolated_state
    namespace = _ready_receipt()
    calls: list[tuple[str, str, str]] = []

    async def probe(*, api_key: str, region: str, namespace: str) -> MetadataProbeResult:
        calls.append((api_key, region, namespace))
        return MetadataProbeResult(state=MetadataProbeState.INDEX_UP_TO_DATE)

    dependencies = replace(
        default_doctor_dependencies(),
        capability_inspector_factory=lambda settings: LocalCapabilityInspector(
            settings,
            runtime_available=lambda: True,
        ),
        metadata_probe=probe,
    )
    execution = await run_doctor(
        _settings(namespace),
        mode=DoctorMode.LIVE_TINY,
        dataset_version_id=None,
        live=True,
        dependencies=dependencies,
    )

    assert execution.exit_code == 0
    assert calls == [(_KEY, _REGION, namespace)]
    assert [check.state for check in execution.report.checks] == [
        DoctorCheckState.READY,
        DoctorCheckState.READY,
    ]
