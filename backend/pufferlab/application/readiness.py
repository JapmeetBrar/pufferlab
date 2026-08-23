"""Pure, provider-free local capability inspection."""

from __future__ import annotations

from typing import Protocol

from pufferlab.config import Settings
from pufferlab.contracts.capabilities import (
    CAPABILITY_ACTION_ORDER,
    CAPABILITY_REQUIREMENT_ORDER,
    CapabilitiesResponse,
    CapabilityRequirementCode,
    CapabilityState,
    LivePlaygroundCapability,
)
from pufferlab.retrieval.preflight import (
    RuntimeAvailabilityCheck,
    local_search_requirements,
    sentence_transformers_available,
)

_OWNED_TINY_REQUIREMENT_ORDER = (
    CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
    CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
    CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
)


class CapabilityInspector(Protocol):
    def inspect(self) -> CapabilitiesResponse: ...


class OwnedTinyRequirementResolver(Protocol):
    """M4-C seam for authenticated, value-free receipt requirements."""

    def __call__(self, settings: Settings) -> tuple[CapabilityRequirementCode, ...]: ...


def unresolved_owned_tiny_requirements(
    settings: Settings,
) -> tuple[CapabilityRequirementCode, ...]:
    """Fail closed until M4-C supplies the authenticated receipt resolver."""
    del settings
    return (CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,)


class LocalCapabilityInspector:
    """Inspect only local settings, package discovery, and an injected receipt reader."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_available: RuntimeAvailabilityCheck = sentence_transformers_available,
        owned_tiny_requirements: OwnedTinyRequirementResolver = (
            unresolved_owned_tiny_requirements
        ),
    ) -> None:
        self._settings = settings
        self._runtime_available = runtime_available
        self._owned_tiny_requirements = owned_tiny_requirements

    def inspect(self) -> CapabilitiesResponse:
        requirements = local_search_requirements(
            self._settings,
            runtime_available=self._runtime_available,
        )
        if not requirements:
            requirements = self._resolve_owned_tiny_requirements()
        if not requirements:
            capability = LivePlaygroundCapability(
                state=CapabilityState.LOCALLY_CONFIGURED,
                requirements=(),
                next_action=None,
            )
        else:
            first_index = CAPABILITY_REQUIREMENT_ORDER.index(requirements[0])
            capability = LivePlaygroundCapability(
                state=CapabilityState.ACTION_REQUIRED,
                requirements=requirements,
                next_action=CAPABILITY_ACTION_ORDER[first_index],
            )
        return CapabilitiesResponse(live_playground=capability)

    def _resolve_owned_tiny_requirements(self) -> tuple[CapabilityRequirementCode, ...]:
        requirements = self._owned_tiny_requirements(self._settings)
        if len(requirements) != len(set(requirements)):
            raise ValueError("owned-tiny readiness requirements must be unique")
        if any(requirement not in _OWNED_TINY_REQUIREMENT_ORDER for requirement in requirements):
            raise ValueError("owned-tiny readiness returned an unsupported requirement")
        ordered = tuple(
            requirement
            for requirement in _OWNED_TINY_REQUIREMENT_ORDER
            if requirement in requirements
        )
        if requirements != ordered:
            raise ValueError("owned-tiny readiness requirements are out of order")
        return requirements
