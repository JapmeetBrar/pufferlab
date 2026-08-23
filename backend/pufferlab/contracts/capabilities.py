"""Provider-free local capability contracts."""

from enum import StrEnum

from pydantic import ConfigDict, Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion


class CapabilityState(StrEnum):
    LOCALLY_CONFIGURED = "locally_configured"
    ACTION_REQUIRED = "action_required"


class CapabilityRequirementCode(StrEnum):
    API_KEY = "api_key"
    SEARCH_NAMESPACE = "search_namespace"
    REGION = "region"
    LIVE_SEARCH_RUNTIME = "live_search_runtime"
    OWNED_TINY_RECEIPT_INVALID = "owned_tiny_receipt_invalid"
    OWNED_TINY_CREDENTIAL_MISMATCH = "owned_tiny_credential_mismatch"
    OWNED_TINY_REGION_MISMATCH = "owned_tiny_region_mismatch"


class CapabilityActionCode(StrEnum):
    CONFIGURE_API_KEY = "configure_api_key"
    CONFIGURE_SEARCH_NAMESPACE = "configure_search_namespace"
    CONFIGURE_REGION = "configure_region"
    INSTALL_LIVE_SEARCH_RUNTIME = "install_live_search_runtime"
    RESOLVE_OWNED_TINY_RECEIPT = "resolve_owned_tiny_receipt"
    USE_OWNED_TINY_CREDENTIAL = "use_owned_tiny_credential"
    USE_OWNED_TINY_REGION = "use_owned_tiny_region"


CAPABILITY_REQUIREMENT_ORDER = tuple(CapabilityRequirementCode)
CAPABILITY_ACTION_ORDER = tuple(CapabilityActionCode)
_ACTION_BY_REQUIREMENT = dict(
    zip(CAPABILITY_REQUIREMENT_ORDER, CAPABILITY_ACTION_ORDER, strict=True)
)


class _FrozenCapabilityModel(ContractModel):
    model_config = ConfigDict(frozen=True)


class LivePlaygroundCapability(_FrozenCapabilityModel):
    """Local readiness only; this contract never asserts remote provider health."""

    state: CapabilityState
    requirements: tuple[CapabilityRequirementCode, ...] = Field(
        max_length=len(CAPABILITY_REQUIREMENT_ORDER)
    )
    next_action: CapabilityActionCode | None

    @model_validator(mode="after")
    def validate_state_and_action(self) -> "LivePlaygroundCapability":
        if len(self.requirements) != len(set(self.requirements)):
            raise ValueError("capability requirements must be unique")
        ordered = sorted(
            self.requirements,
            key=CAPABILITY_REQUIREMENT_ORDER.index,
        )
        if self.requirements != tuple(ordered):
            raise ValueError("capability requirements must use the frozen contract order")
        if not self.requirements:
            if self.state is not CapabilityState.LOCALLY_CONFIGURED or self.next_action is not None:
                raise ValueError("locally configured capability requires no action")
            return self
        expected_action = _ACTION_BY_REQUIREMENT[self.requirements[0]]
        if self.state is not CapabilityState.ACTION_REQUIRED:
            raise ValueError("unmet capability requirements require action")
        if self.next_action is not expected_action:
            raise ValueError("next action must match the first unmet requirement")
        return self


class CapabilitiesResponse(_FrozenCapabilityModel):
    contract_version: ContractVersion = 1
    live_playground: LivePlaygroundCapability
