"""Public error and warning contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from pufferlab.contracts.common import ContractModel, JsonValue


class ApiErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_REQUIRED = "configuration_required"
    NOT_FOUND = "not_found"
    NAMESPACE_NOT_READY = "namespace_not_ready"
    PROVIDER_ERROR = "provider_error"
    RATE_LIMITED = "rate_limited"
    RUN_CONFLICT = "run_conflict"
    INTERNAL_ERROR = "internal_error"


class ApiErrorDetail(ContractModel):
    code: ApiErrorCode
    message: str
    retryable: bool
    trace_id: UUID
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ApiWarning(ContractModel):
    code: str
    message: str
