"""Persistence-owned records that are not part of the HTTP contract."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pufferlab.contracts.common import ContractModel, JsonValue


class QueryOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class QueryOutcome(ContractModel):
    """One durable config/query result; the eval layer owns the payload schema."""

    run_id: UUID
    config_id: UUID
    query_id: UUID
    status: QueryOutcomeStatus
    payload: dict[str, JsonValue]
    created_at: datetime
