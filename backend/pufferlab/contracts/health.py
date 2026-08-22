"""Health endpoint contract."""

from typing import Literal

from pufferlab.contracts.common import ContractModel, ContractVersion


class HealthResponse(ContractModel):
    contract_version: ContractVersion = 1
    status: Literal["ok"] = "ok"
    version: str
