"""Provider-free local capability route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pufferlab.api.dependencies import get_capability_inspector
from pufferlab.application.readiness import CapabilityInspector
from pufferlab.contracts.capabilities import CapabilitiesResponse

router = APIRouter(tags=["capabilities"])


@router.get(
    "/capabilities",
    operation_id="get_local_capabilities",
    response_model=CapabilitiesResponse,
)
async def get_local_capabilities(
    inspector: Annotated[CapabilityInspector, Depends(get_capability_inspector)],
) -> CapabilitiesResponse:
    return inspector.inspect()
