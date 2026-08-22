"""Health route."""

from fastapi import APIRouter

from pufferlab import __version__
from pufferlab.contracts.health import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, operation_id="get_health")
async def get_health() -> HealthResponse:
    return HealthResponse(version=__version__)
