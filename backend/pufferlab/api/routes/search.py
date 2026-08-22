"""Interactive search comparison route."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from pufferlab.api.dependencies import get_search_backend
from pufferlab.contracts.errors import ApiErrorDetail
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.retrieval.types import SearchBackend

router = APIRouter(tags=["search"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {"model": ApiErrorDetail, "description": "The request failed unexpectedly."},
    404: {"model": ApiErrorDetail, "description": "A retrieval config or namespace was not found."},
    422: {"model": ApiErrorDetail, "description": "The comparison request is invalid."},
    429: {"model": ApiErrorDetail, "description": "The provider rate limit was exceeded."},
    502: {"model": ApiErrorDetail, "description": "The provider request failed."},
    503: {"model": ApiErrorDetail, "description": "Search is temporarily unavailable."},
}


@router.post(
    "/search/compare",
    operation_id="compare_search_configs",
    response_model=SearchCompareResponse,
    responses=_ERROR_RESPONSES,
)
async def compare_search_configs(
    request: SearchCompareRequest,
    backend: Annotated[SearchBackend, Depends(get_search_backend)],
) -> SearchCompareResponse:
    return await backend.compare(request)
