"""Retrieval configuration discovery."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pufferlab.api.dependencies import get_search_backend
from pufferlab.contracts.errors import ApiErrorDetail
from pufferlab.contracts.retrieval import RetrievalConfigListResponse
from pufferlab.retrieval.types import SearchBackend

router = APIRouter(tags=["retrieval"])


@router.get(
    "/configs",
    operation_id="list_retrieval_configs",
    response_model=RetrievalConfigListResponse,
    responses={
        500: {"model": ApiErrorDetail, "description": "The request failed unexpectedly."},
        503: {"model": ApiErrorDetail, "description": "Search runtime is unavailable."},
    },
)
async def list_retrieval_configs(
    backend: Annotated[SearchBackend, Depends(get_search_backend)],
) -> RetrievalConfigListResponse:
    return RetrievalConfigListResponse(configs=list(backend.list_configs()))
