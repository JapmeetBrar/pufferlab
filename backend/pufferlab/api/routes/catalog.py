"""Provider-free persisted dataset, query-set, and evaluation-config catalogs."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends

from pufferlab.api.evaluation_dependencies import get_evaluation_views
from pufferlab.api.evaluation_facades import EvaluationViewFacade
from pufferlab.contracts.catalog import (
    DatasetDetailResponse,
    DatasetListResponse,
    QuerySetListResponse,
    RetrievalConfigCatalogResponse,
)
from pufferlab.contracts.errors import ApiErrorDetail

router = APIRouter(tags=["evaluation catalogs"])

_CATALOG_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ApiErrorDetail, "description": "The requested revision was not found."},
    409: {"model": ApiErrorDetail, "description": "The persisted catalog is not canonical."},
    422: {"model": ApiErrorDetail, "description": "The catalog request is invalid."},
    500: {"model": ApiErrorDetail, "description": "The request failed unexpectedly."},
    503: {"model": ApiErrorDetail, "description": "Stored catalog data is unavailable."},
}


@router.get(
    "/datasets",
    operation_id="list_evaluation_datasets",
    response_model=DatasetListResponse,
    responses=_CATALOG_ERRORS,
)
def list_evaluation_datasets(
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> DatasetListResponse:
    return views.list_datasets()


@router.get(
    "/datasets/{dataset_version_id}",
    operation_id="get_evaluation_dataset",
    response_model=DatasetDetailResponse,
    responses=_CATALOG_ERRORS,
)
def get_evaluation_dataset(
    dataset_version_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> DatasetDetailResponse:
    return views.get_dataset(dataset_version_id)


@router.get(
    "/query-sets",
    operation_id="list_evaluation_query_sets",
    response_model=QuerySetListResponse,
    responses=_CATALOG_ERRORS,
)
def list_evaluation_query_sets(
    dataset_version_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> QuerySetListResponse:
    return views.list_query_sets(dataset_version_id)


@router.get(
    "/datasets/{dataset_version_id}/configs",
    operation_id="list_dataset_evaluation_configs",
    response_model=RetrievalConfigCatalogResponse,
    responses=_CATALOG_ERRORS,
)
def list_dataset_evaluation_configs(
    dataset_version_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> RetrievalConfigCatalogResponse:
    return views.list_dataset_configs(dataset_version_id)
