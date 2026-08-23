"""Versioned durable evaluation read surface and injected control signatures."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from pufferlab.api.evaluation_dependencies import (
    get_evaluation_controls,
    get_evaluation_views,
)
from pufferlab.api.evaluation_facades import EvaluationControlFacade, EvaluationViewFacade
from pufferlab.contracts.errors import ApiErrorDetail
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    EvalRunDetailResponse,
    EvalRunExportResponse,
    EvalRunListQuery,
    EvalRunListResponse,
    EvalRunQueryDetailResponse,
    RegressionOrder,
    RegressionQuery,
    RegressionResponse,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
)

router = APIRouter(prefix="/eval-runs", tags=["evaluation runs"])

_EVALUATION_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ApiErrorDetail, "description": "The requested evaluation record was not found."},
    409: {"model": ApiErrorDetail, "description": "The requested evaluation action conflicts."},
    422: {"model": ApiErrorDetail, "description": "The evaluation request is invalid."},
    500: {"model": ApiErrorDetail, "description": "The request failed unexpectedly."},
    503: {"model": ApiErrorDetail, "description": "Evaluation data or runtime is unavailable."},
}


@router.post(
    "",
    operation_id="create_evaluation_run",
    response_model=CreateEvalRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_EVALUATION_ERRORS,
)
async def create_evaluation_run(
    request: CreateEvalRunRequest,
    controls: Annotated[EvaluationControlFacade, Depends(get_evaluation_controls)],
) -> CreateEvalRunResponse:
    return await controls.create_eval_run(request)


@router.get(
    "",
    operation_id="list_evaluation_runs",
    response_model=EvalRunListResponse,
    responses=_EVALUATION_ERRORS,
)
def list_evaluation_runs(
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> EvalRunListResponse:
    return views.list_eval_runs(EvalRunListQuery(limit=limit))


@router.get(
    "/{run_id}",
    operation_id="get_evaluation_run",
    response_model=EvalRunDetailResponse,
    responses=_EVALUATION_ERRORS,
)
def get_evaluation_run(
    run_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> EvalRunDetailResponse:
    return views.get_eval_run(run_id)


@router.post(
    "/{run_id}/cancel",
    operation_id="cancel_evaluation_run",
    response_model=CancelEvalRunResponse,
    responses=_EVALUATION_ERRORS,
)
async def cancel_evaluation_run(
    run_id: UUID,
    controls: Annotated[EvaluationControlFacade, Depends(get_evaluation_controls)],
) -> CancelEvalRunResponse:
    return await controls.cancel_eval_run(run_id)


@router.get(
    "/{run_id}/regressions",
    operation_id="get_evaluation_regressions",
    response_model=RegressionResponse,
    responses=_EVALUATION_ERRORS,
)
def get_evaluation_regressions(
    run_id: UUID,
    candidate_config_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
    order: RegressionOrder = RegressionOrder.REGRESSIONS,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RegressionResponse:
    return views.get_regressions(
        run_id,
        RegressionQuery(
            candidate_config_id=candidate_config_id,
            order=order,
            limit=limit,
        ),
    )


@router.get(
    "/{run_id}/queries/{query_id}",
    operation_id="get_evaluation_run_query",
    response_model=EvalRunQueryDetailResponse,
    responses=_EVALUATION_ERRORS,
)
def get_evaluation_run_query(
    run_id: UUID,
    query_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> EvalRunQueryDetailResponse:
    return views.get_query_detail(run_id, query_id)


@router.get(
    "/{run_id}/export",
    operation_id="export_evaluation_run",
    response_model=EvalRunExportResponse,
    responses=_EVALUATION_ERRORS,
)
def export_evaluation_run(
    run_id: UUID,
    views: Annotated[EvaluationViewFacade, Depends(get_evaluation_views)],
) -> EvalRunExportResponse:
    return views.export_eval_run(run_id)


@router.post(
    "/{run_id}/queries/{query_id}/replay",
    operation_id="replay_evaluation_run_query",
    response_model=EvalRunQueryReplayResponse,
    responses=_EVALUATION_ERRORS,
)
async def replay_evaluation_run_query(
    run_id: UUID,
    query_id: UUID,
    request: EvalRunQueryReplayRequest,
    controls: Annotated[EvaluationControlFacade, Depends(get_evaluation_controls)],
) -> EvalRunQueryReplayResponse:
    return await controls.replay_eval_query(run_id, query_id, request)
