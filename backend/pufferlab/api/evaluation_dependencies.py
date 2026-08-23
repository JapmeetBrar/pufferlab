"""FastAPI dependencies for provider-free evaluation views and injected controls."""

from fastapi import Request

from pufferlab.api.evaluation_facades import EvaluationControlFacade, EvaluationViewFacade
from pufferlab.application.view_errors import EvaluationViewError
from pufferlab.contracts.errors import ApiErrorCode


def get_evaluation_views(request: Request) -> EvaluationViewFacade:
    views: EvaluationViewFacade | None = getattr(request.app.state, "evaluation_views", None)
    if views is None:
        raise _missing_dependency("evaluation read service", "resolve_evaluation_views")
    return views


def get_evaluation_controls(request: Request) -> EvaluationControlFacade:
    controls: EvaluationControlFacade | None = getattr(
        request.app.state,
        "evaluation_controls",
        None,
    )
    if controls is None:
        raise _missing_dependency("evaluation control runtime", "resolve_evaluation_controls")
    return controls


def _missing_dependency(name: str, operation: str) -> EvaluationViewError:
    return EvaluationViewError(
        code=ApiErrorCode.INTERNAL_ERROR,
        message=f"{name} is not configured",
        http_status=503,
        operation=operation,
    )
