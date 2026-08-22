"""Render safe domain failures as the versioned public error contract."""

from uuid import uuid4

from fastapi.responses import JSONResponse

from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail
from pufferlab.providers.errors import ProviderError
from pufferlab.retrieval.errors import SearchError


def search_error_response(error: SearchError) -> JSONResponse:
    detail = error.to_api_error(uuid4())
    return JSONResponse(
        status_code=error.details.http_status,
        content=detail.model_dump(mode="json"),
    )


def provider_error_response(error: ProviderError) -> JSONResponse:
    status_code = {
        ApiErrorCode.NOT_FOUND: 404,
        ApiErrorCode.RATE_LIMITED: 429,
        ApiErrorCode.NAMESPACE_NOT_READY: 503,
    }.get(error.details.code, 502)
    detail = ApiErrorDetail(
        code=error.details.code,
        message=str(error),
        retryable=error.details.retryable,
        trace_id=uuid4(),
        details={"operation": error.details.operation},
    )
    return JSONResponse(status_code=status_code, content=detail.model_dump(mode="json"))


def validation_error_response() -> JSONResponse:
    detail = ApiErrorDetail(
        code=ApiErrorCode.VALIDATION_ERROR,
        message="request validation failed",
        retryable=False,
        trace_id=uuid4(),
        details={"operation": "validate_request"},
    )
    return JSONResponse(status_code=422, content=detail.model_dump(mode="json"))


def internal_error_response() -> JSONResponse:
    detail = ApiErrorDetail(
        code=ApiErrorCode.INTERNAL_ERROR,
        message="request failed unexpectedly",
        retryable=False,
        trace_id=uuid4(),
        details={"operation": "request"},
    )
    return JSONResponse(status_code=500, content=detail.model_dump(mode="json"))
