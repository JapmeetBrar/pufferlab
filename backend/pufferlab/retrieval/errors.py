"""Safe failures raised by the retrieval layer."""

from dataclasses import dataclass
from uuid import UUID

from pufferlab.contracts.common import JsonValue
from pufferlab.contracts.errors import ApiErrorCode, ApiErrorDetail


@dataclass(frozen=True, slots=True)
class SearchErrorDetails:
    code: ApiErrorCode
    retryable: bool
    http_status: int
    operation: str


class SearchError(Exception):
    """A public-safe search error with no copied third-party exception text."""

    def __init__(self, message: str, details: SearchErrorDetails) -> None:
        super().__init__(message)
        self.details = details

    def to_api_error(self, trace_id: UUID) -> ApiErrorDetail:
        details: dict[str, JsonValue] = {"operation": self.details.operation}
        return ApiErrorDetail(
            code=self.details.code,
            message=str(self),
            retryable=self.details.retryable,
            trace_id=trace_id,
            details=details,
        )


def invalid_search(message: str) -> SearchError:
    return SearchError(
        message,
        SearchErrorDetails(
            code=ApiErrorCode.VALIDATION_ERROR,
            retryable=False,
            http_status=422,
            operation="compare",
        ),
    )


def config_not_found() -> SearchError:
    return SearchError(
        "retrieval configuration was not found",
        SearchErrorDetails(
            code=ApiErrorCode.NOT_FOUND,
            retryable=False,
            http_status=404,
            operation="resolve_config",
        ),
    )


def embedding_failed() -> SearchError:
    return SearchError(
        "query embedding failed",
        SearchErrorDetails(
            code=ApiErrorCode.INTERNAL_ERROR,
            retryable=False,
            http_status=503,
            operation="embed_query",
        ),
    )


def search_unavailable() -> SearchError:
    return SearchError(
        "search backend is not configured",
        SearchErrorDetails(
            code=ApiErrorCode.INTERNAL_ERROR,
            retryable=False,
            http_status=503,
            operation="search_runtime",
        ),
    )


def invalid_provider_result() -> SearchError:
    return SearchError(
        "retrieval provider returned an invalid result",
        SearchErrorDetails(
            code=ApiErrorCode.PROVIDER_ERROR,
            retryable=False,
            http_status=502,
            operation="map_search_result",
        ),
    )


def provider_failed(operation: str) -> SearchError:
    return SearchError(
        "retrieval provider request failed",
        SearchErrorDetails(
            code=ApiErrorCode.PROVIDER_ERROR,
            retryable=False,
            http_status=502,
            operation=operation,
        ),
    )
