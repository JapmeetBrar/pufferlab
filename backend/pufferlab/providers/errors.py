"""Redacted translation of turbopuffer failures into public contract codes."""

from dataclasses import dataclass

from turbopuffer import APIConnectionError, APIError, RateLimitError

from pufferlab.contracts.errors import ApiErrorCode


@dataclass(frozen=True, slots=True)
class ProviderErrorDetails:
    code: ApiErrorCode
    retryable: bool
    operation: str
    status_code: int | None = None


class ProviderError(Exception):
    """A safe provider failure that never includes the SDK exception body or request."""

    def __init__(self, message: str, details: ProviderErrorDetails) -> None:
        super().__init__(message)
        self.details = details


def map_turbopuffer_error(error: APIError, *, operation: str) -> ProviderError:
    """Map an SDK failure without copying its potentially sensitive message or body."""

    raw_status_code = getattr(error, "status_code", None)
    status_code = raw_status_code if isinstance(raw_status_code, int) else None

    if isinstance(error, RateLimitError) or status_code == 429:
        return ProviderError(
            "turbopuffer rate limit exceeded",
            ProviderErrorDetails(
                code=ApiErrorCode.RATE_LIMITED,
                retryable=True,
                operation=operation,
                status_code=status_code,
            ),
        )

    if status_code == 404:
        return ProviderError(
            "turbopuffer namespace was not found",
            ProviderErrorDetails(
                code=ApiErrorCode.NOT_FOUND,
                retryable=False,
                operation=operation,
                status_code=status_code,
            ),
        )

    if status_code == 202:
        return ProviderError(
            "turbopuffer namespace index is not ready",
            ProviderErrorDetails(
                code=ApiErrorCode.NAMESPACE_NOT_READY,
                retryable=True,
                operation=operation,
                status_code=status_code,
            ),
        )

    retryable = isinstance(error, APIConnectionError) or status_code in {408, 409, 425}
    retryable = retryable or (status_code is not None and status_code >= 500)
    return ProviderError(
        "turbopuffer request failed",
        ProviderErrorDetails(
            code=ApiErrorCode.PROVIDER_ERROR,
            retryable=retryable,
            operation=operation,
            status_code=status_code,
        ),
    )
