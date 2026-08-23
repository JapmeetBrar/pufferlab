"""Safe application failures for provider-free evaluation and catalog views."""

from pufferlab.contracts.errors import ApiErrorCode


class EvaluationViewError(RuntimeError):
    """A direct public-safe failure that never retains an internal exception."""

    def __init__(
        self,
        *,
        code: ApiErrorCode,
        message: str,
        http_status: int,
        operation: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.operation = operation
        self.retryable = retryable


def evaluation_not_found(*, message: str, operation: str) -> EvaluationViewError:
    return EvaluationViewError(
        code=ApiErrorCode.NOT_FOUND,
        message=message,
        http_status=404,
        operation=operation,
    )


def evaluation_invalid(*, message: str, operation: str) -> EvaluationViewError:
    return EvaluationViewError(
        code=ApiErrorCode.VALIDATION_ERROR,
        message=message,
        http_status=422,
        operation=operation,
    )


def evaluation_conflict(*, message: str, operation: str) -> EvaluationViewError:
    return EvaluationViewError(
        code=ApiErrorCode.RUN_CONFLICT,
        message=message,
        http_status=409,
        operation=operation,
    )


def evaluation_unavailable(*, operation: str) -> EvaluationViewError:
    return EvaluationViewError(
        code=ApiErrorCode.INTERNAL_ERROR,
        message="stored evaluation data is temporarily unavailable",
        http_status=503,
        operation=operation,
    )
