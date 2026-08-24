"""Provider-free guards for cost-bearing evaluation route signatures."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pufferlab.application.view_errors import EvaluationViewError, evaluation_conflict
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    EvalRunDetailResponse,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
    ExpectedDocumentDiagnosticRequest,
    ExpectedDocumentDiagnosticResponse,
)


class EvaluationOriginViews(Protocol):
    def query_set_data_origin(self, query_set_id: UUID) -> DataOrigin: ...

    def get_eval_run(self, run_id: UUID) -> EvalRunDetailResponse: ...


class ProviderFreeEvaluationControls:
    """Reject synthetic cost paths directly; leave live controls to M3-D/M3-E."""

    def __init__(self, views: EvaluationOriginViews) -> None:
        self._views = views

    async def create_eval_run(self, request: CreateEvalRunRequest) -> CreateEvalRunResponse:
        if self._views.query_set_data_origin(request.query_set_id) is DataOrigin.SYNTHETIC_DEMO:
            raise self._synthetic_read_only("create_eval_run")
        raise self._control_unavailable("create_eval_run")

    async def cancel_eval_run(self, run_id: UUID) -> CancelEvalRunResponse:
        detail = self._views.get_eval_run(run_id)
        if detail.result.data_origin is DataOrigin.SYNTHETIC_DEMO:
            raise self._synthetic_read_only("cancel_eval_run")
        raise self._control_unavailable("cancel_eval_run")

    async def replay_eval_query(
        self,
        run_id: UUID,
        query_id: UUID,
        request: EvalRunQueryReplayRequest,
    ) -> EvalRunQueryReplayResponse:
        del query_id, request
        detail = self._views.get_eval_run(run_id)
        if detail.result.data_origin is DataOrigin.SYNTHETIC_DEMO:
            raise self._synthetic_read_only("replay_eval_query")
        raise self._control_unavailable("replay_eval_query")

    async def diagnose_expected_document(
        self,
        run_id: UUID,
        query_id: UUID,
        document_id: UUID,
        request: ExpectedDocumentDiagnosticRequest,
    ) -> ExpectedDocumentDiagnosticResponse:
        del query_id, document_id, request
        detail = self._views.get_eval_run(run_id)
        if detail.result.data_origin is DataOrigin.SYNTHETIC_DEMO:
            raise self._synthetic_read_only("diagnose_expected_document")
        raise self._control_unavailable("diagnose_expected_document")

    @staticmethod
    def _synthetic_read_only(operation: str) -> EvaluationViewError:
        return evaluation_conflict(
            message="synthetic demo evaluations are read/export-only",
            operation=operation,
        )

    @staticmethod
    def _control_unavailable(operation: str) -> EvaluationViewError:
        return EvaluationViewError(
            code=ApiErrorCode.INTERNAL_ERROR,
            message="evaluation control runtime is not available",
            http_status=503,
            operation=operation,
        )
