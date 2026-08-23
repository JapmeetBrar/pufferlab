"""Injected HTTP-facing protocols for evaluation reads and later control runtimes."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pufferlab.contracts.catalog import (
    DatasetDetailResponse,
    DatasetListResponse,
    QuerySetListResponse,
    RetrievalConfigCatalogResponse,
)
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.evals import (
    CancelEvalRunResponse,
    CreateEvalRunRequest,
    CreateEvalRunResponse,
    EvalRunDetailResponse,
    EvalRunExportResponse,
    EvalRunListQuery,
    EvalRunListResponse,
    EvalRunQueryDetailResponse,
    RegressionQuery,
    RegressionResponse,
)
from pufferlab.contracts.forensics import (
    EvalRunQueryReplayRequest,
    EvalRunQueryReplayResponse,
)


class EvaluationViewFacade(Protocol):
    def list_datasets(self) -> DatasetListResponse: ...

    def get_dataset(self, dataset_version_id: UUID) -> DatasetDetailResponse: ...

    def list_query_sets(self, dataset_version_id: UUID) -> QuerySetListResponse: ...

    def list_dataset_configs(
        self,
        dataset_version_id: UUID,
    ) -> RetrievalConfigCatalogResponse: ...

    def list_eval_runs(self, query: EvalRunListQuery) -> EvalRunListResponse: ...

    def get_eval_run(self, run_id: UUID) -> EvalRunDetailResponse: ...

    def get_regressions(self, run_id: UUID, query: RegressionQuery) -> RegressionResponse: ...

    def get_query_detail(self, run_id: UUID, query_id: UUID) -> EvalRunQueryDetailResponse: ...

    def export_eval_run(self, run_id: UUID) -> EvalRunExportResponse: ...

    def query_set_data_origin(self, query_set_id: UUID) -> DataOrigin: ...


class EvaluationControlFacade(Protocol):
    async def create_eval_run(self, request: CreateEvalRunRequest) -> CreateEvalRunResponse: ...

    async def cancel_eval_run(self, run_id: UUID) -> CancelEvalRunResponse: ...

    async def replay_eval_query(
        self,
        run_id: UUID,
        query_id: UUID,
        request: EvalRunQueryReplayRequest,
    ) -> EvalRunQueryReplayResponse: ...
