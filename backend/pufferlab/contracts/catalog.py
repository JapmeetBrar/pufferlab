"""Versioned provider-free dataset and query-set catalog responses."""

from uuid import UUID

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel, ContractVersion
from pufferlab.contracts.datasets import DataOrigin, DatasetVersion
from pufferlab.contracts.evals import QuerySet
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode

_CANONICAL_QUERY_COUNT = 50
_CANONICAL_CONFIG_MODES = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)


class DatasetCatalogItem(ContractModel):
    dataset: DatasetVersion
    data_origin: DataOrigin

    @model_validator(mode="after")
    def validate_origin(self) -> "DatasetCatalogItem":
        if self.data_origin is not self.dataset.data_origin:
            raise ValueError("catalog origin must match the dataset revision")
        return self


class DatasetListResponse(ContractModel):
    contract_version: ContractVersion = 1
    datasets: list[DatasetCatalogItem] = Field(max_length=100)


class DatasetDetailResponse(ContractModel):
    contract_version: ContractVersion = 1
    dataset: DatasetVersion
    data_origin: DataOrigin

    @model_validator(mode="after")
    def validate_origin(self) -> "DatasetDetailResponse":
        if self.data_origin is not self.dataset.data_origin:
            raise ValueError("detail origin must match the dataset revision")
        return self


class QuerySetCatalogItem(ContractModel):
    query_set: QuerySet
    data_origin: DataOrigin

    @model_validator(mode="after")
    def validate_canonical_query_count(self) -> "QuerySetCatalogItem":
        if self.query_set.query_count != _CANONICAL_QUERY_COUNT:
            raise ValueError("P0 catalog query sets must contain exactly 50 queries")
        return self


class QuerySetListResponse(ContractModel):
    contract_version: ContractVersion = 1
    dataset_version_id: UUID
    query_sets: list[QuerySetCatalogItem] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_dataset_scope(self) -> "QuerySetListResponse":
        if any(
            item.query_set.dataset_version_id != self.dataset_version_id for item in self.query_sets
        ):
            raise ValueError("every query set must belong to the requested dataset revision")
        return self


class RetrievalConfigCatalogResponse(ContractModel):
    contract_version: ContractVersion = 1
    dataset_version_id: UUID
    data_origin: DataOrigin
    configs: list[RetrievalConfigSummary] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_canonical_config_order(self) -> "RetrievalConfigCatalogResponse":
        if tuple(config.mode for config in self.configs) != _CANONICAL_CONFIG_MODES:
            raise ValueError("P0 configs must use BM25, ANN, RRF, reranker contract order")
        config_ids = [config.id for config in self.configs]
        if len(config_ids) != len(set(config_ids)):
            raise ValueError("P0 configs must have distinct immutable IDs")
        return self
