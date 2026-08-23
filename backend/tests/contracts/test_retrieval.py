from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pufferlab.contracts.catalog import RetrievalConfigCatalogResponse
from pufferlab.contracts.datasets import DataOrigin
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
)
from pydantic import ValidationError


def test_retrieval_mode_rejects_incompatible_specs() -> None:
    with pytest.raises(ValidationError, match="do not match mode"):
        RetrievalConfig(
            id=uuid4(),
            revision=1,
            name="bad vector config",
            dataset_version_id=uuid4(),
            mode=RetrievalMode.VECTOR,
            lexical=LexicalSpec(),
            config_hash="hash",
            created_at=datetime.now(UTC),
        )


def test_config_catalog_exposes_dataset_scope_and_data_origin() -> None:
    dataset_id = uuid4()
    response = RetrievalConfigCatalogResponse(
        dataset_version_id=dataset_id,
        data_origin=DataOrigin.SYNTHETIC_DEMO,
        configs=[
            RetrievalConfigSummary(
                id=uuid4(),
                revision=1,
                name=f"Synthetic {mode.value}",
                mode=mode,
                config_hash=str(index) * 64,
            )
            for index, mode in enumerate(
                (
                    RetrievalMode.BM25,
                    RetrievalMode.VECTOR,
                    RetrievalMode.HYBRID_RRF,
                    RetrievalMode.HYBRID_RERANK,
                ),
                start=1,
            )
        ],
    )

    payload = response.model_dump(mode="json")
    assert payload["contract_version"] == 1
    assert payload["dataset_version_id"] == str(dataset_id)
    assert payload["data_origin"] == "synthetic_demo"
    with pytest.raises(ValidationError, match="contract order"):
        RetrievalConfigCatalogResponse(
            dataset_version_id=dataset_id,
            data_origin=DataOrigin.SYNTHETIC_DEMO,
            configs=list(reversed(response.configs)),
        )
