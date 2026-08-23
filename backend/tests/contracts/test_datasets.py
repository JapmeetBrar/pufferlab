from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pufferlab.contracts.catalog import (
    DatasetCatalogItem,
    DatasetDetailResponse,
    DatasetListResponse,
)
from pufferlab.contracts.datasets import (
    DataOrigin,
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pydantic import ValidationError


def make_index_profile(distance_metric: str) -> IndexProfile:
    return IndexProfile.model_validate(
        {
            "id": "tiny-v1",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "main",
            "vector_dimensions": 384,
            "vector_dtype": "f32",
            "distance_metric": distance_metric,
            "fts_profile": {},
            "schema_hash": "schema-hash",
        }
    )


@pytest.mark.parametrize("distance_metric", ["cosine_distance", "euclidean_squared"])
def test_index_profile_accepts_supported_dense_metrics(distance_metric: str) -> None:
    profile = make_index_profile(distance_metric)

    assert profile.distance_metric == distance_metric


def test_index_profile_rejects_dot_product() -> None:
    with pytest.raises(ValidationError, match="distance_metric"):
        make_index_profile("dot_product")


def test_fts_profile_accepts_max_token_length_255() -> None:
    profile = FtsProfile(max_token_length=255)

    assert profile.max_token_length == 255


def test_fts_profile_rejects_max_token_length_256() -> None:
    with pytest.raises(ValidationError, match="max_token_length"):
        FtsProfile(max_token_length=256)


def _dataset(*, origin: DataOrigin, namespace: str) -> DatasetVersion:
    return DatasetVersion(
        id=uuid4(),
        slug="contract-dataset",
        version="v1",
        data_origin=origin,
        namespace=namespace,
        index_profile=make_index_profile("cosine_distance"),
        document_count=50,
        corpus_hash="corpus-hash",
        status=DatasetStatus.READY,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
    )


def test_live_dataset_default_preserves_old_m2_canonical_json() -> None:
    old_payload = _dataset(origin=DataOrigin.LIVE, namespace="pufferlab-live").model_dump(
        mode="json"
    )

    assert "data_origin" not in old_payload
    restored = DatasetVersion.model_validate(old_payload)
    assert restored.data_origin is DataOrigin.LIVE
    assert restored.model_dump(mode="json") == old_payload


def test_synthetic_dataset_is_explicit_and_cannot_claim_a_namespace() -> None:
    synthetic = _dataset(origin=DataOrigin.SYNTHETIC_DEMO, namespace="")

    assert synthetic.model_dump(mode="json")["data_origin"] == "synthetic_demo"
    with pytest.raises(ValidationError, match="cannot claim a provider namespace"):
        _dataset(origin=DataOrigin.SYNTHETIC_DEMO, namespace="fake-live-namespace")
    with pytest.raises(ValidationError, match="require a provider namespace"):
        _dataset(origin=DataOrigin.LIVE, namespace="")


def test_dataset_catalog_propagates_origin_in_every_projection() -> None:
    synthetic = _dataset(origin=DataOrigin.SYNTHETIC_DEMO, namespace="")
    item = DatasetCatalogItem(dataset=synthetic, data_origin=DataOrigin.SYNTHETIC_DEMO)

    listed = DatasetListResponse(datasets=[item])
    detailed = DatasetDetailResponse(
        dataset=synthetic,
        data_origin=DataOrigin.SYNTHETIC_DEMO,
    )

    assert listed.model_dump(mode="json")["datasets"][0]["data_origin"] == "synthetic_demo"
    assert detailed.model_dump(mode="json")["data_origin"] == "synthetic_demo"
    with pytest.raises(ValidationError, match="origin must match"):
        DatasetCatalogItem(dataset=synthetic, data_origin=DataOrigin.LIVE)
