import pytest
from pufferlab.contracts.datasets import FtsProfile, IndexProfile
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
