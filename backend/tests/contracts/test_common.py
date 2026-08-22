import pytest
from pufferlab.contracts.common import ObservedScore, ScoreDirection, ScoreKind, ScoreSource
from pydantic import ValidationError


def test_vector_distance_requires_lower_is_better() -> None:
    with pytest.raises(ValidationError, match="lower_is_better"):
        ObservedScore(
            kind=ScoreKind.VECTOR_DISTANCE,
            value=0.2,
            direction=ScoreDirection.HIGHER_IS_BETTER,
            source=ScoreSource.TURBOPUFFER_DIST,
        )


def test_bm25_requires_higher_is_better() -> None:
    score = ObservedScore(
        kind=ScoreKind.BM25,
        value=4.2,
        direction=ScoreDirection.HIGHER_IS_BETTER,
        source=ScoreSource.COMPUTE_ATTRIBUTE,
    )

    assert score.value == 4.2
