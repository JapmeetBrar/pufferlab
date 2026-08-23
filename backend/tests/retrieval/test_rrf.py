import pytest
from pufferlab.retrieval.rrf import reconstruct_rrf


def test_weighted_rrf_matches_documented_hand_calculation_and_is_deterministic() -> None:
    result = reconstruct_rrf(
        (("a", "b", "c"), ("b", "d", "a")),
        rank_constant=10,
        weights=(2.0, 1.0),
    )

    assert [entry.document_id for entry in result] == ["a", "b", "c", "d"]
    by_id = {entry.document_id: entry for entry in result}
    assert by_id["a"].score == pytest.approx(2 / 11 + 1 / 13)
    assert by_id["a"].source_ranks == (1, 3)
    assert by_id["b"].score == pytest.approx(2 / 12 + 1 / 11)
    assert by_id["b"].source_ranks == (2, 1)
    assert by_id["c"].score == pytest.approx(2 / 13)
    assert by_id["d"].score == pytest.approx(1 / 12)
    assert result == reconstruct_rrf(
        (("a", "b", "c"), ("b", "d", "a")),
        rank_constant=10,
        weights=(2.0, 1.0),
    )


@pytest.mark.parametrize(
    ("ranked_lists", "rank_constant", "weights", "message"),
    [
        ((("a",),), 60, (1.0,), "at least two"),
        ((("a",), ("b",)), 60, (1.0,), "one weight"),
        ((("a",), ("b",)), 0, (1.0, 1.0), "positive"),
        ((("a",), ("b",)), 60, (1.0, float("nan")), "finite"),
        ((("a", "a"), ("b",)), 60, (1.0, 1.0), "duplicate"),
    ],
)
def test_rrf_rejects_invalid_inputs(
    ranked_lists: tuple[tuple[str, ...], ...],
    rank_constant: int,
    weights: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reconstruct_rrf(
            ranked_lists,
            rank_constant=rank_constant,
            weights=weights,
        )
