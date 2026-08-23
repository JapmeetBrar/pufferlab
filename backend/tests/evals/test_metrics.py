from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from uuid import UUID

import pytest
from pufferlab.evals import EvaluationWarningCode, Judgment, evaluate_ranking


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _reference_ndcg(
    ranking: Sequence[UUID],
    judgments: Sequence[Judgment],
    *,
    cutoff: int = 10,
) -> float:
    """Small exhaustive reference, intentionally independent of production helpers."""

    grades = {judgment.document_id: judgment.relevance_grade for judgment in judgments}

    def dcg(ordered_grades: Sequence[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(ordered_grades[:cutoff], start=1)
        )

    observed = dcg([grades.get(document_id, 0) for document_id in ranking])
    judged_grades = tuple(grades.values())
    ideal = max(dcg(permutation) for permutation in itertools.permutations(judged_grades))
    return observed / ideal


def test_metrics_match_hand_calculated_graded_example() -> None:
    first, second, third, unjudged = (_uuid(value) for value in range(1, 5))
    metrics = evaluate_ranking(
        [second, first, unjudged, third],
        [Judgment(first, 3), Judgment(second, 2), Judgment(third, 1)],
    )

    # DCG = 3 + 7/log2(3) + 1/log2(5); IDCG = 7 + 3/log2(3) + 1/log2(4).
    assert metrics.ndcg_at_10 == pytest.approx(0.8354477690556398)
    assert metrics.recall_at_50 == 1.0
    assert metrics.mrr_at_10 == 1.0
    assert metrics.warnings == ()


def test_ndcg_matches_independent_exhaustive_reference() -> None:
    generator = random.Random(20260822)
    judged_ids = [_uuid(value) for value in range(1, 6)]
    unjudged_ids = [_uuid(value) for value in range(10, 13)]

    for _ in range(25):
        judgments = [Judgment(document_id, generator.randint(1, 4)) for document_id in judged_ids]
        ranking = judged_ids + unjudged_ids
        generator.shuffle(ranking)

        actual = evaluate_ranking(ranking, judgments)

        assert actual.ndcg_at_10 == pytest.approx(_reference_ndcg(ranking, judgments), abs=1e-12)


def test_recall_and_mrr_use_positive_grade_and_their_exact_cutoffs() -> None:
    documents = [_uuid(value) for value in range(1, 53)]
    relevant_at_10 = documents[9]
    relevant_at_51 = documents[50]

    metrics = evaluate_ranking(
        documents,
        [Judgment(relevant_at_10, 1), Judgment(relevant_at_51, 2)],
    )

    assert metrics.recall_at_50 == 0.5
    assert metrics.mrr_at_10 == 0.1

    beyond_mrr = evaluate_ranking(
        documents,
        [Judgment(documents[10], 1)],
    )
    assert beyond_mrr.mrr_at_10 == 0.0


def test_empty_and_missing_results_are_zero_when_positive_qrels_exist() -> None:
    relevant = _uuid(1)

    empty = evaluate_ranking([], [Judgment(relevant, 1)])
    missing = evaluate_ranking([_uuid(2)], [Judgment(relevant, 1)])

    assert (empty.ndcg_at_10, empty.recall_at_50, empty.mrr_at_10) == (0.0, 0.0, 0.0)
    assert (missing.ndcg_at_10, missing.recall_at_50, missing.mrr_at_10) == (
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize(
    "judgments",
    [[], [Judgment(_uuid(1), 0)]],
)
def test_no_positive_qrels_return_null_metrics_with_warning(
    judgments: list[Judgment],
) -> None:
    metrics = evaluate_ranking([_uuid(1)], judgments)

    assert metrics.ndcg_at_10 is None
    assert metrics.recall_at_50 is None
    assert metrics.mrr_at_10 is None
    assert [warning.code for warning in metrics.warnings] == [
        EvaluationWarningCode.NO_POSITIVE_QRELS
    ]


def test_duplicate_retrieval_consumes_rank_without_repeated_credit() -> None:
    first, second, unjudged = _uuid(1), _uuid(2), _uuid(3)

    metrics = evaluate_ranking(
        [unjudged, first, first, second],
        [Judgment(first, 1), Judgment(second, 1)],
    )

    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    expected_ideal = 1 + 1 / math.log2(3)
    assert metrics.ndcg_at_10 == pytest.approx(expected_dcg / expected_ideal)
    assert metrics.recall_at_50 == 1.0
    assert metrics.mrr_at_10 == 0.5
    assert [warning.code for warning in metrics.warnings] == [
        EvaluationWarningCode.DUPLICATE_RETRIEVED_DOCUMENT
    ]


def test_identical_duplicate_qrels_coalesce_and_conflicts_fail_closed() -> None:
    relevant = _uuid(1)

    metrics = evaluate_ranking(
        [relevant],
        [Judgment(relevant, 2), Judgment(relevant, 2), Judgment(relevant, 2)],
    )

    assert metrics.ndcg_at_10 == 1.0
    assert [warning.code for warning in metrics.warnings] == [EvaluationWarningCode.DUPLICATE_QREL]

    with pytest.raises(ValueError, match="conflicting qrels"):
        evaluate_ranking(
            [relevant],
            [Judgment(relevant, 1), Judgment(relevant, 2)],
        )


def test_equal_grade_ties_do_not_depend_on_document_identity() -> None:
    first, second = _uuid(1), _uuid(2)
    judgments = [Judgment(first, 2), Judgment(second, 2)]

    first_order = evaluate_ranking([first, second], judgments)
    second_order = evaluate_ranking([second, first], judgments)

    assert first_order == second_order


def test_extreme_positive_grades_still_produce_finite_normalized_ndcg() -> None:
    first, second = _uuid(1), _uuid(2)

    metrics = evaluate_ranking(
        [second, first],
        [Judgment(first, 10**100), Judgment(second, 1)],
    )

    assert metrics.ndcg_at_10 is not None
    assert math.isfinite(metrics.ndcg_at_10)
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0


@pytest.mark.parametrize("grade", [-1, True, 1.5])
def test_judgments_reject_invalid_grades(grade: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Judgment(_uuid(1), grade)  # type: ignore[arg-type]
