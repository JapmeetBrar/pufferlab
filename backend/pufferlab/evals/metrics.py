"""Auditable per-query information-retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID

from pufferlab.evals.models import (
    EvaluationWarning,
    EvaluationWarningCode,
    Judgment,
    QueryMetrics,
)

NDCG_CUTOFF = 10
RECALL_CUTOFF = 50
MRR_CUTOFF = 10


def _scaled_exponential_gain(grade: int, maximum_grade: int) -> float:
    """Return ``(2**grade - 1) / 2**maximum_grade`` without overflowing."""

    scaled_power = 0.0 if maximum_grade - grade > 1074 else 2.0 ** (grade - maximum_grade)
    scaled_offset = 0.0 if maximum_grade > 1074 else 2.0**-maximum_grade
    return scaled_power - scaled_offset


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _normalize_judgments(
    judgments: Sequence[Judgment],
) -> tuple[dict[UUID, int], list[EvaluationWarning]]:
    grades: dict[UUID, int] = {}
    warnings: list[EvaluationWarning] = []
    warned_ids: set[UUID] = set()

    for judgment in judgments:
        prior_grade = grades.get(judgment.document_id)
        if prior_grade is not None and prior_grade != judgment.relevance_grade:
            raise ValueError(
                f"conflicting qrels for document {judgment.document_id}: "
                f"{prior_grade} and {judgment.relevance_grade}"
            )
        if prior_grade is not None and judgment.document_id not in warned_ids:
            warnings.append(
                EvaluationWarning(
                    code=EvaluationWarningCode.DUPLICATE_QREL,
                    message="duplicate identical qrel was coalesced",
                )
            )
            warned_ids.add(judgment.document_id)
        grades[judgment.document_id] = judgment.relevance_grade

    return grades, warnings


def evaluate_ranking(
    ranked_document_ids: Sequence[UUID],
    judgments: Sequence[Judgment],
) -> QueryMetrics:
    """Compute NDCG@10, Recall@50, and MRR@10 for one ordered ranking.

    NDCG uses exponential gain ``2**grade - 1`` and logarithmic discount
    ``1 / log2(rank + 1)``. Recall and reciprocal rank treat every positive grade as relevant.
    Unjudged and grade-zero documents are non-relevant.

    A duplicate retrieved ID consumes its original rank position but receives gain/relevance credit
    only on its first occurrence. This prevents duplicate hits from inflating quality. A query with
    no positive qrels returns three null values and a ``no_positive_qrels`` warning.
    """

    grades, warnings = _normalize_judgments(judgments)
    positive_grades = [grade for grade in grades.values() if grade > 0]
    if not positive_grades:
        warnings.append(
            EvaluationWarning(
                code=EvaluationWarningCode.NO_POSITIVE_QRELS,
                message="quality metrics are undefined because the query has no positive qrels",
            )
        )
        return QueryMetrics(
            ndcg_at_10=None,
            recall_at_50=None,
            mrr_at_10=None,
            warnings=tuple(warnings),
        )

    maximum_grade = max(positive_grades)
    seen: set[UUID] = set()
    duplicate_warning_added = False
    ranked_grades: list[int] = []
    for document_id in ranked_document_ids:
        if document_id in seen:
            ranked_grades.append(0)
            if not duplicate_warning_added:
                warnings.append(
                    EvaluationWarning(
                        code=EvaluationWarningCode.DUPLICATE_RETRIEVED_DOCUMENT,
                        message=(
                            "duplicate retrieved document consumed its rank but received no "
                            "repeated relevance credit"
                        ),
                    )
                )
                duplicate_warning_added = True
            continue
        seen.add(document_id)
        ranked_grades.append(grades.get(document_id, 0))

    dcg = math.fsum(
        _scaled_exponential_gain(grade, maximum_grade) * _discount(rank)
        for rank, grade in enumerate(ranked_grades[:NDCG_CUTOFF], start=1)
        if grade > 0
    )
    ideal_grades = sorted(positive_grades, reverse=True)[:NDCG_CUTOFF]
    ideal_dcg = math.fsum(
        _scaled_exponential_gain(grade, maximum_grade) * _discount(rank)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    relevant_ids = {document_id for document_id, grade in grades.items() if grade > 0}
    retrieved_relevant = {
        document_id
        for document_id in ranked_document_ids[:RECALL_CUTOFF]
        if document_id in relevant_ids
    }
    first_relevant_rank = next(
        (
            rank
            for rank, document_id in enumerate(ranked_document_ids[:MRR_CUTOFF], start=1)
            if document_id in relevant_ids
        ),
        None,
    )

    return QueryMetrics(
        ndcg_at_10=dcg / ideal_dcg,
        recall_at_50=len(retrieved_relevant) / len(relevant_ids),
        mrr_at_10=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        warnings=tuple(warnings),
    )
