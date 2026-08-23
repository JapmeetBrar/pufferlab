"""Pure reciprocal-rank fusion reconstruction for observed debug lists."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pufferlab.providers.types import DocumentId


@dataclass(frozen=True, slots=True)
class RrfEntry:
    document_id: DocumentId
    score: float
    source_ranks: tuple[int | None, ...]


def reconstruct_rrf(
    ranked_document_ids: Sequence[Sequence[DocumentId]],
    *,
    rank_constant: int,
    weights: Sequence[float],
) -> tuple[RrfEntry, ...]:
    """Reconstruct weighted RRF from 1-based ranks.

    The score is ``sum(weight / (rank_constant + rank))`` for every source list in
    which a document appears. Ties use source rank and then document ID only to make
    local debug output deterministic; turbopuffer does not document a server tie-break.
    """

    if len(ranked_document_ids) < 2:
        raise ValueError("RRF requires at least two ranked lists")
    if len(weights) != len(ranked_document_ids):
        raise ValueError("RRF requires one weight per ranked list")
    if isinstance(rank_constant, bool) or rank_constant < 1:
        raise ValueError("RRF rank_constant must be positive")
    if any(
        isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0 for weight in weights
    ):
        raise ValueError("RRF weights must be finite and positive")

    ranks_by_document: dict[DocumentId, list[int | None]] = {}
    for source_index, document_ids in enumerate(ranked_document_ids):
        seen: set[DocumentId] = set()
        for rank, document_id in enumerate(document_ids, start=1):
            if isinstance(document_id, bool) or not isinstance(document_id, str | int):
                raise ValueError("RRF document IDs must be strings or integers")
            if document_id in seen:
                raise ValueError("RRF source lists must not contain duplicate document IDs")
            seen.add(document_id)
            ranks = ranks_by_document.setdefault(
                document_id,
                [None for _ in ranked_document_ids],
            )
            ranks[source_index] = rank

    entries = [
        RrfEntry(
            document_id=document_id,
            score=sum(
                weight / (rank_constant + rank)
                for weight, rank in zip(weights, ranks, strict=True)
                if rank is not None
            ),
            source_ranks=tuple(ranks),
        )
        for document_id, ranks in ranks_by_document.items()
    ]
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                -entry.score,
                min(rank for rank in entry.source_ranks if rank is not None),
                str(entry.document_id),
            ),
        )
    )
