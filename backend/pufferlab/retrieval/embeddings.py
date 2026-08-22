"""Pinned BGE query embedding boundary."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from pufferlab.embeddings import (
    LazySentenceTransformer,
    SentenceEncoderFactory,
    to_float_vector,
)
from pufferlab.retrieval.types import QueryEmbedding

_RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerQueryEmbedder:
    """Load one exact model revision on first use and reuse it thereafter."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        model_factory: SentenceEncoderFactory | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.model = model
        self.revision = revision
        self.dimensions = dimensions
        self._clock = clock
        self._model = LazySentenceTransformer(
            model=model,
            revision=revision,
            model_factory=model_factory,
        )

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        start = self._clock()
        vector = await self._model.encode(
            f"{_RETRIEVAL_QUERY_PREFIX}{query_text}",
            batch_size=1,
        )
        values = to_float_vector(vector)
        return QueryEmbedding(
            vector=values,
            client_duration_ms=max(0.0, (self._clock() - start) * 1000),
        )
