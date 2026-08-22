"""Pinned BGE passage embedding boundary for dataset ingestion."""

from collections.abc import Sequence

from pufferlab.embeddings import (
    LazySentenceTransformer,
    SentenceEncoderFactory,
    to_float_matrix,
)


class SentenceTransformerDocumentEmbedder:
    """Embed unprefixed BGE passages in normalized, model-native batches."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        batch_size: int = 32,
        model_factory: SentenceEncoderFactory | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.revision = revision
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._model = LazySentenceTransformer(
            model=model,
            revision=revision,
            model_factory=model_factory,
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        matrix = await self._model.encode(
            tuple(texts),
            batch_size=self.batch_size,
        )
        return to_float_matrix(matrix)
