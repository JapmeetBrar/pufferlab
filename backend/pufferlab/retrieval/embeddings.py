"""Lazy, pinned sentence-transformers query embedding boundary."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Protocol, runtime_checkable

from pufferlab.retrieval.types import QueryEmbedding

_RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class _Encoder(Protocol):
    def encode(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


class _ModelFactory(Protocol):
    def __call__(self, model_name_or_path: str, *, revision: str) -> _Encoder: ...


@runtime_checkable
class _ListConvertible(Protocol):
    def tolist(self) -> object: ...


class SentenceTransformerQueryEmbedder:
    """Load one exact model revision on first use and reuse it thereafter."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        model_factory: _ModelFactory | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.model = model
        self.revision = revision
        self.dimensions = dimensions
        self._model_factory = model_factory
        self._clock = clock
        self._encoder: _Encoder | None = None
        self._load_lock = asyncio.Lock()

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        start = self._clock()
        encoder = await self._load_encoder()
        vector = await asyncio.to_thread(
            encoder.encode,
            f"{_RETRIEVAL_QUERY_PREFIX}{query_text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        values = _to_float_tuple(vector)
        return QueryEmbedding(
            vector=values,
            client_duration_ms=max(0.0, (self._clock() - start) * 1000),
        )

    async def _load_encoder(self) -> _Encoder:
        if self._encoder is not None:
            return self._encoder
        async with self._load_lock:
            if self._encoder is None:
                factory = self._model_factory or _load_sentence_transformer_factory()
                self._encoder = await asyncio.to_thread(
                    factory,
                    self.model,
                    revision=self.revision,
                )
        return self._encoder


def _load_sentence_transformer_factory() -> _ModelFactory:
    module = importlib.import_module("sentence_transformers")
    sentence_transformer = module.SentenceTransformer
    if not callable(sentence_transformer):
        raise TypeError("sentence_transformers.SentenceTransformer is not callable")

    def factory(model_name_or_path: str, *, revision: str) -> _Encoder:
        encoder = sentence_transformer(model_name_or_path, revision=revision)
        if not isinstance(encoder, _Encoder):
            raise TypeError("SentenceTransformer does not implement encode")
        return encoder

    return factory


def _to_float_tuple(value: object) -> tuple[float, ...]:
    if isinstance(value, _ListConvertible):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("embedding model returned a non-vector value")
    values = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise TypeError("embedding model returned a non-numeric vector value")
        values.append(float(item))
    return tuple(values)
