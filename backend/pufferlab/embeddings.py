"""Shared lazy loading for the optional sentence-transformers runtime."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


class SentenceEncoderFactory(Protocol):
    def __call__(self, model_name_or_path: str, *, revision: str) -> SentenceEncoder: ...


@runtime_checkable
class ListConvertible(Protocol):
    def tolist(self) -> object: ...


class LazySentenceTransformer:
    """Load one exact model revision on first use and reuse it across embedding calls."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        model_factory: SentenceEncoderFactory | None = None,
    ) -> None:
        self.model = model
        self.revision = revision
        self._model_factory = model_factory
        self._encoder: SentenceEncoder | None = None
        self._load_lock = asyncio.Lock()
        self._encode_lock = asyncio.Lock()

    async def encode(
        self,
        sentences: str | Sequence[str],
        *,
        batch_size: int,
    ) -> object:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        encoder = await self._load_encoder()
        async with self._encode_lock:
            return await asyncio.to_thread(
                encoder.encode,
                sentences,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    async def _load_encoder(self) -> SentenceEncoder:
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


def to_float_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, ListConvertible):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("embedding model returned a non-vector value")
    values = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise TypeError("embedding model returned a non-numeric vector value")
        values.append(float(item))
    return tuple(values)


def to_float_matrix(value: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, ListConvertible):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("embedding model returned a non-matrix value")
    return tuple(to_float_vector(row) for row in value)


def _load_sentence_transformer_factory() -> SentenceEncoderFactory:
    module = importlib.import_module("sentence_transformers")
    sentence_transformer = module.SentenceTransformer
    if not callable(sentence_transformer):
        raise TypeError("sentence_transformers.SentenceTransformer is not callable")

    def factory(model_name_or_path: str, *, revision: str) -> SentenceEncoder:
        encoder = sentence_transformer(model_name_or_path, revision=revision)
        if not isinstance(encoder, SentenceEncoder):
            raise TypeError("SentenceTransformer does not implement encode")
        return encoder

    return factory
