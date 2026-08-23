"""Pinned, text-only local cross-encoder reranker adapter."""

from __future__ import annotations

import asyncio
import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from pufferlab.providers.types import DocumentId

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    document_id: DocumentId
    title: str
    body: str


@dataclass(frozen=True, slots=True)
class RerankScore:
    document_id: DocumentId
    score: float


@dataclass(frozen=True, slots=True)
class RerankResult:
    scores: tuple[RerankScore, ...]
    client_duration_ms: float


class Reranker(Protocol):
    model: str
    revision: str

    async def rerank(
        self,
        *,
        query_text: str,
        candidates: Sequence[RerankCandidate],
    ) -> RerankResult: ...


@runtime_checkable
class _CrossEncoder(Protocol):
    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object: ...


class CrossEncoderFactory(Protocol):
    def __call__(self, model_name_or_path: str, *, revision: str) -> _CrossEncoder: ...


class SentenceTransformersReranker:
    """Load one exact model revision lazily and reuse it for text-only scoring."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_RERANKER_MODEL,
        revision: str = DEFAULT_RERANKER_REVISION,
        model_factory: CrossEncoderFactory | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not model or not revision:
            raise ValueError("reranker model and revision must not be empty")
        self.model = model
        self.revision = revision
        self._model_factory = model_factory or _load_cross_encoder
        self._model: _CrossEncoder | None = None
        self._lock = asyncio.Lock()
        self._clock = clock

    async def rerank(
        self,
        *,
        query_text: str,
        candidates: Sequence[RerankCandidate],
    ) -> RerankResult:
        if not query_text:
            raise ValueError("reranker query must not be empty")
        if len({candidate.document_id for candidate in candidates}) != len(candidates):
            raise ValueError("reranker candidates must have unique document IDs")

        start = self._clock()
        async with self._lock:
            values = await asyncio.to_thread(self._predict, query_text, tuple(candidates))
        duration_ms = max(0.0, (self._clock() - start) * 1000.0)
        scores = tuple(
            RerankScore(document_id=candidate.document_id, score=value)
            for candidate, value in zip(candidates, values, strict=True)
        )
        return RerankResult(scores=scores, client_duration_ms=duration_ms)

    def _predict(
        self,
        query_text: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[float, ...]:
        if not candidates:
            return ()
        if self._model is None:
            self._model = self._model_factory(self.model, revision=self.revision)
        pairs = [(query_text, _candidate_text(candidate)) for candidate in candidates]
        raw_scores = self._model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
            raise ValueError("reranker returned an invalid score collection")
        scores: list[float] = []
        for value in values:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError("reranker returned a non-scalar score")
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("reranker returned a non-finite score")
            scores.append(score)
        if len(scores) != len(candidates):
            raise ValueError("reranker returned the wrong number of scores")
        return tuple(scores)


def _candidate_text(candidate: RerankCandidate) -> str:
    return f"{candidate.title}\n\n{candidate.body}" if candidate.title else candidate.body


def _load_cross_encoder(model_name_or_path: str, *, revision: str) -> _CrossEncoder:
    module = importlib.import_module("sentence_transformers")
    cross_encoder = module.CrossEncoder
    if not callable(cross_encoder):
        raise TypeError("sentence_transformers.CrossEncoder is not callable")
    model = cross_encoder(model_name_or_path, revision=revision)
    if not isinstance(model, _CrossEncoder):
        raise TypeError("CrossEncoder does not implement predict")
    return model
