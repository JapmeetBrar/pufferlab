import asyncio
import threading
from collections.abc import Iterator

import pytest
from pufferlab.providers.rerankers import RerankCandidate, SentenceTransformersReranker


class FakeScores:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> FakeScores:
        self.calls.append((inputs, kwargs))
        return FakeScores([0.25 + index for index, _ in enumerate(inputs)])


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.encoder = FakeCrossEncoder()

    def __call__(self, model_name_or_path: str, *, revision: str) -> FakeCrossEncoder:
        self.calls.append((model_name_or_path, revision))
        return self.encoder


def _clock(values: list[float]):
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


@pytest.mark.asyncio
async def test_reranker_uses_exact_revision_text_only_inputs_and_cached_model() -> None:
    factory = FakeFactory()
    reranker = SentenceTransformersReranker(
        model="reranker-model",
        revision="exact-revision",
        model_factory=factory,
        clock=_clock([1.0, 1.004, 2.0, 2.006]),
    )
    candidates = (
        RerankCandidate(document_id="one", title="Title one", body="Body one"),
        RerankCandidate(document_id="two", title="", body="Body two"),
    )

    first = await reranker.rerank(query_text="shell pipes", candidates=candidates)
    second = await reranker.rerank(query_text="chmod", candidates=candidates[:1])

    assert factory.calls == [("reranker-model", "exact-revision")]
    assert factory.encoder.calls == [
        (
            [("shell pipes", "Title one\n\nBody one"), ("shell pipes", "Body two")],
            {"show_progress_bar": False, "convert_to_numpy": True},
        ),
        (
            [("chmod", "Title one\n\nBody one")],
            {"show_progress_bar": False, "convert_to_numpy": True},
        ),
    ]
    assert [(score.document_id, score.score) for score in first.scores] == [
        ("one", 0.25),
        ("two", 1.25),
    ]
    assert first.client_duration_ms == pytest.approx(4.0)
    assert second.client_duration_ms == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_reranker_rejects_duplicate_candidates_before_model_load() -> None:
    factory = FakeFactory()
    reranker = SentenceTransformersReranker(
        model="reranker-model",
        revision="exact-revision",
        model_factory=factory,
    )
    duplicate = RerankCandidate(document_id="same", title="Title", body="Body")

    with pytest.raises(ValueError, match="unique"):
        await reranker.rerank(query_text="query", candidates=(duplicate, duplicate))

    assert factory.calls == []


@pytest.mark.asyncio
async def test_cancellation_does_not_allow_concurrent_use_of_cached_model() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    class BlockingCrossEncoder:
        def __init__(self) -> None:
            self._state_lock = threading.Lock()
            self.calls = 0
            self.active = 0
            self.max_active = 0

        def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> FakeScores:
            del kwargs
            with self._state_lock:
                self.calls += 1
                call_number = self.calls
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                if call_number == 1:
                    first_entered.set()
                    if not release_first.wait(timeout=5):
                        raise TimeoutError("test did not release first prediction")
                else:
                    second_entered.set()
                return FakeScores([0.25 for _ in inputs])
            finally:
                with self._state_lock:
                    self.active -= 1

    encoder = BlockingCrossEncoder()
    reranker = SentenceTransformersReranker(
        model="reranker-model",
        revision="exact-revision",
        model_factory=lambda model, *, revision: encoder,
    )
    candidate = RerankCandidate(document_id="one", title="Title", body="Body")
    first = asyncio.create_task(reranker.rerank(query_text="first", candidates=(candidate,)))

    try:
        assert await asyncio.to_thread(first_entered.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(reranker.rerank(query_text="second", candidates=(candidate,)))
        assert not await asyncio.to_thread(second_entered.wait, 0.1)
        release_first.set()
        result = await asyncio.wait_for(second, timeout=2)
    finally:
        release_first.set()

    assert second_entered.is_set()
    assert result.scores[0].score == 0.25
    assert encoder.calls == 2
    assert encoder.max_active == 1
