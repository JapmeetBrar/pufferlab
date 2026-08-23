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
