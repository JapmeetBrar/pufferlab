from collections.abc import Iterator

import pytest
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PREFIX = "Represent this sentence for searching relevant passages: "


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def encode(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append((sentences, normalize_embeddings, show_progress_bar))
        return [0.5, -0.25, 0.125]


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.encoder = FakeEncoder()

    def __call__(self, model_name_or_path: str, *, revision: str) -> FakeEncoder:
        self.calls.append((model_name_or_path, revision))
        return self.encoder


def _clock(values: list[float]):
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


@pytest.mark.asyncio
async def test_embedder_uses_exact_revision_retrieval_prefix_and_normalization() -> None:
    factory = FakeFactory()
    embedder = SentenceTransformerQueryEmbedder(
        model=MODEL,
        revision=REVISION,
        dimensions=3,
        model_factory=factory,
        clock=_clock([5.0, 5.012, 6.0, 6.004]),
    )

    first = await embedder.embed_query("how do pipes work")
    second = await embedder.embed_query("what is chmod")

    assert factory.calls == [(MODEL, REVISION)]
    assert factory.encoder.calls == [
        (f"{PREFIX}how do pipes work", True, False),
        (f"{PREFIX}what is chmod", True, False),
    ]
    assert first.vector == second.vector == (0.5, -0.25, 0.125)
    assert first.client_duration_ms == pytest.approx(12.0)
    assert second.client_duration_ms == pytest.approx(4.0)
