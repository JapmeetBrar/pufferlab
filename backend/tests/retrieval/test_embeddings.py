from collections.abc import Iterator, Sequence

import pytest
from pufferlab.datasets.embeddings import SentenceTransformerDocumentEmbedder
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PREFIX = "Represent this sentence for searching relevant passages: "


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool, bool]] = []

    def encode(
        self,
        sentences: str,
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append((sentences, batch_size, normalize_embeddings, show_progress_bar))
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
        (f"{PREFIX}how do pipes work", 1, True, False),
        (f"{PREFIX}what is chmod", 1, True, False),
    ]
    assert first.vector == second.vector == (0.5, -0.25, 0.125)
    assert first.client_duration_ms == pytest.approx(12.0)
    assert second.client_duration_ms == pytest.approx(4.0)


class FakePassageEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int, bool, bool]] = []

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        assert not isinstance(sentences, str)
        texts = tuple(sentences)
        self.calls.append((texts, batch_size, normalize_embeddings, show_progress_bar))
        return [[float(index), 0.5, -0.25] for index, _ in enumerate(texts)]


class FakePassageFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.encoder = FakePassageEncoder()

    def __call__(self, model_name_or_path: str, *, revision: str) -> FakePassageEncoder:
        self.calls.append((model_name_or_path, revision))
        return self.encoder


@pytest.mark.asyncio
async def test_document_embedder_uses_exact_revision_unprefixed_batches_and_normalization() -> None:
    factory = FakePassageFactory()
    embedder = SentenceTransformerDocumentEmbedder(
        model=MODEL,
        revision=REVISION,
        dimensions=3,
        batch_size=2,
        model_factory=factory,
    )

    first = await embedder.embed(("Title one\n\nPassage one", "Title two\n\nPassage two"))
    second = await embedder.embed(("Another passage",))

    assert factory.calls == [(MODEL, REVISION)]
    assert factory.encoder.calls == [
        (("Title one\n\nPassage one", "Title two\n\nPassage two"), 2, True, False),
        (("Another passage",), 2, True, False),
    ]
    assert first == ((0.0, 0.5, -0.25), (1.0, 0.5, -0.25))
    assert second == ((0.0, 0.5, -0.25),)
    assert embedder.dimensions == 3
