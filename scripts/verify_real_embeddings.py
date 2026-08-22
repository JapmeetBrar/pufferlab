"""Load the exact fixture model and verify real query/passage embeddings safely."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Final

from pufferlab.datasets.embeddings import SentenceTransformerDocumentEmbedder
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.retrieval.embeddings import SentenceTransformerQueryEmbedder

_ROOT: Final = Path(__file__).parents[1]
_MODEL: Final = "BAAI/bge-small-en-v1.5"
_REVISION: Final = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_DIMENSIONS: Final = 384
_QUERY_TEXT: Final = "How can I find the program listening on port 8080?"
_PASSAGE_IDS: Final = ("tiny-002", "tiny-005")


def _validated_norm(values: object, *, label: str) -> float:
    if not isinstance(values, list | tuple):
        raise RuntimeError(f"{label} embedding is not a vector")
    if len(values) != _DIMENSIONS:
        raise RuntimeError(f"{label} embedding has the wrong dimensions")
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"{label} embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in numeric))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"{label} embedding is not unit-normalized")
    return norm


async def _verify() -> None:
    corpus = load_fixture_corpus(_ROOT / "fixtures" / "tiny-corpus")
    manifest = corpus.manifest
    if (
        manifest.embedding.model != _MODEL
        or manifest.embedding.revision != _REVISION
        or manifest.embedding.dimensions != _DIMENSIONS
    ):
        raise RuntimeError("fixture embedding manifest no longer matches the verifier")

    documents = {document.external_id: document for document in corpus.documents}
    passage_texts = tuple(
        f"{documents[external_id].title}\n\n{documents[external_id].body}"
        for external_id in _PASSAGE_IDS
    )
    query_embedder = SentenceTransformerQueryEmbedder(
        model=_MODEL,
        revision=_REVISION,
        dimensions=_DIMENSIONS,
    )
    document_embedder = SentenceTransformerDocumentEmbedder(
        model=_MODEL,
        revision=_REVISION,
        dimensions=_DIMENSIONS,
        batch_size=len(passage_texts),
    )

    query = await query_embedder.embed_query(_QUERY_TEXT)
    passages = await document_embedder.embed(passage_texts)
    query_norm = _validated_norm(query.vector, label="query")
    passage_norms = tuple(
        _validated_norm(vector, label=f"passage {external_id}")
        for external_id, vector in zip(_PASSAGE_IDS, passages, strict=True)
    )

    print("real_embedding_verification=passed")
    print(f"model={_MODEL}")
    print(f"revision={_REVISION}")
    print(f"dimensions={_DIMENSIONS}")
    print(f"query_norm={query_norm:.6f}")
    print("passage_norms=" + ",".join(f"{norm:.6f}" for norm in passage_norms))


def main() -> None:
    asyncio.run(_verify())


if __name__ == "__main__":
    main()
