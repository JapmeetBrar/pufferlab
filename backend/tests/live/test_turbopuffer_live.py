"""Opt-in smoke coverage against a uniquely owned real turbopuffer namespace."""

from __future__ import annotations

import asyncio
import os
import secrets
from time import monotonic
from typing import cast

import pytest
from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderQueryResult, ProviderSchema, WriteDocument
from pufferlab.retrieval.rrf import RrfEntry, reconstruct_rrf

_LIVE_NAMESPACE_PREFIX = "pufferlab-live-test-"


def _unit_vector(dimensions: int, hot_dimension: int) -> list[float]:
    return [float(position == hot_dimension) for position in range(dimensions)]


def _assert_tie_safe_rrf_parity(
    server: ProviderQueryResult,
    reconstructed: tuple[RrfEntry, ...],
) -> None:
    expected = reconstructed[: len(server.documents)]
    start = 0
    while start < len(expected):
        end = start + 1
        while end < len(expected) and expected[end].score == expected[start].score:
            end += 1
        assert {document.id for document in server.documents[start:end]} == {
            entry.document_id for entry in expected[start:end]
        }
        start = end

    score_by_id = {entry.document_id: entry.score for entry in expected}
    assert {document.id for document in server.documents} == set(score_by_id)
    for document in server.documents:
        assert document.score.value == pytest.approx(score_by_id[document.id])


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_write_bm25_ann_server_rrf_parity_and_exact_cleanup() -> None:
    if os.environ.get("PUFFERLAB_RUN_LIVE") != "1":
        pytest.skip("set PUFFERLAB_RUN_LIVE=1 to run live turbopuffer tests")
    settings = Settings()
    secret = settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        pytest.fail("TURBOPUFFER_API_KEY is required in the environment or ignored .env")

    api_key = secret.get_secret_value()
    region = settings.turbopuffer_region
    namespace_id = f"{_LIVE_NAMESPACE_PREFIX}{secrets.token_hex(12)}"
    created_namespace_id: str | None = None
    provider = TurbopufferProvider(api_key=api_key, region=region)
    puffer_vector = _unit_vector(2, 0)
    walrus_vector = _unit_vector(2, 1)
    hybrid_vector = [
        (left + right) / 2 for left, right in zip(puffer_vector, walrus_vector, strict=True)
    ]
    full_text_search = {
        "tokenizer": "word_v4",
        "case_sensitive": False,
        "language": "english",
        "stemming": False,
        "remove_stopwords": False,
        "ascii_folding": False,
        "max_token_length": 39,
        "k1": 1.2,
        "b": 0.75,
        "k3": 8.0,
    }
    schema = cast(
        ProviderSchema,
        {
            "title": {
                "type": "string",
                "filterable": False,
                "full_text_search": full_text_search,
            },
            "body": {
                "type": "string",
                "filterable": False,
                "full_text_search": full_text_search,
            },
            "vector": {"type": "[2]f32", "ann": True, "filterable": False},
        },
    )

    try:
        write_result = await provider.write_documents(
            namespace=namespace_id,
            documents=(
                WriteDocument(
                    id="puffer-doc",
                    attributes={
                        "title": "Pufferfish storage",
                        "body": "A pufferfish search engine stores indexes on object storage.",
                        "vector": puffer_vector,
                    },
                ),
                WriteDocument(
                    id="walrus-doc",
                    attributes={
                        "title": "Walrus notes",
                        "body": "A walrus rests near the cold ocean.",
                        "vector": walrus_vector,
                    },
                ),
                WriteDocument(
                    id="hybrid-doc",
                    attributes={
                        "title": "Search comparison",
                        "body": "Compare lexical and vector search for pufferfish queries.",
                        "vector": hybrid_vector,
                    },
                ),
            ),
            schema=schema,
            distance_metric="cosine_distance",
        )
        created_namespace_id = namespace_id
        assert write_result.rows_affected == 3

        deadline = monotonic() + 90.0
        while True:
            metadata = await provider.namespace_metadata(namespace_id)
            if metadata.ready:
                break
            if monotonic() >= deadline:
                pytest.fail("live namespace index did not become ready before timeout")
            await asyncio.sleep(0.5)

        bm25 = await provider.query_bm25(
            namespace=namespace_id,
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="pufferfish search",
            top_k=3,
            include_attributes=("title", "body"),
        )
        ann = await provider.query_ann(
            namespace=namespace_id,
            vector_attribute="vector",
            query_vector=puffer_vector,
            top_k=3,
            include_attributes=("title", "body"),
            distance_metric="cosine_distance",
        )
        hybrid = await provider.query_hybrid_rrf(
            namespace=namespace_id,
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="pufferfish search",
            vector_attribute="vector",
            query_vector=puffer_vector,
            candidate_k=3,
            result_k=3,
            include_attributes=("title", "body"),
            rank_constant=60,
            weights=(1.0, 1.0),
            distance_metric="cosine_distance",
        )
        probe = await provider.probe_hybrid_candidates(
            namespace=namespace_id,
            lexical_fields=(("title", 2.0), ("body", 1.0)),
            query_text="pufferfish search",
            vector_attribute="vector",
            query_vector=puffer_vector,
            candidate_k=3,
            include_attributes=("title", "body"),
            distance_metric="cosine_distance",
        )
        reconstructed = reconstruct_rrf(
            (
                tuple(document.id for document in probe.bm25_documents),
                tuple(document.id for document in probe.ann_documents),
            ),
            rank_constant=60,
            weights=(1.0, 1.0),
        )

        assert bm25.documents
        assert ann.documents
        assert bm25.documents[0].score.direction.value == "higher_is_better"
        assert ann.documents[0].score.direction.value == "lower_is_better"
        _assert_tie_safe_rrf_parity(hybrid, reconstructed)
        assert all(document.score.kind.value == "rrf" for document in hybrid.documents)
        assert bm25.client_duration_ms >= 0
        assert ann.client_duration_ms >= 0
        assert hybrid.client_duration_ms >= 0
        assert probe.client_duration_ms >= 0
    finally:
        try:
            if created_namespace_id is not None:
                if not created_namespace_id.startswith(_LIVE_NAMESPACE_PREFIX):
                    pytest.fail("refusing to delete a namespace outside the live-test prefix")
                try:
                    await provider.delete_namespace(created_namespace_id)
                except ProviderError as error:
                    if error.details.code is not ApiErrorCode.NOT_FOUND:
                        raise
                for attempt in range(30):
                    try:
                        await provider.namespace_metadata(created_namespace_id)
                    except ProviderError as error:
                        if error.details.code is ApiErrorCode.NOT_FOUND:
                            break
                        raise
                    if attempt + 1 < 30:
                        await asyncio.sleep(0.5)
                else:
                    pytest.fail("live namespace cleanup was not confirmed as not found")
        finally:
            await provider.close()
