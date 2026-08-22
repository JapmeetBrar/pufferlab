"""Opt-in smoke coverage against a uniquely owned real turbopuffer namespace."""

from __future__ import annotations

import asyncio
import os
import secrets
from time import monotonic
from typing import cast

import pytest
from pufferlab.providers.turbopuffer import TurbopufferProvider
from pufferlab.providers.types import ProviderSchema, WriteDocument

_LIVE_NAMESPACE_PREFIX = "pufferlab-live-test-"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_write_bm25_ann_and_exact_cleanup() -> None:
    if os.environ.get("PUFFERLAB_RUN_LIVE") != "1":
        pytest.skip("set PUFFERLAB_RUN_LIVE=1 to run live turbopuffer tests")
    api_key = os.environ.get("TURBOPUFFER_API_KEY")
    if not api_key:
        pytest.skip("TURBOPUFFER_API_KEY is required for live turbopuffer tests")

    region = os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1")
    namespace_id = f"{_LIVE_NAMESPACE_PREFIX}{secrets.token_hex(12)}"
    created_namespace_id: str | None = None
    provider = TurbopufferProvider(api_key=api_key, region=region)
    schema = cast(
        ProviderSchema,
        {
            "title": {"type": "string", "filterable": False},
            "body": {
                "type": "string",
                "filterable": False,
                "full_text_search": {
                    "tokenizer": "word_v4",
                    "case_sensitive": False,
                    "language": "english",
                    "stemming": False,
                    "remove_stopwords": False,
                    "ascii_folding": False,
                    "max_token_length": 39,
                    "k1": 1.2,
                    "b": 0.75,
                },
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
                        "vector": [1.0, 0.0],
                    },
                ),
                WriteDocument(
                    id="walrus-doc",
                    attributes={
                        "title": "Walrus notes",
                        "body": "A walrus rests near the cold ocean.",
                        "vector": [0.0, 1.0],
                    },
                ),
                WriteDocument(
                    id="hybrid-doc",
                    attributes={
                        "title": "Search comparison",
                        "body": "Compare lexical and vector search for pufferfish queries.",
                        "vector": [0.8, 0.2],
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
            text_attribute="body",
            query_text="pufferfish search",
            top_k=3,
            include_attributes=("title", "body"),
        )
        ann = await provider.query_ann(
            namespace=namespace_id,
            vector_attribute="vector",
            query_vector=(1.0, 0.0),
            top_k=3,
            include_attributes=("title", "body"),
            distance_metric="cosine_distance",
        )

        assert bm25.documents
        assert ann.documents
        assert bm25.documents[0].score.direction.value == "higher_is_better"
        assert ann.documents[0].score.direction.value == "lower_is_better"
        assert bm25.client_duration_ms >= 0
        assert ann.client_duration_ms >= 0
    finally:
        try:
            if created_namespace_id is not None:
                if not created_namespace_id.startswith(_LIVE_NAMESPACE_PREFIX):
                    pytest.fail("refusing to delete a namespace outside the live-test prefix")
                await provider.delete_namespace(created_namespace_id)
        finally:
            await provider.close()
