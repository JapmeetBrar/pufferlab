"""Deterministic retrieval configurations for one ingested fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid5

from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.providers.types import ConsistencyLevel, DistanceMetric

_CONFIG_ID_NAMESPACE = UUID("224ff18e-4b57-55bb-a6ec-4e40384550da")


@dataclass(frozen=True, slots=True)
class SearchCatalogProfile:
    """Search-relevant fields copied from a validated fixture manifest."""

    dataset_slug: str
    dataset_version: str
    namespace_schema_hash: str
    text_attribute: str
    vector_attribute: str
    embedding_model: str
    embedding_revision: str
    embedding_dimensions: int
    distance_metric: DistanceMetric


@dataclass(frozen=True, slots=True)
class SeededSearchConfig:
    summary: RetrievalConfigSummary
    mode: Literal[RetrievalMode.BM25, RetrievalMode.VECTOR]
    result_k: int
    consistency: ConsistencyLevel
    text_attribute: str | None = None
    vector_attribute: str | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_dimensions: int | None = None
    distance_metric: DistanceMetric | None = None


class SearchConfigCatalog:
    def __init__(self, configs: tuple[SeededSearchConfig, ...]) -> None:
        self._configs = configs
        self._by_id = {config.summary.id: config for config in configs}
        if len(self._by_id) != len(configs):
            raise ValueError("retrieval config identifiers must be unique")

    @property
    def configs(self) -> tuple[SeededSearchConfig, ...]:
        return self._configs

    def get(self, config_id: UUID) -> SeededSearchConfig | None:
        return self._by_id.get(config_id)

    def summaries(self) -> tuple[RetrievalConfigSummary, ...]:
        return tuple(config.summary for config in self._configs)


def build_search_catalog(
    profile: SearchCatalogProfile,
    *,
    result_k: int = 10,
    consistency: ConsistencyLevel = "strong",
) -> SearchConfigCatalog:
    if result_k < 1:
        raise ValueError("result_k must be positive")
    if profile.embedding_dimensions < 1:
        raise ValueError("embedding_dimensions must be positive")

    common = {
        "dataset_slug": profile.dataset_slug,
        "dataset_version": profile.dataset_version,
        "namespace_schema_hash": profile.namespace_schema_hash,
        "result_k": result_k,
        "consistency": consistency,
    }
    bm25_payload = {
        **common,
        "mode": RetrievalMode.BM25.value,
        "text_attribute": profile.text_attribute,
    }
    vector_payload = {
        **common,
        "mode": RetrievalMode.VECTOR.value,
        "vector_attribute": profile.vector_attribute,
        "embedding_model": profile.embedding_model,
        "embedding_revision": profile.embedding_revision,
        "embedding_dimensions": profile.embedding_dimensions,
        "distance_metric": profile.distance_metric,
    }

    bm25 = SeededSearchConfig(
        summary=_summary(
            name=f"BM25 · {profile.text_attribute}",
            mode=RetrievalMode.BM25,
            payload=bm25_payload,
        ),
        mode=RetrievalMode.BM25,
        result_k=result_k,
        consistency=consistency,
        text_attribute=profile.text_attribute,
    )
    vector = SeededSearchConfig(
        summary=_summary(
            name=f"Vector · {profile.embedding_model}",
            mode=RetrievalMode.VECTOR,
            payload=vector_payload,
        ),
        mode=RetrievalMode.VECTOR,
        result_k=result_k,
        consistency=consistency,
        vector_attribute=profile.vector_attribute,
        embedding_model=profile.embedding_model,
        embedding_revision=profile.embedding_revision,
        embedding_dimensions=profile.embedding_dimensions,
        distance_metric=profile.distance_metric,
    )
    return SearchConfigCatalog((bm25, vector))


def _summary(
    *,
    name: str,
    mode: RetrievalMode,
    payload: dict[str, object],
) -> RetrievalConfigSummary:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return RetrievalConfigSummary(
        id=uuid5(_CONFIG_ID_NAMESPACE, canonical),
        revision=1,
        name=name,
        mode=mode,
        config_hash=config_hash,
    )
