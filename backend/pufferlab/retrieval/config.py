"""Deterministic retrieval configurations for one ingested fixture."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from uuid import UUID, uuid5

from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.providers.rerankers import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION
from pufferlab.providers.types import ConsistencyLevel, DistanceMetric, LexicalFieldWeights

_CONFIG_ID_NAMESPACE = UUID("224ff18e-4b57-55bb-a6ec-4e40384550da")


@dataclass(frozen=True, slots=True)
class SeededSearchConfig:
    summary: RetrievalConfigSummary
    mode: RetrievalMode
    result_k: int
    candidate_k: int
    consistency: ConsistencyLevel
    lexical_fields: LexicalFieldWeights | None = None
    vector_attribute: str | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_dimensions: int | None = None
    distance_metric: DistanceMetric | None = None
    rrf_rank_constant: int | None = None
    rrf_weights: tuple[float, float] | None = None
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranker_depth: int | None = None


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


@dataclass(frozen=True, slots=True)
class BoundSearchCatalog:
    """Persistable retrieval revisions and their exact executable counterparts."""

    configs: tuple[RetrievalConfig, ...]
    catalog: SearchConfigCatalog

    def __post_init__(self) -> None:
        persisted_summaries = tuple(_contract_summary(config) for config in self.configs)
        if persisted_summaries != self.catalog.summaries():
            raise ValueError("persisted and executable retrieval configurations must match")


def build_search_catalog(
    manifest: DatasetManifest,
    *,
    result_k: int = 10,
    candidate_k: int = 100,
    rrf_rank_constant: int = 60,
    rrf_weights: tuple[float, float] = (1.0, 1.0),
    reranker_depth: int = 50,
    consistency: ConsistencyLevel = "strong",
    lexical: LexicalSpec | None = None,
) -> SearchConfigCatalog:
    if result_k < 1:
        raise ValueError("result_k must be positive")
    if candidate_k < result_k:
        raise ValueError("candidate_k must be greater than or equal to result_k")
    if isinstance(rrf_rank_constant, bool) or rrf_rank_constant < 1:
        raise ValueError("rrf_rank_constant must be positive")
    if len(rrf_weights) != 2 or any(
        isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0
        for weight in rrf_weights
    ):
        raise ValueError("rrf_weights must contain two finite positive values")
    if reranker_depth < result_k or reranker_depth > candidate_k:
        raise ValueError("reranker_depth must be between result_k and candidate_k")
    write_spec = compile_namespace_write_spec(manifest)
    lexical_spec = lexical or LexicalSpec()
    lexical_fields = _compile_lexical_fields(manifest, lexical_spec)
    lexical_payload = {
        "title_weight": lexical_spec.title_weight,
        "body_weight": lexical_spec.body_weight,
    }

    common = {
        "dataset_slug": manifest.slug,
        "dataset_version": manifest.version,
        "namespace_schema_hash": write_spec.schema_hash,
        "result_k": result_k,
        "candidate_k": candidate_k,
        "consistency": consistency,
    }
    bm25_payload = {
        **common,
        "mode": RetrievalMode.BM25.value,
        "lexical": lexical_payload,
    }
    vector_payload = {
        **common,
        "mode": RetrievalMode.VECTOR.value,
        "vector_attribute": manifest.vector.attribute,
        "embedding_model": manifest.embedding.model,
        "embedding_revision": manifest.embedding.revision,
        "embedding_dimensions": manifest.embedding.dimensions,
        "distance_metric": manifest.vector.distance_metric,
    }
    hybrid_payload = {
        **vector_payload,
        "mode": RetrievalMode.HYBRID_RRF.value,
        "lexical": lexical_payload,
        "rrf_execution": "server",
        "rrf_rank_constant": rrf_rank_constant,
        "rrf_weights": rrf_weights,
    }
    rerank_payload = {
        **hybrid_payload,
        "mode": RetrievalMode.HYBRID_RERANK.value,
        "reranker_provider": "sentence_transformers",
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_revision": DEFAULT_RERANKER_REVISION,
        "reranker_depth": reranker_depth,
    }

    bm25 = SeededSearchConfig(
        summary=_summary(
            name="BM25 · weighted title + body",
            mode=RetrievalMode.BM25,
            payload=bm25_payload,
        ),
        mode=RetrievalMode.BM25,
        result_k=result_k,
        candidate_k=candidate_k,
        consistency=consistency,
        lexical_fields=lexical_fields,
    )
    vector = SeededSearchConfig(
        summary=_summary(
            name=f"Vector · {manifest.embedding.model}",
            mode=RetrievalMode.VECTOR,
            payload=vector_payload,
        ),
        mode=RetrievalMode.VECTOR,
        result_k=result_k,
        candidate_k=candidate_k,
        consistency=consistency,
        vector_attribute=manifest.vector.attribute,
        embedding_model=manifest.embedding.model,
        embedding_revision=manifest.embedding.revision,
        embedding_dimensions=manifest.embedding.dimensions,
        distance_metric=manifest.vector.distance_metric,
    )
    hybrid = SeededSearchConfig(
        summary=_summary(
            name="Hybrid · server RRF",
            mode=RetrievalMode.HYBRID_RRF,
            payload=hybrid_payload,
        ),
        mode=RetrievalMode.HYBRID_RRF,
        result_k=result_k,
        candidate_k=candidate_k,
        consistency=consistency,
        lexical_fields=lexical_fields,
        vector_attribute=manifest.vector.attribute,
        embedding_model=manifest.embedding.model,
        embedding_revision=manifest.embedding.revision,
        embedding_dimensions=manifest.embedding.dimensions,
        distance_metric=manifest.vector.distance_metric,
        rrf_rank_constant=rrf_rank_constant,
        rrf_weights=rrf_weights,
    )
    rerank = SeededSearchConfig(
        summary=_summary(
            name=f"Hybrid + reranker · {DEFAULT_RERANKER_MODEL}",
            mode=RetrievalMode.HYBRID_RERANK,
            payload=rerank_payload,
        ),
        mode=RetrievalMode.HYBRID_RERANK,
        result_k=result_k,
        candidate_k=candidate_k,
        consistency=consistency,
        lexical_fields=lexical_fields,
        vector_attribute=manifest.vector.attribute,
        embedding_model=manifest.embedding.model,
        embedding_revision=manifest.embedding.revision,
        embedding_dimensions=manifest.embedding.dimensions,
        distance_metric=manifest.vector.distance_metric,
        rrf_rank_constant=rrf_rank_constant,
        rrf_weights=rrf_weights,
        reranker_model=DEFAULT_RERANKER_MODEL,
        reranker_revision=DEFAULT_RERANKER_REVISION,
        reranker_depth=reranker_depth,
    )
    return SearchConfigCatalog((bm25, vector, hybrid, rerank))


def bind_retrieval_catalog(
    dataset_version: DatasetVersion,
    manifest: DatasetManifest,
    *,
    namespace: str | None = None,
    result_k: int = 50,
    candidate_k: int = 100,
    rrf_rank_constant: int = 60,
    rrf_weights: tuple[float, float] = (1.0, 1.0),
    reranker_depth: int = 50,
    consistency: ConsistencyLevel = "strong",
    lexical: LexicalSpec | None = None,
) -> BoundSearchCatalog:
    """Bind the frozen four-config suite to one proven-ready dataset revision."""

    _validate_dataset_binding(dataset_version, manifest, namespace=namespace)
    lexical_spec = lexical or LexicalSpec()
    # Reuse the executable compiler for all provider-specific validation, then replace only
    # identities with dataset-version-bound immutable contract identities below.
    executable = build_search_catalog(
        manifest,
        result_k=result_k,
        candidate_k=candidate_k,
        rrf_rank_constant=rrf_rank_constant,
        rrf_weights=rrf_weights,
        reranker_depth=reranker_depth,
        consistency=consistency,
        lexical=lexical_spec,
    )

    vector_spec = VectorSpec(
        attribute=manifest.vector.attribute,
        embedding_model=manifest.embedding.model,
    )
    rrf_spec = RrfSpec(rank_constant=rrf_rank_constant, weights=rrf_weights)
    reranker_spec = RerankerSpec(
        provider="sentence_transformers",
        model=DEFAULT_RERANKER_MODEL,
        revision=DEFAULT_RERANKER_REVISION,
        depth=reranker_depth,
    )
    specifications = (
        (RetrievalMode.BM25, lexical_spec, None, None, None),
        (RetrievalMode.VECTOR, None, vector_spec, None, None),
        (RetrievalMode.HYBRID_RRF, lexical_spec, vector_spec, rrf_spec, None),
        (
            RetrievalMode.HYBRID_RERANK,
            lexical_spec,
            vector_spec,
            rrf_spec,
            reranker_spec,
        ),
    )

    persisted: list[RetrievalConfig] = []
    rebound: list[SeededSearchConfig] = []
    for seeded, (mode, lexical_value, vector, rrf, reranker) in zip(
        executable.configs,
        specifications,
        strict=True,
    ):
        identity_payload = {
            "candidate_k": candidate_k,
            "consistency": consistency,
            "dataset_version_id": str(dataset_version.id),
            "filters": None,
            "lexical": (
                lexical_value.model_dump(mode="json") if lexical_value is not None else None
            ),
            "mode": mode.value,
            "name": seeded.summary.name,
            "reranker": reranker.model_dump(mode="json") if reranker is not None else None,
            "result_k": result_k,
            "revision": 1,
            "rrf": rrf.model_dump(mode="json") if rrf is not None else None,
            "vector": vector.model_dump(mode="json") if vector is not None else None,
        }
        canonical = _canonical_json(identity_payload)
        config_hash = hashlib.sha256(canonical.encode()).hexdigest()
        config = RetrievalConfig(
            id=uuid5(_CONFIG_ID_NAMESPACE, canonical),
            revision=1,
            name=seeded.summary.name,
            dataset_version_id=dataset_version.id,
            mode=mode,
            result_k=result_k,
            candidate_k=candidate_k,
            consistency=consistency,
            filters=None,
            lexical=lexical_value,
            vector=vector,
            rrf=rrf,
            reranker=reranker,
            config_hash=config_hash,
            created_at=dataset_version.created_at,
        )
        persisted.append(config)
        rebound.append(replace(seeded, summary=_contract_summary(config)))
    return BoundSearchCatalog(tuple(persisted), SearchConfigCatalog(tuple(rebound)))


def _compile_lexical_fields(
    manifest: DatasetManifest,
    lexical: LexicalSpec,
) -> LexicalFieldWeights:
    requested = (
        ("title", lexical.title_weight),
        ("body", lexical.body_weight),
    )
    available = set(manifest.fts.attributes)
    missing = [
        attribute for attribute, weight in requested if weight > 0 and attribute not in available
    ]
    if missing:
        raise ValueError(
            "positive lexical weights require full-text-search attributes: " + ", ".join(missing)
        )
    compiled = tuple((attribute, weight) for attribute, weight in requested if weight > 0)
    if not compiled:
        raise ValueError("at least one lexical weight must be positive")
    return compiled


def _summary(
    *,
    name: str,
    mode: RetrievalMode,
    payload: dict[str, object],
) -> RetrievalConfigSummary:
    canonical = _canonical_json(payload)
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return RetrievalConfigSummary(
        id=uuid5(_CONFIG_ID_NAMESPACE, canonical),
        revision=1,
        name=name,
        mode=mode,
        config_hash=config_hash,
    )


def _contract_summary(config: RetrievalConfig) -> RetrievalConfigSummary:
    return RetrievalConfigSummary(
        id=config.id,
        revision=config.revision,
        name=config.name,
        mode=config.mode,
        config_hash=config.config_hash,
    )


def _validate_dataset_binding(
    dataset_version: DatasetVersion,
    manifest: DatasetManifest,
    *,
    namespace: str | None,
) -> None:
    if dataset_version.status is not DatasetStatus.READY:
        raise ValueError("retrieval configurations require a READY dataset version")
    if namespace is not None and namespace != dataset_version.namespace:
        raise ValueError("dataset version namespace does not match the requested namespace")
    write_spec = compile_namespace_write_spec(manifest)
    profile = dataset_version.index_profile
    expected_profile_id = f"{manifest.slug}-{write_spec.schema_hash[:16]}"
    compatibility = {
        "dataset slug": (dataset_version.slug, manifest.slug),
        "dataset version": (dataset_version.version, manifest.version),
        "index profile id": (profile.id, expected_profile_id),
        "embedding provider": (profile.embedding_provider, manifest.embedding.provider),
        "embedding model": (profile.embedding_model, manifest.embedding.model),
        "embedding revision": (profile.embedding_revision, manifest.embedding.revision),
        "vector attribute": (profile.vector_attribute, manifest.vector.attribute),
        "vector dimensions": (profile.vector_dimensions, manifest.embedding.dimensions),
        "vector dtype": (profile.vector_dtype, manifest.vector.dtype),
        "distance metric": (profile.distance_metric, manifest.vector.distance_metric),
        "namespace schema hash": (profile.schema_hash, write_spec.schema_hash),
    }
    for field_name, (actual, expected) in compatibility.items():
        if actual != expected:
            raise ValueError(f"dataset version {field_name} does not match the manifest")

    manifest_fts = manifest.fts
    profile_fts = profile.fts_profile
    fts_compatibility = {
        "tokenizer": (profile_fts.tokenizer, manifest_fts.tokenizer),
        "case sensitivity": (profile_fts.case_sensitive, manifest_fts.case_sensitive),
        "language": (profile_fts.language, manifest_fts.language),
        "stemming": (profile_fts.stemming, manifest_fts.stemming),
        "stopword removal": (profile_fts.remove_stopwords, manifest_fts.remove_stopwords),
        "ASCII folding": (profile_fts.ascii_folding, manifest_fts.ascii_folding),
        "maximum token length": (profile_fts.max_token_length, manifest_fts.max_token_length),
        "k1": (profile_fts.k1, manifest_fts.k1),
        "b": (profile_fts.b, manifest_fts.b),
        "k3": (profile_fts.k3, manifest_fts.k3),
    }
    for field_name, (actual, expected) in fts_compatibility.items():
        if actual != expected:
            raise ValueError(f"dataset version FTS {field_name} does not match the manifest")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
