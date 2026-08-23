from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
    RrfSpec,
)
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.models import DatasetManifest
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.retrieval.config import bind_retrieval_catalog

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "fixtures" / "tiny-corpus"
IDENTITY_NAMESPACE = UUID("b8ccefc4-7fcc-44b5-a6a4-39dcb32fdcc7")
CREATED_AT = datetime(2014, 9, 26, tzinfo=UTC)


def _dataset_version(
    manifest: DatasetManifest,
    *,
    namespace: str = "pufferlab-bound-one",
    identity: str = "dataset-one",
) -> DatasetVersion:
    write_spec = compile_namespace_write_spec(manifest)
    return DatasetVersion(
        id=uuid5(IDENTITY_NAMESPACE, identity),
        slug=manifest.slug,
        version=manifest.version,
        namespace=namespace,
        index_profile=IndexProfile(
            id=f"{manifest.slug}-{write_spec.schema_hash[:16]}",
            embedding_provider=manifest.embedding.provider,
            embedding_model=manifest.embedding.model,
            embedding_revision=manifest.embedding.revision,
            vector_attribute=manifest.vector.attribute,
            vector_dimensions=manifest.embedding.dimensions,
            vector_dtype=manifest.vector.dtype,
            distance_metric=manifest.vector.distance_metric,
            fts_profile=FtsProfile(
                tokenizer=manifest.fts.tokenizer,
                case_sensitive=manifest.fts.case_sensitive,
                language=manifest.fts.language,
                stemming=manifest.fts.stemming,
                remove_stopwords=manifest.fts.remove_stopwords,
                ascii_folding=manifest.fts.ascii_folding,
                max_token_length=manifest.fts.max_token_length,
                k1=manifest.fts.k1,
                b=manifest.fts.b,
                k3=manifest.fts.k3,
            ),
            schema_hash=write_spec.schema_hash,
        ),
        document_count=20,
        corpus_hash="fixture-corpus-hash",
        status=DatasetStatus.READY,
        created_at=CREATED_AT,
    )


def _summary(config: RetrievalConfig) -> RetrievalConfigSummary:
    return RetrievalConfigSummary(
        id=config.id,
        revision=config.revision,
        name=config.name,
        mode=config.mode,
        config_hash=config.config_hash,
    )


def test_binder_compiles_complete_contracts_and_exact_executable_summaries() -> None:
    manifest = load_fixture_corpus(FIXTURE_DIR).manifest
    dataset = _dataset_version(manifest)

    bound = bind_retrieval_catalog(dataset, manifest)

    assert tuple(config.mode for config in bound.configs) == (
        RetrievalMode.BM25,
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    )
    assert bound.catalog.summaries() == tuple(_summary(config) for config in bound.configs)
    assert all(config.dataset_version_id == dataset.id for config in bound.configs)
    assert all(config.result_k == 50 for config in bound.configs)
    assert all(config.candidate_k == 100 for config in bound.configs)
    assert all(config.created_at == dataset.created_at for config in bound.configs)
    assert bound.configs[0].lexical == LexicalSpec(title_weight=2.0, body_weight=1.0)
    assert bound.configs[2].rrf == RrfSpec(
        execution="server",
        rank_constant=60,
        weights=(1.0, 1.0),
    )
    assert bound.configs[3].reranker is not None
    assert bound.configs[3].reranker.depth == 50
    assert all(config.result_k == 50 for config in bound.catalog.configs)
    assert bound.catalog.configs[3].reranker_depth == 50
    assert len({config.id for config in bound.configs}) == 4
    assert all(len(config.config_hash) == 64 for config in bound.configs)


def test_binder_is_idempotent_and_dataset_revision_and_namespace_bound() -> None:
    manifest = load_fixture_corpus(FIXTURE_DIR).manifest
    first_dataset = _dataset_version(manifest)
    second_dataset = _dataset_version(
        manifest,
        namespace="pufferlab-bound-two",
        identity="dataset-two",
    )

    first = bind_retrieval_catalog(first_dataset, manifest)
    repeated = bind_retrieval_catalog(first_dataset, manifest)
    second = bind_retrieval_catalog(second_dataset, manifest)

    assert [config.model_dump(mode="json") for config in first.configs] == [
        config.model_dump(mode="json") for config in repeated.configs
    ]
    assert first.catalog.configs == repeated.catalog.configs
    assert {config.id for config in first.configs}.isdisjoint(
        config.id for config in second.configs
    )
    assert {config.config_hash for config in first.configs}.isdisjoint(
        config.config_hash for config in second.configs
    )


def test_binder_rejects_namespace_dataset_and_index_profile_mismatches() -> None:
    manifest = load_fixture_corpus(FIXTURE_DIR).manifest
    dataset = _dataset_version(manifest)

    with pytest.raises(ValueError, match="namespace"):
        bind_retrieval_catalog(dataset, manifest, namespace="pufferlab-other")
    with pytest.raises(ValueError, match="READY"):
        bind_retrieval_catalog(
            dataset.model_copy(update={"status": DatasetStatus.INDEXING}),
            manifest,
        )
    with pytest.raises(ValueError, match="dataset slug"):
        bind_retrieval_catalog(
            dataset.model_copy(update={"slug": "another-dataset"}),
            manifest,
        )
    with pytest.raises(ValueError, match="schema hash"):
        bind_retrieval_catalog(
            dataset.model_copy(
                update={
                    "index_profile": dataset.index_profile.model_copy(
                        update={"schema_hash": "wrong-schema"}
                    )
                }
            ),
            manifest,
        )
