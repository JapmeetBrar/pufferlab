"""Small PufferLab-authored ranks and qrels for the offline dashboard demo.

This module deliberately contains no vectors, provider responses, measured timings, metrics, or
aggregate summaries. Those values either do not exist for the synthetic demo or are derived by the
normal evaluation engine when the seed is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from pufferlab.contracts.evals import JudgedQuery, Qrel
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID, document_uuid
from pufferlab.datasets.models import (
    DatasetManifest,
    EmbeddingProfile,
    FtsProfile,
    SourceDocument,
    VectorProfile,
)

SYNTHETIC_DEMO_CREATED_AT = datetime(2026, 8, 23, tzinfo=UTC)
SYNTHETIC_DEMO_VERSION = "pufferlab-synthetic-demo-v1"
_DOCUMENT_COUNT = 60
_QUERY_COUNT = 50
_RANKING_DEPTH = 50
_MODE_ORDER = (
    RetrievalMode.BM25,
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
)


@dataclass(frozen=True, slots=True)
class AuthoredSyntheticQuery:
    """One judged query and four authored rankings, with no copied metric values."""

    judged_query: JudgedQuery
    rankings: tuple[tuple[RetrievalMode, tuple[UUID, ...]], ...]

    def __post_init__(self) -> None:
        if tuple(mode for mode, _ranking in self.rankings) != _MODE_ORDER:
            raise ValueError("synthetic rankings must retain canonical config mode order")
        for _mode, ranking in self.rankings:
            if len(ranking) != _RANKING_DEPTH or len(ranking) != len(set(ranking)):
                raise ValueError("synthetic rankings must contain 50 unique document IDs")

    def ranking_for(self, mode: RetrievalMode) -> tuple[UUID, ...]:
        return dict(self.rankings)[mode]


@dataclass(frozen=True, slots=True)
class AuthoredSyntheticDemo:
    """The complete checked-in synthetic inputs used to derive durable demo state."""

    manifest: DatasetManifest
    documents: tuple[SourceDocument, ...]
    queries: tuple[AuthoredSyntheticQuery, ...]

    def __post_init__(self) -> None:
        if len(self.documents) != _DOCUMENT_COUNT:
            raise ValueError("the synthetic demo requires exactly 60 authored documents")
        if len(self.queries) != _QUERY_COUNT:
            raise ValueError("the synthetic demo requires exactly 50 authored queries")
        document_ids = {
            document_uuid(self.manifest.version, document.external_id)
            for document in self.documents
        }
        if len(document_ids) != len(self.documents):
            raise ValueError("synthetic document identities must be unique")
        query_ids = [item.judged_query.id for item in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("synthetic query identities must be unique")
        for item in self.queries:
            if any(qrel.document_id not in document_ids for qrel in item.judged_query.qrels):
                raise ValueError("synthetic qrels must reference authored documents")
            if any(
                document_id not in document_ids
                for _mode, ranking in item.rankings
                for document_id in ranking
            ):
                raise ValueError("synthetic rankings must reference authored documents")


def _manifest() -> DatasetManifest:
    # This intentionally mirrors the reviewed live index schema while naming only synthetic,
    # PufferLab-authored material.
    return DatasetManifest(
        format_version=1,
        slug="pufferlab-synthetic-demo",
        version=SYNTHETIC_DEMO_VERSION,
        title="PufferLab offline synthetic evaluation demo",
        license="CC0-1.0",
        source_url=(
            "https://github.com/JapmeetBrar/pufferlab/tree/main/backend/pufferlab/synthetic_demo"
        ),
        embedding=EmbeddingProfile(
            provider="sentence_transformers",
            model="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            dimensions=384,
        ),
        vector=VectorProfile(
            attribute="vector",
            dtype="f16",
            distance_metric="cosine_distance",
        ),
        fts=FtsProfile(
            attributes=["title", "body"],
            tokenizer="word_v4",
            case_sensitive=False,
            language="english",
            stemming=False,
            remove_stopwords=False,
            ascii_folding=False,
            max_token_length=39,
            k1=1.2,
            b=0.75,
            k3=8.0,
        ),
    )


def _documents(manifest: DatasetManifest) -> tuple[SourceDocument, ...]:
    return tuple(
        SourceDocument(
            external_id=f"demo-document-{index:03d}",
            title=f"Synthetic troubleshooting note {index:03d}",
            body=(
                f"PufferLab-authored offline example {index:03d}. "
                f"It discusses synthetic command token demo-{index:03d}."
            ),
            source_url=(
                "https://example.invalid/pufferlab/synthetic/"
                f"{manifest.version}/demo-document-{index:03d}"
            ),
            attributes={"synthetic": True, "ordinal": index},
        )
        for index in range(1, _DOCUMENT_COUNT + 1)
    )


def _queries(
    manifest: DatasetManifest,
    documents: tuple[SourceDocument, ...],
) -> tuple[AuthoredSyntheticQuery, ...]:
    document_ids = tuple(
        document_uuid(manifest.version, document.external_id) for document in documents
    )
    queries: list[AuthoredSyntheticQuery] = []
    for query_index in range(_QUERY_COUNT):
        primary = document_ids[query_index]
        secondary = document_ids[(query_index + 10) % len(document_ids)]
        no_positive_qrels = query_index == _QUERY_COUNT - 1
        qrels = [
            Qrel(document_id=primary, relevance_grade=0 if no_positive_qrels else 2),
            Qrel(document_id=secondary, relevance_grade=0 if no_positive_qrels else 1),
        ]
        external_id = f"demo-query-{query_index + 1:03d}"
        judged_query = JudgedQuery(
            id=uuid5(
                PUFFERLAB_NAMESPACE_UUID,
                f"judged-query:{manifest.version}:{external_id}",
            ),
            external_id=external_id,
            text=(f"Find the synthetic troubleshooting note for demo token {query_index + 1:03d}."),
            tags=["synthetic_demo", "offline"],
            qrels=qrels,
        )
        rank_positions = (
            (1 + query_index % 10, 21 + query_index % 10),
            (2 + query_index % 8, 13 + query_index % 8),
            (1 + query_index % 5, 7 + query_index % 5),
            (1, 2),
        )
        rankings = tuple(
            (
                mode,
                _ranking(
                    document_ids,
                    primary=primary,
                    secondary=secondary,
                    primary_rank=primary_rank,
                    secondary_rank=secondary_rank,
                    rotation=query_index,
                ),
            )
            for mode, (primary_rank, secondary_rank) in zip(
                _MODE_ORDER,
                rank_positions,
                strict=True,
            )
        )
        queries.append(AuthoredSyntheticQuery(judged_query=judged_query, rankings=rankings))
    return tuple(queries)


def _ranking(
    document_ids: tuple[UUID, ...],
    *,
    primary: UUID,
    secondary: UUID,
    primary_rank: int,
    secondary_rank: int,
    rotation: int,
) -> tuple[UUID, ...]:
    if primary_rank == secondary_rank:
        raise ValueError("synthetic relevant ranks must be distinct")
    rotated = document_ids[rotation:] + document_ids[:rotation]
    fillers = iter(
        document_id for document_id in rotated if document_id not in {primary, secondary}
    )
    ranking: list[UUID] = []
    for rank in range(1, _RANKING_DEPTH + 1):
        if rank == primary_rank:
            ranking.append(primary)
        elif rank == secondary_rank:
            ranking.append(secondary)
        else:
            ranking.append(next(fillers))
    return tuple(ranking)


_MANIFEST = _manifest()
_DOCUMENTS = _documents(_MANIFEST)
AUTHORED_SYNTHETIC_DEMO = AuthoredSyntheticDemo(
    manifest=_MANIFEST,
    documents=_DOCUMENTS,
    queries=_queries(_MANIFEST, _DOCUMENTS),
)
