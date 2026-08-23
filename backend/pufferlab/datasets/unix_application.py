"""Application boundary for preparing, ingesting, and seeding CQADupStack Unix."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import JudgedQuery, Qrel, QuerySet
from pufferlab.datasets.checkpoints import IngestionCheckpointStore
from pufferlab.datasets.cqadupstack import (
    CuratedQueryManifest,
    CurationTag,
    DatasetPreparationError,
    ProcessedPackLock,
    ProcessedQrel,
    SourceLock,
    iter_processed_qrels,
    load_curated_query_manifest,
    load_curated_unix_corpus,
    load_processed_pack_lock,
    load_source_lock,
    source_lock_sha256,
)
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID, document_uuid
from pufferlab.datasets.ingestion import (
    IngestionCheckpoint,
    IngestionReport,
    IngestionService,
    ProgressObserver,
)
from pufferlab.datasets.models import FixtureCorpus
from pufferlab.datasets.schema import compile_namespace_write_spec

# This is the immutable upstream revision date, not the wall-clock time of a local ingestion.
UNIX_REVISION_CREATED_AT = datetime(2014, 9, 26, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class CuratedJudgedQuerySeed:
    """One contract-native judged query plus its authored curation explanation."""

    judged_query: JudgedQuery
    primary_tag: CurationTag
    tags: tuple[CurationTag, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class UnixEvaluationSeed:
    """Immutable contract-native inputs for M2-E persistence and evaluation."""

    dataset_version: DatasetVersion
    query_set: QuerySet
    curated_queries: tuple[CuratedJudgedQuerySeed, ...]

    @property
    def judged_queries(self) -> tuple[JudgedQuery, ...]:
        return tuple(item.judged_query for item in self.curated_queries)


@dataclass(frozen=True, slots=True)
class CuratedUnixLocalPack:
    """Verified ignored local materialization, including official graded qrels."""

    corpus: FixtureCorpus
    curated_manifest: CuratedQueryManifest
    qrels: tuple[ProcessedQrel, ...]


@dataclass(frozen=True, slots=True)
class UnixIngestionResult:
    """Ready ingestion evidence and the exact immutable evaluation seed it enables."""

    report: IngestionReport
    evaluation_seed: UnixEvaluationSeed


def authenticate_persisted_unix_query_set(
    dataset: DatasetVersion,
    query_set: QuerySet,
    judged_queries: tuple[JudgedQuery, ...],
    *,
    curated_manifest: CuratedQueryManifest,
    checked_source_lock: SourceLock,
) -> None:
    """Authenticate the complete persisted suite against its checked ID-only source."""
    if curated_manifest.query_set_content_sha256 is None:
        raise DatasetPreparationError("curated query-set content anchor is missing")
    if curated_manifest.source_lock_sha256 != source_lock_sha256(checked_source_lock):
        raise DatasetPreparationError("curated query source lock does not match the checked source")
    if query_set.dataset_version_id != dataset.id:
        raise DatasetPreparationError("query set is bound to a different dataset revision")
    if len(judged_queries) != curated_manifest.query_count:
        raise DatasetPreparationError("persisted query count does not match the curated source")

    seen_query_ids: set[UUID] = set()
    for query, selection in zip(judged_queries, curated_manifest.entries, strict=True):
        expected_query_id = uuid5(
            PUFFERLAB_NAMESPACE_UUID,
            f"judged-query:{dataset.version}:{selection.query_id}",
        )
        if (
            query.id != expected_query_id
            or query.id in seen_query_ids
            or query.external_id != selection.query_id
            or tuple(query.tags) != selection.tags
            or query.filters is not None
        ):
            raise DatasetPreparationError(
                "persisted judged-query identity does not match the curated source"
            )
        seen_query_ids.add(query.id)
        qrel_ids = [qrel.document_id for qrel in query.qrels]
        if not qrel_ids or len(qrel_ids) != len(set(qrel_ids)):
            raise DatasetPreparationError("persisted judged-query qrels are not canonical")

    content_hash = unix_query_set_content_sha256(judged_queries, curated_manifest)
    expected_query_set_id = uuid5(
        PUFFERLAB_NAMESPACE_UUID,
        f"query-set:{dataset.id}:{content_hash}",
    )
    if (
        content_hash != curated_manifest.query_set_content_sha256
        or query_set.content_hash != content_hash
        or query_set.id != expected_query_set_id
        or query_set.name != "CQADupStack Unix curated 50"
        or query_set.version != curated_manifest.selection_version
        or query_set.query_count != curated_manifest.query_count
        or query_set.created_at != UNIX_REVISION_CREATED_AT
    ):
        raise DatasetPreparationError(
            "persisted query set does not match the immutable curated source"
        )


def load_curated_unix_local_pack(
    processed_path: Path,
    *,
    source_lock: SourceLock,
    processed_pack_lock: ProcessedPackLock,
    dataset_manifest_path: Path,
    curated_manifest_path: Path,
) -> CuratedUnixLocalPack:
    """Load one verified ignored pack without flattening its official qrel grades."""
    corpus = load_curated_unix_corpus(
        processed_path,
        source_lock=source_lock,
        processed_pack_lock=processed_pack_lock,
        dataset_manifest_path=dataset_manifest_path,
        curated_manifest_path=curated_manifest_path,
    )
    # The corpus loader has already bound and recomputed this exact curated manifest.
    curated_manifest = load_curated_query_manifest(curated_manifest_path)
    qrels = tuple(iter_processed_qrels(processed_path))
    return CuratedUnixLocalPack(
        corpus=corpus,
        curated_manifest=curated_manifest,
        qrels=qrels,
    )


def build_ready_unix_evaluation_seed(
    local_pack: CuratedUnixLocalPack,
    *,
    namespace: str,
) -> UnixEvaluationSeed:
    """Purely build stable READY contracts after the application service proves readiness."""
    if not namespace.strip():
        raise ValueError("namespace must not be blank")
    corpus = local_pack.corpus
    write_spec = compile_namespace_write_spec(corpus.manifest)
    dataset_identity = {
        "corpus_hash": corpus.corpus_hash,
        "dataset_version": corpus.manifest.version,
        "namespace": namespace,
        "schema_hash": write_spec.schema_hash,
    }
    dataset_id = uuid5(
        PUFFERLAB_NAMESPACE_UUID,
        f"dataset-version:{_canonical_hash(dataset_identity)}",
    )
    dataset_version = DatasetVersion(
        id=dataset_id,
        slug=corpus.manifest.slug,
        version=corpus.manifest.version,
        namespace=namespace,
        index_profile=IndexProfile(
            id=f"{corpus.manifest.slug}-{write_spec.schema_hash[:16]}",
            embedding_provider=corpus.manifest.embedding.provider,
            embedding_model=corpus.manifest.embedding.model,
            embedding_revision=corpus.manifest.embedding.revision,
            vector_attribute=corpus.manifest.vector.attribute,
            vector_dimensions=corpus.manifest.embedding.dimensions,
            vector_dtype=corpus.manifest.vector.dtype,
            distance_metric=corpus.manifest.vector.distance_metric,
            fts_profile=FtsProfile(
                tokenizer=corpus.manifest.fts.tokenizer,
                case_sensitive=corpus.manifest.fts.case_sensitive,
                language=corpus.manifest.fts.language,
                stemming=corpus.manifest.fts.stemming,
                remove_stopwords=corpus.manifest.fts.remove_stopwords,
                ascii_folding=corpus.manifest.fts.ascii_folding,
                max_token_length=corpus.manifest.fts.max_token_length,
                k1=corpus.manifest.fts.k1,
                b=corpus.manifest.fts.b,
                k3=corpus.manifest.fts.k3,
            ),
            schema_hash=write_spec.schema_hash,
        ),
        document_count=len(corpus.documents),
        corpus_hash=corpus.corpus_hash,
        status=DatasetStatus.READY,
        created_at=UNIX_REVISION_CREATED_AT,
    )

    query_by_external_id = {query.external_id: query for query in corpus.queries}
    if len(query_by_external_id) != len(corpus.queries):
        raise DatasetPreparationError("curated corpus contains duplicate query source IDs")
    selection_by_id = {
        selection.query_id: selection for selection in local_pack.curated_manifest.entries
    }
    if set(query_by_external_id) != set(selection_by_id):
        raise DatasetPreparationError("curated corpus and ID-only manifest do not match")

    qrels_by_query: defaultdict[str, list[ProcessedQrel]] = defaultdict(list)
    for qrel in local_pack.qrels:
        qrels_by_query[qrel.query_id].append(qrel)

    curated_queries: list[CuratedJudgedQuerySeed] = []
    for selection in local_pack.curated_manifest.entries:
        query = query_by_external_id[selection.query_id]
        source_qrels = tuple(
            sorted(
                qrels_by_query[selection.query_id],
                key=lambda qrel: _source_id_sort_key(qrel.document_id),
            )
        )
        if not source_qrels:
            raise DatasetPreparationError("curated query has no official graded qrels")
        expected_external_ids = tuple(query.expected_external_ids)
        actual_external_ids = tuple(qrel.document_id for qrel in source_qrels)
        if actual_external_ids != expected_external_ids:
            raise DatasetPreparationError(
                "graded qrels do not match the retained curated-query judgments"
            )
        judged_query = JudgedQuery(
            id=uuid5(
                PUFFERLAB_NAMESPACE_UUID,
                f"judged-query:{corpus.manifest.version}:{query.external_id}",
            ),
            external_id=query.external_id,
            text=query.text,
            tags=list(selection.tags),
            qrels=[
                Qrel(
                    document_id=document_uuid(corpus.manifest.version, qrel.document_id),
                    relevance_grade=qrel.relevance,
                )
                for qrel in source_qrels
            ],
        )
        curated_queries.append(
            CuratedJudgedQuerySeed(
                judged_query=judged_query,
                primary_tag=selection.primary_tag,
                tags=selection.tags,
                reason=selection.reason,
            )
        )

    query_set_content_hash = unix_query_set_content_sha256(
        tuple(item.judged_query for item in curated_queries),
        local_pack.curated_manifest,
    )
    if (
        local_pack.curated_manifest.query_set_content_sha256 is not None
        and query_set_content_hash != local_pack.curated_manifest.query_set_content_sha256
    ):
        raise DatasetPreparationError(
            "built query set does not match the checked curated content anchor"
        )
    query_set = QuerySet(
        id=uuid5(
            PUFFERLAB_NAMESPACE_UUID,
            f"query-set:{dataset_id}:{query_set_content_hash}",
        ),
        name="CQADupStack Unix curated 50",
        version=local_pack.curated_manifest.selection_version,
        dataset_version_id=dataset_id,
        query_count=len(curated_queries),
        content_hash=query_set_content_hash,
        created_at=UNIX_REVISION_CREATED_AT,
    )
    return UnixEvaluationSeed(
        dataset_version=dataset_version,
        query_set=query_set,
        curated_queries=tuple(curated_queries),
    )


class UnixDatasetApplicationService:
    """Own local-pack materialization, atomic resume, stable upserts, and READY seed creation."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        checkpoint_store: IngestionCheckpointStore,
        *,
        processed_path: Path,
        source_lock: SourceLock,
        processed_pack_lock: ProcessedPackLock,
        dataset_manifest_path: Path,
        curated_manifest_path: Path,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._checkpoint_store = checkpoint_store
        self._processed_path = processed_path
        self._source_lock = source_lock
        self._processed_pack_lock = processed_pack_lock
        self._dataset_manifest_path = dataset_manifest_path
        self._curated_manifest_path = curated_manifest_path

    @classmethod
    def from_paths(
        cls,
        ingestion_service: IngestionService,
        checkpoint_store: IngestionCheckpointStore,
        *,
        processed_path: Path,
        source_lock_path: Path,
        processed_pack_lock_path: Path,
        dataset_manifest_path: Path,
        curated_manifest_path: Path,
    ) -> UnixDatasetApplicationService:
        """Load the reviewed checked-in locks for the normal application composition root."""
        return cls(
            ingestion_service,
            checkpoint_store,
            processed_path=processed_path,
            source_lock=load_source_lock(source_lock_path),
            processed_pack_lock=load_processed_pack_lock(processed_pack_lock_path),
            dataset_manifest_path=dataset_manifest_path,
            curated_manifest_path=curated_manifest_path,
        )

    def materialize_local_pack(self) -> CuratedUnixLocalPack:
        return load_curated_unix_local_pack(
            self._processed_path,
            source_lock=self._source_lock,
            processed_pack_lock=self._processed_pack_lock,
            dataset_manifest_path=self._dataset_manifest_path,
            curated_manifest_path=self._curated_manifest_path,
        )

    async def ingest(
        self,
        *,
        namespace: str,
        on_progress: ProgressObserver | None = None,
    ) -> UnixIngestionResult:
        local_pack = self.materialize_local_pack()
        corpus = local_pack.corpus
        write_spec = compile_namespace_write_spec(corpus.manifest)
        checkpoint = self._checkpoint_store.load(
            namespace=namespace,
            dataset_version=corpus.manifest.version,
            corpus_hash=corpus.corpus_hash,
            schema_hash=write_spec.schema_hash,
        )
        report = await self._ingestion_service.ingest(
            corpus,
            namespace=namespace,
            on_progress=on_progress,
            resume_from=checkpoint,
            on_checkpoint=self._save_checkpoint,
        )
        if not report.ready:
            raise RuntimeError("ingestion service returned without proving namespace readiness")
        return UnixIngestionResult(
            report=report,
            evaluation_seed=build_ready_unix_evaluation_seed(
                local_pack,
                namespace=namespace,
            ),
        )

    def _save_checkpoint(self, checkpoint: IngestionCheckpoint) -> None:
        self._checkpoint_store.save(checkpoint)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def unix_query_set_content_sha256(
    judged_queries: tuple[JudgedQuery, ...],
    curated_manifest: CuratedQueryManifest,
) -> str:
    """Rebuild the canonical hash used by the immutable query-set UUID."""
    if len(judged_queries) != len(curated_manifest.entries):
        raise DatasetPreparationError("judged queries do not match the curated selection")
    return _canonical_hash(
        {
            "curated_queries": [
                {
                    "judged_query": judged_query.model_dump(mode="json"),
                    "primary_tag": selection.primary_tag,
                    "reason": selection.reason,
                    "tags": list(selection.tags),
                }
                for judged_query, selection in zip(
                    judged_queries,
                    curated_manifest.entries,
                    strict=True,
                )
            ],
            "selection_sha256": curated_manifest.selection_sha256,
            "selection_version": curated_manifest.selection_version,
            "source_lock_sha256": curated_manifest.source_lock_sha256,
        }
    )


def _source_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)
