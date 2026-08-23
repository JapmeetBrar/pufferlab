from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pufferlab.datasets.unix_application as unix_application_module
import pytest
from pufferlab.contracts.datasets import DatasetStatus, DatasetVersion
from pufferlab.contracts.evals import JudgedQuery, QuerySet
from pufferlab.datasets.checkpoints import IngestionCheckpointStore
from pufferlab.datasets.cqadupstack import (
    CuratedQueryManifest,
    ProcessedPackLock,
    SourceLock,
    curate_query_ids,
    curated_selection_sha256,
    iter_processed_qrels,
    iter_processed_queries,
    prepare_unix_pack,
    source_lock_sha256,
)
from pufferlab.datasets.ingestion import (
    EmbeddedDocument,
    IngestionService,
    NamespaceReadiness,
)
from pufferlab.datasets.schema import NamespaceWriteSpec
from pufferlab.datasets.unix_application import (
    UnixDatasetApplicationService,
    build_ready_unix_evaluation_seed,
    load_curated_unix_local_pack,
)
from pufferlab.persistence import Database, PufferLabRepository

from .test_cqadupstack import _processed_pack_lock, _synthetic_archive

REPOSITORY_ROOT = Path(__file__).parents[3]
DATASET_MANIFEST_PATH = REPOSITORY_ROOT / "datasets/cqadupstack-unix/dataset-manifest.json"


class _Embedder:
    dimensions = 384

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]


class _Writer:
    def __init__(self) -> None:
        self.documents: dict[UUID, EmbeddedDocument] = {}
        self.schema_hash = ""
        self.upsert_calls = 0

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        write_spec: NamespaceWriteSpec,
    ) -> None:
        del namespace
        self.upsert_calls += 1
        self.schema_hash = write_spec.schema_hash
        self.documents.update((document.id, document) for document in documents)

    async def inspect_readiness(
        self,
        namespace: str,
        *,
        expected_document_ids: frozenset[UUID],
    ) -> NamespaceReadiness:
        del namespace
        del expected_document_ids
        return NamespaceReadiness(
            document_count=len(self.documents),
            document_ids=frozenset(self.documents),
            schema_hash=self.schema_hash,
            metadata_ready=True,
            indexes_ready=True,
        )


def test_builder_preserves_grades_curation_metadata_and_contract_native_seed(
    tmp_path: Path,
) -> None:
    processed, curated_path, source_lock, processed_pack_lock = _prepared_curated_pack(tmp_path)
    local_pack = load_curated_unix_local_pack(
        processed,
        source_lock=source_lock,
        processed_pack_lock=processed_pack_lock,
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        curated_manifest_path=curated_path,
    )

    first = build_ready_unix_evaluation_seed(local_pack, namespace="caller-owned-unix")
    second = build_ready_unix_evaluation_seed(local_pack, namespace="caller-owned-unix")
    reordered = build_ready_unix_evaluation_seed(
        replace(local_pack, qrels=tuple(reversed(local_pack.qrels))),
        namespace="caller-owned-unix",
    )

    assert first == second == reordered
    assert isinstance(first.dataset_version, DatasetVersion)
    assert isinstance(first.query_set, QuerySet)
    assert all(isinstance(query, JudgedQuery) for query in first.judged_queries)
    assert first.dataset_version.status is DatasetStatus.READY
    assert first.dataset_version.document_count == 60
    assert first.query_set.query_count == 50
    assert len(first.query_set.content_hash) == 64

    source_grades: dict[str, list[int]] = {}
    for qrel in local_pack.qrels:
        source_grades.setdefault(qrel.query_id, []).append(qrel.relevance)
    for curated in first.curated_queries:
        judged = curated.judged_query
        assert [qrel.relevance_grade for qrel in judged.qrels] == source_grades[judged.external_id]
        assert tuple(judged.tags) == curated.tags
        assert curated.primary_tag in curated.tags
        assert curated.reason.startswith("Selected")
    assert any(qrel.relevance_grade > 1 for judged in first.judged_queries for qrel in judged.qrels)

    database = Database(tmp_path / "seed.sqlite3")
    database.migrate()
    try:
        repository = PufferLabRepository(database.session_factory)
        repository.put_dataset_version(first.dataset_version)
        repository.put_query_set(first.query_set, first.judged_queries)
        persisted_set, persisted_queries = repository.get_query_set(first.query_set.id)
    finally:
        database.dispose()
    assert persisted_set == first.query_set
    assert persisted_queries == list(first.judged_queries)


@pytest.mark.asyncio
async def test_application_service_owns_materialization_checkpoint_resume_and_no_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed, curated_path, source_lock, processed_pack_lock = _prepared_curated_pack(tmp_path)
    monkeypatch.setattr(unix_application_module, "load_source_lock", lambda path: source_lock)
    monkeypatch.setattr(
        unix_application_module,
        "load_processed_pack_lock",
        lambda path: processed_pack_lock,
    )
    writer = _Writer()
    ingestion = IngestionService(_Embedder(), writer, batch_size=20, max_concurrency=2)
    application = UnixDatasetApplicationService.from_paths(
        ingestion,
        IngestionCheckpointStore(tmp_path.resolve()),
        processed_path=processed,
        source_lock_path=tmp_path / "source-lock.json",
        processed_pack_lock_path=tmp_path / "processed-pack-lock.json",
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        curated_manifest_path=curated_path,
    )

    first = await application.ingest(namespace="caller-owned-unix")
    calls_after_first_run = writer.upsert_calls
    second = await application.ingest(namespace="caller-owned-unix")

    assert first.report.ready
    assert first.report.documents_completed == 60
    assert first.evaluation_seed.query_set.query_count == 50
    assert second.report.ready
    assert second.report.resumed_documents == 60
    assert writer.upsert_calls == calls_after_first_run
    assert second.evaluation_seed == first.evaluation_seed
    assert not hasattr(application, "delete_namespace")
    assert len(tuple((tmp_path / "ingestion-checkpoints").glob("*.json"))) == 1


def _prepared_curated_pack(
    tmp_path: Path,
) -> tuple[Path, Path, SourceLock, ProcessedPackLock]:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    processed = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    query_rows = tuple(
        (query.external_id, query.text) for query in iter_processed_queries(processed)
    )
    qrels: dict[str, list[str]] = {}
    for qrel in iter_processed_qrels(processed):
        qrels.setdefault(qrel.query_id, []).append(qrel.document_id)
    entries = curate_query_ids(query_rows, qrels)
    curated = CuratedQueryManifest(
        format_version=1,
        selection_version="pufferlab-curated-50-v1",
        source_lock_sha256=source_lock_sha256(source_lock),
        query_count=50,
        selection_sha256=curated_selection_sha256(entries),
        entries=entries,
    )
    curated_path = tmp_path / "curated.json"
    curated_path.write_text(curated.model_dump_json(), encoding="utf-8")
    return processed, curated_path, source_lock, _processed_pack_lock(source_lock, processed)
