import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from pufferlab.datasets.cqadupstack import (
    ARCHIVE_URL,
    BEIR_COMMIT,
    CONTENT_LICENSE,
    CORPUS_MEMBER,
    CQADUPSTACK_COMMIT,
    QRELS_MEMBER,
    QUERIES_MEMBER,
    ArchiveLock,
    AttributionAvailability,
    CuratedQuery,
    CuratedQueryManifest,
    DatasetPreparationError,
    ForbiddenTokenWindow,
    MemberLock,
    PreprocessingLock,
    ProcessedFile,
    ProcessedPackLock,
    ProcessedPackLockFile,
    ProcessedPackManifest,
    RepositoryLock,
    SourceLock,
    curate_query_ids,
    curated_selection_sha256,
    iter_processed_documents,
    iter_processed_qrels,
    iter_processed_queries,
    load_curated_unix_corpus,
    load_unix_dataset_manifest,
    prepare_unix_pack,
    processed_content_sha256,
    source_lock_sha256,
    transformation_specification_sha256,
    verify_archive,
    verify_processed_pack,
)


def test_public_unix_manifest_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text('{"slug":"first","slug":"second"}', encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="duplicate key"):
        load_unix_dataset_manifest(manifest)


def test_prepare_is_deterministic_content_addressed_and_attributed(tmp_path: Path) -> None:
    archive, source_lock = _synthetic_archive(tmp_path)

    first = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    second = prepare_unix_pack(archive, tmp_path / "processed", source_lock)

    assert first == second
    assert first.name.startswith("cqadupstack-unix-")
    documents = tuple(iter_processed_documents(first))
    queries = tuple(iter_processed_queries(first))
    qrels = tuple(iter_processed_qrels(first))
    assert [document.external_id for document in documents] == ["2", "10"]
    assert [query.external_id for query in queries] == ["3", "20"]
    assert [(qrel.query_id, qrel.document_id) for qrel in qrels] == [("3", "2"), ("20", "10")]
    attribution = documents[0].attribution
    assert attribution.source_dataset == "CQADupStack"
    assert attribution.source_subset == "unix"
    assert attribution.canonical_post_url == "https://unix.stackexchange.com/questions/2"
    assert attribution.content_license == CONTENT_LICENSE
    assert attribution.author_display_name is None
    assert attribution.contribution_timestamp is None
    assert attribution.revision_timestamp is None
    assert attribution.attribution_metadata_status == "unavailable_in_pinned_archive"
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_sha256"] == first.name.removeprefix("cqadupstack-unix-")
    assert manifest["source_lock_sha256"] == source_lock_sha256(source_lock)
    assert manifest["preprocessing_sha256"] == transformation_specification_sha256()


def test_archive_size_md5_and_sha256_are_checked_before_zip_open(tmp_path: Path) -> None:
    archive, source_lock = _synthetic_archive(tmp_path)
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(archive.read_bytes()[:-1])

    with pytest.raises(DatasetPreparationError, match="archive is incomplete"):
        verify_archive(truncated, source_lock)

    wrong_md5 = source_lock.model_copy(
        update={"archive": source_lock.archive.model_copy(update={"published_md5": "0" * 32})}
    )
    with pytest.raises(DatasetPreparationError, match="published MD5"):
        verify_archive(archive, wrong_md5)

    wrong_sha256 = source_lock.model_copy(
        update={
            "archive": source_lock.archive.model_copy(
                update={"completed_download_sha256": "0" * 64}
            )
        }
    )
    with pytest.raises(DatasetPreparationError, match="local SHA-256"):
        verify_archive(archive, wrong_sha256)


def test_qrels_may_only_reference_retained_records(tmp_path: Path) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, dangling_qrel=True)

    with pytest.raises(DatasetPreparationError, match="was not retained"):
        prepare_unix_pack(archive, tmp_path / "processed", source_lock)


def test_member_metadata_is_locked_after_whole_archive_verification(tmp_path: Path) -> None:
    archive, source_lock = _synthetic_archive(tmp_path)
    corpus_lock = source_lock.members["corpus"].model_copy(update={"crc32": "00000000"})
    drifted = source_lock.model_copy(
        update={"members": {**source_lock.members, "corpus": corpus_lock}}
    )

    with pytest.raises(DatasetPreparationError, match="member metadata drifted"):
        prepare_unix_pack(archive, tmp_path / "processed", drifted)


def test_curation_is_deterministic_id_only_and_covers_four_strata() -> None:
    queries = tuple(
        (str(index), "sudo command fails during a multi step backup operation")
        for index in range(100, 180)
    )
    qrels = {query_id: ("document-a", "document-b") for query_id, _ in queries}

    first = curate_query_ids(queries, qrels)
    second = curate_query_ids(reversed(queries), qrels)

    assert first == second
    assert len(first) == 50
    assert len({entry.query_id for entry in first}) == 50
    assert {entry.primary_tag for entry in first} == {
        "exact_token",
        "semantic",
        "hybrid",
        "reranker",
    }
    serialized = json.dumps([entry.model_dump(mode="json") for entry in first])
    assert "query_text" not in serialized
    assert "title" not in serialized
    assert "body" not in serialized


def test_curated_manifest_rejects_text_bearing_or_hash_drift() -> None:
    entries = tuple(
        CuratedQuery(
            query_id=str(index),
            primary_tag=("exact_token", "semantic", "hybrid", "reranker")[index % 4],
            tags=(("exact_token", "semantic", "hybrid", "reranker")[index % 4],),
            reason="PufferLab-authored selection reason.",
        )
        for index in range(1, 51)
    )
    payload = {
        "format_version": 1,
        "selection_version": "pufferlab-curated-50-v1",
        "source_lock_sha256": "a" * 64,
        "query_count": 50,
        "selection_sha256": curated_selection_sha256(entries),
        "query_set_content_sha256": "b" * 64,
        "entries": tuple(entry.model_dump() for entry in entries),
    }
    CuratedQueryManifest.model_validate(payload)
    payload["entries"][0]["text"] = "synthetic but forbidden manifest field"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CuratedQueryManifest.model_validate(payload)


def test_processed_manifest_rejects_noncanonical_file_inventory() -> None:
    file = ProcessedFile(name="documents.jsonl", records=1, sha256="a" * 64)
    payload = {
        "format_version": 1,
        "dataset": "CQADupStack",
        "subset": "unix",
        "archive_sha256": "b" * 64,
        "source_lock_sha256": "c" * 64,
        "preprocessing_version": "pufferlab-cqadupstack-unix-v1",
        "preprocessing_sha256": "d" * 64,
        "content_sha256": "e" * 64,
        "files": (file, file, file),
    }

    with pytest.raises(ValueError, match="file inventory must be canonical"):
        ProcessedPackManifest.model_validate(payload)


def test_processed_pack_and_id_only_manifest_load_for_ingestion(tmp_path: Path) -> None:
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
        query_set_content_sha256=None,
        entries=entries,
    )
    curated_path = tmp_path / "curated.json"
    curated_path.write_text(curated.model_dump_json(), encoding="utf-8")

    corpus = load_curated_unix_corpus(
        processed,
        source_lock=source_lock,
        processed_pack_lock=_processed_pack_lock(source_lock, processed),
        dataset_manifest_path=Path(__file__).parents[3]
        / "datasets/cqadupstack-unix/dataset-manifest.json",
        curated_manifest_path=curated_path,
    )

    assert len(corpus.documents) == 60
    assert len(corpus.queries) == 50
    assert all(query.expected_external_ids for query in corpus.queries)
    attribution = corpus.documents[0].attributes["attribution"]
    assert isinstance(attribution, dict)
    assert attribution["attribution_metadata_status"] == "unavailable_in_pinned_archive"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("archive_sha256", "0" * 64, "archive SHA-256 drifted"),
        ("source_lock_sha256", "3" * 64, "source-lock SHA-256 drifted"),
        ("preprocessing_sha256", "1" * 64, "preprocessing SHA-256 drifted"),
        ("content_sha256", "2" * 64, "content SHA-256 is not canonical"),
    ),
)
def test_loader_rejects_each_self_declared_manifest_identity_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    original = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    expected_pack = _processed_pack_lock(source_lock, original)
    mutated = tmp_path / field / original.name
    mutated.parent.mkdir()
    shutil.copytree(original, mutated)
    manifest = _processed_manifest(mutated).model_copy(update={field: value})
    (mutated / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match=message):
        verify_processed_pack(mutated, source_lock=source_lock, expected_pack=expected_pack)


def test_public_corpus_loader_rejects_reviewed_provenance_reproduction(
    tmp_path: Path,
) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    original = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    expected_pack = _processed_pack_lock(source_lock, original)
    mutated = tmp_path / "review-reproduction" / original.name
    mutated.parent.mkdir()
    shutil.copytree(original, mutated)
    manifest = _processed_manifest(mutated).model_copy(
        update={
            "archive_sha256": "0" * 64,
            "preprocessing_sha256": "1" * 64,
            "content_sha256": "2" * 64,
        }
    )
    (mutated / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="archive SHA-256 drifted"):
        load_curated_unix_corpus(
            mutated,
            source_lock=source_lock,
            processed_pack_lock=expected_pack,
            dataset_manifest_path=Path(__file__).parents[3]
            / "datasets/cqadupstack-unix/dataset-manifest.json",
            curated_manifest_path=tmp_path / "never-read-curated.json",
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_loader_rejects_missing_or_extra_processed_pack_files(
    tmp_path: Path,
    mutation: str,
) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    original = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    expected_pack = _processed_pack_lock(source_lock, original)
    mutated = tmp_path / mutation / original.name
    mutated.parent.mkdir()
    shutil.copytree(original, mutated)
    if mutation == "missing":
        (mutated / "qrels.jsonl").unlink()
    else:
        (mutated / "unexpected.txt").write_text("synthetic extra", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="directory inventory is not exact"):
        verify_processed_pack(mutated, source_lock=source_lock, expected_pack=expected_pack)


def test_loader_rejects_wrong_content_address_directory(tmp_path: Path) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    original = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    expected_pack = _processed_pack_lock(source_lock, original)
    mutated = tmp_path / "processed" / "cqadupstack-unix-wrong-address"
    shutil.copytree(original, mutated)

    with pytest.raises(DatasetPreparationError, match="directory is not its content address"):
        verify_processed_pack(mutated, source_lock=source_lock, expected_pack=expected_pack)


def test_loader_rejects_changed_rows_with_self_updated_manifest_and_directory(
    tmp_path: Path,
) -> None:
    archive, source_lock = _synthetic_archive(tmp_path, query_count=60)
    original = prepare_unix_pack(archive, tmp_path / "processed", source_lock)
    expected_pack = _processed_pack_lock(source_lock, original)
    staging = tmp_path / "self-updated" / original.name
    staging.parent.mkdir()
    shutil.copytree(original, staging)
    documents_path = staging / "documents.jsonl"
    documents = documents_path.read_text(encoding="utf-8")
    documents_path.write_text(
        documents.replace("Synthetic body 0", "Synthetic body tampered", 1),
        encoding="utf-8",
    )
    original_manifest = _processed_manifest(staging)
    actual_files = _actual_processed_files(staging)
    changed_content_sha256 = processed_content_sha256(
        source_lock_hash=source_lock_sha256(source_lock),
        preprocessing_hash=transformation_specification_sha256(),
        files=actual_files,
    )
    changed_manifest = original_manifest.model_copy(
        update={"files": actual_files, "content_sha256": changed_content_sha256}
    )
    (staging / "manifest.json").write_text(changed_manifest.model_dump_json(), encoding="utf-8")
    changed_path = staging.with_name(f"cqadupstack-unix-{changed_content_sha256}")
    staging.rename(changed_path)

    with pytest.raises(DatasetPreparationError, match="reviewed pack identity"):
        verify_processed_pack(changed_path, source_lock=source_lock, expected_pack=expected_pack)


def _synthetic_archive(
    tmp_path: Path,
    *,
    dangling_qrel: bool = False,
    query_count: int = 2,
) -> tuple[Path, SourceLock]:
    archive = tmp_path / "synthetic-input.bin"
    if query_count == 2:
        documents = (
            {
                "_id": "10",
                "title": "Synthetic ten",
                "text": "Synthetic body ten",
                "metadata": {},
            },
            {
                "_id": "2",
                "title": "Synthetic two",
                "text": "Synthetic body two",
                "metadata": {},
            },
        )
        queries = (
            {"_id": "20", "text": "Synthetic query twenty", "metadata": {"discarded": {}}},
            {"_id": "3", "text": "Synthetic query three", "metadata": {"discarded": {}}},
        )
        qrel_document = "missing" if dangling_qrel else "2"
        qrels = f"query-id\tcorpus-id\tscore\r\n20\t10\t1\r\n3\t{qrel_document}\t1\r\n"
    else:
        documents = tuple(
            {
                "_id": str(10_000 + index),
                "title": f"Synthetic title {index}",
                "text": f"Synthetic body {index}",
                "metadata": {},
            }
            for index in range(query_count)
        )
        queries = tuple(
            {
                "_id": str(20_000 + index),
                "text": "sudo command fails during a multi step backup operation",
                "metadata": {"discarded": {}},
            }
            for index in range(query_count)
        )
        qrels = "query-id\tcorpus-id\tscore\r\n" + "".join(
            f"{20_000 + index}\t{10_000 + index}\t{1 + (index % 3)}\r\n"
            f"{20_000 + index}\t{10_000 + ((index + 1) % query_count)}\t1\r\n"
            for index in range(query_count)
        )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(CORPUS_MEMBER, _jsonl(documents))
        output.writestr(QUERIES_MEMBER, _jsonl(queries))
        output.writestr(QRELS_MEMBER, qrels)

    archive_bytes = archive.read_bytes()
    member_records = {
        "corpus": query_count,
        "queries": query_count,
        "qrels": query_count if query_count == 2 else query_count * 2,
    }
    member_names = {"corpus": CORPUS_MEMBER, "queries": QUERIES_MEMBER, "qrels": QRELS_MEMBER}
    with zipfile.ZipFile(archive) as loaded:
        members = {
            key: MemberLock(
                path=name,
                records=member_records[key],
                compressed_bytes=loaded.getinfo(name).compress_size,
                uncompressed_bytes=loaded.getinfo(name).file_size,
                crc32=f"{loaded.getinfo(name).CRC:08x}",
            )
            for key, name in member_names.items()
        }
    source_lock = SourceLock.model_construct(
        format_version=1,
        dataset="CQADupStack",
        subset="unix",
        archive=ArchiveLock(
            url=ARCHIVE_URL,
            bytes=len(archive_bytes),
            published_md5=hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest(),
            completed_download_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            last_modified="Tue, 20 Apr 2021 14:25:04 GMT",
            etag='"607ee440-13e82d1a8"',
        ),
        beir=RepositoryLock(
            url="https://github.com/beir-cellar/beir",
            commit=BEIR_COMMIT,
            license_url="https://github.com/beir-cellar/beir/blob/ef83d293/LICENSE",
        ),
        cqadupstack=RepositoryLock(
            url="https://github.com/D1Doris/CQADupStack",
            commit=CQADUPSTACK_COMMIT,
            license_url="https://github.com/D1Doris/CQADupStack/blob/f73fc5b2/LICENSE.md",
        ),
        beir_md5_registry_url="https://github.com/beir-cellar/beir/blob/ef83d293/md5.csv",
        paper_doi="10.1145/2838931.2838934",
        source_dump_date="2014-09-26",
        source_site="Unix & Linux Stack Exchange",
        source_site_url="https://unix.stackexchange.com/",
        content_license=CONTENT_LICENSE,
        content_license_urls=(
            "https://creativecommons.org/licenses/by-sa/2.5/",
            "https://creativecommons.org/licenses/by-sa/3.0/",
        ),
        stack_exchange_license_chronology_url="https://stackoverflow.com/help/licensing",
        beir_paper_url="https://arxiv.org/abs/2104.08663",
        members=members,
        preprocessing=PreprocessingLock(
            version="pufferlab-cqadupstack-unix-v1",
            specification_sha256=transformation_specification_sha256(),
        ),
        attribution_availability=AttributionAvailability(
            observed_corpus_fields=("_id", "title", "text", "metadata"),
            observed_query_fields=("_id", "text", "metadata"),
            author_display_name="unavailable_in_pinned_archive",
            contribution_timestamp="unavailable_in_pinned_archive",
            revision_timestamp="unavailable_in_pinned_archive",
            license_selection="CC-BY-SA-2.5-or-3.0_due_to_missing_timestamp",
        ),
        forbidden_source_token_windows=(ForbiddenTokenWindow(token_count=2, sha256="b" * 64),),
    )
    return archive, source_lock


def _jsonl(records: tuple[dict[str, object], ...]) -> str:
    return "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records)


def _processed_manifest(processed: Path) -> ProcessedPackManifest:
    return ProcessedPackManifest.model_validate_json(
        (processed / "manifest.json").read_text(encoding="utf-8")
    )


def _actual_processed_files(processed: Path) -> tuple[ProcessedFile, ...]:
    files: list[ProcessedFile] = []
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl"):
        path = processed / name
        data = path.read_bytes()
        files.append(
            ProcessedFile(
                name=name,
                records=len(data.splitlines()),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(files)


def _processed_pack_lock(source_lock: SourceLock, processed: Path) -> ProcessedPackLock:
    manifest = _processed_manifest(processed)
    return ProcessedPackLock(
        format_version=1,
        dataset="CQADupStack",
        subset="unix",
        source_lock_sha256=source_lock_sha256(source_lock),
        archive_sha256=source_lock.archive.completed_download_sha256,
        preprocessing_sha256=transformation_specification_sha256(),
        content_sha256=manifest.content_sha256,
        files=tuple(
            ProcessedPackLockFile(name=file.name, records=file.records) for file in manifest.files
        ),
    )
