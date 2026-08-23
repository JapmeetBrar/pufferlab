"""Fail-closed preparation of the pinned CQADupStack Unix evaluation pack."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from pufferlab.contracts.common import JsonValue
from pufferlab.datasets.identity import corpus_hash
from pufferlab.datasets.models import DatasetManifest, FixtureCorpus, FixtureQuery, SourceDocument

BEIR_COMMIT = "ef83d29307061c65d04b035b4f4e7c18bd8374af"
CQADUPSTACK_COMMIT = "f73fc5b2cc708c61d33bc76a3de93de0bf5bf584"
ARCHIVE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
BEIR_URL = f"https://github.com/beir-cellar/beir/tree/{BEIR_COMMIT}"
BEIR_LICENSE_URL = f"https://github.com/beir-cellar/beir/blob/{BEIR_COMMIT}/LICENSE"
BEIR_MD5_REGISTRY_URL = (
    f"https://github.com/beir-cellar/beir/blob/{BEIR_COMMIT}/examples/dataset/md5.csv"
)
CQADUPSTACK_URL = f"https://github.com/D1Doris/CQADupStack/tree/{CQADUPSTACK_COMMIT}"
CQADUPSTACK_LICENSE_URL = (
    f"https://github.com/D1Doris/CQADupStack/blob/{CQADUPSTACK_COMMIT}/LICENSE.md"
)
CC_BY_SA_25_URL = "https://creativecommons.org/licenses/by-sa/2.5/"
CC_BY_SA_30_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
STACK_EXCHANGE_LICENSE_URL = "https://stackoverflow.com/help/licensing"
FORBIDDEN_SOURCE_WINDOWS = (
    (7, "24b6ba6ae914d0cf90e562a337b6d3ac5fe280a30a2fcd18fffba15701552970"),
    (5, "c3e9237ee91ae4478585746ec2f4acc8d0112ece8c89ccd2830203fc6535d1b7"),
)
ARCHIVE_BYTES = 5_343_728_040
ARCHIVE_MD5 = "4e41456d7df8ee7760a7f866133bda78"
CORPUS_MEMBER = "cqadupstack/unix/corpus.jsonl"
QUERIES_MEMBER = "cqadupstack/unix/queries.jsonl"
QRELS_MEMBER = "cqadupstack/unix/qrels/test.tsv"
CORPUS_RECORDS = 47_382
QUERY_RECORDS = 1_072
QREL_RECORDS = 1_693
TRANSFORMATION_VERSION: Literal["pufferlab-cqadupstack-unix-v1"] = "pufferlab-cqadupstack-unix-v1"
SOURCE_DATASET: Literal["CQADupStack"] = "CQADupStack"
SOURCE_SUBSET: Literal["unix"] = "unix"
SOURCE_SITE: Literal["Unix & Linux Stack Exchange"] = "Unix & Linux Stack Exchange"
SOURCE_DUMP_DATE: Literal["2014-09-26"] = "2014-09-26"
CONTENT_LICENSE: Literal["CC-BY-SA-2.5 OR CC-BY-SA-3.0"] = "CC-BY-SA-2.5 OR CC-BY-SA-3.0"
ATTRIBUTION_METADATA_STATUS: Literal["unavailable_in_pinned_archive"] = (
    "unavailable_in_pinned_archive"
)

_TRANSFORMATION_SPEC: dict[str, JsonValue] = {
    "format_version": 1,
    "unicode_normalization": "NFC",
    "line_endings": "LF",
    "whitespace": "preserve",
    "discarded_upstream_fields": ["metadata"],
    "attribution_metadata": ATTRIBUTION_METADATA_STATUS,
    "qrel_policy": "require_retained_query_and_document",
    "ordering": ["numeric_source_id", "lexical_source_id_fallback"],
}


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, Field(min_length=1), AfterValidator(_non_blank)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Md5 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ArchiveLock(_StrictModel):
    url: NonBlank
    bytes: int = Field(gt=0)
    published_md5: Md5
    completed_download_sha256: Sha256
    last_modified: Literal["Tue, 20 Apr 2021 14:25:04 GMT"]
    etag: Literal['"607ee440-13e82d1a8"']


class RepositoryLock(_StrictModel):
    url: NonBlank
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    license_url: NonBlank


class MemberLock(_StrictModel):
    path: NonBlank
    records: int = Field(gt=0)
    compressed_bytes: int = Field(gt=0)
    uncompressed_bytes: int = Field(gt=0)
    crc32: Annotated[str, Field(pattern=r"^[0-9a-f]{8}$")]


class PreprocessingLock(_StrictModel):
    version: Literal["pufferlab-cqadupstack-unix-v1"]
    specification_sha256: Sha256


class ForbiddenTokenWindow(_StrictModel):
    token_count: int = Field(ge=2)
    sha256: Sha256


class AttributionAvailability(_StrictModel):
    observed_corpus_fields: tuple[Literal["_id", "title", "text", "metadata"], ...]
    observed_query_fields: tuple[Literal["_id", "text", "metadata"], ...]
    author_display_name: Literal["unavailable_in_pinned_archive"]
    contribution_timestamp: Literal["unavailable_in_pinned_archive"]
    revision_timestamp: Literal["unavailable_in_pinned_archive"]
    license_selection: Literal["CC-BY-SA-2.5-or-3.0_due_to_missing_timestamp"]


class SourceLock(_StrictModel):
    format_version: Literal[1]
    dataset: Literal["CQADupStack"]
    subset: Literal["unix"]
    archive: ArchiveLock
    beir: RepositoryLock
    cqadupstack: RepositoryLock
    beir_md5_registry_url: NonBlank
    paper_doi: Literal["10.1145/2838931.2838934"]
    source_dump_date: Literal["2014-09-26"]
    source_site: Literal["Unix & Linux Stack Exchange"]
    source_site_url: Literal["https://unix.stackexchange.com/"]
    content_license: Literal["CC-BY-SA-2.5 OR CC-BY-SA-3.0"]
    content_license_urls: tuple[NonBlank, NonBlank]
    stack_exchange_license_chronology_url: NonBlank
    beir_paper_url: Literal["https://arxiv.org/abs/2104.08663"]
    members: dict[Literal["corpus", "queries", "qrels"], MemberLock]
    preprocessing: PreprocessingLock
    attribution_availability: AttributionAvailability
    forbidden_source_token_windows: tuple[ForbiddenTokenWindow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def pins_match_reviewed_acquisition_chain(self) -> SourceLock:
        expected_members = {
            "corpus": (CORPUS_MEMBER, CORPUS_RECORDS, 17_356_767, 52_466_222, "6bdfe142"),
            "queries": (
                QUERIES_MEMBER,
                QUERY_RECORDS,
                290_036_323,
                826_061_363,
                "fe744c49",
            ),
            "qrels": (QRELS_MEMBER, QREL_RECORDS, 10_142, 26_039, "e6d85907"),
        }
        actual_members = {
            key: (
                value.path,
                value.records,
                value.compressed_bytes,
                value.uncompressed_bytes,
                value.crc32,
            )
            for key, value in self.members.items()
        }
        if self.archive.url != ARCHIVE_URL:
            raise ValueError("archive URL drifted from the reviewed BEIR source")
        if self.archive.bytes != ARCHIVE_BYTES or self.archive.published_md5 != ARCHIVE_MD5:
            raise ValueError("archive size or published MD5 drifted from the reviewed BEIR lock")
        if self.beir.commit != BEIR_COMMIT:
            raise ValueError("BEIR revision drifted from the reviewed source")
        if self.cqadupstack.commit != CQADUPSTACK_COMMIT:
            raise ValueError("CQADupStack revision drifted from the reviewed source")
        if self.beir.url != BEIR_URL or self.beir.license_url != BEIR_LICENSE_URL:
            raise ValueError("BEIR repository or license URL drifted from the reviewed source")
        if (
            self.cqadupstack.url != CQADUPSTACK_URL
            or self.cqadupstack.license_url != CQADUPSTACK_LICENSE_URL
        ):
            raise ValueError(
                "CQADupStack repository or license URL drifted from the reviewed source"
            )
        if self.beir_md5_registry_url != BEIR_MD5_REGISTRY_URL:
            raise ValueError("BEIR MD5 registry URL drifted from the reviewed source")
        if self.content_license_urls != (CC_BY_SA_25_URL, CC_BY_SA_30_URL):
            raise ValueError("Stack Exchange content license URLs drifted")
        if self.stack_exchange_license_chronology_url != STACK_EXCHANGE_LICENSE_URL:
            raise ValueError("Stack Exchange license chronology URL drifted")
        if self.attribution_availability.observed_corpus_fields != (
            "_id",
            "title",
            "text",
            "metadata",
        ) or self.attribution_availability.observed_query_fields != (
            "_id",
            "text",
            "metadata",
        ):
            raise ValueError("observed attribution field inventory drifted")
        actual_forbidden_windows = tuple(
            (window.token_count, window.sha256) for window in self.forbidden_source_token_windows
        )
        if actual_forbidden_windows != FORBIDDEN_SOURCE_WINDOWS:
            raise ValueError("known source-text exposure scan windows drifted")
        if actual_members != expected_members:
            raise ValueError("Unix archive member inventory drifted from the reviewed source")
        if self.preprocessing.specification_sha256 != transformation_specification_sha256():
            raise ValueError("preprocessing specification drifted from its source lock")
        return self


class Attribution(_StrictModel):
    source_dataset: Literal["CQADupStack"] = SOURCE_DATASET
    source_subset: Literal["unix"] = SOURCE_SUBSET
    original_post_id: NonBlank
    canonical_post_url: NonBlank
    source_site: Literal["Unix & Linux Stack Exchange"] = SOURCE_SITE
    source_dump_date: Literal["2014-09-26"] = SOURCE_DUMP_DATE
    transformation_version: Literal["pufferlab-cqadupstack-unix-v1"] = TRANSFORMATION_VERSION
    content_hash: Sha256
    content_license: Literal["CC-BY-SA-2.5 OR CC-BY-SA-3.0"] = CONTENT_LICENSE
    attribution_metadata_status: Literal["unavailable_in_pinned_archive"] = (
        ATTRIBUTION_METADATA_STATUS
    )
    author_display_name: None = None
    contribution_timestamp: None = None
    revision_timestamp: None = None


class ProcessedDocument(_StrictModel):
    external_id: NonBlank
    title: str
    body: NonBlank
    source_url: NonBlank
    attribution: Attribution

    @model_validator(mode="after")
    def identity_is_consistent(self) -> ProcessedDocument:
        if self.external_id != self.attribution.original_post_id:
            raise ValueError("document source identity does not match attribution")
        if self.source_url != self.attribution.canonical_post_url:
            raise ValueError("document source URL does not match attribution")
        if self.attribution.content_hash != _canonical_hash([self.title, self.body]):
            raise ValueError("document content hash does not match retained text")
        return self


class ProcessedQuery(_StrictModel):
    external_id: NonBlank
    text: NonBlank
    source_url: NonBlank
    attribution: Attribution

    @model_validator(mode="after")
    def identity_is_consistent(self) -> ProcessedQuery:
        if self.external_id != self.attribution.original_post_id:
            raise ValueError("query source identity does not match attribution")
        if self.source_url != self.attribution.canonical_post_url:
            raise ValueError("query source URL does not match attribution")
        if self.attribution.content_hash != _canonical_hash([self.text]):
            raise ValueError("query content hash does not match retained text")
        return self


class ProcessedQrel(_StrictModel):
    query_id: NonBlank
    document_id: NonBlank
    relevance: int = Field(ge=1)
    source_dataset: Literal["CQADupStack"] = SOURCE_DATASET
    source_subset: Literal["unix"] = SOURCE_SUBSET
    transformation_version: Literal["pufferlab-cqadupstack-unix-v1"] = TRANSFORMATION_VERSION


class ProcessedFile(_StrictModel):
    name: NonBlank
    records: int = Field(gt=0)
    sha256: Sha256


class ProcessedPackManifest(_StrictModel):
    format_version: Literal[1]
    dataset: Literal["CQADupStack"]
    subset: Literal["unix"]
    archive_sha256: Sha256
    source_lock_sha256: Sha256
    preprocessing_version: Literal["pufferlab-cqadupstack-unix-v1"]
    preprocessing_sha256: Sha256
    content_sha256: Sha256
    files: tuple[ProcessedFile, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def file_inventory_is_canonical(self) -> ProcessedPackManifest:
        if tuple(file.name for file in self.files) != (
            "documents.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
        ):
            raise ValueError("processed pack file inventory must be canonical")
        return self


class ProcessedPackLockFile(_StrictModel):
    name: NonBlank
    records: int = Field(gt=0)


class ProcessedPackLock(_StrictModel):
    """Independently reviewed identity for the real deterministic Unix output."""

    format_version: Literal[1]
    dataset: Literal["CQADupStack"]
    subset: Literal["unix"]
    source_lock_sha256: Sha256
    archive_sha256: Sha256
    preprocessing_sha256: Sha256
    content_sha256: Sha256
    files: tuple[ProcessedPackLockFile, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def file_inventory_is_canonical(self) -> ProcessedPackLock:
        if tuple(file.name for file in self.files) != (
            "documents.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
        ):
            raise ValueError("processed pack lock file inventory must be canonical")
        return self


type CurationTag = Literal["exact_token", "semantic", "hybrid", "reranker"]


class CuratedQuery(_StrictModel):
    query_id: NonBlank
    primary_tag: CurationTag
    tags: tuple[CurationTag, ...] = Field(min_length=1)
    reason: NonBlank

    @model_validator(mode="after")
    def primary_tag_is_present(self) -> CuratedQuery:
        if self.primary_tag not in self.tags:
            raise ValueError("primary tag must be present in tags")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("curation tags must be unique")
        return self


class CuratedQueryManifest(_StrictModel):
    format_version: Literal[1]
    selection_version: Literal["pufferlab-curated-50-v1"]
    source_lock_sha256: Sha256
    query_count: Literal[50]
    selection_sha256: Sha256
    # The exact licensed query/qrel payload stays local. This checked digest anchors the complete
    # contract-native query set without copying that payload into the repository.
    query_set_content_sha256: Sha256 | None = None
    entries: tuple[CuratedQuery, ...] = Field(min_length=50, max_length=50)

    @model_validator(mode="after")
    def selection_is_canonical(self) -> CuratedQueryManifest:
        ids = tuple(entry.query_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("curated query IDs must be unique")
        if ids != tuple(sorted(ids, key=_source_id_sort_key)):
            raise ValueError("curated query IDs must use canonical source-ID ordering")
        if self.selection_sha256 != curated_selection_sha256(self.entries):
            raise ValueError("curated selection hash does not match its entries")
        if set(entry.primary_tag for entry in self.entries) != {
            "exact_token",
            "semantic",
            "hybrid",
            "reranker",
        }:
            raise ValueError("curated selection must cover every analysis stratum")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedArchive:
    path: Path
    bytes: int
    md5: str
    sha256: str


class DatasetPreparationError(RuntimeError):
    """The local archive or processed pack does not satisfy the pinned source lock."""


def transformation_specification_sha256() -> str:
    return _canonical_hash(_TRANSFORMATION_SPEC)


def load_source_lock(path: Path) -> SourceLock:
    return _load_model_json(path, SourceLock)


def load_curated_query_manifest(path: Path) -> CuratedQueryManifest:
    return _load_model_json(path, CuratedQueryManifest)


def load_processed_pack_lock(path: Path) -> ProcessedPackLock:
    return _load_model_json(path, ProcessedPackLock)


def source_lock_sha256(source_lock: SourceLock) -> str:
    return _canonical_hash(source_lock.model_dump(mode="json"))


def curated_selection_sha256(entries: Iterable[CuratedQuery]) -> str:
    return _canonical_hash([entry.model_dump(mode="json", exclude_none=False) for entry in entries])


def processed_content_sha256(
    *,
    source_lock_hash: str,
    preprocessing_hash: str,
    files: tuple[ProcessedFile, ...],
) -> str:
    """Hash the exact canonical identity used by generated and loaded packs."""
    return _canonical_hash(
        {
            "format_version": 1,
            "source_lock_sha256": source_lock_hash,
            "preprocessing_sha256": preprocessing_hash,
            "files": [file.model_dump(mode="json") for file in files],
        }
    )


def verify_archive(path: Path, source_lock: SourceLock) -> VerifiedArchive:
    """Verify the complete archive before any ZIP member is opened."""
    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise DatasetPreparationError("CQADupStack archive is missing or unreadable") from error
    if actual_bytes != source_lock.archive.bytes:
        raise DatasetPreparationError(
            f"CQADupStack archive is incomplete: expected {source_lock.archive.bytes} bytes, "
            f"observed {actual_bytes}"
        )

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                md5.update(chunk)
                sha256.update(chunk)
    except OSError as error:
        raise DatasetPreparationError("CQADupStack archive could not be hashed") from error

    actual_md5 = md5.hexdigest()
    actual_sha256 = sha256.hexdigest()
    if actual_md5 != source_lock.archive.published_md5:
        raise DatasetPreparationError("CQADupStack archive does not match BEIR's published MD5")
    if actual_sha256 != source_lock.archive.completed_download_sha256:
        raise DatasetPreparationError("CQADupStack archive does not match the local SHA-256 lock")
    return VerifiedArchive(
        path=path,
        bytes=actual_bytes,
        md5=actual_md5,
        sha256=actual_sha256,
    )


def prepare_unix_pack(
    archive_path: Path,
    output_parent: Path,
    source_lock: SourceLock,
) -> Path:
    """Create an ignored, content-addressed Unix pack from one verified archive."""
    verified = verify_archive(archive_path, source_lock)
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cqadupstack-unix-staging-", dir=output_parent))
    try:
        manifest = _process_verified_archive(verified, staging, source_lock)
        final_path = output_parent / f"cqadupstack-unix-{manifest.content_sha256}"
        if final_path.exists():
            existing = _load_model_json(final_path / "manifest.json", ProcessedPackManifest)
            if existing != manifest:
                raise DatasetPreparationError(
                    "content-addressed output exists with a conflicting manifest"
                )
            _verify_processed_pack_files(final_path, existing)
            shutil.rmtree(staging)
            return final_path
        staging.rename(final_path)
        return final_path
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def iter_processed_documents(path: Path) -> Iterator[ProcessedDocument]:
    yield from _iter_model_jsonl(path / "documents.jsonl", ProcessedDocument)


def iter_processed_queries(path: Path) -> Iterator[ProcessedQuery]:
    yield from _iter_model_jsonl(path / "queries.jsonl", ProcessedQuery)


def iter_processed_qrels(path: Path) -> Iterator[ProcessedQrel]:
    yield from _iter_model_jsonl(path / "qrels.jsonl", ProcessedQrel)


def verify_processed_pack(
    processed_path: Path,
    *,
    source_lock: SourceLock,
    expected_pack: ProcessedPackLock,
) -> ProcessedPackManifest:
    """Bind a local pack to reviewed provenance and content before any row is exposed."""
    expected_source_lock_hash = source_lock_sha256(source_lock)
    expected_preprocessing_hash = transformation_specification_sha256()
    if expected_pack.source_lock_sha256 != expected_source_lock_hash:
        raise DatasetPreparationError("processed pack lock belongs to a different source lock")
    if expected_pack.archive_sha256 != source_lock.archive.completed_download_sha256:
        raise DatasetPreparationError("processed pack lock archive SHA-256 drifted")
    if (
        expected_pack.preprocessing_sha256 != source_lock.preprocessing.specification_sha256
        or expected_pack.preprocessing_sha256 != expected_preprocessing_hash
    ):
        raise DatasetPreparationError("processed pack lock preprocessing SHA-256 drifted")
    expected_record_inventory = (
        ("documents.jsonl", source_lock.members["corpus"].records),
        ("queries.jsonl", source_lock.members["queries"].records),
        ("qrels.jsonl", source_lock.members["qrels"].records),
    )
    if (
        tuple((file.name, file.records) for file in expected_pack.files)
        != expected_record_inventory
    ):
        raise DatasetPreparationError("processed pack lock record inventory drifted")

    _require_exact_processed_pack_inventory(processed_path)
    manifest = _load_model_json(processed_path / "manifest.json", ProcessedPackManifest)
    if manifest.archive_sha256 != source_lock.archive.completed_download_sha256:
        raise DatasetPreparationError("processed manifest archive SHA-256 drifted")
    if manifest.source_lock_sha256 != expected_source_lock_hash:
        raise DatasetPreparationError("processed manifest source-lock SHA-256 drifted")
    if (
        manifest.preprocessing_version != source_lock.preprocessing.version
        or manifest.preprocessing_version != TRANSFORMATION_VERSION
    ):
        raise DatasetPreparationError("processed manifest preprocessing version drifted")
    if (
        manifest.preprocessing_sha256 != source_lock.preprocessing.specification_sha256
        or manifest.preprocessing_sha256 != expected_preprocessing_hash
    ):
        raise DatasetPreparationError("processed manifest preprocessing SHA-256 drifted")

    actual_files = _inspect_processed_pack_files(processed_path)
    if manifest.files != actual_files:
        raise DatasetPreparationError("processed manifest file identity does not match local rows")
    actual_content_sha256 = processed_content_sha256(
        source_lock_hash=expected_source_lock_hash,
        preprocessing_hash=expected_preprocessing_hash,
        files=actual_files,
    )
    if manifest.content_sha256 != actual_content_sha256:
        raise DatasetPreparationError("processed manifest content SHA-256 is not canonical")
    if expected_pack.content_sha256 != actual_content_sha256:
        raise DatasetPreparationError("processed rows do not match the reviewed pack identity")
    expected_basename = f"cqadupstack-unix-{actual_content_sha256}"
    if processed_path.name != expected_basename:
        raise DatasetPreparationError("processed pack directory is not its content address")
    return manifest


def load_curated_unix_corpus(
    processed_path: Path,
    *,
    source_lock: SourceLock,
    processed_pack_lock: ProcessedPackLock,
    dataset_manifest_path: Path,
    curated_manifest_path: Path,
) -> FixtureCorpus:
    """Load the ignored processed corpus plus only the checked-in curated query IDs."""
    verify_processed_pack(
        processed_path,
        source_lock=source_lock,
        expected_pack=processed_pack_lock,
    )
    dataset_manifest = _load_model_json(dataset_manifest_path, DatasetManifest)
    curated_manifest = _verify_curated_query_selection(
        processed_path,
        curated_manifest_path,
        expected_source_lock_hash=source_lock_sha256(source_lock),
    )

    documents = tuple(
        SourceDocument(
            external_id=document.external_id,
            title=document.title,
            body=document.body,
            source_url=document.source_url,
            attributes={"attribution": document.attribution.model_dump(mode="json")},
        )
        for document in iter_processed_documents(processed_path)
    )
    query_by_id = {query.external_id: query for query in iter_processed_queries(processed_path)}
    qrels_by_query: defaultdict[str, list[str]] = defaultdict(list)
    for qrel in iter_processed_qrels(processed_path):
        qrels_by_query[qrel.query_id].append(qrel.document_id)

    queries: list[FixtureQuery] = []
    for selection in curated_manifest.entries:
        query = query_by_id.get(selection.query_id)
        if query is None:
            raise DatasetPreparationError("curated query ID is absent from the processed pack")
        expected_ids = tuple(sorted(qrels_by_query[selection.query_id], key=_source_id_sort_key))
        if not expected_ids:
            raise DatasetPreparationError("curated query has no retained official judgments")
        queries.append(
            FixtureQuery(
                external_id=query.external_id,
                text=query.text,
                expected_external_ids=list(expected_ids),
            )
        )
    return FixtureCorpus(
        manifest=dataset_manifest,
        documents=documents,
        queries=tuple(queries),
        corpus_hash=corpus_hash(documents),
    )


def load_unix_dataset_manifest(path: Path) -> DatasetManifest:
    """Load the checked Unix schema/model manifest through the strict JSON boundary."""
    return _load_model_json(path, DatasetManifest)


def verify_curated_query_manifest(
    processed_path: Path,
    curated_manifest_path: Path,
    *,
    source_lock: SourceLock,
    processed_pack_lock: ProcessedPackLock,
) -> CuratedQueryManifest:
    """Recompute the checked-in ID-only curation against ignored local source text."""
    verify_processed_pack(
        processed_path,
        source_lock=source_lock,
        expected_pack=processed_pack_lock,
    )
    return _verify_curated_query_selection(
        processed_path,
        curated_manifest_path,
        expected_source_lock_hash=source_lock_sha256(source_lock),
    )


def _verify_curated_query_selection(
    processed_path: Path,
    curated_manifest_path: Path,
    *,
    expected_source_lock_hash: str,
) -> CuratedQueryManifest:
    curated_manifest = load_curated_query_manifest(curated_manifest_path)
    if curated_manifest.source_lock_sha256 != expected_source_lock_hash:
        raise DatasetPreparationError("curated query manifest belongs to a different source lock")
    queries = tuple(
        (query.external_id, query.text) for query in iter_processed_queries(processed_path)
    )
    qrels: defaultdict[str, list[str]] = defaultdict(list)
    for qrel in iter_processed_qrels(processed_path):
        qrels[qrel.query_id].append(qrel.document_id)
    expected = curate_query_ids(queries, qrels)
    if curated_manifest.entries != expected:
        raise DatasetPreparationError(
            "curated query manifest does not match deterministic four-stratum selection"
        )
    return curated_manifest


def curate_query_ids(
    queries: Iterable[tuple[str, str]],
    qrels: Mapping[str, Iterable[str]],
) -> tuple[CuratedQuery, ...]:
    """Reproduce the deterministic four-stratum, text-free checked-in selection."""
    candidates: list[tuple[str, tuple[CurationTag, ...]]] = []
    for query_id, text in queries:
        tags = _curation_tags(text, judgment_count=sum(1 for _ in qrels.get(query_id, ())))
        candidates.append((query_id, tags))

    selection_version = "pufferlab-curated-50-v1"
    tag_order: tuple[CurationTag, ...] = (
        "exact_token",
        "semantic",
        "hybrid",
        "reranker",
    )
    by_tag = {
        tag: sorted(
            (candidate for candidate in candidates if tag in candidate[1]),
            key=lambda candidate: hashlib.sha256(
                f"{selection_version}:{tag}:{candidate[0]}".encode()
            ).digest(),
        )
        for tag in tag_order
    }
    offsets: Counter[CurationTag] = Counter()
    selected: list[CuratedQuery] = []
    selected_ids: set[str] = set()
    while len(selected) < 50:
        made_progress = False
        for primary_tag in tag_order:
            pool = by_tag[primary_tag]
            offset = offsets[primary_tag]
            while offset < len(pool) and pool[offset][0] in selected_ids:
                offset += 1
            offsets[primary_tag] = offset
            if offset >= len(pool):
                continue
            query_id, tags = pool[offset]
            offsets[primary_tag] += 1
            selected_ids.add(query_id)
            selected.append(
                CuratedQuery(
                    query_id=query_id,
                    primary_tag=primary_tag,
                    tags=tags,
                    reason=_CURATION_REASONS[primary_tag],
                )
            )
            made_progress = True
            if len(selected) == 50:
                break
        if not made_progress:
            raise DatasetPreparationError("fewer than 50 queries satisfy the curation strata")
    return tuple(sorted(selected, key=lambda item: _source_id_sort_key(item.query_id)))


_CURATION_REASONS: dict[CurationTag, str] = {
    "exact_token": "Selected for lexical structure that makes exact-token retrieval observable.",
    "semantic": "Selected for multi-token natural-language intent suited to semantic retrieval.",
    "hybrid": "Selected for lexical anchors plus broader intent that make fusion observable.",
    "reranker": "Selected because multiple judgments make second-stage ordering measurable.",
}


def _process_verified_archive(
    verified: VerifiedArchive,
    staging: Path,
    source_lock: SourceLock,
) -> ProcessedPackManifest:
    member_paths = {member.path for member in source_lock.members.values()}
    try:
        archive = zipfile.ZipFile(verified.path)
    except (OSError, zipfile.BadZipFile) as error:
        raise DatasetPreparationError("verified archive is not a readable ZIP file") from error

    with archive, ExitStack() as stack:
        actual_members = set(archive.namelist())
        missing = member_paths - actual_members
        if missing:
            raise DatasetPreparationError("verified archive is missing a locked Unix member")
        for member in source_lock.members.values():
            info = archive.getinfo(member.path)
            if (
                info.file_size != member.uncompressed_bytes
                or info.compress_size != member.compressed_bytes
                or f"{info.CRC:08x}" != member.crc32
            ):
                raise DatasetPreparationError("verified archive member metadata drifted")

        document_output = stack.enter_context((staging / "documents.jsonl").open("wb"))
        query_output = stack.enter_context((staging / "queries.jsonl").open("wb"))
        qrel_output = stack.enter_context((staging / "qrels.jsonl").open("wb"))
        document_ids = _process_documents(archive, document_output)
        query_ids = _process_queries(archive, query_output)
        qrel_count = _process_qrels(archive, qrel_output, document_ids, query_ids)

    counts = {
        "documents.jsonl": len(document_ids),
        "queries.jsonl": len(query_ids),
        "qrels.jsonl": qrel_count,
    }
    expected_counts = {
        "documents.jsonl": source_lock.members["corpus"].records,
        "queries.jsonl": source_lock.members["queries"].records,
        "qrels.jsonl": source_lock.members["qrels"].records,
    }
    if counts != expected_counts:
        raise DatasetPreparationError("processed Unix record counts drifted from the source lock")

    files = tuple(
        ProcessedFile(name=name, records=counts[name], sha256=_file_sha256(staging / name))
        for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl")
    )
    locked_source_hash = source_lock_sha256(source_lock)
    preprocessing_hash = transformation_specification_sha256()
    manifest = ProcessedPackManifest(
        format_version=1,
        dataset=SOURCE_DATASET,
        subset=SOURCE_SUBSET,
        archive_sha256=verified.sha256,
        source_lock_sha256=locked_source_hash,
        preprocessing_version=TRANSFORMATION_VERSION,
        preprocessing_sha256=preprocessing_hash,
        content_sha256=processed_content_sha256(
            source_lock_hash=locked_source_hash,
            preprocessing_hash=preprocessing_hash,
            files=files,
        ),
        files=files,
    )
    (staging / "manifest.json").write_bytes(_canonical_json_line(manifest.model_dump(mode="json")))
    return manifest


def _process_documents(archive: zipfile.ZipFile, output: BinaryIO) -> set[str]:
    retained: set[str] = set()
    records: list[ProcessedDocument] = []
    with archive.open(CORPUS_MEMBER) as binary:
        for raw in io.TextIOWrapper(binary, encoding="utf-8", newline=""):
            value = _strict_json_line(raw)
            _require_exact_keys(value, {"_id", "title", "text", "metadata"}, "corpus")
            source_id = _required_source_id(value.get("_id"), "corpus")
            title = _normalized_text(value.get("title"), allow_blank=True, field="corpus title")
            body = _normalized_text(value.get("text"), allow_blank=False, field="corpus text")
            _require_discarded_metadata(value.get("metadata"), source="corpus")
            if source_id in retained:
                raise DatasetPreparationError("corpus contains a duplicate source ID")
            retained.add(source_id)
            source_url = _canonical_post_url(source_id)
            attribution = _attribution(source_id, _canonical_hash([title, body]))
            records.append(
                ProcessedDocument(
                    external_id=source_id,
                    title=title,
                    body=body,
                    source_url=source_url,
                    attribution=attribution,
                )
            )
    for record in sorted(records, key=lambda item: _source_id_sort_key(item.external_id)):
        output.write(_canonical_json_line(record.model_dump(mode="json")))
    return retained


def _process_queries(archive: zipfile.ZipFile, output: BinaryIO) -> set[str]:
    retained: set[str] = set()
    records: list[ProcessedQuery] = []
    with archive.open(QUERIES_MEMBER) as binary:
        for raw in io.TextIOWrapper(binary, encoding="utf-8", newline=""):
            value = _strict_json_line(raw)
            _require_exact_keys(value, {"_id", "text", "metadata"}, "queries")
            source_id = _required_source_id(value.get("_id"), "query")
            text = _normalized_text(value.get("text"), allow_blank=False, field="query text")
            _require_discarded_metadata(value.get("metadata"), source="queries")
            if source_id in retained:
                raise DatasetPreparationError("queries contain a duplicate source ID")
            retained.add(source_id)
            source_url = _canonical_post_url(source_id)
            records.append(
                ProcessedQuery(
                    external_id=source_id,
                    text=text,
                    source_url=source_url,
                    attribution=_attribution(source_id, _canonical_hash([text])),
                )
            )
    for record in sorted(records, key=lambda item: _source_id_sort_key(item.external_id)):
        output.write(_canonical_json_line(record.model_dump(mode="json")))
    return retained


def _process_qrels(
    archive: zipfile.ZipFile,
    output: BinaryIO,
    document_ids: set[str],
    query_ids: set[str],
) -> int:
    records: list[ProcessedQrel] = []
    seen: set[tuple[str, str]] = set()
    with archive.open(QRELS_MEMBER) as binary:
        wrapper = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        reader = csv.DictReader(wrapper, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise DatasetPreparationError("qrels header drifted from the pinned BEIR shape")
        for value in reader:
            if set(value) != {"query-id", "corpus-id", "score"}:
                raise DatasetPreparationError("qrels row drifted from the pinned BEIR shape")
            query_id = _required_source_id(value["query-id"], "qrel query")
            document_id = _required_source_id(value["corpus-id"], "qrel document")
            if query_id not in query_ids or document_id not in document_ids:
                raise DatasetPreparationError("qrel references a record that was not retained")
            identity = (query_id, document_id)
            if identity in seen:
                raise DatasetPreparationError("qrels contain a duplicate query-document pair")
            seen.add(identity)
            try:
                relevance = int(value["score"])
            except (TypeError, ValueError):
                raise DatasetPreparationError("qrel score is not an integer") from None
            records.append(
                ProcessedQrel(
                    query_id=query_id,
                    document_id=document_id,
                    relevance=relevance,
                )
            )
    for record in sorted(
        records,
        key=lambda item: (
            _source_id_sort_key(item.query_id),
            _source_id_sort_key(item.document_id),
        ),
    ):
        output.write(_canonical_json_line(record.model_dump(mode="json")))
    return len(records)


def _attribution(source_id: str, content_hash: str) -> Attribution:
    return Attribution(
        original_post_id=source_id,
        canonical_post_url=_canonical_post_url(source_id),
        content_hash=content_hash,
    )


def _canonical_post_url(source_id: str) -> str:
    return f"https://unix.stackexchange.com/questions/{source_id}"


def _curation_tags(text: str, *, judgment_count: int) -> tuple[CurationTag, ...]:
    import re

    tokens = re.findall(r"[A-Za-z0-9_./$@#:+-]+", text)
    codeish = re.compile(
        r"(?:[/\\$_.:=@#]|\b(?:sudo|grep|ssh|sed|awk|bash|shell|linux|unix|mount|chmod|"
        r"chown|apt|yum|tar|rsync|cron|regex|ip|dns|ssl|tcp|udp|git|vim|find|xargs)\b)",
        re.IGNORECASE,
    )
    exact = bool(
        codeish.search(text)
        or any(any(character.isdigit() for character in token) for token in tokens)
    )
    semantic = len(tokens) >= 5
    values: list[CurationTag] = []
    if exact:
        values.append("exact_token")
    if semantic:
        values.append("semantic")
    if exact and semantic:
        values.append("hybrid")
    if judgment_count >= 2:
        values.append("reranker")
    return tuple(values)


def _require_exact_processed_pack_inventory(path: Path) -> None:
    expected_names = {"documents.jsonl", "manifest.json", "qrels.jsonl", "queries.jsonl"}
    try:
        if path.is_symlink() or not path.is_dir():
            raise DatasetPreparationError("processed pack path must be a real directory")
        entries = tuple(path.iterdir())
    except OSError as error:
        raise DatasetPreparationError("processed pack directory is unreadable") from error
    if {entry.name for entry in entries} != expected_names:
        raise DatasetPreparationError("processed pack directory inventory is not exact")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise DatasetPreparationError("processed pack entries must be regular files")


def _inspect_processed_pack_files(path: Path) -> tuple[ProcessedFile, ...]:
    files: list[ProcessedFile] = []
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl"):
        candidate = path / name
        try:
            with candidate.open("rb") as stream:
                record_count = sum(1 for _ in stream)
        except OSError as error:
            raise DatasetPreparationError(f"processed file {name} is unreadable") from error
        if record_count < 1:
            raise DatasetPreparationError(f"processed file {name} contains no records")
        files.append(
            ProcessedFile(
                name=name,
                records=record_count,
                sha256=_file_sha256(candidate),
            )
        )
    return tuple(files)


def _verify_processed_pack_files(path: Path, manifest: ProcessedPackManifest) -> None:
    _require_exact_processed_pack_inventory(path)
    if _inspect_processed_pack_files(path) != manifest.files:
        raise DatasetPreparationError("content-addressed pack files do not match manifest")


def _iter_model_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> Iterator[ModelT]:
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DatasetPreparationError(f"processed file {path.name} is unreadable") from error
    with stream:
        for line_number, raw in enumerate(stream, start=1):
            try:
                yield model.model_validate(_strict_json_line(raw))
            except ValueError as error:
                raise DatasetPreparationError(
                    f"processed file {path.name}:{line_number} is invalid"
                ) from error


def _strict_json_line(raw: str) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        loaded: dict[str, object] = {}
        for key, value in pairs:
            if key in loaded:
                raise DatasetPreparationError("JSON record contains a duplicate key")
            loaded[key] = value
        return loaded

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: _raise_nonfinite(token),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetPreparationError("dataset member contains invalid JSON") from error
    if not isinstance(value, dict):
        raise DatasetPreparationError("dataset JSONL member must contain objects")
    return value


def _raise_nonfinite(token: str) -> None:
    raise DatasetPreparationError(f"dataset JSON contains non-finite number {token!r}")


def _required_source_id(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetPreparationError(f"{source} source ID is missing or invalid")
    return value


def _normalized_text(value: object, *, allow_blank: bool, field: str) -> str:
    if not isinstance(value, str):
        raise DatasetPreparationError(f"{field} is missing or invalid")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if not allow_blank and not normalized.strip():
        raise DatasetPreparationError(f"{field} must not be blank")
    return normalized


def _require_discarded_metadata(value: object, *, source: str) -> None:
    if not isinstance(value, dict):
        raise DatasetPreparationError(f"{source} metadata is missing or invalid")


def _require_exact_keys(value: Mapping[str, object], expected: set[str], source: str) -> None:
    if set(value) != expected:
        raise DatasetPreparationError(f"{source} record fields drifted from the pinned BEIR shape")


def _source_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_line(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DatasetPreparationError(f"could not hash processed file {path.name}") from error
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetPreparationError(f"could not read {path.name}") from error
    value = _strict_json_line(raw)
    return value


def _load_model_json[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetPreparationError(f"could not read {path.name}") from error
    _strict_json_line(raw)
    return model.model_validate_json(raw)
