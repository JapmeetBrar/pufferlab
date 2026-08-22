import json
import shutil
from pathlib import Path

import pytest
from pufferlab.datasets.identity import corpus_hash, document_uuid
from pufferlab.datasets.loader import DatasetLoadError, load_fixture_corpus

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "tiny-corpus"


def test_tiny_corpus_has_golden_hash_and_ids() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)

    assert len(corpus.documents) == 20
    assert len(corpus.queries) == 5
    assert corpus.corpus_hash == "fc7817ade91368cef13c52e48d4dc154189ec3ba803a4e293407dd7c09242512"
    assert str(corpus.document_id("tiny-001")) == "82b689d4-74ea-5fe6-a7e0-9fcbacf0244e"
    assert str(corpus.document_id("tiny-005")) == "d6b469d7-1f16-5eda-bdff-244250161fa0"
    assert str(corpus.document_id("tiny-020")) == "8ad15114-c3b1-52ce-a882-0bbacf892843"
    assert corpus.manifest.schema_hash == (
        "606fa669dfb2ac63494a8228260d46fac3bb1e4d3b3f13d3a6db3f9f6ee19e41"
    )
    assert corpus_hash(reversed(corpus.documents)) == corpus.corpus_hash


def test_document_uuid_includes_dataset_version() -> None:
    first = document_uuid("version-one", "source-123")

    assert first == document_uuid("version-one", "source-123")
    assert first != document_uuid("version-two", "source-123")


def test_all_query_expectations_map_to_loaded_documents() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)
    ingested_ids = {corpus.document_id(document.external_id) for document in corpus.documents}

    for expected_ids in corpus.expected_document_ids.values():
        assert expected_ids
        assert set(expected_ids) <= ingested_ids


def test_loader_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    documents_path = fixture / "documents.jsonl"
    lines = documents_path.read_text(encoding="utf-8").splitlines()
    documents_path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="duplicate external_id: 'tiny-001'"):
        load_fixture_corpus(fixture)


def test_loader_rejects_unknown_jsonl_fields(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    documents_path = fixture / "documents.jsonl"
    lines = documents_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["raw_vector"] = [0.1, 0.2]
    lines[0] = json.dumps(first)
    documents_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Extra inputs are not permitted"):
        load_fixture_corpus(fixture)


def test_loader_rejects_missing_query_targets(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    queries_path = fixture / "queries.jsonl"
    queries = queries_path.read_text(encoding="utf-8").replace("tiny-005", "does-not-exist", 1)
    queries_path.write_text(queries, encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="references missing documents: does-not-exist"):
        load_fixture_corpus(fixture)


def test_loader_rejects_blank_jsonl_lines(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    documents_path = fixture / "documents.jsonl"
    content = documents_path.read_text(encoding="utf-8")
    documents_path.write_text(content.replace("\n", "\n\n", 1), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="blank lines are not allowed"):
        load_fixture_corpus(fixture)


def test_loader_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    documents_path = fixture / "documents.jsonl"
    content = documents_path.read_text(encoding="utf-8")
    documents_path.write_text(
        content.replace('"synthetic":true', '"weight":NaN,"synthetic":true', 1),
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="non-finite JSON number 'NaN' is not allowed"):
        load_fixture_corpus(fixture)


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination
