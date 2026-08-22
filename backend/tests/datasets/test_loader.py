import json
import math
import shutil
from pathlib import Path

import pytest
from pufferlab.datasets.identity import canonical_schema_hash, corpus_hash, document_uuid
from pufferlab.datasets.loader import DatasetLoadError, load_fixture_corpus
from pufferlab.datasets.models import SourceDocument
from pufferlab.datasets.schema import compile_namespace_write_spec
from pydantic import ValidationError

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "tiny-corpus"


def test_tiny_corpus_has_golden_hash_and_ids() -> None:
    corpus = load_fixture_corpus(FIXTURE_ROOT)

    assert len(corpus.documents) == 20
    assert len(corpus.queries) == 5
    assert corpus.corpus_hash == "fc7817ade91368cef13c52e48d4dc154189ec3ba803a4e293407dd7c09242512"
    assert str(corpus.document_id("tiny-001")) == "82b689d4-74ea-5fe6-a7e0-9fcbacf0244e"
    assert str(corpus.document_id("tiny-005")) == "d6b469d7-1f16-5eda-bdff-244250161fa0"
    assert str(corpus.document_id("tiny-020")) == "8ad15114-c3b1-52ce-a882-0bbacf892843"
    assert compile_namespace_write_spec(corpus.manifest).schema_hash == (
        "0251f57f6166bf8f1ab8351ae0a4a797cfcf691fb0699bcfc59a4083945eea1d"
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


@pytest.mark.parametrize(
    ("file_name", "old", "new"),
    [
        (
            "manifest.json",
            '"dimensions": 384',
            '"dimensions": 384, "dimensions": 12',
        ),
        (
            "documents.jsonl",
            '"topic":"find"',
            '"topic":"find","topic":"duplicate"',
        ),
        (
            "queries.jsonl",
            '"external_id":"query-001"',
            '"external_id":"query-001","external_id":"duplicate"',
        ),
    ],
)
def test_loader_rejects_duplicate_keys_at_every_json_level(
    tmp_path: Path,
    file_name: str,
    old: str,
    new: str,
) -> None:
    fixture = _copy_fixture(tmp_path)
    path = fixture / file_name
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="duplicate JSON object key"):
        load_fixture_corpus(fixture)


@pytest.mark.parametrize(
    ("file_name", "old", "constant"),
    [
        ("manifest.json", '"k1": 1.2', '"k1": {value}'),
        ("documents.jsonl", '"synthetic":true', '"weight":{value},"synthetic":true'),
        ("queries.jsonl", '"text":', '"weight":{value},"text":'),
    ],
)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_literal_nonfinite_numbers_in_every_file(
    tmp_path: Path,
    file_name: str,
    old: str,
    constant: str,
    value: str,
) -> None:
    fixture = _copy_fixture(tmp_path)
    path = fixture / file_name
    replacement = constant.format(value=value)
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError, match="non-finite JSON number"):
        load_fixture_corpus(fixture)


@pytest.mark.parametrize(
    ("file_name", "old", "new"),
    [
        ("manifest.json", '"k1": 1.2', '"k1": 1e999'),
        ("documents.jsonl", '"synthetic":true', '"weight":1e999,"synthetic":true'),
        ("queries.jsonl", '"text":', '"weight":1e999,"text":'),
    ],
)
def test_loader_rejects_exponent_overflow_in_every_file(
    tmp_path: Path,
    file_name: str,
    old: str,
    new: str,
) -> None:
    fixture = _copy_fixture(tmp_path)
    path = fixture / file_name
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="numeric overflow"):
        load_fixture_corpus(fixture)


def test_models_reject_programmatic_nonfinite_nested_values() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        SourceDocument(
            external_id="test",
            title="Title",
            body="Body",
            source_url="https://example.com/test",
            attributes={"nested": {"weight": math.inf}},
        )


def test_canonical_hashes_refuse_nonfinite_values_even_if_validation_is_bypassed() -> None:
    document = SourceDocument.model_construct(
        external_id="test",
        title="Title",
        body="Body",
        source_url="https://example.com/test",
        attributes={"weight": math.nan},
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        corpus_hash([document])
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_schema_hash({"weight": math.inf})


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination
