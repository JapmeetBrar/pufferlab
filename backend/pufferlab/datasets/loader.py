"""Strict JSON and JSONL loading for local fixture packs."""

import json
import math
from pathlib import Path
from typing import NoReturn

from pydantic import BaseModel, ValidationError

from pufferlab.datasets.identity import corpus_hash
from pufferlab.datasets.models import DatasetManifest, FixtureCorpus, FixtureQuery, SourceDocument


class DatasetLoadError(ValueError):
    """A dataset pack is malformed or internally inconsistent."""


def load_fixture_corpus(directory: Path) -> FixtureCorpus:
    manifest = _load_json(directory / "manifest.json", DatasetManifest)
    documents = _load_jsonl(directory / "documents.jsonl", SourceDocument)
    queries = _load_jsonl(directory / "queries.jsonl", FixtureQuery)

    _require_unique(documents, field_name="external_id", source="documents.jsonl")
    _require_unique(queries, field_name="external_id", source="queries.jsonl")
    document_ids = {document.external_id for document in documents}
    for query in queries:
        missing = sorted(set(query.expected_external_ids) - document_ids)
        if missing:
            joined = ", ".join(missing)
            raise DatasetLoadError(
                f"queries.jsonl query {query.external_id!r} references missing documents: {joined}"
            )

    ordered_documents = tuple(sorted(documents, key=lambda item: item.external_id))
    ordered_queries = tuple(sorted(queries, key=lambda item: item.external_id))
    return FixtureCorpus(
        manifest=manifest,
        documents=ordered_documents,
        queries=ordered_queries,
        corpus_hash=corpus_hash(ordered_documents),
    )


def _load_json[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        raw = path.read_text(encoding="utf-8")
        value = _strict_json_loads(raw)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise DatasetLoadError(f"could not load {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetLoadError(f"{path.name} must contain one JSON object")
    try:
        return model_type.model_validate(value)
    except ValidationError as error:
        raise DatasetLoadError(f"invalid {path.name}: {error}") from error


def _load_jsonl[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetLoadError(f"could not load {path.name}: {error}") from error
    if not lines:
        raise DatasetLoadError(f"{path.name} must not be empty")

    loaded: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetLoadError(f"{path.name}:{line_number} blank lines are not allowed")
        try:
            value = _strict_json_loads(line)
        except ValueError as error:
            raise DatasetLoadError(f"invalid JSON at {path.name}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise DatasetLoadError(f"{path.name}:{line_number} must contain a JSON object")
        try:
            loaded.append(model_type.model_validate(value))
        except ValidationError as error:
            raise DatasetLoadError(
                f"invalid record at {path.name}:{line_number}: {error}"
            ) from error
    return tuple(loaded)


def _require_unique[ModelT: BaseModel](
    values: tuple[ModelT, ...],
    *,
    field_name: str,
    source: str,
) -> None:
    seen: set[object] = set()
    for value in values:
        candidate = getattr(value, field_name)
        if candidate in seen:
            raise DatasetLoadError(f"{source} contains duplicate {field_name}: {candidate!r}")
        seen.add(candidate)


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    loaded: dict[str, object] = {}
    for key, value in pairs:
        if key in loaded:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        loaded[key] = value
    return loaded


def _reject_nonfinite_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number produced by numeric overflow is not allowed")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_nonfinite_values(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite_values(nested)


def _strict_json_loads(raw: str) -> object:
    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    _reject_nonfinite_values(value)
    return value
