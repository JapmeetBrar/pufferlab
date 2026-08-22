"""Stable identifiers and content hashes for dataset revisions."""

import hashlib
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

if TYPE_CHECKING:
    from pufferlab.datasets.models import SourceDocument

# This namespace is a project constant. Changing it would change every document identity.
PUFFERLAB_NAMESPACE_UUID = UUID("7f52416d-2d83-5e8f-9cc4-2fce71e82c36")


def document_uuid(dataset_version: str, external_id: str) -> UUID:
    """Map a dataset revision and source identity to a stable UUIDv5."""
    if not dataset_version.strip():
        raise ValueError("dataset_version must not be blank")
    if not external_id.strip():
        raise ValueError("external_id must not be blank")
    return uuid5(PUFFERLAB_NAMESPACE_UUID, f"{dataset_version}:{external_id}")


def canonical_corpus_bytes(documents: Iterable["SourceDocument"]) -> bytes:
    """Serialize a corpus independently of JSONL order or formatting."""
    ordered = sorted(documents, key=lambda document: document.external_id)
    payload = {
        "format_version": 1,
        "documents": [document.model_dump(mode="json", exclude_none=False) for document in ordered],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{encoded}\n".encode()


def corpus_hash(documents: Iterable["SourceDocument"]) -> str:
    """Return the SHA-256 digest of the canonical corpus representation."""
    return hashlib.sha256(canonical_corpus_bytes(documents)).hexdigest()


def canonical_schema_hash(schema_payload: dict[str, object]) -> str:
    """Hash explicit index settings using canonical JSON."""
    encoded = json.dumps(
        schema_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
