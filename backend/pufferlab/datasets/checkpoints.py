"""Atomic local checkpoints for resumable stable-ID corpus upserts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pufferlab.datasets.ingestion import IngestionCheckpoint


class _CheckpointPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format_version: Literal[1]
    namespace: Annotated[str, Field(min_length=1)]
    dataset_version: Annotated[str, Field(min_length=1)]
    corpus_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    schema_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    completed_document_ids: tuple[UUID, ...]

    @field_validator("namespace", "dataset_version")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("completed_document_ids")
    @classmethod
    def require_canonical_unique_ids(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if len(value) != len(set(value)):
            raise ValueError("checkpoint document IDs must be unique")
        if value != tuple(sorted(value, key=str)):
            raise ValueError("checkpoint document IDs must be canonically ordered")
        return value


class IngestionCheckpointStore:
    """Persist checkpoints under one caller-chosen ignored data directory.

    This type deliberately exposes no namespace deletion operation. A caller-supplied namespace is
    used only as input to a fixed-length checkpoint filename hash and to bind resume validation.
    """

    def __init__(self, data_dir: Path) -> None:
        if not data_dir.is_absolute():
            raise ValueError("checkpoint data directory must be absolute")
        self._directory = data_dir / "ingestion-checkpoints"

    def save(self, checkpoint: IngestionCheckpoint) -> Path:
        payload = _CheckpointPayload(
            format_version=1,
            namespace=checkpoint.namespace,
            dataset_version=checkpoint.dataset_version,
            corpus_hash=checkpoint.corpus_hash,
            schema_hash=checkpoint.schema_hash,
            completed_document_ids=checkpoint.completed_document_ids,
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        target = self._path_for(
            namespace=payload.namespace,
            dataset_version=payload.dataset_version,
            corpus_hash=payload.corpus_hash,
            schema_hash=payload.schema_hash,
        )
        encoded = (
            json.dumps(
                payload.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._directory,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_descriptor = os.open(self._directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def load(
        self,
        *,
        namespace: str,
        dataset_version: str,
        corpus_hash: str,
        schema_hash: str,
    ) -> IngestionCheckpoint | None:
        target = self._path_for(
            namespace=namespace,
            dataset_version=dataset_version,
            corpus_hash=corpus_hash,
            schema_hash=schema_hash,
        )
        try:
            raw = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        payload = _CheckpointPayload.model_validate_json(raw)
        return IngestionCheckpoint(
            format_version=payload.format_version,
            namespace=payload.namespace,
            dataset_version=payload.dataset_version,
            corpus_hash=payload.corpus_hash,
            schema_hash=payload.schema_hash,
            completed_document_ids=payload.completed_document_ids,
        )

    def _path_for(
        self,
        *,
        namespace: str,
        dataset_version: str,
        corpus_hash: str,
        schema_hash: str,
    ) -> Path:
        identity = json.dumps(
            [namespace, dataset_version, corpus_hash, schema_hash],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return self._directory / f"{hashlib.sha256(identity).hexdigest()}.json"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"checkpoint contains duplicate key {key!r}")
        value[key] = item
    return value
