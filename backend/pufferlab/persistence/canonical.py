"""Canonical JSON and time encodings for durable contract payloads."""

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from pufferlab.persistence.errors import PersistenceValidationError


def canonical_json(value: BaseModel | Any) -> str:
    """Serialize JSON deterministically without accepting non-finite numbers."""
    serializable = _normalize_json_value(value)
    return json.dumps(
        serializable,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_utc(value: datetime, *, field_name: str) -> str:
    """Encode an aware datetime as a stable UTC ISO-8601 string."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise PersistenceValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize_json_value(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return canonical_utc(value, field_name="serialized datetime")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _normalize_json_value(value.value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value
