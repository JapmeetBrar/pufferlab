"""Canonical JSON and time encodings for durable contract payloads."""

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from pufferlab.persistence.errors import PersistenceValidationError


def canonical_json(value: BaseModel | Any) -> str:
    """Serialize JSON deterministically without accepting non-finite numbers."""
    serializable = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
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
