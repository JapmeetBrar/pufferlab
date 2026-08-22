"""Provider-facing types that keep SDK details behind a narrow boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from turbopuffer.types import AttributeSchemaParam

from pufferlab.contracts.common import JsonValue, ObservedScore

type ConsistencyLevel = Literal["strong", "eventual"]
type DistanceMetric = Literal["cosine_distance", "euclidean_squared"]
type DocumentId = str | int
type ProviderSchema = Mapping[str, AttributeSchemaParam]


@dataclass(frozen=True, slots=True)
class WriteDocument:
    """A complete document upsert.

    Values may include a vector, but callers must not log this object or expose it through API
    responses.
    """

    id: DocumentId
    attributes: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if "id" in self.attributes:
            raise ValueError("document attributes must not override id")


@dataclass(frozen=True, slots=True)
class ProviderDocument:
    """A query row with vectors and provider-only fields removed."""

    id: DocumentId
    attributes: Mapping[str, JsonValue]
    score: ObservedScore


@dataclass(frozen=True, slots=True)
class ProviderQueryResult:
    documents: tuple[ProviderDocument, ...]
    client_duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderWriteResult:
    rows_affected: int
    client_duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderDeleteResult:
    client_duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderNamespaceMetadata:
    approx_row_count: int
    index_status: Literal["updating", "up-to-date"]
    unindexed_bytes: int | None
    schema: Mapping[str, JsonValue]
    client_duration_ms: float

    @property
    def ready(self) -> bool:
        return self.index_status == "up-to-date"
