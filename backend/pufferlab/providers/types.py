"""Provider-facing types that keep SDK details behind a narrow boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Required, TypedDict

from pufferlab.contracts.common import JsonValue, ObservedScore

type ConsistencyLevel = Literal["strong", "eventual"]
type DistanceMetric = Literal["cosine_distance", "euclidean_squared"]
type DocumentId = str | int
type LexicalFieldWeights = tuple[tuple[str, float], ...]


class FullTextSearchSchema(TypedDict, total=False):
    tokenizer: str
    case_sensitive: bool
    language: str
    stemming: bool
    remove_stopwords: bool
    ascii_folding: bool
    max_token_length: int
    k1: float
    b: float
    k3: float


class AnnIndexSchema(TypedDict, total=False):
    distance_metric: DistanceMetric


class AttributeSchema(TypedDict, total=False):
    """The supported, provider-neutral subset of turbopuffer attribute schema."""

    type: Required[str]
    filterable: bool
    full_text_search: bool | FullTextSearchSchema
    ann: bool | AnnIndexSchema


type ProviderAttributeSchema = str | AttributeSchema
type ProviderSchema = Mapping[str, ProviderAttributeSchema]


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
class ProviderHybridProbeResult:
    """Debug-only raw lists from one same-snapshot BM25/ANN multi-query."""

    bm25_documents: tuple[ProviderDocument, ...]
    ann_documents: tuple[ProviderDocument, ...]
    client_duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderWriteResult:
    rows_affected: int
    client_duration_ms: float


@dataclass(frozen=True, slots=True)
class ProviderDocumentIdInventory:
    """Strong-consistency ID inventory bounded one row beyond the expected corpus size."""

    document_ids: tuple[DocumentId, ...]
    document_count: int
    truncated: bool
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
