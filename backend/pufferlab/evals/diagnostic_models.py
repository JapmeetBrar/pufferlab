"""Provider-neutral, bounded inputs for pure expected-document analysis."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pufferlab.contracts.forensics import (
    CandidateCutoffEvidence,
    DiagnosticCandidateSubquerySummary,
    DiagnosticSubqueryRole,
    DiagnosticTargetLookup,
    DiagnosticTargetLookupSubquerySummary,
    FilterPredicateEvidence,
    ForensicObservation,
    QualifiedRrfEvidence,
)
from pufferlab.contracts.retrieval import RetrievalMode


class DiagnosticAnalysisErrorCode(StrEnum):
    """Value-free reasons why pure diagnostic analysis failed closed."""

    INVALID_BINDING = "invalid_binding"
    INVALID_MODE = "invalid_mode"
    INVALID_TARGET = "invalid_target"
    INVALID_FILTER = "invalid_filter"
    INVALID_CANDIDATES = "invalid_candidates"
    INVALID_RRF = "invalid_rrf"
    INVALID_OUTPUT = "invalid_output"


class DiagnosticAnalysisError(ValueError):
    """A bounded analysis failure that retains no submitted evidence."""

    __slots__ = ("code",)

    def __init__(self, code: DiagnosticAnalysisErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class TruthValue(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class AttributePresence(StrEnum):
    MISSING = "missing"
    PRESENT_NULL = "present_null"
    PRESENT_VALUE = "present_value"


class FilterValueType(StrEnum):
    STRING = "string"
    DATETIME = "datetime"
    UUID = "uuid"
    BOOL = "bool"
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    STRING_ARRAY = "[]string"
    DATETIME_ARRAY = "[]datetime"
    UUID_ARRAY = "[]uuid"
    BOOL_ARRAY = "[]bool"
    INT_ARRAY = "[]int"
    UINT_ARRAY = "[]uint"
    FLOAT_ARRAY = "[]float"


@dataclass(frozen=True, slots=True)
class DiagnosticBinding:
    config_id: UUID
    target_document_id: UUID
    observed_at: datetime
    trace_id: UUID


@dataclass(frozen=True, slots=True)
class PreservedAttribute:
    """One SDK-decoded field with omitted/null/value presence retained exactly."""

    presence: AttributePresence
    value: object = dataclass_field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class FilterFieldSchema:
    field: str
    value_type: FilterValueType
    filterable: bool


@dataclass(frozen=True, slots=True)
class ObservedFilterAttribute:
    field: str
    attribute: PreservedAttribute = dataclass_field(repr=False)


@dataclass(frozen=True, slots=True)
class FilterDefinitionInput:
    node: object = dataclass_field(repr=False)
    schema: tuple[FilterFieldSchema, ...]


@dataclass(frozen=True, slots=True)
class FilterAnalysisInput(FilterDefinitionInput):
    attributes: tuple[ObservedFilterAttribute, ...] = dataclass_field(repr=False)


@dataclass(frozen=True, slots=True)
class TargetLookupInput:
    available: bool
    bm25_score: float | None = None
    vector_distance: float | None = None


@dataclass(frozen=True, slots=True)
class CandidateRow:
    document_id: UUID = dataclass_field(repr=False)
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class CandidateListInput:
    ordinal: int
    role: DiagnosticSubqueryRole
    requested_limit: int
    rows: tuple[CandidateRow, ...] = dataclass_field(repr=False)


@dataclass(frozen=True, slots=True)
class RrfInputs:
    bm25_weight: float
    ann_weight: float
    rank_constant: int
    cutoff: int = 50


@dataclass(frozen=True, slots=True)
class DiagnosticAnalysisInput:
    binding: DiagnosticBinding
    mode: RetrievalMode
    include_no_filter_counterfactual: bool
    target: TargetLookupInput = dataclass_field(repr=False)
    candidate_lists: tuple[CandidateListInput, ...] = dataclass_field(repr=False)
    stored_filter: FilterAnalysisInput | None = dataclass_field(default=None, repr=False)
    rrf: RrfInputs | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticAnalysisResult:
    """Target-only facts ready for M5-D source binding and response construction."""

    filter_root_result: TruthValue | None
    target: DiagnosticTargetLookup
    subqueries: tuple[
        DiagnosticTargetLookupSubquerySummary | DiagnosticCandidateSubquerySummary,
        ...,
    ]
    filter_evidence: tuple[FilterPredicateEvidence, ...]
    candidate_evidence: tuple[CandidateCutoffEvidence, ...]
    qualified_rrf_evidence: tuple[QualifiedRrfEvidence, ...]
    observations: tuple[ForensicObservation, ...]
