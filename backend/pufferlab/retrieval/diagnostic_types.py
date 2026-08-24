"""Provider-neutral inputs and complete internal rows for one expected-document diagnostic."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Protocol
from uuid import UUID

from pufferlab.contracts.common import (
    JsonValue,
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterNode,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.contracts.forensics import DiagnosticSubqueryRole
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.providers.types import DistanceMetric, LexicalFieldWeights

_RESULT_K = 50
_CANDIDATE_K = 100
_MAX_FILTER_PREDICATES = 16
_MAX_FILTER_NODES = 31
_MAX_FILTER_DEPTH = 8
_MAX_ATTRIBUTE_DEPTH = 8
_MAX_ATTRIBUTE_OBJECT_ITEMS = 256
_MAX_ATTRIBUTE_ARRAY_ITEMS = 10_000
_MAX_DIAGNOSTIC_SCORE = 1_000_000_000_000.0
_MAX_DIAGNOSTIC_DURATION_MS = 600_000.0
_SAFE_FILTER_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_SAFE_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_REGION = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_RESERVED_DIAGNOSTIC_FIELDS = {
    "id",
    "__pufferlab_diagnostic_bm25",
    "__pufferlab_diagnostic_vector_distance",
}
_MISSING = object()


class DiagnosticAttributeState(StrEnum):
    MISSING = "missing"
    PRESENT_NULL = "present_null"
    PRESENT_VALUE = "present_value"


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticAttributeValue:
    """One exact-lookup attribute with provider field-presence preserved."""

    field: str
    state: DiagnosticAttributeState
    value: JsonValue | None = None

    def __post_init__(self) -> None:
        if type(self.field) is not str or _SAFE_FILTER_FIELD.fullmatch(self.field) is None:
            raise ValueError("diagnostic attribute field is invalid")
        if type(self.state) is not DiagnosticAttributeState:
            raise ValueError("diagnostic attribute state is invalid")
        if self.state is DiagnosticAttributeState.PRESENT_VALUE:
            if self.value is None:
                raise ValueError("present diagnostic attribute requires a value")
            _validate_json_value(self.value)
        elif self.value is not None:
            raise ValueError("missing/null diagnostic attribute cannot carry a value")


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticTargetObservation:
    """Target lookup result. Attribute values never cross the M5-C/D public boundary."""

    target_document_id: UUID
    available: bool
    bm25_score: ObservedScore | None
    vector_distance: ObservedScore | None
    attributes: tuple[DiagnosticAttributeValue, ...]

    def __post_init__(self) -> None:
        if type(self.target_document_id) is not UUID:
            raise ValueError("diagnostic target ID must be a UUID")
        if type(self.available) is not bool:
            raise ValueError("diagnostic target availability must be boolean")
        if not self.available and (
            self.bm25_score is not None or self.vector_distance is not None or self.attributes
        ):
            raise ValueError("unavailable diagnostic target cannot carry observed values")
        if type(self.attributes) is not tuple or not all(
            type(item) is DiagnosticAttributeValue for item in self.attributes
        ):
            raise ValueError("diagnostic target attributes are invalid")
        fields = tuple(item.field for item in self.attributes)
        if len(fields) != len(set(fields)):
            raise ValueError("diagnostic target attributes must be unique")
        if self.bm25_score is not None:
            _validate_observed_score(
                self.bm25_score,
                kind=ScoreKind.BM25,
                source=ScoreSource.COMPUTE_ATTRIBUTE,
                direction=ScoreDirection.HIGHER_IS_BETTER,
            )
        if self.vector_distance is not None:
            _validate_observed_score(
                self.vector_distance,
                kind=ScoreKind.VECTOR_DISTANCE,
                source=ScoreSource.COMPUTE_ATTRIBUTE,
                direction=ScoreDirection.LOWER_IS_BETTER,
            )


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticCandidateRow:
    document_id: UUID
    rank: int
    score: ObservedScore

    def __post_init__(self) -> None:
        if type(self.document_id) is not UUID:
            raise ValueError("diagnostic candidate ID must be a UUID")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("diagnostic candidate rank must be positive")
        _validate_observed_score(
            self.score,
            kind=None,
            source=ScoreSource.TURBOPUFFER_DIST,
            direction=None,
        )
        if self.score.kind not in {ScoreKind.BM25, ScoreKind.VECTOR_DISTANCE}:
            raise ValueError("diagnostic candidate score kind is invalid")
        if self.score.kind is ScoreKind.BM25 and self.score.value <= 0:
            raise ValueError("diagnostic BM25 candidate scores must be positive")


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticCandidateList:
    role: DiagnosticSubqueryRole
    requested_limit: int
    rows: tuple[DiagnosticCandidateRow, ...]

    def __post_init__(self) -> None:
        if type(self.role) is not DiagnosticSubqueryRole:
            raise ValueError("diagnostic candidate role is invalid")
        if isinstance(self.requested_limit, bool) or self.requested_limit not in {
            _RESULT_K,
            _CANDIDATE_K,
        }:
            raise ValueError("diagnostic candidate limit is invalid")
        if type(self.rows) is not tuple or not all(
            type(row) is DiagnosticCandidateRow for row in self.rows
        ):
            raise ValueError("diagnostic candidate rows are invalid")
        if len(self.rows) > self.requested_limit:
            raise ValueError("diagnostic candidate rows exceed the requested limit")
        if tuple(row.rank for row in self.rows) != tuple(range(1, len(self.rows) + 1)):
            raise ValueError("diagnostic candidate ranks must be contiguous")
        if len({row.document_id for row in self.rows}) != len(self.rows):
            raise ValueError("diagnostic candidate IDs must be unique")
        bm25 = self.role in {
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        }
        ann = self.role in {
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
        }
        if not bm25 and not ann:
            raise ValueError("diagnostic candidate role is invalid")
        expected_kind = ScoreKind.BM25 if bm25 else ScoreKind.VECTOR_DISTANCE
        expected_direction = (
            ScoreDirection.HIGHER_IS_BETTER if bm25 else ScoreDirection.LOWER_IS_BETTER
        )
        for row in self.rows:
            _validate_observed_score(
                row.score,
                kind=expected_kind,
                source=ScoreSource.TURBOPUFFER_DIST,
                direction=expected_direction,
            )
        if not monotonic(tuple(row.score.value for row in self.rows), descending=bm25):
            raise ValueError("diagnostic candidate scores must be monotonic")


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticProviderResult:
    namespace: str
    target: DiagnosticTargetObservation
    candidate_lists: tuple[DiagnosticCandidateList, ...]
    client_duration_ms: float

    def __post_init__(self) -> None:
        if not is_valid_diagnostic_namespace(self.namespace):
            raise ValueError("diagnostic result namespace is invalid")
        if type(self.target) is not DiagnosticTargetObservation:
            raise ValueError("diagnostic target result is invalid")
        if type(self.candidate_lists) is not tuple or not all(
            type(item) is DiagnosticCandidateList for item in self.candidate_lists
        ):
            raise ValueError("diagnostic candidate results are invalid")
        roles = tuple(item.role for item in self.candidate_lists)
        if len(roles) != len(set(roles)):
            raise ValueError("diagnostic candidate roles must be unique")
        if (
            type(self.client_duration_ms) not in {int, float}
            or not _is_finite_number(self.client_duration_ms)
            or float(self.client_duration_ms) < 0
            or float(self.client_duration_ms) > _MAX_DIAGNOSTIC_DURATION_MS
        ):
            raise ValueError("diagnostic client duration must be finite and nonnegative")


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticProviderRequest:
    namespace: str
    query_text: str
    target_document_id: UUID
    mode: RetrievalMode
    lexical_fields: LexicalFieldWeights | None
    vector_attribute: str | None
    query_vector: tuple[float, ...] | None
    distance_metric: DistanceMetric | None
    stored_filter: FilterNode | None
    include_no_filter_counterfactual: bool
    result_k: int = _RESULT_K
    candidate_k: int = _CANDIDATE_K

    def __post_init__(self) -> None:
        if not is_valid_diagnostic_namespace(self.namespace):
            raise ValueError("diagnostic namespace is invalid")
        if type(self.query_text) is not str or not self.query_text:
            raise ValueError("diagnostic query text is required")
        if type(self.target_document_id) is not UUID:
            raise ValueError("diagnostic target ID must be a UUID")
        if type(self.mode) is not RetrievalMode:
            raise ValueError("diagnostic retrieval mode is invalid")
        if type(self.include_no_filter_counterfactual) is not bool:
            raise ValueError("diagnostic no-filter option must be boolean")
        if (
            type(self.result_k) is not int
            or type(self.candidate_k) is not int
            or self.result_k != _RESULT_K
            or self.candidate_k != _CANDIDATE_K
        ):
            raise ValueError("diagnostic result/candidate bounds must be 50/100")
        if self.include_no_filter_counterfactual and self.stored_filter is None:
            raise ValueError("no-filter diagnostic requires one stored query filter")
        diagnostic_filter_fields(self.stored_filter)

        lexical_required = self.mode in {
            RetrievalMode.BM25,
            RetrievalMode.HYBRID_RRF,
            RetrievalMode.HYBRID_RERANK,
        }
        vector_required = self.mode in {
            RetrievalMode.VECTOR,
            RetrievalMode.HYBRID_RRF,
            RetrievalMode.HYBRID_RERANK,
        }
        if (self.lexical_fields is not None) is not lexical_required:
            raise ValueError("diagnostic lexical inputs do not match the selected mode")
        if lexical_required:
            lexical_fields = self.lexical_fields
            if (
                type(lexical_fields) is not tuple
                or not lexical_fields
                or not all(type(item) is tuple and len(item) == 2 for item in lexical_fields)
            ):
                raise ValueError("diagnostic lexical fields must be nonempty and unique")
            for field, weight in lexical_fields:
                if (
                    type(field) is not str
                    or not field
                    or type(weight) not in {int, float}
                    or not _is_finite_number(weight)
                    or float(weight) <= 0
                ):
                    raise ValueError("diagnostic lexical field weights are invalid")
            if len({field for field, _ in lexical_fields}) != len(lexical_fields):
                raise ValueError("diagnostic lexical fields must be nonempty and unique")

        if (
            (self.vector_attribute is not None)
            and (self.query_vector is not None)
            and (self.distance_metric is not None)
        ) is not vector_required:
            raise ValueError("diagnostic vector inputs do not match the selected mode")
        if vector_required:
            vector_attribute = self.vector_attribute
            query_vector = self.query_vector
            if (
                type(vector_attribute) is not str
                or not vector_attribute
                or type(query_vector) is not tuple
                or not query_vector
                or type(self.distance_metric) is not str
                or self.distance_metric not in {"cosine_distance", "euclidean_squared"}
            ):
                raise ValueError("diagnostic vector inputs must be nonempty")
            if any(
                type(value) not in {int, float} or not _is_finite_number(value)
                for value in query_vector
            ):
                raise ValueError("diagnostic query vector must be finite")

    @property
    def roles(self) -> tuple[DiagnosticSubqueryRole, ...]:
        return diagnostic_roles(self.mode, self.include_no_filter_counterfactual)

    @property
    def candidate_limit(self) -> int:
        return (
            self.result_k
            if self.mode in {RetrievalMode.BM25, RetrievalMode.VECTOR}
            else self.candidate_k
        )

    @property
    def filter_fields(self) -> tuple[str, ...]:
        return diagnostic_filter_fields(self.stored_filter)


class DiagnosticProvider(Protocol):
    async def query(self, request: DiagnosticProviderRequest) -> DiagnosticProviderResult: ...

    async def close(self) -> None: ...


def diagnostic_roles(
    mode: RetrievalMode,
    include_no_filter_counterfactual: bool,
) -> tuple[DiagnosticSubqueryRole, ...]:
    stored = {
        RetrievalMode.BM25: (DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,),
        RetrievalMode.VECTOR: (DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,),
        RetrievalMode.HYBRID_RRF: (
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ),
        RetrievalMode.HYBRID_RERANK: (
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        ),
    }[mode]
    if not include_no_filter_counterfactual:
        return (DiagnosticSubqueryRole.TARGET_LOOKUP, *stored)
    no_filter = tuple(
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES
        if "bm25" in role.value
        else DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES
        for role in stored
    )
    return (DiagnosticSubqueryRole.TARGET_LOOKUP, *stored, *no_filter)


def diagnostic_filter_fields(node: FilterNode | None) -> tuple[str, ...]:
    if node is None:
        return ()
    fields: list[str] = []
    seen: set[str] = set()
    predicate_count = 0
    node_count = 0
    stack: list[tuple[FilterNode, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_FILTER_NODES or depth > _MAX_FILTER_DEPTH:
            raise ValueError("diagnostic stored filter exceeds the finite node/depth bound")
        if type(current) is FilterPredicate:
            predicate_count += 1
            if predicate_count > _MAX_FILTER_PREDICATES:
                raise ValueError("diagnostic stored filter exceeds the predicate bound")
            field = getattr(current, "field", _MISSING)
            op = getattr(current, "op", _MISSING)
            value = getattr(current, "value", _MISSING)
            kind = getattr(current, "kind", _MISSING)
            if (
                kind != "predicate"
                or type(field) is not str
                or _SAFE_FILTER_FIELD.fullmatch(field) is None
                or field in _RESERVED_DIAGNOSTIC_FIELDS
                or type(op) is not PredicateOp
                or value is _MISSING
            ):
                raise ValueError("diagnostic stored filter predicate is invalid")
            _validate_predicate_value(op, value)
            if field not in seen:
                seen.add(field)
                fields.append(field)
            continue
        if type(current) is not FilterLogical:
            raise ValueError("diagnostic stored filter contains an unsupported node")
        kind = getattr(current, "kind", _MISSING)
        op = getattr(current, "op", _MISSING)
        children = getattr(current, "children", _MISSING)
        if (
            kind != "logical"
            or type(op) is not LogicalOp
            or type(children) is not list
            or not children
            or len(children) > _MAX_FILTER_NODES
            or (op is LogicalOp.NOT and len(children) != 1)
        ):
            raise ValueError("diagnostic stored filter logical node is invalid")
        if not all(type(child) in {FilterPredicate, FilterLogical} for child in children):
            raise ValueError("diagnostic stored filter logical children are invalid")
        stack.extend((child, depth + 1) for child in reversed(children))
    return tuple(fields)


def is_valid_diagnostic_namespace(value: object) -> bool:
    return (
        type(value) is str
        and value not in {".", ".."}
        and _SAFE_NAMESPACE.fullmatch(value) is not None
    )


def is_valid_diagnostic_region(value: object) -> bool:
    """Return whether a stored diagnostic region is one exact official DNS label."""

    return type(value) is str and _SAFE_REGION.fullmatch(value) is not None


def _validate_predicate_value(op: PredicateOp, value: object) -> None:
    _validate_json_value(value)
    if op in {PredicateOp.IN, PredicateOp.CONTAINS_ANY}:
        if type(value) is not list:
            raise ValueError("diagnostic stored filter array operand is invalid")
        scalar_types = {_json_scalar_type(item) for item in value}
        if None in scalar_types or len(scalar_types) > 1:
            raise ValueError("diagnostic stored filter array operands must share one scalar type")
        return
    if type(value) in {list, dict}:
        raise ValueError("diagnostic stored filter scalar operand is invalid")
    if op in {PredicateOp.LT, PredicateOp.LTE, PredicateOp.GT, PredicateOp.GTE} and (
        value is None or type(value) not in {int, float, str}
    ):
        raise ValueError("diagnostic stored filter range operand is invalid")


def _json_scalar_type(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is bool:
        return "bool"
    if type(value) in {int, float}:
        return "number"
    if type(value) is str:
        return "string"
    return None


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_ATTRIBUTE_DEPTH:
        raise ValueError("diagnostic JSON value exceeds the depth bound")
    if value is None or type(value) in {str, bool}:
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("diagnostic JSON value must be finite")
        return
    if type(value) is list:
        if len(value) > _MAX_ATTRIBUTE_ARRAY_ITEMS:
            raise ValueError("diagnostic JSON array exceeds the item bound")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > _MAX_ATTRIBUTE_OBJECT_ITEMS or not all(type(key) is str for key in value):
            raise ValueError("diagnostic JSON object is invalid")
        for item in value.values():
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("diagnostic value is not strict JSON")


def _validate_observed_score(
    score: object,
    *,
    kind: ScoreKind | None,
    source: ScoreSource,
    direction: ScoreDirection | None,
) -> None:
    if type(score) is not ObservedScore:
        raise ValueError("diagnostic score is invalid")
    observed_kind = getattr(score, "kind", _MISSING)
    observed_value = getattr(score, "value", _MISSING)
    observed_direction = getattr(score, "direction", _MISSING)
    observed_source = getattr(score, "source", _MISSING)
    if (
        not isinstance(observed_kind, ScoreKind)
        or not isinstance(observed_direction, ScoreDirection)
        or not isinstance(observed_source, ScoreSource)
        or (kind is not None and observed_kind is not kind)
        or observed_source is not source
        or (direction is not None and observed_direction is not direction)
    ):
        raise ValueError("diagnostic score metadata is invalid")
    expected_direction = (
        ScoreDirection.LOWER_IS_BETTER
        if observed_kind is ScoreKind.VECTOR_DISTANCE
        else ScoreDirection.HIGHER_IS_BETTER
    )
    if observed_direction is not expected_direction:
        raise ValueError("diagnostic score direction is invalid")
    require_finite_nonnegative(observed_value)


def require_exact_uuid(value: object) -> UUID:
    if not isinstance(value, str):
        raise ValueError("diagnostic row ID must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError("diagnostic row ID must be a canonical UUID string") from error
    if str(parsed) != value:
        raise ValueError("diagnostic row ID must be a canonical UUID string")
    return parsed


def require_finite_nonnegative(value: object, *, positive: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("diagnostic score must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("diagnostic score must be finite") from error
    if (
        not math.isfinite(result)
        or result < 0
        or result > _MAX_DIAGNOSTIC_SCORE
        or (positive and result == 0)
    ):
        raise ValueError("diagnostic score is outside its finite domain")
    return result


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def monotonic(values: Sequence[float], *, descending: bool) -> bool:
    pairs = pairwise(values)
    return all(left >= right if descending else left <= right for left, right in pairs)
