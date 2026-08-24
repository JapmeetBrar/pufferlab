"""Pure target-scoped filter, cutoff, and qualified-RRF analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from typing import NoReturn, TypeGuard
from uuid import UUID

from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.contracts.forensics import (
    CandidateCutoffEvidence,
    CutoffRelationEvidenceValue,
    DiagnosticCandidateScope,
    DiagnosticCandidateSubquerySummary,
    DiagnosticCutoffRelation,
    DiagnosticPredicateResult,
    DiagnosticSignal,
    DiagnosticSubqueryRole,
    DiagnosticTargetLookup,
    DiagnosticTargetLookupSubquerySummary,
    DiagnosticTargetUnavailableReason,
    DirectScoreEvidenceValue,
    EvidenceCertainty,
    EvidenceItem,
    EvidenceOrigin,
    FilterPredicateEvidence,
    FilterPredicateEvidenceValue,
    ForensicCode,
    ForensicObservation,
    QualifiedRrfEvidence,
    RrfContributionEvidenceValue,
    ScoreEvidenceValue,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.contracts.search import RetrievalStage
from pufferlab.evals.diagnostic_models import (
    AttributePresence,
    CandidateListInput,
    CandidateRow,
    DiagnosticAnalysisError,
    DiagnosticAnalysisErrorCode,
    DiagnosticAnalysisInput,
    DiagnosticAnalysisResult,
    DiagnosticBinding,
    FilterAnalysisInput,
    FilterDefinitionInput,
    FilterFieldSchema,
    FilterValueType,
    ObservedFilterAttribute,
    PreservedAttribute,
    RrfInputs,
    TargetLookupInput,
    TruthValue,
)

_MAX_FILTER_PREDICATES = 16
_MAX_FILTER_NODES = 31
_MAX_FILTER_DEPTH = 8
_MAX_FILTER_COLLECTION_VALUES = 10_000
_MAX_SCORE = 1_000_000_000_000.0
_SCORE_REL_TOL = 1e-12
_SCORE_ABS_TOL = 1e-15
_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_DATETIME_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"(?:T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?P<fraction>\.\d{1,9})?(?P<zone>Z|[+-]\d{2}:\d{2})?)?$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_ARRAY_TYPES = {
    FilterValueType.STRING_ARRAY,
    FilterValueType.DATETIME_ARRAY,
    FilterValueType.UUID_ARRAY,
    FilterValueType.BOOL_ARRAY,
    FilterValueType.INT_ARRAY,
    FilterValueType.UINT_ARRAY,
    FilterValueType.FLOAT_ARRAY,
}
_ORDERABLE_TYPES = {
    FilterValueType.STRING,
    FilterValueType.DATETIME,
    FilterValueType.INT,
    FilterValueType.UINT,
    FilterValueType.FLOAT,
}
_SCALAR_FOR_ARRAY = {
    FilterValueType.STRING_ARRAY: FilterValueType.STRING,
    FilterValueType.DATETIME_ARRAY: FilterValueType.DATETIME,
    FilterValueType.UUID_ARRAY: FilterValueType.UUID,
    FilterValueType.BOOL_ARRAY: FilterValueType.BOOL,
    FilterValueType.INT_ARRAY: FilterValueType.INT,
    FilterValueType.UINT_ARRAY: FilterValueType.UINT,
    FilterValueType.FLOAT_ARRAY: FilterValueType.FLOAT,
}

_ROLE_SCOPE_SIGNAL = {
    DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES: (
        DiagnosticCandidateScope.STORED_QUERY,
        DiagnosticSignal.BM25,
    ),
    DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES: (
        DiagnosticCandidateScope.STORED_QUERY,
        DiagnosticSignal.ANN,
    ),
    DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES: (
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
        DiagnosticSignal.BM25,
    ),
    DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES: (
        DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL,
        DiagnosticSignal.ANN,
    ),
}
_EXPECTED_CANDIDATE_ROLES = {
    (RetrievalMode.BM25, False): (DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,),
    (RetrievalMode.BM25, True): (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
    ),
    (RetrievalMode.VECTOR, False): (DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,),
    (RetrievalMode.VECTOR, True): (
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RRF, False): (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RRF, True): (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RERANK, False): (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
    ),
    (RetrievalMode.HYBRID_RERANK, True): (
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.STORED_QUERY_ANN_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    ),
}

_STATEMENTS = {
    ForensicCode.FILTER_PREDICATE_FAILED: (
        "The selected target did not match a stored-query filter predicate."
    ),
    ForensicCode.NO_LEXICAL_SCORE: (
        "The selected target had no positive lexical score in this diagnostic."
    ),
    ForensicCode.OUTSIDE_LEXICAL_CANDIDATES: (
        "The selected target scored outside the lexical candidate boundary."
    ),
    ForensicCode.OUTSIDE_VECTOR_CANDIDATES: (
        "The selected target scored outside the vector candidate boundary."
    ),
    ForensicCode.ANN_CANDIDATE_MISS: (
        "The selected target beat the observed ANN boundary but was absent from that list."
    ),
    ForensicCode.OUTSIDE_FUSION_TOP_K: (
        "The selected target scored outside the qualified client-computed fusion boundary."
    ),
    ForensicCode.NOT_OBSERVABLE: (
        "The selected target's exclusion is not observable from this diagnostic."
    ),
}
_UNAVAILABLE_STATEMENT = "The selected target was unavailable in this diagnostic snapshot."


class _InvalidAnalysis(Exception):
    __slots__ = ("code",)

    def __init__(self, code: DiagnosticAnalysisErrorCode) -> None:
        self.code = code


class _ControlOutcome(StrEnum):
    NONE = "none"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    SYSTEM_EXIT = "system_exit"
    MEMORY_ERROR = "memory_error"


@dataclass(frozen=True, slots=True)
class _ValidatedPredicate:
    ordinal: int
    path: tuple[int, ...]
    field_name: str
    operator: PredicateOp
    value_type: FilterValueType
    value: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ValidatedLogical:
    operator: LogicalOp
    children: tuple[_ValidatedNode, ...]


type _ValidatedNode = _ValidatedPredicate | _ValidatedLogical


@dataclass(frozen=True, slots=True)
class _FilterDefinition:
    root: _ValidatedNode
    predicates: tuple[_ValidatedPredicate, ...]
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FilterResult:
    root: TruthValue
    evidence: tuple[FilterPredicateEvidence, ...]
    witness_ordinals: frozenset[int]


@dataclass(frozen=True, slots=True)
class _CandidateFacts:
    source: CandidateListInput = field(repr=False)
    scope: DiagnosticCandidateScope
    signal: DiagnosticSignal
    rows: tuple[CandidateRow, ...] = field(repr=False)
    target_row: CandidateRow | None = field(default=None, repr=False)


def _fail(code: DiagnosticAnalysisErrorCode) -> NoReturn:
    raise _InvalidAnalysis(code)


def _is_strict_number(value: object) -> TypeGuard[int | float]:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or type(value) not in {int, float}
    ):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _bounded_score(value: object) -> float:
    if not _is_strict_number(value) or value < 0 or value > _MAX_SCORE:
        _fail(DiagnosticAnalysisErrorCode.INVALID_TARGET)
    return float(value)


def _scores_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_SCORE_REL_TOL, abs_tol=_SCORE_ABS_TOL)


def _validate_binding(binding: object) -> DiagnosticBinding:
    if not isinstance(binding, DiagnosticBinding) or type(binding) is not DiagnosticBinding:
        _fail(DiagnosticAnalysisErrorCode.INVALID_BINDING)
    if (
        type(binding.config_id) is not UUID
        or type(binding.target_document_id) is not UUID
        or type(binding.trace_id) is not UUID
        or type(binding.observed_at) is not datetime
        or binding.observed_at.tzinfo is None
        or binding.observed_at.utcoffset() is None
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_BINDING)
    return binding


def _validate_mode(
    value: DiagnosticAnalysisInput,
) -> tuple[RetrievalMode, tuple[DiagnosticSubqueryRole, ...]]:
    if (
        type(value.mode) is not RetrievalMode
        or type(value.include_no_filter_counterfactual) is not bool
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_MODE)
    expected = _EXPECTED_CANDIDATE_ROLES.get((value.mode, value.include_no_filter_counterfactual))
    if expected is None:
        _fail(DiagnosticAnalysisErrorCode.INVALID_MODE)
    return value.mode, expected


def _validate_target(target: object, mode: RetrievalMode) -> TargetLookupInput:
    if (
        not isinstance(target, TargetLookupInput)
        or type(target) is not TargetLookupInput
        or type(target.available) is not bool
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_TARGET)
    required = {
        RetrievalMode.BM25: (True, False),
        RetrievalMode.VECTOR: (False, True),
        RetrievalMode.HYBRID_RRF: (True, True),
        RetrievalMode.HYBRID_RERANK: (True, True),
    }[mode]
    actual = (target.bm25_score is not None, target.vector_distance is not None)
    if not target.available:
        if actual != (False, False):
            _fail(DiagnosticAnalysisErrorCode.INVALID_TARGET)
        return target
    if actual != required:
        _fail(DiagnosticAnalysisErrorCode.INVALID_TARGET)
    if target.bm25_score is not None:
        _bounded_score(target.bm25_score)
    if target.vector_distance is not None:
        _bounded_score(target.vector_distance)
    return target


def _score(signal: DiagnosticSignal, value: float, *, direct: bool) -> ObservedScore:
    return ObservedScore(
        kind={
            DiagnosticSignal.BM25: ScoreKind.BM25,
            DiagnosticSignal.ANN: ScoreKind.VECTOR_DISTANCE,
            DiagnosticSignal.RRF: ScoreKind.RRF,
        }[signal],
        value=value,
        direction=(
            ScoreDirection.LOWER_IS_BETTER
            if signal is DiagnosticSignal.ANN
            else ScoreDirection.HIGHER_IS_BETTER
        ),
        source=(
            ScoreSource.COMPUTE_ATTRIBUTE
            if direct
            else ScoreSource.CLIENT_COMPUTED
            if signal is DiagnosticSignal.RRF
            else ScoreSource.TURBOPUFFER_DIST
        ),
    )


def _target_model(binding: DiagnosticBinding, target: TargetLookupInput) -> DiagnosticTargetLookup:
    return DiagnosticTargetLookup(
        config_id=binding.config_id,
        target_document_id=binding.target_document_id,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        available=target.available,
        unavailable_reason=(
            None
            if target.available
            else DiagnosticTargetUnavailableReason.TARGET_UNAVAILABLE_IN_DIAGNOSTIC_SNAPSHOT
        ),
        bm25_score=(
            None
            if target.bm25_score is None
            else _score(DiagnosticSignal.BM25, float(target.bm25_score), direct=True)
        ),
        vector_distance=(
            None
            if target.vector_distance is None
            else _score(DiagnosticSignal.ANN, float(target.vector_distance), direct=True)
        ),
    )


def _validate_filter_definition(value: object) -> _FilterDefinition | None:
    if value is None:
        return None
    if (
        not isinstance(value, FilterDefinitionInput)
        or type(value) not in {FilterDefinitionInput, FilterAnalysisInput}
        or type(value.schema) is not tuple
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if not 1 <= len(value.schema) <= _MAX_FILTER_PREDICATES:
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    schema: dict[str, FilterValueType] = {}
    for item in value.schema:
        if (
            type(item) is not FilterFieldSchema
            or type(item.field) is not str
            or _FIELD_PATTERN.fullmatch(item.field) is None
            or type(item.value_type) is not FilterValueType
            or type(item.filterable) is not bool
            or item.filterable is not True
            or item.field in schema
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        schema[item.field] = item.value_type

    predicates: list[_ValidatedPredicate] = []
    node_count = 0

    def walk(node: object, *, depth: int, path: tuple[int, ...]) -> _ValidatedNode:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_FILTER_NODES or depth > _MAX_FILTER_DEPTH:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        if isinstance(node, FilterPredicate) and type(node) is FilterPredicate:
            if (
                node.kind != "predicate"
                or type(node.field) is not str
                or _FIELD_PATTERN.fullmatch(node.field) is None
                or type(node.op) is not PredicateOp
            ):
                _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
            value_type = schema.get(node.field)
            if value_type is None:
                _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
            normalized_value = _validate_filter_operand(node.op, node.value, value_type)
            if len(predicates) >= _MAX_FILTER_PREDICATES:
                _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
            predicate = _ValidatedPredicate(
                ordinal=len(predicates),
                path=path,
                field_name=node.field,
                operator=node.op,
                value_type=value_type,
                value=normalized_value,
            )
            predicates.append(predicate)
            return predicate
        if not isinstance(node, FilterLogical) or type(node) is not FilterLogical:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        if (
            node.kind != "logical"
            or type(node.op) is not LogicalOp
            or type(node.children) is not list
            or not node.children
            or (node.op is LogicalOp.NOT and len(node.children) != 1)
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        children = tuple(
            walk(child, depth=depth + 1, path=(*path, index))
            for index, child in enumerate(node.children)
        )
        return _ValidatedLogical(operator=node.op, children=children)

    root = walk(value.node, depth=1, path=(0,))
    if not predicates or set(schema) != {predicate.field_name for predicate in predicates}:
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    return _FilterDefinition(
        root=root,
        predicates=tuple(predicates),
        fields=tuple(schema),
    )


def _validate_filter_operand(
    operator: PredicateOp,
    value: object,
    value_type: FilterValueType,
) -> object:
    is_array = value_type in _ARRAY_TYPES
    if operator is PredicateOp.IN:
        if (
            is_array
            or not isinstance(value, list)
            or type(value) is not list
            or len(value) > _MAX_FILTER_COLLECTION_VALUES
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return tuple(_normalize_scalar(item, value_type, allow_null=False) for item in value)
    if operator is PredicateOp.CONTAINS_ANY:
        if (
            not is_array
            or not isinstance(value, list)
            or type(value) is not list
            or len(value) > _MAX_FILTER_COLLECTION_VALUES
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        scalar_type = _SCALAR_FOR_ARRAY[value_type]
        return tuple(_normalize_scalar(item, scalar_type, allow_null=False) for item in value)
    if operator in {PredicateOp.EQ, PredicateOp.NOT_EQ}:
        if is_array:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return _normalize_scalar(value, value_type, allow_null=True)
    if operator in {PredicateOp.LT, PredicateOp.LTE, PredicateOp.GT, PredicateOp.GTE}:
        if value_type not in _ORDERABLE_TYPES:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return _normalize_scalar(value, value_type, allow_null=False)
    _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)


def _normalize_scalar(value: object, value_type: FilterValueType, *, allow_null: bool) -> object:
    if value is None:
        if allow_null:
            return None
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.STRING:
        if type(value) is str:
            return value
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.DATETIME:
        if type(value) is not str:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return _datetime_milliseconds(value)
    if value_type is FilterValueType.UUID:
        if type(value) is not str or len(value) > 128:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        try:
            return UUID(value)
        except (AttributeError, ValueError):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.BOOL:
        if type(value) is bool:
            return value
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.INT:
        if type(value) is int:
            return value
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.UINT:
        if type(value) is int and value >= 0:
            return value
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value_type is FilterValueType.FLOAT:
        if _is_strict_number(value):
            return float(value)
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)


def _datetime_milliseconds(value: str) -> int:
    match = _DATETIME_PATTERN.fullmatch(value)
    if match is None:
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    parts = match.groupdict()
    fraction = parts["fraction"]
    microsecond = 0 if fraction is None else int(fraction[1:7].ljust(6, "0"))
    zone = parts["zone"]
    if zone in {None, "Z"}:
        tz = UTC
    else:
        sign = 1 if zone[0] == "+" else -1
        hours, minutes = (int(part) for part in zone[1:].split(":"))
        if hours > 23 or minutes > 59:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        parsed = datetime(
            int(parts["year"]),
            int(parts["month"]),
            int(parts["day"]),
            int(parts["hour"] or 0),
            int(parts["minute"] or 0),
            int(parts["second"] or 0),
            microsecond,
            tzinfo=tz,
        ).astimezone(UTC)
    except (OverflowError, ValueError):
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    delta = parsed - _EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _validate_observed_attributes(
    value: FilterAnalysisInput,
    definition: _FilterDefinition,
    *,
    target_available: bool,
) -> dict[str, PreservedAttribute]:
    if type(value.attributes) is not tuple:
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if not target_available:
        if value.attributes:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return {}
    if len(value.attributes) != len(definition.fields):
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    observed: dict[str, PreservedAttribute] = {}
    schema = {item.field: item.value_type for item in value.schema}
    for item in value.attributes:
        if (
            type(item) is not ObservedFilterAttribute
            or type(item.field) is not str
            or item.field in observed
            or item.field not in schema
            or type(item.attribute) is not PreservedAttribute
            or type(item.attribute.presence) is not AttributePresence
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        attribute = item.attribute
        if attribute.presence in {AttributePresence.MISSING, AttributePresence.PRESENT_NULL}:
            if attribute.value is not None:
                _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        elif attribute.presence is AttributePresence.PRESENT_VALUE:
            attribute = PreservedAttribute(
                presence=attribute.presence,
                value=_validate_observed_value(attribute.value, schema[item.field]),
            )
        else:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        observed[item.field] = attribute
    if set(observed) != set(definition.fields):
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    return observed


def _validate_observed_value(value: object, value_type: FilterValueType) -> object:
    if value_type in _ARRAY_TYPES:
        if type(value) is not tuple or len(value) > _MAX_FILTER_COLLECTION_VALUES:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        scalar_type = _SCALAR_FOR_ARRAY[value_type]
        return tuple(_normalize_scalar(item, scalar_type, allow_null=False) for item in value)
    return _normalize_scalar(value, value_type, allow_null=False)


def _evaluate_filter(
    value: FilterAnalysisInput,
    definition: _FilterDefinition,
    binding: DiagnosticBinding,
) -> _FilterResult:
    observed = _validate_observed_attributes(value, definition, target_available=True)
    results: dict[int, TruthValue] = {}

    def evaluate(node: _ValidatedNode) -> TruthValue:
        if isinstance(node, _ValidatedPredicate):
            result = _evaluate_predicate(node, observed[node.field_name])
            results[node.ordinal] = result
            return result
        children = tuple(evaluate(child) for child in node.children)
        if node.operator is LogicalOp.NOT:
            return _kleene_not(children[0])
        if node.operator is LogicalOp.AND:
            return _kleene_and(children)
        return _kleene_or(children)

    root = evaluate(definition.root)
    evidence = tuple(
        FilterPredicateEvidence(
            config_id=binding.config_id,
            target_document_id=binding.target_document_id,
            observed_at=binding.observed_at,
            trace_id=binding.trace_id,
            predicate_ordinal=predicate.ordinal,
            predicate_path=predicate.path,
            field=predicate.field_name,
            operator=predicate.operator,
            result=_public_predicate_result(results[predicate.ordinal]),
            certainty=(
                EvidenceCertainty.INSUFFICIENT
                if results[predicate.ordinal] is TruthValue.UNKNOWN
                else EvidenceCertainty.OBSERVED
            ),
        )
        for predicate in definition.predicates
    )
    return _FilterResult(
        root=root,
        evidence=evidence,
        witness_ordinals=_truth_witness_ordinals(definition.root, results, root),
    )


def _truth_witness_ordinals(
    root: _ValidatedNode,
    predicate_results: dict[int, TruthValue],
    desired: TruthValue,
) -> frozenset[int]:
    def result_for(node: _ValidatedNode) -> TruthValue:
        if isinstance(node, _ValidatedPredicate):
            return predicate_results[node.ordinal]
        children = tuple(result_for(child) for child in node.children)
        if node.operator is LogicalOp.NOT:
            return _kleene_not(children[0])
        if node.operator is LogicalOp.AND:
            return _kleene_and(children)
        return _kleene_or(children)

    witnesses: set[int] = set()

    def visit(node: _ValidatedNode, wanted: TruthValue) -> None:
        if result_for(node) is not wanted:
            return
        if isinstance(node, _ValidatedPredicate):
            if wanted in {TruthValue.FALSE, TruthValue.UNKNOWN}:
                witnesses.add(node.ordinal)
            return
        if node.operator is LogicalOp.NOT:
            visit(node.children[0], _kleene_not(wanted))
            return
        child_results = tuple((child, result_for(child)) for child in node.children)
        if wanted is TruthValue.UNKNOWN:
            for child, result in child_results:
                if result is TruthValue.UNKNOWN:
                    visit(child, TruthValue.UNKNOWN)
            return
        if node.operator is LogicalOp.AND:
            selected = TruthValue.FALSE if wanted is TruthValue.FALSE else TruthValue.TRUE
        else:
            selected = TruthValue.FALSE if wanted is TruthValue.FALSE else TruthValue.TRUE
        for child, result in child_results:
            if result is selected:
                visit(child, selected)

    visit(root, desired)
    return frozenset(witnesses)


def _evaluate_predicate(
    predicate: _ValidatedPredicate,
    attribute: PreservedAttribute,
) -> TruthValue:
    operator = predicate.operator
    right = predicate.value
    if attribute.presence is AttributePresence.MISSING:
        if operator is PredicateOp.EQ:
            return TruthValue.TRUE if right is None else TruthValue.FALSE
        if operator is PredicateOp.NOT_EQ:
            return TruthValue.FALSE if right is None else TruthValue.TRUE
        if operator in {PredicateOp.LT, PredicateOp.LTE, PredicateOp.GT, PredicateOp.GTE}:
            return TruthValue.UNKNOWN
        return TruthValue.FALSE
    if attribute.presence is AttributePresence.PRESENT_NULL:
        if operator is PredicateOp.EQ:
            return TruthValue.UNKNOWN if right is None else TruthValue.FALSE
        if operator is PredicateOp.NOT_EQ:
            return TruthValue.TRUE
        if operator in {PredicateOp.LT, PredicateOp.LTE}:
            return TruthValue.TRUE
        if operator in {PredicateOp.GT, PredicateOp.GTE}:
            return TruthValue.UNKNOWN
        return TruthValue.FALSE

    left = attribute.value
    if operator is PredicateOp.EQ:
        return _truth(left == right)
    if operator is PredicateOp.NOT_EQ:
        return _truth(left != right)
    if operator is PredicateOp.LT:
        return _truth(left < right)  # type: ignore[operator]
    if operator is PredicateOp.LTE:
        return _truth(left <= right)  # type: ignore[operator]
    if operator is PredicateOp.GT:
        return _truth(left > right)  # type: ignore[operator]
    if operator is PredicateOp.GTE:
        return _truth(left >= right)  # type: ignore[operator]
    if operator is PredicateOp.IN:
        assert isinstance(right, tuple) and type(right) is tuple
        return _truth(left in set(right))
    assert (
        operator is PredicateOp.CONTAINS_ANY
        and isinstance(left, tuple)
        and type(left) is tuple
        and isinstance(right, tuple)
        and type(right) is tuple
    )
    return _truth(not set(left).isdisjoint(right))


def _truth(value: bool) -> TruthValue:
    return TruthValue.TRUE if value else TruthValue.FALSE


def _kleene_not(value: TruthValue) -> TruthValue:
    if value is TruthValue.TRUE:
        return TruthValue.FALSE
    if value is TruthValue.FALSE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def _kleene_and(values: tuple[TruthValue, ...]) -> TruthValue:
    if TruthValue.FALSE in values:
        return TruthValue.FALSE
    if all(value is TruthValue.TRUE for value in values):
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def _kleene_or(values: tuple[TruthValue, ...]) -> TruthValue:
    if TruthValue.TRUE in values:
        return TruthValue.TRUE
    if all(value is TruthValue.FALSE for value in values):
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def _public_predicate_result(value: TruthValue) -> DiagnosticPredicateResult:
    return {
        TruthValue.TRUE: DiagnosticPredicateResult.MATCHED,
        TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
        TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
    }[value]


def _stored_filter_result(
    scope: DiagnosticCandidateScope,
    value: TruthValue | None,
) -> DiagnosticPredicateResult | None:
    if scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL or value is None:
        return None
    return _public_predicate_result(value)


def _validate_candidate_lists(
    value: object,
    *,
    expected_roles: tuple[DiagnosticSubqueryRole, ...],
    mode: RetrievalMode,
    binding: DiagnosticBinding,
    target_available: bool,
) -> tuple[_CandidateFacts, ...]:
    if (
        not isinstance(value, tuple)
        or type(value) is not tuple
        or len(value) != len(expected_roles)
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
    expected_limit = 50 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 100
    facts: list[_CandidateFacts] = []
    for ordinal, (candidate_list, expected_role) in enumerate(
        zip(value, expected_roles, strict=True), start=1
    ):
        if (
            not isinstance(candidate_list, CandidateListInput)
            or type(candidate_list) is not CandidateListInput
            or type(candidate_list.ordinal) is not int
            or candidate_list.ordinal != ordinal
            or type(candidate_list.role) is not DiagnosticSubqueryRole
            or candidate_list.role is not expected_role
            or type(candidate_list.requested_limit) is not int
            or candidate_list.requested_limit != expected_limit
            or type(candidate_list.rows) is not tuple
            or len(candidate_list.rows) > expected_limit
        ):
            _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
        scope, signal = _ROLE_SCOPE_SIGNAL[candidate_list.role]
        seen: set[UUID] = set()
        target_row: CandidateRow | None = None
        previous: float | None = None
        for expected_rank, row in enumerate(candidate_list.rows, start=1):
            if (
                type(row) is not CandidateRow
                or type(row.document_id) is not UUID
                or type(row.rank) is not int
                or row.rank != expected_rank
                or row.document_id in seen
                or not _is_strict_number(row.score)
                or row.score < 0
                or row.score > _MAX_SCORE
                or (signal is DiagnosticSignal.BM25 and row.score == 0)
            ):
                _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
            score = float(row.score)
            if previous is not None and (
                (signal is DiagnosticSignal.BM25 and score > previous)
                or (signal is DiagnosticSignal.ANN and score < previous)
            ):
                _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
            previous = score
            seen.add(row.document_id)
            if row.document_id == binding.target_document_id:
                target_row = row
        if not target_available and target_row is not None:
            _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
        facts.append(
            _CandidateFacts(
                source=candidate_list,
                scope=scope,
                signal=signal,
                rows=candidate_list.rows,
                target_row=target_row,
            )
        )
    return tuple(facts)


def _candidate_summary(facts: _CandidateFacts) -> DiagnosticCandidateSubquerySummary:
    target_score = (
        None
        if facts.target_row is None
        else _score(facts.signal, float(facts.target_row.score), direct=False)
    )
    boundary = facts.rows[-1] if len(facts.rows) == facts.source.requested_limit else None
    return DiagnosticCandidateSubquerySummary(
        ordinal=facts.source.ordinal,
        role=facts.source.role,
        requested_limit=facts.source.requested_limit,
        returned_count=len(facts.rows),
        target_present=facts.target_row is not None,
        target_rank=None if facts.target_row is None else facts.target_row.rank,
        target_score=target_score,
        boundary_score=(
            None if boundary is None else _score(facts.signal, float(boundary.score), direct=False)
        ),
    )


def _candidate_evidence(
    facts: _CandidateFacts,
    *,
    binding: DiagnosticBinding,
    target: TargetLookupInput,
    filter_root: TruthValue | None,
) -> CandidateCutoffEvidence:
    raw_direct = (
        target.bm25_score if facts.signal is DiagnosticSignal.BM25 else target.vector_distance
    )
    if raw_direct is None:
        _fail(DiagnosticAnalysisErrorCode.INVALID_TARGET)
    direct = float(raw_direct)
    if facts.target_row is not None and not _scores_equal(float(facts.target_row.score), direct):
        _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
    stored_filter_applies = (
        facts.scope is DiagnosticCandidateScope.STORED_QUERY and filter_root is not None
    )
    if stored_filter_applies and filter_root is TruthValue.FALSE and facts.target_row is not None:
        _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
    boundary = facts.rows[-1] if len(facts.rows) == facts.source.requested_limit else None
    boundary_score = None if boundary is None else float(boundary.score)
    filter_ineligible = (
        stored_filter_applies
        and filter_root in {TruthValue.FALSE, TruthValue.UNKNOWN}
        and facts.target_row is None
    )
    eligibility_override = (
        filter_ineligible
        and boundary_score is not None
        and not _scores_equal(direct, boundary_score)
        and (
            (facts.signal is DiagnosticSignal.BM25 and direct > boundary_score)
            or (facts.signal is DiagnosticSignal.ANN and direct < boundary_score)
        )
    )
    relation = (
        DiagnosticCutoffRelation.NOT_OBSERVABLE
        if eligibility_override
        else _candidate_relation(
            signal=facts.signal,
            direct_score=direct,
            target_present=facts.target_row is not None,
            boundary_score=boundary_score,
        )
    )
    certainty = (
        EvidenceCertainty.INSUFFICIENT
        if relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
        else EvidenceCertainty.COUNTERFACTUAL
        if facts.scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
        else EvidenceCertainty.OBSERVED
    )
    return CandidateCutoffEvidence(
        config_id=binding.config_id,
        target_document_id=binding.target_document_id,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        subquery_ordinal=facts.source.ordinal,
        role=facts.source.role,
        scope=facts.scope,
        stored_filter_result=_stored_filter_result(facts.scope, filter_root),
        signal=facts.signal,
        requested_limit=facts.source.requested_limit,
        returned_count=len(facts.rows),
        target_present=facts.target_row is not None,
        target_rank=None if facts.target_row is None else facts.target_row.rank,
        target_score=(
            None
            if facts.target_row is None
            else _score(facts.signal, float(facts.target_row.score), direct=False)
        ),
        direct_score=_score(facts.signal, direct, direct=True),
        boundary_score=(
            None if boundary is None else _score(facts.signal, float(boundary.score), direct=False)
        ),
        relation=relation,
        certainty=certainty,
    )


def _candidate_relation(
    *,
    signal: DiagnosticSignal,
    direct_score: float,
    target_present: bool,
    boundary_score: float | None,
) -> DiagnosticCutoffRelation:
    if target_present:
        return DiagnosticCutoffRelation.TARGET_PRESENT
    if signal is DiagnosticSignal.BM25 and direct_score == 0:
        return DiagnosticCutoffRelation.NO_LEXICAL_SCORE
    if boundary_score is None or _scores_equal(direct_score, boundary_score):
        return DiagnosticCutoffRelation.NOT_OBSERVABLE
    if signal is DiagnosticSignal.BM25:
        if direct_score > boundary_score:
            _fail(DiagnosticAnalysisErrorCode.INVALID_CANDIDATES)
        return DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
    if direct_score < boundary_score:
        return DiagnosticCutoffRelation.ANN_CANDIDATE_MISS
    return DiagnosticCutoffRelation.OUTSIDE_CANDIDATES


def _validate_rrf_inputs(value: object, *, hybrid: bool) -> RrfInputs | None:
    if not hybrid:
        if value is not None:
            _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
        return None
    if not isinstance(value, RrfInputs) or type(value) is not RrfInputs:
        _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
    if (
        not _is_strict_number(value.bm25_weight)
        or not 0 < value.bm25_weight <= 100
        or not _is_strict_number(value.ann_weight)
        or not 0 < value.ann_weight <= 100
        or type(value.rank_constant) is not int
        or not 1 <= value.rank_constant <= 10_000
        or type(value.cutoff) is not int
        or value.cutoff != 50
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
    return value


def _qualified_rrf(
    *,
    binding: DiagnosticBinding,
    scope: DiagnosticCandidateScope,
    bm25: _CandidateFacts,
    ann: _CandidateFacts,
    inputs: RrfInputs,
    filter_root: TruthValue | None,
) -> QualifiedRrfEvidence:
    if (
        bm25.scope is not scope
        or ann.scope is not scope
        or bm25.signal is not DiagnosticSignal.BM25
        or ann.signal is not DiagnosticSignal.ANN
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
    scores: dict[UUID, list[float]] = {}
    ranks: dict[UUID, tuple[int | None, int | None]] = {}
    for row in bm25.rows:
        scores.setdefault(row.document_id, []).append(
            float(inputs.bm25_weight) / (inputs.rank_constant + row.rank)
        )
        ranks[row.document_id] = (row.rank, ranks.get(row.document_id, (None, None))[1])
    for row in ann.rows:
        scores.setdefault(row.document_id, []).append(
            float(inputs.ann_weight) / (inputs.rank_constant + row.rank)
        )
        ranks[row.document_id] = (ranks.get(row.document_id, (None, None))[0], row.rank)
    totals = {document_id: math.fsum(parts) for document_id, parts in scores.items()}
    if any(not math.isfinite(score) or score <= 0 for score in totals.values()):
        _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
    ordered_scores = sorted(totals.values(), reverse=True)
    returned_count = min(len(ordered_scores), inputs.cutoff)
    boundary_value = (
        ordered_scores[inputs.cutoff - 1] if len(ordered_scores) >= inputs.cutoff else None
    )
    target_ranks = ranks.get(binding.target_document_id, (None, None))
    if (
        scope is DiagnosticCandidateScope.STORED_QUERY
        and filter_root is TruthValue.FALSE
        and target_ranks != (None, None)
    ):
        _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
    target_score_value = totals.get(binding.target_document_id, 0.0)
    target_in_union = binding.target_document_id in totals
    target_present = False
    target_rank: int | None = None
    if target_in_union:
        strictly_higher = sum(
            score > target_score_value and not _scores_equal(score, target_score_value)
            for score in totals.values()
        )
        tied = sum(_scores_equal(score, target_score_value) for score in totals.values())
        membership_certain = len(totals) <= inputs.cutoff or strictly_higher + tied <= inputs.cutoff
        clearly_outside = strictly_higher >= inputs.cutoff
        if membership_certain:
            target_present = True
            target_rank = strictly_higher + 1
        elif not clearly_outside:
            # The tolerance-equal score group straddles the cutoff. No UUID tie-break is invented.
            target_present = False
        else:
            target_present = False
    relation = (
        DiagnosticCutoffRelation.TARGET_PRESENT
        if target_present
        else DiagnosticCutoffRelation.NOT_OBSERVABLE
        if boundary_value is None or _scores_equal(target_score_value, boundary_value)
        else DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
    )
    certainty = (
        EvidenceCertainty.INSUFFICIENT
        if relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
        else EvidenceCertainty.COUNTERFACTUAL
        if scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
        else EvidenceCertainty.OBSERVED
    )
    return QualifiedRrfEvidence(
        config_id=binding.config_id,
        target_document_id=binding.target_document_id,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        scope=scope,
        stored_filter_result=_stored_filter_result(scope, filter_root),
        bm25_rank=target_ranks[0],
        ann_rank=target_ranks[1],
        bm25_weight=float(inputs.bm25_weight),
        ann_weight=float(inputs.ann_weight),
        rank_constant=inputs.rank_constant,
        returned_count=returned_count,
        target_present=target_present,
        target_rank=target_rank,
        target_score=_score(DiagnosticSignal.RRF, target_score_value, direct=False),
        boundary_score=(
            None
            if boundary_value is None
            else _score(DiagnosticSignal.RRF, boundary_value, direct=False)
        ),
        relation=relation,
        certainty=certainty,
    )


def _filter_observation(
    evidence: FilterPredicateEvidence,
    binding: DiagnosticBinding,
    root_result: TruthValue,
    witness_ordinals: frozenset[int],
) -> ForensicObservation | None:
    if evidence.predicate_ordinal not in witness_ordinals:
        return None
    expected = {
        TruthValue.TRUE: None,
        TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
        TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
    }[root_result]
    if expected is None or evidence.result is not expected:
        return None
    code = (
        ForensicCode.FILTER_PREDICATE_FAILED
        if evidence.result is DiagnosticPredicateResult.NOT_MATCHED
        else ForensicCode.NOT_OBSERVABLE
    )
    return ForensicObservation(
        config_id=binding.config_id,
        document_id=binding.target_document_id,
        code=code,
        statement=_STATEMENTS[code],
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        evidence=[
            EvidenceItem(
                label=f"filter_predicate_{evidence.predicate_ordinal}",
                value=FilterPredicateEvidenceValue(
                    predicate_ordinal=evidence.predicate_ordinal,
                    predicate_path=evidence.predicate_path,
                    field=evidence.field,
                    operator=evidence.operator,
                    result=evidence.result,
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=binding.observed_at,
                trace_id=binding.trace_id,
            )
        ],
        certainty=(
            EvidenceCertainty.INSUFFICIENT
            if code is ForensicCode.NOT_OBSERVABLE
            else EvidenceCertainty.OBSERVED
        ),
    )


def _candidate_observation(
    evidence: CandidateCutoffEvidence,
    binding: DiagnosticBinding,
    filter_root: TruthValue | None,
) -> ForensicObservation | None:
    if evidence.relation is DiagnosticCutoffRelation.TARGET_PRESENT:
        return None
    if (
        evidence.scope is DiagnosticCandidateScope.STORED_QUERY
        and filter_root is TruthValue.FALSE
        and evidence.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    ):
        return None
    code = {
        DiagnosticCutoffRelation.NO_LEXICAL_SCORE: ForensicCode.NO_LEXICAL_SCORE,
        DiagnosticCutoffRelation.OUTSIDE_CANDIDATES: (
            ForensicCode.OUTSIDE_LEXICAL_CANDIDATES
            if evidence.signal is DiagnosticSignal.BM25
            else ForensicCode.OUTSIDE_VECTOR_CANDIDATES
        ),
        DiagnosticCutoffRelation.ANN_CANDIDATE_MISS: ForensicCode.ANN_CANDIDATE_MISS,
        DiagnosticCutoffRelation.NOT_OBSERVABLE: ForensicCode.NOT_OBSERVABLE,
    }[evidence.relation]
    return ForensicObservation(
        config_id=binding.config_id,
        document_id=binding.target_document_id,
        code=code,
        statement=_STATEMENTS[code],
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        evidence=[
            EvidenceItem(
                label=f"cutoff_{evidence.scope.value}_{evidence.signal.value}",
                value=CutoffRelationEvidenceValue(
                    scope=evidence.scope,
                    signal=evidence.signal,
                    relation=evidence.relation,
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=binding.observed_at,
                trace_id=binding.trace_id,
            ),
            EvidenceItem(
                label=f"direct_{evidence.signal.value}_score",
                value=DirectScoreEvidenceValue(
                    signal=evidence.signal,
                    score=evidence.direct_score,
                ),
                origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
                observed_at=binding.observed_at,
                trace_id=binding.trace_id,
            ),
        ],
        certainty=evidence.certainty,
    )


def _rrf_observation(
    evidence: QualifiedRrfEvidence,
    binding: DiagnosticBinding,
    filter_root: TruthValue | None,
) -> ForensicObservation | None:
    if evidence.relation is DiagnosticCutoffRelation.TARGET_PRESENT:
        return None
    if (
        evidence.scope is DiagnosticCandidateScope.STORED_QUERY
        and filter_root is TruthValue.FALSE
        and evidence.relation is DiagnosticCutoffRelation.NOT_OBSERVABLE
    ):
        return None
    code = (
        ForensicCode.OUTSIDE_FUSION_TOP_K
        if evidence.relation is DiagnosticCutoffRelation.OUTSIDE_CANDIDATES
        else ForensicCode.NOT_OBSERVABLE
    )
    items: list[EvidenceItem] = [
        EvidenceItem(
            label=f"cutoff_{evidence.scope.value}_rrf",
            value=CutoffRelationEvidenceValue(
                scope=evidence.scope,
                signal=DiagnosticSignal.RRF,
                relation=evidence.relation,
            ),
            origin=EvidenceOrigin.CLIENT_COMPUTED,
            observed_at=binding.observed_at,
            trace_id=binding.trace_id,
        ),
        EvidenceItem(
            label=f"qualified_rrf_score_{evidence.scope.value}",
            value=ScoreEvidenceValue(stage=RetrievalStage.RRF, score=evidence.target_score),
            origin=EvidenceOrigin.CLIENT_COMPUTED,
            observed_at=binding.observed_at,
            trace_id=binding.trace_id,
        ),
    ]
    for stage, rank, weight in (
        (RetrievalStage.BM25_CANDIDATES, evidence.bm25_rank, evidence.bm25_weight),
        (RetrievalStage.VECTOR_CANDIDATES, evidence.ann_rank, evidence.ann_weight),
    ):
        if rank is None:
            continue
        items.append(
            EvidenceItem(
                label=f"{evidence.scope.value}_{stage.value}_rrf_contribution",
                value=RrfContributionEvidenceValue(
                    stage=stage,
                    rank=rank,
                    weight=weight,
                    rank_constant=evidence.rank_constant,
                    contribution=weight / (evidence.rank_constant + rank),
                ),
                origin=EvidenceOrigin.CLIENT_COMPUTED,
                observed_at=binding.observed_at,
                trace_id=binding.trace_id,
            )
        )
    return ForensicObservation(
        config_id=binding.config_id,
        document_id=binding.target_document_id,
        code=code,
        statement=_STATEMENTS[code],
        origin=EvidenceOrigin.CLIENT_COMPUTED,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        evidence=items,
        certainty=evidence.certainty,
    )


def _unavailable_observation(binding: DiagnosticBinding) -> ForensicObservation:
    return ForensicObservation(
        config_id=binding.config_id,
        document_id=binding.target_document_id,
        code=ForensicCode.NOT_OBSERVABLE,
        statement=_UNAVAILABLE_STATEMENT,
        origin=EvidenceOrigin.LIVE_EXPECTED_DOCUMENT_DIAGNOSTIC,
        observed_at=binding.observed_at,
        trace_id=binding.trace_id,
        evidence=[],
        certainty=EvidenceCertainty.INSUFFICIENT,
    )


def _analyze(value: object) -> DiagnosticAnalysisResult:
    if not isinstance(value, DiagnosticAnalysisInput) or type(value) is not DiagnosticAnalysisInput:
        _fail(DiagnosticAnalysisErrorCode.INVALID_BINDING)
    binding = _validate_binding(value.binding)
    mode, expected_roles = _validate_mode(value)
    target_input = _validate_target(value.target, mode)
    filter_definition = _validate_filter_definition(value.stored_filter)
    if value.include_no_filter_counterfactual and filter_definition is None:
        _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
    if value.stored_filter is not None and filter_definition is not None:
        _validate_observed_attributes(
            value.stored_filter,
            filter_definition,
            target_available=target_input.available,
        )
    candidates = _validate_candidate_lists(
        value.candidate_lists,
        expected_roles=expected_roles,
        mode=mode,
        binding=binding,
        target_available=target_input.available,
    )
    hybrid = mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
    rrf_inputs = _validate_rrf_inputs(value.rrf, hybrid=hybrid)

    target = _target_model(binding, target_input)
    summaries = tuple(_candidate_summary(candidate) for candidate in candidates)
    lookup_summary = DiagnosticTargetLookupSubquerySummary(
        returned_count=1 if target_input.available else 0,
        target_present=target_input.available,
    )
    subqueries: tuple[
        DiagnosticTargetLookupSubquerySummary | DiagnosticCandidateSubquerySummary,
        ...,
    ] = (
        lookup_summary,
        *summaries,
    )
    if not target_input.available:
        return DiagnosticAnalysisResult(
            filter_root_result=None,
            target=target,
            subqueries=subqueries,
            filter_evidence=(),
            candidate_evidence=(),
            qualified_rrf_evidence=(),
            observations=(_unavailable_observation(binding),),
        )

    filter_result = (
        None
        if value.stored_filter is None or filter_definition is None
        else _evaluate_filter(value.stored_filter, filter_definition, binding)
    )
    candidate_evidence = tuple(
        _candidate_evidence(
            candidate,
            binding=binding,
            target=target_input,
            filter_root=None if filter_result is None else filter_result.root,
        )
        for candidate in candidates
    )
    qualified: list[QualifiedRrfEvidence] = []
    if rrf_inputs is not None:
        # In HYBRID_RERANK this is only would-be top-50 admission from raw inputs. The diagnostic
        # never executes or makes a finding about reranker/final order.
        by_scope_signal = {
            (candidate.scope, candidate.signal): candidate for candidate in candidates
        }
        scopes = [DiagnosticCandidateScope.STORED_QUERY]
        if value.include_no_filter_counterfactual:
            scopes.append(DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL)
        for scope in scopes:
            bm25 = by_scope_signal.get((scope, DiagnosticSignal.BM25))
            ann = by_scope_signal.get((scope, DiagnosticSignal.ANN))
            if bm25 is None or ann is None:
                _fail(DiagnosticAnalysisErrorCode.INVALID_RRF)
            qualified.append(
                _qualified_rrf(
                    binding=binding,
                    scope=scope,
                    bm25=bm25,
                    ann=ann,
                    inputs=rrf_inputs,
                    filter_root=(
                        None
                        if scope is DiagnosticCandidateScope.NO_FILTER_COUNTERFACTUAL
                        or filter_result is None
                        else filter_result.root
                    ),
                )
            )
    filter_evidence = () if filter_result is None else filter_result.evidence
    filter_observations = (
        ()
        if filter_result is None
        else tuple(
            _filter_observation(
                item,
                binding,
                filter_result.root,
                filter_result.witness_ordinals,
            )
            for item in filter_evidence
        )
    )
    observations = tuple(
        observation
        for observation in (
            *filter_observations,
            *(
                _candidate_observation(
                    item,
                    binding,
                    None if filter_result is None else filter_result.root,
                )
                for item in candidate_evidence
            ),
            *(
                _rrf_observation(
                    item,
                    binding,
                    None if filter_result is None else filter_result.root,
                )
                for item in qualified
            ),
        )
        if observation is not None
    )
    return DiagnosticAnalysisResult(
        filter_root_result=None if filter_result is None else filter_result.root,
        target=target,
        subqueries=subqueries,
        filter_evidence=filter_evidence,
        candidate_evidence=candidate_evidence,
        qualified_rrf_evidence=tuple(qualified),
        observations=observations,
    )


def _analyze_outcome(
    value: object,
) -> tuple[
    DiagnosticAnalysisResult | None,
    DiagnosticAnalysisErrorCode | None,
    _ControlOutcome,
]:
    try:
        return _analyze(value), None, _ControlOutcome.NONE
    except _InvalidAnalysis as error:
        code = error.code
        _detach_exception(error)
        del error
        return None, code, _ControlOutcome.NONE
    except KeyboardInterrupt as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.SYSTEM_EXIT
    except MemoryError as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.MEMORY_ERROR
    except Exception as error:
        _detach_exception(error)
        del error
        return None, DiagnosticAnalysisErrorCode.INVALID_OUTPUT, _ControlOutcome.NONE


def _preflight_outcome(
    value: object,
) -> tuple[tuple[str, ...] | None, DiagnosticAnalysisErrorCode | None, _ControlOutcome]:
    try:
        if type(value) is not FilterDefinitionInput:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        definition = _validate_filter_definition(value)
        if definition is None:
            _fail(DiagnosticAnalysisErrorCode.INVALID_FILTER)
        return definition.fields, None, _ControlOutcome.NONE
    except _InvalidAnalysis as error:
        code = error.code
        _detach_exception(error)
        del error
        return None, code, _ControlOutcome.NONE
    except KeyboardInterrupt as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.KEYBOARD_INTERRUPT
    except SystemExit as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.SYSTEM_EXIT
    except MemoryError as error:
        _detach_exception(error)
        del error
        return None, None, _ControlOutcome.MEMORY_ERROR
    except Exception as error:
        _detach_exception(error)
        del error
        return None, DiagnosticAnalysisErrorCode.INVALID_OUTPUT, _ControlOutcome.NONE


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None


def _raise_control(control: _ControlOutcome) -> NoReturn:
    if control is _ControlOutcome.KEYBOARD_INTERRUPT:
        raise KeyboardInterrupt() from None
    if control is _ControlOutcome.SYSTEM_EXIT:
        raise SystemExit(1) from None
    if control is _ControlOutcome.MEMORY_ERROR:
        raise MemoryError() from None
    raise AssertionError("unreachable control outcome")


def preflight_filter_definition(value: FilterDefinitionInput) -> tuple[str, ...]:
    """Validate one bounded stored-filter definition before any sensitive factory exists.

    The returned field names are the exact schema-bound lookup projection. Observed target
    attributes are deliberately not accepted by this definition-only entry point.
    """

    fields, error_code, control = _preflight_outcome(value)
    del value
    if control is not _ControlOutcome.NONE:
        _raise_control(control)
    if error_code is not None:
        raise DiagnosticAnalysisError(error_code) from None
    assert fields is not None
    return fields


def analyze_diagnostic(value: DiagnosticAnalysisInput) -> DiagnosticAnalysisResult:
    """Analyze one exact diagnostic snapshot without I/O or provider behavior.

    The caller supplies already-authenticated, provider-neutral inputs. Full candidate rows are
    consumed only for validation and qualified client computation; the result contains no
    unrelated document IDs or filter/attribute values. Invalid inputs unwind before the fixed,
    value-free error is raised.
    """

    result, error_code, control = _analyze_outcome(value)
    del value
    if control is not _ControlOutcome.NONE:
        _raise_control(control)
    if error_code is not None:
        raise DiagnosticAnalysisError(error_code) from None
    assert result is not None
    return result
