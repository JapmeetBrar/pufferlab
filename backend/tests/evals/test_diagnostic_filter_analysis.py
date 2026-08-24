from __future__ import annotations

import traceback
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pufferlab.contracts.filters import FilterLogical, FilterPredicate, LogicalOp, PredicateOp
from pufferlab.contracts.forensics import (
    DiagnosticPredicateResult,
    DiagnosticSubqueryRole,
    FilterPredicateEvidenceValue,
    ForensicCode,
)
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.evals.diagnostic_analysis import analyze_diagnostic, preflight_filter_definition
from pufferlab.evals.diagnostic_models import (
    AttributePresence,
    CandidateListInput,
    CandidateRow,
    DiagnosticAnalysisError,
    DiagnosticAnalysisErrorCode,
    DiagnosticAnalysisInput,
    DiagnosticBinding,
    FilterAnalysisInput,
    FilterDefinitionInput,
    FilterFieldSchema,
    FilterValueType,
    ObservedFilterAttribute,
    PreservedAttribute,
    TargetLookupInput,
    TruthValue,
)

_CONFIG = UUID(int=101)
_TARGET = UUID(int=102)
_TRACE = UUID(int=103)
_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)


def _analyze_filter(
    node: object,
    *,
    schemas: tuple[FilterFieldSchema, ...],
    attributes: tuple[ObservedFilterAttribute, ...],
):
    def input_for(*, target_in_candidates: bool) -> DiagnosticAnalysisInput:
        return DiagnosticAnalysisInput(
            binding=DiagnosticBinding(
                config_id=_CONFIG,
                target_document_id=_TARGET,
                observed_at=_NOW,
                trace_id=_TRACE,
            ),
            mode=RetrievalMode.BM25,
            include_no_filter_counterfactual=False,
            target=TargetLookupInput(available=True, bm25_score=3.0),
            candidate_lists=(
                CandidateListInput(
                    ordinal=1,
                    role=DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
                    requested_limit=50,
                    rows=(
                        (CandidateRow(document_id=_TARGET, rank=1, score=3.0),)
                        if target_in_candidates
                        else ()
                    ),
                ),
            ),
            stored_filter=FilterAnalysisInput(
                node=node,
                schema=schemas,
                attributes=attributes,
            ),
        )

    try:
        return analyze_diagnostic(input_for(target_in_candidates=True))
    except DiagnosticAnalysisError as error:
        # A false stored filter and a present stored candidate are structurally contradictory.
        # Filter-only tests rerun that exact case with the target absent so candidate facts do not
        # obscure the independently asserted filter truth table.
        if error.code is not DiagnosticAnalysisErrorCode.INVALID_CANDIDATES:
            raise
        return analyze_diagnostic(input_for(target_in_candidates=False))


def _one(
    *,
    operator: PredicateOp,
    right: object,
    value_type: FilterValueType,
    presence: AttributePresence,
    observed: object = None,
):
    field = "attribute"
    predicate = FilterPredicate.model_construct(
        kind="predicate",
        field=field,
        op=operator,
        value=right,
    )
    return _analyze_filter(
        predicate,
        schemas=(FilterFieldSchema(field=field, value_type=value_type, filterable=True),),
        attributes=(
            ObservedFilterAttribute(
                field=field,
                attribute=PreservedAttribute(presence=presence, value=observed),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("operator", "right", "value_type", "expected"),
    [
        (PredicateOp.EQ, None, FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.NOT_EQ, None, FilterValueType.STRING, TruthValue.FALSE),
        (PredicateOp.EQ, "x", FilterValueType.STRING, TruthValue.FALSE),
        (PredicateOp.NOT_EQ, "x", FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.LT, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.LTE, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.GT, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.GTE, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.IN, ["x"], FilterValueType.STRING, TruthValue.FALSE),
        (
            PredicateOp.CONTAINS_ANY,
            ["x"],
            FilterValueType.STRING_ARRAY,
            TruthValue.FALSE,
        ),
    ],
)
def test_frozen_missing_truth_table(
    operator: PredicateOp,
    right: object,
    value_type: FilterValueType,
    expected: TruthValue,
) -> None:
    result = _one(
        operator=operator,
        right=right,
        value_type=value_type,
        presence=AttributePresence.MISSING,
    )

    assert result.filter_root_result is expected
    assert (
        result.filter_evidence[0].result
        is {
            TruthValue.TRUE: DiagnosticPredicateResult.MATCHED,
            TruthValue.FALSE: DiagnosticPredicateResult.NOT_MATCHED,
            TruthValue.UNKNOWN: DiagnosticPredicateResult.NOT_OBSERVABLE,
        }[expected]
    )


@pytest.mark.parametrize(
    ("operator", "right", "value_type", "expected"),
    [
        (PredicateOp.EQ, None, FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.NOT_EQ, None, FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.EQ, "x", FilterValueType.STRING, TruthValue.FALSE),
        (PredicateOp.NOT_EQ, "x", FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.LT, "x", FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.LTE, "x", FilterValueType.STRING, TruthValue.TRUE),
        (PredicateOp.GT, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.GTE, "x", FilterValueType.STRING, TruthValue.UNKNOWN),
        (PredicateOp.IN, ["x"], FilterValueType.STRING, TruthValue.FALSE),
        (
            PredicateOp.CONTAINS_ANY,
            ["x"],
            FilterValueType.STRING_ARRAY,
            TruthValue.FALSE,
        ),
    ],
)
def test_frozen_present_null_truth_table(
    operator: PredicateOp,
    right: object,
    value_type: FilterValueType,
    expected: TruthValue,
) -> None:
    result = _one(
        operator=operator,
        right=right,
        value_type=value_type,
        presence=AttributePresence.PRESENT_NULL,
    )

    assert result.filter_root_result is expected


@pytest.mark.parametrize(
    ("operator", "right", "value_type", "matching", "nonmatching"),
    [
        (PredicateOp.EQ, "b", FilterValueType.STRING, "b", "a"),
        (PredicateOp.NOT_EQ, "b", FilterValueType.STRING, "a", "b"),
        (PredicateOp.LT, "b", FilterValueType.STRING, "a", "c"),
        (PredicateOp.LTE, "b", FilterValueType.STRING, "b", "c"),
        (PredicateOp.GT, "b", FilterValueType.STRING, "c", "a"),
        (PredicateOp.GTE, "b", FilterValueType.STRING, "b", "a"),
        (PredicateOp.IN, ["a", "b"], FilterValueType.STRING, "a", "c"),
        (
            PredicateOp.CONTAINS_ANY,
            ["a", "b"],
            FilterValueType.STRING_ARRAY,
            ("b", "c"),
            ("c", "d"),
        ),
    ],
)
def test_present_values_use_exact_schema_typed_semantics(
    operator: PredicateOp,
    right: object,
    value_type: FilterValueType,
    matching: object,
    nonmatching: object,
) -> None:
    assert (
        _one(
            operator=operator,
            right=right,
            value_type=value_type,
            presence=AttributePresence.PRESENT_VALUE,
            observed=matching,
        ).filter_root_result
        is TruthValue.TRUE
    )
    assert (
        _one(
            operator=operator,
            right=right,
            value_type=value_type,
            presence=AttributePresence.PRESENT_VALUE,
            observed=nonmatching,
        ).filter_root_result
        is TruthValue.FALSE
    )


@pytest.mark.parametrize(
    ("operator", "value_type", "observed"),
    [
        (PredicateOp.IN, FilterValueType.STRING, "value"),
        (PredicateOp.CONTAINS_ANY, FilterValueType.STRING_ARRAY, ("value",)),
    ],
)
def test_empty_collection_operands_are_valid_and_match_nothing(
    operator: PredicateOp,
    value_type: FilterValueType,
    observed: object,
) -> None:
    result = _one(
        operator=operator,
        right=[],
        value_type=value_type,
        presence=AttributePresence.PRESENT_VALUE,
        observed=observed,
    )
    assert result.filter_root_result is TruthValue.FALSE


@pytest.mark.parametrize(
    ("operator", "value_type", "observed"),
    [
        (PredicateOp.IN, FilterValueType.INT, 9_999),
        (PredicateOp.CONTAINS_ANY, FilterValueType.INT_ARRAY, (9_999,)),
    ],
)
def test_collection_operand_bound_accepts_10000_and_rejects_10001(
    operator: PredicateOp,
    value_type: FilterValueType,
    observed: object,
) -> None:
    accepted = _one(
        operator=operator,
        right=list(range(10_000)),
        value_type=value_type,
        presence=AttributePresence.PRESENT_VALUE,
        observed=observed,
    )
    assert accepted.filter_root_result is TruthValue.TRUE

    with pytest.raises(DiagnosticAnalysisError) as rejected:
        _one(
            operator=operator,
            right=list(range(10_001)),
            value_type=value_type,
            presence=AttributePresence.PRESENT_VALUE,
            observed=observed,
        )
    assert rejected.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (FilterValueType.STRING, "value"),
        (FilterValueType.DATETIME, "2026-08-23T12:00:00Z"),
        (FilterValueType.UUID, "00000000-0000-0000-0000-000000000001"),
        (FilterValueType.BOOL, True),
        (FilterValueType.INT, -1),
        (FilterValueType.UINT, 1),
        (FilterValueType.FLOAT, 1.5),
    ],
)
def test_every_scalar_schema_type_has_exact_present_value_equality(
    value_type: FilterValueType,
    value: object,
) -> None:
    result = _one(
        operator=PredicateOp.EQ,
        right=value,
        value_type=value_type,
        presence=AttributePresence.PRESENT_VALUE,
        observed=value,
    )
    assert result.filter_root_result is TruthValue.TRUE


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (FilterValueType.STRING_ARRAY, "value"),
        (FilterValueType.DATETIME_ARRAY, "2026-08-23T12:00:00Z"),
        (FilterValueType.UUID_ARRAY, "00000000-0000-0000-0000-000000000001"),
        (FilterValueType.BOOL_ARRAY, True),
        (FilterValueType.INT_ARRAY, -1),
        (FilterValueType.UINT_ARRAY, 1),
        (FilterValueType.FLOAT_ARRAY, 1.5),
    ],
)
def test_every_array_schema_type_has_exact_contains_any_semantics(
    value_type: FilterValueType,
    value: object,
) -> None:
    result = _one(
        operator=PredicateOp.CONTAINS_ANY,
        right=[value],
        value_type=value_type,
        presence=AttributePresence.PRESENT_VALUE,
        observed=(value,),
    )
    assert result.filter_root_result is TruthValue.TRUE


def _truth_leaf(name: str, truth: TruthValue) -> FilterPredicate:
    if truth is TruthValue.TRUE:
        operator, value = PredicateOp.EQ, None
    elif truth is TruthValue.FALSE:
        operator, value = PredicateOp.EQ, "value"
    else:
        operator, value = PredicateOp.LT, "value"
    return FilterPredicate(field=name, op=operator, value=value)


def _multi_filter(node: object, fields: tuple[str, ...]):
    return _analyze_filter(
        node,
        schemas=tuple(
            FilterFieldSchema(field=name, value_type=FilterValueType.STRING, filterable=True)
            for name in fields
        ),
        attributes=tuple(
            ObservedFilterAttribute(
                field=name,
                attribute=PreservedAttribute(AttributePresence.MISSING),
            )
            for name in fields
        ),
    )


@pytest.mark.parametrize("truth", tuple(TruthValue))
def test_strong_kleene_not(truth: TruthValue) -> None:
    result = _multi_filter(
        FilterLogical(op=LogicalOp.NOT, children=[_truth_leaf("a", truth)]),
        ("a",),
    )
    assert (
        result.filter_root_result
        is {
            TruthValue.TRUE: TruthValue.FALSE,
            TruthValue.FALSE: TruthValue.TRUE,
            TruthValue.UNKNOWN: TruthValue.UNKNOWN,
        }[truth]
    )


@pytest.mark.parametrize("left", tuple(TruthValue))
@pytest.mark.parametrize("right", tuple(TruthValue))
def test_strong_kleene_and_or_exhaustive(left: TruthValue, right: TruthValue) -> None:
    children = [_truth_leaf("a", left), _truth_leaf("b", right)]
    and_result = _multi_filter(FilterLogical(op=LogicalOp.AND, children=children), ("a", "b"))
    or_result = _multi_filter(FilterLogical(op=LogicalOp.OR, children=children), ("a", "b"))
    expected_and = (
        TruthValue.FALSE
        if TruthValue.FALSE in {left, right}
        else TruthValue.TRUE
        if left is right is TruthValue.TRUE
        else TruthValue.UNKNOWN
    )
    expected_or = (
        TruthValue.TRUE
        if TruthValue.TRUE in {left, right}
        else TruthValue.FALSE
        if left is right is TruthValue.FALSE
        else TruthValue.UNKNOWN
    )
    assert and_result.filter_root_result is expected_and
    assert or_result.filter_root_result is expected_or


def test_logical_not_false_root_keeps_true_leaf_and_emits_no_filter_failed_claim() -> None:
    result = _multi_filter(
        FilterLogical(op=LogicalOp.NOT, children=[_truth_leaf("a", TruthValue.TRUE)]),
        ("a",),
    )

    assert result.filter_root_result is TruthValue.FALSE
    assert result.filter_evidence[0].result is DiagnosticPredicateResult.MATCHED
    assert all(
        item.code is not ForensicCode.FILTER_PREDICATE_FAILED for item in result.observations
    )


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected_root", "expected_code", "expected_ordinal"),
    [
        (LogicalOp.OR, TruthValue.TRUE, TruthValue.FALSE, TruthValue.TRUE, None, None),
        (LogicalOp.OR, TruthValue.TRUE, TruthValue.UNKNOWN, TruthValue.TRUE, None, None),
        (
            LogicalOp.OR,
            TruthValue.FALSE,
            TruthValue.UNKNOWN,
            TruthValue.UNKNOWN,
            ForensicCode.NOT_OBSERVABLE,
            1,
        ),
        (
            LogicalOp.AND,
            TruthValue.FALSE,
            TruthValue.UNKNOWN,
            TruthValue.FALSE,
            ForensicCode.FILTER_PREDICATE_FAILED,
            0,
        ),
        (
            LogicalOp.AND,
            TruthValue.TRUE,
            TruthValue.UNKNOWN,
            TruthValue.UNKNOWN,
            ForensicCode.NOT_OBSERVABLE,
            1,
        ),
    ],
)
def test_filter_findings_are_gated_by_root_without_losing_atomic_evidence(
    operator: LogicalOp,
    left: TruthValue,
    right: TruthValue,
    expected_root: TruthValue,
    expected_code: ForensicCode | None,
    expected_ordinal: int | None,
) -> None:
    result = _multi_filter(
        FilterLogical(
            op=operator,
            children=[_truth_leaf("a", left), _truth_leaf("b", right)],
        ),
        ("a", "b"),
    )

    assert result.filter_root_result is expected_root
    assert len(result.filter_evidence) == 2
    if expected_code is None:
        assert result.observations == ()
    else:
        assert [observation.code for observation in result.observations] == [expected_code]
        value = result.observations[0].evidence[0].value
        assert isinstance(value, FilterPredicateEvidenceValue)
        assert value.predicate_ordinal == expected_ordinal


def _balanced_filter(names: tuple[str, ...]) -> object:
    nodes: list[object] = [
        FilterPredicate(field=name, op=PredicateOp.EQ, value=None) for name in names
    ]
    while len(nodes) > 1:
        nodes = [
            FilterLogical(op=LogicalOp.AND, children=nodes[index : index + 2])
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0]


def test_filter_bounds_accept_16_predicates_31_nodes_and_depth_8() -> None:
    names = tuple(f"field_{index}" for index in range(16))
    full = _multi_filter(_balanced_filter(names), names)
    assert len(full.filter_evidence) == 16

    deep: object = FilterPredicate(field="deep", op=PredicateOp.EQ, value=None)
    for _ in range(7):
        deep = FilterLogical(op=LogicalOp.NOT, children=[deep])
    assert _multi_filter(deep, ("deep",)).filter_root_result is TruthValue.FALSE


def test_filter_bounds_reject_17_predicates_32_nodes_and_depth_9() -> None:
    names = tuple(f"field_{index}" for index in range(16))
    thirty_two = FilterLogical(op=LogicalOp.NOT, children=[_balanced_filter(names)])
    with pytest.raises(DiagnosticAnalysisError) as nodes:
        _multi_filter(thirty_two, names)
    assert nodes.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER

    names_17 = tuple(f"wide_{index}" for index in range(17))
    with pytest.raises(DiagnosticAnalysisError):
        _multi_filter(_balanced_filter(names_17), names_17)

    deep: object = FilterPredicate(field="deep", op=PredicateOp.EQ, value=None)
    for _ in range(8):
        deep = FilterLogical(op=LogicalOp.NOT, children=[deep])
    with pytest.raises(DiagnosticAnalysisError):
        _multi_filter(deep, ("deep",))


@pytest.mark.parametrize(
    ("operator", "right", "value_type", "observed"),
    [
        (PredicateOp.EQ, True, FilterValueType.INT, 1),
        (PredicateOp.EQ, 1, FilterValueType.BOOL, True),
        (PredicateOp.LT, None, FilterValueType.INT, 1),
        (PredicateOp.IN, [1, True], FilterValueType.INT, 1),
        (PredicateOp.CONTAINS_ANY, [1, True], FilterValueType.INT_ARRAY, (1,)),
        (PredicateOp.CONTAINS_ANY, [1], FilterValueType.INT_ARRAY, [1]),
        (PredicateOp.EQ, float("nan"), FilterValueType.FLOAT, 1.0),
        (PredicateOp.EQ, 1.0, FilterValueType.FLOAT, float("nan")),
        (PredicateOp.EQ, -1, FilterValueType.UINT, 1),
        (PredicateOp.EQ, 1, FilterValueType.UINT, -1),
    ],
)
def test_invalid_and_coercion_filter_shapes_fail_closed(
    operator: PredicateOp,
    right: object,
    value_type: FilterValueType,
    observed: object,
) -> None:
    with pytest.raises(DiagnosticAnalysisError) as raised:
        _one(
            operator=operator,
            right=right,
            value_type=value_type,
            presence=AttributePresence.PRESENT_VALUE,
            observed=observed,
        )
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_constructed_logical_and_presence_attacks_fail_as_fixed_errors() -> None:
    predicate = FilterPredicate(field="a", op=PredicateOp.EQ, value=None)
    malformed = FilterLogical.model_construct(
        kind="logical", op=LogicalOp.AND, children=(predicate,)
    )
    with pytest.raises(DiagnosticAnalysisError) as logical:
        _multi_filter(malformed, ("a",))
    assert logical.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER

    with pytest.raises(DiagnosticAnalysisError) as presence:
        _analyze_filter(
            predicate,
            schemas=(FilterFieldSchema("a", FilterValueType.STRING, True),),
            attributes=(
                ObservedFilterAttribute(
                    "a",
                    PreservedAttribute("missing", None),  # type: ignore[arg-type]
                ),
            ),
        )
    assert presence.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER


@pytest.mark.parametrize("node_type", ["predicate", "logical"])
@pytest.mark.parametrize("construction", ["model_copy", "model_construct"])
def test_forged_kind_discriminators_fail_before_hostile_comparison_in_both_entrypoints(
    node_type: str,
    construction: str,
) -> None:
    calls: list[str] = []
    marker = f"PRIVATE_{node_type.upper()}_{construction.upper()}_KIND"

    class HostileKind(str):
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return True

        def __ne__(self, other: object) -> bool:
            calls.append("ne")
            return False

    forged_kind = HostileKind(marker)
    predicate = FilterPredicate(field="safe", op=PredicateOp.EQ, value="x")
    if node_type == "predicate":
        node = (
            predicate.model_copy(update={"kind": forged_kind})
            if construction == "model_copy"
            else FilterPredicate.model_construct(
                kind=forged_kind,
                field="safe",
                op=PredicateOp.EQ,
                value="x",
            )
        )
    else:
        logical = FilterLogical(op=LogicalOp.AND, children=[predicate])
        node = (
            logical.model_copy(update={"kind": forged_kind})
            if construction == "model_copy"
            else FilterLogical.model_construct(
                kind=forged_kind,
                op=LogicalOp.AND,
                children=[predicate],
            )
        )
    assert calls == []
    assert type(node.kind) is HostileKind

    schema = (FilterFieldSchema("safe", FilterValueType.STRING, True),)
    for invoke in (
        lambda: preflight_filter_definition(FilterDefinitionInput(node=node, schema=schema)),
        lambda: _analyze_filter(
            node,
            schemas=schema,
            attributes=(
                ObservedFilterAttribute(
                    "safe",
                    PreservedAttribute(AttributePresence.PRESENT_VALUE, "x"),
                ),
            ),
        ),
    ):
        with pytest.raises(DiagnosticAnalysisError) as raised:
            invoke()
        error = raised.value
        assert error.code is DiagnosticAnalysisErrorCode.INVALID_FILTER
        assert error.__cause__ is error.__context__ is None
        assert marker not in str(error)
        assert marker not in repr(error)
        assert marker not in "".join(traceback.format_exception(error))
        current = error.__traceback__
        while current is not None:
            if current.tb_frame.f_code.co_filename.endswith("diagnostic_analysis.py"):
                assert marker not in repr(current.tb_frame.f_locals)
            current = current.tb_next
    assert calls == []


@pytest.mark.parametrize(
    "attacked_field", ["schema_field", "predicate_field", "predicate_op", "logical_op"]
)
def test_adjacent_filter_discriminators_authenticate_exact_types_before_use(
    attacked_field: str,
) -> None:
    calls: list[str] = []

    class HostileScalar(str):
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return True

        def __ne__(self, other: object) -> bool:
            calls.append("ne")
            return False

        def __hash__(self) -> int:
            calls.append("hash")
            return super().__hash__()

    hostile = HostileScalar(f"PRIVATE_{attacked_field.upper()}_MARKER")
    predicate = FilterPredicate(field="safe", op=PredicateOp.EQ, value="x")
    schema = (FilterFieldSchema("safe", FilterValueType.STRING, True),)
    node: object = predicate
    if attacked_field == "schema_field":
        schema = (FilterFieldSchema(hostile, FilterValueType.STRING, True),)
    elif attacked_field == "predicate_field":
        node = predicate.model_copy(update={"field": hostile})
    elif attacked_field == "predicate_op":
        node = predicate.model_copy(update={"op": hostile})
    else:
        node = FilterLogical(op=LogicalOp.AND, children=[predicate]).model_copy(
            update={"op": hostile}
        )
    assert calls == []

    with pytest.raises(DiagnosticAnalysisError) as raised:
        preflight_filter_definition(FilterDefinitionInput(node=node, schema=schema))
    assert raised.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert calls == []


def test_signed_truth_witnesses_follow_nested_not_polarity_without_false_leaf_claims() -> None:
    false_leaf = _truth_leaf("a", TruthValue.FALSE)
    double_not = FilterLogical(
        op=LogicalOp.NOT,
        children=[FilterLogical(op=LogicalOp.NOT, children=[false_leaf])],
    )
    recovered = _multi_filter(double_not, ("a",))
    assert recovered.filter_root_result is TruthValue.FALSE
    assert [item.code for item in recovered.observations] == [ForensicCode.FILTER_PREDICATE_FAILED]

    non_witness = _multi_filter(
        FilterLogical(
            op=LogicalOp.NOT,
            children=[
                FilterLogical(
                    op=LogicalOp.OR,
                    children=[
                        _truth_leaf("a", TruthValue.TRUE),
                        _truth_leaf("b", TruthValue.FALSE),
                    ],
                )
            ],
        ),
        ("a", "b"),
    )
    assert non_witness.filter_root_result is TruthValue.FALSE
    assert [item.result for item in non_witness.filter_evidence] == [
        DiagnosticPredicateResult.MATCHED,
        DiagnosticPredicateResult.NOT_MATCHED,
    ]
    assert non_witness.observations == ()


@pytest.mark.parametrize(
    ("operator", "right", "observed", "expected"),
    [
        (PredicateOp.EQ, "2026-08-23", "2026-08-23T00:00:00", TruthValue.TRUE),
        (
            PredicateOp.EQ,
            "2026-08-23T12:00:00-07:00",
            "2026-08-23T19:00:00Z",
            TruthValue.TRUE,
        ),
        (
            PredicateOp.EQ,
            "1970-01-01T00:00:00.0001Z",
            "1970-01-01T00:00:00.0009Z",
            TruthValue.TRUE,
        ),
        (
            PredicateOp.LT,
            "1970-01-01T00:00:00Z",
            "1969-12-31T23:59:59.9999Z",
            TruthValue.TRUE,
        ),
        (
            PredicateOp.GT,
            "2026-08-23T18:59:59.999Z",
            "2026-08-23T12:00:00-07:00",
            TruthValue.TRUE,
        ),
    ],
)
def test_datetime_values_use_provider_compatible_iso8601_integer_milliseconds(
    operator: PredicateOp,
    right: str,
    observed: str,
    expected: TruthValue,
) -> None:
    result = _one(
        operator=operator,
        right=right,
        value_type=FilterValueType.DATETIME,
        presence=AttributePresence.PRESENT_VALUE,
        observed=observed,
    )
    assert result.filter_root_result is expected


def test_datetime_and_uuid_array_values_share_typed_normalization() -> None:
    datetimes = _one(
        operator=PredicateOp.CONTAINS_ANY,
        right=["2026-08-23T19:00:00Z"],
        value_type=FilterValueType.DATETIME_ARRAY,
        presence=AttributePresence.PRESENT_VALUE,
        observed=("2026-08-23T12:00:00-07:00",),
    )
    uuids = _one(
        operator=PredicateOp.CONTAINS_ANY,
        right=["00000000-0000-0000-0000-0000000000ab"],
        value_type=FilterValueType.UUID_ARRAY,
        presence=AttributePresence.PRESENT_VALUE,
        observed=("000000000000000000000000000000AB",),
    )
    assert datetimes.filter_root_result is uuids.filter_root_result is TruthValue.TRUE


@pytest.mark.parametrize(
    "invalid",
    [
        "2026-02-30",
        "2026-08-23 12:00:00Z",
        "2026-08-23T24:00:00Z",
        "2026-08-23T12:00:00+24:00",
        "2026-08-23T12:00:00.1234567890Z",
        "not-a-datetime",
    ],
)
def test_invalid_datetime_lexemes_fail_closed_without_string_comparison(invalid: str) -> None:
    with pytest.raises(DiagnosticAnalysisError) as operand:
        _one(
            operator=PredicateOp.EQ,
            right=invalid,
            value_type=FilterValueType.DATETIME,
            presence=AttributePresence.PRESENT_VALUE,
            observed="2026-08-23",
        )
    assert operand.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER

    with pytest.raises(DiagnosticAnalysisError) as attribute:
        _one(
            operator=PredicateOp.EQ,
            right="2026-08-23",
            value_type=FilterValueType.DATETIME,
            presence=AttributePresence.PRESENT_VALUE,
            observed=invalid,
        )
    assert attribute.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER


def test_uuid_values_use_typed_identity_without_narrow_lexical_canonicalization() -> None:
    equivalent = _one(
        operator=PredicateOp.EQ,
        right="00000000-0000-0000-0000-0000000000ab",
        value_type=FilterValueType.UUID,
        presence=AttributePresence.PRESENT_VALUE,
        observed="{00000000-0000-0000-0000-0000000000AB}",
    )
    assert equivalent.filter_root_result is TruthValue.TRUE

    with pytest.raises(DiagnosticAnalysisError) as invalid:
        _one(
            operator=PredicateOp.EQ,
            right="not-a-uuid",
            value_type=FilterValueType.UUID,
            presence=AttributePresence.PRESENT_VALUE,
            observed="00000000-0000-0000-0000-0000000000ab",
        )
    assert invalid.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER


def test_definition_preflight_is_schema_only_exact_and_filterable() -> None:
    definition = FilterDefinitionInput(
        node=FilterLogical(
            op=LogicalOp.AND,
            children=[
                FilterPredicate(field="published_at", op=PredicateOp.GTE, value="2026-08-23"),
                FilterPredicate(
                    field="document_uuid",
                    op=PredicateOp.EQ,
                    value="000000000000000000000000000000ab",
                ),
            ],
        ),
        schema=(
            FilterFieldSchema("published_at", FilterValueType.DATETIME, True),
            FilterFieldSchema("document_uuid", FilterValueType.UUID, True),
        ),
    )
    assert preflight_filter_definition(definition) == ("published_at", "document_uuid")

    for attacked in (
        FilterDefinitionInput(
            node=definition.node,
            schema=(
                FilterFieldSchema("published_at", FilterValueType.DATETIME, False),
                FilterFieldSchema("document_uuid", FilterValueType.UUID, True),
            ),
        ),
        FilterDefinitionInput(
            node=FilterPredicate(field="missing", op=PredicateOp.EQ, value="value"),
            schema=(FilterFieldSchema("published_at", FilterValueType.STRING, True),),
        ),
        FilterDefinitionInput(
            node=FilterPredicate(field="published_at", op=PredicateOp.CONTAINS_ANY, value=[]),
            schema=(FilterFieldSchema("published_at", FilterValueType.DATETIME, True),),
        ),
    ):
        with pytest.raises(DiagnosticAnalysisError) as rejected:
            preflight_filter_definition(attacked)
        assert rejected.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER

    with pytest.raises(DiagnosticAnalysisError):
        preflight_filter_definition(
            FilterAnalysisInput(
                node=definition.node,
                schema=definition.schema,
                attributes=(),
            )  # type: ignore[arg-type]
        )


def test_definition_preflight_enforces_the_exact_ast_bounds_before_evaluation() -> None:
    fields = tuple(f"preflight_{index}" for index in range(16))
    accepted = FilterDefinitionInput(
        node=_balanced_filter(fields),
        schema=tuple(FilterFieldSchema(field, FilterValueType.STRING, True) for field in fields),
    )
    assert preflight_filter_definition(accepted) == fields

    fields_17 = (*fields, "preflight_16")
    with pytest.raises(DiagnosticAnalysisError) as too_many:
        preflight_filter_definition(
            FilterDefinitionInput(
                node=_balanced_filter(fields_17),
                schema=tuple(
                    FilterFieldSchema(field, FilterValueType.STRING, True) for field in fields_17
                ),
            )
        )
    assert too_many.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER

    deep: object = FilterPredicate(field="deep", op=PredicateOp.EQ, value=None)
    for _ in range(8):
        deep = FilterLogical(op=LogicalOp.NOT, children=[deep])
    with pytest.raises(DiagnosticAnalysisError) as too_deep:
        preflight_filter_definition(
            FilterDefinitionInput(
                node=deep,
                schema=(FilterFieldSchema("deep", FilterValueType.STRING, True),),
            )
        )
    assert too_deep.value.code is DiagnosticAnalysisErrorCode.INVALID_FILTER
