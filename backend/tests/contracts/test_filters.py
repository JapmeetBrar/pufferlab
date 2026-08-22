import pytest
from pufferlab.contracts.filters import FilterNode, FilterPredicate, LogicalOp
from pydantic import TypeAdapter, ValidationError


def test_filter_ast_parses_by_kind() -> None:
    value = TypeAdapter(FilterNode).validate_python(
        {
            "kind": "logical",
            "op": "and",
            "children": [{"kind": "predicate", "field": "source", "op": "eq", "value": "unix"}],
        }
    )

    assert value.op is LogicalOp.AND


def test_not_requires_exactly_one_child() -> None:
    with pytest.raises(ValidationError, match="exactly one child"):
        TypeAdapter(FilterNode).validate_python(
            {
                "kind": "logical",
                "op": "not",
                "children": [
                    {"kind": "predicate", "field": "source", "op": "eq", "value": "unix"},
                    {"kind": "predicate", "field": "source", "op": "eq", "value": "tex"},
                ],
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        ["valid", float("-inf")],
        {"nested": [float("nan")]},
    ],
)
def test_filter_values_reject_nonfinite_numbers_recursively(value: object) -> None:
    with pytest.raises(ValidationError):
        FilterPredicate(field="external_id", op="eq", value=value)


@pytest.mark.parametrize(
    ("op", "value", "message"),
    [
        ("in", "not-an-array", "require an array"),
        ("contains_any", "not-an-array", "require an array"),
        ("eq", ["not-a-scalar"], "require a scalar"),
        ("gte", {"not": "a scalar"}, "require a scalar"),
    ],
)
def test_filter_operator_value_shapes_are_validated(
    op: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        FilterPredicate(field="external_id", op=op, value=value)
