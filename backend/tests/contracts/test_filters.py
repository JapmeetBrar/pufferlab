import pytest
from pufferlab.contracts.filters import FilterNode, LogicalOp
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
