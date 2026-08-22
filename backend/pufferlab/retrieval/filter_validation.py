"""Schema-bound validation for the neutral filter AST."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pufferlab.contracts.common import JsonValue
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterNode,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.datasets.schema import NamespaceAttributeWriteSpec, NamespaceWriteSpec
from pufferlab.retrieval.errors import invalid_search

_EQUALITY_OPERATORS = {PredicateOp.EQ, PredicateOp.NOT_EQ}
_ORDER_OPERATORS = {
    PredicateOp.LT,
    PredicateOp.LTE,
    PredicateOp.GT,
    PredicateOp.GTE,
}
_ORDERABLE_TYPES = {"string", "datetime", "int", "uint", "float"}


class FixtureFilterValidator:
    """Validate filter fields, operators, and values before provider execution."""

    def __init__(self, write_spec: NamespaceWriteSpec) -> None:
        self._attributes = dict(write_spec.attributes)

    def validate(self, node: FilterNode) -> None:
        if isinstance(node, FilterPredicate):
            self._validate_predicate(node)
            return
        if not isinstance(node, FilterLogical):
            raise invalid_search("filter expression is invalid")
        if node.op not in {LogicalOp.AND, LogicalOp.OR, LogicalOp.NOT}:
            raise invalid_search("filter expression is invalid")
        if not node.children:
            raise invalid_search("filter expression is invalid")
        if node.op is LogicalOp.NOT and len(node.children) != 1:
            raise invalid_search("filter expression is invalid")
        for child in node.children:
            self.validate(child)

    def _validate_predicate(self, predicate: FilterPredicate) -> None:
        specification = self._attributes.get(predicate.field)
        if specification is None:
            raise invalid_search("filter field is not available")
        if specification.filterable is not True:
            raise invalid_search("filter field is not filterable")
        if not _is_finite_json(predicate.value):
            raise invalid_search("filter value must contain only finite JSON numbers")
        if not _operator_accepts_value(
            predicate.op,
            predicate.value,
            specification=specification,
        ):
            raise invalid_search("filter value is invalid for its field and operator")


def _operator_accepts_value(
    operator: PredicateOp,
    value: JsonValue,
    *,
    specification: NamespaceAttributeWriteSpec,
) -> bool:
    attribute_type = specification.type
    is_array = attribute_type.startswith("[]")
    scalar_type = attribute_type[2:] if is_array else attribute_type

    if operator is PredicateOp.IN:
        return (
            not is_array
            and isinstance(value, list)
            and all(_matches_scalar(item, scalar_type, allow_null=False) for item in value)
        )
    if operator is PredicateOp.CONTAINS_ANY:
        return (
            is_array
            and isinstance(value, list)
            and all(_matches_scalar(item, scalar_type, allow_null=False) for item in value)
        )
    if operator in _EQUALITY_OPERATORS:
        return _matches_scalar(value, scalar_type, allow_null=True) if not is_array else False
    if operator in _ORDER_OPERATORS:
        return (
            not is_array
            and scalar_type in _ORDERABLE_TYPES
            and _matches_scalar(value, scalar_type, allow_null=False)
        )
    return False


def _matches_scalar(value: JsonValue, scalar_type: str, *, allow_null: bool) -> bool:
    if value is None:
        return allow_null
    if isinstance(value, Mapping | list):
        return False
    if scalar_type in {"string", "datetime", "uuid"}:
        return isinstance(value, str)
    if scalar_type == "bool":
        return isinstance(value, bool)
    if scalar_type in {"int", "uint"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if scalar_type == "float":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return False


def _is_finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if value is None or isinstance(value, str | int | bool):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return all(_is_finite_json(item) for item in value)
    return False
