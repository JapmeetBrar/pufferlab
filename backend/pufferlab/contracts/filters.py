"""Validated, provider-neutral filter expression contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pufferlab.contracts.common import ContractModel, JsonValue


class PredicateOp(StrEnum):
    EQ = "eq"
    NOT_EQ = "not_eq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    CONTAINS_ANY = "contains_any"


class LogicalOp(StrEnum):
    AND = "and"
    OR = "or"
    NOT = "not"


class FilterPredicate(ContractModel):
    kind: Literal["predicate"] = "predicate"
    field: str = Field(min_length=1)
    op: PredicateOp
    value: JsonValue

    @model_validator(mode="after")
    def validate_value_shape(self) -> FilterPredicate:
        array_operators = {PredicateOp.IN, PredicateOp.CONTAINS_ANY}
        if self.op in array_operators and not isinstance(self.value, list):
            raise ValueError(f"{self.op.value} filters require an array value")
        if self.op not in array_operators and isinstance(self.value, list | dict):
            raise ValueError(f"{self.op.value} filters require a scalar value")
        return self


class FilterLogical(ContractModel):
    kind: Literal["logical"] = "logical"
    op: LogicalOp
    children: list[FilterPredicate | FilterLogical] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_arity(self) -> FilterLogical:
        if self.op is LogicalOp.NOT and len(self.children) != 1:
            raise ValueError("not filters require exactly one child")
        return self


type FilterNode = Annotated[FilterPredicate | FilterLogical, Field(discriminator="kind")]
