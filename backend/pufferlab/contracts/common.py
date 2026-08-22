"""Shared primitive contracts."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

type ContractVersion = Literal[1]
type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


class ContractModel(BaseModel):
    """Base model with strict input handling."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ScoreKind(StrEnum):
    BM25 = "bm25"
    VECTOR_DISTANCE = "vector_distance"
    RRF = "rrf"
    RERANKER = "reranker"


class ScoreDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class ScoreSource(StrEnum):
    TURBOPUFFER_DIST = "turbopuffer_dist"
    COMPUTE_ATTRIBUTE = "compute_attribute"
    CLIENT_COMPUTED = "client_computed"
    RERANKER = "reranker"


class ObservedScore(ContractModel):
    kind: ScoreKind
    value: float
    direction: ScoreDirection
    source: ScoreSource

    @model_validator(mode="after")
    def validate_direction(self) -> "ObservedScore":
        expected = (
            ScoreDirection.LOWER_IS_BETTER
            if self.kind is ScoreKind.VECTOR_DISTANCE
            else ScoreDirection.HIGHER_IS_BETTER
        )
        if self.direction is not expected:
            msg = f"{self.kind.value} scores must use {expected.value}"
            raise ValueError(msg)
        return self
