"""Reserved P1 forensic observation contracts."""

from enum import StrEnum
from typing import Literal

from pufferlab.contracts.common import ContractModel, JsonValue


class ForensicCode(StrEnum):
    FILTER_PREDICATE_FAILED = "filter_predicate_failed"
    NO_LEXICAL_SCORE = "no_lexical_score"
    OUTSIDE_LEXICAL_CANDIDATES = "outside_lexical_candidates"
    OUTSIDE_VECTOR_CANDIDATES = "outside_vector_candidates"
    OUTSIDE_FUSION_TOP_K = "outside_fusion_top_k"
    RERANKED_DOWN = "reranked_down"
    NOT_OBSERVABLE = "not_observable"


class EvidenceItem(ContractModel):
    label: str
    value: JsonValue
    source: Literal[
        "query_response",
        "compute_attribute",
        "local_filter_evaluation",
        "counterfactual_query",
        "client_computation",
        "reranker",
    ]


class ForensicObservation(ContractModel):
    code: ForensicCode
    statement: str
    evidence: list[EvidenceItem]
    certainty: Literal["observed", "counterfactual", "insufficient"]
