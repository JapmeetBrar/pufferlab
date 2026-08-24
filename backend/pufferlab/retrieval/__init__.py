"""Interactive retrieval services."""

from pufferlab.retrieval.config import (
    BoundSearchCatalog,
    SearchConfigCatalog,
    SeededSearchConfig,
    bind_retrieval_catalog,
    build_search_catalog,
    derive_bound_retrieval_configs,
)
from pufferlab.retrieval.diagnostic_types import is_valid_diagnostic_region
from pufferlab.retrieval.service import SearchCompareService
from pufferlab.retrieval.types import SearchExecuteRequest, SearchExecuteResult

__all__ = [
    "BoundSearchCatalog",
    "SearchCompareService",
    "SearchConfigCatalog",
    "SearchExecuteRequest",
    "SearchExecuteResult",
    "SeededSearchConfig",
    "bind_retrieval_catalog",
    "build_search_catalog",
    "derive_bound_retrieval_configs",
    "is_valid_diagnostic_region",
]
