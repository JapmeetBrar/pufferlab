"""External provider adapters."""

from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.providers.turbopuffer import TurbopufferProvider, filter_to_turbopuffer
from pufferlab.providers.types import (
    AnnIndexSchema,
    AttributeSchema,
    FullTextSearchSchema,
    ProviderDeleteResult,
    ProviderDocument,
    ProviderNamespaceMetadata,
    ProviderQueryResult,
    ProviderWriteResult,
    WriteDocument,
)

__all__ = [
    "AnnIndexSchema",
    "AttributeSchema",
    "FullTextSearchSchema",
    "ProviderDeleteResult",
    "ProviderDocument",
    "ProviderError",
    "ProviderErrorDetails",
    "ProviderNamespaceMetadata",
    "ProviderQueryResult",
    "ProviderWriteResult",
    "TurbopufferProvider",
    "WriteDocument",
    "filter_to_turbopuffer",
]
