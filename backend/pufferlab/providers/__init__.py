"""External provider adapters."""

from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.providers.metadata_probe import (
    MetadataProbeConfigurationError,
    MetadataProbeRequestError,
    MetadataProbeResult,
    MetadataProbeState,
    is_valid_metadata_probe_region,
    metadata_request_sanitizer,
    probe_namespace_metadata,
)
from pufferlab.providers.turbopuffer import TurbopufferProvider, filter_to_turbopuffer
from pufferlab.providers.types import (
    AnnIndexSchema,
    AttributeSchema,
    FullTextSearchSchema,
    ProviderDeleteResult,
    ProviderDocument,
    ProviderDocumentIdInventory,
    ProviderNamespaceMetadata,
    ProviderQueryResult,
    ProviderWriteResult,
    WriteDocument,
)

__all__ = [
    "AnnIndexSchema",
    "AttributeSchema",
    "FullTextSearchSchema",
    "MetadataProbeConfigurationError",
    "MetadataProbeRequestError",
    "MetadataProbeResult",
    "MetadataProbeState",
    "ProviderDeleteResult",
    "ProviderDocument",
    "ProviderDocumentIdInventory",
    "ProviderError",
    "ProviderErrorDetails",
    "ProviderNamespaceMetadata",
    "ProviderQueryResult",
    "ProviderWriteResult",
    "TurbopufferProvider",
    "WriteDocument",
    "filter_to_turbopuffer",
    "is_valid_metadata_probe_region",
    "metadata_request_sanitizer",
    "probe_namespace_metadata",
]
