"""Deterministic dataset loading, identity, and ingestion primitives."""

from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID, document_uuid
from pufferlab.datasets.ingestion import IngestionService
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec

__all__ = [
    "PUFFERLAB_NAMESPACE_UUID",
    "IngestionService",
    "NamespaceWriteSpec",
    "compile_namespace_write_spec",
    "document_uuid",
    "load_fixture_corpus",
]
