"""Deterministic dataset loading, identity, and ingestion primitives."""

from pufferlab.datasets.checkpoints import IngestionCheckpointStore
from pufferlab.datasets.cqadupstack import load_curated_unix_corpus, prepare_unix_pack
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID, document_uuid
from pufferlab.datasets.ingestion import IngestionCheckpoint, IngestionService
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.datasets.schema import NamespaceWriteSpec, compile_namespace_write_spec
from pufferlab.datasets.turbopuffer_writer import TurbopufferNamespaceWriter

__all__ = [
    "PUFFERLAB_NAMESPACE_UUID",
    "IngestionCheckpoint",
    "IngestionCheckpointStore",
    "IngestionService",
    "NamespaceWriteSpec",
    "TurbopufferNamespaceWriter",
    "compile_namespace_write_spec",
    "document_uuid",
    "load_curated_unix_corpus",
    "load_fixture_corpus",
    "prepare_unix_pack",
]
