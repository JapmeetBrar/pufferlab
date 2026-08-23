"""FastAPI dependencies shared by search routes."""

from fastapi import Request

from pufferlab.application.readiness import CapabilityInspector
from pufferlab.retrieval.errors import search_unavailable
from pufferlab.retrieval.types import SearchBackend


def get_capability_inspector(request: Request) -> CapabilityInspector:
    inspector: CapabilityInspector | None = getattr(
        request.app.state,
        "capability_inspector",
        None,
    )
    if inspector is None:
        raise RuntimeError("local capability inspector is not configured")
    return inspector


def get_search_backend(request: Request) -> SearchBackend:
    backend: SearchBackend | None = getattr(request.app.state, "search_backend", None)
    if backend is None:
        raise search_unavailable()
    return backend
