"""FastAPI dependencies shared by search routes."""

from fastapi import Request

from pufferlab.retrieval.errors import search_unavailable
from pufferlab.retrieval.types import SearchBackend


def get_search_backend(request: Request) -> SearchBackend:
    backend: SearchBackend | None = getattr(request.app.state, "search_backend", None)
    if backend is None:
        raise search_unavailable()
    return backend
