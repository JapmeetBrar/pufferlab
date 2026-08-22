"""PufferLab FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pufferlab import __version__
from pufferlab.api.errors import (
    internal_error_response,
    provider_error_response,
    search_error_response,
    validation_error_response,
)
from pufferlab.api.router import api_router
from pufferlab.config import Settings, get_settings
from pufferlab.providers.errors import ProviderError
from pufferlab.retrieval.errors import SearchError
from pufferlab.retrieval.types import SearchBackend


def create_app(
    settings: Settings | None = None,
    *,
    search_backend: SearchBackend | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            backend: SearchBackend | None = app.state.search_backend
            if backend is not None:
                await backend.close()

    app = FastAPI(
        title="PufferLab API",
        summary="Search evaluation and query-forensics workbench",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.search_backend = search_backend

    @app.exception_handler(SearchError)
    async def handle_search_error(_: Request, error: SearchError) -> JSONResponse:
        return search_error_response(error)

    @app.exception_handler(ProviderError)
    async def handle_provider_error(_: Request, error: ProviderError) -> JSONResponse:
        return provider_error_response(error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return validation_error_response()

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        return internal_error_response()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(api_router)
    return app


app = create_app()
