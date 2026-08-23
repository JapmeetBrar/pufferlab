"""Provider-free browser-test API with fatal construction and mutation tripwires."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Never

from fastapi import Request
from pufferlab.application.evaluation_runtime import EvaluationApiRuntime
from pufferlab.config import Settings
from pufferlab.datasets.loader import load_fixture_corpus
from pufferlab.main import create_app
from pufferlab.retrieval.runtime import RuntimeSearchBackend
from starlette.responses import PlainTextResponse, Response

_MARKER = Path(os.environ["PUFFERLAB_E2E_GUARD_MARKER"])


def _trip_guard(name: str) -> Never:
    _MARKER.write_text(f"unexpected={name}\n", encoding="utf-8")
    raise RuntimeError("provider-free browser guard tripped")


def _poison_provider_factory(*_args: object, **_kwargs: object) -> Never:
    _trip_guard("provider_factory")


def _poison_embedder_factory(*_args: object, **_kwargs: object) -> Never:
    _trip_guard("embedder_factory")


def _poison_reranker_factory(*_args: object, **_kwargs: object) -> Never:
    _trip_guard("reranker_factory")


def _poison_search_backend_factory(*_args: object, **_kwargs: object) -> Never:
    _trip_guard("search_backend_factory")


settings = Settings()
fixture = load_fixture_corpus(settings.pufferlab_fixture_dir)
search_backend = RuntimeSearchBackend(
    settings=settings,
    manifest=fixture.manifest,
    provider_factory=_poison_provider_factory,
    embedder_factory=_poison_embedder_factory,
    reranker_factory=_poison_reranker_factory,
    optional_runtime_available=lambda: False,
)
evaluation_runtime = EvaluationApiRuntime(
    settings,
    provider_factory=_poison_provider_factory,
    embedder_factory=_poison_embedder_factory,
    reranker_factory=_poison_reranker_factory,
    search_backend_factory=_poison_search_backend_factory,
)
app = create_app(
    settings,
    search_backend=search_backend,
    evaluation_runtime=evaluation_runtime,
)


@app.middleware("http")
async def reject_browser_mutations(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        try:
            _trip_guard("browser_post")
        except RuntimeError:
            return PlainTextResponse("browser mutations are disabled", status_code=405)
    return await call_next(request)
