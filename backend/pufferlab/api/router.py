"""Versioned API router."""

from fastapi import APIRouter

from pufferlab.api.routes.catalog import router as catalog_router
from pufferlab.api.routes.configs import router as configs_router
from pufferlab.api.routes.eval_runs import router as eval_runs_router
from pufferlab.api.routes.health import router as health_router
from pufferlab.api.routes.search import router as search_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(configs_router)
api_router.include_router(search_router)
api_router.include_router(catalog_router)
api_router.include_router(eval_runs_router)
