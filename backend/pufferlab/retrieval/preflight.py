"""Provider-free local configuration checks shared by readiness and search."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from pufferlab.config import Settings
from pufferlab.contracts.capabilities import CapabilityRequirementCode

type RuntimeAvailabilityCheck = Callable[[], bool]


def sentence_transformers_available() -> bool:
    """Detect the optional live-search package without importing it or a model."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def local_search_requirements(
    settings: Settings,
    *,
    runtime_available: RuntimeAvailabilityCheck = sentence_transformers_available,
) -> tuple[CapabilityRequirementCode, ...]:
    """Return value-free unmet requirements in the frozen contract order."""
    requirements: list[CapabilityRequirementCode] = []
    if settings.turbopuffer_api_key is None:
        requirements.append(CapabilityRequirementCode.API_KEY)
    if not _text_is_nonblank(settings.pufferlab_search_namespace):
        requirements.append(CapabilityRequirementCode.SEARCH_NAMESPACE)
    if not _text_is_nonblank(settings.turbopuffer_region):
        requirements.append(CapabilityRequirementCode.REGION)
    if not _runtime_is_available(runtime_available):
        requirements.append(CapabilityRequirementCode.LIVE_SEARCH_RUNTIME)
    return tuple(requirements)


def _text_is_nonblank(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _runtime_is_available(check: RuntimeAvailabilityCheck) -> bool:
    try:
        return check() is True
    except Exception:
        return False
