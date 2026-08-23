from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pufferlab.application.readiness import LocalCapabilityInspector
from pufferlab.config import Settings
from pufferlab.contracts.capabilities import (
    CapabilityActionCode,
    CapabilityRequirementCode,
    CapabilityState,
)
from pufferlab.main import create_app
from pydantic import SecretStr


def _configured_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "turbopuffer_api_key": "readiness-test-key",
        "pufferlab_search_namespace": "readiness-test-namespace",
        "turbopuffer_region": "gcp-us-west1",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _no_receipt_requirements(
    settings: Settings,
) -> tuple[CapabilityRequirementCode, ...]:
    del settings
    return ()


def test_capability_inspector_reports_configured_without_remote_health_claims() -> None:
    response = LocalCapabilityInspector(
        _configured_settings(),
        runtime_available=lambda: True,
        owned_tiny_requirements=_no_receipt_requirements,
    ).inspect()

    assert response.live_playground.state is CapabilityState.LOCALLY_CONFIGURED
    assert response.live_playground.requirements == ()
    assert response.live_playground.next_action is None
    assert response.model_dump(mode="json") == {
        "contract_version": 1,
        "live_playground": {
            "state": "locally_configured",
            "requirements": [],
            "next_action": None,
        },
    }


def test_capability_inspector_treats_blank_values_as_missing_in_frozen_order() -> None:
    resolver_called = False

    def receipt_resolver(
        settings: Settings,
    ) -> tuple[CapabilityRequirementCode, ...]:
        nonlocal resolver_called
        del settings
        resolver_called = True
        return ()

    response = LocalCapabilityInspector(
        _configured_settings(
            turbopuffer_api_key=" \t ",
            pufferlab_search_namespace=" \n ",
            turbopuffer_region=" ",
        ),
        runtime_available=lambda: False,
        owned_tiny_requirements=receipt_resolver,
    ).inspect()

    assert response.live_playground.requirements == (
        CapabilityRequirementCode.API_KEY,
        CapabilityRequirementCode.SEARCH_NAMESPACE,
        CapabilityRequirementCode.REGION,
        CapabilityRequirementCode.LIVE_SEARCH_RUNTIME,
    )
    assert response.live_playground.next_action is CapabilityActionCode.CONFIGURE_API_KEY
    assert resolver_called is False
    serialized = response.model_dump_json()
    assert "readiness-test" not in serialized


def test_settings_normalizes_an_already_wrapped_blank_secret_before_inspection() -> None:
    settings = _configured_settings(turbopuffer_api_key=SecretStr(" \t "))

    assert settings.turbopuffer_api_key is None


class _FailIfUnwrappedSecret(SecretStr):
    def get_secret_value(self) -> str:
        raise AssertionError("provider-free readiness unwrapped the API key")


def test_capability_inspector_never_unwraps_a_configured_secret() -> None:
    settings = _configured_settings()
    settings.turbopuffer_api_key = _FailIfUnwrappedSecret("opaque-test-key")

    response = LocalCapabilityInspector(
        settings,
        runtime_available=lambda: True,
        owned_tiny_requirements=_no_receipt_requirements,
    ).inspect()

    assert response.live_playground.state is CapabilityState.LOCALLY_CONFIGURED


def test_default_runtime_discovery_does_not_import_the_optional_package() -> None:
    was_imported = "sentence_transformers" in sys.modules

    LocalCapabilityInspector(_configured_settings()).inspect()

    assert ("sentence_transformers" in sys.modules) is was_imported


def test_default_receipt_resolver_fails_closed_until_m4_c() -> None:
    response = LocalCapabilityInspector(
        _configured_settings(),
        runtime_available=lambda: True,
    ).inspect()

    assert response.live_playground.requirements == (
        CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
    )
    assert response.live_playground.next_action is (CapabilityActionCode.RESOLVE_OWNED_TINY_RECEIPT)


def test_receipt_requirements_preserve_the_frozen_tail_order() -> None:
    response = LocalCapabilityInspector(
        _configured_settings(),
        runtime_available=lambda: True,
        owned_tiny_requirements=lambda settings: (
            CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
            CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
        ),
    ).inspect()

    assert response.live_playground.requirements == (
        CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
        CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
    )
    assert response.live_playground.next_action is (CapabilityActionCode.USE_OWNED_TINY_CREDENTIAL)


@pytest.mark.parametrize(
    "resolver",
    [
        lambda settings: (CapabilityRequirementCode.API_KEY,),
        lambda settings: (
            CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
            CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
        ),
        lambda settings: (
            CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
            CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
        ),
    ],
)
def test_receipt_resolver_cannot_escape_its_allowlisted_order(
    resolver: Callable[[Settings], tuple[CapabilityRequirementCode, ...]],
) -> None:
    with pytest.raises(ValueError, match="owned-tiny readiness"):
        LocalCapabilityInspector(
            _configured_settings(),
            runtime_available=lambda: True,
            owned_tiny_requirements=resolver,
        ).inspect()


class _NoopSearchBackend:
    async def close(self) -> None:
        return None


def test_default_capability_request_constructs_no_provider_model_or_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "must-not-exist"

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("capability request crossed a side-effect boundary")

    monkeypatch.setattr("pufferlab.main.Database.from_settings", fail)
    monkeypatch.setattr("pufferlab.main.EvaluationApiRuntime", fail)
    monkeypatch.setattr("pufferlab.main.RuntimeSearchBackend.from_settings", fail)
    monkeypatch.setattr("pufferlab.retrieval.runtime.TurbopufferProvider", fail)
    monkeypatch.setattr("pufferlab.retrieval.runtime.SentenceTransformerQueryEmbedder", fail)
    monkeypatch.setattr("pufferlab.retrieval.runtime.SentenceTransformersReranker", fail)
    app = create_app(
        Settings.model_validate(
            {
                "pufferlab_data_dir": data_dir,
                "turbopuffer_api_key": None,
                "pufferlab_search_namespace": None,
            }
        ),
        search_backend=_NoopSearchBackend(),
        evaluation_views=object(),  # type: ignore[arg-type]
        evaluation_controls=object(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["live_playground"] == {
        "state": "action_required",
        "requirements": ["api_key", "search_namespace", "live_search_runtime"],
        "next_action": "configure_api_key",
    }
    assert not data_dir.exists()


def test_capability_route_uses_injected_inspector_without_provider_work() -> None:
    inspector = LocalCapabilityInspector(
        _configured_settings(),
        runtime_available=lambda: True,
        owned_tiny_requirements=_no_receipt_requirements,
    )
    app = create_app(
        _configured_settings(),
        search_backend=_NoopSearchBackend(),
        evaluation_views=object(),  # type: ignore[arg-type]
        evaluation_controls=object(),  # type: ignore[arg-type]
        capability_inspector=inspector,
    )

    response = TestClient(app).get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["live_playground"]["state"] == "locally_configured"
