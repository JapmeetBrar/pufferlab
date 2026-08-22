from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pufferlab.config import Settings
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.providers.types import ProviderDeleteResult

from scripts import live_namespace_session, verify_live_api


def _hit(*, mode: str, external_id: str) -> dict[str, object]:
    kind = "bm25" if mode == "bm25" else "vector_distance"
    direction = "higher_is_better" if mode == "bm25" else "lower_is_better"
    return {
        "document_id": "36a3ec84-361d-5171-81af-564961663f06",
        "external_id": external_id,
        "final_rank": 1,
        "final_score": {
            "kind": kind,
            "value": 0.5,
            "direction": direction,
            "source": "turbopuffer_dist",
        },
    }


def _result(*, mode: str, identifier: str, external_id: str) -> dict[str, object]:
    return {
        "config": {"id": identifier, "mode": mode},
        "hits": [_hit(mode=mode, external_id=external_id)],
        "timings": [
            {
                "stage": "turbopuffer",
                "duration_ms": 2.5,
                "measurement": "client_wall_clock",
            },
            {"stage": "total", "duration_ms": 4.0, "measurement": "client_wall_clock"},
        ],
    }


def test_live_api_verifier_executes_exact_public_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    bm25_id = "c90061ae-ec7a-518c-a215-043304c0be57"
    vector_id = "dd99afdb-d427-5aa1-b2b4-fbaae665be69"
    observed_payloads: list[dict[str, verify_live_api.JsonValue] | None] = []

    def fake_request(
        method: str,
        url: str,
        payload: dict[str, verify_live_api.JsonValue] | None = None,
    ) -> dict[str, verify_live_api.JsonValue]:
        observed_payloads.append(payload)
        if url.endswith("/health"):
            return {"contract_version": 1, "status": "ok", "version": "0.1.0"}
        if url.endswith("/configs"):
            return {
                "contract_version": 1,
                "configs": [
                    {"id": bm25_id, "mode": "bm25"},
                    {"id": vector_id, "mode": "vector"},
                ],
            }
        assert method == "POST"
        return {
            "contract_version": 1,
            "query_text": "How can I find the program listening on port 8080?",
            "results": [
                _result(mode="bm25", identifier=bm25_id, external_id="tiny-002"),
                _result(mode="vector", identifier=vector_id, external_id="tiny-002"),
            ],
        }

    monkeypatch.setattr(verify_live_api, "_request_json", fake_request)

    result = verify_live_api.verify("http://127.0.0.1:8000")

    assert result["live_api_verification"] == "passed"
    assert observed_payloads[-1] == {
        "contract_version": 1,
        "query_text": "How can I find the program listening on port 8080?",
        "config_ids": [bm25_id, vector_id],
        "debug_provenance": True,
    }


def test_live_api_verifier_rejects_remote_origins_and_private_fields() -> None:
    with pytest.raises(verify_live_api.VerificationError, match="loopback"):
        verify_live_api.verify("https://example.com")
    with pytest.raises(verify_live_api.VerificationError, match="private field"):
        verify_live_api._reject_private_fields({"attributes": {"vector": [0.1]}})


class _DeletedProvider:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.closed = False

    async def delete_namespace(self, namespace: str) -> ProviderDeleteResult:
        self.deleted.append(namespace)
        return ProviderDeleteResult(client_duration_ms=1.0)

    async def namespace_metadata(self, namespace: str) -> None:
        raise ProviderError(
            "turbopuffer namespace was not found",
            ProviderErrorDetails(
                code=ApiErrorCode.NOT_FOUND,
                retryable=False,
                operation="metadata",
                status_code=404,
            ),
        )

    async def close(self) -> None:
        self.closed = True


class _CloseFailureProvider(_DeletedProvider):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError("safe close failure")


@pytest.mark.asyncio
async def test_live_session_owns_one_exact_namespace_and_confirms_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    session = live_namespace_session.create_session(
        path,
        token_factory=lambda _: "a" * 24,
    )
    assert session.namespace == "pufferlab-tiny-" + "a" * 24
    assert live_namespace_session.load_session(path) == session
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        live_namespace_session.create_session(path)

    provider = _DeletedProvider()

    def provider_factory(*, api_key: str, region: str) -> _DeletedProvider:
        assert api_key == "secret"
        assert region == "gcp-us-central1"
        return provider

    cleaned = await live_namespace_session.cleanup_session(
        path,
        settings=Settings(_env_file=None, turbopuffer_api_key="secret"),
        provider_factory=provider_factory,
    )

    assert cleaned == session
    assert provider.deleted == [session.namespace]
    assert provider.closed
    assert not path.exists()


@pytest.mark.asyncio
async def test_live_session_retains_record_until_provider_closes(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    live_namespace_session.create_session(path, token_factory=lambda _: "b" * 24)
    provider = _CloseFailureProvider()

    def provider_factory(*, api_key: str, region: str) -> _CloseFailureProvider:
        del api_key, region
        return provider

    with pytest.raises(RuntimeError, match="close failure"):
        await live_namespace_session.cleanup_session(
            path,
            settings=Settings(_env_file=None, turbopuffer_api_key="secret"),
            provider_factory=provider_factory,
        )

    assert provider.closed
    assert path.exists()


def test_live_session_refuses_tampered_cleanup_target(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps({"format_version": 1, "namespace": "production"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="refusing cleanup"):
        live_namespace_session.load_session(path)
