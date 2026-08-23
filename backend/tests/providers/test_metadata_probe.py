from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from types import TracebackType

import httpx
import pufferlab.providers.metadata_probe as metadata_probe_module
import pytest
from pufferlab.providers.metadata_probe import (
    MetadataProbeConfigurationError,
    MetadataProbeState,
    is_valid_metadata_probe_region,
    probe_namespace_metadata,
)

_API_KEY = "test-only-metadata-credential"
_REGION = "gcp-us-west1"
_NAMESPACE = "fixture/name?revision=1"
_EXPECTED_URL = (
    "https://gcp-us-west1.turbopuffer.com/v2/namespaces/fixture%2Fname%3Frevision=1/metadata"
)

type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        handler: Handler,
        *,
        close_started: asyncio.Event | None = None,
        close_release: asyncio.Event | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._handler = handler
        self._close_started = close_started
        self._close_release = close_release
        self._close_error = close_error
        self.requests: list[httpx.Request] = []
        self.close_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._handler(request)

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_started is not None:
            self._close_started.set()
        if self._close_release is not None:
            await self._close_release.wait()
        if self._close_error is not None:
            raise self._close_error


def _probe_traceback_frames(traceback_value: TracebackType | None) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    while traceback_value is not None:
        frame = traceback_value.tb_frame
        if frame.f_code.co_filename.endswith("/pufferlab/providers/metadata_probe.py"):
            frames.append(dict(frame.f_locals))
        traceback_value = traceback_value.tb_next
    return frames


def _assert_probe_traceback_scrubbed(
    error: BaseException,
    *,
    secrets: tuple[str, ...],
    transport: RecordingTransport,
) -> None:
    frames = _probe_traceback_frames(error.__traceback__)
    assert frames
    for frame in frames:
        for value in frame.values():
            assert value is not transport
            assert not isinstance(value, httpx.Request | httpx.AsyncClient | RecordingTransport)
            if isinstance(value, str):
                for secret in secrets:
                    if secret:
                        assert secret not in value


def _metadata_body(*, status: str = "up-to-date") -> dict[str, object]:
    index: dict[str, object] = {"status": status}
    if status == "updating":
        index["unindexed_bytes"] = 128
    return {
        "approx_logical_bytes": 256,
        "approx_row_count": 3,
        "created_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:01Z",
        "encryption": {"mode": "default"},
        "index": index,
        "schema": {"body": {"type": "string", "full_text_search": True}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_status", "expected_state"),
    [
        ("up-to-date", MetadataProbeState.INDEX_UP_TO_DATE),
        ("updating", MetadataProbeState.INDEX_UPDATING),
    ],
)
async def test_probe_uses_exact_one_shot_sdk_request_and_normalizes_index_state(
    monkeypatch: pytest.MonkeyPatch,
    provider_status: str,
    expected_state: MetadataProbeState,
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "wrong-environment-credential")
    monkeypatch.setenv("TURBOPUFFER_REGION", "wrong-environment-region")
    monkeypatch.setenv("TURBOPUFFER_BASE_URL", "https://redirect.invalid/{region}")
    monkeypatch.setenv(
        "TURBOPUFFER_CUSTOM_HEADERS",
        "authorization: Bearer wrong-custom-credential\nX-Provider-Leak: forbidden",
    )
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == _EXPECTED_URL
        assert await request.aread() == b""
        assert set(request.headers) == {"accept", "authorization", "host", "user-agent"}
        assert request.headers.get_list("authorization") == [f"Bearer {_API_KEY}"]
        assert request.headers["host"] == "gcp-us-west1.turbopuffer.com"
        timeout = request.extensions["timeout"]
        assert isinstance(timeout, dict)
        assert set(timeout.values()) == {10.0}
        return httpx.Response(200, json=_metadata_body(status=provider_status))

    transport = RecordingTransport(handler)
    result = await probe_namespace_metadata(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )

    assert result.state is expected_state
    assert result.metadata_reachable is True
    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    rendered = repr(result)
    assert _API_KEY not in rendered
    assert _NAMESPACE not in rendered
    assert "wrong-custom-credential" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_state"),
    [
        ("not-found", MetadataProbeState.NOT_FOUND),
        ("rate-limit", MetadataProbeState.REMOTE_FAILURE),
        ("server", MetadataProbeState.REMOTE_FAILURE),
        ("timeout", MetadataProbeState.REMOTE_FAILURE),
        ("connect", MetadataProbeState.REMOTE_FAILURE),
        ("redirect", MetadataProbeState.REMOTE_FAILURE),
    ],
)
async def test_retryable_redirect_and_not_found_paths_make_one_outbound_attempt(
    failure: str,
    expected_state: MetadataProbeState,
) -> None:
    provider_secret = "provider-body-value-must-not-survive"

    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout(provider_secret, request=request)
        if failure == "connect":
            raise httpx.ConnectError(provider_secret, request=request)
        status = {"not-found": 404, "rate-limit": 429, "server": 500, "redirect": 307}[failure]
        headers = {"location": "https://redirect.invalid/stolen"} if status == 307 else None
        return httpx.Response(
            status,
            headers=headers,
            json={"error": provider_secret},
        )

    transport = RecordingTransport(handler)
    result = await probe_namespace_metadata(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )

    assert result.state is expected_state
    assert result.metadata_reachable is False
    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert provider_secret not in str(result)
    assert provider_secret not in repr(result)


@pytest.mark.asyncio
async def test_malformed_sdk_metadata_is_a_detached_remote_failure() -> None:
    provider_secret = "malformed-provider-body-secret"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "index": {"status": provider_secret},
                "schema": {provider_secret: {"type": "string"}},
            },
        )

    transport = RecordingTransport(handler)
    result = await probe_namespace_metadata(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )

    assert result.state is MetadataProbeState.REMOTE_FAILURE
    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert provider_secret not in str(result)
    assert provider_secret not in repr(result)


@pytest.mark.parametrize(
    "region",
    [
        "",
        "-gcp-us-west1",
        "gcp-us-west1-",
        "GCP-us-west1",
        "gcp.us-west1",
        "gcp/us-west1",
        "gcp@us-west1",
        "gcp us-west1",
        "gcp-us-west1\nredirect.invalid",
        "gcp-us-wést1",
        "a" * 64,
    ],
)
@pytest.mark.asyncio
async def test_hostile_region_fails_before_client_or_transport(
    region: str,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be reached")

    transport = RecordingTransport(handler)
    with pytest.raises(MetadataProbeConfigurationError) as raised:
        await probe_namespace_metadata(
            api_key=_API_KEY,
            region=region,
            namespace=_NAMESPACE,
            transport=transport,
        )

    assert is_valid_metadata_probe_region(region) is False
    assert len(transport.requests) == 0
    assert transport.close_calls == 0
    if region:
        assert region not in str(raised.value)
        assert region not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_probe_traceback_scrubbed(
        raised.value,
        secrets=(_API_KEY, _NAMESPACE, region),
        transport=transport,
    )


@pytest.mark.parametrize(
    ("api_key", "namespace"),
    [
        ("", _NAMESPACE),
        ("   ", _NAMESPACE),
        (_API_KEY, ""),
        (_API_KEY, "   "),
        (_API_KEY, "n" * 129),
        (_API_KEY, "é" * 65),
    ],
)
@pytest.mark.asyncio
async def test_blank_key_and_blank_or_overlong_namespace_fail_before_client_construction(
    api_key: str,
    namespace: str,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be reached")

    transport = RecordingTransport(handler)
    with pytest.raises(MetadataProbeConfigurationError) as raised:
        await probe_namespace_metadata(
            api_key=api_key,
            region=_REGION,
            namespace=namespace,
            transport=transport,
        )

    assert len(transport.requests) == 0
    assert transport.close_calls == 0
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_probe_traceback_scrubbed(
        raised.value,
        secrets=(api_key, namespace, _REGION),
        transport=transport,
    )


def test_region_accepts_bounded_lowercase_dns_labels() -> None:
    assert is_valid_metadata_probe_region("gcp-us-west1") is True
    assert is_valid_metadata_probe_region("a") is True
    assert is_valid_metadata_probe_region("a" * 63) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("POST", _EXPECTED_URL, b'{"rows":[]}'),
        (
            "POST",
            "https://gcp-us-west1.turbopuffer.com/v2/namespaces/fixture/query",
            b"{}",
        ),
        (
            "DELETE",
            "https://gcp-us-west1.turbopuffer.com/v2/namespaces/fixture",
            b"",
        ),
        ("GET", _EXPECTED_URL, b"unexpected-body"),
        (
            "GET",
            "https://gcp-us-west1.turbopuffer.com/v2/namespaces/other/metadata",
            b"",
        ),
    ],
)
async def test_request_sanitizer_rejects_write_query_delete_body_and_other_target_before_transport(
    method: str,
    url: str,
    body: bytes,
) -> None:
    outbound_calls = 0

    async def poison_handler(_: httpx.Request) -> httpx.Response:
        nonlocal outbound_calls
        outbound_calls += 1
        raise AssertionError("poisoned provider operation reached transport")

    sanitizer = metadata_probe_module._metadata_request_sanitizer(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
    )
    async with httpx.AsyncClient(
        event_hooks={"request": [sanitizer]},
        transport=httpx.MockTransport(poison_handler),
    ) as client:
        with pytest.raises(metadata_probe_module._MetadataProbeRequestError) as raised:
            await client.request(method, url, content=body)

    assert outbound_calls == 0
    assert _API_KEY not in str(raised.value)
    assert _NAMESPACE not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_cancellation_during_request_closes_once_and_detaches_sdk_graph() -> None:
    request_started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    transport = RecordingTransport(handler)
    task = asyncio.create_task(
        probe_namespace_metadata(
            api_key=_API_KEY,
            region=_REGION,
            namespace=_NAMESPACE,
            transport=transport,
        )
    )
    await request_started.wait()
    task.cancel("cancel-message-must-not-survive")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert transport.close_calls == 1
    assert str(raised.value) == ""
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    rendered = "".join(traceback.format_exception(raised.value, chain=True))
    assert _API_KEY not in rendered
    assert _NAMESPACE not in rendered
    assert "cancel-message-must-not-survive" not in rendered
    _assert_probe_traceback_scrubbed(
        raised.value,
        secrets=(_API_KEY, _NAMESPACE, "cancel-message-must-not-survive"),
        transport=transport,
    )


@pytest.mark.asyncio
async def test_repeated_cancellation_during_close_is_drained_before_cancellation_returns() -> None:
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_metadata_body())

    transport = RecordingTransport(
        handler,
        close_started=close_started,
        close_release=close_release,
    )
    task = asyncio.create_task(
        probe_namespace_metadata(
            api_key=_API_KEY,
            region=_REGION,
            namespace=_NAMESPACE,
            transport=transport,
        )
    )
    await close_started.wait()
    task.cancel("first-sensitive-cancel")
    await asyncio.sleep(0)
    task.cancel("second-sensitive-cancel")
    await asyncio.sleep(0)

    assert task.done() is False
    assert transport.close_calls == 1
    close_release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert str(raised.value) == ""
    assert transport.close_calls == 1
    _assert_probe_traceback_scrubbed(
        raised.value,
        secrets=(_API_KEY, _NAMESPACE, "first-sensitive-cancel", "second-sensitive-cancel"),
        transport=transport,
    )


@pytest.mark.asyncio
async def test_close_failure_becomes_safe_remote_failure_without_exception_graph() -> None:
    close_secret = "close-exception-secret"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_metadata_body())

    transport = RecordingTransport(handler, close_error=RuntimeError(close_secret))
    result = await probe_namespace_metadata(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )

    assert result.state is MetadataProbeState.REMOTE_FAILURE
    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert close_secret not in str(result)
    assert close_secret not in repr(result)
