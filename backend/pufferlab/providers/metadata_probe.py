"""One-shot, read-only turbopuffer namespace metadata probe.

This module deliberately does not reuse :class:`TurbopufferProvider`. Interactive
search and ingestion benefit from the SDK's normal retry policy; an operator
diagnostic instead needs one auditable request, a fixed destination, and an owned
client that is drained before control returns to its caller.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote

import httpx
from turbopuffer import APIError, AsyncTurbopuffer, NotFoundError

_TIMEOUT_SECONDS = 10.0
_OFFICIAL_BASE_URL_TEMPLATE = "https://{region}.turbopuffer.com"
_REGION_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MAX_NAMESPACE_UTF8_BYTES = 128
_NAMESPACE_PATH_SAFE = "!$&'()*+,;=:@"
_USER_AGENT = "pufferlab-metadata-probe/1"

type _MetadataRequestHook = Callable[[httpx.Request], Awaitable[None]]


class MetadataProbeState(StrEnum):
    """Finite, value-free outcome of one metadata request."""

    INDEX_UP_TO_DATE = "index_up_to_date"
    INDEX_UPDATING = "index_updating"
    NOT_FOUND = "not_found"
    REMOTE_FAILURE = "remote_failure"


@dataclass(frozen=True, slots=True)
class MetadataProbeResult:
    """A safe probe result that contains no target or provider payload."""

    state: MetadataProbeState

    @property
    def metadata_reachable(self) -> bool:
        return self.state in {
            MetadataProbeState.INDEX_UP_TO_DATE,
            MetadataProbeState.INDEX_UPDATING,
        }


class MetadataProbeConfigurationError(ValueError):
    """Raised before client construction when local probe input is unsafe."""

    def __init__(self) -> None:
        super().__init__("turbopuffer metadata probe configuration is invalid")


class _MetadataProbeRequestError(RuntimeError):
    """Raised before transport when the SDK request violates the frozen shape."""

    def __init__(self) -> None:
        super().__init__("turbopuffer metadata probe request was rejected")


def is_valid_metadata_probe_region(region: str) -> bool:
    """Return whether ``region`` is one bounded lowercase DNS label."""

    return isinstance(region, str) and _REGION_PATTERN.fullmatch(region) is not None


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    result: MetadataProbeResult
    configuration_error: bool = False
    cancellation_observed: bool = False


def _metadata_request_sanitizer(
    *,
    api_key: str,
    region: str,
    namespace: str,
) -> _MetadataRequestHook:
    """Build an HTTPX request hook that freezes the metadata request boundary.

    The SDK reads custom headers from the process environment. Replacing the
    complete header collection in a request hook prevents a differently-cased
    ``Authorization`` override from surviving HTTPX's case-insensitive merge.
    The hook also runs before transport, so an SDK method/path/body drift cannot
    turn this read-only capability into another provider operation.
    """

    if not is_valid_metadata_probe_region(region):
        raise MetadataProbeConfigurationError()
    expected_url = _metadata_url(region=region, namespace=namespace)

    async def sanitize(request: httpx.Request) -> None:
        body = await request.aread()
        if request.method != "GET" or request.url != expected_url or body:
            raise _MetadataProbeRequestError()
        request.headers = httpx.Headers(
            {
                "Host": expected_url.netloc.decode("ascii"),
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _USER_AGENT,
            }
        )

    return sanitize


async def probe_namespace_metadata(
    *,
    api_key: str,
    region: str,
    namespace: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> MetadataProbeResult:
    """Issue at most one metadata request and drain the owned client.

    The returned value and every locally-created exception are deliberately
    value-free. Provider bodies and SDK exception graphs are consumed inside
    this boundary. Cancellation remains cancellation, but a fresh exception is
    raised after close drains so the SDK request graph is not retained.
    """

    outcome = await _probe_namespace_metadata_inner(
        api_key=api_key,
        region=region,
        namespace=namespace,
        transport=transport,
    )

    # A safe exception still retains the locals of every traceback frame. Scrub
    # all caller-supplied and transport-bearing references before a separate
    # value-free helper raises configuration/cancellation signals.
    api_key = ""
    region = ""
    namespace = ""
    transport = None

    if outcome.configuration_error:
        _raise_configuration_error()
    if outcome.cancellation_observed:
        _raise_cancelled()
    return outcome.result


async def _probe_namespace_metadata_inner(
    *,
    api_key: str,
    region: str,
    namespace: str,
    transport: httpx.AsyncBaseTransport | None,
) -> _ProbeOutcome:
    if not _valid_local_configuration(api_key=api_key, region=region, namespace=namespace):
        return _ProbeOutcome(
            result=MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE),
            configuration_error=True,
        )

    request_hook = _metadata_request_sanitizer(
        api_key=api_key,
        region=region,
        namespace=namespace,
    )
    try:
        http_client = httpx.AsyncClient(
            event_hooks={"request": [request_hook]},
            follow_redirects=False,
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            transport=transport,
            trust_env=False,
        )
    except Exception:
        return _ProbeOutcome(result=MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE))

    client: AsyncTurbopuffer | None = None
    result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)
    cancellation_observed = False
    try:
        try:
            client = AsyncTurbopuffer(
                api_key=api_key,
                region=region,
                base_url=_OFFICIAL_BASE_URL_TEMPLATE,
                http_client=http_client,
                max_retries=0,
                timeout=_TIMEOUT_SECONDS,
                _strict_response_validation=True,
            )
        except Exception:
            result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)
        else:
            try:
                response = await client.namespace(namespace).metadata(timeout=_TIMEOUT_SECONDS)
                result = _normalize_metadata(response)
            except asyncio.CancelledError:
                cancellation_observed = True
            except NotFoundError:
                result = MetadataProbeResult(state=MetadataProbeState.NOT_FOUND)
            except APIError:
                result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)
            except Exception:
                result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)
    finally:
        close = client.close if client is not None else http_client.aclose
        close_cancelled, close_failed = await _drain_close(close)
        cancellation_observed = cancellation_observed or close_cancelled
        if close_failed:
            result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)

    return _ProbeOutcome(
        result=result,
        cancellation_observed=cancellation_observed,
    )


def _valid_local_configuration(*, api_key: str, region: str, namespace: str) -> bool:
    if not api_key.strip() or not is_valid_metadata_probe_region(region) or not namespace.strip():
        return False
    try:
        namespace_bytes = namespace.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(namespace_bytes) <= _MAX_NAMESPACE_UTF8_BYTES


def _raise_configuration_error() -> None:
    raise MetadataProbeConfigurationError()


def _raise_cancelled() -> None:
    raise asyncio.CancelledError()


def _metadata_url(*, region: str, namespace: str) -> httpx.URL:
    encoded_namespace = quote(namespace, safe=_NAMESPACE_PATH_SAFE)
    return httpx.URL(f"https://{region}.turbopuffer.com/v2/namespaces/{encoded_namespace}/metadata")


def _normalize_metadata(response: object) -> MetadataProbeResult:
    index = getattr(response, "index", None)
    status = getattr(index, "status", None)
    if status == "up-to-date":
        return MetadataProbeResult(state=MetadataProbeState.INDEX_UP_TO_DATE)
    if status == "updating":
        return MetadataProbeResult(state=MetadataProbeState.INDEX_UPDATING)
    return MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)


async def _drain_close(
    close: Callable[[], Awaitable[None]],
) -> tuple[bool, bool]:
    async def close_owned_client() -> None:
        await close()

    close_task: asyncio.Task[None] = asyncio.create_task(
        close_owned_client(),
        name="pufferlab-metadata-probe-close",
    )
    cancellation_observed = False
    close_failed = False
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancellation_observed = True
        except BaseException:
            close_failed = True

    if close_task.cancelled():
        close_failed = True
    else:
        close_failed = close_task.exception() is not None or close_failed
    del close_task
    return cancellation_observed, close_failed
