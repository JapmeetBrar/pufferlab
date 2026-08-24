"""One-shot hardened turbopuffer adapter for expected-document diagnostics."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import NoReturn, Protocol, cast
from urllib.parse import quote

import httpx
from turbopuffer import AsyncTurbopuffer
from turbopuffer.types import NamespaceMultiQueryResponse, Row
from turbopuffer.types.namespace_multi_query_response import Result
from turbopuffer.types.query_billing import QueryBilling
from turbopuffer.types.query_performance import QueryPerformance

from pufferlab.contracts.common import (
    JsonValue,
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.forensics import DiagnosticSubqueryRole
from pufferlab.providers.turbopuffer import _weighted_bm25_rank_by, filter_to_turbopuffer
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticAttributeValue,
    DiagnosticCandidateList,
    DiagnosticCandidateRow,
    DiagnosticProviderRequest,
    DiagnosticProviderResult,
    DiagnosticTargetObservation,
    is_valid_diagnostic_namespace,
    monotonic,
    require_exact_uuid,
    require_finite_nonnegative,
)

_TIMEOUT_SECONDS = 10.0
_OFFICIAL_BASE_URL_TEMPLATE = "https://{region}.turbopuffer.com"
_REGION_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_NAMESPACE_PATH_SAFE = "!$&'()*+,;=:@"
_USER_AGENT = "pufferlab-expected-document-diagnostic/1"
_BM25_COMPUTE_FIELD = "__pufferlab_diagnostic_bm25"
_VECTOR_COMPUTE_FIELD = "__pufferlab_diagnostic_vector_distance"
_SCORE_REL_TOL = 1e-12
_SCORE_ABS_TOL = 1e-15

type _RequestHook = Callable[[httpx.Request], Awaitable[None]]


class _AsyncNamespace(Protocol):
    async def multi_query(self, **kwargs: object) -> object: ...


class _AsyncClient(Protocol):
    def namespace(self, namespace: str) -> _AsyncNamespace: ...

    async def close(self) -> None: ...


class _SdkRow(Protocol):
    model_fields_set: set[str]
    model_extra: dict[str, object] | None

    def model_dump(
        self,
        *,
        by_alias: bool = False,
        exclude_unset: bool = False,
        warnings: bool | str = True,
    ) -> dict[str, object]: ...


class DiagnosticProviderConfigurationError(ValueError):
    def __init__(self) -> None:
        super().__init__("expected-document diagnostic provider configuration is invalid")


class DiagnosticProviderFailure(RuntimeError):
    def __init__(self) -> None:
        super().__init__("expected-document diagnostic provider request failed")


class _DiagnosticRequestRejected(RuntimeError):
    def __init__(self) -> None:
        super().__init__("expected-document diagnostic request was rejected")


class _RequestBodyGuard:
    """Single-use value-only binding between one validated input and its HTTP body."""

    __slots__ = ("_expected",)

    def __init__(self) -> None:
        self._expected: bytes | None = None

    def bind(self, expected: dict[str, JsonValue]) -> None:
        if self._expected is not None:
            raise _DiagnosticRequestRejected()
        self._expected = _canonical_body_digest(expected)
        expected = {}

    def take(self) -> bytes:
        expected = self._expected
        self._expected = None
        if expected is None:
            raise _DiagnosticRequestRejected()
        return expected

    def clear(self) -> None:
        self._expected = None


@dataclass(frozen=True, slots=True)
class _CreateOutcome:
    provider: TurbopufferDiagnosticProvider | None = None
    configuration_error: bool = False
    failure: bool = False
    cancelled: bool = False
    keyboard_interrupt: bool = False
    system_exit: bool = False


@dataclass(frozen=True, slots=True)
class _QueryOutcome:
    result: DiagnosticProviderResult | None = None
    failure: bool = False
    cancelled: bool = False
    keyboard_interrupt: bool = False
    system_exit: bool = False


def _diagnostic_url(*, region: str, namespace: str) -> httpx.URL:
    encoded_namespace = quote(namespace, safe=_NAMESPACE_PATH_SAFE)
    url = httpx.URL(
        f"https://{region}.turbopuffer.com/v2/namespaces/{encoded_namespace}/query"
        "?stainless_overload=multiQuery"
    )
    expected_raw_path = (
        f"/v2/namespaces/{encoded_namespace}/query?stainless_overload=multiQuery".encode("ascii")
    )
    if url.raw_path != expected_raw_path:
        raise DiagnosticProviderConfigurationError()
    return url


def _request_sanitizer(
    *,
    api_key: str,
    region: str,
    namespace: str,
    body_guard: _RequestBodyGuard,
) -> _RequestHook:
    expected_url = _diagnostic_url(region=region, namespace=namespace)

    async def sanitize(request: httpx.Request) -> None:
        expected_digest = body_guard.take()
        body = await request.aread()
        if request.method != "POST" or request.url != expected_url or not body:
            raise _DiagnosticRequestRejected()
        try:
            payload = json.loads(
                body,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise _DiagnosticRequestRejected() from error
        if type(payload) is not dict or _canonical_body_digest(payload) != expected_digest:
            raise _DiagnosticRequestRejected()
        expected_digest = b""
        request.headers = httpx.Headers(
            {
                "Host": expected_url.netloc.decode("ascii"),
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            }
        )

    return sanitize


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("diagnostic request JSON is invalid")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise ValueError("diagnostic request JSON is invalid")


def _canonical_body_digest(value: object) -> bytes:
    checked = _strict_request_json_value(value)
    if type(checked) is not dict:
        raise _DiagnosticRequestRejected()
    encoded = json.dumps(
        checked,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(encoded).digest()
    encoded = b""
    checked = {}
    return digest


def _strict_request_json_value(value: object, *, depth: int = 0) -> JsonValue:
    if depth > 32:
        raise _DiagnosticRequestRejected()
    if value is None or type(value) in {str, bool}:
        return cast(JsonValue, value)
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _DiagnosticRequestRejected()
        return value
    if type(value) is dict:
        if len(value) > 256 or not all(type(key) is str for key in value):
            raise _DiagnosticRequestRejected()
        return {
            cast(str, key): _strict_request_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if type(value) is list:
        if len(value) > 10_000:
            raise _DiagnosticRequestRejected()
        return [_strict_request_json_value(item, depth=depth + 1) for item in value]
    raise _DiagnosticRequestRejected()


class TurbopufferDiagnosticProvider:
    """Request-bound provider that owns one strict SDK and HTTP client."""

    def __init__(
        self,
        *,
        client: _AsyncClient,
        namespace: str,
        clock: Callable[[], float] = perf_counter,
        body_guard: _RequestBodyGuard | None = None,
    ) -> None:
        self._client: _AsyncClient | None = client
        self._namespace_name = namespace
        self._namespace: _AsyncNamespace | None = client.namespace(namespace)
        self._clock = clock
        self._body_guard = body_guard
        self._closed = False
        self._called = False

    @classmethod
    async def create(
        cls,
        *,
        api_key: str,
        region: str,
        namespace: str,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> TurbopufferDiagnosticProvider:
        outcome = await _create_inner(
            api_key=api_key,
            region=region,
            namespace=namespace,
            transport=transport,
            clock=clock,
        )
        api_key = ""
        region = ""
        namespace = ""
        transport = None
        clock = perf_counter
        if outcome.configuration_error:
            _raise_configuration_error()
        if outcome.cancelled:
            _raise_cancelled()
        if outcome.keyboard_interrupt:
            _raise_keyboard_interrupt()
        if outcome.system_exit:
            _raise_system_exit()
        if outcome.failure or outcome.provider is None:
            _raise_provider_failure()
        return outcome.provider

    async def query(self, request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        outcome = await _query_inner(self, request)
        request = cast(DiagnosticProviderRequest, None)
        self = cast(TurbopufferDiagnosticProvider, None)
        if outcome.cancelled:
            _raise_cancelled()
        if outcome.keyboard_interrupt:
            _raise_keyboard_interrupt()
        if outcome.system_exit:
            _raise_system_exit()
        if outcome.failure or outcome.result is None:
            _raise_provider_failure()
        return outcome.result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        self._namespace = None
        self._namespace_name = ""
        body_guard = self._body_guard
        self._body_guard = None
        if body_guard is not None:
            body_guard.clear()
        if client is None:
            return
        error, cancelled = await _drain_close(client.close)
        client = None
        if error is None and not cancelled:
            return
        keyboard = False
        system = False
        if error is not None:
            error_cancelled, keyboard, system = _classify_control(error)
            cancelled = cancelled or error_cancelled
        error = None
        if cancelled:
            _raise_cancelled()
        if keyboard:
            _raise_keyboard_interrupt()
        if system:
            _raise_system_exit()
        _raise_provider_failure()

    def _request_kwargs(self, request: DiagnosticProviderRequest) -> dict[str, object]:
        queries: list[dict[str, object]] = [self._target_query(request)]
        for role in request.roles[1:]:
            queries.append(self._candidate_query(request, role))
        return {
            "queries": queries,
            "consistency": {"level": "strong"},
            "timeout": _TIMEOUT_SECONDS,
        }

    @staticmethod
    def _expected_request_body(request: DiagnosticProviderRequest) -> dict[str, JsonValue]:
        queries: list[JsonValue] = [_expected_target_query(request)]
        queries.extend(_expected_candidate_query(request, role) for role in request.roles[1:])
        return {"queries": queries, "consistency": {"level": "strong"}}

    @staticmethod
    def _target_query(request: DiagnosticProviderRequest) -> dict[str, object]:
        computed: dict[str, object] = {}
        if request.lexical_fields is not None:
            computed[_BM25_COMPUTE_FIELD] = _weighted_bm25_rank_by(
                request.lexical_fields,
                request.query_text,
            )
        if request.query_vector is not None:
            assert request.vector_attribute is not None
            computed[_VECTOR_COMPUTE_FIELD] = (
                request.vector_attribute,
                "VectorDist",
                list(request.query_vector),
            )
        query: dict[str, object] = {
            "rank_by": ("id", "asc"),
            "filters": ("id", "Eq", str(request.target_document_id)),
            "limit": 1,
            "include_attributes": list(request.filter_fields),
            "compute_attributes": computed,
        }
        if request.query_vector is not None:
            query["distance_metric"] = request.distance_metric
        return query

    @staticmethod
    def _candidate_query(
        request: DiagnosticProviderRequest,
        role: DiagnosticSubqueryRole,
    ) -> dict[str, object]:
        no_filter = role in {
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
        }
        bm25 = role in {
            DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
            DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        }
        query: dict[str, object] = {
            "limit": request.candidate_limit,
            "include_attributes": [],
        }
        if bm25:
            assert request.lexical_fields is not None
            query["rank_by"] = _weighted_bm25_rank_by(
                request.lexical_fields,
                request.query_text,
            )
        else:
            assert request.vector_attribute is not None
            assert request.query_vector is not None
            query["rank_by"] = (
                request.vector_attribute,
                "ANN",
                list(request.query_vector),
            )
            query["distance_metric"] = request.distance_metric
        if not no_filter and request.stored_filter is not None:
            query["filters"] = filter_to_turbopuffer(request.stored_filter)
        return query


def _expected_target_query(request: DiagnosticProviderRequest) -> dict[str, JsonValue]:
    computed: dict[str, JsonValue] = {}
    if request.lexical_fields is not None:
        computed[_BM25_COMPUTE_FIELD] = _wire_json_value(
            _weighted_bm25_rank_by(request.lexical_fields, request.query_text)
        )
    if request.query_vector is not None:
        assert request.vector_attribute is not None
        computed[_VECTOR_COMPUTE_FIELD] = [
            request.vector_attribute,
            "VectorDist",
            list(request.query_vector),
        ]
    query: dict[str, JsonValue] = {
        "rank_by": ["id", "asc"],
        "filters": ["id", "Eq", str(request.target_document_id)],
        "limit": 1,
        "include_attributes": list(request.filter_fields),
        "compute_attributes": computed,
    }
    if request.query_vector is not None:
        assert request.distance_metric is not None
        query["distance_metric"] = request.distance_metric
    return query


def _expected_candidate_query(
    request: DiagnosticProviderRequest,
    role: DiagnosticSubqueryRole,
) -> dict[str, JsonValue]:
    no_filter = role in {
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_ANN_CANDIDATES,
    }
    bm25 = role in {
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
    }
    query: dict[str, JsonValue] = {
        "limit": request.candidate_limit,
        "include_attributes": [],
    }
    if bm25:
        assert request.lexical_fields is not None
        query["rank_by"] = _wire_json_value(
            _weighted_bm25_rank_by(request.lexical_fields, request.query_text)
        )
    else:
        assert request.vector_attribute is not None
        assert request.query_vector is not None
        assert request.distance_metric is not None
        query["rank_by"] = [
            request.vector_attribute,
            "ANN",
            list(request.query_vector),
        ]
        query["distance_metric"] = request.distance_metric
    if not no_filter and request.stored_filter is not None:
        query["filters"] = _wire_json_value(filter_to_turbopuffer(request.stored_filter))
    return query


def _wire_json_value(value: object) -> JsonValue:
    if value is None or type(value) in {str, bool, int}:
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("diagnostic request contains a nonfinite value")
        return value
    if type(value) in {tuple, list}:
        items = cast(tuple[object, ...] | list[object], value)
        return [_wire_json_value(item) for item in items]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("diagnostic request contains an invalid object")
        return {cast(str, key): _wire_json_value(item) for key, item in value.items()}
    raise ValueError("diagnostic request is not JSON-compatible")


async def _create_inner(
    *,
    api_key: str,
    region: str,
    namespace: str,
    transport: httpx.AsyncBaseTransport | None,
    clock: Callable[[], float],
) -> _CreateOutcome:
    if (
        not _valid_api_key(api_key)
        or type(region) is not str
        or _REGION_PATTERN.fullmatch(region) is None
        or not is_valid_diagnostic_namespace(namespace)
    ):
        return _CreateOutcome(configuration_error=True)
    request_hook: _RequestHook | None = None
    body_guard: _RequestBodyGuard | None = None
    http_client: httpx.AsyncClient | None = None
    sdk: _AsyncClient | None = None
    caught_error: BaseException | None = None
    try:
        body_guard = _RequestBodyGuard()
        request_hook = _request_sanitizer(
            api_key=api_key,
            region=region,
            namespace=namespace,
            body_guard=body_guard,
        )
        http_client = httpx.AsyncClient(
            event_hooks={"request": [request_hook]},
            follow_redirects=False,
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            limits=httpx.Limits(
                max_connections=1,
                max_keepalive_connections=1,
                keepalive_expiry=5.0,
            ),
            transport=transport,
            trust_env=False,
        )
        sdk = cast(
            _AsyncClient,
            AsyncTurbopuffer(
                api_key=api_key,
                region=region,
                base_url=_OFFICIAL_BASE_URL_TEMPLATE,
                http_client=http_client,
                max_retries=0,
                timeout=_TIMEOUT_SECONDS,
                _strict_response_validation=True,
            ),
        )
        provider = TurbopufferDiagnosticProvider(
            client=sdk,
            namespace=namespace,
            clock=clock,
            body_guard=body_guard,
        )
        return _CreateOutcome(provider=provider)
    except BaseException as error:
        caught_error = error
        _detach_exception(error)

    close_error: BaseException | None = None
    close_cancelled = False
    if sdk is not None:
        close_error, close_cancelled = await _drain_close(sdk.close)
    elif http_client is not None:
        close_error, close_cancelled = await _drain_close(http_client.aclose)
    api_key = ""
    region = ""
    namespace = ""
    request_hook = None
    if body_guard is not None:
        body_guard.clear()
    body_guard = None
    http_client = None
    sdk = None
    assert caught_error is not None
    cancelled, keyboard, system = _classify_control(caught_error)
    caught_error = None
    if not (cancelled or keyboard or system):
        if close_cancelled:
            cancelled = True
        elif close_error is not None:
            cancelled, keyboard, system = _classify_control(close_error)
    close_error = None
    return _CreateOutcome(
        failure=not (cancelled or keyboard or system),
        cancelled=cancelled,
        keyboard_interrupt=keyboard,
        system_exit=system,
    )


async def _query_inner(
    provider: TurbopufferDiagnosticProvider,
    request: DiagnosticProviderRequest,
) -> _QueryOutcome:
    body_guard = provider._body_guard
    try:
        request.__post_init__()
        if provider._closed or provider._called or provider._namespace is None:
            raise ValueError("diagnostic provider is unavailable")
        if request.namespace != provider._namespace_name:
            raise ValueError("diagnostic namespace does not match the bound provider")
        provider._called = True
        if body_guard is not None:
            body_guard.bind(provider._expected_request_body(request))
        started = provider._clock()
        response = await provider._namespace.multi_query(**provider._request_kwargs(request))
        if body_guard is not None:
            body_guard.clear()
        duration_ms = (provider._clock() - started) * 1000.0
        result = _decode_response(
            response,
            request=request,
            client_duration_ms=duration_ms,
        )
        response = None
        return _QueryOutcome(result=result)
    except BaseException as error:
        if body_guard is not None:
            body_guard.clear()
        cancelled, keyboard, system = _classify_control(error)
        _detach_exception(error)
        return _QueryOutcome(
            failure=not (cancelled or keyboard or system),
            cancelled=cancelled,
            keyboard_interrupt=keyboard,
            system_exit=system,
        )


def _decode_response(
    response: object,
    *,
    request: DiagnosticProviderRequest,
    client_duration_ms: float,
) -> DiagnosticProviderResult:
    if type(response) is not NamespaceMultiQueryResponse:
        raise ValueError("diagnostic multi-query response has invalid results")
    _require_sdk_model_metadata(
        response,
        required_fields={"billing", "performance", "results"},
    )
    if (
        type(response.billing) is not QueryBilling
        or type(response.performance) is not QueryPerformance
    ):
        raise ValueError("diagnostic multi-query response has invalid metadata")
    _require_sdk_model_metadata(
        response.billing,
        required_fields=set(QueryBilling.model_fields),
    )
    _require_sdk_model_metadata(
        response.performance,
        required_fields=set(QueryPerformance.model_fields),
    )
    results = response.results
    if type(results) is not list:
        raise ValueError("diagnostic multi-query response has invalid results")
    if len(results) != len(request.roles):
        raise ValueError("diagnostic multi-query response count is invalid")
    row_sets = tuple(
        _result_rows(
            result,
            max_rows=1 if ordinal == 0 else request.candidate_limit,
        )
        for ordinal, result in enumerate(results)
    )
    target = _decode_target(row_sets[0], request)
    candidates = tuple(
        _decode_candidates(rows, request=request, role=role)
        for role, rows in zip(request.roles[1:], row_sets[1:], strict=True)
    )
    target_ids = {
        row.document_id
        for candidate in candidates
        for row in candidate.rows
        if row.document_id == request.target_document_id
    }
    if not target.available and target_ids:
        raise ValueError("diagnostic candidate contains a target unavailable to exact lookup")
    if target.available:
        for candidate in candidates:
            matching = next(
                (row for row in candidate.rows if row.document_id == request.target_document_id),
                None,
            )
            if matching is None:
                continue
            direct = (
                target.bm25_score
                if matching.score.kind is ScoreKind.BM25
                else target.vector_distance
            )
            if direct is None or not math.isclose(
                matching.score.value,
                direct.value,
                rel_tol=_SCORE_REL_TOL,
                abs_tol=_SCORE_ABS_TOL,
            ):
                raise ValueError("diagnostic target candidate score contradicts exact lookup")
    return DiagnosticProviderResult(
        namespace=request.namespace,
        target=target,
        candidate_lists=candidates,
        client_duration_ms=client_duration_ms,
    )


def _result_rows(result: object, *, max_rows: int) -> tuple[object, ...]:
    if type(result) is not Result:
        raise ValueError("diagnostic multi-query result has invalid rows")
    _require_sdk_model_metadata(result, required_fields={"rows"})
    if result.aggregations is not None or result.aggregation_groups is not None:
        raise ValueError("diagnostic multi-query result is not rows-only")
    rows = result.rows
    if rows is None:
        return ()
    if type(rows) is not list or len(rows) > max_rows:
        raise ValueError("diagnostic multi-query result has invalid rows")
    return tuple(rows)


def _decode_target(
    rows: tuple[object, ...],
    request: DiagnosticProviderRequest,
) -> DiagnosticTargetObservation:
    if len(rows) > 1:
        raise ValueError("diagnostic target lookup returned too many rows")
    if not rows:
        return DiagnosticTargetObservation(
            target_document_id=request.target_document_id,
            available=False,
            bm25_score=None,
            vector_distance=None,
            attributes=(),
        )
    view, fields_set, extra = _sdk_row_view(rows[0])
    if require_exact_uuid(view.get("id")) != request.target_document_id:
        raise ValueError("diagnostic target lookup returned the wrong ID")
    if "$dist" in view:
        raise ValueError("attribute-ranked diagnostic lookup cannot contain $dist")
    allowed = {"id", *request.filter_fields}
    bm25_score: ObservedScore | None = None
    vector_distance: ObservedScore | None = None
    if request.lexical_fields is not None:
        allowed.add(_BM25_COMPUTE_FIELD)
        bm25_score = ObservedScore(
            kind=ScoreKind.BM25,
            value=require_finite_nonnegative(view.get(_BM25_COMPUTE_FIELD)),
            direction=ScoreDirection.HIGHER_IS_BETTER,
            source=ScoreSource.COMPUTE_ATTRIBUTE,
        )
    if request.query_vector is not None:
        allowed.add(_VECTOR_COMPUTE_FIELD)
        vector_distance = ObservedScore(
            kind=ScoreKind.VECTOR_DISTANCE,
            value=require_finite_nonnegative(view.get(_VECTOR_COMPUTE_FIELD)),
            direction=ScoreDirection.LOWER_IS_BETTER,
            source=ScoreSource.COMPUTE_ATTRIBUTE,
        )
    if set(view) - allowed:
        raise ValueError("diagnostic target lookup returned unrequested fields")
    attributes = tuple(
        _decode_attribute(field, view=view, fields_set=fields_set, extra=extra)
        for field in request.filter_fields
    )
    return DiagnosticTargetObservation(
        target_document_id=request.target_document_id,
        available=True,
        bm25_score=bm25_score,
        vector_distance=vector_distance,
        attributes=attributes,
    )


def _decode_attribute(
    field: str,
    *,
    view: Mapping[str, object],
    fields_set: set[str],
    extra: Mapping[str, object],
) -> DiagnosticAttributeValue:
    present = field in fields_set or field in extra
    if not present:
        if field in view:
            raise ValueError("diagnostic row presence views disagree")
        return DiagnosticAttributeValue(field=field, state=DiagnosticAttributeState.MISSING)
    if field not in view:
        raise ValueError("diagnostic row presence views disagree")
    value = _strict_json_value(view[field])
    if value is None:
        return DiagnosticAttributeValue(field=field, state=DiagnosticAttributeState.PRESENT_NULL)
    return DiagnosticAttributeValue(
        field=field,
        state=DiagnosticAttributeState.PRESENT_VALUE,
        value=value,
    )


def _decode_candidates(
    rows: tuple[object, ...],
    *,
    request: DiagnosticProviderRequest,
    role: DiagnosticSubqueryRole,
) -> DiagnosticCandidateList:
    if len(rows) > request.candidate_limit:
        raise ValueError("diagnostic candidate rows exceed the request bound")
    bm25 = role in {
        DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
    }
    kind = ScoreKind.BM25 if bm25 else ScoreKind.VECTOR_DISTANCE
    direction = ScoreDirection.HIGHER_IS_BETTER if bm25 else ScoreDirection.LOWER_IS_BETTER
    decoded: list[DiagnosticCandidateRow] = []
    scores: list[float] = []
    for rank, raw in enumerate(rows, start=1):
        view, _, _ = _sdk_row_view(raw)
        if set(view) != {"id", "$dist"}:
            raise ValueError("diagnostic candidate row contains unexpected fields")
        score_value = require_finite_nonnegative(view["$dist"], positive=bm25)
        scores.append(score_value)
        decoded.append(
            DiagnosticCandidateRow(
                document_id=require_exact_uuid(view["id"]),
                rank=rank,
                score=ObservedScore(
                    kind=kind,
                    value=score_value,
                    direction=direction,
                    source=ScoreSource.TURBOPUFFER_DIST,
                ),
            )
        )
    if not monotonic(scores, descending=bm25):
        raise ValueError("diagnostic candidate scores are not monotonic")
    return DiagnosticCandidateList(
        role=role,
        requested_limit=request.candidate_limit,
        rows=tuple(decoded),
    )


def _sdk_row_view(row: object) -> tuple[dict[str, object], set[str], dict[str, object]]:
    if type(row) is not Row:
        raise ValueError("diagnostic response row is not SDK-compatible")
    sdk_row = cast(_SdkRow, row)
    fields_set = getattr(sdk_row, "model_fields_set", None)
    extra = getattr(sdk_row, "model_extra", None)
    if (
        type(fields_set) is not set
        or not all(type(field) is str for field in fields_set)
        or (extra is not None and type(extra) is not dict)
    ):
        raise ValueError("diagnostic SDK row presence metadata is invalid")
    checked_extra = {} if extra is None else _strict_json_object(extra)
    view = sdk_row.model_dump(by_alias=True, exclude_unset=True, warnings=False)
    if type(view) is not dict:
        raise ValueError("diagnostic SDK row serialization is invalid")
    checked_view = _strict_json_object(view)
    if fields_set != set(checked_view) or set(checked_extra) != set(checked_view) - {
        "id",
        "vector",
    }:
        raise ValueError("diagnostic SDK row presence views disagree")
    return checked_view, set(fields_set), checked_extra


def _require_sdk_model_metadata(model: object, *, required_fields: set[str]) -> None:
    fields_set = getattr(model, "model_fields_set", None)
    extra = getattr(model, "model_extra", None)
    if (
        type(fields_set) is not set
        or not all(type(field) is str for field in fields_set)
        or fields_set != required_fields
        or (extra is not None and type(extra) is not dict)
        or (type(extra) is dict and len(extra) != 0)
    ):
        raise ValueError("diagnostic SDK response metadata is invalid")


def _strict_json_object(value: object) -> dict[str, object]:
    checked = _strict_json_value(value)
    if type(checked) is not dict:
        raise ValueError("diagnostic attribute object is invalid")
    return cast(dict[str, object], checked)


def _strict_json_value(value: object, *, depth: int = 0) -> JsonValue:
    if depth > 8:
        raise ValueError("diagnostic attribute value is too deeply nested")
    if value is None or type(value) in {str, bool}:
        return cast(JsonValue, value)
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("diagnostic attribute contains a nonfinite value")
        return value
    if type(value) is dict:
        if len(value) > 256 or not all(type(key) is str for key in value):
            raise ValueError("diagnostic attribute object is invalid")
        return {
            cast(str, key): _strict_json_value(item, depth=depth + 1) for key, item in value.items()
        }
    if type(value) is list:
        if len(value) > 10_000:
            raise ValueError("diagnostic attribute array is too large")
        return [_strict_json_value(item, depth=depth + 1) for item in value]
    raise ValueError("diagnostic attribute value is not JSON-compatible")


def _valid_api_key(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 4096
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


async def _drain_close(
    close: Callable[[], Awaitable[None]],
) -> tuple[BaseException | None, bool]:
    operation: Awaitable[None] | None = None
    captured_operation: Awaitable[BaseException | None] | None = None
    start_error: BaseException | None = None
    try:
        operation = close()
        captured_operation = _capture_close(operation)
        task = asyncio.create_task(captured_operation)
    except BaseException as caught_start:
        _detach_exception(caught_start)
        start_error = caught_start
        if captured_operation is None:
            if operation is not None:
                _dispose_unstarted_awaitable(operation)
            return start_error, False
        try:
            task = asyncio.ensure_future(captured_operation)
        except BaseException as caught_fallback:
            _detach_exception(caught_fallback)
            _dispose_unstarted_awaitable(captured_operation)
            fallback_error = await _capture_close(cast(Awaitable[None], operation))
            if fallback_error is not None and any(_classify_control(fallback_error)):
                return fallback_error, False
            return start_error, False
    cancelled = False
    error: BaseException | None = None
    captured_ready = False
    while True:
        try:
            error = await asyncio.shield(task)
            captured_ready = True
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                break
    if task.done() and not captured_ready:
        error = task.result()
    task = cast(asyncio.Task[BaseException | None], None)
    close = cast(Callable[[], Awaitable[None]], None)
    if start_error is not None and error is None and not cancelled:
        error = start_error
    return error, cancelled


async def _capture_close(operation: Awaitable[None]) -> BaseException | None:
    try:
        await operation
    except BaseException as error:
        _detach_exception(error)
        return error
    return None


def _dispose_unstarted_awaitable(value: object) -> None:
    if inspect.iscoroutine(value):
        value.close()
    elif isinstance(value, asyncio.Future) and not value.done():
        value.cancel()


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None


def _classify_control(error: BaseException) -> tuple[bool, bool, bool]:
    return (
        isinstance(error, asyncio.CancelledError),
        isinstance(error, KeyboardInterrupt),
        isinstance(error, SystemExit),
    )


def _raise_configuration_error() -> NoReturn:
    raise DiagnosticProviderConfigurationError()


def _raise_provider_failure() -> NoReturn:
    raise DiagnosticProviderFailure()


def _raise_cancelled() -> NoReturn:
    raise asyncio.CancelledError()


def _raise_keyboard_interrupt() -> NoReturn:
    raise KeyboardInterrupt()


def _raise_system_exit() -> NoReturn:
    raise SystemExit(1)
