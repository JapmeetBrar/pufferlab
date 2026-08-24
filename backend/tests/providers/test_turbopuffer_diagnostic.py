from __future__ import annotations

import asyncio
import copy
import json
import traceback
from collections.abc import Awaitable, Callable
from typing import cast
from uuid import UUID

import httpx
import pufferlab.providers.turbopuffer_diagnostic as diagnostic_module
import pytest
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.contracts.forensics import DiagnosticSubqueryRole
from pufferlab.contracts.retrieval import RetrievalMode
from pufferlab.providers.turbopuffer_diagnostic import (
    DiagnosticProviderConfigurationError,
    DiagnosticProviderFailure,
    TurbopufferDiagnosticProvider,
)
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticProviderRequest,
)
from turbopuffer.types import NamespaceMultiQueryResponse, Row
from turbopuffer.types.namespace_multi_query_response import Result

_API_KEY = "test-only-diagnostic-credential"
_REGION = "gcp-us-west1"
_NAMESPACE = "m5-diagnostic_fixture-1"
_TARGET = UUID("00000000-0000-0000-0000-000000000001")
_OTHER = UUID("00000000-0000-0000-0000-000000000002")
_URL = (
    "https://gcp-us-west1.turbopuffer.com/v2/namespaces/m5-diagnostic_fixture-1/query"
    "?stainless_overload=multiQuery"
)

type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.close_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self.handler(request)

    async def aclose(self) -> None:
        self.close_calls += 1


def _billing() -> dict[str, int]:
    return {
        "billable_logical_bytes_queried": 1,
        "billable_logical_bytes_returned": 1,
    }


def _performance() -> dict[str, object]:
    return {
        "client_total_ms": 1.0,
        "client_compress_ms": 0.0,
        "client_response_ms": 1.0,
        "client_body_read_ms": 0.0,
        "client_deserialize_ms": 0.0,
        "approx_namespace_size": 1,
        "cache_hit_ratio": 0.0,
        "cache_temperature": "hot",
        "exhaustive_search_count": 0,
        "query_execution_ms": 1,
        "server_total_ms": 1,
    }


def _response(results: list[dict[str, object]]) -> dict[str, object]:
    return {"billing": _billing(), "performance": _performance(), "results": results}


def _request(
    mode: RetrievalMode,
    *,
    include_no_filter: bool = False,
    stored_filter: FilterPredicate | FilterLogical | None = None,
) -> DiagnosticProviderRequest:
    lexical = mode in {
        RetrievalMode.BM25,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    vector = mode in {
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    return DiagnosticProviderRequest(
        namespace=_NAMESPACE,
        query_text="expected shell document",
        target_document_id=_TARGET,
        mode=mode,
        lexical_fields=(("title", 2.0), ("body", 1.0)) if lexical else None,
        vector_attribute="vector" if vector else None,
        query_vector=(0.25, -0.5) if vector else None,
        distance_metric="cosine_distance" if vector else None,
        stored_filter=stored_filter,
        include_no_filter_counterfactual=include_no_filter,
    )


def _target_row(request: DiagnosticProviderRequest, **attributes: object) -> dict[str, object]:
    row: dict[str, object] = {"id": str(_TARGET), **attributes}
    if request.lexical_fields is not None:
        row["__pufferlab_diagnostic_bm25"] = 4.0
    if request.query_vector is not None:
        row["__pufferlab_diagnostic_vector_distance"] = 0.25
    return row


def _candidate_rows(role: DiagnosticSubqueryRole) -> list[dict[str, object]]:
    if "bm25" in role.value:
        return [
            {"id": str(_TARGET), "$dist": 4.0},
            {"id": str(_OTHER), "$dist": 2.0},
        ]
    return [
        {"id": str(_TARGET), "$dist": 0.25},
        {"id": str(_OTHER), "$dist": 0.75},
    ]


def _successful_body(request: DiagnosticProviderRequest) -> dict[str, object]:
    attributes = {field: "unix" for field in request.filter_fields}
    return _response(
        [
            {"rows": [_target_row(request, **attributes)]},
            *({"rows": _candidate_rows(role)} for role in request.roles[1:]),
        ]
    )


def _balanced_max_filter() -> FilterLogical:
    level: list[FilterPredicate | FilterLogical] = [
        FilterPredicate(field=f"field_{ordinal}", op=PredicateOp.EQ, value=ordinal)
        for ordinal in range(16)
    ]
    while len(level) > 1:
        level = [
            FilterLogical(op=LogicalOp.AND, children=level[index : index + 2])
            for index in range(0, len(level), 2)
        ]
    result = level[0]
    assert isinstance(result, FilterLogical)
    return result


def _depth_eight_filter() -> FilterLogical:
    node: FilterPredicate | FilterLogical = FilterPredicate(
        field="category",
        op=PredicateOp.EQ,
        value="unix",
    )
    for _ in range(7):
        node = FilterLogical(op=LogicalOp.NOT, children=[node])
    assert isinstance(node, FilterLogical)
    return node


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("include_no_filter", [False, True])
async def test_one_strong_sdk_multi_query_uses_exact_mode_shapes_and_hardened_transport(
    monkeypatch: pytest.MonkeyPatch,
    mode: RetrievalMode,
    include_no_filter: bool,
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "wrong-env-credential")
    monkeypatch.setenv("TURBOPUFFER_REGION", "wrong-env-region")
    monkeypatch.setenv("TURBOPUFFER_BASE_URL", "https://redirect.invalid/{region}")
    monkeypatch.setenv("TURBOPUFFER_CUSTOM_HEADERS", "X-Leak: forbidden")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    stored_filter = FilterPredicate(field="category", op=PredicateOp.EQ, value="unix")
    request = _request(
        mode,
        include_no_filter=include_no_filter,
        stored_filter=stored_filter,
    )

    async def handler(outbound: httpx.Request) -> httpx.Response:
        assert outbound.method == "POST"
        assert str(outbound.url) == _URL
        assert outbound.url.raw_path == (
            b"/v2/namespaces/m5-diagnostic_fixture-1/query?stainless_overload=multiQuery"
        )
        assert set(outbound.headers) == {
            "host",
            "accept",
            "authorization",
            "user-agent",
            "content-type",
            "content-length",
        }
        assert outbound.headers.get_list("authorization") == [f"Bearer {_API_KEY}"]
        assert outbound.headers["content-type"] == "application/json"
        body_bytes = await outbound.aread()
        assert outbound.headers["content-length"] == str(len(body_bytes))
        body = json.loads(body_bytes)
        assert set(body) == {"queries", "consistency"}
        assert body["consistency"] == {"level": "strong"}
        queries = body["queries"]
        assert isinstance(queries, list)
        assert len(queries) == len(request.roles)
        lookup = queries[0]
        assert lookup["rank_by"] == ["id", "asc"]
        assert lookup["filters"] == ["id", "Eq", str(_TARGET)]
        assert lookup["limit"] == 1
        assert lookup["include_attributes"] == ["category"]
        assert "rerank_by" not in body
        for role, query in zip(request.roles[1:], queries[1:], strict=True):
            assert query["limit"] == request.candidate_limit
            assert query["include_attributes"] == []
            no_filter = role.value.startswith("no_filter_")
            assert ("filters" not in query) is no_filter
            if not no_filter:
                assert query["filters"] == ["category", "Eq", "unix"]
            if "bm25" in role.value:
                assert "BM25" in repr(query["rank_by"])
            else:
                assert query["rank_by"] == ["vector", "ANN", [0.25, -0.5]]
                assert query["distance_metric"] == "cosine_distance"
        return httpx.Response(200, json=_successful_body(request))

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    result = await provider.query(request)
    await provider.close()
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert tuple(item.role for item in result.candidate_lists) == request.roles[1:]
    assert all(item.requested_limit == request.candidate_limit for item in result.candidate_lists)
    assert [item.state for item in result.target.attributes] == [
        DiagnosticAttributeState.PRESENT_VALUE
    ]
    rendered = repr(result)
    assert _API_KEY not in rendered
    assert "unix" not in rendered
    assert str(_OTHER) not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "eventual_consistency",
        "root_rerank",
        "changed_limit",
        "changed_filter",
        "extra_query_field",
        "removed_query",
        "reordered_queries",
    ],
)
async def test_request_bound_body_guard_rejects_sdk_payload_drift_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request = _request(
        RetrievalMode.HYBRID_RRF,
        include_no_filter=True,
        stored_filter=FilterPredicate(field="category", op=PredicateOp.EQ, value="unix"),
    )
    transport_calls = 0

    async def poison_handler(_: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("request-body-drift-reached-transport-marker")

    transport = RecordingTransport(poison_handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    request_kwargs = copy.deepcopy(provider._request_kwargs(request))
    queries = cast(list[dict[str, object]], request_kwargs["queries"])
    if mutation == "eventual_consistency":
        request_kwargs["consistency"] = {"level": "eventual"}
    elif mutation == "root_rerank":
        request_kwargs["rerank_by"] = ("RRF", {})
    elif mutation == "changed_limit":
        queries[1]["limit"] = 1
    elif mutation == "changed_filter":
        queries[1]["filters"] = ("category", "Eq", "other")
    elif mutation == "extra_query_field":
        queries[1]["secret_marker"] = True
    elif mutation == "removed_query":
        queries.pop()
    elif mutation == "reordered_queries":
        queries[1], queries[2] = queries[2], queries[1]

    monkeypatch.setattr(
        TurbopufferDiagnosticProvider,
        "_request_kwargs",
        lambda _self, _request: request_kwargs,
    )
    with pytest.raises(DiagnosticProviderFailure) as raised:
        await provider.query(request)
    await provider.close()

    assert transport_calls == 0
    assert transport.requests == []
    assert transport.close_calls == 1
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_filter",
    [
        _balanced_max_filter(),
        _depth_eight_filter(),
        FilterPredicate(field="tag", op=PredicateOp.IN, value=list(range(10_000))),
    ],
)
async def test_request_body_guard_accepts_frozen_maximum_filter_shapes(
    stored_filter: FilterPredicate | FilterLogical,
) -> None:
    request = _request(RetrievalMode.BM25, stored_filter=stored_filter)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_successful_body(request))

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    await provider.query(request)
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribute_fragment", "expected_state"),
    [
        ({}, DiagnosticAttributeState.MISSING),
        ({"category": None}, DiagnosticAttributeState.PRESENT_NULL),
    ],
)
async def test_pinned_sdk_preserves_omitted_versus_explicit_null_attribute_presence(
    monkeypatch: pytest.MonkeyPatch,
    attribute_fragment: dict[str, object],
    expected_state: DiagnosticAttributeState,
) -> None:
    original_dump = Row.model_dump
    dump_calls = 0

    def checked_dump(self: Row, *args: object, **kwargs: object) -> dict[str, object]:
        nonlocal dump_calls
        dump_calls += 1
        assert kwargs.get("exclude_unset") is True
        return cast(dict[str, object], original_dump(self, *args, **kwargs))

    monkeypatch.setattr(Row, "model_dump", checked_dump)
    stored_filter = FilterPredicate(field="category", op=PredicateOp.EQ, value=None)
    request = _request(RetrievalMode.BM25, stored_filter=stored_filter)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                [
                    {"rows": [_target_row(request, **attribute_fragment)]},
                    {"rows": None},
                ]
            ),
        )

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    result = await provider.query(request)
    await provider.close()

    assert result.target.attributes[0].state is expected_state
    assert result.candidate_lists[0].rows == ()
    assert dump_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "namespace",
    [".", "..", "a/b", r"a\b", "a%2fb", "a\ncontrol", "münchen"],
)
async def test_path_dangerous_names_fail_before_sdk_or_transport(namespace: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be reached")

    transport = RecordingTransport(handler)
    with pytest.raises(DiagnosticProviderConfigurationError) as raised:
        await TurbopufferDiagnosticProvider.create(
            api_key=_API_KEY,
            region=_REGION,
            namespace=namespace,
            transport=transport,
        )

    assert transport.requests == []
    assert transport.close_calls == 0
    assert namespace not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["api_key", "region", "namespace"])
@pytest.mark.parametrize("hostile_kind", ["object", "subclass"])
async def test_malformed_creation_values_are_fixed_and_absent_from_traceback_locals(
    field: str,
    hostile_kind: str,
) -> None:
    marker = f"hostile-{field}-creation-marker"

    class HostileString(str):
        def strip(self, *_: object, **__: object) -> str:
            raise RuntimeError(marker)

        def encode(self, *_: object, **__: object) -> bytes:
            raise RuntimeError(marker)

    hostile: object = object() if hostile_kind == "object" else HostileString(marker)
    arguments: dict[str, object] = {
        "api_key": _API_KEY,
        "region": _REGION,
        "namespace": _NAMESPACE,
    }
    arguments[field] = hostile

    with pytest.raises(DiagnosticProviderConfigurationError) as raised:
        await TurbopufferDiagnosticProvider.create(
            api_key=cast(str, arguments["api_key"]),
            region=cast(str, arguments["region"]),
            namespace=cast(str, arguments["namespace"]),
        )

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    for frame, _ in traceback.walk_tb(raised.value.__traceback__):
        if frame.f_code.co_filename == diagnostic_module.__file__:
            assert marker not in repr(frame.f_locals)
            assert hostile not in frame.f_locals.values()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_count",
        "too_many_lookup",
        "wrong_lookup_id",
        "lookup_dist",
        "unexpected_lookup_field",
        "duplicate_candidate",
        "nonmonotonic_bm25",
        "nonmonotonic_ann",
        "score_mismatch",
        "candidate_without_lookup",
        "lookup_aggregation",
        "candidate_aggregation",
        "top_level_extra",
    ],
)
async def test_malformed_or_contradictory_sdk_responses_fail_closed(mutation: str) -> None:
    request = _request(RetrievalMode.HYBRID_RRF)
    body = cast(dict[str, object], _successful_body(request))
    results = cast(list[dict[str, object]], body["results"])
    if mutation == "wrong_count":
        results.pop()
    elif mutation == "too_many_lookup":
        results[0]["rows"] = [_target_row(request), _target_row(request)]
    elif mutation == "wrong_lookup_id":
        results[0]["rows"] = [{**_target_row(request), "id": str(_OTHER)}]
    elif mutation == "lookup_dist":
        results[0]["rows"] = [{**_target_row(request), "$dist": 1.0}]
    elif mutation == "unexpected_lookup_field":
        results[0]["rows"] = [{**_target_row(request), "secret": "must-not-survive"}]
    elif mutation == "duplicate_candidate":
        results[1]["rows"] = [
            {"id": str(_OTHER), "$dist": 3.0},
            {"id": str(_OTHER), "$dist": 2.0},
        ]
    elif mutation == "nonmonotonic_bm25":
        results[1]["rows"] = [
            {"id": str(_TARGET), "$dist": 4.0},
            {"id": str(_OTHER), "$dist": 5.0},
        ]
    elif mutation == "nonmonotonic_ann":
        results[2]["rows"] = [
            {"id": str(_TARGET), "$dist": 0.25},
            {"id": str(_OTHER), "$dist": 0.1},
        ]
    elif mutation == "score_mismatch":
        cast(list[dict[str, object]], results[1]["rows"])[0]["$dist"] = 3.0
    elif mutation == "candidate_without_lookup":
        results[0]["rows"] = None
    elif mutation == "lookup_aggregation":
        results[0]["aggregations"] = {"secret_marker": 1}
    elif mutation == "candidate_aggregation":
        results[1]["aggregation_groups"] = [{"secret_marker": 1}]
    elif mutation == "top_level_extra":
        body["secret_marker"] = 1

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    with pytest.raises(DiagnosticProviderFailure) as raised:
        await provider.query(request)
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["redirect", "rate_limit", "server", "connect"])
async def test_remote_failures_never_retry_redirect_or_expose_provider_body(failure: str) -> None:
    request = _request(RetrievalMode.BM25)
    marker = "provider-response-secret-marker"

    async def handler(outbound: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError(marker, request=outbound)
        status = {"redirect": 307, "rate_limit": 429, "server": 500}[failure]
        headers = {"location": "https://redirect.invalid/stolen"} if status == 307 else None
        return httpx.Response(status, headers=headers, json={"error": marker})

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    with pytest.raises(DiagnosticProviderFailure) as raised:
        await provider.query(request)
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert marker not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", [float("nan"), float("inf"), -1.0, RuntimeError("clock-marker")]
)
async def test_invalid_or_throwing_terminal_clock_fails_after_one_attempt_and_closes(
    terminal: float | BaseException,
) -> None:
    request = _request(RetrievalMode.BM25)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0
        if isinstance(terminal, BaseException):
            raise terminal
        return terminal

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_successful_body(request))

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
        clock=clock,
    )
    with pytest.raises(DiagnosticProviderFailure) as raised:
        await provider.query(request)
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1
    assert "clock-marker" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_provider_is_one_shot_and_namespace_bound() -> None:
    request = _request(RetrievalMode.BM25)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_successful_body(request))

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    await provider.query(request)
    with pytest.raises(DiagnosticProviderFailure):
        await provider.query(request)
    await provider.close()

    assert len(transport.requests) == 1
    assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_mutated_nested_filter_is_revalidated_before_outbound_request() -> None:
    stored_filter = FilterPredicate(field="category", op=PredicateOp.EQ, value="unix")
    request = _request(RetrievalMode.BM25, stored_filter=stored_filter)
    stored_filter.field = "unsafe field"

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be reached")

    transport = RecordingTransport(handler)
    provider = await TurbopufferDiagnosticProvider.create(
        api_key=_API_KEY,
        region=_REGION,
        namespace=_NAMESPACE,
        transport=transport,
    )
    with pytest.raises(DiagnosticProviderFailure):
        await provider.query(request)
    await provider.close()

    assert transport.requests == []
    assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_namespace_construction_failure_closes_sdk_once_even_if_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSdk:
        close_calls = 0

        def __init__(self, **_: object) -> None:
            pass

        def namespace(self, _: str) -> object:
            raise RuntimeError("provider-secret-namespace-error")

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("provider-secret-close-error")

    sdk = BrokenSdk()
    monkeypatch.setattr(diagnostic_module, "AsyncTurbopuffer", lambda **_: sdk)

    with pytest.raises(DiagnosticProviderFailure) as raised:
        await TurbopufferDiagnosticProvider.create(
            api_key=_API_KEY,
            region=_REGION,
            namespace=_NAMESPACE,
        )

    assert sdk.close_calls == 1
    assert _API_KEY not in str(raised.value)
    assert "provider-secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_construction_failure_cleanup_drains_under_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    class BlockingSdk:
        close_calls = 0
        close_completed = False

        def namespace(self, _: str) -> object:
            raise RuntimeError("provider-secret-namespace-error")

        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await close_release.wait()
            self.close_completed = True

    sdk = BlockingSdk()
    monkeypatch.setattr(diagnostic_module, "AsyncTurbopuffer", lambda **_: sdk)
    task = asyncio.create_task(
        TurbopufferDiagnosticProvider.create(
            api_key=_API_KEY,
            region=_REGION,
            namespace=_NAMESPACE,
        )
    )
    await asyncio.wait_for(close_started.wait(), timeout=5)
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sdk.close_calls == 1
    assert sdk.close_completed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [KeyboardInterrupt("close-marker"), SystemExit("close-marker")])
async def test_construction_cleanup_controls_are_captured_and_reraised_fresh(
    monkeypatch: pytest.MonkeyPatch,
    control: BaseException,
) -> None:
    class ControlSdk:
        close_calls = 0

        def namespace(self, _: str) -> object:
            raise RuntimeError("provider-secret-namespace-error")

        async def close(self) -> None:
            self.close_calls += 1
            raise control

    sdk = ControlSdk()
    monkeypatch.setattr(diagnostic_module, "AsyncTurbopuffer", lambda **_: sdk)

    with pytest.raises(type(control)) as raised:
        await TurbopufferDiagnosticProvider.create(
            api_key=_API_KEY,
            region=_REGION,
            namespace=_NAMESPACE,
        )

    assert sdk.close_calls == 1
    assert raised.value is not control
    assert "close-marker" not in str(raised.value)
    assert control.__traceback__ is None
    assert control.__cause__ is None
    assert control.__context__ is None


@pytest.mark.asyncio
async def test_normal_provider_close_drains_owned_client_under_repeated_cancellation() -> None:
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    class Namespace:
        async def multi_query(self, **_: object) -> object:
            raise AssertionError("query is not part of this close test")

    class Client:
        close_calls = 0
        close_completed = False

        def namespace(self, _: str) -> Namespace:
            return Namespace()

        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await close_release.wait()
            self.close_completed = True

    client = Client()
    provider = TurbopufferDiagnosticProvider(client=client, namespace=_NAMESPACE)
    task = asyncio.create_task(provider.close())
    await asyncio.wait_for(close_started.wait(), timeout=5)
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await provider.close()

    assert client.close_calls == 1
    assert client.close_completed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    [
        "response_subclass",
        "response_missing_fields",
        "response_missing_billing",
        "response_missing_performance",
        "response_extra_empty_list",
        "response_extra_hostile_truth",
        "results_tuple",
        "results_hostile",
        "result_subclass",
        "result_missing_rows",
        "result_extra_empty_list",
        "result_extra_hostile_truth",
        "rows_tuple",
        "rows_hostile",
        "rows_overbound",
        "row_subclass",
        "fields_set_not_set",
        "fields_set_extra",
        "fields_set_missing",
        "model_extra_not_dict",
        "model_extra_missing",
        "model_extra_extra",
        "json_value_subclass",
        "json_object_subclass",
        "json_key_subclass",
    ],
)
async def test_forged_sdk_models_and_containers_fail_without_hostile_consumption(
    attack: str,
) -> None:
    request = _request(
        RetrievalMode.BM25,
        stored_filter=FilterPredicate(field="category", op=PredicateOp.EQ, value="unix"),
    )
    response: object = NamespaceMultiQueryResponse.model_validate(_successful_body(request))
    hostile_consumed = False

    class HostileContainer:
        def __iter__(self) -> object:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError("sdk-hostile-container-marker")

        def __len__(self) -> int:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError("sdk-hostile-container-marker")

    class HostileList(list[object]):
        def __iter__(self) -> object:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError("sdk-hostile-json-marker")

    class HostileDict(dict[str, object]):
        def items(self) -> object:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError("sdk-hostile-json-marker")

    class HostileTruth:
        def __bool__(self) -> bool:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError("sdk-hostile-truth-marker")

    class HostileKey(str):
        pass

    class ResponseSubclass(NamespaceMultiQueryResponse):
        pass

    class ResultSubclass(Result):
        pass

    class RowSubclass(Row):
        pass

    if attack == "response_subclass":
        response = ResponseSubclass.model_validate(_successful_body(request))
    elif attack == "response_missing_fields":
        response = object.__new__(NamespaceMultiQueryResponse)
    elif attack in {"response_missing_billing", "response_missing_performance"}:
        valid = cast(NamespaceMultiQueryResponse, response)
        values: dict[str, object] = {
            "billing": valid.billing,
            "performance": valid.performance,
            "results": valid.results,
        }
        values.pop(attack.removeprefix("response_missing_"))
        response = NamespaceMultiQueryResponse.model_construct(**values)
    elif attack == "response_extra_empty_list":
        object.__setattr__(response, "__pydantic_extra__", [])
    elif attack == "response_extra_hostile_truth":
        object.__setattr__(response, "__pydantic_extra__", HostileTruth())
    else:
        assert type(response) is NamespaceMultiQueryResponse
        first_result = response.results[0]
        assert first_result.rows is not None
        first_row = first_result.rows[0]
        if attack == "results_tuple":
            object.__setattr__(response, "results", tuple(response.results))
        elif attack == "results_hostile":
            object.__setattr__(response, "results", HostileContainer())
        elif attack == "result_subclass":
            response.results[0] = ResultSubclass.model_validate(first_result.model_dump())
        elif attack == "result_missing_rows":
            response.results[0] = Result.model_construct()
        elif attack == "result_extra_empty_list":
            object.__setattr__(first_result, "__pydantic_extra__", [])
        elif attack == "result_extra_hostile_truth":
            object.__setattr__(first_result, "__pydantic_extra__", HostileTruth())
        elif attack == "rows_tuple":
            object.__setattr__(first_result, "rows", tuple(first_result.rows))
        elif attack == "rows_hostile":
            object.__setattr__(first_result, "rows", HostileContainer())
        elif attack == "rows_overbound":
            object.__setattr__(response.results[1], "rows", [first_row] * 51)
        elif attack == "row_subclass":
            first_result.rows[0] = RowSubclass.model_validate(first_row.model_dump())
        elif attack == "fields_set_not_set":
            object.__setattr__(first_row, "__pydantic_fields_set__", [])
        elif attack == "fields_set_extra":
            object.__setattr__(
                first_row,
                "__pydantic_fields_set__",
                {*first_row.model_fields_set, "phantom"},
            )
        elif attack == "fields_set_missing":
            object.__setattr__(
                first_row,
                "__pydantic_fields_set__",
                first_row.model_fields_set - {"id"},
            )
        elif attack == "model_extra_not_dict":
            object.__setattr__(first_row, "__pydantic_extra__", HostileContainer())
        elif attack == "model_extra_missing":
            assert first_row.model_extra is not None
            del first_row.model_extra["category"]
        elif attack == "model_extra_extra":
            assert first_row.model_extra is not None
            first_row.model_extra["phantom"] = "value"
        elif attack == "json_value_subclass":
            assert first_row.model_extra is not None
            first_row.model_extra["category"] = HostileList(["unix"])
        elif attack == "json_object_subclass":
            assert first_row.model_extra is not None
            first_row.model_extra["category"] = HostileDict(value="unix")
        elif attack == "json_key_subclass":
            assert first_row.model_extra is not None
            del first_row.model_extra["category"]
            first_row.model_extra[HostileKey("category")] = "unix"

    class Namespace:
        calls = 0

        async def multi_query(self, **_: object) -> object:
            self.calls += 1
            return response

    class Client:
        close_calls = 0

        def __init__(self) -> None:
            self.bound = Namespace()

        def namespace(self, _: str) -> Namespace:
            return self.bound

        async def close(self) -> None:
            self.close_calls += 1

    client = Client()
    provider = TurbopufferDiagnosticProvider(client=client, namespace=_NAMESPACE)
    with pytest.raises(DiagnosticProviderFailure) as raised:
        await provider.query(request)
    await provider.close()

    assert client.bound.calls == 1
    assert client.close_calls == 1
    assert hostile_consumed is False
    assert "sdk-hostile" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
