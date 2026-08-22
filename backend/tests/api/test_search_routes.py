from uuid import UUID

from fastapi.testclient import TestClient
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.contracts.search import SearchCompareRequest, SearchCompareResponse
from pufferlab.main import create_app
from pufferlab.providers.errors import ProviderError, ProviderErrorDetails
from pufferlab.retrieval.errors import invalid_search

BM25_ID = UUID("a0489c14-a523-58f4-a59d-879a496cdb89")
VECTOR_ID = UUID("11e1e5c5-8390-5450-b596-51cfd0d6e94c")


class FakeSearchBackend:
    def __init__(self) -> None:
        self.requests: list[SearchCompareRequest] = []
        self.closed = False
        self.failure: Exception | None = None

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return (
            RetrievalConfigSummary(
                id=BM25_ID,
                revision=1,
                name="BM25 · body",
                mode=RetrievalMode.BM25,
                config_hash="a" * 64,
            ),
            RetrievalConfigSummary(
                id=VECTOR_ID,
                revision=1,
                name="Vector · BAAI/bge-small-en-v1.5",
                mode=RetrievalMode.VECTOR,
                config_hash="b" * 64,
            ),
        )

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return SearchCompareResponse(
            query_text=request.query_text,
            query_id=request.query_id,
            results=[],
            rank_movements=[],
            overlap=[],
            observability_notice="Only observed evidence is returned.",
        )

    async def close(self) -> None:
        self.closed = True


def _request_body() -> dict[str, object]:
    return {
        "contract_version": 1,
        "query_text": "how do pipes work",
        "config_ids": [str(BM25_ID), str(VECTOR_ID)],
        "debug_provenance": True,
    }


def test_config_discovery_and_compare_use_versioned_contracts() -> None:
    backend = FakeSearchBackend()
    with TestClient(create_app(search_backend=backend)) as client:
        configs_response = client.get("/api/v1/configs")
        compare_response = client.post("/api/v1/search/compare", json=_request_body())

    assert configs_response.status_code == 200
    assert configs_response.json() == {
        "contract_version": 1,
        "configs": [summary.model_dump(mode="json") for summary in backend.list_configs()],
    }
    assert compare_response.status_code == 200
    assert compare_response.json()["query_text"] == "how do pipes work"
    assert compare_response.json()["contract_version"] == 1
    assert backend.requests[0].config_ids == [BM25_ID, VECTOR_ID]
    assert backend.closed


def test_provider_error_is_direct_contract_and_does_not_expose_chained_secret() -> None:
    backend = FakeSearchBackend()
    try:
        raise RuntimeError("sdk-body-secret")
    except RuntimeError:
        backend.failure = ProviderError(
            "turbopuffer request failed",
            ProviderErrorDetails(
                code=ApiErrorCode.RATE_LIMITED,
                retryable=True,
                operation="query_ann",
                status_code=429,
            ),
        )

    response = TestClient(create_app(search_backend=backend)).post(
        "/api/v1/search/compare", json=_request_body()
    )

    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"
    assert response.json()["message"] == "turbopuffer request failed"
    assert response.json()["retryable"] is True
    assert response.json()["details"] == {"operation": "query_ann"}
    UUID(response.json()["trace_id"])
    assert "detail" not in response.json()
    assert "sdk-body-secret" not in response.text


def test_search_error_and_request_validation_use_direct_error_contract() -> None:
    backend = FakeSearchBackend()
    backend.failure = invalid_search("config_ids must be distinct")
    service_response = TestClient(create_app(search_backend=backend)).post(
        "/api/v1/search/compare", json=_request_body()
    )
    validation_response = TestClient(create_app(search_backend=backend)).post(
        "/api/v1/search/compare", json={"query_text": "missing config ids"}
    )

    assert service_response.status_code == 422
    assert service_response.json()["code"] == "validation_error"
    assert service_response.json()["message"] == "config_ids must be distinct"
    assert validation_response.status_code == 422
    assert validation_response.json()["code"] == "validation_error"
    assert validation_response.json()["message"] == "request validation failed"
    assert "detail" not in validation_response.json()


def test_missing_search_backend_dependency_returns_safe_503() -> None:
    app = create_app()
    app.state.search_backend = None
    response = TestClient(app).get("/api/v1/configs")

    assert response.status_code == 503
    assert response.json()["code"] == "internal_error"
    assert response.json()["message"] == "search backend is not configured"
    assert response.json()["retryable"] is False


def test_unexpected_error_is_redacted() -> None:
    backend = FakeSearchBackend()
    backend.failure = RuntimeError("unexpected-secret-value")

    with TestClient(create_app(search_backend=backend), raise_server_exceptions=False) as client:
        response = client.post("/api/v1/search/compare", json=_request_body())

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert response.json()["message"] == "request failed unexpectedly"
    assert "unexpected-secret-value" not in response.text


def test_openapi_includes_config_and_compare_contracts() -> None:
    schema = create_app().openapi()

    assert schema["paths"]["/api/v1/configs"]["get"]["operationId"] == ("list_retrieval_configs")
    assert schema["paths"]["/api/v1/search/compare"]["post"]["operationId"] == (
        "compare_search_configs"
    )
    assert "SearchCompareRequest" in schema["components"]["schemas"]
    assert "SearchCompareResponse" in schema["components"]["schemas"]
    assert "ApiErrorDetail" in schema["components"]["schemas"]
