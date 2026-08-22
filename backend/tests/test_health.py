from fastapi.testclient import TestClient
from pufferlab.main import create_app


def test_health_contract() -> None:
    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": 1,
        "status": "ok",
        "version": "0.1.0",
    }


def test_openapi_uses_versioned_health_path() -> None:
    schema = create_app().openapi()

    assert "/api/v1/health" in schema["paths"]
    assert schema["paths"]["/api/v1/health"]["get"]["operationId"] == "get_health"
