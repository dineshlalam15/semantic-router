"""Integration tests for the 2 FastAPI endpoints: /health and /route."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient with FastAPI lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify GET /health returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MiniLM" in data["encoder_model"]
    assert data["total_routes"] >= 6
    assert data["total_models"] >= 8


def test_post_route_coding_query(client):
    """Verify POST /route correctly classifies and selects model for coding query."""
    payload = {
        "query": "How do I implement an LRU cache in Python?"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert data["domain"] == "software_engineering"
    assert data["recommended_model"] == "claude-3-5-sonnet-20241022"
    assert data["llm_provider"] == "Anthropic"
    assert set(data.keys()) == {"query", "domain", "recommended_model", "llm_provider"}


def test_post_route_image_query(client):
    """Verify POST /route correctly classifies and selects model for image generation query."""
    payload = {
        "query": "Generate a photorealistic portrait of an astronaut on Mars in watercolor style."
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["domain"] == "image_generation"
    assert data["recommended_model"] == "dall-e-3"
    assert data["llm_provider"] == "OpenAI"
    assert set(data.keys()) == {"query", "domain", "recommended_model", "llm_provider"}


def test_post_route_missing_query_field(client):
    """Verify missing query field returns 422 validation error."""
    response = client.post("/route", json={})
    assert response.status_code == 422
