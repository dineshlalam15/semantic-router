"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient with FastAPI lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify GET /health returns operational status and provider details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["encoder_loaded"] is True
    assert data["total_routes"] >= 5
    assert data["total_utterances"] >= 60
    assert data["total_models"] >= 8
    assert "OpenAI" in data["configured_providers"]
    assert "Gemini" in data["configured_providers"]
    assert "Anthropic" in data["configured_providers"]
    assert "LiteLLM" in data["configured_providers"]


def test_routes_list_endpoint(client):
    """Verify GET /routes lists configured routes."""
    response = client.get("/routes")
    assert response.status_code == 200
    routes = response.json()
    assert len(routes) >= 5
    route_names = [r["name"] for r in routes]
    assert "software_engineering" in route_names
    assert "stem_mathematics" in route_names
    assert "legal_document_analysis" in route_names


def test_models_list_endpoint(client):
    """Verify GET /models lists configured models."""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 8
    providers = {m["provider"] for m in models}
    assert {"OpenAI", "Gemini", "Anthropic", "LiteLLM"}.issubset(providers)


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
    assert data["recommended_model"] is not None
    assert data["llm_provider"] is not None
    assert data["confidence_score"] > 0.6
    assert isinstance(data["ranked_domains"], list)
    assert len(data["ranked_domains"]) >= 5
    assert isinstance(data["candidate_models"], list)
    assert data["routing_time_ms"] > 0


def test_post_route_strategy_override(client):
    """Verify POST /route supports overriding selection strategy."""
    payload = {
        "query": "How do I implement an LRU cache in Python?",
        "strategy": "cost"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["selection_strategy"] == "cost"


def test_post_route_empty_query_rejected(client):
    """Verify empty query returns 422 validation error."""
    response = client.post("/route", json={"query": ""})
    assert response.status_code == 422


def test_post_route_whitespace_query_rejected(client):
    """Verify whitespace-only query returns 422 validation error."""
    response = client.post("/route", json={"query": "   \n\t  "})
    assert response.status_code == 422


def test_post_route_invalid_strategy_rejected(client):
    """Verify invalid strategy name returns 422 validation error."""
    response = client.post("/route", json={"query": "Explain quantum physics", "strategy": "magic_speed"})
    assert response.status_code == 422
