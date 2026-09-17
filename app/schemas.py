"""Pydantic schemas for request, response, and health check."""

from pydantic import BaseModel


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    query: str
    domain: str
    recommended_model: str
    llm_provider: str


class HealthResponse(BaseModel):
    status: str
    encoder_model: str
    total_routes: int
    total_models: int
