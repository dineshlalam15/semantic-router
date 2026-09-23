"""Pydantic schemas for request, response, and health check."""

from typing import List, Optional
from pydantic import BaseModel


class RouteRequest(BaseModel):
    query: str
    include_available_models: Optional[bool] = True


class AvailableModel(BaseModel):
    model: str
    provider: str


class RouteResponse(BaseModel):
    query: str
    domain: str
    available_models: Optional[List[AvailableModel]] = None
    recommended_model: str
    llm_provider: str


class HealthResponse(BaseModel):
    status: str
    encoder_model: str
    total_routes: int
    total_models: int
