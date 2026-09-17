"""Pydantic schemas and contracts for configuration, requests, and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Configuration Schemas
# ============================================================================

class RouteConfig(BaseModel):
    """Configuration for an individual semantic route."""
    name: str = Field(..., min_length=1, description="Unique domain identifier (e.g. software_engineering)")
    description: Optional[str] = Field(None, description="Human-readable description of the domain")
    priority: int = Field(default=1, description="Tie-breaker priority (higher is preferred)")
    utterances: List[str] = Field(..., min_length=1, description="Sample representative phrases for vector matching")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Route name cannot be empty or whitespace.")
        return v_clean

    @field_validator("utterances")
    @classmethod
    def validate_utterances(cls, v: List[str]) -> List[str]:
        cleaned = [u.strip() for u in v if u and u.strip()]
        if not cleaned:
            raise ValueError("Route must have at least one non-empty utterance.")
        return cleaned


class RoutesConfigFile(BaseModel):
    """Container schema for routes.yaml."""
    routes: List[RouteConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_routes(self) -> "RoutesConfigFile":
        seen = set()
        for r in self.routes:
            if r.name in seen:
                raise ValueError(f"Duplicate route name detected: '{r.name}'. Route names must be unique.")
            seen.add(r.name)
        return self


class ModelMetadata(BaseModel):
    """Operational and performance metadata for an LLM model."""
    context_window: Optional[int] = Field(None, ge=1000)
    input_cost_per_m: Optional[float] = Field(None, ge=0.0)
    output_cost_per_m: Optional[float] = Field(None, ge=0.0)
    latency_p90_ms: Optional[int] = Field(None, ge=0)
    reasoning_capability: Optional[float] = Field(None, ge=0.0, le=10.0)
    coding_capability: Optional[float] = Field(None, ge=0.0, le=10.0)
    vision_capability: Optional[bool] = False
    tool_use: Optional[bool] = False
    availability: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    priority: Optional[int] = Field(default=1, ge=1)


class ModelConfig(BaseModel):
    """Configuration for a single LLM model entry."""
    provider: str = Field(..., min_length=1, description="Provider name: OpenAI, Gemini, Anthropic, LiteLLM")
    model_name: str = Field(..., min_length=1, description="Model identifier")
    domains: List[str] = Field(default_factory=list, description="Supported semantic domains")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)

    @field_validator("provider", "model_name")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Field cannot be empty or whitespace.")
        return v_clean


class ModelsConfigFile(BaseModel):
    """Container schema for models.yaml."""
    models: List[ModelConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_models(self) -> "ModelsConfigFile":
        seen = set()
        for m in self.models:
            key = (m.provider.lower(), m.model_name.lower())
            if key in seen:
                raise ValueError(f"Duplicate model detected: '{m.model_name}' under provider '{m.provider}'.")
            seen.add(key)
        return self


class EncoderConfig(BaseModel):
    """Configuration for HuggingFace encoder."""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu")
    batch_size: int = Field(default=32, ge=1)


class BalancedWeightsConfig(BaseModel):
    """Weights for the balanced selection strategy."""
    quality: float = Field(default=0.5, ge=0.0)
    cost: float = Field(default=0.25, ge=0.0)
    latency: float = Field(default=0.25, ge=0.0)


class RoutingSettings(BaseModel):
    """Settings governing router behavior and selection strategies."""
    default_selection_strategy: str = Field(
        default="quality",
        description="Selection strategy: quality, cost, latency, priority, or balanced"
    )
    balanced_weights: BalancedWeightsConfig = Field(default_factory=BalancedWeightsConfig)

    @field_validator("default_selection_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"quality", "cost", "latency", "priority", "balanced"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid strategy '{v}'. Must be one of: {valid}")
        return v.lower()


class ServerSettings(BaseModel):
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class SettingsConfigFile(BaseModel):
    """Container schema for settings.yaml."""
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


# ============================================================================
# API Request & Response Schemas
# ============================================================================

class RouteRequest(BaseModel):
    """API request payload for POST /route."""
    query: str = Field(..., min_length=1, description="Incoming user query to route")
    strategy: Optional[str] = Field(
        None,
        description="Optional selection strategy override: 'quality', 'cost', 'latency', 'priority', 'balanced'"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or purely whitespace.")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def validate_strategy_override(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid = {"quality", "cost", "latency", "priority", "balanced"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid strategy override '{v}'. Must be one of: {valid}")
        return v.lower()


class DomainCandidate(BaseModel):
    """Details of a candidate domain and its semantic score."""
    domain: str
    similarity_score: float
    top_matched_utterance: Optional[str] = None


class ModelCandidate(BaseModel):
    """Brief representation of an eligible candidate model."""
    provider: str
    model_name: str
    priority: Optional[int] = None
    input_cost_per_m: Optional[float] = None
    output_cost_per_m: Optional[float] = None
    latency_p90_ms: Optional[int] = None
    reasoning_capability: Optional[float] = None


class RouteResponse(BaseModel):
    """API response payload for POST /route."""
    query: str
    domain: str
    recommended_model: str
    llm_provider: str
    selection_strategy: Optional[str] = None
    confidence_score: Optional[float] = None
    ranked_domains: Optional[List[DomainCandidate]] = None
    candidate_models: Optional[List[ModelCandidate]] = None
    routing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """API response payload for GET /health."""
    status: str
    encoder_loaded: bool
    encoder_model: str
    total_routes: int
    total_utterances: int
    total_models: int
    configured_providers: List[str]
