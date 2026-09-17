"""FastAPI route definitions for semantic LLM routing API."""

import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, status

from app.schemas import (
    RouteRequest,
    RouteResponse,
    HealthResponse,
    RoutesConfigFile,
    ModelsConfigFile,
)
from app.semantic_router import SemanticDomainRouter
from app.model_selector import ModelSelector, ModelSelectionError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_router(request: Request) -> SemanticDomainRouter:
    """Dependency retrieving the initialized SemanticDomainRouter from app state."""
    r = getattr(request.app.state, "semantic_router", None)
    if not r:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic router is not initialized or still starting up."
        )
    return r


def get_model_selector(request: Request) -> ModelSelector:
    """Dependency retrieving the initialized ModelSelector from app state."""
    ms = getattr(request.app.state, "model_selector", None)
    if not ms:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model selector is not initialized or still starting up."
        )
    return ms


@router.post(
    "/route",
    response_model=RouteResponse,
    status_code=status.HTTP_200_OK,
    summary="Route a user query to the optimal LLM model and domain",
    description="Performs non-threshold semantic routing using HuggingFace sentence representations, "
                "identifies the domain, and selects the optimal LLM model from configured providers."
)
async def route_query(
    payload: RouteRequest,
    semantic_router: SemanticDomainRouter = Depends(get_router),
    model_selector: ModelSelector = Depends(get_model_selector),
) -> RouteResponse:
    start_time = time.perf_counter()

    try:
        # Step 1: Semantic Domain Classification (Deterministic Argmax Ranking)
        routing_result = semantic_router.route(payload.query)

        # Step 2: Model Selection based on domain and strategy
        selected_model, active_strategy, candidate_models = model_selector.select_model(
            domain=routing_result.selected_domain,
            strategy_override=payload.strategy
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Structured logging (without logging sensitive user text or embeddings)
        query_preview = payload.query[:60] + "..." if len(payload.query) > 60 else payload.query
        logger.info(
            f"Routed query: preview='{query_preview}' -> Domain='{routing_result.selected_domain}' "
            f"(score={routing_result.confidence_score:.3f}) -> Model='{selected_model.model_name}' "
            f"Provider='{selected_model.provider}' Strategy='{active_strategy}' Latency={elapsed_ms:.2f}ms"
        )

        return RouteResponse(
            query=payload.query,
            domain=routing_result.selected_domain,
            recommended_model=selected_model.model_name,
            llm_provider=selected_model.provider,
            selection_strategy=active_strategy,
            confidence_score=routing_result.confidence_score,
            ranked_domains=routing_result.ranked_candidates,
            candidate_models=candidate_models,
            routing_time_ms=round(elapsed_ms, 2),
        )

    except ModelSelectionError as err:
        logger.error(f"Model selection error during routing: {err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err)
        )
    except Exception as exc:
        logger.error(f"Unexpected error during routing: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal routing error: {str(exc)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health and readiness probe"
)
async def health_check(request: Request) -> HealthResponse:
    semantic_router = getattr(request.app.state, "semantic_router", None)
    model_selector = getattr(request.app.state, "model_selector", None)
    config = getattr(request.app.state, "config", None)

    encoder_loaded = semantic_router is not None
    encoder_model = config.settings.encoder.model_name if config else "unknown"
    total_routes = len(config.routes) if config else 0
    total_utterances = (
        sum(len(r.utterances) for r in config.routes) if config else 0
    )
    total_models = len(config.models) if config else 0
    providers = sorted(list({m.provider for m in config.models})) if config else []

    return HealthResponse(
        status="healthy" if encoder_loaded else "initializing",
        encoder_loaded=encoder_loaded,
        encoder_model=encoder_model,
        total_routes=total_routes,
        total_utterances=total_utterances,
        total_models=total_models,
        configured_providers=providers,
    )


@router.get("/routes", summary="Inspect configured semantic routes")
async def list_routes(request: Request):
    config = getattr(request.app.state, "config", None)
    if not config:
        raise HTTPException(status_code=503, detail="Configurations not loaded.")
    return [
        {
            "name": r.name,
            "description": r.description,
            "priority": r.priority,
            "utterance_count": len(r.utterances),
            "sample_utterances": r.utterances[:3],
        }
        for r in config.routes
    ]


@router.get("/models", summary="Inspect configured models and providers")
async def list_models(request: Request):
    config = getattr(request.app.state, "config", None)
    if not config:
        raise HTTPException(status_code=503, detail="Configurations not loaded.")
    return [
        {
            "provider": m.provider,
            "model_name": m.model_name,
            "domains": m.domains,
            "capabilities": m.capabilities,
            "metadata": m.metadata.model_dump(exclude_none=True),
        }
        for m in config.models
    ]
