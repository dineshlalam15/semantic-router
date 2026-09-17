"""FastAPI application with health check and routing endpoints."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import RouteRequest, RouteResponse, HealthResponse
from app.router import SemanticRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router: SemanticRouter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    logger.info("Starting Semantic LLM Router...")
    router = SemanticRouter()
    logger.info("Semantic LLM Router ready.")
    yield


app = FastAPI(
    title="Semantic LLM Router",
    version="1.0.0",
    description="Demo Semantic Router for LLM selection",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check() -> HealthResponse:
    """Returns service health and configuration overview."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return HealthResponse(
        status="healthy",
        encoder_model=getattr(router.encoder, "name", "sentence-transformers/all-MiniLM-L6-v2"),
        total_routes=len(router.routes),
        total_models=len(router.models),
    )


@app.post("/route", response_model=RouteResponse, summary="Route query to optimal LLM model")
async def route_query(payload: RouteRequest) -> RouteResponse:
    """Takes a user query, identifies the domain, and returns the recommended model and provider."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        return router.route(payload.query)
    except Exception as exc:
        logger.error(f"Routing failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
