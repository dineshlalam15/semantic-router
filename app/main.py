"""FastAPI application entrypoint with lifespan lifecycle and initialization."""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config_loader import load_configurations, ConfigurationError
from app.encoder import SemanticEncoder, EncoderInitializationError
from app.semantic_router import SemanticDomainRouter
from app.model_selector import ModelSelector
from app.routes import router

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("semantic_router_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling startup initialization and graceful shutdown."""
    logger.info("Initializing Semantic LLM Router service...")

    config_dir = Path(os.getenv("CONFIG_DIR", "config"))

    try:
        # 1. Load and validate configurations
        logger.info(f"Loading configuration files from '{config_dir.resolve()}'...")
        app_config = load_configurations(config_dir=config_dir)
        app.state.config = app_config
        logger.info(
            f"Successfully validated {len(app_config.routes)} routes and "
            f"{len(app_config.models)} models across providers."
        )

        # 2. Initialize HuggingFace encoder once at startup
        encoder_name = app_config.settings.encoder.model_name
        device = os.getenv("DEVICE", app_config.settings.encoder.device)
        logger.info(f"Initializing SemanticEncoder ('{encoder_name}') on device: {device}...")
        encoder = SemanticEncoder(model_name=encoder_name, device=device)
        app.state.encoder = encoder

        # 3. Pre-compute route utterance embeddings once at startup
        logger.info("Building SemanticDomainRouter and embedding route utterances...")
        domain_router = SemanticDomainRouter(routes=app_config.routes, encoder=encoder)
        app.state.semantic_router = domain_router

        # 4. Initialize ModelSelector with configured strategies
        logger.info("Initializing ModelSelector with pluggable strategies...")
        default_strat = app_config.settings.routing.default_selection_strategy
        weights = app_config.settings.routing.balanced_weights
        model_selector = ModelSelector(
            models=app_config.models,
            default_strategy=default_strat,
            balanced_weights=weights
        )
        app.state.model_selector = model_selector

        logger.info("Semantic LLM Router startup complete. Ready for routing requests.")

    except (ConfigurationError, EncoderInitializationError) as err:
        logger.critical(f"FATAL: Application failed to start due to configuration/encoder error: {err}")
        raise err
    except Exception as exc:
        logger.critical(f"FATAL: Unexpected error during application startup: {exc}", exc_info=True)
        raise exc

    yield

    logger.info("Shutting down Semantic LLM Router service. Cleanup complete.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    application = FastAPI(
        title="Semantic LLM Router",
        description=(
            "A decoupled, configuration-driven decision-making layer that routes user queries "
            "to the optimal LLM model using HuggingFace semantic embeddings without manual thresholds "
            "or external LLM API calls."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
