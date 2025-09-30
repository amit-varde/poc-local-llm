"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import get_config
from ..inference import InferenceEngine
from .routes import router
from .middleware import setup_middleware
from .utils import get_inference_engine

logger = logging.getLogger(__name__)

# Global inference engine instance
inference_engine: InferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global inference_engine

    logger.info("Starting Local LLM Pipeline API")

    # Initialize inference engine
    inference_engine = InferenceEngine()

    # Load default model if configured
    config = get_config()
    settings = config.load_settings()

    if settings.model.default_model:
        try:
            await inference_engine.load_model(settings.model.default_model)
            logger.info(f"Loaded default model: {settings.model.default_model}")
        except Exception as e:
            logger.warning(f"Failed to load default model: {e}")

    # Store in app state
    app.state.inference_engine = inference_engine

    yield

    # Cleanup
    if inference_engine:
        await inference_engine.unload_model()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    settings = config.load_settings()

    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        openapi_url=settings.api.openapi_url,
        docs_url=settings.api.docs_url,
        redoc_url=settings.api.redoc_url,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    # Setup custom middleware
    setup_middleware(app)

    # Include routes
    app.include_router(router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

    return app