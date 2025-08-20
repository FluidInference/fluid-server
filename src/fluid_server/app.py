"""
Main FastAPI application
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import health
from .api.v1 import audio, chat, models
from .config import ServerConfig
from .managers.runtime_manager import RuntimeManager

logger = logging.getLogger(__name__)

# Global runtime manager instance
runtime_manager: RuntimeManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for startup and shutdown

    Args:
        app: FastAPI application instance
    """
    # Startup
    global runtime_manager

    config: ServerConfig = app.state.config
    logger.info("Starting Fluid Server...")
    logger.info(f"Model path: {config.model_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info("Device assignment: GPU for LLM, NPU for Whisper")

    # Initialize runtime manager
    runtime_manager = RuntimeManager(config)
    await runtime_manager.initialize()

    # Store in app state for dependency injection
    app.state.runtime_manager = runtime_manager

    yield

    # Shutdown
    logger.info("Shutting down Fluid Server...")
    if runtime_manager:
        await runtime_manager.unload_all()


def create_app(config: ServerConfig) -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        config: Server configuration

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Fluid Server",
        description="OpenAI-compatible API server with multi-model support",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    # Add CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependency to get runtime manager
    async def get_runtime_manager() -> RuntimeManager:
        """Get runtime manager dependency"""
        return app.state.runtime_manager

    # Override dependencies in routers
    chat.router.dependency_overrides = {RuntimeManager: get_runtime_manager}
    models.router.dependency_overrides = {RuntimeManager: get_runtime_manager}
    # For audio, we need to override the specific function
    from .api.v1.audio import get_runtime_manager as audio_get_runtime_manager

    audio.router.dependency_overrides = {audio_get_runtime_manager: get_runtime_manager}
    health.router.dependency_overrides = {RuntimeManager: get_runtime_manager}

    # Include routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(audio.router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {"message": str(exc), "type": type(exc).__name__, "code": "internal_error"}
            },
        )

    return app


# Global app factory function for uvicorn string reference
def create_worker_app() -> FastAPI:
    """Create app instance for worker process using environment config"""
    import os
    from pathlib import Path

    config = ServerConfig(
        host=os.getenv("FLUID_HOST", "127.0.0.1"),
        port=int(os.getenv("FLUID_PORT", "8080")),
        model_path=Path(os.getenv("FLUID_MODEL_PATH", "./models")),
        cache_dir=Path(os.getenv("FLUID_CACHE_DIR")) if os.getenv("FLUID_CACHE_DIR") else None,
        llm_model=os.getenv("FLUID_LLM_MODEL", "qwen3-8b-int4-ov"),
        whisper_model=os.getenv("FLUID_WHISPER_MODEL", "whisper-tiny"),
        warm_up=os.getenv("FLUID_WARM_UP", "true").lower() == "true",
        idle_timeout_minutes=int(os.getenv("FLUID_IDLE_TIMEOUT", "5")),
    )

    logger.info(f"Initializing worker app with config: {config}")
    return create_app(config)


# For PyInstaller compatibility, directly assign the factory function
app = create_worker_app
