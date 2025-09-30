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
from .api.v1 import audio, chat, embeddings, models, vector_store
from .config import ServerConfig
from .managers.embedding_manager import EmbeddingManager
from .managers.runtime_manager import RuntimeManager
from .storage.lancedb_client import LanceDBClient

logger = logging.getLogger(__name__)

# Global manager instances
runtime_manager: RuntimeManager | None = None
embedding_manager: EmbeddingManager | None = None
lancedb_client: LanceDBClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for startup and shutdown

    Args:
        app: FastAPI application instance
    """
    # Startup
    global runtime_manager, embedding_manager, lancedb_client

    config: ServerConfig = app.state.config
    logger.info("Starting Fluid Server...")
    logger.info(f"Data root: {config.data_root}")
    logger.info(f"Model path: {config.model_path}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Embeddings enabled: {config.enable_embeddings}")
    logger.info("Device assignment: GPU for LLM, NPU for Whisper")

    # Initialize runtime manager
    runtime_manager = RuntimeManager(config)
    await runtime_manager.initialize()

    # Initialize embedding manager if enabled
    if config.enable_embeddings:
        try:
            embedding_manager = EmbeddingManager(config)
            await embedding_manager.initialize()

            # Initialize LanceDB client
            lancedb_client = LanceDBClient(
                db_path=config.embeddings_db_path,
                db_name=config.embeddings_db_name
            )
            await lancedb_client.initialize()

            logger.info("Embedding system initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding system: {e}")
            embedding_manager = None
            lancedb_client = None
    else:
        logger.info("Embeddings disabled in configuration")

    # Store in app state for dependency injection
    app.state.runtime_manager = runtime_manager
    app.state.embedding_manager = embedding_manager
    app.state.lancedb_client = lancedb_client

    yield

    # Shutdown
    logger.info("Shutting down Fluid Server...")
    if runtime_manager:
        await runtime_manager.unload_all()
    if embedding_manager:
        await embedding_manager.shutdown()
    if lancedb_client:
        await lancedb_client.close()


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

    # Override dependencies for the whole app
    from .api.health import get_runtime_manager as health_get_runtime_manager
    from .api.v1.audio import get_runtime_manager as audio_get_runtime_manager
    from .api.v1.chat import get_runtime_manager as chat_get_runtime_manager
    from .api.v1.models import get_runtime_manager as models_get_runtime_manager

    # Import embedding dependencies if they exist
    try:
        from .api.v1.embeddings import get_embedding_manager
        from .api.v1.embeddings import get_runtime_manager as embeddings_get_runtime_manager
        from .api.v1.vector_store import get_embedding_manager as vs_get_embedding_manager
        from .api.v1.vector_store import get_lancedb_client
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        EMBEDDINGS_AVAILABLE = False

    # Dependency functions
    def get_runtime_manager() -> RuntimeManager:
        """Get runtime manager dependency"""
        return app.state.runtime_manager

    def get_embedding_manager_dep() -> EmbeddingManager:
        """Get embedding manager dependency"""
        return app.state.embedding_manager

    def get_lancedb_client_dep() -> LanceDBClient:
        """Get LanceDB client dependency"""
        return app.state.lancedb_client

    # Set up dependency overrides
    overrides = {
        chat_get_runtime_manager: get_runtime_manager,
        models_get_runtime_manager: get_runtime_manager,
        audio_get_runtime_manager: get_runtime_manager,
        health_get_runtime_manager: get_runtime_manager,
    }

    # Add embedding dependencies if available
    if EMBEDDINGS_AVAILABLE:
        overrides.update({
            get_embedding_manager: get_embedding_manager_dep,
            embeddings_get_runtime_manager: get_runtime_manager,
            vs_get_embedding_manager: get_embedding_manager_dep,
            get_lancedb_client: get_lancedb_client_dep,
        })

    app.dependency_overrides = overrides

    # Include routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(audio.router)

    # Include embedding routers if embeddings are available
    if EMBEDDINGS_AVAILABLE and config.enable_embeddings:
        app.include_router(embeddings.router)
        app.include_router(vector_store.router)

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
        port=int(os.getenv("FLUID_PORT", "3847")),
        data_root=Path(os.getenv("FLUID_DATA_ROOT", "./data")),
        model_path=Path(os.getenv("FLUID_MODEL_PATH", "")) if os.getenv("FLUID_MODEL_PATH") else None,
        cache_dir=Path(os.getenv("FLUID_CACHE_DIR", "")) if os.getenv("FLUID_CACHE_DIR") else None,
        llm_model=os.getenv("FLUID_LLM_MODEL", "qwen3-8b-int4-ov"),
        whisper_model=os.getenv("FLUID_WHISPER_MODEL", "whisper-tiny"),
        enable_embeddings=os.getenv("FLUID_ENABLE_EMBEDDINGS", "true").lower() == "true",
        embedding_model=os.getenv("FLUID_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        warm_up=os.getenv("FLUID_WARM_UP", "true").lower() == "true",
        idle_timeout_minutes=int(os.getenv("FLUID_IDLE_TIMEOUT", "5")),
    )

    logger.info(f"Initializing worker app with config: {config}")
    return create_app(config)


# For PyInstaller compatibility, directly assign the factory function
app = create_worker_app
