"""
Main FastAPI application
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import ServerConfig
from .managers.runtime_manager import RuntimeManager
from .api import health
from .api.v1 import chat, models, audio

logger = logging.getLogger(__name__)

# Global runtime manager instance
runtime_manager: Optional[RuntimeManager] = None


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
    logger.info(f"Device: {config.device}")
    
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
        lifespan=lifespan
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
    audio.router.dependency_overrides = {RuntimeManager: get_runtime_manager}
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
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "code": "internal_error"
                }
            }
        )
    
    return app