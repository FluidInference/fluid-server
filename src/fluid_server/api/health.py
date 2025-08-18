"""
Health check endpoint
"""
import logging
from fastapi import APIRouter
from ..models.openai import HealthStatus
from ..managers.runtime_manager import RuntimeManager
from .. import __version__

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint
        
    Returns:
        Server health status
    """
    # Get available devices
    available_devices = _get_available_devices()
    
    # For now, return basic health status
    # TODO: Get runtime manager properly via dependency injection
    return HealthStatus(
        status="ready",
        models_loaded={},
        device="unknown",
        available_devices=available_devices,
        version=__version__
    )


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fluid Server - OpenAI-compatible API",
        "version": __version__,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "transcription": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "health": "/health"
        }
    }


def _get_available_devices() -> list[str]:
    """Get list of available OpenVINO devices"""
    try:
        import openvino as ov
        core = ov.Core()
        return core.available_devices
    except Exception as e:
        logger.warning(f"Could not get available devices: {e}")
        return []