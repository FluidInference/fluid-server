"""
Health check endpoint
"""
import logging
from fastapi import APIRouter, Request, Depends
from ..models.openai import HealthStatus
from ..managers.runtime_manager import RuntimeManager
from .. import __version__

logger = logging.getLogger(__name__)
router = APIRouter()


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.state.runtime_manager


@router.get("/health", response_model=HealthStatus)
async def health_check(runtime_manager: RuntimeManager = Depends(get_runtime_manager)) -> HealthStatus:
    """
    Health check endpoint
        
    Returns:
        Server health status including warm-up progress
    """
    # Get available devices
    available_devices = _get_available_devices()
    
    # Determine current device and models loaded
    current_device = "unknown"
    models_loaded = {}
    
    if runtime_manager.current_runtime:
        current_device = runtime_manager.current_runtime.device
        model_type = "llm" if hasattr(runtime_manager.current_runtime, 'pipeline') and hasattr(runtime_manager.current_runtime.pipeline, 'generate') else "whisper"
        models_loaded[model_type] = True
    
    # Determine overall status based on warm-up
    if runtime_manager.warmup_status.get("in_progress", False):
        status = "warming_up"
    elif runtime_manager.warmup_status.get("whisper") == "ready" or runtime_manager.warmup_status.get("llm") == "ready":
        status = "ready"
    else:
        status = "ready"  # Default to ready if no warm-up info
    
    # Create warmup status copy for response (exclude start_time for cleaner response)
    warmup_status_response = None
    if runtime_manager.warmup_status.get("start_time") or runtime_manager.warmup_status.get("in_progress"):
        warmup_status_response = {
            "in_progress": runtime_manager.warmup_status.get("in_progress", False),
            "whisper": runtime_manager.warmup_status.get("whisper", "pending"),
            "llm": runtime_manager.warmup_status.get("llm", "pending"),
            "current_task": runtime_manager.warmup_status.get("current_task", "")
        }
        # Add elapsed time if warm-up is running
        if runtime_manager.warmup_status.get("start_time") and runtime_manager.warmup_status.get("in_progress"):
            import time
            elapsed = int(time.time() - runtime_manager.warmup_status["start_time"])
            warmup_status_response["elapsed_seconds"] = elapsed
    
    return HealthStatus(
        status=status,
        models_loaded=models_loaded,
        device=current_device,
        available_devices=available_devices,
        warmup_status=warmup_status_response,
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