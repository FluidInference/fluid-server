"""
Models listing and management endpoints
"""
import time
import logging
from typing import Annotated
from pathlib import Path
from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download, HfApi
from ...models.openai import ModelsResponse, ModelInfo
from ...managers.runtime_manager import RuntimeManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


class ModelDownloadRequest(BaseModel):
    """Request to download a model from HuggingFace"""
    repo_id: str  # e.g., "FluidInference/Qwen3-1.7B-fp16-ov"
    model_type: str = "llm"  # llm, whisper, embedding
    local_name: str | None = None  # Optional custom name for local storage


class ModelDownloadResponse(BaseModel):
    """Response from model download"""
    success: bool
    message: str
    local_path: str | None = None
    model_id: str | None = None


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.state.runtime_manager


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    runtime_manager: Annotated[RuntimeManager, Depends(get_runtime_manager)]
) -> ModelsResponse:
    """
    List available models (OpenAI-compatible)
    
    Args:
        runtime_manager: Runtime manager instance (injected)
        
    Returns:
        List of available models
    """
    models = []
    
    # Add LLM models
    for model_name in runtime_manager.available_models.get("llm", []):
        models.append(ModelInfo(
            id=model_name,
            created=int(time.time()),
            owned_by="local",
            model_type="llm"
        ))
    
    # Add Whisper models
    for model_name in runtime_manager.available_models.get("whisper", []):
        models.append(ModelInfo(
            id=f"whisper-{model_name}",
            created=int(time.time()),
            owned_by="local",
            model_type="whisper"
        ))
    
    # Add embedding models (future)
    for model_name in runtime_manager.available_models.get("embedding", []):
        models.append(ModelInfo(
            id=f"embedding-{model_name}",
            created=int(time.time()),
            owned_by="local",
            model_type="embedding"
        ))
    
    logger.debug(f"Returning {len(models)} available models")
    
    return ModelsResponse(data=models)


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model(
    request: ModelDownloadRequest,
    runtime_manager: Annotated[RuntimeManager, Depends(get_runtime_manager)]
) -> ModelDownloadResponse:
    """
    Download a model from HuggingFace Hub
    
    Args:
        request: Download request with repo_id and options
        runtime_manager: Runtime manager instance (injected)
        
    Returns:
        Download result with local path
    """
    try:
        # Validate model type
        if request.model_type not in ["llm", "whisper", "embedding"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_type: {request.model_type}. Must be 'llm', 'whisper', or 'embedding'"
            )
        
        # Get model directory
        model_base_path = runtime_manager.config.model_path / request.model_type
        model_base_path.mkdir(parents=True, exist_ok=True)
        
        # Determine local name
        if request.local_name:
            local_name = request.local_name
        else:
            # Extract model name from repo_id (e.g., "FluidInference/Qwen3-1.7B-fp16-ov" -> "Qwen3-1.7B-fp16-ov")
            local_name = request.repo_id.split("/")[-1]
        
        local_path = model_base_path / local_name
        
        # Check if model already exists
        if local_path.exists():
            logger.info(f"Model {request.repo_id} already exists at {local_path}")
            # Refresh available models in runtime manager
            runtime_manager.discover_models()
            return ModelDownloadResponse(
                success=True,
                message=f"Model already exists at {local_path}",
                local_path=str(local_path),
                model_id=local_name
            )
        
        logger.info(f"Downloading model {request.repo_id} to {local_path}")
        
        # Verify repo exists before downloading
        api = HfApi()
        try:
            repo_info = api.repo_info(request.repo_id)
            logger.info(f"Found repo: {repo_info.id} ({repo_info.modelId})")
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Repository {request.repo_id} not found on HuggingFace Hub: {str(e)}"
            )
        
        # Download the model
        downloaded_path = snapshot_download(
            repo_id=request.repo_id,
            local_dir=str(local_path),
            resume_download=True,
            local_dir_use_symlinks=False  # Copy files instead of symlinks for Windows compatibility
        )
        
        logger.info(f"Successfully downloaded {request.repo_id} to {downloaded_path}")
        
        # Refresh available models in runtime manager
        runtime_manager.discover_models()
        
        return ModelDownloadResponse(
            success=True,
            message=f"Successfully downloaded {request.repo_id}",
            local_path=str(local_path),
            model_id=local_name
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to download model {request.repo_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {str(e)}"
        )