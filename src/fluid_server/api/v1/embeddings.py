"""
OpenAI-compatible embeddings endpoint
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ...managers.embedding_manager import EmbeddingManager
from ...managers.runtime_manager import RuntimeManager
from ...models.openai import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


def get_embedding_manager(request: Request) -> EmbeddingManager:
    """Dependency to get embedding manager"""
    return request.app.embedding_manager


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.runtime_manager


@router.post("/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)]
) -> EmbeddingResponse:
    """
    Create embeddings for text inputs (OpenAI-compatible)
    
    This endpoint is compatible with OpenAI's embeddings API and can be used
    as a drop-in replacement.
    """
    try:
        # Validate that embeddings are enabled
        if not embedding_manager.config.enable_embeddings:
            raise HTTPException(
                status_code=503,
                detail="Embeddings functionality is disabled"
            )

        # Ensure input is a list
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # Generate embeddings
        start_time = time.time()
        embeddings = await embedding_manager.get_text_embeddings(
            texts=inputs,
            model_name=request.model
        )
        processing_time = time.time() - start_time
        
        # Create response data
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    embedding=embedding,
                    index=i
                )
            )
        
        # Calculate usage statistics (approximate)
        total_tokens = sum(len(text.split()) for text in inputs)
        usage = EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
        
        # Create response
        response = EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=usage
        )
        
        logger.info(
            f"Generated embeddings for {len(inputs)} inputs "
            f"using model '{request.model}' in {processing_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/batch")
async def create_embeddings_batch(
    requests: List[EmbeddingRequest],
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)]
) -> List[EmbeddingResponse]:
    """
    Create embeddings for multiple requests in batch
    """
    try:
        if not embedding_manager.config.enable_embeddings:
            raise HTTPException(
                status_code=503,
                detail="Embeddings functionality is disabled"
            )
        
        responses = []
        for request in requests:
            # Process each request individually but return as batch
            response = await create_embeddings(request, embedding_manager)
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch embeddings: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/embeddings/models")
async def list_embedding_models(
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)]
) -> JSONResponse:
    """
    List available embedding models
    """
    try:
        info = embedding_manager.get_embedding_info()
        
        models = []
        for model_type, model_list in info["available_models"].items():
            for model_name in model_list:
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fluid-server",
                    "model_type": f"embedding_{model_type}"
                })
        
        return JSONResponse(content={
            "object": "list",
            "data": models
        })
        
    except Exception as e:
        logger.error(f"Error listing embedding models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings/info")
async def get_embedding_info(
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)]
) -> JSONResponse:
    """
    Get detailed information about embedding system status
    """
    try:
        info = embedding_manager.get_embedding_info()
        return JSONResponse(content=info)
        
    except Exception as e:
        logger.error(f"Error getting embedding info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
