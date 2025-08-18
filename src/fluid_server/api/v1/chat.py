"""
OpenAI-compatible chat completions endpoint
"""
from typing import AsyncIterator, Annotated
import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from ...models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage
)
from ...managers.runtime_manager import RuntimeManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.state.runtime_manager


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    completion_request: ChatCompletionRequest,
    runtime_manager: Annotated[RuntimeManager, Depends(get_runtime_manager)]
):
    """
    OpenAI-compatible chat completions endpoint
    
    Args:
        completion_request: Chat completion request
        runtime_manager: Runtime manager instance (injected)
        
    Returns:
        Chat completion response or streaming response
    """
    try:
        # Verify the requested model exists
        if completion_request.model not in runtime_manager.available_models.get("llm", []):
            available_models = runtime_manager.available_models.get("llm", [])
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{completion_request.model}' not found. Available models: {available_models}"
            )
        
        # Get LLM runtime for the specific model
        llm = await runtime_manager.get_llm(completion_request.model)
        
        # Build prompt from messages
        prompt = _build_prompt(completion_request.messages)
        
        # Handle streaming
        if completion_request.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    llm,
                    prompt,
                    completion_request
                ),
                media_type="text/event-stream"
            )
        
        # Non-streaming generation
        response_text = await llm.generate(
            prompt=prompt,
            max_tokens=completion_request.max_tokens or 512,
            temperature=completion_request.temperature or 0.7,
            top_p=completion_request.top_p or 0.95
        )
        
        # Build response
        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        )
        
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _build_prompt(messages: list[ChatMessage]) -> str:
    """
    Build a prompt string from chat messages
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
    
    prompt += "Assistant: "
    return prompt


async def _stream_chat_completion(
    llm,
    prompt: str,
    request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """
    Stream chat completion responses in SSE format
    
    Args:
        llm: LLM runtime instance
        prompt: Formatted prompt
        request: Original request
        
    Yields:
        SSE formatted response chunks
    """
    import uuid
    
    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Send initial chunk
        initial_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None
            )]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Stream tokens
        async for token in llm.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95
        ):
            chunk = ChatCompletionStreamResponse(
                id=response_id,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason=None
                )]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Send final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_msg = {"error": str(e)}
        yield f"data: {json.dumps(error_msg)}\n\n"