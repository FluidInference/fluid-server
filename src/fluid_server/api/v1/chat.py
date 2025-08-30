"""
OpenAI-compatible chat completions endpoint
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...managers.runtime_manager import RuntimeManager
from ...models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.state.runtime_manager


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    completion_request: ChatCompletionRequest,
    runtime_manager: Annotated[RuntimeManager, Depends(get_runtime_manager)],
    request: Request,
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
        # Check if the model is in available models OR if it can be downloaded
        available_llm_models = runtime_manager.available_models.get("llm", [])
        
        # Import the constants to check if model can be downloaded
        from ...utils.model_downloader import DEFAULT_MODEL_REPOS
        model_can_be_downloaded = completion_request.model in DEFAULT_MODEL_REPOS.get("llm", {})
        
        if completion_request.model not in available_llm_models and not model_can_be_downloaded:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{completion_request.model}' not found and not available for download. Available models: {available_llm_models}",
            )

        # Check if model is still downloading or loading
        model_key = f"llm:{completion_request.model}"
        if model_key in runtime_manager.download_status:
            status = runtime_manager.download_status[model_key]
            if status == "downloading":
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{completion_request.model}' is currently downloading. Please try again later.",
                )
            elif status == "loading":
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{completion_request.model}' is currently loading (this can take 1-2 minutes). Please try again shortly.",
                )
            elif status == "failed":
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{completion_request.model}' failed to load. Please check logs.",
                )

        # Get LLM runtime for the specific model with error recovery
        try:
            llm = await runtime_manager.get_llm(completion_request.model)
        except RuntimeError as e:
            logger.warning(f"LLM model not available: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model '{completion_request.model}' is temporarily unavailable. Please try again later.",
            ) from e

        # Build prompt from messages
        prompt = _build_prompt(completion_request.messages)

        # Handle streaming
        if completion_request.stream:
            return StreamingResponse(
                _stream_chat_completion(llm, prompt, completion_request, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        # Non-streaming generation
        response_text = await llm.generate(
            prompt=prompt,
            max_tokens=completion_request.max_tokens or 512,
            temperature=completion_request.temperature or 0.7,
            top_p=completion_request.top_p or 0.95,
        )

        # Build response
        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        )

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
    llm, prompt: str, request: ChatCompletionRequest, http_request: Request
) -> AsyncIterator[str]:
    """
    Stream chat completion responses in SSE format with keepalive heartbeat

    Args:
        llm: LLM runtime instance
        prompt: Formatted prompt
        request: Original request
        http_request: FastAPI request object for connection monitoring

    Yields:
        SSE formatted response chunks with keepalive heartbeat
    """
    import uuid

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Send initial chunk
        initial_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0, delta={"role": "assistant", "content": ""}, finish_reason=None
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # Create token stream and heartbeat stream
        token_stream = llm.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
        )
        
        # Buffer for sentence-based streaming
        sentence_buffer = ""
        last_flush_time = time.time()
        last_heartbeat_time = time.time()
        
        # Stream tokens with heartbeat and sentence buffering
        async for token in token_stream:
            # Check for client disconnection
            if await http_request.is_disconnected():
                logger.info("Client disconnected, stopping stream")
                break
                
            # Add token to sentence buffer
            sentence_buffer += token
            current_time = time.time()
            
            # Send heartbeat if needed (every 20 seconds)
            if current_time - last_heartbeat_time >= 20.0:
                yield ": keepalive\n\n"
                last_heartbeat_time = current_time
            
            # Check if we should flush the buffer
            should_flush = (
                # Found sentence boundary
                any(punct in sentence_buffer for punct in ['.', '!', '?', '\n']) or
                # Buffer timeout (1.5 seconds since last flush)
                (current_time - last_flush_time >= 1.5 and sentence_buffer.strip())
            )
            
            if should_flush:
                # Send buffered content
                chunk = ChatCompletionStreamResponse(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0, delta={"content": sentence_buffer}, finish_reason=None
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                
                sentence_buffer = ""
                last_flush_time = current_time
        
        # Flush any remaining content
        if sentence_buffer.strip():
            chunk = ChatCompletionStreamResponse(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0, delta={"content": sentence_buffer}, finish_reason=None
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Send final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=request.model,
            choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_msg = {"error": str(e)}
        yield f"data: {json.dumps(error_msg)}\n\n"
