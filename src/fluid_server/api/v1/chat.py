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
    ContentBlock,
    TextContent,
    ImageUrlContent,
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

        # Build prompt from messages and extract images
        prompt, images = _build_prompt_and_extract_images(completion_request.messages)

        # Handle streaming
        if completion_request.stream:
            return StreamingResponse(
                _stream_chat_completion(llm, prompt, completion_request, request, images),
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
            images=images,
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


def _build_prompt_and_extract_images(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    """
    Build a prompt string from chat messages and extract images

    Args:
        messages: List of chat messages

    Returns:
        Tuple of (formatted prompt string, list of image URLs)
    """
    prompt = ""
    images = []
    
    for msg in messages:
        text_content = ""
        
        # Handle content as string or list of content blocks
        if isinstance(msg.content, str):
            text_content = msg.content
        elif isinstance(msg.content, list):
            # Process content blocks
            for block in msg.content:
                if isinstance(block, dict):
                    # Handle dict format from JSON
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "image_url":
                        image_url_obj = block.get("image_url", {})
                        if "url" in image_url_obj:
                            images.append(image_url_obj["url"])
                elif hasattr(block, 'type'):
                    # Handle Pydantic model format
                    if isinstance(block, TextContent):
                        text_content += block.text
                    elif isinstance(block, ImageUrlContent):
                        images.append(block.image_url["url"])
        
        # Add text content to prompt
        if text_content:
            if msg.role == "system":
                prompt += f"System: {text_content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {text_content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {text_content}\n\n"

    prompt += "Assistant: "
    return prompt, images


async def _stream_chat_completion(
    llm, prompt: str, request: ChatCompletionRequest, http_request: Request, images: list[str] | None = None
) -> AsyncIterator[str]:
    """
    Stream chat completion responses in SSE format with keepalive heartbeat and improved cancellation

    Args:
        llm: LLM runtime instance
        prompt: Formatted prompt
        request: Original request
        http_request: FastAPI request object for connection monitoring

    Yields:
        SSE formatted response chunks with keepalive heartbeat
    """
    import uuid

    token_stream = None
    stream_cancelled = False
    
    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Check initial connection state
        if await http_request.is_disconnected():
            logger.info("Client already disconnected before streaming started")
            return
        
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

        # Create token stream (this is an async generator, not awaitable)
        token_stream = llm.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            images=images,
        )
        
        # Buffer for sentence-based streaming
        sentence_buffer = ""
        last_flush_time = time.time()
        last_heartbeat_time = time.time()
        last_disconnect_check = time.time()
        token_count = 0
        
        # Stream tokens with heartbeat and sentence buffering
        async for token in token_stream:
            token_count += 1
            current_time = time.time()
            
            # Check for client disconnection more frequently during long generations
            # Check every 10 tokens or every 2 seconds, whichever comes first
            should_check_disconnect = (
                token_count % 10 == 0 or 
                (current_time - last_disconnect_check >= 2.0)
            )
            
            if should_check_disconnect:
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected after {token_count} tokens, cancelling stream")
                    stream_cancelled = True
                    break
                last_disconnect_check = current_time
                
            # Add token to sentence buffer
            sentence_buffer += token
            
            # Send heartbeat if needed (every 20 seconds)
            if current_time - last_heartbeat_time >= 20.0:
                # Check connection before sending heartbeat
                if await http_request.is_disconnected():
                    logger.info("Client disconnected during heartbeat check, cancelling stream")
                    stream_cancelled = True
                    break
                    
                yield ": keepalive\n\n"
                last_heartbeat_time = current_time
            
            # Check if we should flush the buffer
            should_flush = (
                # Found sentence boundary
                any(punct in sentence_buffer for punct in ['.', '!', '?', '\n']) or
                # Buffer timeout (1.5 seconds since last flush)
                (current_time - last_flush_time >= 1.5 and sentence_buffer.strip()) or
                # Force flush every 50 tokens to prevent excessive buffering
                token_count % 50 == 0
            )
            
            if should_flush:
                # Check connection before sending chunk
                if await http_request.is_disconnected():
                    logger.info("Client disconnected during chunk send, cancelling stream")
                    stream_cancelled = True
                    break
                
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
        
        # Only send completion if stream wasn't cancelled
        if not stream_cancelled:
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
            
            logger.info(f"Stream completed successfully with {token_count} tokens")
        else:
            logger.info(f"Stream cancelled after {token_count} tokens")

    except asyncio.CancelledError:
        logger.info("Stream cancelled by client")
        stream_cancelled = True
        # Don't re-raise, just end the stream gracefully
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        # Only send error response if client is still connected
        try:
            if not (stream_cancelled or await http_request.is_disconnected()):
                error_msg = {"error": str(e)}
                yield f"data: {json.dumps(error_msg)}\n\n"
        except Exception:
            logger.error("Failed to send error response to client")
    finally:
        # Attempt to clean up the token stream
        if token_stream is not None:
            try:
                # If the stream has a close method, call it
                if hasattr(token_stream, 'aclose'):
                    await token_stream.aclose()
                elif hasattr(token_stream, 'close'):
                    token_stream.close()
            except Exception as e:
                logger.debug(f"Error closing token stream: {e}")
                
        logger.debug("Stream cleanup completed")
