"""
OpenAI-compatible API models with full type hints
"""

import time
import uuid
from typing import Any, Union

from pydantic import BaseModel, Field


# ============== Multimodal Content Models ==============
class TextContent(BaseModel):
    """Text content block in a chat message"""
    
    type: str = Field("text", description="Content type (always 'text')")
    text: str = Field(..., description="The text content")


class ImageUrlContent(BaseModel):
    """Image URL content block in a chat message"""
    
    type: str = Field("image_url", description="Content type (always 'image_url')")
    image_url: dict[str, str] = Field(..., description="Image URL object containing 'url' field")


# Union type for content blocks
ContentBlock = Union[TextContent, ImageUrlContent]


# ============== Chat Completion Models ==============
class ChatMessage(BaseModel):
    """Chat message in a conversation"""

    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: Union[str, list[ContentBlock]] = Field(..., description="The content of the message (string or array of content blocks)")
    name: str | None = Field(None, description="Optional name of the message author")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    model: str = Field(..., description="ID of the model to use")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float | None = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float | None = Field(0.95, ge=0, le=1, description="Nucleus sampling probability")
    n: int | None = Field(1, ge=1, description="Number of completions to generate")
    stream: bool | None = Field(False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    max_tokens: int | None = Field(512, description="Maximum tokens to generate")
    presence_penalty: float | None = Field(0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: float | None = Field(0, ge=-2, le=2, description="Frequency penalty")
    user: str | None = Field(None, description="Unique identifier for the end-user")
    seed: int | None = Field(None, description="Random seed for reproducibility")


class ChatCompletionResponseChoice(BaseModel):
    """A choice in a chat completion response"""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: dict[str, int] | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A choice in a streaming chat completion response"""

    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


# ============== Audio Transcription Models ==============
class TranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription request"""

    file: bytes = Field(..., description="Audio file to transcribe")
    model: str = Field("whisper-1", description="Model to use for transcription")
    language: str | None = Field(None, description="Language of the audio")
    prompt: str | None = Field(None, description="Optional prompt to guide transcription")
    response_format: str | None = Field(
        "json", description="Response format (json, text, srt, vtt)"
    )
    temperature: float | None = Field(0, description="Sampling temperature")


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response"""

    text: str = Field(..., description="Transcribed text")
    language: str | None = Field(None, description="Detected language")
    duration: float | None = Field(None, description="Audio duration in seconds")
    segments: list[dict[str, Any]] | None = Field(
        None, description="Transcription segments with timestamps"
    )


# ============== Model Information ==============
class ModelInfo(BaseModel):
    """Information about an available model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    model_type: str | None = Field(None, description="Type of model (llm, whisper, embedding)")


class ModelsResponse(BaseModel):
    """List of available models"""

    object: str = "list"
    data: list[ModelInfo]


# ============== Health/Status Models ==============
class HealthStatus(BaseModel):
    """Server health status"""

    status: str = Field(..., description="Overall health status")
    models_loaded: dict[str, bool] = Field(default_factory=dict, description="Loaded model status")
    device: str = Field(..., description="Current device")
    available_devices: list[str] = Field(default_factory=list, description="Available devices")
    memory_usage_gb: float | None = Field(None, description="Current memory usage")
    warmup_status: dict[str, Any] | None = Field(None, description="Model warm-up status")
    version: str = Field(..., description="Server version")
