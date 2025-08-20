"""
OpenAI-compatible API models with full type hints
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import time
import uuid


# ============== Chat Completion Models ==============
class ChatMessage(BaseModel):
    """Chat message in a conversation"""

    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the message author")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, ge=0, le=1, description="Nucleus sampling probability")
    n: Optional[int] = Field(1, ge=1, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0, ge=-2, le=2, description="Frequency penalty")
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ChatCompletionResponseChoice(BaseModel):
    """A choice in a chat completion response"""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionStreamChoice(BaseModel):
    """A choice in a streaming chat completion response"""

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============== Audio Transcription Models ==============
class TranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription request"""

    file: bytes = Field(..., description="Audio file to transcribe")
    model: str = Field("whisper-1", description="Model to use for transcription")
    language: Optional[str] = Field(None, description="Language of the audio")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    response_format: Optional[str] = Field(
        "json", description="Response format (json, text, srt, vtt)"
    )
    temperature: Optional[float] = Field(0, description="Sampling temperature")


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response"""

    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    segments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Transcription segments with timestamps"
    )


# ============== Model Information ==============
class ModelInfo(BaseModel):
    """Information about an available model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    model_type: Optional[str] = Field(None, description="Type of model (llm, whisper, embedding)")


class ModelsResponse(BaseModel):
    """List of available models"""

    object: str = "list"
    data: List[ModelInfo]


# ============== Health/Status Models ==============
class HealthStatus(BaseModel):
    """Server health status"""

    status: str = Field(..., description="Overall health status")
    models_loaded: Dict[str, bool] = Field(default_factory=dict, description="Loaded model status")
    device: str = Field(..., description="Current device")
    available_devices: List[str] = Field(default_factory=list, description="Available devices")
    memory_usage_gb: Optional[float] = Field(None, description="Current memory usage")
    warmup_status: Optional[Dict[str, Any]] = Field(None, description="Model warm-up status")
    version: str = Field(..., description="Server version")
