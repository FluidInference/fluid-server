"""
OpenAI-compatible API models with full type hints
"""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============== Chat Completion Models ==============
class ChatMessage(BaseModel):
    """Chat message in a conversation"""

    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")
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


# ============== Embedding Models ==============
class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""

    input: str | list[str] = Field(..., description="Text input(s) to embed")
    model: str = Field(..., description="ID of the model to use")
    encoding_format: str | None = Field("float", description="Format to return embeddings in")
    dimensions: int | None = Field(None, description="Number of dimensions for embedding")
    user: str | None = Field(None, description="Unique identifier for the end-user")


class EmbeddingData(BaseModel):
    """Single embedding data point"""

    object: str = "embedding"
    embedding: list[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="Index of the embedding in the input list")


class EmbeddingUsage(BaseModel):
    """Usage information for embedding request"""

    prompt_tokens: int = Field(..., description="Number of tokens in the input")
    total_tokens: int = Field(..., description="Total number of tokens used")


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""

    object: str = "list"
    data: list[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    usage: EmbeddingUsage = Field(..., description="Usage statistics")


    input: str | dict[str, Any] = Field(..., description="Input data (text, image bytes, or audio bytes)")
    input_type: str = Field(..., description="Type of input: text, image, or audio")
    model: str = Field(..., description="ID of the model to use")
    encoding_format: str | None = Field("float", description="Format to return embeddings in")
    dimensions: int | None = Field(None, description="Number of dimensions for embedding")
    user: str | None = Field(None, description="Unique identifier for the end-user")


# ============== Vector Store Models ==============
class VectorStoreDocument(BaseModel):
    """Document for vector storage"""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    content_type: str = Field("text", description="Type of content")


class VectorStoreInsertRequest(BaseModel):
    """Request to insert documents into vector store"""

    collection: str = Field(..., description="Collection name")
    documents: list[VectorStoreDocument] = Field(..., description="Documents to insert")
    model: str | None = Field(None, description="Embedding model to use")


class VectorStoreInsertResponse(BaseModel):
    """Response for vector store insertion"""

    inserted_count: int = Field(..., description="Number of documents inserted")
    collection: str = Field(..., description="Collection name")
    ids: list[str] = Field(..., description="IDs of inserted documents")


class VectorStoreSearchRequest(BaseModel):
    """Request to search vector store"""

    collection: str = Field(..., description="Collection name")
    query: str = Field(..., description="Query text")
    query_type: Literal["text"] = Field("text", description="Type of query (text only)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    filter: str | None = Field(None, description="Optional filter condition")
    model: str | None = Field(None, description="Embedding model to use for query")


class VectorStoreSearchResult(BaseModel):
    """Single search result"""

    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] | None = Field(None, description="Document metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    content_type: str = Field(..., description="Type of content")


class VectorStoreSearchResponse(BaseModel):
    """Response for vector store search"""

    results: list[VectorStoreSearchResult] = Field(..., description="Search results")
    collection: str = Field(..., description="Collection name")
    query_type: Literal["text"] = Field("text", description="Type of query used")
    total_results: int = Field(..., description="Total number of results found")


class CollectionInfo(BaseModel):
    """Information about a vector store collection"""

    name: str = Field(..., description="Collection name")
    num_documents: int = Field(..., description="Number of documents in collection")
    embedding_dimension: int | None = Field(None, description="Embedding vector dimension")
    content_types: list[str] = Field(default_factory=list, description="Types of content stored")


class CollectionListResponse(BaseModel):
    """Response listing all collections"""

    collections: list[CollectionInfo] = Field(..., description="List of collections")
    total_collections: int = Field(..., description="Total number of collections")


class CreateCollectionRequest(BaseModel):
    """Request to create a new collection"""

    name: str = Field(..., description="Collection name")
    dimension: int = Field(..., description="Embedding vector dimension")
    content_type: str = Field("text", description="Primary content type for this collection")
    overwrite: bool = Field(False, description="Whether to overwrite existing collection")
