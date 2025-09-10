# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fluid Server is an OpenAI-compatible API server designed to run AI models with OpenVINO backend on Windows. The server provides REST endpoints for chat completions, audio transcription, text embeddings, and vector storage with model management capabilities including automatic downloading, caching, and memory-efficient runtime switching.

## Development Commands

### Setup and Dependencies
```powershell
# Install dependencies
uv sync

# Install development dependencies  
uv add --dev ty
```

### Running the Server
```powershell
# Development mode with auto-reload
uv run python -m fluid_server --reload

# Development mode with custom options
uv run python -m fluid_server --model-path ./models

# Using the convenience script
.\scripts\start_server.ps1
```

### Type Checking
```powershell
# Run type check
.\scripts\typecheck.ps1
# Or directly: uv run ty
```

### Building Executable
```powershell
# Build standalone .exe with PyInstaller
.\scripts\build.ps1
```

### Linting and Formatting
```powershell
# Run code formatting and linting with ruff
uv run ruff format .
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Testing
```powershell
# Test the built executable
.\scripts\test_exe.ps1

# Test with actual models
.\scripts\test_with_models.ps1

# Manual testing endpoints
curl http://localhost:8080/health
curl http://localhost:8080/v1/models

# Kill server if needed
.\scripts\kill_server.ps1
```

## Architecture Overview

### Core Components

**Application Structure (`src/fluid_server/`)**:
- `app.py` - FastAPI application factory with CORS, exception handling, and dependency injection
- `config.py` - Server configuration dataclass with defaults for model paths, device selection, and runtime parameters
- `__main__.py` - CLI entry point with argument parsing and uvicorn server startup

**Runtime Management (`managers/`)**:
- `RuntimeManager` - Central coordinator supporting both LLM and Whisper models loaded simultaneously
- Automatic model downloading via HuggingFace Hub with progress tracking
- Background warm-up process for faster first requests
- Idle cleanup with configurable timeout to manage memory usage

**Model Runtimes (`runtimes/`)**:
- `BaseRuntime` - Abstract base class with load/unload lifecycle and idle tracking
- `OpenVINOLLMRuntime` - LLM inference using OpenVINO GenAI
- `OpenVINOWhisperRuntime` - Audio transcription using OpenVINO Whisper models
- `LlamaCppRuntime` - Alternative LLM backend using llama.cpp
- `QNNWhisperRuntime` - ARM64-specific Whisper backend using Qualcomm Neural Network SDK

**API Endpoints (`api/`)**:
- `v1/chat.py` - OpenAI-compatible chat completions with streaming support
- `v1/audio.py` - Audio transcription endpoints
- `v1/models.py` - Model listing and management
- `v1/embeddings.py` - Text and multimodal embeddings generation
- `v1/vector_store.py` - LanceDB vector database operations
- `health.py` - Health check with OpenVINO status

**Storage and Embeddings (`storage/`, `managers/`)**:
- `LanceDBClient` - Vector database client for multimodal storage
- `EmbeddingManager` - Manages text and multimodal embedding models
- `BaseEmbeddingRuntime` - Abstract base for embedding runtimes
- `OpenVINOEmbeddingRuntime` - Text embeddings using OpenVINO backend
- `WhisperEmbeddingRuntime` - Audio embeddings for semantic search

### Model Organization

Expected directory structure under `model_path`:
```
models/
├── llm/
│   ├── qwen3-8b-int8-ov/          # LLM model directories
│   └── phi-4-mini/
├── whisper/
│   ├── whisper-tiny/              # Whisper model directories  
│   └── whisper-large-v3/
├── embeddings/                    # Text embedding models
│   └── sentence-transformers/
└── cache/                         # Compiled model cache
```

**Data Directory Structure** under `data_root`:
```
data/
├── models/                        # Model files (see above)
├── cache/                         # Runtime cache and compiled models
└── databases/                     # LanceDB vector databases
    └── embeddings/                # Default embeddings database
```

### Key Design Patterns

- **Multi-Runtime Architecture**: Separate LLM, Whisper, and embedding models can be loaded simultaneously for optimal performance
- **Background Model Management**: Automatic downloading, warm-up, and idle cleanup with progress tracking
- **Device-Specific Runtime Selection**: QNN backend automatically used on ARM64, OpenVINO/llama.cpp on other architectures
- **OpenAI Compatibility**: Request/response formats match OpenAI API for drop-in replacement
- **PyInstaller Ready**: Handles frozen executable detection for simplified deployment
- **Graceful Degradation**: Falls back to alternative runtimes if primary backend unavailable
- **Vector Database Integration**: LanceDB provides multimodal storage for embeddings, text, and metadata
- **Memory Management**: Configurable idle timeout and memory limits prevent resource exhaustion

### Configuration

Server behavior controlled via `ServerConfig` dataclass:
- Model paths and device selection (CPU/GPU/NPU)
- Memory limits and idle timeout settings
- Generation defaults (max_tokens, temperature, top_p)
- Feature flags (warm_up, idle cleanup, embeddings)
- Embedding model configuration and vector database settings
- Default models: `qwen3-8b-int4-ov` (LLM), `whisper-large-v3-turbo-fp16-ov-npu` (Whisper), `sentence-transformers/all-MiniLM-L6-v2` (embeddings)

Command-line arguments override configuration defaults. The server validates model availability on startup and provides informative warnings for missing models.

## Important Development Notes

- **Python Version**: Project requires exactly Python 3.10 (`==3.10.*`)
- **Architecture Support**: QNN backend only available on ARM64 with conditional imports to prevent PyInstaller issues
- **Model Management**: Use `RuntimeManager` and `EmbeddingManager` for all model operations - they handle downloading, loading, and resource management
- **Memory Optimization**: Prefer the multi-runtime architecture over single model switching for production use
- **Error Handling**: All runtimes implement graceful loading/unloading with proper resource cleanup
- **Testing Architecture**: ARM64 systems test QNN backend, x64 Intel systems test OpenVINO backend
- **Vector Database**: LanceDB integration requires embedding models to be loaded before vector operations
- **Multimodal Support**: Text embeddings via sentence-transformers, image embeddings via CLIP, audio embeddings via Whisper
- **Build System**: PyInstaller creates single-file executable with architecture detection and runtime selection

## Code Style Guidelines

**Type Hints and Imports**:
- Use absolute imports from `fluid_server` package
- Required type hints for all function signatures
- Use `Path` objects for filesystem paths, `Optional[T]` for nullable types

**Async and Threading**:
- Use `async/await` for I/O operations
- Run OpenVINO inference in ThreadPoolExecutor for CPU-bound operations
- Handle device selection gracefully (CPU/GPU/NPU)

**Error Handling and Logging**:
- Use module-level `logger = logging.getLogger(__name__)`
- Log errors with `logger.error()` before raising exceptions
- Use specific exception types, avoid generic `Exception`

**FastAPI Patterns**:
- Use Pydantic models for request/response validation
- Leverage dependency injection for shared state (managers, clients)
- Implement proper lifespan management for resource cleanup