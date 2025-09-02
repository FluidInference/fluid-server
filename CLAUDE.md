# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fluid Server is an OpenAI-compatible API server designed to run AI models with OpenVINO backend on Windows. The server provides REST endpoints for chat completions and audio transcription, with model management capabilities including automatic downloading, caching, and memory-efficient runtime switching.

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
- `health.py` - Health check with OpenVINO status

### Model Organization

Expected directory structure under `model_path`:
```
models/
├── llm/
│   ├──  qwen3-8b-int8-ov/          # LLM model directories
│   └── phi-4-mini/
├── whisper/
│   ├── whisper-tiny/      # Whisper model directories  
│   └── whisper-large-v3/
└── cache/                 # Compiled model cache
```

### Key Design Patterns

- **Dual Runtime Architecture**: Separate LLM and Whisper models can be loaded simultaneously for optimal performance
- **Background Model Management**: Automatic downloading, warm-up, and idle cleanup with progress tracking
- **Device-Specific Runtime Selection**: QNN backend automatically used on ARM64, OpenVINO/llama.cpp on other architectures
- **OpenAI Compatibility**: Request/response formats match OpenAI API for drop-in replacement
- **PyInstaller Ready**: Handles frozen executable detection for simplified deployment
- **Graceful Degradation**: Falls back to alternative runtimes if primary backend unavailable

### Configuration

Server behavior controlled via `ServerConfig` dataclass:
- Model paths and device selection (CPU/GPU/NPU)
- Memory limits and idle timeout settings
- Generation defaults (max_tokens, temperature, top_p)
- Feature flags (warm_up, idle cleanup)

Command-line arguments override configuration defaults. The server validates model availability on startup and provides informative warnings for missing models.

## Important Development Notes

- **Python Version**: Project requires exactly Python 3.10 (`==3.10.*`)
- **Architecture Support**: QNN backend only available on ARM64 with conditional imports to prevent PyInstaller issues
- **Model Management**: Use `RuntimeManager` for all model operations - it handles downloading, loading, and resource management
- **Memory Optimization**: Prefer the dual runtime architecture over single model switching for production use
- **Error Handling**: All runtimes implement graceful loading/unloading with proper resource cleanup
- Check the system architecture before making assumptions. ARM we test QNN, x64 intel we test openvino