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

### Testing
```powershell
# Test the built executable
.\scripts\test_exe.ps1

# Manual testing endpoints
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

## Architecture Overview

### Core Components

**Application Structure (`src/fluid_server/`)**:
- `app.py` - FastAPI application factory with CORS, exception handling, and dependency injection
- `config.py` - Server configuration dataclass with defaults for model paths, device selection, and runtime parameters
- `__main__.py` - CLI entry point with argument parsing and uvicorn server startup

**Runtime Management (`managers/`)**:
- `RuntimeManager` - Central coordinator that loads/unloads models on-demand for memory efficiency
- Supports single model in memory at a time with configurable idle timeout
- Handles model discovery and validation on startup

**Model Runtimes (`runtimes/`)**:
- `BaseRuntime` - Abstract base class for all model backends
- `OpenVINOLLMRuntime` - LLM inference using OpenVINO GenAI
- `OpenVINOWhisperRuntime` - Audio transcription using OpenVINO Whisper models

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

- **Single Model Runtime**: Only one model loaded at a time to optimize memory usage
- **Warm-up by Default**: Models loaded on startup by default (disable with `--no-warm-up`)
- **OpenAI Compatibility**: Request/response formats match OpenAI API for drop-in replacement
- **PyInstaller Ready**: Handles frozen executable detection for simplified deployment

### Configuration

Server behavior controlled via `ServerConfig` dataclass:
- Model paths and device selection (CPU/GPU/NPU)
- Memory limits and idle timeout settings
- Generation defaults (max_tokens, temperature, top_p)
- Feature flags (warm_up, idle cleanup)

Command-line arguments override configuration defaults. The server validates model availability on startup and provides informative warnings for missing models.