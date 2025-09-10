# Development Guide

This guide covers everything you need to know for developing with Fluid Server, from initial setup to building and testing.

## Prerequisites

### System Requirements
- **OS**: Windows 10/11
- **Python**: 3.10+ with `uv` package manager
- **Memory**: 8GB+ RAM (16GB recommended for 8B models)
- **Storage**: 10GB+ free space for models

### Hardware-Specific Requirements

#### Intel NPU Support
- **Runtime**: OpenVINO 2025.2.0+ runtime
- **Hardware**: Intel Arc graphics or Intel NPU

#### Qualcomm NPU Support
- **Runtime**: ONNX Runtime QNN (bundled with dependencies)
- **Hardware**: Snapdragon X Elite device with HTP (Hexagon Tensor Processor)

### Installing Prerequisites

#### Install uv Package Manager
```powershell
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.ps1 | powershell
```

#### Install OpenVINO (for Intel Devices)
```powershell
# Download and install OpenVINO runtime from Intel's website
# https://docs.openvino.ai/2025/get-started/install-openvino.html
```

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/FluidInference/fluid-server.git
cd fluid-server
```

### 2. Install Dependencies
```powershell
# Install all dependencies including development tools
uv sync

# Install development dependencies separately (if needed)
uv add --dev ty
```

### 3. Verify Installation
```powershell
# Check that dependencies are installed correctly
uv run python -c "import fluid_server; print('Setup successful')"
```

## Development Workflow

### Running the Development Server

#### Basic Development Mode
```powershell
# Run with auto-reload for development
uv run python -m fluid_server --reload
```

#### Development with Custom Options
```powershell
# Run with custom model path and debug logging
uv run python -m fluid_server --model-path ./models --log-level DEBUG --reload
```

#### Using Convenience Scripts
```powershell
# Use the convenience script
.\scripts\start_server.ps1
```

### Development Server Options
- `--reload` - Auto-reload on code changes (development only)
- `--model-path` - Custom path to model directory
- `--log-level` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--host` - Server host (default: 127.0.0.1)
- `--port` - Server port (default: 8080)

## Code Quality and Testing

### Type Checking
```powershell
# Run type checking with ty
.\scripts\typecheck.ps1

# Or run directly
uv run ty
```

### Code Formatting and Linting
```powershell
# Format code with ruff
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check --fix .
```

### Testing the Server

#### Development Testing
```powershell
# Test health endpoint
curl http://localhost:8080/health

# Test models endpoint
curl http://localhost:8080/v1/models

# Test basic chat completion
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"current\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'
```

#### Automated Testing Scripts
```powershell
# Test with actual models (requires model downloads)
.\scripts\test_with_models.ps1

# Kill server if needed
.\scripts\kill_server.ps1
```

## Building and Distribution

### Building the Executable

#### Standard Build
```powershell
# Build standalone .exe with PyInstaller
.\scripts\build.ps1
```

The build script creates `dist/fluid-server.exe` (approximately 276 MB with OpenVINO + QNN bundled).

#### Build Configuration
The build process:
1. Installs dependencies with `uv sync`
2. Runs type checking with `ty`
3. Creates executable with PyInstaller
4. Includes all necessary runtime libraries

### Testing the Built Executable

#### Quick Test
```powershell
# Test the built executable
.\scripts\test_exe.ps1
```

#### Manual Testing
```powershell
# Run the executable directly
.\dist\fluid-server.exe

# Run with custom options
.\dist\fluid-server.exe --host 127.0.0.1 --port 8080 --log-level DEBUG
```

## Project Structure

### Source Code Organization
```
src/fluid_server/
├── __main__.py              # CLI entry point
├── app.py                   # FastAPI application factory
├── config.py                # Server configuration
├── api/                     # API endpoints
│   ├── v1/
│   │   ├── chat.py         # Chat completions
│   │   ├── audio.py        # Audio transcription
│   │   ├── models.py       # Model management
│   │   └── embeddings.py   # Text embeddings
│   └── health.py           # Health checks
├── managers/               # Core business logic
│   ├── runtime_manager.py  # Model loading/unloading
│   └── embedding_manager.py # Embedding generation
├── runtimes/              # Model runtime implementations
│   ├── base.py            # Abstract base runtime
│   ├── openvino_llm.py    # OpenVINO LLM runtime
│   ├── openvino_whisper.py # OpenVINO Whisper runtime
│   ├── llamacpp.py        # Llama.cpp runtime
│   └── qnn_whisper.py     # QNN Whisper runtime
├── storage/               # Data persistence
│   └── lancedb_client.py  # LanceDB vector storage
└── utils/                 # Utilities
    ├── model_utils.py     # Model discovery/downloading
    └── platform_utils.py  # Platform detection
```

### Model Directory Structure
```
models/
├── llm/                   # Language models
│   ├── qwen3-8b-int8-ov/  # OpenVINO LLM models
│   └── phi-4-mini/        # Additional LLM models
├── whisper/               # Audio transcription models
│   ├── whisper-large-v3-turbo-ov-npu/    # OpenVINO Whisper
│   ├── whisper-large-v3-turbo-qnn/       # QNN Whisper
│   └── whisper-tiny/                     # Smaller models
├── embeddings/            # Text embedding models
│   └── sentence-transformers_all-MiniLM-L6-v2/
└── cache/                 # Compiled model cache
```

## Development Configuration

### Environment Variables
```powershell
# Set development environment variables
$env:PYTHONPATH = "src"
$env:FLUID_LOG_LEVEL = "DEBUG"
$env:FLUID_MODEL_PATH = "./models"
```

### IDE Configuration

#### VS Code Settings
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false,
    "python.formatting.provider": "black",
    "python.formatting.blackPath": ".venv/Scripts/black.exe"
}
```

## Debugging and Troubleshooting

### Common Development Issues

#### Module Import Errors
```powershell
# Ensure PYTHONPATH includes src directory
$env:PYTHONPATH = "src"
uv run python -m fluid_server
```

#### Model Loading Issues
```powershell
# Run with debug logging to see model loading details
uv run python -m fluid_server --log-level DEBUG
```

#### Port Already in Use
```powershell
# Kill any existing server processes
.\scripts\kill_server.ps1

# Or use a different port
uv run python -m fluid_server --port 8081
```

### Debug Logging
```python
import logging

# Enable debug logging for specific components
logging.getLogger("fluid_server.managers.runtime_manager").setLevel(logging.DEBUG)
logging.getLogger("fluid_server.runtimes").setLevel(logging.DEBUG)
```

### Performance Profiling
```powershell
# Run with performance profiling
uv run python -m cProfile -o profile_output.pstats -m fluid_server

# Analyze profile results
uv run python -c "import pstats; pstats.Stats('profile_output.pstats').sort_stats('cumulative').print_stats(20)"
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Format code with `ruff format`
- Ensure all linting checks pass with `ruff check`

### Commit Guidelines
- Use conventional commit messages
- Include relevant tests for new features
- Ensure all existing tests pass
- Update documentation for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with proper testing
4. Ensure all checks pass (linting, type checking, tests)
5. Submit a pull request with a clear description

## Advanced Development Topics

### Adding New Model Runtimes
1. Implement the `BaseRuntime` abstract class
2. Add the runtime to the `RuntimeManager`
3. Update configuration options
4. Add appropriate tests

### Extending API Endpoints
1. Create new endpoint modules in `api/v1/`
2. Register routes in the main application
3. Add request/response models using Pydantic
4. Include comprehensive error handling

### Performance Optimization
- Use async/await for I/O operations
- Implement connection pooling for external services
- Cache frequently accessed data
- Monitor memory usage with large models

This development guide provides the foundation for contributing to and extending Fluid Server. For specific implementation details, refer to the existing codebase and follow the established patterns.