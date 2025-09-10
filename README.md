# Fluid Server — AI server for your Windows apps

**THIS PROJECT IS UNDER ACTIVE DEVELOPMENT**

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference)

A portable, packaged OpenAI-compatible server for Windows desktop applications. Provides optimal model configurations for each chipset with AI accelerator support.

## Features

**Core Capabilities**
- **LLM Chat Completions** - OpenAI-compatible API with streaming
- **Audio Transcription** - Whisper models with NPU acceleration
- **Text Embeddings** - Vector embeddings for search and RAG
- **Vector Database** - LanceDB integration for multimodal storage

**Hardware Acceleration**
- **Intel NPU** via OpenVINO backend
- **Qualcomm NPU** via QNN (Snapdragon X Elite)
- **Cross-platform** - Runtime detection

**Easy Integration**
- Single binary deployment
- OpenAI SDK compatible
- .NET, Python, Node.js support

## Quick Start

### 1. Download or Build

**Option A: Download Release**
- Download `fluid-server.exe` from [releases](https://github.com/FluidInference/fluid-server/releases)

**Option B: Run from Source**
```powershell
# Install dependencies and run
uv sync
uv run
```

### 2. Run the Server

```powershell
# Run with default settings
.\dist\fluid-server.exe

# Or with custom options
.\dist\fluid-server.exe --host 127.0.0.1 --port 8080
```

### 3. Test the API

- **Health Check**: http://localhost:8080/health
- **API Docs**: http://localhost:8080/docs
- **Models**: http://localhost:8080/v1/models

## Usage Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b-int8-ov", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Python Integration

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")

# Chat with streaming
for chunk in client.chat.completions.create(
    model="qwen3-8b-int8-ov",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

### Audio Transcription

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3-turbo-qnn"
```

## Documentation

📖 **Comprehensive Guides**
- [NPU Support Guide](docs/npu-support.md) - Intel & Qualcomm NPU configuration
- [Integration Guide](docs/integration-guide.md) - Python, .NET, Node.js examples
- [Development Guide](docs/development.md) - Setup, building, and contributing
- [LanceDB Integration](docs/lancedb.md) - Vector database and embeddings
- [GGUF Model Support](docs/GGUF-model-support.md) - Using any GGUF model
- [Compilation Guide](docs/compilation-guide.md) - Build system details

## FAQ

**Why Python?** Best ML ecosystem support and PyInstaller packaging.

**Why not llama.cpp?** We support multiple runtimes and AI accelerators beyond GGML.

## Acknowledgements

Built using `ty`, `FastAPI`, `Pydantic`, `ONNX Runtime`, `OpenAI Whisper`, and various other AI libraries.

**Runtime Technologies:**
- `OpenVINO` - Intel NPU and GPU acceleration
- `Qualcomm QNN` - Snapdragon NPU optimization with HTP backend
- `ONNX Runtime` - Cross-platform AI inference
