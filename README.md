# Fluid Server - AI server for your Desktop Apps

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference)

The goal is to bring a portable, packaged OpenAI-like server for any desktop application to integrate with, providing the most optimal model configurations for each chipset. We prioritize AI accelerators where possible; where they're not available, the client will prefer GPU-based execution. Currently, most AI accelerators have a strict limit on the context window for LLMs, so we will focus on GPU-based execution with native runtimes in the meantime.

Designed to bundle into a single binary for easy integration into existing desktop applications.

We are starting with support for the OpenVINO backend and will eventually move into Qualcomm, then AMD. For Mac-related solutions, please see [FluidAudio](https://github.com/FluidInference/FluidAudio)

## Quick Start

### Prerequisites

- Windows 10/11
- Python 3.10+ with `uv` package manager
- OpenVINO 2025.2.0+ runtime
- 8GB+ RAM (16GB recommended for 8B models)

### Development

1. Install dependencies:

```powershell
uv sync
```

1. Run the development server:

```powershell
uv run python src/main.py
```

1. Test endpoints:

- <http://localhost:8080> - Welcome page
- <http://localhost:8080/health> - Health check with OpenVINO status
- <http://localhost:8080/docs> - Interactive API documentation
- <http://localhost:8080/v1/models> - List models (mock)
- POST <http://localhost:8080/v1/test> - Test OpenVINO operation

### Build Executable

Run the build script:

```powershell
.\build.ps1
```

This creates `dist/fluid-server.exe` (~65MB with OpenVINO bundled).

### Test Executable

Quick test with the provided script:

```powershell
.\test_exe.ps1
```

Or run manually:

```powershell
.\dist\fluid-server.exe
```

Command-line options:

```powershell
.\dist\fluid-server.exe --host 127.0.0.1 --port 8080
```

## Example Usage

### Testing with curl

```powershell
# Check server health
curl http://localhost:8080/health

# Chat completion (non-streaming)
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model": " qwen3-8b-int8-ov", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'
```

### Integration with OpenAI SDK

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"  # Can be anything for local server
)

# Use like regular OpenAI
response = client.chat.completions.create(
    model=" qwen3-8b-int8-ov",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming supported
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Integration with .NET Application

```csharp
// Use with OpenAI SDK for .NET
var client = new OpenAIClient(
    new Uri("http://localhost:8080/v1"),
    new ApiKeyCredential("local")
);

var response = await client.GetChatCompletionsAsync(
    " qwen3-8b-int8-ov",
    new ChatCompletionsOptions {
        Messages = { new ChatRequestUserMessage("Hello!") },
        MaxTokens = 100
    }
);
```

### FAQ

### Why Python?

Very valid question. It's just the easiest to support. Most ML work is done in Python, so it's the best supported by all the various runtimes we want to support. And with PyInstaller, being able to bundle it into a single .exe file is very helpful.

C++ and Rust are also other options we have considered, but they will require more investment and the team isn't familiar enough with Rust to make that jump. We may end up building a C++ server later on as well, but we want to avoid any heavy lifting on the inference side as much as possible.

Solutions like `uv`, `ty`, `fastapi` and `Pydantic` have made Python much more manageable as well.

### Why not just llama.cpp or whisper.cpp?

Solid options, but the goal is to support other runtimes and model formats beyond GGML ones. We want to best leverage AI accelerators available on various devices, and this is the simplest way to achieve that.

## Acknowledgements

Built using `ty`, `FastAPI`, `Pydantic` and various other AI libraries.

`OpenVINO`, `Qualcomm QNN` (and all their other work).

[**SearchSavior/OpenArc**](https://github.com/SearchSavior/OpenArc) - for the idea!
