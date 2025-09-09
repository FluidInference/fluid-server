# Fluid Server — AI server for your Windows apps

**THIS PROJECT IS UNDER ACTIVE DEVELOPMENT**

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference)

The goal is to provide a portable, packaged OpenAI‑like server that any Windows desktop application can integrate with, offering optimal model configurations for each chipset. We prioritize AI accelerators where possible; for LLM inference we currently use llama.cpp.

We plan to provide features including LLM, transcription, text‑to‑speech, speaker diarization, VAD, and more.  

It is designed to bundle into a single binary for easy integration into existing desktop applications.

**Currently supported NPU runtimes for transcription:**
- **Intel NPU** via OpenVINO backend
- **Qualcomm NPU** via QNN (Snapdragon X Elite)

The server automatically detects your model format and selects the appropriate runtime for optimal performance. For macOS, see [FluidAudio](https://github.com/FluidInference/FluidAudio).

We built this due to fragmented support across Windows devices. There is no clear standard for running local inference across chipsets—especially on AI accelerators.

## NPU Support

Fluid Server supports multiple NPU runtimes for optimal performance on different hardware:

### Intel NPU (OpenVINO)
- **Models**: Uses OpenVINO IR format (.xml/.bin files)
- **Location**: `models/whisper/whisper-large-v3-turbo-fp16-ov-npu/`
- **Performance**: Optimized for Intel NPU and integrated graphics

### Qualcomm NPU (QNN)
- **Models**: Uses ONNX format with device-specific compilation
- **Location**: `models/whisper/whisper-large-v3-turbo-qnn/snapdragon-x-elite/`
- **Performance**: 16× real‑time transcription on Snapdragon X Elite
- **Hardware**: Snapdragon X Elite devices with HTP (Hexagon Tensor Processor)

## Quick Start

### Prerequisites

- Windows 10/11
- Python 3.10+ with `uv` package manager
- **For Intel NPU:** OpenVINO 2025.2.0+ runtime
- **For Qualcomm NPU:** ONNX Runtime QNN (bundled with dependencies)
- 8GB+ RAM (16GB recommended for 8B models)
- **Recommended:** Snapdragon X Elite device for optimal QNN performance

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

This creates `dist/fluid-server.exe` (approximately 276 MB with OpenVINO + QNN bundled).

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

# Chat completion (non‑streaming)
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model": "qwen3-8b-int8-ov", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'

# Audio transcription with QNN model
curl -X POST http://localhost:8080/v1/audio/transcriptions `
  -F "file=@audio.wav" `
  -F "model=whisper-large-v3-turbo-qnn" `
  -F "response_format=json"
```

### Integration with OpenAI SDK

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"  # Can be anything for local server
)

# Chat completion
response = client.chat.completions.create(
    model="qwen3-8b-int8-ov",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming supported
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# Audio transcription with QNN
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo-qnn",
        file=audio_file
    )
    print(transcript.text)
```

### Integration with .NET Application

```csharp
// Use with OpenAI SDK for .NET
var client = new OpenAIClient(
    new Uri("http://localhost:8080/v1"),
    new ApiKeyCredential("local")
);

var response = await client.GetChatCompletionsAsync(
    "qwen3-8b-int8-ov",
    new ChatCompletionsOptions {
        Messages = { new ChatRequestUserMessage("Hello!") },
        MaxTokens = 100
    }
);
```

### FAQ

### Why Python?

Good question. It is the easiest to support. Most ML work is done in Python, so it is the best supported across the runtimes we target. PyInstaller lets us bundle everything into a single .exe, which is very helpful.

C++ and Rust are options we have considered, but they require more investment, and the team is not yet familiar enough with Rust to make that jump. We may build a C++ server later, but we want to avoid heavy lifting on the inference side where possible.

Tools like `uv`, `ty`, `FastAPI`, and `Pydantic` have also made Python much more manageable.

### Why not just llama.cpp or whisper.cpp?

Those are solid options, but the goal is to support other runtimes and model formats beyond GGML. We want to leverage AI accelerators available on various devices, and this is the simplest way to achieve that.

## Acknowledgements

Built using `ty`, `FastAPI`, `Pydantic`, `ONNX Runtime`, `OpenAI Whisper`, and various other AI libraries.

**Runtime Technologies:**
- `OpenVINO` - Intel NPU and GPU acceleration
- `Qualcomm QNN` - Snapdragon NPU optimization with HTP backend
- `ONNX Runtime` - Cross-platform AI inference

[**SearchSavior/OpenArc**](https://github.com/SearchSavior/OpenArc) - for the idea!
