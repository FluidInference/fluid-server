# Fluid Server - AI server integrated into your Windows Apps

** THIS IS STILL UNDER DEVELOPMENT **

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference)

The goal is to bring a portable, packaged OpenAI-like server for any desktop application to integrate with, providing the most optimal model configurations for each chipset. We prioritize AI accelerators where possible.

The goal is to eventually provide all the necessary features like LLM, Transcription, Text to Speech, Speaker diarization, VAD, etc...  

Designed to bundle into a single binary for easy integration into existing desktop applications.



**Currently Supported NPU Runtimes:**
- **Intel NPU** via OpenVINO backend
- **Qualcomm NPU** via QNN (Snapdragon X Elite)

The server automatically detects your model format and selects the appropriate runtime for optimal performance. For Mac-related solutions, please see [FluidAudio](https://github.com/FluidInference/FluidAudio)

We built this because of the lack of support for Windows because of the fragmentation in support across Windows desktops and there doesn't seem to be a standard yet for running inference locally across all the chipsets, especially on AI accelerators. 

## NPU Support

Fluid Server supports multiple NPU runtimes for optimal performance on different hardware:

### Intel NPU (OpenVINO)
- **Models**: Uses OpenVINO IR format (.xml/.bin files)
- **Location**: `models/whisper/whisper-large-v3-turbo-fp16-ov-npu/`
- **Performance**: Optimized for Intel NPU and integrated graphics

### Qualcomm NPU (QNN)
- **Models**: Uses ONNX format with device-specific compilation
- **Location**: `models/whisper/whisper-large-v3-turbo-qnn/snapdragon-x-elite/`
- **Performance**: 16x+ real-time transcription on Snapdragon X Elite
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

This creates `dist/fluid-server.exe` (~276MB with OpenVINO + QNN bundled).

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

Very valid question. It's just the easiest to support. Most ML work is done in Python, so it's the best supported by all the various runtimes we want to support. And with PyInstaller, being able to bundle it into a single .exe file is very helpful.

C++ and Rust are also other options we have considered, but they will require more investment and the team isn't familiar enough with Rust to make that jump. We may end up building a C++ server later on as well, but we want to avoid any heavy lifting on the inference side as much as possible.

Solutions like `uv`, `ty`, `fastapi` and `Pydantic` have made Python much more manageable as well.

### Why not just llama.cpp or whisper.cpp?

Solid options, but the goal is to support other runtimes and model formats beyond GGML ones. We want to best leverage AI accelerators available on various devices, and this is the simplest way to achieve that.

## Acknowledgements

Built using `ty`, `FastAPI`, `Pydantic`, `ONNX Runtime`, `OpenAI Whisper` and various other AI libraries.

**Runtime Technologies:**
- `OpenVINO` - Intel NPU and GPU acceleration
- `Qualcomm QNN` - Snapdragon NPU optimization with HTP backend
- `ONNX Runtime` - Cross-platform AI inference

[**SearchSavior/OpenArc**](https://github.com/SearchSavior/OpenArc) - for the idea!
