# Fluid Server - OpenAI-Compatible API Server

An OpenAI-compatible API server with OpenVINO backend for local LLM inference on Windows. Designed to replace direct Python integration with a clean REST API.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **OpenVINO Backend** - Hardware acceleration on CPU, GPU, NPU
- **Model Management** - Automatic downloading and caching from Hugging Face
- **Streaming Support** - Real-time token streaming for chat completions
- **Memory Optimization** - Automatic model unloading after idle timeout
- **PyInstaller Ready** - Bundle as single .exe file for deployment

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

2. Run the development server:
```powershell
uv run python src/main.py
```

3. Test endpoints:
- http://localhost:8080 - Welcome page
- http://localhost:8080/health - Health check with OpenVINO status
- http://localhost:8080/docs - Interactive API documentation
- http://localhost:8080/v1/models - List models (mock)
- POST http://localhost:8080/v1/test - Test OpenVINO operation

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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API info |
| `/health` | GET | Server health with device status |
| `/v1/models` | GET | List available models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |

## Project Structure

```
fluid-server-windows/
├── src/
│   └── main.py          # FastAPI application
├── build.ps1            # PyInstaller build script
├── pyproject.toml       # Dependencies
├── .gitignore
└── README.md
```

## Example Usage

### Testing with curl
```powershell
# Check server health
curl http://localhost:8080/health

# Chat completion (non-streaming)
curl -X POST http://localhost:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'
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
    model="qwen3-8b",
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
    "qwen3-8b",
    new ChatCompletionsOptions {
        Messages = { new ChatRequestUserMessage("Hello!") },
        MaxTokens = 100
    }
);
```

## Troubleshooting

### Build Issues

If PyInstaller fails:
1. Ensure all dependencies are installed: `uv sync`
2. Check Windows Defender isn't blocking the build
3. Try building with console mode first: Remove `--noconsole` from build.ps1
4. Clean build artifacts: `Remove-Item -Recurse build, dist, *.spec`

### Runtime Issues

If the .exe doesn't start:
1. Check Windows Event Viewer for errors
2. Run with console for debugging: Edit build.ps1 and remove `--noconsole`
3. Verify Visual C++ Redistributables are installed
4. Check if port 8080 is already in use: `netstat -an | findstr :8080`

## Next Steps

- Add real OpenVINO model loading
- Implement actual inference endpoints
- Add model management
- Configure for Windows Store packaging