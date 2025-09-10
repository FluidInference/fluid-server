# Fluid Server Documentation

This directory contains comprehensive documentation for the Fluid Server project, including compilation guides, feature documentation, and troubleshooting information.

## Documents

### [Compilation Guide](./compilation-guide.md)
Complete guide for building PyInstaller executables, including:
- Critical import fixes for PyInstaller compatibility
- Build performance optimization
- Common compilation issues and solutions
- Build environment setup requirements

### [GGUF Model Support](./GGUF-model-support.md)
Complete guide for using any GGUF model from HuggingFace Hub:
- Flexible model format support (repo, repo/file, legacy names)
- Popular model recommendations and quantization guidance
- Automatic download and caching system
- Performance optimization for different hardware configurations

## Quick Reference

### Build the Executable
```powershell
.\scripts\build.ps1
```

### Use Any GGUF Model
```bash
./fluid-server.exe --llm-model "unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
```

### Test Streaming
```bash
curl -X POST http://localhost:3847/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "current", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Architecture Overview

```
Fluid Server
├── FastAPI Application (app.py)
├── Runtime Manager (managers/)
│   ├── Model loading/unloading
│   └── Memory management
├── Model Runtimes (runtimes/)
│   ├── LlamaCpp (GGUF models)
│   └── OpenVINO (optimized models)
├── API Endpoints (api/)
│   ├── Chat completions (/v1/chat/completions)
│   ├── Model management (/v1/models)
│   └── Health checks (/health)
└── Utilities (utils/)
    ├── Model discovery
    ├── Model downloading
    └── Platform detection
```
