# Fluid Server Documentation

This directory contains comprehensive documentation for the Fluid Server project, including compilation guides, feature documentation, and troubleshooting information.

## Documents

### 📋 [Compilation Guide](./compilation-guide.md)
Complete guide for building PyInstaller executables, including:
- Critical import fixes for PyInstaller compatibility
- Build performance optimization
- Common compilation issues and solutions
- Build environment setup requirements

### 🚀 [Streaming Improvements](./streaming-improvements.md)
Documentation of streaming fixes that resolve client connection issues:
- SSE keepalive heartbeat implementation
- Progressive token streaming with async queues
- Sentence-based buffering and timeout handling
- Performance impact analysis and testing results

### 🤖 [GGUF Model Support](./GGUF-model-support.md)
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

## Key Improvements Made

### ✅ Fixed Client Connection Drops
- Added SSE keepalive heartbeat every 20 seconds
- Implemented progressive token streaming 
- Added sentence-based buffering with smart flushing
- Improved error handling and connection monitoring

### ✅ Resolved PyInstaller Import Issues
- Implemented try/except import pattern for development vs executable modes
- Documented all critical compilation considerations
- Provided multiple build methods and troubleshooting guides

### ✅ Enhanced Model Support
- Support for any GGUF model from HuggingFace without predefined mappings
- Flexible model identifier formats (repo, repo/file, legacy)
- Automatic download and caching with resume capability
- Hardware-optimized configurations (GPU/CPU, memory management)

### ✅ Improved Default Configuration
- Changed default port from 8080 to 3847 to avoid conflicts
- Optimized streaming parameters for better performance
- Enhanced logging and monitoring capabilities

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

## Deployment Ready Features

The current implementation includes all production-ready features:

- ✅ **Standalone Executable**: Complete PyInstaller build (309.8 MB)
- ✅ **OpenAI API Compatibility**: Drop-in replacement for OpenAI API
- ✅ **Streaming Support**: Fixed progressive streaming with keepalive
- ✅ **GGUF Model Support**: Use any quantized model from HuggingFace
- ✅ **GPU Acceleration**: Vulkan backend for LlamaCpp, OpenVINO optimization
- ✅ **Memory Management**: Single model in memory with idle timeout
- ✅ **Error Handling**: Graceful connection handling and recovery
- ✅ **Configuration**: Flexible command-line and environment configuration

## Support and Troubleshooting

### Common Issues
1. **Import Errors**: See [Compilation Guide](./compilation-guide.md#critical-import-issue-and-solution)
2. **Connection Drops**: See [Streaming Improvements](./streaming-improvements.md#solutions-implemented)
3. **Model Loading**: See [GGUF Model Support](./GGUF-model-support.md#error-handling)

### Getting Help
- Review the relevant documentation file for your issue
- Check the build logs and warnings
- Verify system requirements and dependencies
- Test in development mode before building executable

## Contributing

When making changes to the Fluid Server:

1. **Update Documentation**: Keep these docs current with any changes
2. **Test Executable**: Always test PyInstaller builds after changes
3. **Verify Streaming**: Test streaming functionality with real clients
4. **Document Issues**: Add new compilation or runtime issues to relevant docs

## Version History

### Current Version
- **Streaming Fixes**: SSE keepalive, progressive tokens, sentence buffering
- **GGUF Support**: Any HuggingFace GGUF model without mappings
- **PyInstaller Compatibility**: Fixed import issues for executable builds
- **Port Change**: Default port 3847 instead of 8080
- **Production Ready**: Complete standalone executable with all dependencies