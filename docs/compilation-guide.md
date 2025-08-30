# PyInstaller Compilation Guide

This document outlines the key issues encountered during PyInstaller compilation and their solutions for the Fluid Server project.

## Critical Import Issue and Solution

### Problem
When running the compiled executable, we encountered this error:
```
ImportError: attempted relative import with no known parent package
```

This occurred in `src/fluid_server/__main__.py` with these relative imports:
```python
from .app import create_app
from .config import ServerConfig
from .utils.model_discovery import ModelDiscovery
```

### Root Cause
PyInstaller packages the code differently than the development environment. When the executable runs, Python doesn't recognize the module structure that allows relative imports to work.

### Solution
Implemented a try/except pattern in `src/fluid_server/__main__.py` to handle both development and compiled environments:

```python
try:
    # Try relative imports (development mode)
    from .app import create_app
    from .config import ServerConfig
    from .utils.model_discovery import ModelDiscovery
except ImportError:
    # Fallback to absolute imports (PyInstaller executable)
    from fluid_server.app import create_app
    from fluid_server.config import ServerConfig
    from fluid_server.utils.model_discovery import ModelDiscovery
```

## Build Performance and Process

### Build Time
- **Full build from scratch**: ~15-20 minutes
- **Analysis phase**: ~80% of build time
- **Building phase**: ~20% of build time

### Build Size
- **Final executable**: ~310 MB
- **Includes**: OpenVINO, LlamaCpp, PyTorch, SciPy, FastAPI, and all ML dependencies

### Memory Requirements
- **Build process**: Requires significant RAM (8GB+ recommended)
- **Analysis phase**: Most memory-intensive part

## Build Commands

### Method 1: Using Build Script
```powershell
.\scripts\build.ps1
```

### Method 2: Direct PyInstaller with Spec File
```bash
uv run pyinstaller fluid-server.spec --noconfirm
```

### Method 3: Manual PyInstaller Command
```bash
uv run pyinstaller src/fluid_server/__main__.py --name fluid-server --onefile
```

## PyInstaller Spec File Configuration

Key configurations in `fluid-server.spec`:

```python
# Include source code as data
datas = [('src\\fluid_server', 'fluid_server')]

# Essential hidden imports
hiddenimports = [
    'fluid_server.app',
    'fluid_server.managers',
    'fluid_server.runtimes',
    'fluid_server.runtimes.llamacpp_llm',
    'openvino',
    'openvino_genai',
    'llama_cpp',
    # ... other dependencies
]

# Collect all data/binaries for ML packages
for pkg in ['openvino', 'openvino_genai', 'openvino_tokenizers', 'librosa', 'scipy', 'soundfile', 'llama_cpp']:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
```

## Common Build Issues and Solutions

### Issue 1: Missing Hidden Imports
**Error**: Module not found errors during runtime
**Solution**: Add missing modules to `hiddenimports` list in spec file

### Issue 2: DLL Loading Issues
**Error**: DLL not found errors
**Solution**: Ensure all required libraries are included in `binaries` list

### Issue 3: Data Files Missing
**Error**: Configuration files or model files not found
**Solution**: Add required data files to `datas` list

### Issue 4: Import Structure Problems
**Error**: Relative import errors
**Solution**: Use the try/except import pattern documented above

## Build Environment Requirements

### Python Environment
- **Python**: 3.10+ (tested with 3.10.16)
- **Package Manager**: uv (recommended) or pip
- **Virtual Environment**: Required

### System Requirements
- **OS**: Windows 10/11 (x64)
- **RAM**: 8GB+ (16GB recommended for build process)
- **Disk Space**: 5GB+ free space for build artifacts
- **Architecture**: x64 (ARM64 builds require additional configuration)

### Key Dependencies
- **PyInstaller**: 6.15.0+
- **OpenVINO**: Latest stable
- **LlamaCpp**: Latest with Vulkan support
- **FastAPI**: Latest
- **Uvicorn**: Latest

## Testing the Built Executable

### Basic Functionality Test
```bash
# Test help output
.\dist\fluid-server.exe --help

# Test startup (will fail if port in use)
.\dist\fluid-server.exe --no-warm-up
```

### Full Integration Test
```bash
# Test with specific model
.\dist\fluid-server.exe --llm-model "unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf" --no-warm-up --port 3848
```

## Build Artifacts

### Generated Files
- `dist/fluid-server.exe` - Final executable
- `build/` - Temporary build files (can be deleted)
- `*.spec` - PyInstaller specification file

### Build Logs
- Warnings logged to `build/fluid-server/warn-fluid-server.txt`
- Cross-reference graph at `build/fluid-server/xref-fluid-server.html`

## Performance Considerations

### Runtime Performance
- **Startup time**: ~2-3 seconds (vs ~1 second in development)
- **Memory usage**: Similar to development mode
- **Model loading**: Same performance as development

### Build Optimization
- Use `--noconfirm` to skip prompts
- Consider using `--clean` for clean builds
- Build on SSD for better performance

## Troubleshooting

### Common Solutions
1. **Clean build**: Delete `build/` and `dist/` directories
2. **Update dependencies**: Ensure all packages are latest versions
3. **Check logs**: Review PyInstaller warnings and errors
4. **Test imports**: Verify all modules can be imported in development

### Debug Mode
For debugging issues, build with:
```bash
uv run pyinstaller fluid-server.spec --debug=all --noconfirm
```

## Best Practices

1. **Version Control**: Keep `fluid-server.spec` in version control
2. **Automated Builds**: Use the build script for consistency
3. **Testing**: Always test the executable before deployment
4. **Documentation**: Document any spec file changes
5. **Clean Builds**: Perform clean builds for releases

## Notes

- The executable is self-contained and doesn't require Python installation
- All streaming improvements and GGUF model support are included
- Default port is 3847 to avoid conflicts with common services
- The executable supports all command-line options from the development version