# GGUF Model Support Documentation

This document explains the GGUF model support implementation and how to use any GGUF model from HuggingFace Hub.

## Overview

The Fluid Server now supports any GGUF (GPT-Generated Unified Format) model from HuggingFace Hub without requiring predefined mappings. This provides flexibility to use the latest quantized models.

## Supported Model Formats

### Format 1: Repository with Auto-Detection
```bash
--llm-model "microsoft/Phi-3-mini-4k-instruct-gguf"
```
- Automatically detects the main GGUF file in the repository
- Works when repository has a single primary GGUF file

### Format 2: Explicit File Specification
```bash
--llm-model "unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
```
- Specifies exact GGUF file when repository has multiple files
- Format: `repository_owner/repository_name/filename.gguf`

### Format 3: Legacy Named Models (Backward Compatibility)
```bash
--llm-model "gemma-3-4b-it-gguf"
```
- Uses predefined mappings in `model_downloader.py`
- Maintained for backward compatibility

## Implementation Details

### Model Identifier Parsing

**Location**: `src/fluid_server/runtimes/llamacpp_llm.py`

```python
def _parse_model_identifier(self) -> tuple[str | None, str | None]:
    """Parse model identifier to extract repo_id and filename"""
    
    if "/" in self.model_name:
        parts = self.model_name.split("/")
        
        if len(parts) >= 3:
            # Format: repo_owner/repo_name/filename.gguf
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
            return repo_id, filename
        elif len(parts) == 2:
            # Format: repo_owner/repo_name (auto-detect)
            repo_id = self.model_name
            filename = None  # Let llama-cpp auto-detect
            return repo_id, filename
    
    # Fallback to mappings
    repo_id = DEFAULT_MODEL_REPOS.get("llm", {}).get(self.model_name)
    filename = GGUF_FILE_MAPPINGS.get(self.model_name)
    
    return repo_id, filename
```

### Model Loading

**LlamaCpp Integration**: Uses `llama-cpp-python`'s `from_pretrained` method:

```python
# Auto-detect GGUF file
if not gguf_filename:
    self.llama = self.Llama.from_pretrained(
        repo_id=repo_id,
        cache_dir=str(self.model_path.resolve()),
        resume_download=True,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
# Explicit filename
else:
    self.llama = self.Llama.from_pretrained(
        repo_id=repo_id,
        filename=gguf_filename,
        cache_dir=str(self.model_path.resolve()),
        resume_download=True,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
```

## Configuration

### Backend Selection
- **GPU**: Uses Vulkan backend with `n_gpu_layers=-1`
- **CPU**: Uses CPU backend with `n_gpu_layers=0`
- **Auto**: Based on `--device` parameter (default: GPU)

### Model Parameters
```python
DEFAULT_CONTEXT_SIZE = 4096      # Context window
DEFAULT_BATCH_SIZE = 512         # Batch processing size
DEFAULT_GPU_LAYERS = -1          # All layers on GPU (if available)
VULKAN_BACKEND = True            # Use Vulkan for GPU acceleration
```

### Cache Management
- **Download Cache**: `{model_path}/cache/` (configurable)
- **Resume Downloads**: Enabled by default
- **Permissions**: Uses local cache to avoid global permission issues

## Popular GGUF Models

### Recommended Models

#### Small Models (< 5GB)
```bash
# Phi-3 Mini (3.8B parameters)
--llm-model "microsoft/Phi-3-mini-4k-instruct-gguf"

# Gemma 2B
--llm-model "google/gemma-2b-it-GGUF/gemma-2b-it-q4_k_m.gguf"
```

#### Medium Models (5-15GB)
```bash
# Gemma 7B
--llm-model "google/gemma-7b-it-GGUF/gemma-7b-it-q4_k_m.gguf"

# Llama 3.2 3B
--llm-model "unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

#### Large Models (15GB+)
```bash
# Llama 3.1 8B
--llm-model "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Qwen2.5 7B
--llm-model "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf"
```

### Quantization Levels

#### Available Quantizations (by file size)
- **Q2_K**: Smallest, lowest quality
- **Q3_K_S/Q3_K_M**: Small, good quality
- **Q4_K_S/Q4_K_M**: Balanced size/quality ⭐ **Recommended**
- **Q5_K_S/Q5_K_M**: Larger, better quality
- **Q6_K**: Large, high quality
- **Q8_0**: Largest, highest quality

#### Selection Guide
```bash
# For limited RAM (< 8GB)
--llm-model "repo/model/model-Q4_K_S.gguf"

# Balanced performance (8-16GB RAM)  
--llm-model "repo/model/model-Q4_K_M.gguf"  # Most common

# High quality (16GB+ RAM)
--llm-model "repo/model/model-Q5_K_M.gguf"
```

## Model Discovery and Validation

### Automatic Detection
```python
# Models discovered at startup
available_models = ModelDiscovery.find_models(config.model_path, config.llm_model)

# Added even if not found locally (for download)
logger.info(f"Added GGUF model (not found locally): {model_name}")
```

### Validation Process
1. **Parse Format**: Determine repo_id and filename
2. **Check Local Cache**: Look for existing download
3. **Validate Repository**: Verify HuggingFace repository exists
4. **Download on Demand**: Download when first accessed
5. **Load Model**: Initialize llama-cpp with specified parameters

## Environment Variables

### HuggingFace Configuration
```bash
# Optional: Enable faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: Authentication for private models
export HF_TOKEN="your_huggingface_token"

# Optional: Custom cache location
export HF_HOME="/path/to/cache"
```

### Fluid Server Configuration
```bash
# Model cache directory
export FLUID_CACHE_DIR="/path/to/model/cache"

# Model path
export FLUID_MODEL_PATH="/path/to/models"

# Default LLM model
export FLUID_LLM_MODEL="unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
```

## Download Behavior

### Automatic Download
- **Triggered**: When model is first accessed (not at startup)
- **Resumable**: Supports partial downloads
- **Progress**: Logged during download
- **Cache**: Stored in configured cache directory

### Manual Pre-download
```bash
# Pre-download a model (optional)
python -c "
from llama_cpp import Llama
Llama.from_pretrained(
    repo_id='unsloth/gemma-3-4b-it-GGUF',
    filename='gemma-3-4b-it-Q4_K_M.gguf'
)
"
```

## Error Handling

### Common Issues and Solutions

#### Repository Not Found
```
ERROR: Repository unsloth/invalid-model not found on HuggingFace Hub
```
**Solution**: Verify repository name and accessibility

#### File Not Found in Repository
```
ERROR: File model-invalid.gguf not found in repository
```
**Solution**: Check available files in the HuggingFace repository

#### Insufficient Memory
```
ERROR: Failed to allocate memory for model
```
**Solution**: Use smaller quantization (Q4_K_S instead of Q5_K_M)

#### Download Timeout
```
ERROR: Download timeout
```
**Solution**: Check internet connection, retry with `resume_download=True`

## Performance Optimization

### GPU Acceleration
```python
# Vulkan backend (default for GPU)
n_gpu_layers = -1  # All layers on GPU

# CUDA backend (if available)
# Requires llama-cpp-python compiled with CUDA support
```

### Memory Management
```python
# Optimize context size based on available RAM
n_ctx = 4096      # 16GB+ RAM
n_ctx = 2048      # 8-16GB RAM  
n_ctx = 1024      # <8GB RAM
```

### Batch Processing
```python
# Optimize batch size for throughput
n_batch = 512     # Default
n_batch = 256     # Lower memory usage
n_batch = 1024    # Higher throughput (if memory allows)
```

## Backward Compatibility

### Legacy Model Mappings
**Location**: `src/fluid_server/utils/model_downloader.py`

```python
DEFAULT_MODEL_REPOS = {
    "llm": {
        "gemma-3-4b-it-gguf": "ggml-org/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-qat-GGUF": "unsloth/gemma-3-4b-it-qat-GGUF"
    }
}

GGUF_FILE_MAPPINGS = {
    "gemma-3-4b-it-gguf": "gemma-3-4b-it-Q4_K_M.gguf",
    "gemma-3-4b-it-qat-GGUF": "gemma-3-4b-it-qat-BF16.gguf",
}
```

### Migration Guide
```bash
# Old format (still works)
--llm-model "gemma-3-4b-it-gguf"

# New format (recommended)
--llm-model "ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
```

## Testing

### Basic Model Loading Test
```bash
# Test model parsing and loading
./fluid-server.exe --llm-model "microsoft/Phi-3-mini-4k-instruct-gguf" --no-warm-up
```

### Streaming Test
```bash
# Test with API call
curl -X POST http://localhost:3847/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "current",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Future Enhancements

### Planned Features
1. **Model Library**: Built-in catalog of popular GGUF models
2. **Auto-Quantization**: Automatic quantization selection based on available RAM  
3. **Multi-Model Support**: Load multiple GGUF models simultaneously
4. **Model Validation**: Pre-download validation of model compatibility
5. **Progress Tracking**: Better download progress reporting

### Community Integration
- Model recommendations based on hardware
- Community ratings and performance benchmarks
- Automatic model updates and notifications