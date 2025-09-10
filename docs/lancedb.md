# LanceDB Integration Guide for Fluid Server

## Overview

Fluid Server integrates **LanceDB** as its primary vector database solution for storing and retrieving high-dimensional embeddings. LanceDB provides a modern, embedded vector database specifically designed for AI applications with native multimodal support and superior performance characteristics.

## Why LanceDB Over Chroma?

### 1. **Native .NET Client Support**
LanceDB offers comprehensive client libraries including full .NET support, making it ideal for Windows desktop applications that need to integrate with C# and .NET frameworks. Chroma only provides a client-side solution for .NET environments.

### 2. **Native Multimodal Embeddings**
LanceDB supports multimodal embeddings (text, image, audio) natively without requiring additional configuration or separate collections. This allows unified storage and cross-modal search capabilities.

### 3. **Superior Performance**
- **Embedded Architecture**: LanceDB runs as an embedded solution with lower latency and no network overhead
- **Columnar Storage**: Uses Apache Arrow and Lance format for efficient storage and retrieval
- **Optimized Indexing**: Advanced indexing algorithms specifically designed for high-dimensional vectors

### 4. **Simplified Deployment**
As an embedded solution, LanceDB eliminates the need for separate database server infrastructure, making deployment and management significantly simpler.

## Architecture Overview

### Core Components

```
Fluid Server Architecture
├── API Layer (FastAPI)
│   ├── /v1/embeddings          # OpenAI-compatible embeddings
│   ├── /v1/embeddings/multimodal # Multimodal embedding support
│   └── /v1/vector_store/*      # Vector storage operations
├── Embedding Manager
│   ├── Text Embeddings (OpenVINO)
│   ├── Image Embeddings (CLIP-based)
│   └── Audio Embeddings (Whisper-based)
└── LanceDB Storage Layer
    ├── Collections (Tables)
    ├── Vector Search Engine
    └── Document Storage
```

### Model Directory Structure

```
models/
├── embeddings/
│   ├── sentence-transformers_all-MiniLM-L6-v2/  # Text models
│   ├── openai_clip-vit-base-patch32/             # Multimodal models
│   └── openai_whisper-base/                      # Audio models
└── cache/                                        # Compiled model cache
```

## Installation and Configuration

### Dependencies

LanceDB is automatically installed with Fluid Server:

```toml
# pyproject.toml
dependencies = [
    "lancedb>=0.14.0",
    "sentence-transformers>=2.2.0",
    "pillow>=10.0.0",
]
```

### Configuration

Enable embeddings in your server configuration:

```python
# Server startup
config = ServerConfig(
    enable_embeddings=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    multimodal_model="openai/clip-vit-base-patch32",
    embedding_device="CPU",  # or "GPU"
    embeddings_db_path=Path("./data/embeddings"),
    embeddings_db_name="vectors"
)
```

## API Usage Examples

### 1. Text Embeddings

#### Generate Text Embeddings
```bash
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Machine learning with Python"],
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

#### Store Documents with Automatic Embedding
```bash
curl -X POST "http://localhost:8080/v1/vector_store/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "documents", 
    "documents": [
      {
        "content": "LanceDB provides efficient vector storage",
        "metadata": {"source": "documentation", "category": "database"}
      },
      {
        "content": "Fluid Server enables AI model deployment on Windows",
        "metadata": {"source": "readme", "category": "deployment"}
      }
    ],
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

### 2. Multimodal Embeddings

#### Image Embeddings
```bash
curl -X POST "http://localhost:8080/v1/embeddings/multimodal" \
  -F "input_type=image" \
  -F "model=openai/clip-vit-base-patch32" \
  -F "file=@image.jpg"
```

### 3. Vector Search

#### Text-based Search
```bash
curl -X POST "http://localhost:8080/v1/vector_store/search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "documents",
    "query": "vector database performance",
    "query_type": "text",
    "limit": 5,
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

#### Cross-modal Search (Image to Text)
```bash
curl -X POST "http://localhost:8080/v1/vector_store/search/multimodal" \
  -F "collection=documents" \
  -F "query_type=image" \
  -F "limit=10" \
  -F "model=openai/clip-vit-base-patch32" \
  -F "file_query=@query_image.jpg"
```

## Collection Management

### Create Collections
```bash
curl -X POST "http://localhost:8080/v1/vector_store/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_collection",
    "dimension": 384,
    "content_type": "text",
    "overwrite": false
  }'
```

### List Collections
```bash
curl -X GET "http://localhost:8080/v1/vector_store/collections"
```

### Get Collection Statistics
```bash
curl -X GET "http://localhost:8080/v1/vector_store/my_collection/stats"
```

## Programmatic Usage (Python)

## Advanced Features

### 1. Filtering

LanceDB supports SQL-like filtering expressions:

```python
# Filter by metadata
results = await lancedb_client.search_vectors(
    collection_name="documents",
    query_vector=query_vector,
    limit=10,
    filter_condition="metadata->>'category' = 'technical'"
)
```

### 2. Batch Operations

```python
# Batch insert
documents = [VectorDocument(...) for _ in range(1000)]
await lancedb_client.insert_documents("large_collection", documents)

# Batch embedding generation
texts = ["text " + str(i) for i in range(100)]
embeddings = await embedding_manager.get_text_embeddings(texts)
```

### 3. Memory Management

The server automatically manages embedding model memory:

```python
# Models are automatically loaded/unloaded based on usage
config = ServerConfig(
    idle_timeout_minutes=30,  # Unload models after 30 minutes of inactivity
    max_memory_gb=8.0         # Maximum memory usage
)
```

### Debug Commands

```bash
# Check available models
curl -X GET "http://localhost:8080/v1/embeddings/models"

# Verify collection status
curl -X GET "http://localhost:8080/v1/vector_store/collections"
```