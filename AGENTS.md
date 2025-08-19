# Development Guide for AI Coding Agents

## Build & Test Commands
```bash
uv sync                           # Install dependencies
uv run python -m fluid_server     # Run server
uv run ty                         # Type check
uv run pytest tests/ -v           # Run all tests (when available)
uv run pytest tests/test_x.py    # Run single test file
```

## Code Style
- **Imports**: Use absolute imports from `fluid_server` package, group by: stdlib, third-party, local
- **Type Hints**: Required for all function signatures, use `Optional[T]` for nullable, `Path` for filesystem paths
- **Docstrings**: Use triple quotes with brief description for all public functions/classes
- **Error Handling**: Log errors with `logger.error()` before raising, use specific exceptions
- **Async**: Use `async/await` for I/O operations, ThreadPoolExecutor for CPU-bound OpenVINO ops
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **FastAPI**: Use Pydantic models for request/response validation, dependency injection for shared state
- **OpenVINO**: Run inference in dedicated thread pools, handle device selection (CPU/GPU/NPU)
- **Logging**: Use module-level `logger = logging.getLogger(__name__)`, avoid print statements
- **File Paths**: Always use `Path` objects, never hardcode separators, resolve to absolute paths