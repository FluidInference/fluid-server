# Repository Guidelines

## Project Structure & Module Organization
- `src/fluid_server/`: FastAPI application packages orchestrating inference, routing, and device management. Add new modules close to related services and keep dependency injection consistent.
- `tests/`: Pytest suites; mirror the package layout when adding coverage. Start new features with `tests/test_<feature>.py`.
- `docs/`: Deep-dive references (compilation guide, NPU support, integration patterns); update alongside behavioral changes.
- `scripts/`: Helper utilities for packaging and automation; prefer extending these rather than duplicating shell logic.

## Build, Test, and Development Commands
- `uv sync`: Install the pinned Python 3.10 toolchain and dependencies.
- `uv run python -m fluid_server`: Launch the dev server with default configuration for manual verification.
- `uv run ty`: Execute static type analysis; treat failures as blockers.
- `uv run pytest tests/ -v`: Run the full test suite with verbose reporting before opening a PR.
- `uv run pytest tests/test_x.py`: Iterate on a focused module while developing.

## Coding Style & Naming Conventions
- Target Python 3.10, 4-space indentation, and the 100-character limit defined in `pyproject.toml`.
- Use absolute imports from `fluid_server`, ordered stdlib -> third-party -> local; run `ruff format` and `ruff check` to enforce style.
- Annotate every function signature, apply `Optional[...]` for nullable values, and pass filesystem data as `Path`.
- Provide concise docstrings for public APIs, log via `logger.error()` before re-raising, and keep async I/O under `async`/`await`; delegate heavy OpenVINO work to thread pools.

## Testing Guidelines
- Name test files `test_<feature>.py` and write descriptive pytest functions.
- Share fixtures within each module or a local `conftest.py` to keep scope clear.
- Expect new behavior to ship with tests; maintain coverage parity with the touched modules.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative commit subjects (e.g., `add vector DB fallback`), referencing PR numbers or issues in parentheses when relevant.
- Keep commits focused; separate generated artifacts from code changes.
- PR descriptions should cover context, approach, and validation steps; attach logs or screenshots for UI/API changes and link issues via `Fixes #id` when applicable.
