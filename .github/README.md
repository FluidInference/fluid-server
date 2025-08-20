# GitHub Actions Workflows

This repository uses GitHub Actions for continuous integration and deployment. Here's an overview of the workflows:

## Workflows

### 1. PR Checks (`pr.yml`)
**Trigger:** Pull requests to `main` branch
**Purpose:** Validates code quality and builds before merging

- **Linting & Type Checking:** Runs `ruff` for code formatting/linting and `ty` for type checking
- **Multi-platform Build Tests:** Tests builds on Linux, Windows, and macOS (x64 and ARM64)
- **Server Health Checks:** Verifies the server starts correctly on each platform
- **PyInstaller Builds:** Creates executables and uploads as artifacts

### 2. Main Branch CI (`main.yml`)
**Trigger:** Pushes to `main` branch
**Purpose:** Auto-formats code and creates release artifacts

- **Auto-format:** Automatically formats code with `ruff` and commits changes
- **Type Checking:** Ensures type safety after formatting
- **Release Builds:** Creates production executables for all platforms
- **Draft Releases:** Creates draft GitHub releases when tags are pushed

### 3. Release Builds (`release.yml`)
**Trigger:** Manual dispatch or GitHub release creation
**Purpose:** Creates official release artifacts for all platforms

- **Cross-platform Builds:** Linux, Windows, macOS (both x64 and ARM64)
- **ARM64 Support:** Uses QEMU emulation for Linux ARM64, native runners for macOS ARM64
- **Automatic Upload:** Uploads artifacts to GitHub releases

### 4. Test Workflow (`test.yml`)
**Trigger:** Manual dispatch
**Purpose:** Quick testing of CI pipeline

Options:
- `quick`: Linting and type checks only
- `full`: All checks plus builds
- `build-only`: Just build testing

## Development Commands

```bash
# Install dependencies (including ruff)
uv sync

# Run formatting
uv run ruff format src/ tests/

# Run linting
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Run type checking
uv run ty
```

## Automated Maintenance

### Dependabot
Configured to automatically create PRs for:
- Python dependencies (weekly)
- GitHub Actions updates (weekly)

### Auto-formatting
The main branch automatically formats code on merge to maintain consistency.

## Platform Support

All workflows support:
- **Linux:** x64 and ARM64
- **Windows:** x64 and ARM64 (cross-compiled)
- **macOS:** x64 (Intel) and ARM64 (Apple Silicon)

## Notes

- Linting issues are automatically fixed on merge to `main`
- PR checks must pass before merging
- Release artifacts are retained for 30 days
- Build artifacts from PRs are retained for 7 days