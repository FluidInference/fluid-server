"""
Entry point for fluid_server module
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

try:
    # Try relative imports (development mode)
    from .__version__ import __version__
    from .app import create_app
    from .config import ServerConfig
    from .utils.model_discovery import ModelDiscovery
except ImportError:
    # Fallback to absolute imports (PyInstaller executable)
    from fluid_server.__version__ import __version__
    from fluid_server.app import create_app
    from fluid_server.config import ServerConfig
    from fluid_server.utils.model_discovery import ModelDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point with command-line argument parsing"""

    # Fix stdout/stderr for frozen executables
    if getattr(sys, 'frozen', False):
        import io
        if sys.stdout is None:
            sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.BytesIO()), encoding='utf-8')
        if sys.stderr is None:
            sys.stderr = io.TextIOWrapper(io.BufferedWriter(io.BytesIO()), encoding='utf-8')

    parser = argparse.ArgumentParser(
        description="Fluid Server - OpenAI-compatible API with multiple model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use models in default location (./models)
  %(prog)s

  # Use custom model path
  %(prog)s --model-path C:\\MyApp\\ai-models

  # Specify different LLM model
  %(prog)s --llm-model phi-4-mini

  # Disable warm-up and use custom log level
  %(prog)s --no-warm-up --log-level DEBUG

  # Use specific device
  %(prog)s --device CPU
        """,
    )

    # Version
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Server configuration group
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    server_group.add_argument(
        "--port", type=int, default=3847, help="Port to bind to (default: 3847)"
    )
    server_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, set to 2+ for concurrent processing)",
    )
    server_group.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Logging group
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    # Model paths group
    path_group = parser.add_argument_group("Model Paths")
    path_group.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models"),
        help="Base path containing model directories (llm/, whisper/, etc.)",
    )
    path_group.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for compiled models (default: model-path/cache)",
    )

    # Model selection group
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--llm-model",
        default="unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
        help="LLM model to use (directory name in model-path/llm/)",
    )
    model_group.add_argument(
        "--whisper-model",
        default="whisper-tiny",
        help="Whisper model to use (directory name in model-path/whisper/)",
    )
    model_group.add_argument(
        "--device",
        choices=["AUTO", "CPU", "GPU", "NPU"],
        default="AUTO",
        help="Device to use for inference (default: AUTO)",
    )

    # Performance tuning group
    perf_group = parser.add_argument_group("Performance & Memory")
    perf_group.add_argument(
        "--no-warm-up", action="store_true", help="Disable warm-up of models on startup"
    )
    perf_group.add_argument(
        "--idle-timeout",
        type=int,
        default=5,
        help="Minutes before unloading idle models (0 to disable)",
    )
    perf_group.add_argument(
        "--idle-check-interval",
        type=int,
        default=60,
        help="Seconds between idle model checks (default: 60)",
    )
    perf_group.add_argument(
        "--max-memory",
        type=float,
        default=4.0,
        help="Maximum memory usage in GB for models (default: 4.0)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Apply log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # Create configuration
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_path=args.model_path.resolve(),
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        device=args.device,
        warm_up=not args.no_warm_up,
        idle_timeout_minutes=args.idle_timeout,
        idle_check_interval_seconds=args.idle_check_interval,
        max_memory_gb=args.max_memory,
    )

    # Validate that model path exists - create if missing instead of exiting
    if not config.model_path.exists():
        logger.warning(f"Model path does not exist: {config.model_path}")
        logger.info("Creating model directory automatically...")
        try:
            config.model_path.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for model types
            (config.model_path / "llm").mkdir(exist_ok=True)
            (config.model_path / "whisper").mkdir(exist_ok=True)
            (config.model_path / "embedding").mkdir(exist_ok=True)
            logger.info(f"Created model directories at {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to create model directory: {e}")
            logger.info(
                "Server will continue but no models will be available until directory is created"
            )

    # Discover available models
    logger.info(f"Discovering models in {config.model_path}")
    available_models = ModelDiscovery.find_models(config.model_path, config.llm_model)

    # Log available models
    if available_models.get("llm"):
        logger.info(f"Available LLM models: {', '.join(available_models['llm'])}")
    else:
        logger.warning("No LLM models found")

    if available_models.get("whisper"):
        logger.info(f"Available Whisper models: {', '.join(available_models['whisper'])}")
    else:
        logger.info("No Whisper models found (transcription will not be available)")

    # Validate requested models exist
    if config.llm_model not in available_models.get("llm", []):
        logger.warning(f"LLM model '{config.llm_model}' not found in {config.model_path / 'llm'}")
        if available_models.get("llm"):
            logger.info(f"Available LLM models: {available_models['llm']}")
            logger.info("Continuing without LLM support...")
        else:
            logger.info("No LLM models available - chat completions will not work")

    # Set environment variables for worker configuration
    import os

    os.environ["FLUID_HOST"] = config.host
    os.environ["FLUID_PORT"] = str(config.port)
    os.environ["FLUID_MODEL_PATH"] = str(config.model_path)
    if config.cache_dir:
        os.environ["FLUID_CACHE_DIR"] = str(config.cache_dir)
    os.environ["FLUID_LLM_MODEL"] = config.llm_model
    os.environ["FLUID_WHISPER_MODEL"] = config.whisper_model
    os.environ["FLUID_WARM_UP"] = str(config.warm_up).lower()
    os.environ["FLUID_IDLE_TIMEOUT"] = str(config.idle_timeout_minutes)

    # Check if running as frozen executable
    is_frozen = getattr(sys, "frozen", False)

    # Determine number of workers
    num_workers = args.workers
    if num_workers > 1 and is_frozen:
        logger.info(f"Using {num_workers} workers with PyInstaller frozen executable")
    elif num_workers > 1:
        logger.info(f"Using {num_workers} workers for concurrent processing")
    else:
        logger.info("Using single worker (concurrent processing disabled)")

    # Configure uvicorn logging - handle frozen executable case
    if is_frozen:
        # Simple logging config for frozen executable without uvicorn-specific formatters
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "class": "logging.Formatter",
                },
                "access": {
                    "format": '%(asctime)s - %(name)s - %(levelname)s - "%(message)s"',
                    "class": "logging.Formatter",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }
    else:
        # Standard uvicorn config for development
        from uvicorn.config import LOGGING_CONFIG

        log_config = LOGGING_CONFIG.copy()
        log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
        log_config["formatters"]["access"]["fmt"] = (
            '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        )

    # Run server
    if num_workers > 1 and not is_frozen:
        # Use string reference for multiple workers in development
        uvicorn.run(
            "src.fluid_server.app:app",
            host=config.host,
            port=config.port,
            reload=args.reload,
            workers=num_workers,
            log_level="info",
            log_config=log_config,
            access_log=True,
        )
    else:
        # Single process mode - works for both development and PyInstaller
        # Note: For PyInstaller builds, we use optimized async execution within single process
        if num_workers > 1 and is_frozen:
            logger.info(
                f"Using optimized async execution (requested {num_workers} workers, using 1 process with concurrent task handling)"
            )

        app = create_app(config)
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=args.reload and not is_frozen,
            workers=1,
            log_level="info",
            log_config=log_config,
            access_log=True,
        )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Required for PyInstaller to prevent infinite spawn loop
    main()
