"""
Entry point for fluid_server module
"""
import argparse
import sys
import logging
from pathlib import Path
import uvicorn
from fluid_server.config import ServerConfig
from fluid_server.app import create_app
from fluid_server.utils.model_discovery import ModelDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point with command-line argument parsing"""
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
  
  # Use GPU with preloading
  %(prog)s --device GPU --preload
        """
    )
    
    # Server options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    
    # Model paths - simple!
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models"),
        help="Base path containing model directories (llm/, whisper/, etc.)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for compiled models (default: model-path/cache)"
    )
    
    # Model selection
    parser.add_argument(
        "--llm-model",
        default="qwen3-8b",
        help="LLM model to use (directory name in model-path/llm/)"
    )
    parser.add_argument(
        "--whisper-model",
        default="whisper-tiny",
        help="Whisper model to use (directory name in model-path/whisper/)"
    )
    
    # Runtime options
    parser.add_argument(
        "--device",
        default="GPU",
        choices=["CPU", "GPU", "NPU"],
        help="Device to run models on (default: GPU)"
    )
    
    # Features
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload models on startup"
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=5,
        help="Minutes before unloading idle models (0 to disable)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        device=args.device,
        preload_models=args.preload,
        idle_timeout_minutes=args.idle_timeout
    )
    
    # Validate that model path exists
    if not config.model_path.exists():
        logger.error(f"Model path does not exist: {config.model_path}")
        logger.info("Please create the directory or specify a different path with --model-path")
        sys.exit(1)
    
    # Discover available models
    logger.info(f"Discovering models in {config.model_path}")
    available_models = ModelDiscovery.find_models(config.model_path)
    
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
            logger.info(f"Continuing without LLM support...")
        else:
            logger.info("No LLM models available - chat completions will not work")
    
    # Create and run application
    app = create_app(config)
    
    # Check if running as frozen executable
    is_frozen = getattr(sys, 'frozen', False)
    
    # Configure uvicorn logging - handle frozen executable case
    if is_frozen:
        # Simple logging config for frozen executable without uvicorn-specific formatters
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "class": "logging.Formatter"
                },
                "access": {
                    "format": '%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                    "class": "logging.Formatter"
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
        log_config = uvicorn.config.LOGGING_CONFIG.copy()
        log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
        log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    
    # Run server
    uvicorn.run(
        app if is_frozen else "fluid_server.__main__:app",
        host=config.host,
        port=config.port,
        reload=args.reload and not is_frozen,
        log_level="info",
        log_config=log_config,
        access_log=True
    )


# For uvicorn reload
app = None
if __name__ != "__main__":
    # Create app for uvicorn when imported
    config = ServerConfig()
    app = create_app(config)


if __name__ == "__main__":
    main()