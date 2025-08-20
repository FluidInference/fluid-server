"""
Runtime manager for handling multiple AI models
"""

import asyncio
import logging
import time
from typing import Any

from ..config import ServerConfig
from ..runtimes.base import BaseRuntime
from ..runtimes.openvino_llm import OpenVINOLLMRuntime
from ..runtimes.openvino_whisper import OpenVINOWhisperRuntime
from ..utils.model_discovery import ModelDiscovery
from ..utils.model_downloader import ModelDownloader
from ..utils.retry import retry_async

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Manages LLM and Whisper model runtimes, keeping both loaded simultaneously for optimal performance"""

    def __init__(self, config: ServerConfig) -> None:
        """
        Initialize runtime manager

        Args:
            config: Server configuration
        """
        self.config = config
        # Keep separate runtimes for each model type
        self.llm_runtime: BaseRuntime | None = None
        self.whisper_runtime: BaseRuntime | None = None
        self.loaded_llm_model: str | None = None
        self.loaded_whisper_model: str | None = None
        self.available_models = ModelDiscovery.find_models(config.model_path)
        self.downloader = ModelDownloader(config.model_path, config.cache_dir or config.model_path / "cache")
        self._idle_task: asyncio.Task | None = None
        self.download_status: dict[str, str] = {}  # Track download status for models

        # Legacy compatibility properties
        self.current_runtime: BaseRuntime | None = None
        self.current_model_name: str | None = None

        # Track warm-up status
        self.warmup_status = {
            "in_progress": False,
            "whisper": "pending",  # pending/loading/ready/failed
            "llm": "pending",  # pending/loading/ready/failed
            "current_task": "",
            "start_time": None,
        }

    async def initialize(self) -> None:
        """Initialize runtime manager and optionally warm up models"""
        logger.info(f"Initializing runtime manager with models from {self.config.model_path}")
        logger.info(f"Available models: {self.available_models}")

        # Start idle cleanup monitoring
        self._schedule_idle_cleanup()

        if self.config.warm_up:
            logger.info("Starting background model warm-up...")
            # Start warm-up in background to not block server startup
            asyncio.create_task(self._warm_up_models())

    async def _warm_up_models(self) -> None:
        """Warm up models in the background - download and pre-load both models"""
        try:
            # Start warm-up tracking
            self.warmup_status["in_progress"] = True
            self.warmup_status["start_time"] = time.time()
            self.warmup_status["current_task"] = "Starting model warm-up..."
            logger.info("Starting comprehensive model warm-up...")

            # Phase 1: Download models if needed
            self.warmup_status["current_task"] = "Checking and downloading models if needed..."

            # Check and download LLM if needed
            if self.config.llm_model not in self.available_models.get("llm", []):
                logger.info(
                    f"LLM model '{self.config.llm_model}' not found locally, downloading..."
                )
                self.warmup_status["current_task"] = (
                    f"Downloading LLM model '{self.config.llm_model}'..."
                )
                self.download_status[f"llm:{self.config.llm_model}"] = "downloading"
                llm_available = await self.downloader.ensure_model_available(
                    "llm", self.config.llm_model
                )
                if llm_available:
                    # Refresh model discovery after download
                    self.available_models = ModelDiscovery.find_models(self.config.model_path)
                    self.download_status.pop(f"llm:{self.config.llm_model}", None)
                    logger.info(f"LLM model '{self.config.llm_model}' downloaded successfully")
                else:
                    self.download_status[f"llm:{self.config.llm_model}"] = "failed"
                    self.warmup_status["llm"] = "failed"
                    logger.warning(f"LLM model '{self.config.llm_model}' could not be downloaded")

            # Check and download Whisper if needed
            if self.config.whisper_model not in self.available_models.get("whisper", []):
                logger.info(
                    f"Whisper model '{self.config.whisper_model}' not found locally, downloading..."
                )
                self.warmup_status["current_task"] = (
                    f"Downloading Whisper model '{self.config.whisper_model}'..."
                )
                self.download_status[f"whisper:{self.config.whisper_model}"] = "downloading"
                whisper_available = await self.downloader.ensure_model_available(
                    "whisper", self.config.whisper_model
                )
                if whisper_available:
                    # Refresh model discovery after download
                    self.available_models = ModelDiscovery.find_models(self.config.model_path)
                    self.download_status.pop(f"whisper:{self.config.whisper_model}", None)
                    logger.info(
                        f"Whisper model '{self.config.whisper_model}' downloaded successfully"
                    )
                else:
                    self.download_status[f"whisper:{self.config.whisper_model}"] = "failed"
                    self.warmup_status["whisper"] = "failed"
                    logger.warning(
                        f"Whisper model '{self.config.whisper_model}' could not be downloaded"
                    )

            # Phase 2: Load and warm Whisper first (primary use case)
            if self.config.whisper_model in self.available_models.get("whisper", []):
                try:
                    self.warmup_status["whisper"] = "loading"
                    self.warmup_status["current_task"] = (
                        f"Loading and compiling Whisper model '{self.config.whisper_model}' on NPU (this may take several minutes)..."
                    )
                    logger.info(
                        f"Pre-loading Whisper model '{self.config.whisper_model}' for warm-up"
                    )
                    await self.load_whisper(self.config.whisper_model)
                    self.warmup_status["whisper"] = "ready"
                    logger.info(
                        f"Whisper model '{self.config.whisper_model}' loaded and compiled successfully"
                    )
                except Exception as e:
                    self.warmup_status["whisper"] = "failed"
                    logger.error(f"Failed to warm up Whisper model: {e}")

            # Phase 3: Load and warm LLM (keep both models loaded)
            if self.config.llm_model in self.available_models.get("llm", []):
                try:
                    self.warmup_status["llm"] = "loading"
                    self.warmup_status["current_task"] = (
                        f"Pre-loading LLM model '{self.config.llm_model}' on GPU..."
                    )
                    logger.info(f"Pre-loading LLM model '{self.config.llm_model}' for warm-up")
                    await self.load_llm(self.config.llm_model)
                    self.warmup_status["llm"] = "ready"
                    logger.info(f"LLM model '{self.config.llm_model}' loaded successfully")

                    # Both models are now loaded and will remain in memory
                    logger.info("Both LLM and Whisper models are now loaded and ready")

                except Exception as e:
                    self.warmup_status["llm"] = "failed"
                    logger.error(f"Failed to warm up LLM model: {e}")

            # Phase 5: Complete warm-up
            elapsed = time.time() - self.warmup_status["start_time"]
            self.warmup_status["in_progress"] = False
            self.warmup_status["current_task"] = ""
            logger.info(
                f"Model warm-up completed in {elapsed:.1f}s - Whisper: {self.warmup_status['whisper']}, LLM: {self.warmup_status['llm']}"
            )

        except Exception as e:
            self.warmup_status["in_progress"] = False
            self.warmup_status["current_task"] = f"Error: {str(e)}"
            logger.error(f"Error during model warm-up: {e}")

    async def load_llm(self, model_name: str | None = None) -> OpenVINOLLMRuntime | None:
        """Load an LLM model (keeps both LLM and Whisper in memory)"""
        # Use provided model name or fall back to configured default
        model_to_load = model_name or self.config.llm_model

        # If the same model is already loaded, return it
        if self.llm_runtime is not None and self.loaded_llm_model == model_to_load:
            logger.debug(f"LLM model '{model_to_load}' already loaded")
            self.llm_runtime.last_used = time.time()  # Update last used
            # Update legacy compatibility
            self.current_runtime = self.llm_runtime
            self.current_model_name = model_to_load
            return self.llm_runtime

        # Unload existing LLM if different
        if self.llm_runtime is not None and self.loaded_llm_model != model_to_load:
            logger.info(f"Unloading LLM '{self.loaded_llm_model}' to load '{model_to_load}'")
            await self.llm_runtime.unload()
            self.llm_runtime = None
            self.loaded_llm_model = None

            # Force garbage collection to free memory
            import gc

            gc.collect()

        # Get model path
        model_path = ModelDiscovery.get_model_path(self.config.model_path, "llm", model_to_load)

        if model_path is None:
            logger.error(f"LLM model '{model_to_load}' not found")
            return None  # Return None instead of raising exception

        # Create and load runtime with error recovery and retry logic
        logger.info(f"Loading LLM model '{model_to_load}' on GPU")

        @retry_async(
            max_attempts=3,
            delay=2.0,
            backoff_factor=1.5,
            exceptions=(RuntimeError, ValueError, OSError),
            on_retry=lambda attempt, exc: logger.info(
                f"Retrying LLM model load (attempt {attempt}): {exc}"
            ),
        )
        async def _load_with_retry():
            runtime = OpenVINOLLMRuntime(
                model_path=model_path, cache_dir=self.config.cache_dir or self.config.model_path / "cache", device="GPU"
            )
            await runtime.load()
            return runtime

        try:
            runtime = await _load_with_retry()
            self.llm_runtime = runtime
            self.loaded_llm_model = model_to_load

            # Update legacy compatibility
            self.current_runtime = runtime
            self.current_model_name = model_to_load

            # Start idle cleanup monitoring
            self._schedule_idle_cleanup()

            return runtime
        except Exception as e:
            logger.error(f"Failed to load LLM model '{model_to_load}' after retries: {e}")
            logger.info("Server will continue without this LLM model")
            return None  # Return None instead of crashing

    async def load_whisper(self, model_name: str | None = None) -> OpenVINOWhisperRuntime | None:
        """Load a Whisper model (keeps both LLM and Whisper in memory)"""
        # Use provided model name or fall back to configured default
        model_to_load = model_name or self.config.whisper_model

        # If the same Whisper model is already loaded, return it
        if self.whisper_runtime is not None and self.loaded_whisper_model == model_to_load:
            logger.debug(f"Whisper model '{model_to_load}' already loaded")
            self.whisper_runtime.last_used = time.time()  # Update last used
            # Update legacy compatibility
            self.current_runtime = self.whisper_runtime
            self.current_model_name = model_to_load
            return self.whisper_runtime

        # Unload existing Whisper if different
        if self.whisper_runtime is not None and self.loaded_whisper_model != model_to_load:
            logger.info(
                f"Unloading Whisper '{self.loaded_whisper_model}' to load '{model_to_load}'"
            )
            await self.whisper_runtime.unload()
            self.whisper_runtime = None
            self.loaded_whisper_model = None

            # Force garbage collection to free memory
            import gc

            gc.collect()

        # Get model path
        model_path = ModelDiscovery.get_model_path(self.config.model_path, "whisper", model_to_load)

        if model_path is None:
            logger.error(f"Whisper model '{model_to_load}' not found")
            return None  # Return None instead of raising exception

        # Create and load runtime with error recovery and retry logic
        logger.info(f"Loading Whisper model '{model_to_load}' on NPU")

        @retry_async(
            max_attempts=3,
            delay=2.0,
            backoff_factor=1.5,
            exceptions=(RuntimeError, ValueError, OSError),
            on_retry=lambda attempt, exc: logger.info(
                f"Retrying Whisper model load (attempt {attempt}): {exc}"
            ),
        )
        async def _load_with_retry():
            runtime = OpenVINOWhisperRuntime(
                model_path=model_path, cache_dir=self.config.cache_dir or self.config.model_path / "cache", device="NPU"
            )
            await runtime.load()
            return runtime

        try:
            runtime = await _load_with_retry()
            self.whisper_runtime = runtime
            self.loaded_whisper_model = model_to_load

            # Update legacy compatibility
            self.current_runtime = runtime
            self.current_model_name = model_to_load

            # Start idle cleanup monitoring
            self._schedule_idle_cleanup()

            return runtime
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_to_load}' after retries: {e}")
            logger.info("Server will continue without this Whisper model")
            return None  # Return None instead of crashing

    async def get_llm(self, model_name: str | None = None) -> OpenVINOLLMRuntime:
        """Get LLM runtime, loading if necessary"""
        model_to_get = model_name or self.config.llm_model

        # Check if we have the right LLM loaded
        if self.llm_runtime is None or self.loaded_llm_model != model_to_get:
            runtime = await self.load_llm(model_name)
            if runtime is None:
                raise RuntimeError(f"LLM model '{model_to_get}' is not available")

        # Update last used time
        if self.llm_runtime:
            self.llm_runtime.last_used = time.time()

        return self.llm_runtime

    async def get_whisper(self, model_name: str | None = None) -> OpenVINOWhisperRuntime:
        """Get Whisper runtime, loading if necessary"""
        model_to_get = model_name or self.config.whisper_model

        # Check if we have the right Whisper loaded
        if self.whisper_runtime is None or self.loaded_whisper_model != model_to_get:
            runtime = await self.load_whisper(model_name)
            if runtime is None:
                raise RuntimeError(f"Whisper model '{model_to_get}' is not available")

        # Update last used time
        if self.whisper_runtime:
            self.whisper_runtime.last_used = time.time()

        return self.whisper_runtime

    def _schedule_idle_cleanup(self) -> None:
        """Schedule cleanup of idle model"""
        if self._idle_task:
            self._idle_task.cancel()

        if self.config.idle_timeout_minutes <= 0:
            return

        async def cleanup_idle_model():
            """Monitor and cleanup idle model"""
            while True:
                try:
                    # Check every minute for idle model
                    await asyncio.sleep(60)

                    if self.llm_runtime is None and self.whisper_runtime is None:
                        continue

                    current_time = time.time()
                    idle_threshold = self.config.idle_timeout_minutes * 60

                    # Check if LLM model is idle
                    if self.llm_runtime is not None and hasattr(self.llm_runtime, "last_used"):
                        idle_time = current_time - self.llm_runtime.last_used
                        if idle_time >= idle_threshold:
                            logger.info(
                                f"Unloading idle LLM model '{self.loaded_llm_model}' (idle for {idle_time / 60:.1f} minutes)"
                            )
                            try:
                                await self.llm_runtime.unload()
                                self.llm_runtime = None
                                self.loaded_llm_model = None
                                logger.info("LLM model unloaded due to inactivity")
                            except Exception as e:
                                logger.error(f"Failed to unload LLM model: {e}")

                    # Check if Whisper model is idle
                    if self.whisper_runtime is not None and hasattr(
                        self.whisper_runtime, "last_used"
                    ):
                        idle_time = current_time - self.whisper_runtime.last_used
                        if idle_time >= idle_threshold:
                            logger.info(
                                f"Unloading idle Whisper model '{self.loaded_whisper_model}' (idle for {idle_time / 60:.1f} minutes)"
                            )
                            try:
                                await self.whisper_runtime.unload()
                                self.whisper_runtime = None
                                self.loaded_whisper_model = None
                                logger.info("Whisper model unloaded due to inactivity")
                            except Exception as e:
                                logger.error(f"Failed to unload Whisper model: {e}")

                    # Update legacy compatibility if no models loaded
                    if self.llm_runtime is None and self.whisper_runtime is None:
                        self.current_runtime = None
                        self.current_model_name = None
                        # Force garbage collection
                        import gc

                        gc.collect()
                        logger.info("All models unloaded, memory freed")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in idle cleanup: {e}")

        self._idle_task = asyncio.create_task(cleanup_idle_model())

    async def unload_all(self) -> None:
        """Unload all models"""
        if self.llm_runtime is not None:
            logger.info(f"Unloading LLM model '{self.loaded_llm_model}'")
            await self.llm_runtime.unload()
            self.llm_runtime = None
            self.loaded_llm_model = None

        if self.whisper_runtime is not None:
            logger.info(f"Unloading Whisper model '{self.loaded_whisper_model}'")
            await self.whisper_runtime.unload()
            self.whisper_runtime = None
            self.loaded_whisper_model = None

        # Clear legacy compatibility
        self.current_runtime = None
        self.current_model_name = None

        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None

    def discover_models(self) -> None:
        """Refresh the list of available models"""
        logger.info(f"Discovering models in {self.config.model_path}")
        self.available_models = ModelDiscovery.find_models(self.config.model_path)
        logger.info(f"Available models: {self.available_models}")

    def get_status(self) -> dict[str, Any]:
        """Get status of runtime manager"""
        status = {
            "available_models": self.available_models,
            "loaded_models": {"llm": None, "whisper": None},
            "config": {
                "llm_device": "GPU",
                "whisper_device": "NPU",
                "llm_model": self.config.llm_model,
                "whisper_model": self.config.whisper_model,
                "idle_timeout_minutes": self.config.idle_timeout_minutes,
            },
        }

        if self.llm_runtime is not None:
            status["loaded_models"]["llm"] = {
                "name": self.loaded_llm_model,
                "info": self.llm_runtime.get_info(),
            }

        if self.whisper_runtime is not None:
            status["loaded_models"]["whisper"] = {
                "name": self.loaded_whisper_model,
                "info": self.whisper_runtime.get_info(),
            }

        return status
