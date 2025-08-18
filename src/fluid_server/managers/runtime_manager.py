"""
Runtime manager for handling multiple AI models
"""
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import asyncio
import time
from ..config import ServerConfig
from ..runtimes.base import BaseRuntime
from ..runtimes.openvino_llm import OpenVINOLLMRuntime
from ..runtimes.openvino_whisper import OpenVINOWhisperRuntime
from ..utils.model_discovery import ModelDiscovery

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Manages a single AI model runtime at a time for memory efficiency"""
    
    def __init__(self, config: ServerConfig) -> None:
        """
        Initialize runtime manager
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.current_runtime: Optional[BaseRuntime] = None
        self.current_model_name: Optional[str] = None
        self.available_models = ModelDiscovery.find_models(config.model_path)
        self._idle_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize runtime manager and optionally preload models"""
        logger.info(f"Initializing runtime manager with models from {self.config.model_path}")
        logger.info(f"Available models: {self.available_models}")
        
        # Start idle cleanup monitoring
        self._schedule_idle_cleanup()
        
        if self.config.preload_models:
            logger.info("Preloading models...")
            await self.load_llm()
            # await self.load_whisper()  # Uncomment when Whisper is fully implemented
    
    async def load_llm(self, model_name: str = None) -> OpenVINOLLMRuntime:
        """Load an LLM model (unloads previous model if different)"""
        # Use provided model name or fall back to configured default
        model_to_load = model_name or self.config.llm_model
        
        # If the same model is already loaded, return it
        if (self.current_runtime is not None and 
            self.current_model_name == model_to_load and 
            isinstance(self.current_runtime, OpenVINOLLMRuntime)):
            logger.debug(f"LLM model '{model_to_load}' already loaded")
            self.current_runtime.last_used = time.time()  # Update last used
            return self.current_runtime
        
        # Unload any existing model first
        if self.current_runtime is not None:
            logger.info(f"Unloading current model '{self.current_model_name}' to load '{model_to_load}'")
            await self.current_runtime.unload()
            self.current_runtime = None
            self.current_model_name = None
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
        # Get model path
        model_path = ModelDiscovery.get_model_path(
            self.config.model_path,
            "llm",
            model_to_load
        )
        
        if model_path is None:
            raise ValueError(f"LLM model '{model_to_load}' not found")
        
        # Create and load runtime
        logger.info(f"Loading LLM model '{model_to_load}'")
        runtime = OpenVINOLLMRuntime(
            model_path=model_path,
            cache_dir=self.config.cache_dir,
            device=self.config.device
        )
        
        await runtime.load()
        self.current_runtime = runtime
        self.current_model_name = model_to_load
        
        # Start idle cleanup monitoring
        self._schedule_idle_cleanup()
        
        return runtime
    
    async def load_whisper(self) -> OpenVINOWhisperRuntime:
        """Load the configured Whisper model"""
        if "whisper" in self.runtimes:
            logger.debug("Whisper already loaded")
            return self.runtimes["whisper"]
        
        # Get model path
        model_path = ModelDiscovery.get_model_path(
            self.config.model_path,
            "whisper",
            self.config.whisper_model
        )
        
        if model_path is None:
            raise ValueError(f"Whisper model '{self.config.whisper_model}' not found")
        
        # Create and load runtime
        runtime = OpenVINOWhisperRuntime(
            model_path=model_path,
            cache_dir=self.config.cache_dir,
            device=self.config.device
        )
        
        await runtime.load()
        self.runtimes["whisper"] = runtime
        
        # Schedule idle cleanup
        self._schedule_idle_cleanup()
        
        return runtime
    
    async def get_llm(self, model_name: str = None) -> OpenVINOLLMRuntime:
        """Get LLM runtime, loading if necessary"""
        model_to_get = model_name or self.config.llm_model
        
        # Load the model if it's not the current one
        if (self.current_runtime is None or 
            self.current_model_name != model_to_get or 
            not isinstance(self.current_runtime, OpenVINOLLMRuntime)):
            await self.load_llm(model_name)
        
        return self.current_runtime
    
    
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
                    
                    if self.current_runtime is None:
                        continue
                    
                    current_time = time.time()
                    idle_threshold = self.config.idle_timeout_minutes * 60
                    
                    # Check if current model is idle
                    if hasattr(self.current_runtime, 'last_used'):
                        idle_time = current_time - self.current_runtime.last_used
                        if idle_time >= idle_threshold:
                            logger.info(f"Unloading idle model '{self.current_model_name}' (idle for {idle_time/60:.1f} minutes)")
                            try:
                                await self.current_runtime.unload()
                                self.current_runtime = None
                                self.current_model_name = None
                                
                                # Force garbage collection
                                import gc
                                gc.collect()
                                logger.info("Memory freed, no models loaded")
                            except Exception as e:
                                logger.error(f"Failed to unload model: {e}")
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in idle cleanup: {e}")
        
        self._idle_task = asyncio.create_task(cleanup_idle_model())
    
    async def unload_all(self) -> None:
        """Unload the current model"""
        if self.current_runtime is not None:
            logger.info(f"Unloading current model '{self.current_model_name}'")
            await self.current_runtime.unload()
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of current runtime"""
        status = {
            "available_models": self.available_models,
            "loaded_model": None,
            "config": {
                "device": self.config.device,
                "llm_model": self.config.llm_model,
                "whisper_model": self.config.whisper_model,
                "idle_timeout_minutes": self.config.idle_timeout_minutes
            }
        }
        
        if self.current_runtime is not None:
            status["loaded_model"] = {
                "name": self.current_model_name,
                "info": self.current_runtime.get_info()
            }
        
        return status