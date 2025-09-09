"""
Embedding manager for handling multiple embedding models
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..config import ServerConfig
from ..runtimes.base_embedding import BaseEmbeddingRuntime, EmbeddingType
from ..runtimes.openvino_embedding import OpenVINOEmbeddingRuntime
from ..runtimes.whisper_embedding import WhisperEmbeddingRuntime
from ..utils.model_discovery import ModelDiscovery
from ..utils.model_downloader import ModelDownloader
from ..utils.retry import retry_async

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model runtimes for text, image, and audio embeddings"""

    def __init__(self, config: ServerConfig) -> None:
        """
        Initialize embedding manager

        Args:
            config: Server configuration
        """
        self.config = config
        
        # Keep separate runtimes for different embedding types
        self.text_runtime: BaseEmbeddingRuntime | None = None
        self.multimodal_runtime: BaseEmbeddingRuntime | None = None  # For text + images
        self.audio_runtime: BaseEmbeddingRuntime | None = None
        
        # Track loaded models
        self.loaded_text_model: str | None = None
        self.loaded_multimodal_model: str | None = None
        self.loaded_audio_model: str | None = None
        
        # Model discovery and downloading
        self.available_models: Dict[str, List[str]] = {}
        self.downloader = ModelDownloader(
            config.model_path, config.cache_dir or config.model_path / "cache"
        )
        
        # Idle cleanup task
        self._idle_task: asyncio.Task | None = None
        self.download_status: dict[str, str] = {}  # Track download status for models
        
        # Track warm-up status
        self.warmup_status = {
            "in_progress": False,
            "text": "pending",      # pending/loading/ready/failed
            "multimodal": "pending", # pending/loading/ready/failed  
            "audio": "pending",     # pending/loading/ready/failed
            "current_task": "",
            "start_time": None,
        }

    async def initialize(self) -> None:
        """Initialize embedding manager and optionally warm up models"""
        if not self.config.enable_embeddings:
            logger.info("Embeddings disabled in configuration")
            return
            
        logger.info(f"Initializing embedding manager with models from {self.config.model_path}")
        
        # Discover available embedding models
        self._discover_embedding_models()
        logger.info(f"Available embedding models: {self.available_models}")

        # Start idle cleanup monitoring
        self._schedule_idle_cleanup()

        if self.config.warm_up:
            logger.info("Starting background embedding model warm-up...")
            asyncio.create_task(self._warm_up_models())

    def _discover_embedding_models(self) -> None:
        """Discover available embedding models in the model directory"""
        embeddings_dir = self.config.model_path / "embeddings"
        
        if not embeddings_dir.exists():
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
        self.available_models = {
            "text": [],
            "multimodal": [],
            "audio": []
        }
        
        # Scan for embedding models
        for model_dir in embeddings_dir.iterdir():
            if model_dir.is_dir():
                model_dir_name = model_dir.name
                
                # Convert directory name back to HuggingFace model ID
                model_name = model_dir_name.replace("_", "/")
                
                # Categorize by model name patterns
                if "clip" in model_name.lower() or "multimodal" in model_name.lower():
                    self.available_models["multimodal"].append(model_name)
                elif "whisper" in model_name.lower():
                    self.available_models["audio"].append(model_name)
                else:
                    # Default to text model
                    self.available_models["text"].append(model_name)

    async def _warm_up_models(self) -> None:
        """Warm up embedding models in the background"""
        try:
            self.warmup_status["in_progress"] = True
            self.warmup_status["start_time"] = time.time()
            self.warmup_status["current_task"] = "Starting embedding model warm-up..."
            
            logger.info("Starting comprehensive embedding model warm-up...")

            # Phase 1: Download models if needed
            await self._download_models_if_needed()
            
            # Phase 2: Pre-load models
            await self._preload_models()
            
            self.warmup_status["in_progress"] = False
            elapsed = time.time() - self.warmup_status["start_time"]
            logger.info(f"Embedding model warm-up completed in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Error during embedding model warm-up: {e}")
            self.warmup_status["in_progress"] = False

    async def _download_models_if_needed(self) -> None:
        """Download embedding models if not available locally from HuggingFace"""
        # Check for default text embedding model
        if self.config.embedding_model not in self.available_models.get("text", []):
            logger.info(f"Text embedding model '{self.config.embedding_model}' not found locally, downloading...")
            await self._download_huggingface_model(
                self.config.embedding_model, 
                "text"
            )
            
        # Check for multimodal model
        if self.config.multimodal_model not in self.available_models.get("multimodal", []):
            logger.info(f"Multimodal model '{self.config.multimodal_model}' not found locally, downloading...")
            await self._download_huggingface_model(
                self.config.multimodal_model, 
                "multimodal"
            )

    async def _preload_models(self) -> None:
        """Pre-load embedding models for faster first requests"""
        tasks = []
        
        # Pre-load text model
        if self.available_models.get("text"):
            model_name = self.config.embedding_model
            if model_name in self.available_models["text"]:
                tasks.append(self._load_text_model(model_name))
        
        # Pre-load multimodal model
        if self.available_models.get("multimodal"):
            model_name = self.config.multimodal_model
            # Use first available if configured model not found
            available_multimodal = self.available_models["multimodal"]
            if available_multimodal:
                actual_model = model_name if model_name in available_multimodal else available_multimodal[0]
                tasks.append(self._load_multimodal_model(actual_model))
        
        # Pre-load audio model if available
        if self.available_models.get("audio"):
            model_name = self.available_models["audio"][0]
            tasks.append(self._load_audio_model(model_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_huggingface_model(self, model_id: str, model_type: str) -> None:
        """Download model from HuggingFace Hub"""
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            # Use the existing executor or create one
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EmbeddingDownload")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, self._download_huggingface_model_sync, model_id, model_type)
            
            # Refresh available models after download
            self._discover_embedding_models()
            
        except Exception as e:
            logger.error(f"Failed to download model '{model_id}': {e}")
            raise

    def _download_huggingface_model_sync(self, model_id: str, model_type: str) -> None:
        """Synchronous HuggingFace model download"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            # Create target directory
            target_dir = self.config.model_path / "embeddings" / model_id.replace("/", "_")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {model_type} model '{model_id}' to {target_dir}")
            
            # Download the model using sentence-transformers
            # This will cache it in the HuggingFace cache, then we copy to our structure
            model = SentenceTransformer(model_id)
            
            # Save to our directory structure
            model.save(str(target_dir))
            
            logger.info(f"Successfully downloaded model '{model_id}'")
            
        except Exception as e:
            logger.error(f"Error downloading model '{model_id}': {e}")
            raise

    async def get_text_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate text embeddings"""
        if not self.config.enable_embeddings:
            raise RuntimeError("Embeddings are disabled")
            
        target_model = model_name or self.config.embedding_model
        
        # Load model if not already loaded or different model requested
        if self.loaded_text_model != target_model:
            await self._load_text_model(target_model)
        
        if not self.text_runtime or not self.text_runtime.is_loaded:
            raise RuntimeError("Text embedding model not available")
        
        return await self.text_runtime.embed(texts, EmbeddingType.TEXT)

    async def get_image_embeddings(
        self, 
        image_bytes: bytes,
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate image embeddings"""
        if not self.config.enable_embeddings:
            raise RuntimeError("Embeddings are disabled")
            
        target_model = model_name or self.config.multimodal_model
        
        # Load model if not already loaded or different model requested
        if self.loaded_multimodal_model != target_model:
            await self._load_multimodal_model(target_model)
        
        if not self.multimodal_runtime or not self.multimodal_runtime.is_loaded:
            raise RuntimeError("Multimodal embedding model not available")
        
        return await self.multimodal_runtime.embed(image_bytes, EmbeddingType.IMAGE)

    async def get_audio_embeddings(
        self, 
        audio_bytes: bytes,
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate audio embeddings"""
        if not self.config.enable_embeddings:
            raise RuntimeError("Embeddings are disabled")
            
        # Use first available audio model if none specified
        if not model_name:
            if not self.available_models.get("audio"):
                raise RuntimeError("No audio embedding models available")
            model_name = self.available_models["audio"][0]
        
        # Load model if not already loaded or different model requested
        if self.loaded_audio_model != model_name:
            await self._load_audio_model(model_name)
        
        if not self.audio_runtime or not self.audio_runtime.is_loaded:
            raise RuntimeError("Audio embedding model not available")
        
        return await self.audio_runtime.embed(audio_bytes, EmbeddingType.AUDIO)

    async def _load_text_model(self, model_name: str) -> None:
        """Load text embedding model"""
        try:
            self.warmup_status["text"] = "loading"
            self.warmup_status["current_task"] = f"Loading text embedding model '{model_name}'..."
            
            # Convert HuggingFace model ID to directory name
            model_dir_name = model_name.replace("/", "_")
            model_path = self.config.model_path / "embeddings" / model_dir_name
            
            # If model doesn't exist locally, download it
            if not model_path.exists():
                logger.info(f"Model {model_name} not found locally, downloading...")
                await self._download_huggingface_model(model_name, "text")
            
            # Create runtime
            runtime = OpenVINOEmbeddingRuntime(
                model_path=model_path,
                cache_dir=self.config.cache_dir,
                device=self.config.embedding_device,
                model_type="text"
            )
            
            await runtime.load()
            
            # Replace current runtime
            if self.text_runtime:
                await self.text_runtime.unload()
            
            self.text_runtime = runtime
            self.loaded_text_model = model_name
            self.warmup_status["text"] = "ready"
            
            logger.info(f"Text embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            self.warmup_status["text"] = "failed"
            logger.error(f"Failed to load text embedding model '{model_name}': {e}")
            raise

    async def _load_multimodal_model(self, model_name: str) -> None:
        """Load multimodal embedding model"""
        try:
            self.warmup_status["multimodal"] = "loading"
            self.warmup_status["current_task"] = f"Loading multimodal embedding model '{model_name}'..."
            
            model_path = self.config.model_path / "embeddings" / model_name
            
            # Create runtime
            runtime = OpenVINOEmbeddingRuntime(
                model_path=model_path,
                cache_dir=self.config.cache_dir,
                device=self.config.embedding_device,
                model_type="multimodal"
            )
            
            await runtime.load()
            
            # Replace current runtime
            if self.multimodal_runtime:
                await self.multimodal_runtime.unload()
            
            self.multimodal_runtime = runtime
            self.loaded_multimodal_model = model_name
            self.warmup_status["multimodal"] = "ready"
            
            logger.info(f"Multimodal embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            self.warmup_status["multimodal"] = "failed"
            logger.error(f"Failed to load multimodal embedding model '{model_name}': {e}")
            raise

    async def _load_audio_model(self, model_name: str) -> None:
        """Load audio embedding model"""
        try:
            self.warmup_status["audio"] = "loading"
            self.warmup_status["current_task"] = f"Loading audio embedding model '{model_name}'..."
            
            model_path = self.config.model_path / "embeddings" / model_name
            
            # Create runtime
            runtime = WhisperEmbeddingRuntime(
                model_path=model_path,
                cache_dir=self.config.cache_dir,
                device=self.config.embedding_device,
                max_memory_gb=self.config.max_memory_gb
            )
            
            await runtime.load()
            
            # Replace current runtime
            if self.audio_runtime:
                await self.audio_runtime.unload()
            
            self.audio_runtime = runtime
            self.loaded_audio_model = model_name
            self.warmup_status["audio"] = "ready"
            
            logger.info(f"Audio embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            self.warmup_status["audio"] = "failed"
            logger.error(f"Failed to load audio embedding model '{model_name}': {e}")
            raise

    def _schedule_idle_cleanup(self) -> None:
        """Schedule idle cleanup task"""
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_cleanup_loop())

    async def _idle_cleanup_loop(self) -> None:
        """Background task to unload idle models"""
        while True:
            try:
                await asyncio.sleep(self.config.idle_check_interval_seconds)
                await self._check_and_unload_idle_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in embedding idle cleanup: {e}")

    async def _check_and_unload_idle_models(self) -> None:
        """Check for idle models and unload them"""
        idle_timeout = self.config.idle_timeout_minutes * 60  # Convert to seconds
        
        runtimes = [
            ("text", self.text_runtime),
            ("multimodal", self.multimodal_runtime),
            ("audio", self.audio_runtime)
        ]
        
        for runtime_type, runtime in runtimes:
            if runtime and runtime.is_loaded and runtime.get_idle_time() > idle_timeout:
                logger.info(f"Unloading idle {runtime_type} embedding model: {runtime.model_name}")
                await runtime.unload()
                
                # Reset loaded model tracking
                if runtime_type == "text":
                    self.loaded_text_model = None
                    self.text_runtime = None
                elif runtime_type == "multimodal":
                    self.loaded_multimodal_model = None
                    self.multimodal_runtime = None
                elif runtime_type == "audio":
                    self.loaded_audio_model = None
                    self.audio_runtime = None

    def get_embedding_info(self) -> dict[str, Any]:
        """Get information about loaded embedding models"""
        info = {
            "enabled": self.config.enable_embeddings,
            "available_models": self.available_models,
            "warmup_status": self.warmup_status,
            "loaded_models": {
                "text": self.loaded_text_model,
                "multimodal": self.loaded_multimodal_model,
                "audio": self.loaded_audio_model
            },
            "runtime_info": {}
        }
        
        # Add runtime info for loaded models
        if self.text_runtime:
            info["runtime_info"]["text"] = self.text_runtime.get_info()
        if self.multimodal_runtime:
            info["runtime_info"]["multimodal"] = self.multimodal_runtime.get_info()
        if self.audio_runtime:
            info["runtime_info"]["audio"] = self.audio_runtime.get_info()
            
        return info

    async def shutdown(self) -> None:
        """Shutdown embedding manager and unload all models"""
        logger.info("Shutting down embedding manager...")
        
        # Cancel idle cleanup task
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        
        # Unload all models
        tasks = []
        if self.text_runtime:
            tasks.append(self.text_runtime.unload())
        if self.multimodal_runtime:
            tasks.append(self.multimodal_runtime.unload())
        if self.audio_runtime:
            tasks.append(self.audio_runtime.unload())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Embedding manager shutdown complete")