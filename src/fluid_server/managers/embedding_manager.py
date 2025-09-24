"""
Embedding manager for llama-cpp text embeddings
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import ServerConfig
from ..runtimes.base_embedding import BaseEmbeddingRuntime, EmbeddingType
from ..runtimes.llamacpp_embedding import LlamaCppEmbeddingRuntime

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage llama-cpp based text embedding runtimes"""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config

        self.text_runtime: BaseEmbeddingRuntime | None = None
        self.loaded_text_model: str | None = None
        self.available_models: Dict[str, List[str]] = {"text": []}

        self._idle_task: asyncio.Task | None = None
        self._load_lock = asyncio.Lock()

        # Placeholder to keep API parity with runtime manager
        self.download_status: dict[str, str] = {}

        self.warmup_status = {
            "in_progress": False,
            "text": "pending",
            "current_task": "",
            "start_time": None,
        }

    async def initialize(self) -> None:
        """Initialise the embedding manager and optionally warm up models"""
        if not self.config.enable_embeddings:
            logger.info("Embeddings disabled in configuration")
            return

        embeddings_dir = self.config.model_path / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        discovered = self._discover_text_models(embeddings_dir)
        if self.config.embedding_model not in discovered:
            discovered.append(self.config.embedding_model)
        self.available_models["text"] = list(dict.fromkeys(discovered))  # Preserve order

        self._schedule_idle_cleanup()

        if self.config.warm_up:
            logger.info("Starting background embedding model warm-up...")
            asyncio.create_task(self._warm_up_models())

    def _discover_text_models(self, embeddings_dir: Path) -> List[str]:
        """Discover local llama-cpp embedding models"""
        models: List[str] = []

        for entry in embeddings_dir.iterdir():
            if entry.is_dir() and any(entry.glob("*.gguf")):
                models.append(entry.name.replace("_", "/"))
            elif entry.is_file() and entry.suffix == ".gguf":
                models.append(entry.name)

        return models

    async def _warm_up_models(self) -> None:
        """Load the default embedding model in the background"""
        try:
            self.warmup_status["in_progress"] = True
            self.warmup_status["start_time"] = time.time()
            await self._load_text_model(self.config.embedding_model)
            elapsed = time.time() - (self.warmup_status["start_time"] or time.time())
            logger.info("Embedding model warm-up completed in %.1fs", elapsed)
        except Exception as exc:
            self.warmup_status["text"] = "failed"
            self.warmup_status["current_task"] = f"Failed: {exc}"
            logger.error("Error during embedding model warm-up: %s", exc)
        finally:
            self.warmup_status["in_progress"] = False

    async def get_text_embeddings(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate text embeddings using llama-cpp"""
        if not self.config.enable_embeddings:
            raise RuntimeError("Embeddings are disabled")

        target_model = model_name or self.config.embedding_model

        if (
            self.loaded_text_model != target_model
            or self.text_runtime is None
            or not self.text_runtime.is_loaded
        ):
            await self._load_text_model(target_model)

        if not self.text_runtime:
            raise RuntimeError("Text embedding runtime failed to initialise")

        return await self.text_runtime.embed(texts, EmbeddingType.TEXT)

    async def _load_text_model(self, model_name: str) -> None:
        """Load the specified text embedding model"""
        async with self._load_lock:
            if (
                self.text_runtime
                and self.text_runtime.is_loaded
                and self.loaded_text_model == model_name
            ):
                return

            self.warmup_status["text"] = "loading"
            self.warmup_status["current_task"] = (
                f"Loading text embedding model '{model_name}'..."
            )

            model_dir = self.config.model_path / "embeddings" / model_name.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            runtime = LlamaCppEmbeddingRuntime(
                model_id=model_name,
                model_path=model_dir,
                cache_dir=self.config.cache_dir_resolved,
                device=self.config.embedding_device,
            )

            try:
                await runtime.load()
            except Exception as exc:
                self.warmup_status["text"] = "failed"
                logger.error("Failed to load text embedding model '%s': %s", model_name, exc)
                raise

            if self.text_runtime:
                await self.text_runtime.unload()

            self.text_runtime = runtime
            self.loaded_text_model = model_name
            self.warmup_status["text"] = "ready"
            self.warmup_status["current_task"] = ""

            if model_name not in self.available_models["text"]:
                self.available_models["text"].append(model_name)

            logger.info("Text embedding model '%s' loaded successfully", model_name)

    def _schedule_idle_cleanup(self) -> None:
        """Start idle cleanup loop if required"""
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_cleanup_loop())

    async def _idle_cleanup_loop(self) -> None:
        """Unload idle embedding runtimes"""
        while True:
            try:
                await asyncio.sleep(self.config.idle_check_interval_seconds)
                await self._check_and_unload_idle_models()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in embedding idle cleanup: %s", exc)

    async def _check_and_unload_idle_models(self) -> None:
        """Unload the text runtime if it has been idle for too long"""
        if not self.text_runtime or not self.text_runtime.is_loaded:
            return

        idle_timeout = self.config.idle_timeout_minutes * 60
        if self.text_runtime.get_idle_time() > idle_timeout:
            logger.info(
                "Unloading idle text embedding model: %s",
                self.text_runtime.model_name,
            )
            await self.text_runtime.unload()
            self.text_runtime = None
            self.loaded_text_model = None

    def get_embedding_info(self) -> dict[str, Any]:
        """Return information about available and loaded embedding models"""
        info = {
            "enabled": self.config.enable_embeddings,
            "available_models": self.available_models,
            "warmup_status": self.warmup_status,
            "loaded_models": {
                "text": self.loaded_text_model,
            },
            "runtime_info": {},
        }

        if self.text_runtime:
            info["runtime_info"]["text"] = self.text_runtime.get_info()

        return info

    async def shutdown(self) -> None:
        """Shutdown the embedding manager and unload runtimes"""
        logger.info("Shutting down embedding manager...")

        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass

        if self.text_runtime:
            await self.text_runtime.unload()
            self.text_runtime = None
            self.loaded_text_model = None

        logger.info("Embedding manager shutdown complete")
