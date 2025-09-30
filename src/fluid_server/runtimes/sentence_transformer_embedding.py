"""
Sentence-Transformers runtime for text embedding generation
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from .base_embedding import BaseEmbeddingRuntime, EmbeddingType

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerEmbeddingRuntime(BaseEmbeddingRuntime):
    """Sentence-Transformers runtime for generating text embeddings"""

    def __init__(self, model_id: str, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path=model_path, cache_dir=cache_dir, device=device)
        self.model_id = model_id
        self.model: Any | None = None
        self._SentenceTransformer: Any | None = None
        self.model_lock = Lock()
        self._embedding_dim: int | None = None

        # Convert device to torch format
        self.torch_device = self._convert_device(device)

    def _convert_device(self, device: str) -> str:
        """Convert device string to torch format"""
        device_upper = device.upper()
        if device_upper == "GPU":
            return "cuda"
        elif device_upper in ["CPU", "AUTO"]:
            return "cpu"
        else:
            return device.lower()

    async def load(self) -> None:
        """Load the sentence-transformer model for embedding generation"""
        if self.is_loaded:
            logger.debug("Embedding model %s already loaded", self.model_id)
            return

        try:
            await asyncio.to_thread(self._load_sync)
            self.is_loaded = True
            self.update_last_used()
            logger.info("Loaded sentence-transformer embedding model '%s'", self.model_id)
        except Exception as exc:
            logger.error("Failed to load sentence-transformer embedding model '%s': %s", self.model_id, exc)
            raise

    def _load_sync(self) -> None:
        """Synchronously load the sentence-transformer model"""
        from sentence_transformers import SentenceTransformer

        self._SentenceTransformer = SentenceTransformer

        model_name = self.model_id or DEFAULT_EMBEDDING_MODEL

        with self.model_lock:
            # Use cache_dir for model caching
            cache_folder = str(self.cache_dir.resolve()) if self.cache_dir else None

            self.model = self._SentenceTransformer(
                model_name,
                device=self.torch_device,
                cache_folder=cache_folder
            )

            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            self._embedding_dim = len(test_embedding)

        self.last_used = time.time()

    async def unload(self) -> None:
        """Unload the embedding model"""
        if not self.is_loaded:
            return

        with self.model_lock:
            # Move model to CPU and clear reference
            if self.model is not None:
                self.model = self.model.to("cpu")
                self.model = None

        self.is_loaded = False
        self._embedding_dim = None
        self.update_last_used()
        logger.info("Unloaded sentence-transformer embedding model '%s'", self.model_id)

    def get_info(self) -> dict[str, Any]:
        """Return runtime info"""
        return {
            "runtime_type": "sentence_transformer_embedding",
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "device": self.device,
            "torch_device": self.torch_device,
            "embedding_dimension": self._embedding_dim,
        }

    async def embed(
        self,
        inputs: str | Sequence[str] | bytes,
        embedding_type: EmbeddingType,
    ) -> list[list[float]]:
        """Generate embeddings for the provided text inputs"""
        if embedding_type != EmbeddingType.TEXT:
            raise ValueError("SentenceTransformerEmbeddingRuntime only supports text embeddings")

        if isinstance(inputs, bytes):
            raise ValueError("Text embeddings require string inputs")

        if isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)

        if not texts:
            return []

        if not self.is_loaded:
            await self.load()

        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronously generate embeddings"""
        with self.model_lock:
            if self.model is None:
                raise RuntimeError("Embedding model not loaded")

            self.update_last_used()

            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32
            )

            # Convert numpy arrays to lists if needed
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()

            # Ensure we have a list of lists
            if len(texts) == 1 and not isinstance(embeddings[0], list):
                embeddings = [embeddings]

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension"""
        return self._embedding_dim or 0

    def get_supported_types(self) -> list[EmbeddingType]:
        """Embedding types supported by this runtime"""
        return [EmbeddingType.TEXT]
