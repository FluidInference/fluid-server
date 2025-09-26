"""
llama-cpp runtime for text embedding generation
"""

import asyncio
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Sequence, Union

from .base_embedding import BaseEmbeddingRuntime, EmbeddingType

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_REPO_ID = "unsloth/embeddinggemma-300m-GGUF"
DEFAULT_EMBEDDING_FILENAME = "embeddinggemma-300M-BF16.gguf"

class LlamaCppEmbeddingRuntime(BaseEmbeddingRuntime):
    """llama-cpp runtime for generating text embeddings from GGUF models"""

    def __init__(self, model_id: str, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path=model_path, cache_dir=cache_dir, device=device)
        self.model_id = model_id
        self.llama: Any | None = None
        self._Llama: Any | None = None
        self.model_lock = Lock()
        self._embedding_dim: Optional[int] = None

    async def load(self) -> None:
        """Load the GGUF model for embedding generation"""
        if self.is_loaded:
            logger.debug("Embedding model %s already loaded", self.model_id)
            return

        try:
            await asyncio.to_thread(self._load_sync)
            self.is_loaded = True
            self.update_last_used()
            logger.info("Loaded llama-cpp embedding model '%s'", self.model_id)
        except Exception as exc:
            logger.error("Failed to load llama-cpp embedding model '%s': %s", self.model_id, exc)
            raise

    def _load_sync(self) -> None:
        """Synchronously load the llama-cpp model"""
        from llama_cpp import Llama

        self._Llama = Llama
        repo_id, filename, model_file = self._resolve_model_sources()

        if repo_id is None and model_file is None:
            repo_id = DEFAULT_EMBEDDING_REPO_ID
            filename = DEFAULT_EMBEDDING_FILENAME
        elif repo_id == DEFAULT_EMBEDDING_REPO_ID and filename is None:
            filename = DEFAULT_EMBEDDING_FILENAME

        load_kwargs: dict[str, Any] = {
            "embedding": True,
            "n_ctx": 0,
            "n_batch": 512,
            "verbose": False,
            "n_gpu_layers": 99 if self.device.upper() == "GPU" else 0,
        }

        with self.model_lock:
            if model_file:
                self.llama = self._Llama(
                    model_path=str(model_file.resolve()),
                    **load_kwargs,
                )
            else:
                if repo_id is None:
                    raise ValueError(
                        f"Unable to resolve model source for '{self.model_id}'."
                    )

                cache_dir = self.model_path
                cache_dir.mkdir(parents=True, exist_ok=True)

                if filename:
                    self.llama = self._Llama.from_pretrained(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(cache_dir.resolve()),
                        **load_kwargs,
                    )
                else:
                    self.llama = self._Llama.from_pretrained(
                        repo_id=repo_id,
                        cache_dir=str(cache_dir.resolve()),
                        **load_kwargs,
                    )

        self.last_used = time.time()

    async def unload(self) -> None:
        """Unload the embedding model"""
        if not self.is_loaded:
            return

        with self.model_lock:
            self.llama = None

        self.is_loaded = False
        self._embedding_dim = None
        self.update_last_used()
        logger.info("Unloaded llama-cpp embedding model '%s'", self.model_id)

    def get_info(self) -> dict[str, Any]:
        """Return runtime info"""
        return {
            "runtime_type": "llamacpp_embedding",
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "device": self.device,
            "embedding_dimension": self._embedding_dim,
        }

    async def embed(
        self,
        inputs: Union[str, Sequence[str], bytes],
        embedding_type: EmbeddingType,
    ) -> List[List[float]]:
        """Generate embeddings for the provided text inputs"""
        if embedding_type != EmbeddingType.TEXT:
            raise ValueError("LlamaCppEmbeddingRuntime only supports text embeddings")

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

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronously generate embeddings"""
        with self.model_lock:
            if self.llama is None:
                raise RuntimeError("Embedding model not loaded")

            self.update_last_used()
            result = self.llama.create_embedding(input=texts)

        data = result.get("data", []) if isinstance(result, dict) else []
        embeddings = [entry.get("embedding", []) for entry in data]

        if embeddings and self._embedding_dim is None:
            self._embedding_dim = len(embeddings[0])

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension"""
        return self._embedding_dim or 0

    def get_supported_types(self) -> List[EmbeddingType]:
        """Embedding types supported by this runtime"""
        return [EmbeddingType.TEXT]

    def _resolve_model_sources(self) -> tuple[Optional[str], Optional[str], Optional[Path]]:
        """Resolve local or remote sources for the model"""
        # 1. If model_path points directly to a GGUF file
        if self.model_path.is_file() and self.model_path.suffix == ".gguf":
            return None, None, self.model_path

        # 2. If model_path is a directory containing a GGUF file
        if self.model_path.is_dir():
            gguf_files = sorted(self.model_path.glob("*.gguf"))
            if gguf_files:
                return None, None, gguf_files[0]

        # 3. If model_id is a filesystem path
        potential_path = Path(self.model_id)
        if potential_path.exists():
            if potential_path.is_file() and potential_path.suffix == ".gguf":
                return None, None, potential_path
            if potential_path.is_dir():
                gguf_files = sorted(potential_path.glob("*.gguf"))
                if gguf_files:
                    return None, None, gguf_files[0]

        # 4. Interpret model_id as HuggingFace repo information
        if "/" in self.model_id:
            parts = self.model_id.split("/")
            if parts[-1].endswith(".gguf"):
                repo_id = "/".join(parts[:-1])
                filename = parts[-1]
                return repo_id, filename, None
            return self.model_id, None, None

        return DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_FILENAME, None
