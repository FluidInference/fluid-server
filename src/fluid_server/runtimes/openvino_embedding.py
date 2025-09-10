"""
OpenVINO runtime for text and image embedding models
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, List, Union, Optional
from PIL import Image
import io
import numpy as np

from .base_embedding import BaseEmbeddingRuntime, EmbeddingType

logger = logging.getLogger(__name__)


class OpenVINOEmbeddingRuntime(BaseEmbeddingRuntime):
    """OpenVINO runtime for text and image embedding models"""

    # Class-level dedicated thread pool for embedding operations
    _embedding_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated embedding thread pool"""
        if cls._embedding_executor is None:
            cls._embedding_executor = ThreadPoolExecutor(
                max_workers=2, 
                thread_name_prefix="Embedding"
            )
        return cls._embedding_executor

    def __init__(
        self, 
        model_path: Path, 
        cache_dir: Path, 
        device: str,
        model_type: str = "text"  # "text", "clip", "multimodal"
    ) -> None:
        super().__init__(model_path, cache_dir, device)
        self.model_type = model_type
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self.model_lock = Lock()
        self.last_used = time.time()
        self._embedding_dim: Optional[int] = None

    async def load(self) -> None:
        """Load the embedding model"""
        if self.is_loaded:
            logger.debug(f"Embedding model {self.model_name} already loaded")
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.get_executor(), self._load_model)
            self.is_loaded = True
            logger.info(f"Embedding model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            self.is_loaded = False
            raise

    def _load_model(self) -> None:
        """Load model in thread executor"""
        try:
            # Lazy imports
            import openvino as ov
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.ov = ov
            
            # Note: OpenVINO logging level configuration varies by version
            # Skip setting log level to avoid compatibility issues

            # Use model-specific cache subdirectory
            model_cache = self.cache_dir / "embeddings" / self.model_name
            model_cache.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading embedding model '{self.model_name}' from {self.model_path}")
            logger.info(f"Using cache at {model_cache}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model type: {self.model_type}")

            start_time = time.time()

            # Load based on model type
            if self.model_type == "text":
                self._load_text_model()
            elif self.model_type in ["clip", "multimodal"]:
                self._load_multimodal_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _load_text_model(self) -> None:
        """Load text-only embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load as SentenceTransformer first
            self.model = SentenceTransformer(str(self.model_path), device='cpu')
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self._embedding_dim = len(test_embedding[0])
            
        except Exception as e:
            logger.warning(f"Failed to load as SentenceTransformer: {e}")
            # Fallback to transformers
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModel.from_pretrained(str(self.model_path))
            
            # Get embedding dimension from model config
            self._embedding_dim = self.model.config.hidden_size

    def _load_multimodal_model(self) -> None:
        """Load multimodal (CLIP) embedding model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(str(self.model_path))
            self.model = CLIPModel.from_pretrained(str(self.model_path))
            
            # Get embedding dimension
            self._embedding_dim = self.model.config.text_config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    async def unload(self) -> None:
        """Unload the model to free memory"""
        if not self.is_loaded:
            logger.debug(f"Embedding model {self.model_name} not loaded")
            return

        try:
            with self.model_lock:
                self.model = None
                self.tokenizer = None
                self.processor = None
                self._embedding_dim = None
                self.is_loaded = False
            
            logger.info(f"Embedding model {self.model_name} unloaded")
        except Exception as e:
            logger.error(f"Error unloading embedding model: {e}")
            raise

    async def embed(
        self, 
        inputs: Union[str, List[str], bytes], 
        embedding_type: EmbeddingType
    ) -> List[List[float]]:
        """Generate embeddings for inputs"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        if not self.supports_type(embedding_type):
            raise ValueError(f"Model {self.model_name} does not support {embedding_type.value} embeddings")

        self.update_last_used()

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(), 
                self._embed_sync, 
                inputs, 
                embedding_type
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _embed_sync(
        self, 
        inputs: Union[str, List[str], bytes], 
        embedding_type: EmbeddingType
    ) -> List[List[float]]:
        """Synchronous embedding generation"""
        with self.model_lock:
            if embedding_type == EmbeddingType.TEXT:
                return self._embed_text(inputs)
            elif embedding_type == EmbeddingType.IMAGE:
                return self._embed_image(inputs)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def _embed_text(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        """Generate text embeddings"""
        if isinstance(inputs, str):
            inputs = [inputs]

        if hasattr(self.model, 'encode'):
            # SentenceTransformer model
            embeddings = self.model.encode(inputs, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Transformers model
            import torch
            
            tokens = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.cpu().numpy().tolist()

    def _embed_image(self, inputs: bytes) -> List[List[float]]:
        """Generate image embeddings using CLIP"""
        if self.model_type not in ["clip", "multimodal"]:
            raise ValueError("Image embeddings require CLIP/multimodal model")

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(inputs))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process image
            inputs_processed = self.processor(images=image, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs_processed)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self._embedding_dim is None:
            raise RuntimeError("Model not loaded or dimension not determined")
        return self._embedding_dim

    def get_supported_types(self) -> List[EmbeddingType]:
        """Get supported embedding types"""
        if self.model_type == "text":
            return [EmbeddingType.TEXT]
        elif self.model_type in ["clip", "multimodal"]:
            return [EmbeddingType.TEXT, EmbeddingType.IMAGE]
        else:
            return []

    def get_info(self) -> dict[str, Any]:
        """Get runtime information"""
        return {
            "runtime_type": "openvino_embedding",
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_types": [t.value for t in self.get_supported_types()],
            "embedding_dimension": self._embedding_dim,
            "idle_time": self.get_idle_time()
        }