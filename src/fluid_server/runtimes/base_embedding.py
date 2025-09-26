"""
Base embedding runtime class for all embedding model backends
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Union
from enum import Enum


class EmbeddingType(Enum):
    """Supported embedding types"""
    TEXT = "text"


class BaseEmbeddingRuntime(ABC):
    """Base class for all embedding model runtimes"""

    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        """
        Initialize embedding runtime

        Args:
            model_path: Path to the model directory
            cache_dir: Path to cache compiled models
            device: Device to run on (CPU, GPU, NPU)
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = device
        self.is_loaded = False
        self.last_used = time.time()

    @abstractmethod
    async def load(self) -> None:
        """Load the embedding model into memory"""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model to free memory"""
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Get runtime and model information"""
        pass

    @abstractmethod
    async def embed(
        self, 
        inputs: Union[str, List[str]], 
        embedding_type: EmbeddingType
    ) -> List[List[float]]:
        """
        Generate embeddings for the given inputs
        
        Args:
            inputs: Text string(s)
            embedding_type: Type of embedding to generate
            
        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[EmbeddingType]:
        """Get list of supported embedding types for this runtime"""
        pass

    def update_last_used(self) -> None:
        """Update the last used timestamp"""
        self.last_used = time.time()

    def get_idle_time(self) -> float:
        """Get how long the runtime has been idle in seconds"""
        return time.time() - self.last_used

    @property
    def model_name(self) -> str:
        """Get the model name from path"""
        return self.model_path.name

    def supports_type(self, embedding_type: EmbeddingType) -> bool:
        """Check if this runtime supports a specific embedding type"""
        return embedding_type in self.get_supported_types()