"""
Base runtime class for all AI model backends
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseRuntime(ABC):
    """Base class for all model runtimes"""

    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        """
        Initialize runtime

        Args:
            model_path: Path to the model directory
            cache_dir: Path to cache compiled models
            device: Device to run on (CPU, GPU, NPU)
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = device
        self.is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load the model into memory"""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model to free memory"""
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Get runtime and model information"""
        pass

    async def transcribe(
        self, audio_data: bytes, language: str | None = None, return_timestamps: bool = True
    ) -> dict[str, Any]:
        """Transcribe audio to text (only implemented by Whisper runtimes)"""
        raise NotImplementedError("Transcription not supported by this runtime")

    @property
    def model_name(self) -> str:
        """Get the model name from path"""
        return self.model_path.name
