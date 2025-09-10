"""
Whisper encoder runtime for audio embeddings
"""

import asyncio
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Union
from threading import Lock

from .base_embedding import BaseEmbeddingRuntime, EmbeddingType

logger = logging.getLogger(__name__)


class WhisperEmbeddingRuntime(BaseEmbeddingRuntime):
    """Whisper encoder runtime for audio embeddings"""

    # Class-level dedicated thread pool for audio embedding operations
    _audio_embedding_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated audio embedding thread pool"""
        if cls._audio_embedding_executor is None:
            cls._audio_embedding_executor = ThreadPoolExecutor(
                max_workers=2, 
                thread_name_prefix="AudioEmbedding"
            )
        return cls._audio_embedding_executor

    def __init__(
        self, 
        model_path: Path, 
        cache_dir: Path, 
        device: str,
        max_memory_gb: float = 4.0
    ) -> None:
        super().__init__(model_path, cache_dir, device)
        self.model: Any | None = None
        self.processor: Any | None = None
        self.model_lock = Lock()
        self.last_used = time.time()
        self.max_memory_gb = max_memory_gb
        self._embedding_dim = 1024  # Whisper encoder dimension
        self._load_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the Whisper encoder model"""
        async with self._load_lock:
            if self.is_loaded:
                logger.debug(f"Audio embedding model {self.model_name} already loaded")
                return

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.get_executor(), self._load_model)
                self.is_loaded = True
                logger.info(f"Audio embedding model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load audio embedding model {self.model_name}: {e}")
                self.is_loaded = False
                raise

    def _load_model(self) -> None:
        """Load model in thread executor"""
        try:
            # Lazy imports
            import whisper
            from transformers import WhisperProcessor, WhisperModel
            
            # Use model-specific cache subdirectory
            model_cache = self.cache_dir / "audio_embeddings" / self.model_name
            model_cache.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading audio embedding model '{self.model_name}' from {self.model_path}")
            logger.info(f"Using cache at {model_cache}")
            logger.info(f"Device: {self.device}")

            start_time = time.time()

            try:
                # Try to load using transformers first (for fine-tuned models)
                self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
                self.model = WhisperModel.from_pretrained(str(self.model_path))
                
                # Get embedding dimension from model config
                self._embedding_dim = self.model.config.d_model
                
                logger.info("Loaded Whisper model using transformers")
                
            except Exception as e:
                logger.warning(f"Failed to load with transformers: {e}")
                
                # Fallback to whisper library
                try:
                    # Map model path to whisper model name if it's a standard model
                    model_name = self._get_whisper_model_name()
                    self.model = whisper.load_model(model_name, device="cpu")
                    
                    logger.info(f"Loaded Whisper model '{model_name}' using whisper library")
                    
                except Exception as e2:
                    logger.error(f"Failed to load with whisper library: {e2}")
                    raise

            load_time = time.time() - start_time
            logger.info(f"Audio embedding model loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Error loading audio embedding model: {e}")
            raise

    def _get_whisper_model_name(self) -> str:
        """Map model path to whisper model name"""
        model_name_lower = self.model_name.lower()
        
        if "tiny" in model_name_lower:
            return "tiny"
        elif "base" in model_name_lower:
            return "base"
        elif "small" in model_name_lower:
            return "small"
        elif "medium" in model_name_lower:
            return "medium"
        elif "large" in model_name_lower:
            if "v3" in model_name_lower:
                return "large-v3"
            elif "v2" in model_name_lower:
                return "large-v2"
            else:
                return "large"
        else:
            # Default to base model
            return "base"

    async def unload(self) -> None:
        """Unload the model to free memory"""
        if not self.is_loaded:
            logger.debug(f"Audio embedding model {self.model_name} not loaded")
            return

        try:
            with self.model_lock:
                self.model = None
                self.processor = None
                self.is_loaded = False
            
            logger.info(f"Audio embedding model {self.model_name} unloaded")
        except Exception as e:
            logger.error(f"Error unloading audio embedding model: {e}")
            raise

    async def embed(
        self, 
        inputs: Union[str, List[str], bytes], 
        embedding_type: EmbeddingType
    ) -> List[List[float]]:
        """Generate embeddings for audio inputs"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        if embedding_type != EmbeddingType.AUDIO:
            raise ValueError(f"Whisper embedding model only supports audio inputs")

        self.update_last_used()

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(), 
                self._embed_sync, 
                inputs
            )
        except Exception as e:
            logger.error(f"Error generating audio embeddings: {e}")
            raise

    def _embed_sync(self, audio_bytes: bytes) -> List[List[float]]:
        """Synchronous audio embedding generation"""
        with self.model_lock:
            return self._embed_audio(audio_bytes)

    def _embed_audio(self, audio_bytes: bytes) -> List[List[float]]:
        """Generate audio embeddings using Whisper encoder"""
        try:
            import librosa
            import torch
            import io
            
            # Load audio from bytes
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            
            if hasattr(self, 'processor') and self.processor is not None:
                # Using transformers model
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    # Get encoder outputs (audio embeddings)
                    encoder_outputs = self.model.encoder(
                        inputs["input_features"],
                        output_hidden_states=True
                    )
                    
                    # Use the mean of the last hidden state as the audio embedding
                    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
                    
                return embeddings.cpu().numpy().tolist()
                
            else:
                # Using whisper library
                # Pad or truncate audio to 30 seconds (Whisper's expected length)
                target_length = 16000 * 30  # 30 seconds at 16kHz
                if len(audio_data) < target_length:
                    audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
                else:
                    audio_data = audio_data[:target_length]
                
                # Import whisper for log mel spectrogram
                import whisper
                
                # Convert to log-mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_data)
                
                # Get encoder features
                with torch.no_grad():
                    encoder_output = self.model.encoder(mel.unsqueeze(0))
                    # Mean pool across time dimension to get fixed-size embedding
                    embeddings = encoder_output.mean(dim=1)
                    
                return embeddings.cpu().numpy().tolist()
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self._embedding_dim

    def get_supported_types(self) -> List[EmbeddingType]:
        """Get supported embedding types"""
        return [EmbeddingType.AUDIO]

    def get_info(self) -> dict[str, Any]:
        """Get runtime information"""
        return {
            "runtime_type": "whisper_embedding",
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_types": [t.value for t in self.get_supported_types()],
            "embedding_dimension": self._embedding_dim,
            "idle_time": self.get_idle_time(),
            "max_memory_gb": self.max_memory_gb
        }