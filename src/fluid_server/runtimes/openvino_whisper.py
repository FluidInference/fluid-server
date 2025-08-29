"""
OpenVINO runtime for Whisper models using openvino_genai.WhisperPipeline
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .base import BaseRuntime

logger = logging.getLogger(__name__)


class TranscriptionSegment:
    """Represents a transcribed segment with timing"""

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start_time, "end": self.end_time}


class OpenVINOWhisperRuntime(BaseRuntime):
    """OpenVINO runtime for Whisper models using openvino_genai"""

    # Class-level dedicated thread pool for Whisper operations
    _whisper_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated Whisper thread pool"""
        if cls._whisper_executor is None:
            cls._whisper_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Whisper")
        return cls._whisper_executor

    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path, cache_dir, device)
        self.pipeline: Any | None = None
        self.last_used = time.time()
        self._load_lock = asyncio.Lock()
        self._model_id = "FluidInference/whisper-large-v3-turbo-fp16-ov-npu"
        self._repetition_penalty = 1.5

        # Create model-specific cache directory
        self.model_cache_dir = self.cache_dir / "whisper" / self.model_name
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    async def load(self) -> None:
        """Load the Whisper model using openvino_genai.WhisperPipeline"""
        async with self._load_lock:
            if self.is_loaded and self.pipeline is not None:
                logger.debug(f"Whisper model {self.model_name} already loaded")
                self.last_used = time.time()
                return

            logger.info(f"Loading Whisper '{self.model_name}' from {self.model_path}")
            logger.info(f"Using cache at {self.model_cache_dir}")
            logger.info(f"Device: {self.device}")

            # Validate model exists
            if not self._validate_model_files():
                raise ValueError(f"Invalid Whisper model at {self.model_path}")

            # Set OpenVINO environment variables for optimization
            import os

            os.environ["OPENVINO_THREADING"] = "SEQ"
            os.environ["OPENVINO_CACHE_MODE"] = "OPTIMIZE_SPEED"
            os.environ["OV_CACHE_DIR"] = str(self.model_cache_dir)

            try:
                # Lazy import - only import when actually loading
                import openvino_genai as ov_genai
                self.ov_genai = ov_genai

                logger.info(f"Creating WhisperPipeline (Device: {self.device})")
                start_time = time.time()

                try:
                    # Try primary device (NPU)
                    self.pipeline = self.ov_genai.WhisperPipeline(
                        models_path=self.model_path,
                        device=self.device,
                        CACHE_DIR=str(self.model_cache_dir),
                    )
                    load_time = time.time() - start_time
                    logger.info(f"✓ WhisperPipeline loaded on {self.device} in {load_time:.1f}s")

                except Exception as e:
                    logger.warning(f"Failed to load on {self.device}: {e}")
                    if self.device.upper() != "CPU":
                        logger.info("Falling back to CPU")
                        start_time = time.time()
                        self.pipeline = self.ov_genai.WhisperPipeline(
                            models_path=self.model_path,
                            device="CPU",
                            CACHE_DIR=str(self.model_cache_dir),
                        )
                        load_time = time.time() - start_time
                        logger.info(f"✓ WhisperPipeline loaded on CPU in {load_time:.1f}s")
                    else:
                        raise

                # Run warmup to complete NPU compilation
                await self._warmup_model()

                self.is_loaded = True
                self.last_used = time.time()
                logger.info(f"Whisper model '{self.model_name}' ready for transcription")

            except ImportError as e:
                logger.error(f"Failed to import openvino_genai: {e}")
                raise ValueError(
                    "openvino_genai not available. Install with: pip install openvino-genai"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

    async def transcribe(
        self, audio_data: bytes, language: str | None = None, return_timestamps: bool = True
    ) -> dict[str, Any]:
        """Transcribe audio to text using WhisperPipeline"""
        if not self.is_loaded or self.pipeline is None:
            await self.load()

        self.last_used = time.time()

        try:
            # Assume audio is already in 16kHz mono format as requested by client
            import io
            import wave

            import numpy as np

            logger.debug(f"Processing audio data: {len(audio_data)} bytes")

            # Load WAV file and assume it's already 16kHz mono
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                # Get audio properties for validation
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())

                # Log the actual format for debugging
                logger.info(
                    f"Audio format: {channels} channel(s), {sample_rate}Hz, {sample_width} bytes per sample"
                )

                # Convert to float32 array based on sample width
                if sample_width == 1:
                    # 8-bit unsigned
                    audio_float32 = (
                        np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
                    )
                elif sample_width == 2:
                    # 16-bit signed
                    audio_float32 = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                elif sample_width == 3:
                    # 24-bit (rare, treat as 32-bit)
                    # Pad 24-bit to 32-bit
                    padded_frames = np.frombuffer(frames, dtype=np.uint8)
                    padded_frames = np.pad(
                        padded_frames.reshape(-1, 3), ((0, 0), (0, 1)), mode="constant"
                    )
                    audio_float32 = padded_frames.view(np.int32).astype(np.float32) / 2147483648.0
                elif sample_width == 4:
                    # 32-bit signed integer or float
                    try:
                        # Try as float32 first
                        audio_float32 = np.frombuffer(frames, dtype=np.float32)
                        # Ensure values are in [-1, 1] range
                        if np.max(np.abs(audio_float32)) > 1.1:
                            # Likely int32, convert
                            audio_float32 = (
                                np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                                / 2147483648.0
                            )
                    except (ValueError, TypeError):
                        # Fall back to int32
                        audio_float32 = (
                            np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        )
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                # Convert stereo to mono if needed (simple averaging)
                if channels == 2:
                    audio_float32 = audio_float32.reshape(-1, 2).mean(axis=1)
                elif channels > 2:
                    # For multi-channel, take first channel
                    audio_float32 = audio_float32.reshape(-1, channels)[:, 0]

                # Use the actual sample rate (assume client converted correctly)
                sample_rate = sample_rate

            logger.info(
                f"Audio loaded: {len(audio_float32)} samples at {sample_rate}Hz, duration: {len(audio_float32) / sample_rate:.2f}s"
            )

            # Run transcription in dedicated Whisper thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.get_executor(), self._transcribe_sync, audio_float32, return_timestamps
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def unload(self) -> None:
        """Unload model to free memory"""
        async with self._load_lock:
            if self.pipeline is not None:
                logger.info(f"Unloading Whisper model '{self.model_name}'")
                try:
                    # Clear pipeline reference
                    self.pipeline = None
                    self.is_loaded = False

                    # Force garbage collection
                    import gc

                    gc.collect()

                    logger.info(f"Whisper model '{self.model_name}' unloaded successfully")
                except Exception as e:
                    logger.error(f"Error during unload: {e}")
                    self.pipeline = None
                    self.is_loaded = False

    def get_info(self) -> dict[str, Any]:
        """Get runtime information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "loaded": self.is_loaded,
            "runtime": "openvino_genai",
            "type": "whisper",
            "model_id": self._model_id,
            "cache_dir": str(self.model_cache_dir),
            "last_used": self.last_used if hasattr(self, "last_used") else None,
        }

    def _validate_model_files(self) -> bool:
        """Validate that required Whisper model files exist"""
        required_files = ["openvino_encoder_model.xml", "openvino_decoder_model.xml"]

        for file_name in required_files:
            if not (self.model_path / file_name).exists():
                logger.error(f"Missing required file: {file_name}")
                return False

        logger.debug("All required Whisper model files found")
        return True

    async def _warmup_model(self) -> None:
        """Run a warmup transcription to ensure NPU compilation completes"""
        try:
            logger.info("Running warmup transcription for NPU compilation...")
            start_time = time.time()

            # Create 1 second of silence (16kHz sample rate)
            import numpy as np

            silent_audio = np.zeros(16000, dtype=np.float32)

            # Run warmup in dedicated Whisper thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.get_executor(), self._transcribe_sync, silent_audio, False
            )

            warmup_time = time.time() - start_time
            logger.info(f"✓ Warmup completed in {warmup_time:.1f}s - NPU compilation verified")

        except Exception as e:
            logger.warning(f"Warmup failed: {e} - continuing anyway")

    def _transcribe_sync(self, audio_array, return_timestamps: bool) -> dict[str, Any]:
        """Synchronous transcription method for thread pool execution"""
        try:
            config_params = {
                "return_timestamps": return_timestamps,
                "task": "transcribe",
                "repetition_penalty": self._repetition_penalty,
                "temperature": 0.0,
            }

            logger.info(f"Running transcription with config: {config_params}")

            # Run transcription
            result = self.pipeline.generate(audio_array, **config_params)

            # OpenVINO GenAI WhisperDecodedResults always has 'texts' attribute
            if result.texts:
                if isinstance(result.texts, list):
                    full_text = " ".join(str(text).strip() for text in result.texts)
                else:
                    full_text = str(result.texts).strip()
            else:
                full_text = ""

            logger.info(f"Transcribed text: '{full_text}'")

            # Handle chunks for timestamps if needed
            segments = []
            if return_timestamps and hasattr(result, "chunks") and result.chunks:
                logger.info(f"Processing {len(result.chunks)} chunks for timestamps")
                for i, chunk in enumerate(result.chunks):
                    text = str(chunk.text).strip() if hasattr(chunk, "text") else ""
                    start_time = float(chunk.start_ts) if hasattr(chunk, "start_ts") else 0.0
                    end_time = float(chunk.end_ts) if hasattr(chunk, "end_ts") else 0.0

                    logger.debug(f"Chunk {i}: '{text}' [{start_time:.2f}s - {end_time:.2f}s]")

                    if text:
                        segments.append({"text": text, "start": start_time, "end": end_time})
            else:
                # Single segment for the entire transcription
                segments.append(
                    {"text": full_text, "start": 0.0, "end": len(audio_array) / 16000.0}
                )

            return {
                "text": full_text,
                "language": None,  # Let Whisper detect language
                "segments": segments if return_timestamps else None,
                "duration": len(audio_array) / 16000.0,
            }

        except Exception as e:
            logger.error(f"Sync transcription failed: {e}")
            raise
