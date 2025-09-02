"""
QNN runtime for Whisper models using Qualcomm NPU via QNNExecutionProvider
"""

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseRuntime
from ..utils.platform_utils import get_architecture, is_runtime_available

logger = logging.getLogger(__name__)


class QNNEncoderWrapper:
    """ONNX Runtime wrapper for QNN Whisper encoder"""
    
    def __init__(self, encoder_path: Path):
        logger.info("Initializing QNN Whisper encoder")
        
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError("onnxruntime-qnn not available. Install with: uv add onnxruntime-qnn") from e
        
        # Configure ONNX Runtime session
        options = onnxruntime.SessionOptions()  # type: ignore
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True
        
        # QNN provider configuration for Snapdragon X Elite
        provider_options = [{
            "backend_path": r"QnnHtp.dll",
            "htp_performance_mode": "burst",
            "enable_htp_fp16_precision": "1",
            "high_power_saver": "sustained_high_performance",
            "htp_graph_finalization_optimization_mode": "3"
        }]
        
        try:
            self.session = onnxruntime.InferenceSession(
                str(encoder_path),
                sess_options=options,
                providers=["QNNExecutionProvider"],
                provider_options=provider_options
            )
            
            # Disable fallback to CPU - forces QNN-only execution
            self.session.disable_fallback()
            logger.info("✓ QNN encoder session created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create QNN encoder session: {e}")
            raise
    
    def __call__(self, mel_input: np.ndarray) -> list:
        """Run encoder inference"""
        # Ensure input is float16 as expected by QNN models
        if mel_input.dtype != np.float16:
            mel_input = mel_input.astype(np.float16)
        
        input_name = "input_features"
        
        try:
            outputs = self.session.run(None, {input_name: mel_input})
            return outputs
        except Exception as e:
            logger.error(f"Encoder execution failed: {e}")
            logger.error(f"Input shape: {mel_input.shape}, dtype: {mel_input.dtype}")
            raise


class QNNDecoderWrapper:
    """ONNX Runtime wrapper for QNN Whisper decoder"""
    
    def __init__(self, decoder_path: Path):
        logger.info("Initializing QNN Whisper decoder")
        
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError("onnxruntime-qnn not available. Install with: uv add onnxruntime-qnn") from e
        
        # Configure ONNX Runtime session
        options = onnxruntime.SessionOptions()  # type: ignore
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True
        
        # QNN provider configuration for Snapdragon X Elite
        provider_options = [{
            "backend_path": r"QnnHtp.dll",
            "htp_performance_mode": "burst",
            "enable_htp_fp16_precision": "1", 
            "high_power_saver": "sustained_high_performance",
            "htp_graph_finalization_optimization_mode": "3"
        }]
        
        try:
            self.session = onnxruntime.InferenceSession(
                str(decoder_path),
                sess_options=options,
                providers=["QNNExecutionProvider"],
                provider_options=provider_options
            )
            
            # Disable fallback to CPU - forces QNN-only execution
            self.session.disable_fallback()
            logger.info("✓ QNN decoder session created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create QNN decoder session: {e}")
            raise
    
    def __call__(self, input_dict: dict) -> list:
        """Run decoder inference"""
        try:
            outputs = self.session.run(None, input_dict)
            return outputs
        except Exception as e:
            logger.error(f"Decoder execution failed: {e}")
            traceback.print_exc()
            raise


class QNNWhisperRuntime(BaseRuntime):
    """QNN runtime for Whisper models using Qualcomm NPU"""
    
    # Class-level dedicated thread pool for QNN operations
    _qnn_executor: ThreadPoolExecutor | None = None
    
    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated QNN thread pool"""
        if cls._qnn_executor is None:
            cls._qnn_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="QNN")
        return cls._qnn_executor
    
    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path, cache_dir, device)
        self.encoder: QNNEncoderWrapper | None = None
        self.decoder: QNNDecoderWrapper | None = None
        self.tokenizer: Any | None = None
        self.last_used = time.time()
        self._load_lock = asyncio.Lock()
        
        # Check architecture compatibility
        arch = get_architecture()
        if not is_runtime_available("qnn", arch):
            logger.warning(f"QNN runtime not supported on {arch} architecture, will fail at load time")
        
        # QNN model configuration
        self.num_decoder_blocks = 4    # Large-v3-turbo reduced blocks
        self.num_decoder_heads = 20    # decoder_attention_heads
        self.attention_dim = 1280      # d_model
        self.mean_decode_len = 200     # Max sequence length
        self.device_variant = "snapdragon-x-elite"  # Default device variant
    
    async def load(self) -> None:
        """Load QNN Whisper model"""
        async with self._load_lock:
            if self.is_loaded and self.encoder is not None and self.decoder is not None:
                logger.debug(f"QNN Whisper model {self.model_name} already loaded")
                self.last_used = time.time()
                return
            
            # Check architecture compatibility
            arch = get_architecture()
            if not is_runtime_available("qnn", arch):
                raise RuntimeError(
                    f"QNN runtime requires ARM64 architecture, but running on {arch}. "
                    "QNN is only supported on Windows ARM64 devices with Snapdragon X Elite."
                )
            
            # Lazy import of dependencies
            try:
                import torch
                import whisper
                self.torch = torch
                self.whisper = whisper
            except ImportError as e:
                raise RuntimeError(
                    "QNN runtime requires PyTorch and Whisper. "
                    "Install with: pip install torch whisper"
                ) from e
            
            logger.info(f"Loading QNN Whisper '{self.model_name}' from {self.model_path}")
            logger.info(f"Device variant: {self.device_variant}")
            logger.info(f"Architecture: {arch}")
            
            # Validate model files exist
            if not self._validate_model_files():
                raise ValueError(f"Invalid QNN Whisper model at {self.model_path}")
            
            try:
                # Get device-specific model paths
                device_path = self.model_path / self.device_variant
                encoder_path = device_path / "encoder" / "model.onnx"
                decoder_path = device_path / "decoder" / "model.onnx"
                
                logger.info(f"Loading encoder from: {encoder_path}")
                logger.info(f"Loading decoder from: {decoder_path}")
                
                start_time = time.time()
                
                # Initialize encoder and decoder
                self.encoder = QNNEncoderWrapper(encoder_path)
                self.decoder = QNNDecoderWrapper(decoder_path)
                
                # Initialize tokenizer
                from whisper.decoding import get_tokenizer
                self.tokenizer = get_tokenizer(
                    multilingual=True, task="transcribe"
                )
                
                load_time = time.time() - start_time
                logger.info(f"✓ QNN Whisper loaded successfully in {load_time:.1f}s")
                
                # Run warmup
                await self._warmup_model()
                
                self.is_loaded = True
                self.last_used = time.time()
                logger.info(f"QNN Whisper model '{self.model_name}' ready for transcription")
                
            except Exception as e:
                logger.error(f"Failed to load QNN Whisper model: {e}")
                self.encoder = None
                self.decoder = None
                self.tokenizer = None
                raise
    
    async def transcribe(
        self, audio_data: bytes, language: str | None = None, return_timestamps: bool = True
    ) -> dict[str, Any]:
        """Transcribe audio to text using QNN Whisper"""
        if not self.is_loaded or self.encoder is None or self.decoder is None:
            await self.load()
        
        self.last_used = time.time()
        
        try:
            import io
            import wave
            
            logger.debug(f"Processing audio data: {len(audio_data)} bytes")
            
            # Load WAV file
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                
                logger.info(
                    f"Audio format: {channels} channel(s), {sample_rate}Hz, {sample_width} bytes per sample"
                )
                
                # Convert to float32 array
                if sample_width == 2:  # 16-bit signed
                    audio_float32 = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                elif sample_width == 4:  # 32-bit (try float first, fallback to int)
                    try:
                        audio_float32 = np.frombuffer(frames, dtype=np.float32)
                        if np.max(np.abs(audio_float32)) > 1.1:
                            audio_float32 = (
                                np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                            )
                    except (ValueError, TypeError):
                        audio_float32 = (
                            np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        )
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Convert stereo to mono if needed
                if channels == 2:
                    audio_float32 = audio_float32.reshape(-1, 2).mean(axis=1)
                elif channels > 2:
                    audio_float32 = audio_float32.reshape(-1, channels)[:, 0]
            
            logger.info(
                f"Audio loaded: {len(audio_float32)} samples at {sample_rate}Hz, "
                f"duration: {len(audio_float32) / sample_rate:.2f}s"
            )
            
            # Run transcription in dedicated QNN thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.get_executor(), self._transcribe_sync, audio_float32, return_timestamps
            )
            
            return result
            
        except Exception as e:
            logger.error(f"QNN transcription failed: {e}")
            raise
    
    async def unload(self) -> None:
        """Unload QNN model to free memory"""
        async with self._load_lock:
            if self.encoder is not None or self.decoder is not None:
                logger.info(f"Unloading QNN Whisper model '{self.model_name}'")
                try:
                    self.encoder = None
                    self.decoder = None
                    self.tokenizer = None
                    self.is_loaded = False
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    logger.info(f"QNN Whisper model '{self.model_name}' unloaded successfully")
                except Exception as e:
                    logger.error(f"Error during QNN unload: {e}")
                    self.encoder = None
                    self.decoder = None
                    self.tokenizer = None
                    self.is_loaded = False
    
    def get_info(self) -> dict[str, Any]:
        """Get runtime information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "loaded": self.is_loaded,
            "runtime": "qnn",
            "type": "whisper",
            "device_variant": self.device_variant,
            "last_used": self.last_used if hasattr(self, "last_used") else None,
        }
    
    def _validate_model_files(self) -> bool:
        """Validate that required QNN model files exist"""
        device_path = self.model_path / self.device_variant
        encoder_path = device_path / "encoder" / "model.onnx"
        decoder_path = device_path / "decoder" / "model.onnx"
        
        if not encoder_path.exists():
            logger.error(f"Missing QNN encoder model: {encoder_path}")
            return False
        
        if not decoder_path.exists():
            logger.error(f"Missing QNN decoder model: {decoder_path}")
            return False
        
        logger.debug("All required QNN Whisper model files found")
        return True
    
    async def _warmup_model(self) -> None:
        """Run a warmup transcription to ensure QNN compilation completes"""
        try:
            logger.info("Running QNN warmup transcription...")
            start_time = time.time()
            
            # Create 1 second of silence (16kHz sample rate)
            silent_audio = np.zeros(16000, dtype=np.float32)
            
            # Run warmup in dedicated thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.get_executor(), self._transcribe_sync, silent_audio, False
            )
            
            warmup_time = time.time() - start_time
            logger.info(f"✓ QNN warmup completed in {warmup_time:.1f}s")
            
        except Exception as e:
            logger.warning(f"QNN warmup failed: {e} - continuing anyway")
    
    def _transcribe_sync(self, audio_array: np.ndarray, return_timestamps: bool) -> dict[str, Any]:
        """Synchronous transcription method for thread pool execution"""
        try:
            start_time = time.time()
            
            # Process audio in 30-second windows (Whisper standard)
            window_size = 30 * 16000  # 30 seconds at 16kHz
            full_transcript = []
            
            for i in range(0, len(audio_array), window_size):
                chunk = audio_array[i:i + window_size]
                
                # Pad or trim chunk to exactly 30 seconds
                chunk = self.whisper.pad_or_trim(chunk)
                
                # Generate mel spectrogram with 128 bins for turbo model
                mel = self.whisper.log_mel_spectrogram(chunk, n_mels=128).to("cpu")
                
                # Encode using QNN encoder
                mel_batch = mel.unsqueeze(0).numpy().astype(np.float16)  # Shape: (1, 128, time_steps)
                
                try:
                    encoder_output = self.encoder(mel_batch)
                    
                    # Pair outputs into (k, v) tuples for each layer
                    kv_cache_cross = tuple(
                        (encoder_output[j], encoder_output[j+1])
                        for j in range(0, len(encoder_output), 2)
                    )
                    
                    # Decode tokens
                    tokens = self._decode_tokens(kv_cache_cross)
                    
                    # Decode text for this chunk
                    chunk_text = self.tokenizer.decode(tokens).strip()
                    
                    # Strip out language and special tokens
                    import re
                    chunk_text = re.sub(r'<\|[^|]*\|>', '', chunk_text).strip()
                    
                    if chunk_text:
                        full_transcript.append(chunk_text)
                        
                except Exception as e:
                    logger.error(f"Failed to process audio chunk: {e}")
                    continue
            
            # Combine all chunks
            final_text = " ".join(full_transcript)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            audio_duration = len(audio_array) / 16000.0
            rtfx = audio_duration / processing_time if processing_time > 0 else 0
            
            logger.info(f"QNN transcription completed: {rtfx:.2f}x real-time")
            
            # Create segments
            segments = []
            if return_timestamps:
                segments.append({"text": final_text, "start": 0.0, "end": audio_duration})
            
            return {
                "text": final_text,
                "language": None,  # Let Whisper detect language
                "segments": segments if return_timestamps else None,
                "duration": audio_duration,
            }
            
        except Exception as e:
            logger.error(f"QNN sync transcription failed: {e}")
            raise
    
    def _decode_tokens(self, kv_cache_cross: tuple) -> list[int]:
        """Decode tokens using QNN decoder"""
        # Initialize with start-of-transcript token
        output_ids = self.torch.tensor([[self.tokenizer.sot]], dtype=self.torch.int32)
        decoded_tokens = []
        
        # Initialize caches
        sample_len = self.mean_decode_len
        attention_dim = self.attention_dim
        num_decoder_heads = self.num_decoder_heads
        num_decoder_blocks = self.num_decoder_blocks
        
        # Initialize self attention caches
        k_cache_self = self.torch.zeros((
            num_decoder_heads, 1, attention_dim // num_decoder_heads, sample_len - 1,
        ), dtype=self.torch.float16)
        
        v_cache_self = self.torch.zeros((
            num_decoder_heads, 1, sample_len - 1, attention_dim // num_decoder_heads,
        ), dtype=self.torch.float16)
        
        kv_cache_self = tuple(
            (k_cache_self, v_cache_self) for _ in range(num_decoder_blocks)
        )
        
        # Initialize attention mask
        mask_neg = -100.0
        attention_mask = self.torch.full(
            (1, 1, 1, sample_len), mask_neg, dtype=self.torch.float16,
        )
        
        for n in range(sample_len - 1):
            try:
                # Get current token
                input_ids = output_ids[:, n : n + 1]
                position_ids = self.torch.tensor([n], dtype=self.torch.int32)
                
                # Update attention mask
                attention_mask[:, :, :, sample_len - n - 1] = 0.0
                
                # Build decoder input dict
                decoder_input_dict = {
                    "input_ids": input_ids.numpy().astype(np.int32),
                    "attention_mask": attention_mask.numpy().astype(np.float16),
                    "position_ids": position_ids.numpy().astype(np.int32),
                }
                
                # Add self attention caches
                for i, (k, v) in enumerate(kv_cache_self):
                    k_array = k.numpy() if hasattr(k, 'numpy') else k
                    v_array = v.numpy() if hasattr(v, 'numpy') else v
                    decoder_input_dict[f"k_cache_self_{i}_in"] = k_array.astype(np.float16)
                    decoder_input_dict[f"v_cache_self_{i}_in"] = v_array.astype(np.float16)
                
                # Add cross attention caches
                for i, (k, v) in enumerate(kv_cache_cross):
                    k_array = k.astype(np.float16) if k.dtype != np.float16 else k
                    v_array = v.astype(np.float16) if v.dtype != np.float16 else v
                    decoder_input_dict[f"k_cache_cross_{i}"] = k_array
                    decoder_input_dict[f"v_cache_cross_{i}"] = v_array
                
                # Run decoder
                decoder_output = self.decoder(decoder_input_dict)
                
                if len(decoder_output) != 9:
                    logger.error(f"Expected 9 outputs, got {len(decoder_output)}")
                    break
                
                # Extract outputs
                logits = decoder_output[0]
                
                # Reconstruct kv_cache_self from outputs
                kv_cache_self = tuple(
                    (self.torch.from_numpy(decoder_output[1 + i*2]), 
                     self.torch.from_numpy(decoder_output[1 + i*2 + 1]))
                    for i in range(num_decoder_blocks)
                )
                
                # Get next token
                if logits.shape[0] == 0:
                    logger.error("Empty logits tensor received")
                    break
                
                logits_tensor = self.torch.from_numpy(logits).squeeze()
                output_id = self.torch.argmax(logits_tensor, dim=-1)
                
                if output_id == self.tokenizer.eot:
                    break
                
                # Ensure proper shape for concatenation
                if output_id.dim() == 0:
                    output_id = output_id.unsqueeze(0).unsqueeze(0)
                elif output_id.dim() == 1:
                    output_id = output_id.unsqueeze(0)
                
                output_ids = self.torch.cat((output_ids, output_id), -1)
                decoded_tokens.append(int(output_id.item()))
                
            except Exception as e:
                logger.error(f"Decoder failed at token {n}: {e}")
                break
        
        return decoded_tokens