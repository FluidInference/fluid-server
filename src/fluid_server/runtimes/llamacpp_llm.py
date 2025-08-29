"""
llama-cpp runtime for GGUF LLM models with Vulkan backend
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from .base import BaseRuntime

logger = logging.getLogger(__name__)


class LlamaCppRuntime(BaseRuntime):
    """llama-cpp runtime for GGUF LLM models using Vulkan backend"""

    # Class-level dedicated thread pool for LlamaCpp operations
    _llamacpp_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated LlamaCpp thread pool"""
        if cls._llamacpp_executor is None:
            cls._llamacpp_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="LlamaCpp"
            )
        return cls._llamacpp_executor

    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path, cache_dir, device)
        self.llama = None
        self.model_lock = Lock()
        self.last_used = time.time()

    async def load(self) -> None:
        """Load the GGUF model with Vulkan backend"""
        if self.is_loaded:
            logger.debug(f"Model {self.model_name} already loaded")
            return

        try:
            # Lazy import - only import when actually loading
            from llama_cpp import Llama
            self.Llama = Llama

            # Find the GGUF file in the model directory
            gguf_files = list(self.model_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {self.model_path}")

            gguf_file = gguf_files[0]  # Use the first GGUF file found
            logger.info(f"Loading GGUF model '{self.model_name}' from {gguf_file}")
            logger.info(f"Device: {self.device} (Vulkan backend)")

            start_time = time.time()

            # Configure for Vulkan backend
            # Note: llama-cpp-python needs to be compiled with Vulkan support
            # Set n_gpu_layers to offload layers to GPU via Vulkan
            n_gpu_layers = -1 if self.device == "GPU" else 0

            self.llama = self.Llama(
                model_path=str(gguf_file),
                n_ctx=4096,  # Context length
                n_batch=512,  # Batch size
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=False,
                # Vulkan-specific settings would go here if supported
                # For now, rely on GPU layers offloading
            )

            load_time = time.time() - start_time
            logger.info(f"GGUF model loaded successfully in {load_time:.1f}s")

            self.is_loaded = True
            self.last_used = time.time()

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            self.llama = None
            raise

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Synchronous text generation for use in executor"""
        with self.model_lock:
            if self.llama is None:
                raise RuntimeError("Model not initialized")

            self.last_used = time.time()

            try:
                # Generate with llama-cpp
                result = self.llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.1,
                    echo=False,  # Don't echo the prompt
                    stream=False,
                )

                # Extract the generated text
                if isinstance(result, dict) and "choices" in result:
                    return result["choices"][0]["text"]
                else:
                    return str(result)

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Generate text completion"""
        if not self.is_loaded:
            await self.load()

        # Run generation in dedicated executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.get_executor(), self._generate_sync, prompt, max_tokens, temperature, top_p
        )

        return result

    def generate_stream_sync(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> list[str]:
        """Synchronous streaming generation for use in executor"""
        tokens: list[str] = []

        with self.model_lock:
            if self.llama is None:
                raise RuntimeError("Model not initialized")

            self.last_used = time.time()

            # Generate with streaming
            stream = self.llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1,
                echo=False,
                stream=True,
            )

            # Collect all tokens
            for output in stream:
                if isinstance(output, dict) and "choices" in output:
                    token = output["choices"][0]["text"]
                    if token:  # Only add non-empty tokens
                        tokens.append(token)

        return tokens

    async def generate_stream(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> AsyncIterator[str]:
        """Stream text generation token by token"""
        if not self.is_loaded:
            await self.load()

        try:
            # Run synchronous generation in dedicated executor
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(
                self.get_executor(),
                self.generate_stream_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
            )

            # Yield tokens one by one
            for token in tokens:
                yield token
                await asyncio.sleep(0)  # Allow other async operations

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise

    async def unload(self) -> None:
        """Unload model to free memory"""
        with self.model_lock:
            if self.llama:
                logger.info(f"Unloading GGUF model '{self.model_name}'")
                # llama-cpp doesn't have explicit unload, just delete reference
                self.llama = None
                self.is_loaded = False

                # Force garbage collection
                import gc

                gc.collect()

    def get_info(self) -> dict[str, Any]:
        """Get runtime information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "loaded": self.is_loaded,
            "runtime": "llamacpp",
            "backend": "vulkan" if self.device == "GPU" else "cpu",
            "last_used": self.last_used if self.is_loaded else None,
        }