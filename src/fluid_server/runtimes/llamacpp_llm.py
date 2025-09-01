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
try:
    # Optional mapping for default HF repos and GGUF filenames
    from ..utils.model_downloader import DEFAULT_MODEL_REPOS, GGUF_FILE_MAPPINGS
except Exception:
    # Keep runtime usable even if downloader utilities change
    DEFAULT_MODEL_REPOS = {"llm": {}}
    GGUF_FILE_MAPPINGS = {}

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

    def __init__(self, model_path: Path, cache_dir: Path, device: str, model_name: str) -> None:
        super().__init__(model_path, cache_dir, device)
        self.llama = None
        self.model_lock = Lock()
        self.last_used = time.time()
        self._model_name = model_name  # Store the actual model identifier

    @property
    def model_name(self) -> str:
        """Get the model name"""
        return self._model_name

    async def load(self) -> None:
        """Load the GGUF model with Vulkan backend using llama-cpp-python's from_pretrained"""
        if self.is_loaded:
            logger.debug(f"Model {self.model_name} already loaded")
            return
        try:
            # Lazy import - only import when actually loading
            from llama_cpp import Llama
            self.Llama = Llama

            start_time = time.time()
            
            # Parse repo_id and filename from model name or use mappings
            repo_id, gguf_filename = self._parse_model_identifier()

            if not repo_id:
                raise ValueError(
                    f"Could not determine repo_id for model '{self.model_name}'. "
                    f"Use 'repo_id/filename.gguf' or 'repo_id' format, or ensure model has mapping in DEFAULT_MODEL_REPOS."
                )

            logger.info(f"Loading GGUF model '{self.model_name}' via from_pretrained")
            logger.info(f"Repo: {repo_id}, File: {gguf_filename}")
            logger.info(f"Cache dir: {self.model_path} | Device: {self.device} (Vulkan backend)")

            # Configure for Vulkan backend
            n_gpu_layers = -1 if self.device == "GPU" else 0

            # Use llama-cpp's from_pretrained for all GGUF models
            if gguf_filename:
                # Explicit filename provided
                self.llama = self.Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=gguf_filename,
                    # Keep downloads within workspace to avoid default global cache permissions
                    cache_dir=str(self.model_path.resolve()),
                    # Allow resume; callers can set HF_HUB_ENABLE_HF_TRANSFER/HF_TOKEN via env if needed
                    resume_download=True,
                    # Generation params
                    n_ctx=4096,
                    n_batch=512,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            else:
                # Auto-detect GGUF file from repo
                self.llama = self.Llama.from_pretrained(
                    repo_id=repo_id,
                    # Keep downloads within workspace to avoid default global cache permissions
                    cache_dir=str(self.model_path.resolve()),
                    # Allow resume; callers can set HF_HUB_ENABLE_HF_TRANSFER/HF_TOKEN via env if needed
                    resume_download=True,
                    # Generation params
                    n_ctx=4096,
                    n_batch=512,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )

            load_time = time.time() - start_time
            logger.info(f"GGUF model loaded successfully in {load_time:.1f}s")

            self.is_loaded = True
            self.last_used = time.time()

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            self.llama = None
            raise

    def _parse_model_identifier(self) -> tuple[str | None, str | None]:
        """Parse model identifier to extract repo_id and filename
        
        Supports formats:
        - repo_owner/repo_name (auto-detect GGUF file)  
        - repo_owner/repo_name/filename.gguf (explicit file)
        - model_name (use mappings)
        
        Returns:
            Tuple of (repo_id, filename) or (None, None) if parsing fails
        """
        # Check if model_name contains '/' indicating repo format
        if "/" in self.model_name:
            parts = self.model_name.split("/")
            
            if len(parts) >= 3:
                # Format: repo_owner/repo_name/filename.gguf
                repo_id = "/".join(parts[:-1])  # Everything except the last part
                filename = parts[-1]            # Last part is the filename
                return repo_id, filename
            elif len(parts) == 2:
                # Format: repo_owner/repo_name (auto-detect GGUF file)
                repo_id = self.model_name
                filename = None  # Let llama-cpp auto-detect
                return repo_id, filename
        
        # Fallback to mappings for backward compatibility
        repo_id = DEFAULT_MODEL_REPOS.get("llm", {}).get(self.model_name)
        filename = GGUF_FILE_MAPPINGS.get(self.model_name)
        
        return repo_id, filename

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
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, token_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Synchronous streaming generation for use in executor - now uses queue"""
        pending_futures = []
        
        def _safe_queue_put(item, timeout=1.0):
            """Safely put item in queue and wait for completion"""
            try:
                future = asyncio.run_coroutine_threadsafe(
                    token_queue.put(item), loop
                )
                pending_futures.append(future)
                # Wait for the put operation to complete
                future.result(timeout=timeout)
                return True
            except Exception as e:
                logger.error(f"Failed to send item to queue: {e}")
                return False
        
        try:
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

                # Send tokens to queue as they're generated
                for output in stream:
                    if isinstance(output, dict) and "choices" in output:
                        token = output["choices"][0]["text"]
                        if token:  # Only send non-empty tokens
                            if not _safe_queue_put(token):
                                break  # Stop on queue error
                                
        except Exception as e:
            # Signal completion/error by sending None
            _safe_queue_put(None)
            raise
        finally:
            # Cancel any pending futures
            for future in pending_futures:
                if not future.done():
                    future.cancel()
            
            # Signal completion by sending None
            try:
                completion_future = asyncio.run_coroutine_threadsafe(
                    token_queue.put(None), loop
                )
                completion_future.result(timeout=1.0)
            except Exception as e:
                logger.error(f"Failed to send completion signal: {e}")

    async def generate_stream(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> AsyncIterator[str]:
        """Stream text generation token by token using async queue"""
        if not self.is_loaded:
            await self.load()

        # Create queue for token passing
        token_queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 tokens
        loop = asyncio.get_event_loop()

        try:
            # Start synchronous generation in dedicated executor
            task = loop.run_in_executor(
                self.get_executor(),
                self.generate_stream_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                token_queue,
                loop,
            )

            # Yield tokens as they become available
            while True:
                try:
                    # Get token with timeout to handle hung generators
                    token = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                    
                    if token is None:  # End of stream signal
                        break
                        
                    yield token
                    await asyncio.sleep(0)  # Allow other async operations
                    
                except asyncio.TimeoutError:
                    logger.warning("Token generation timeout - ending stream")
                    break
                except Exception as e:
                    logger.error(f"Error receiving token: {e}")
                    break

            # Ensure the executor task completes
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Generator task did not complete cleanly")
            except Exception as e:
                logger.error(f"Generator task error: {e}")

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
