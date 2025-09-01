"""
OpenVINO runtime for LLM models
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


class OpenVINOLLMRuntime(BaseRuntime):
    """OpenVINO runtime for LLM models"""

    # Class-level dedicated thread pool for LLM operations
    _llm_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated LLM thread pool"""
        if cls._llm_executor is None:
            cls._llm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="LLM")
        return cls._llm_executor

    def __init__(self, model_path: Path, cache_dir: Path, device: str) -> None:
        super().__init__(model_path, cache_dir, device)
        self.pipeline: Any | None = None
        self.pipeline_lock = Lock()
        self.last_used = time.time()

    async def load(self) -> None:
        """Load the LLM model"""
        if self.is_loaded:
            logger.debug(f"Model {self.model_name} already loaded")
            return

        try:
            # Lazy imports - only import when actually loading
            import openvino as ov
            import openvino_genai as ov_genai
            
            self.ov = ov
            self.ov_genai = ov_genai

            # Set OpenVINO logging level
            try:
                ov.properties.log.level = 0  # type: ignore
            except AttributeError:
                # Fallback for older OpenVINO versions
                pass

            # Use model-specific cache subdirectory
            model_cache = self.cache_dir / "llm" / self.model_name
            model_cache.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading LLM '{self.model_name}' from {self.model_path}")
            logger.info(f"Using cache at {model_cache}")
            logger.info(f"Device: {self.device}")

            start_time = time.time()

            # Check if compiled cache exists
            cache_files = list(model_cache.glob("*.blob"))
            has_cache = len(cache_files) > 0

            if has_cache:
                logger.info(f"Found {len(cache_files)} cached blob(s)")
            else:
                logger.info("No cached model found, will compile on first run")

            # Create scheduler config for memory optimization
            scheduler_config = self.ov_genai.SchedulerConfig()
            scheduler_config.enable_prefix_caching = False
            scheduler_config.num_kv_blocks = 1024
            scheduler_config.cache_size = 5  # GB
            scheduler_config.use_cache_eviction = True

            # Configure cache eviction
            cache_eviction_config = self.ov_genai.CacheEvictionConfig(
                start_size=256,
                recent_size=1024,
                max_cache_size=4096,
                aggregation_mode=self.ov_genai.AggregationMode.NORM_SUM,
            )
            scheduler_config.cache_eviction_config = cache_eviction_config

            # Load with configuration
            config_dict = {
                "scheduler_config": scheduler_config,
                "CACHE_DIR": str(model_cache),
                "KV_CACHE_PRECISION": "u8",
            }

            self.pipeline = self.ov_genai.LLMPipeline(self.model_path, self.device, **config_dict)

            load_time = time.time() - start_time
            logger.info(f"LLM loaded successfully in {load_time:.1f}s")

            # Warm up if no cache
            if not has_cache:
                logger.info("Running warm-up generation...")
                warmup_config = self.ov_genai.GenerationConfig()
                warmup_config.max_new_tokens = 1
                self.pipeline.generate("Hello", warmup_config)
                logger.info("Warm-up complete, cache created")

            self.is_loaded = True
            self.last_used = time.time()

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.pipeline = None
            raise

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Synchronous text generation for use in executor"""
        with self.pipeline_lock:
            if self.pipeline is None:
                raise RuntimeError("Pipeline not initialized")

            self.last_used = time.time()

            try:
                config = self.ov_genai.GenerationConfig()
                config.max_new_tokens = max_tokens
                config.temperature = temperature
                config.top_p = top_p
                config.repetition_penalty = 1.1
                config.do_sample = True

                result = self.pipeline.generate(prompt, config)

                # Extract text from result
                if isinstance(result, str):
                    return result
                else:
                    return str(result)

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, images: list[str] | None = None) -> str:
        """Generate text completion"""
        if not self.is_loaded:
            await self.load()

        # Images not supported in OpenVINO LLM runtime
        if images:
            logger.warning("Images passed to OpenVINO LLM runtime but not supported - ignoring images")

        # Run generation in dedicated LLM executor to avoid blocking
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

        config = self.ov_genai.GenerationConfig()
        config.max_new_tokens = max_tokens
        config.temperature = temperature
        config.top_p = top_p
        config.repetition_penalty = 1.1
        config.do_sample = True

        def streamer_callback(token: str) -> bool:
            """Callback for streaming tokens - sends to queue immediately"""
            if token:  # Only send non-empty tokens
                if not _safe_queue_put(token):
                    return True  # Stop generation on queue error
            return False  # Continue generation

        try:
            with self.pipeline_lock:
                if self.pipeline is None:
                    raise RuntimeError("Pipeline not initialized")

                self.last_used = time.time()
                self.pipeline.generate(prompt, config, streamer_callback)
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
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, images: list[str] | None = None
    ) -> AsyncIterator[str]:
        """Stream text generation token by token using async queue"""
        if not self.is_loaded:
            await self.load()

        # Images not supported in OpenVINO LLM runtime
        if images:
            logger.warning("Images passed to OpenVINO LLM runtime but not supported - ignoring images")

        # Create queue for token passing
        token_queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 tokens
        loop = asyncio.get_event_loop()

        try:
            # Start synchronous generation in dedicated LLM executor
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
        with self.pipeline_lock:
            if self.pipeline:
                logger.info(f"Unloading LLM model '{self.model_name}'")
                self.pipeline = None
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
            "runtime": "openvino",
            "last_used": self.last_used if self.is_loaded else None,
        }
