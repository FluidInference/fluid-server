"""
Server configuration management
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    """Simple server configuration"""

    # Server
    host: str = "127.0.0.1"
    port: int = 8080

    # Model paths
    model_path: Path = Path("./models")  # Base path for all models
    cache_dir: Path | None = None  # Defaults to model_path/cache

    # Model selection
    llm_model: str = "qwen3-8b-int4-ov"  # Which LLM to load
    whisper_model: str = "whisper-large-v3-turbo-fp16-ov-npu"  # Which Whisper to load

    # Features
    warm_up: bool = True  # Warm up models on startup
    max_memory_gb: float = 8.0  # Memory limit
    idle_timeout_minutes: int = 5  # Idle timeout for unloading

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.95

    # Private field to track if __post_init__ has been called
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Set defaults after initialization"""
        # Prevent multiple initialization
        if self._initialized:
            return

        # Convert to Path object and ensure it's absolute
        path_obj = Path(self.model_path)
        if not path_obj.is_absolute():
            self.model_path = path_obj.resolve()
        else:
            self.model_path = path_obj

        if self.cache_dir is None:
            self.cache_dir = self.model_path / "cache"
        else:
            cache_path_obj = Path(self.cache_dir)
            if not cache_path_obj.is_absolute():
                self.cache_dir = cache_path_obj.resolve()
            else:
                self.cache_dir = cache_path_obj

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Mark as initialized
        self._initialized = True
