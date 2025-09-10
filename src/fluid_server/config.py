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
    port: int = 3847

    # Data paths
    data_root: Path = Path("./data")  # Root directory for all server data
    model_path: Path | None = None  # Defaults to data_root/models
    cache_dir: Path | None = None  # Defaults to data_root/cache
    embeddings_db_path: Path | None = None  # Defaults to data_root/databases

    # Model selection
    llm_model: str = "qwen3-8b-int4-ov"  # Which LLM to load
    whisper_model: str = "whisper-large-v3-turbo-fp16-ov-npu"  # Which Whisper to load
    device: str = "AUTO"  # Device for inference: AUTO, CPU, GPU, NPU

    # Embeddings configuration
    enable_embeddings: bool = True  # Enable embeddings functionality
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Default text embedding model (384 dim) 22M parameters
    embedding_device: str = "CPU"  # Device for embeddings: AUTO, CPU, GPU, NPU (CPU more stable for sentence-transformers)
    embeddings_db_name: str = "embeddings"  # LanceDB database name
    multimodal_model: str = "openai/clip-vit-base-patch32"  # For image embeddings
    
    # Features
    warm_up: bool = True  # Warm up models on startup
    max_memory_gb: float = 4.0  # Memory limit
    idle_timeout_minutes: int = 5  # Idle timeout for unloading
    idle_check_interval_seconds: int = 60  # How often to check for idle models

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

        # Convert data_root to Path object and ensure it's absolute
        data_root_obj = Path(self.data_root)
        if not data_root_obj.is_absolute():
            self.data_root = data_root_obj.resolve()
        else:
            self.data_root = data_root_obj

        # Set model_path default if not provided
        if self.model_path is None:
            self.model_path = self.data_root / "models"
        else:
            model_path_obj = Path(self.model_path)
            if not model_path_obj.is_absolute():
                self.model_path = model_path_obj.resolve()
            else:
                self.model_path = model_path_obj

        # Set cache_dir default if not provided
        if self.cache_dir is None:
            self.cache_dir = self.data_root / "cache"
        else:
            cache_path_obj = Path(self.cache_dir)
            if not cache_path_obj.is_absolute():
                self.cache_dir = cache_path_obj.resolve()
            else:
                self.cache_dir = cache_path_obj

        # Set embeddings_db_path default if not provided
        if self.embeddings_db_path is None:
            self.embeddings_db_path = self.data_root / "databases"
        else:
            db_path_obj = Path(self.embeddings_db_path)
            if not db_path_obj.is_absolute():
                self.embeddings_db_path = db_path_obj.resolve()
            else:
                self.embeddings_db_path = db_path_obj

        # Create necessary directories
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_embeddings:
            self.embeddings_db_path.mkdir(parents=True, exist_ok=True)

        # Mark as initialized
        self._initialized = True

    @property
    def model_path_resolved(self) -> Path:
        """Get model_path as guaranteed Path (after __post_init__)"""
        assert self.model_path is not None, "model_path should be set after __post_init__"
        return self.model_path

    @property
    def cache_dir_resolved(self) -> Path:
        """Get cache_dir as guaranteed Path (after __post_init__)"""
        assert self.cache_dir is not None, "cache_dir should be set after __post_init__"
        return self.cache_dir

    @property
    def embeddings_db_path_resolved(self) -> Path:
        """Get embeddings_db_path as guaranteed Path (after __post_init__)"""
        assert self.embeddings_db_path is not None, "embeddings_db_path should be set after __post_init__"
        return self.embeddings_db_path
