"""
Server configuration management
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ServerConfig:
    """Simple server configuration"""
    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    
    # Model paths
    model_path: Path = Path("./models")  # Base path for all models
    cache_dir: Optional[Path] = None     # Defaults to model_path/cache
    
    # Model selection
    llm_model: str = "qwen3-8b"          # Which LLM to load
    whisper_model: str = "whisper-tiny"  # Which Whisper to load
    
    # Runtime
    device: str = "GPU"                  # CPU, GPU, or NPU
    
    # Features
    preload_models: bool = False         # Preload on startup
    max_memory_gb: float = 8.0          # Memory limit
    idle_timeout_minutes: int = 5       # Idle timeout for unloading
    
    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    
    def __post_init__(self) -> None:
        """Set defaults after initialization"""
        self.model_path = Path(self.model_path)
        if self.cache_dir is None:
            self.cache_dir = self.model_path / "cache"
        else:
            self.cache_dir = Path(self.cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)