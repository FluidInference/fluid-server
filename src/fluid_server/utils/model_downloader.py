"""
Model downloading utilities for automatic model management
"""

import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

# Default model repositories for each type
DEFAULT_MODEL_REPOS = {
    "llm": {
        "qwen3-8b-int8-ov": "FluidInference/Qwen3-8B-int8-ov",
        "qwen3-8b-int4-ov": "FluidInference/Qwen3-8B-int4-ov",
        "qwen3-4b-int8-ov": "FluidInference/Qwen3-4B-int8-ov",
        "phi-4-mini": "FluidInference/phi-4-mini-instruct-int4-ov-npu",
    },
    "whisper": {
        "whisper-tiny": "FluidInference/whisper-tiny-int4-ov-npu",
        "whisper-large-v3-turbo-fp16-ov-npu": "FluidInference/whisper-large-v3-turbo-fp16-ov-npu",
        "whisper-large-v3-turbo-qnn": "FluidInference/whisper-large-v3-turbo-qnn",
    },
}


class ModelDownloader:
    """Handles automatic downloading of models from HuggingFace Hub"""

    def __init__(self, model_path: Path, cache_dir: Path):
        """
        Initialize model downloader

        Args:
            model_path: Base path for model storage
            cache_dir: Cache directory for downloads
        """
        self.model_path = model_path
        self.cache_dir = cache_dir

    async def download_default_model(self, model_type: str, model_name: str) -> Path | None:
        """
        Download a default model if it doesn't exist locally

        Args:
            model_type: Type of model (llm, whisper, embedding)
            model_name: Name of the model to download

        Returns:
            Path to downloaded model or None if download failed
        """
        # Check if we have a default repo for this model
        repo_mapping = DEFAULT_MODEL_REPOS.get(model_type, {})
        repo_id = repo_mapping.get(model_name)

        if not repo_id:
            logger.warning(
                f"No default repository configured for {model_type} model '{model_name}'"
            )
            return None

        # Check if model already exists locally
        model_dir = self.model_path / model_type / model_name
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Model {model_name} already exists at {model_dir}")
            return model_dir

        try:
            logger.info(f"Auto-downloading {model_type} model '{model_name}' from {repo_id}")

            # Create model directory
            model_dir.mkdir(parents=True, exist_ok=True)

            # Verify repo exists before downloading
            api = HfApi()
            try:
                repo_info = api.repo_info(repo_id)
                logger.info(f"Found repository: {repo_info.id}")
            except Exception as e:
                logger.error(f"Repository {repo_id} not found on HuggingFace Hub: {e}")
                return None

            # Download the model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                resume_download=True,
                local_dir_use_symlinks=False,  # Copy files for Windows compatibility
            )

            logger.info(f"Successfully downloaded {model_name} to {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            logger.error(f"Failed to download {model_type} model '{model_name}': {e}")
            # Clean up failed download directory if it's empty
            try:
                if model_dir.exists() and not any(model_dir.iterdir()):
                    model_dir.rmdir()
            except Exception:
                pass
            return None

    async def ensure_model_available(self, model_type: str, model_name: str) -> bool:
        """
        Ensure a model is available locally, downloading if necessary

        Args:
            model_type: Type of model (llm, whisper, embedding)
            model_name: Name of the model

        Returns:
            True if model is available, False otherwise
        """
        model_path = await self.download_default_model(model_type, model_name)
        return model_path is not None
