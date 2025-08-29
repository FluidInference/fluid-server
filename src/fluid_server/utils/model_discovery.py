"""
Model discovery and validation utilities
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Discover and validate available models in directory structure"""

    @staticmethod
    def find_models(base_path: Path) -> dict[str, list[str]]:
        """
        Find all available models organized by type

        Args:
            base_path: Base directory containing model subdirectories

        Returns:
            Dictionary mapping model type to list of available model names
        """
        models: dict[str, list[str]] = {
            "llm": [],
            "whisper": [],
            "embedding": [],  # Future support
        }

        if not base_path.exists():
            logger.warning(f"Model base path does not exist: {base_path}")
            return models

        # Check LLM models
        llm_path = base_path / "llm"
        if llm_path.exists() and llm_path.is_dir():
            for model_dir in llm_path.iterdir():
                if model_dir.is_dir() and ModelDiscovery._validate_llm_model(model_dir):
                    models["llm"].append(model_dir.name)
                    logger.info(f"Found LLM model: {model_dir.name}")
        else:
            logger.debug(f"No LLM directory found at {llm_path}")

        # Check Whisper models
        whisper_path = base_path / "whisper"
        if whisper_path.exists() and whisper_path.is_dir():
            for model_dir in whisper_path.iterdir():
                if model_dir.is_dir() and ModelDiscovery._validate_whisper_model(model_dir):
                    models["whisper"].append(model_dir.name)
                    logger.info(f"Found Whisper model: {model_dir.name}")
        else:
            logger.debug(f"No Whisper directory found at {whisper_path}")

        # Check embedding models (future)
        embedding_path = base_path / "embedding"
        if embedding_path.exists() and embedding_path.is_dir():
            for model_dir in embedding_path.iterdir():
                if model_dir.is_dir() and ModelDiscovery._validate_embedding_model(model_dir):
                    models["embedding"].append(model_dir.name)
                    logger.info(f"Found embedding model: {model_dir.name}")

        return models

    @staticmethod
    def _validate_llm_model(path: Path) -> bool:
        """
        Check if directory contains valid LLM model (OpenVINO or GGUF)

        Args:
            path: Path to model directory

        Returns:
            True if valid LLM model found
        """
        # Check for GGUF model files
        gguf_files = list(path.glob("*.gguf"))
        if gguf_files:
            logger.debug(f"Found GGUF model: {gguf_files[0].name}")
            return True

        # Check for OpenVINO model files
        has_ov = (path / "openvino_model.xml").exists() and (path / "openvino_model.bin").exists()

        # Check for tokenizer files
        has_tokenizer = (path / "tokenizer.json").exists() or (
            path / "tokenizer_config.json"
        ).exists()

        if not has_ov:
            logger.debug(f"No OpenVINO model files found in {path}")
        if not has_tokenizer:
            logger.debug(f"No tokenizer files found in {path}")

        return has_ov and has_tokenizer

    @staticmethod
    def _validate_whisper_model(path: Path) -> bool:
        """
        Check if directory contains valid Whisper model (OpenVINO or QNN)

        Args:
            path: Path to model directory

        Returns:
            True if valid Whisper model found
        """
        # Check for OpenVINO encoder/decoder models
        has_ov_encoder = (path / "openvino_encoder_model.xml").exists() or (
            path / "encoder_model.xml"
        ).exists()
        has_ov_decoder = (path / "openvino_decoder_model.xml").exists() or (
            path / "decoder_model.xml"
        ).exists()

        # Also check for single OpenVINO model file (some Whisper variants)
        has_ov_single = (path / "openvino_model.xml").exists()

        # Check for QNN models (ONNX format with device-specific structure)
        has_qnn_model = ModelDiscovery._validate_qnn_whisper_model(path)

        return (has_ov_encoder and has_ov_decoder) or has_ov_single or has_qnn_model

    @staticmethod
    def _validate_qnn_whisper_model(path: Path) -> bool:
        """
        Check if directory contains valid QNN Whisper model

        Args:
            path: Path to model directory

        Returns:
            True if valid QNN Whisper model found
        """
        # Check for device-specific QNN models (e.g., snapdragon-x-elite)
        device_variants = ["snapdragon-x-elite"]  # Can be expanded in the future
        
        for device in device_variants:
            device_path = path / device
            if device_path.exists() and device_path.is_dir():
                # Check for encoder and decoder ONNX models
                encoder_path = device_path / "encoder" / "model.onnx"
                decoder_path = device_path / "decoder" / "model.onnx"
                
                if encoder_path.exists() and decoder_path.exists():
                    logger.debug(f"Found QNN model for device: {device}")
                    return True
        
        return False

    @staticmethod
    def _validate_embedding_model(path: Path) -> bool:
        """
        Check if directory contains valid embedding model

        Args:
            path: Path to model directory

        Returns:
            True if valid embedding model found
        """
        # Check for OpenVINO model files
        has_ov = (path / "openvino_model.xml").exists() and (path / "openvino_model.bin").exists()

        # Embeddings might not need tokenizer
        return has_ov

    @staticmethod
    def get_model_path(base_path: Path, model_type: str, model_name: str) -> Path | None:
        """
        Get full path to a specific model

        Args:
            base_path: Base directory for models
            model_type: Type of model (llm, whisper, embedding)
            model_name: Name of the model

        Returns:
            Path to model if exists, None otherwise
        """
        model_path = base_path / model_type / model_name

        if model_path.exists() and model_path.is_dir():
            # Validate based on type
            if model_type == "llm" and ModelDiscovery._validate_llm_model(model_path):
                return model_path
            elif model_type == "whisper" and ModelDiscovery._validate_whisper_model(model_path):
                return model_path
            elif model_type == "embedding" and ModelDiscovery._validate_embedding_model(model_path):
                return model_path
            else:
                logger.warning(f"Model validation failed for {model_path}")
        else:
            logger.debug(f"Model path does not exist: {model_path}")

        return None

    @staticmethod
    def get_llm_runtime_type(model_path: Path) -> str:
        """
        Determine the appropriate runtime type for an LLM model

        Args:
            model_path: Path to the LLM model directory

        Returns:
            Runtime type: "openvino" or "llamacpp"
        """
        # Check for GGUF files first
        gguf_files = list(model_path.glob("*.gguf"))
        if gguf_files:
            return "llamacpp"
        
        # Default to OpenVINO for other valid LLM models
        return "openvino"

    @staticmethod
    def get_whisper_runtime_type(model_path: Path) -> str:
        """
        Determine the appropriate runtime type for a Whisper model

        Args:
            model_path: Path to the Whisper model directory

        Returns:
            Runtime type: "openvino" or "qnn"
        """
        # Import platform utilities
        from .platform_utils import get_architecture, is_runtime_available
        
        # Check for QNN models first (more specific structure) and only on ARM64
        if (get_architecture() == "arm64" and 
            is_runtime_available("qnn") and 
            ModelDiscovery._validate_qnn_whisper_model(model_path)):
            return "qnn"
        
        # Default to OpenVINO for other valid Whisper models
        return "openvino"
