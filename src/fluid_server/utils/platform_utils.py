"""
Platform detection utilities for cross-architecture compatibility
"""

import logging
import os
import platform
from typing import Literal

logger = logging.getLogger(__name__)

ArchType = Literal["x64", "arm64", "unknown"]
RuntimeType = Literal["openvino", "qnn", "llamacpp"]


def get_architecture() -> ArchType:
    """
    Detect the current system architecture
    
    Returns:
        Architecture type: "x64", "arm64", or "unknown"
    """
    machine = platform.machine().lower()
    processor = platform.processor().lower()

    # Check environment variables that might indicate architecture
    processor_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()
    processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()

    # ARM64 detection
    if (
        machine in ["arm64", "aarch64"] or
        "arm64" in processor or
        "aarch64" in processor or
        "arm64" in processor_arch or
        ("arm" in processor_identifier and "64" in processor_identifier)
    ):
        return "arm64"

    # x64 detection
    if (
        machine in ["amd64", "x86_64", "x64"] or
        "amd64" in processor_arch or
        "intel" in processor.lower() or
        "x86" in machine
    ):
        return "x64"

    logger.warning(f"Unknown architecture detected: machine={machine}, processor={processor}")
    return "unknown"


def get_compatible_runtimes(arch: ArchType = None) -> list[RuntimeType]:
    """
    Get list of runtimes that should work on the given architecture
    
    Args:
        arch: Architecture to check, defaults to current architecture
        
    Returns:
        List of compatible runtime types
    """
    if arch is None:
        arch = get_architecture()

    compatible = []

    # OpenVINO should work on both architectures (different devices)
    compatible.append("openvino")

    # llama-cpp should work on both (different backends)
    compatible.append("llamacpp")

    # QNN only works on ARM64 Windows devices
    if arch == "arm64":
        compatible.append("qnn")

    return compatible


def get_preferred_device(arch: ArchType = None) -> dict:
    """
    Get preferred device assignments for the given architecture
    
    Args:
        arch: Architecture to check, defaults to current architecture
        
    Returns:
        Dictionary with preferred device assignments
    """
    if arch is None:
        arch = get_architecture()

    if arch == "arm64":
        return {
            "llm": "CPU",  # ARM64 may not have full GPU support yet
            "whisper": "NPU",  # ARM64 has dedicated NPU
            "fallback": "CPU"
        }
    else:  # x64 or unknown
        return {
            "llm": "GPU",  # x64 typically has good GPU support
            "whisper": "GPU",  # Use GPU as fallback since NPU may not be available
            "fallback": "CPU"
        }


def is_runtime_available(runtime_type: RuntimeType, arch: ArchType = None) -> bool:
    """
    Check if a specific runtime type should be available on the architecture
    
    Args:
        runtime_type: Runtime to check
        arch: Architecture to check, defaults to current architecture
        
    Returns:
        True if runtime should be available
    """
    if arch is None:
        arch = get_architecture()

    return runtime_type in get_compatible_runtimes(arch)


def log_platform_info():
    """Log detailed platform information for debugging"""
    arch = get_architecture()
    compatible = get_compatible_runtimes(arch)
    devices = get_preferred_device(arch)

    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Compatible runtimes: {compatible}")
    logger.info(f"Preferred devices: {devices}")

    # Log environment variables for debugging
    env_vars = ["PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITEW6432"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        logger.debug(f"{var}: {value}")
