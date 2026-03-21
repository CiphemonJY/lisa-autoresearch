"""
LISA Core Module - Large Model Training on Any Hardware

Cross-platform LISA implementation:
- Windows/Linux: PyTorch-based training (federated learning ready)
- macOS Apple Silicon: MLX-based training (original, fast)

The goal: train large models (7B+) on ANY hardware via federated learning.
Even a PC with 16GB RAM can participate as a client or lightweight coordinator.
"""

import platform

# Detect platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Always-available hardware detection (no MLX/torch required)
from .hardware import HardwareInfo, detect_hardware

# Platform-aware trainer selection
def get_trainer():
    """
    Get the best available LISA trainer for this platform.

    macOS Apple Silicon -> MLX trainer (fast, native)
    Windows/Linux      -> PyTorch trainer (CPU compatible, federated-ready)
    """
    if IS_MAC and HAS_MLX and not IS_WINDOWS:
        # Prefer MLX on Apple Silicon
        from .lisa_mlx import LISALayerTrainer as _Trainer
        return _Trainer
    elif HAS_TORCH:
        # PyTorch works everywhere
        from .train_torch import LISALayerTrainer as _Trainer
        return _Trainer
    else:
        raise ImportError(
            "No training framework available. "
            "Install PyTorch: pip install torch "
            "or MLX (macOS): pip install mlx"
        )

def get_offloader():
    """
    Get disk-offloaded trainer for large models.

    PyTorch version works on Windows/Linux.
    Uses layer-group processing to train 7B+ models on 16GB RAM.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for disk offloading. Install: pip install torch")
    from .offload_torch import DiskOffloadedTrainer as _Offloader
    return _Offloader

def get_config():
    """Get LISA configuration object."""
    if HAS_TORCH:
        from .train_torch import LISAConfig as _Config
        return _Config
    else:
        from .trainer import LISAConfig as _Config
        return _Config

def get_hardware_report():
    """Get a quick hardware report as a dict."""
    hw = detect_hardware()
    return {
        "platform": platform.system(),
        "cpu": hw.cpu_brand,
        "cpu_cores": hw.cpu_cores,
        "ram_gb": hw.available_ram_gb,
        "gpu": hw.gpu_name or "None",
        "gpu_type": hw.gpu_type,
        "framework": hw.recommended_framework,
        "max_model": hw.max_model_size,
        "can_federate": True,  # Any hardware can federate
    }

# Public API
__all__ = [
    # Hardware
    "HardwareInfo",
    "detect_hardware",
    "get_hardware_report",
    # Training
    "get_trainer",
    "get_offloader",
    "get_config",
    # Flags
    "HAS_TORCH",
    "HAS_MLX",
    "IS_MAC",
    "IS_WINDOWS",
    "IS_LINUX",
    # Backwards compatibility aliases
    "HardwareDetector",
]

# Aliases
HardwareDetector = HardwareInfo
