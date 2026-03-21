"""
LISA-AutoResearch: Large Model Training on Consumer Hardware

Enables training 32B+ parameter models on 16GB Mac using LISA + disk offload.

Modules:
    lisa: Core LISA training (trainer, offload, hardware detection)
    distributed: P2P and host-based training (p2p, host, discovery, continuous)
    federated: Privacy-preserving training (healthcare, learning, mining, advanced)
    inference: Model inference (engine, parallel, quantize)
    api: Server and async I/O
    utils: Utilities (benchmark, mixed precision, production)
"""

__version__ = "1.0.0"
__author__ = "Ciphemon"

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "HAS_TORCH",
    "__version__",
]
