"""
LISA Package

Train large language models on limited hardware.
"""
__version__ = "1.0.0"

from .src.lisa_70b_v2 import LISATrainer, CONFIG
from .src.lisa_inference_prod import LISAInference

__all__ = ["LISATrainer", "LISAInference", "CONFIG"]
