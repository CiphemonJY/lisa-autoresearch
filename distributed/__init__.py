"""Distributed Training Module - P2P and Host-based Training"""

from .host import ModelHost, TrainingNode
from .continuous import ContinuousMiner

# Discovery classes
from .discovery import PeerInfo, DiscoveryResult, BaseDiscovery, DHTDiscovery, BootstrapDiscovery

__all__ = [
    "ModelHost",
    "TrainingNode",
    "ContinuousMiner",
    "PeerInfo",
    "DiscoveryResult",
    "BaseDiscovery",
    "DHTDiscovery",
    "BootstrapDiscovery",
]
