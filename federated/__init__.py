"""Federated Learning Module - Privacy-Preserving Distributed Training"""

# Import classes from each module
from .healthcare import HealthcareSilo, FederatedCoordinator, PatientDataReference
from .learning import SimpleModel, GradientAggregator, HospitalNode
from .mining import GradientBlock, ProofOfWork, GradientCompressor, GradientMiningPool, GradientChain, GradientMiner
from .advanced import TokenReward, IncentiveSystem, DifferentialPrivacy, SecureAggregation, ConvergenceTracker
from .data import LocalDataNode, DataHost, P2PDataNetwork, DataDistributor

__all__ = [
    # Healthcare
    "HealthcareSilo",
    "FederatedCoordinator",
    "PatientDataReference",
    # Learning
    "SimpleModel",
    "GradientAggregator",
    "HospitalNode",
    # Mining
    "GradientBlock",
    "ProofOfWork",
    "GradientCompressor",
    "GradientMiningPool",
    "GradientChain",
    "GradientMiner",
    # Advanced
    "TokenReward",
    "IncentiveSystem",
    "DifferentialPrivacy",
    "SecureAggregation",
    "ConvergenceTracker",
    # Data
    "LocalDataNode",
    "DataHost",
    "P2PDataNetwork",
    "DataDistributor",
]
