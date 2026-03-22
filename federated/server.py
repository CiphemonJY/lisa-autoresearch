#!/usr/bin/env python3
"""
Federated Learning Server - Real PyTorch Implementation

A federated learning server that:
- Receives gradient updates from multiple clients
- Validates and aggregates them using FedAvg
- Tracks client reputation and participation
- Monitors convergence
- Pushes aggregated model updates back to clients

Works on CPU Windows/Linux. No MLX required.
"""

import os
import sys
import json
import time
import threading
import hashlib
import pickle
import logging
import struct
import socket
import socketserver
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import copy

import numpy as np
import torch
import psutil

from federated.privacy import GradientPrivacy, DPConfig
from federated.byzantine import ByzantineResilientAggregator
from utils.checkpoint_manager import CheckpointManager

# Optional FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: fastapi not installed. HTTP API disabled. Use --demo mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fed-server")


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_name": "distilbert/distilgpt2",
    "num_rounds": 10,
    "min_clients_per_round": 2,
    "max_clients_per_round": 10,
    "round_timeout_secs": 120,
    "round_wait_secs": 60,  # how long to wait for all clients per round
    "aggregation_method": "fedavg",  # fedavg, fedprox, trimmed_mean
    "gradient_noise_tolerance": 1e6,
    "save_model_every": 5,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    # Byzantine resilience
    "byzantine_method": "none",  # none, krum, trimmed_mean, norm
    "byzantine_f": 1,           # max expected malicious clients (krum)
    "byzantine_alpha": 0.1,     # trim fraction for trimmed_mean
    "use_streaming": True,      # Enable chunked transfer for large tensors (>10MB)
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClientInfo:
    """Information about a registered client."""
    client_id: str
    registered_at: float
    last_seen: float
    rounds_participated: int = 0
    total_samples_contributed: int = 0
    reputation: float = 50.0
    is_active: bool = True
    pending_gradient: Optional[Dict] = None


@dataclass
class RoundState:
    """State for a federated learning round."""
    round_number: int
    status: str  # waiting, collecting, aggregating, done, failed
    clients_joined: List[str] = field(default_factory=list)
    clients_completed: List[str] = field(default_factory=list)  # clients that received their update
    clients_disconnected: List[str] = field(default_factory=list)  # clients that dropped mid-round
    gradients_received: int = 0
    gradients_accepted: int = 0
    gradients_rejected: int = 0
    started_at: float = 0
    completed_at: Optional[float] = None
    aggregated_gradient: Optional[bytes] = None


@dataclass
class ServerMetrics:
    """Server-side metrics."""
    total_rounds: int = 0
    total_gradients_received: int = 0
    total_gradients_rejected: int = 0
    avg_round_time: float = 0
    clients_registered: int = 0
    active_clients: int = 0
    convergence_history: List[float] = field(default_factory=list)


# ============================================================================
# Gradient Validation
# ============================================================================

class GradientValidator:
    """Validate incoming gradient updates."""

    def __init__(self, config: Dict):
        self.config = config
        self.noise_threshold = config.get("gradient_noise_tolerance", 1e6)
        self.history: Dict[str, List[float]] = defaultdict(list)

    def validate(self, client_id: str, update: Dict) -> Tuple[bool, str]:
        """
        Validate a gradient update.

        Returns: (is_valid, reason)
        """
        # Check required fields
        required = ["client_id", "round_number", "num_samples", "gradient_norm"]
        for field in required:
            if field not in update:
                return False, f"Missing required field: {field}"

        # Check gradient norm is reasonable
        norm = update.get("gradient_norm", 0)
        if norm > self.noise_threshold:
            return False, f"Gradient norm {norm:.2e} exceeds threshold"

        if norm < 1e-10:
            return False, "Gradient norm too small (likely no training)"

        # Check num_samples
        num_samples = update.get("num_samples", 0)
        if num_samples <= 0:
            return False, "num_samples must be positive"

        # Check loss is improving (loss_after < loss_before)
        if "loss_before" in update and "loss_after" in update:
            if update["loss_after"] > update["loss_before"] + 1.0:
                return False, "Loss increased (possible gradient issue)"

        return True, "Valid"

    def record_norm(self, client_id: str, norm: float):
        """Record gradient norm for history."""
        self.history[client_id].append(norm)
        if len(self.history[client_id]) > 100:
            self.history[client_id] = self.history[client_id][-100:]


# ============================================================================
# Gradient Aggregation
# ============================================================================

class GradientAggregator:
    """Aggregate gradient updates from multiple clients."""

    def __init__(self, method: str = "fedavg", gradient_privacy: Optional[GradientPrivacy] = None,
                 dp_enabled: bool = False, dp_config: Optional[DPConfig] = None):
        self.method = method
        self.compressor = None  # Server-side compression if needed
        self.gradient_privacy = gradient_privacy
        self.dp_enabled = dp_enabled
        self.dp_config = dp_config or DPConfig(enabled=False)

    def _decompress_updates(self, updates: List[Dict]) -> List[Tuple[str, int, Dict]]:
        """Decompress raw update dicts into (client_id, num_samples, state_dict) tuples."""
        decompressed = []
        for u in updates:
            try:
                data = u.get("gradient_data", b"")

                if isinstance(data, dict):
                    # Socket path: gradient_data is already a dict of tensors
                    state_dict = {}
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            state_dict[k] = v.cpu().float()
                        elif isinstance(v, np.ndarray):
                            state_dict[k] = torch.from_numpy(v).float()
                    if state_dict:
                        decompressed.append((u["client_id"], u.get("num_samples", 100), state_dict))
                elif isinstance(data, (bytes, bytearray)):
                    # HTTP path: compressed bytes
                    from federated.client import GradientCompressor
                    default_config = {"compression": {"enabled": True, "sparsification_ratio": 0.05, "quantization_bits": 8, "compression_level": 6}}
                    decompressor = GradientCompressor(default_config)
                    if isinstance(data, str):
                        import base64 as _base64
                        data = _base64.b64decode(data)
                    if data:
                        comp_info = u.get("compression_info", {"method": "sparse-8bit", "sparsification_ratio": 0.05})
                        state_dict = decompressor.decompress(data, comp_info)
                        decompressed.append((u["client_id"], u.get("num_samples", 100), state_dict))
            except Exception as e:
                logger.warning(f"Failed to decompress gradient from {u.get('client_id')}: {e}")
                continue
        return decompressed

    def dp_aggregate(
        self,
        updates: List[Dict],
        reputations: Dict[str, float],
    ) -> Tuple[Optional[bytes], Dict]:
        """
        Aggregate gradients with differential privacy (Gaussian mechanism).

        Clip each client's gradient → weighted sum → add Gaussian noise.
        Returns private aggregated gradient bytes.
        """
        decompressed = self._decompress_updates(updates)
        if not decompressed:
            return None, {"status": "no_valid_gradients"}

        # Extract gradient dicts and weights
        grad_dicts = [state_dict for _, _, state_dict in decompressed]
        client_weights = []
        for client_id, num_samples, _ in decompressed:
            rep = reputations.get(client_id, 50.0) / 50.0
            client_weights.append(num_samples * rep)

        # DP aggregation: clip → sum → add noise
        private_grad = self.gradient_privacy.dp_aggregate(
            grad_dicts,
            noise_multiplier=self.dp_config.noise_multiplier,
            max_grad_norm=self.dp_config.max_grad_norm,
            client_weights=client_weights,
        )

        serialized = pickle.dumps(private_grad)
        stats = {
            "status": "success",
            "method": "dp_fedavg",
            "num_updates": len(decompressed),
            "total_samples": sum(w for w in client_weights),
            "aggregated_size": len(serialized),
            "dp_enabled": True,
        }
        return serialized, stats

    def aggregate(
        self,
        updates: List[Dict],
        reputations: Dict[str, float],
    ) -> Tuple[Optional[bytes], Dict]:
        """
        Aggregate gradients using FedAvg (or DP-FedAvg if enabled).

        Updates may have:
          - gradient_data as bytes (HTTP path, compressed)
          - gradient_data as dict (socket path, raw torch tensors)

        Returns: (aggregated_state_dict_bytes, stats)
        """
        if not updates:
            return None, {"status": "no_updates"}

        # DP path: use DP aggregation instead of plain FedAvg
        if self.dp_enabled and self.gradient_privacy is not None:
            return self.dp_aggregate(updates, reputations)

        decompressed = self._decompress_updates(updates)

        if not decompressed:
            return None, {"status": "no_valid_gradients"}

        # FedAvg: weighted average by sample count, reputation-normalized
        total_samples = sum(s for _, s, _ in decompressed)

        # Initialize aggregated with zero tensors
        first_state = decompressed[0][2]
        aggregated = {}
        for key, val in first_state.items():
            if isinstance(val, np.ndarray):
                aggregated[key] = np.zeros_like(val, dtype=np.float32)
            elif isinstance(val, torch.Tensor):
                aggregated[key] = torch.zeros_like(val, dtype=torch.float32)

        # Compute total weight (sample count * reputation factor)
        total_weight = 0.0
        for client_id, num_samples, _ in decompressed:
            rep = reputations.get(client_id, 50.0) / 50.0  # Normalize to 0.5-1.5
            total_weight += num_samples * rep

        # Weighted sum
        for client_id, num_samples, state_dict in decompressed:
            rep = reputations.get(client_id, 50.0) / 50.0  # Normalize to 0.5-1.5
            weight = (num_samples * rep) / total_weight  # Single normalized weight

            for key in aggregated:
                if key in state_dict:
                    val = state_dict[key]
                    if isinstance(val, torch.Tensor):
                        aggregated[key] = aggregated[key] + val.float() * weight
                    elif isinstance(val, np.ndarray):
                        aggregated[key] = aggregated[key] + val * weight

        # Serialize
        serialized = pickle.dumps(aggregated)

        stats = {
            "status": "success",
            "method": self.method,
            "num_updates": len(decompressed),
            "total_samples": total_samples,
            "aggregated_size": len(serialized),
        }

        return serialized, stats


# ============================================================================
# Federated Server
# ============================================================================

class FederatedServer:
    """
    Main federated learning server.

    Coordinates federated rounds: receives gradients from clients,
    aggregates them, updates the global model, and distributes updates.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()

        self.clients: Dict[str, ClientInfo] = {}
        self.round_state: Dict[int, RoundState] = {}
        self.metrics = ServerMetrics()

        # Differential privacy (must init before aggregator)
        self.dp_config = self.config.get("dp_config") or DPConfig(enabled=False)
        self.gradient_privacy = GradientPrivacy(self.dp_config)
        self._dp_rounds: int = 0  # tracks rounds for epsilon computation

        self.validator = GradientValidator(self.config)
        self.aggregator = GradientAggregator(
            method=self.config.get("aggregation_method", "fedavg"),
            gradient_privacy=self.gradient_privacy,
            dp_enabled=self.dp_config.enabled,
            dp_config=self.dp_config,
        )

        # Byzantine resilience
        byz_method = self.config.get("byzantine_method", "none")
        if byz_method not in ("none", "krum", "trimmed_mean", "norm"):
            byz_method = "none"
        self.byzantine_method = byz_method
        if byz_method != "none":
            self.byzantine_aggregator = ByzantineResilientAggregator(
                method=byz_method,
                f=self.config.get("byzantine_f", 1),
                alpha=self.config.get("byzantine_alpha", 0.1),
            )
            logger.info(
                f"Byzantine resilience enabled: method={byz_method}, "
                f"f={self.config.get('byzantine_f', 1)}, "
                f"alpha={self.config.get('byzantine_alpha', 0.1)}"
            )
        else:
            self.byzantine_aggregator = None

        self.current_model: Optional[bytes] = None
        self.global_round: int = 0
        self.compression_method = self.config.get("compression_method", "none")
        self.compression_k = self.config.get("compression_k", 0.1)
        self.compression_bits = self.config.get("compression_bits", 8)

        self._lock = threading.RLock()

        # Directories
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager with versioning & rollback
        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_dir))

        # Load model
        self._init_model()

        logger.info(f"FederatedServer initialized (model={self.config['model_name']})")
        logger.info(f"  Rounds: {self.config['num_rounds']}")
        logger.info(f"  Min clients/round: {self.config['min_clients_per_round']}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")

    def _init_model(self):
        """Initialize or load the global model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        model_name = self.config["model_name"]
        logger.info(f"Loading global model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            config = AutoConfig.from_pretrained(model_name)
            config.hidden_size = min(config.hidden_size, 512)
            config.num_attention_heads = min(config.num_attention_heads, 8)
            config.num_hidden_layers = min(config.num_hidden_layers, 6)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True,
            )

            # Try to resume from the latest checkpoint
            latest_id = self.checkpoint_manager.get_latest()
            if latest_id is not None:
                logger.info(f"Resuming from checkpoint: {latest_id}")
                try:
                    ckpt_data = self.checkpoint_manager.load(latest_id)
                    self.model.load_state_dict(ckpt_data["model"])
                    self.global_round = ckpt_data["metadata"].get("round", 0)
                    logger.info(
                        f"  Restored round {self.global_round} "
                        f"({latest_id}), perplexity={ckpt_data['metadata'].get('perplexity')}"
                    )
                except Exception as e:
                    logger.warning(f"  Could not restore checkpoint {latest_id}: {e}")

            # Save initial model if no checkpoint to resume from
            if latest_id is None:
                self._save_model(0)

            logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _save_model(self, round_num: int):
        """Save model checkpoint."""
        import torch
        path = self.checkpoint_dir / f"model_round_{round_num}.pt"
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, path)
        self.current_model = pickle.dumps(state)
        logger.info(f"Model saved: {path}")

    def _get_lock(self):
        """Get or create the server lock (lazy initialization for tests)."""
        if not hasattr(self, '_lock'):
            self._lock = threading.RLock()
        return self._lock

    def register_client(self, client_id: str) -> Dict:
        """Register a new client."""
        with self._get_lock():
            if client_id in self.clients:
                self.clients[client_id].last_seen = time.time()
            else:
                self.clients[client_id] = ClientInfo(
                    client_id=client_id,
                    registered_at=time.time(),
                    last_seen=time.time(),
                )
                self.metrics.clients_registered += 1
                self.metrics.active_clients += 1
                logger.info(f"Registered client: {client_id}")

        return {"status": "registered", "client_id": client_id}

    def receive_gradient(self, update: Dict) -> Dict:
        """
        Receive a gradient update from a client.

        This is called when a client submits their gradient.
        """
        client_id = update.get("client_id", "unknown")
        round_num = update.get("round_number", 0)

        # Decode base64 gradient data if provided (convert to bytes for storage)
        import base64 as _base64
        if "gradient_data" in update and update["gradient_data"]:
            try:
                update["gradient_data"] = _base64.b64decode(update["gradient_data"])
            except Exception as e:
                logger.warning(f"Failed to decode base64 gradient data from {client_id}: {e}")

        with self._get_lock():
            # Update client last-seen
            if client_id in self.clients:
                self.clients[client_id].last_seen = time.time()

            # Validate update
            is_valid, reason = self.validator.validate(client_id, update)

            # Record gradient norm
            self.validator.record_norm(client_id, update.get("gradient_norm", 0))

            # Store pending gradient
            if client_id in self.clients:
                self.clients[client_id].pending_gradient = update

            # Update round state
            if round_num not in self.round_state:
                self.round_state[round_num] = RoundState(
                    round_number=round_num,
                    status="collecting",
                    started_at=time.time(),
                )

            rs = self.round_state[round_num]
            rs.gradients_received += 1
            rs.gradients_accepted += 1
            rs.clients_joined.append(client_id)

            self.metrics.total_gradients_received += 1

            # Update client stats
            if client_id in self.clients:
                self.clients[client_id].rounds_participated += 1
                self.clients[client_id].total_samples_contributed += update.get("num_samples", 0)

            logger.info(
                f"Received gradient from {client_id} (round {round_num}): "
                f"norm={update.get('gradient_norm', 0):.4f}, "
                f"samples={update.get('num_samples', 0)}"
            )

            # Check if we have enough gradients to aggregate
            min_clients = self.config.get("min_clients_per_round", 2)
            if len(set(rs.clients_joined)) >= min_clients and rs.status == "collecting":
                self._start_aggregation(round_num)

            return {"status": "accepted", "round": round_num}

    def _start_aggregation(self, round_num: int):
        """Start aggregating gradients for a round."""
        rs = self.round_state[round_num]
        rs.status = "aggregating"

        logger.info(f"Starting aggregation for round {round_num}")

        # Collect all pending gradients
        updates = []
        for cid in set(rs.clients_joined):
            if cid in self.clients and self.clients[cid].pending_gradient:
                updates.append(self.clients[cid].pending_gradient)

        # Get reputations
        reputations = {c.client_id: c.reputation for c in self.clients.values()}

        # Aggregate
        if self.byzantine_aggregator is not None:
            aggregated, byz_stats = self._aggregate_byzantine(updates, reputations)
            byz_method = self.byzantine_method
            excluded = byz_stats.get("num_excluded", 0)
            total = len(updates)
            logger.info(
                f"Round {round_num} Byzantine aggregation ({byz_method}): "
                f"{total - excluded}/{total} clients used — {byz_stats}"
            )
        else:
            aggregated, stats = self.aggregator.aggregate(updates, reputations)
            byz_stats = None

        if aggregated is None:
            rs.status = "failed"
            logger.error(f"Aggregation failed for round {round_num}")
            return

        # Privacy budget accounting
        if self.dp_config.enabled:
            self._dp_rounds += 1
            eps = GradientPrivacy.compute_epsilon(
                self.dp_config.noise_multiplier, self._dp_rounds
            )
            status = self.gradient_privacy.privacy_status(self._dp_rounds)
            logger.info(
                f"Round {round_num}: ε={eps:.2f} (δ=1e-5) [{status['strength']} privacy]"
            )
            for warning in status.get("warnings", []):
                logger.warning(warning)

        # Apply to global model
        self._apply_gradient(aggregated)

        # Save checkpoint
        self._save_model(round_num)

        # Save versioned checkpoint with metadata
        try:
            metrics = {
                "perplexity": None,  # filled by caller if available
                "client_count": stats.get("num_updates", 0),
                "compression": self.compression_method,
                "dp_enabled": self.dp_config.enabled,
                "round_time": round_time,
            }
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            ckpt_id = self.checkpoint_manager.save(model_state, round_num, metrics)
            logger.info(f"  Checkpoint saved: {ckpt_id}")
        except Exception as e:
            logger.warning(f"  Checkpoint save failed: {e}")

        # Complete round
        rs.status = "done"
        rs.completed_at = time.time()
        rs.aggregated_gradient = aggregated

        # Clear pending gradients
        for cid in self.clients:
            self.clients[cid].pending_gradient = None

        # Update metrics
        self.global_round = round_num
        self.metrics.total_rounds += 1

        round_time = rs.completed_at - rs.started_at
        self.metrics.avg_round_time = (
            (self.metrics.avg_round_time * (self.metrics.total_rounds - 1) + round_time)
            / self.metrics.total_rounds
        )

        # Memory profiling
        try:
            _proc = psutil.Process()
            mem_mb = _proc.memory_info().rss / 1e6
            logger.info(f"  Peak memory: {mem_mb:.1f} MB")
        except Exception:
            pass

        logger.info(
            f"Round {round_num} complete: "
            f"{stats['num_updates']} updates, "
            f"{round_time:.1f}s, "
            f"model updated"
        )

    def _apply_gradient(self, gradient_bytes: bytes):
        """Apply aggregated gradient to global model."""
        import torch

        state_dict = pickle.loads(gradient_bytes)

        # Simple SGD update
        current_state = self.model.state_dict()
        lr = 0.01

        for key in current_state:
            if key in state_dict:
                grad = state_dict[key]
                if isinstance(grad, np.ndarray):
                    grad = torch.from_numpy(grad)
                current_state[key] = current_state[key].float() + lr * grad.float()

        self.model.load_state_dict(current_state)

    def _aggregate_byzantine(
        self,
        updates: List[Dict],
        reputations: Dict[str, float],
    ) -> Tuple[Optional[bytes], Dict]:
        """
        Aggregate gradients using Byzantine-resilient method.

        Decompresses raw updates, extracts grad dicts and weights,
        delegates to ByzantineResilientAggregator, serializes result.
        """
        decompressed = self.aggregator._decompress_updates(updates)
        if not decompressed:
            return None, {"status": "no_valid_gradients"}

        grad_dicts = [state_dict for _, _, state_dict in decompressed]
        client_weights = []
        for client_id, num_samples, _ in decompressed:
            rep = reputations.get(client_id, 50.0) / 50.0
            client_weights.append(num_samples * rep)

        result, stats = self.byzantine_aggregator.aggregate(grad_dicts, client_weights)

        serialized = pickle.dumps(result)
        stats["status"] = "success"
        stats["aggregated_size"] = len(serialized)
        stats["num_updates"] = len(decompressed)
        return serialized, stats

    def get_model_update(self, client_id: str, since_round: int = 0) -> Optional[Dict]:
        """
        Get the latest model update for a client.

        Returns serialized model state dict or None.
        """
        with self._get_lock():
            # Return current model
            if self.current_model:
                return {
                    "round": self.global_round,
                    "model_data": self.current_model,
                    "model_size": len(self.current_model),
                }
            return None

    def get_status(self) -> Dict:
        """Get server status."""
        with self._get_lock():
            active = sum(1 for c in self.clients.values() if c.is_active)

            status = {
                "global_round": self.global_round,
                "num_rounds": self.config["num_rounds"],
                "clients_registered": self.metrics.clients_registered,
                "active_clients": active,
                "total_gradients_received": self.metrics.total_gradients_received,
                "total_gradients_rejected": self.metrics.total_gradients_rejected,
                "avg_round_time": self.metrics.avg_round_time,
                "current_model_size_mb": len(self.current_model) / 1e6 if self.current_model else 0,
                "config": {
                    "model_name": self.config["model_name"],
                    "min_clients_per_round": self.config["min_clients_per_round"],
                    "aggregation_method": self.config["aggregation_method"],
                },
            }

            if self.dp_config.enabled:
                eps = GradientPrivacy.compute_epsilon(
                    self.dp_config.noise_multiplier, self._dp_rounds
                )
                status["differential_privacy"] = {
                    "enabled": True,
                    "noise_multiplier": self.dp_config.noise_multiplier,
                    "max_grad_norm": self.dp_config.max_grad_norm,
                    "rounds_completed": self._dp_rounds,
                    "epsilon_estimate": round(eps, 4),
                }
            else:
                status["differential_privacy"] = {"enabled": False}

            return status

    def get_round_status(self, round_num: int) -> Optional[Dict]:
        """Get status of a specific round."""
        with self._get_lock():
            if round_num not in self.round_state:
                return None

            rs = self.round_state[round_num]
            return {
                "round_number": rs.round_number,
                "status": rs.status,
                "clients_joined": len(set(rs.clients_joined)),
                "gradients_received": rs.gradients_received,
                "gradients_accepted": rs.gradients_accepted,
                "gradients_rejected": rs.gradients_rejected,
                "started_at": rs.started_at,
                "completed_at": rs.completed_at,
            }


# ============================================================================
# Socket-Based Federated Server
# ============================================================================

class FederatedSocketHandler(socketserver.BaseRequestHandler):
    """
    Handle one client connection via raw sockets.

    Protocol (matching fed_client.py):
      1. Client sends JSON metadata: {"type": "gradients", "client_id": "...", "round": N}
      2. Server reads N tensor frames: [name_len(4)][name_bytes][size(4)][data]
      3. Server aggregates gradients (FedAvg)
      4. Server sends JSON: {"type": "update", "n_tensors": K, "round": R}
      5. Server sends K tensor frames back

    Disconnect handling:
      - Socket timeout (30s) prevents hanging on slow clients
      - ConnectionResetError/BrokenPipeError/OSError caught at top level
      - Disconnected clients are marked so their gradients are excluded
      - Round continues with remaining clients
    """

    # Socket timeout for recv operations (seconds) - prevents hanging forever
    SOCKET_TIMEOUT = 30

    def handle(self):
        server = self.server.server_instance
        lock = server._get_lock()
        client_ip = self.client_address[0] if self.client_address else "unknown"
        client_id = "unknown"
        round_num = 0

        # Set socket timeout so recv calls don't block forever
        self.request.settimeout(self.SOCKET_TIMEOUT)

        try:
            # --- Auth token exchange (if server has auth_token configured) ---
            if self.server.auth_token is not None:
                token_header = self._recv_exact(4)
                if not token_header or len(token_header) < 4:
                    logger.warning(f"[{client_ip}] Auth failed: connection closed during token read")
                    return
                token_len = struct.unpack("!I", token_header)[0]
                if token_len > 1024 or token_len == 0:
                    logger.warning(f"[{client_ip}] Auth failed: invalid token length {token_len}")
                    return
                token_bytes = self._recv_exact(token_len)
                if not token_bytes or len(token_bytes) < token_len:
                    logger.warning(f"[{client_ip}] Auth failed: incomplete token received")
                    return
                received_token = token_bytes.decode("utf-8", errors="replace")
                if not secrets.compare_digest(received_token, self.server.auth_token):
                    logger.warning(f"[{client_ip}] Auth failed: invalid auth token")
                    return
                logger.info(f"[{client_ip}] Client authenticated successfully")

            # --- Step 1: receive JSON metadata ---
            header = self._recv_exact(4)
            if not header or len(header) < 4:
                return
            meta_len = struct.unpack("!I", header)[0]
            meta_bytes = self._recv_exact(meta_len)
            if not meta_bytes or len(meta_bytes) < meta_len:
                return
            meta = json.loads(meta_bytes.decode("utf-8"))
            msg_type = meta.get("type", "")
            client_id = meta.get("client_id", "unknown")
            round_num = meta.get("round", 0)

            logger.info(f"[{client_id}] Socket connected (round {round_num}, type={msg_type})")

            # Log if DP is enabled on the server
            if server.dp_config.enabled:
                logger.info(
                    f"[{client_id}] DP enabled on server: "
                    f"noise_mult={server.dp_config.noise_multiplier}, "
                    f"clip_norm={server.dp_config.max_grad_norm}"
                )

            if msg_type == "gradients":
                self._handle_gradients(server, lock, client_id, round_num)
            elif msg_type == "disconnect":
                logger.info(f"[{client_id}] Client disconnected gracefully")
                with lock:
                    if round_num in server.round_state:
                        rs = server.round_state[round_num]
                        rs.clients_completed.append(client_id)
            else:
                logger.warning(f"[{client_id}] Unknown message type: {msg_type}")

        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # Client disconnected mid-round - mark as disconnected, don't crash
            logger.warning(f"[{client_id}] Client disconnected (round {round_num}): {e}")
            self._mark_client_disconnected(server, lock, client_id, round_num)
        except Exception as e:
            logger.error(f"[{client_id}] Socket handler error: {e}")
            self._mark_client_disconnected(server, lock, client_id, round_num)

    def _mark_client_disconnected(self, server, lock, client_id: str, round_num: int):
        """Mark a client as disconnected mid-round so their gradient is excluded from aggregation."""
        if round_num <= 0 or client_id == "unknown":
            return
        with lock:
            if round_num in server.round_state:
                rs = server.round_state[round_num]
                if client_id not in rs.clients_disconnected:
                    rs.clients_disconnected.append(client_id)
                    logger.info(f"[{client_id}] disconnected mid-round, skipping")

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes. Returns partial data on timeout/disconnect."""
        data = b""
        while len(data) < n:
            try:
                chunk = self.request.recv(n - len(data))
                if not chunk:
                    return data
                data += chunk
            except socket.timeout:
                # Timed out waiting for data - return what we have
                return data
            except (ConnectionResetError, BrokenPipeError, OSError):
                # Client disconnected - return what we have
                return data
        return data

    def _recv_tensor(self) -> Tuple[str, torch.Tensor]:
        """Receive one tensor: [name_len][name_bytes][size][data]."""
        name_len_data = self._recv_exact(4)
        if len(name_len_data) < 4:
            return "", torch.zeros(0)
        name_len = struct.unpack("!I", name_len_data)[0]
        name_bytes = self._recv_exact(name_len)
        name = name_bytes.decode("utf-8")

        size_data = self._recv_exact(4)
        if len(size_data) < 4:
            return name, torch.zeros(0)
        size = struct.unpack("!I", size_data)[0]
        raw = self._recv_exact(size)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return name, torch.from_numpy(arr)

    # --- Streaming gradient transfer (for large tensors > 10MB) ---
    STREAMING_THRESHOLD = 10 * 1024 * 1024  # 10 MB

    def send_tensor_streaming(self, name: str, tensor: torch.Tensor, chunk_size: int = 65536):
        """
        Send a tensor using chunked streaming to avoid memory spikes.
        Used for tensors larger than STREAMING_THRESHOLD.
        Protocol: [name_len][name_bytes][num_bytes(8)][dtype_str_len][dtype_str][chunk1][chunk2]...
        Each chunk is sent as: [chunk_size bytes]
        """
        name_bytes = name.encode("utf-8")
        self.request.sendall(struct.pack("!I", len(name_bytes)) + name_bytes)

        total_bytes = tensor.numel() * tensor.element_size()
        self.request.sendall(struct.pack("!Q", total_bytes))  # 8-byte unsigned

        dtype_str = str(tensor.dtype).replace("torch.", "")
        dtype_bytes = dtype_str.encode("utf-8")
        self.request.sendall(struct.pack("!I", len(dtype_bytes)) + dtype_bytes)

        shape_bytes = json.dumps(list(tensor.shape)).encode("utf-8")
        self.request.sendall(struct.pack("!I", len(shape_bytes)) + shape_bytes)

        np_bytes = tensor.cpu().numpy().tobytes()
        for offset in range(0, len(np_bytes), chunk_size):
            chunk = np_bytes[offset : offset + chunk_size]
            self.request.sendall(chunk)

    def _recv_tensor_streaming(self, chunk_size: int = 65536) -> Tuple[str, torch.Tensor]:
        """
        Receive a streamed tensor from socket.
        Protocol: [name_len(4)][name][dtype_len(4)][dtype][shape_len(4)][shape][num_bytes(8)][chunks...]
        Returns: (name, tensor)
        """
        name_len_data = self._recv_exact(4)
        if len(name_len_data) < 4:
            return "", torch.zeros(0)
        name_len = struct.unpack("!I", name_len_data)[0]
        name_bytes = self._recv_exact(name_len)
        name = name_bytes.decode("utf-8")

        dtype_len_data = self._recv_exact(4)
        if len(dtype_len_data) < 4:
            return name, torch.zeros(0)
        dtype_len = struct.unpack("!I", dtype_len_data)[0]
        dtype_bytes = self._recv_exact(dtype_len)
        dtype_str = dtype_bytes.decode("utf-8")

        shape_len_data = self._recv_exact(4)
        if len(shape_len_data) < 4:
            return name, torch.zeros(0)
        shape_len = struct.unpack("!I", shape_len_data)[0]
        shape_bytes = self._recv_exact(shape_len)
        shape = json.loads(shape_bytes.decode("utf-8"))

        num_bytes_data = self._recv_exact(8)
        if len(num_bytes_data) < 8:
            return name, torch.zeros(0)
        total_bytes = struct.unpack("!Q", num_bytes_data)[0]

        buffer = b""
        while len(buffer) < total_bytes:
            chunk = self.request.recv(min(chunk_size, total_bytes - len(buffer)))
            if not chunk:
                break
            buffer += chunk

        arr = np.frombuffer(buffer, dtype=dtype_str).copy()
        result = torch.from_numpy(arr).view(shape)
        return name, result

    def _send_tensor(self, name: str, tensor: torch.Tensor):
        """Send one tensor: [name_len][name_bytes][size][data]."""
        name_bytes = name.encode("utf-8")
        self.request.sendall(struct.pack("!I", len(name_bytes)) + name_bytes)

        data = tensor.cpu().numpy().tobytes()
        self.request.sendall(struct.pack("!I", len(data)) + data)

    def _handle_gradients(self, server, lock, client_id: str, round_num: int):
        """Receive gradients, aggregate, send update back.

        Protocol:
          1. handle() already received JSON meta: {"type": "gradients", ...}
          2. Client sends grad_header JSON with small/large tensor split info
          3. Client sends small tensors as pickle blob (if any)
          4. Client sends large tensors as streaming frames
        """
        # --- Receive grad_header ---
        try:
            gh_len_data = self._recv_exact(4)
            if len(gh_len_data) < 4:
                self._mark_client_disconnected(server, lock, client_id, round_num)
                return
            gh_len = struct.unpack("!I", gh_len_data)[0]
            gh_bytes = self._recv_exact(gh_len)
            if len(gh_bytes) < gh_len:
                self._mark_client_disconnected(server, lock, client_id, round_num)
                return
            grad_header = json.loads(gh_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"[{client_id}] Failed to receive grad_header: {e}")
            self._mark_client_disconnected(server, lock, client_id, round_num)
            return

        n_large = grad_header.get("n_large", 0)
        n_small = grad_header.get("n_small", 0)
        large_names = set(grad_header.get("large_names", []))
        logger.info(f"[{client_id}] Receiving gradients: {n_small} small, {n_large} large tensors")

        grad_state = {}

        # --- Receive small tensors as pickle blob ---
        try:
            small_len_data = self._recv_exact(4)
            if len(small_len_data) < 4:
                self._mark_client_disconnected(server, lock, client_id, round_num)
                return
            small_len = struct.unpack("!I", small_len_data)[0]
            if small_len > 0:
                small_raw = b""
                while len(small_raw) < small_len:
                    chunk = self.request.recv(min(65536, small_len - len(small_raw)))
                    if not chunk:
                        break
                    small_raw += chunk
                small_tensors = pickle.loads(small_raw)
                grad_state.update(small_tensors)
        except Exception as e:
            logger.error(f"[{client_id}] Failed to receive small tensors: {e}")
            self._mark_client_disconnected(server, lock, client_id, round_num)
            return

        # --- Receive large tensors via streaming ---
        for i in range(n_large):
            try:
                name, tensor = self._recv_tensor_streaming()
                grad_state[name] = tensor
            except Exception as e:
                logger.warning(f"[{client_id}] Failed to receive large tensor {i}: {e}")
                break

        logger.info(f"[{client_id}] Received {len(grad_state)} gradient tensors (pickle)")

        # Register client
        server.register_client(client_id)

        # Build gradient update dict for FederatedServer.receive_gradient
        grad_norm = float(torch.norm(torch.cat([t.flatten() for t in grad_state.values()])).item())
        update = {
            "client_id": client_id,
            "round_number": round_num,
            "num_samples": 100,  # default; real clients may send more
            "gradient_norm": grad_norm,
            "gradient_data": grad_state,  # dict: will be handled specially by aggregator
        }

        with lock:
            # Validate & record
            is_valid, reason = server.validator.validate(client_id, update)
            server.validator.record_norm(client_id, grad_norm)

            if client_id in server.clients:
                server.clients[client_id].pending_gradient = update

            if round_num not in server.round_state:
                server.round_state[round_num] = RoundState(
                    round_number=round_num,
                    status="collecting",
                    started_at=time.time(),
                )

            rs = server.round_state[round_num]
            rs.gradients_received += 1
            rs.gradients_accepted += 1
            if client_id not in rs.clients_joined:
                rs.clients_joined.append(client_id)
            server.metrics.total_gradients_received += 1

            if client_id in server.clients:
                server.clients[client_id].rounds_participated += 1

            logger.info(
                f"[{client_id}] Gradient recorded (round {round_num}): "
                f"norm={grad_norm:.4f}, tensors={len(grad_state)}"
            )

            # Check if we should aggregate
            min_clients = server.config.get("min_clients_per_round", 2)
            if len(set(rs.clients_joined)) >= min_clients and rs.status == "collecting":
                self._aggregate_and_respond(server, lock, round_num, client_id)
            else:
                # Not enough clients yet - wait briefly then respond
                time.sleep(0.5)
                self._aggregate_and_respond(server, lock, round_num, client_id)

    def _aggregate_and_respond(self, server, lock, round_num: int, completed_client_id: str = None):
        """Aggregate gradients and send model update back to this client."""
        rs = server.round_state.get(round_num)
        if rs is None:
            return

        # Wait for status to settle using configurable round timeout
        round_wait = server.config.get("round_wait_secs", 60)
        waited = 0
        while rs.status == "collecting" and waited < round_wait:
            time.sleep(0.5)
            waited += 0.5

        with lock:
            # Exclude disconnected clients from aggregation
            active_joined = [
                cid for cid in rs.clients_joined
                if cid not in rs.clients_disconnected
            ]
            total_joined = len(set(rs.clients_joined))
            disconnected = list(set(rs.clients_joined) - set(active_joined))

            if rs.status == "collecting":
                # Timeout reached or enough clients - aggregate with whoever we have
                disconnected_names = ", ".join(disconnected) if disconnected else "none"
                if len(active_joined) < total_joined:
                    logger.warning(
                        f"Round {round_num}: timeout with {len(active_joined)}/{total_joined} "
                        f"active clients ({disconnected_names} disconnected)"
                    )
                elif len(active_joined) < server.config.get("min_clients_per_round", 2):
                    min_req = server.config.get("min_clients_per_round", 2)
                    logger.warning(
                        f"Round {round_num}: fewer clients ({len(active_joined)}) than "
                        f"min_clients ({min_req}) - proceeding anyway"
                    )

                updates = []
                for cid in active_joined:
                    if cid in server.clients and server.clients[cid].pending_gradient:
                        updates.append(server.clients[cid].pending_gradient)
                if updates:
                    reputations = {c.client_id: c.reputation for c in server.clients.values()}
                    aggregated, stats = server.aggregator.aggregate(updates, reputations)
                    if aggregated:
                        server._apply_gradient(aggregated)
                        server._save_model(round_num)
                    # Log round summary
                    contrib_names = ", ".join(updates[u].get("client_id", "?") for u in range(len(updates)))
                    disc_names = ", ".join(disconnected) if disconnected else ""
                    logger.info(
                        f"Round {round_num}: {len(updates)}/{total_joined} clients contributed "
                        f"({disc_names} disconnected) - aggregated"
                    )
                else:
                    aggregated = None
                    logger.warning(f"Round {round_num}: no valid gradients to aggregate")
                rs.status = "done"
                rs.completed_at = time.time()
                rs.aggregated_gradient = aggregated
                for cid in server.clients:
                    server.clients[cid].pending_gradient = None
                server.global_round = round_num
                server.metrics.total_rounds += 1

                # Memory profiling
                try:
                    _proc = psutil.Process()
                    mem_mb = _proc.memory_info().rss / 1e6
                    logger.info(f"  Peak memory: {mem_mb:.1f} MB")
                except Exception:
                    pass

            # Mark this client as completed
            if completed_client_id and completed_client_id not in rs.clients_completed:
                rs.clients_completed.append(completed_client_id)

        # Build model update response
        # Use server config; compression sub-agent may override these on the handler
        compression_method = getattr(self, "compression_method", None) or server.config.get("compression_method", "none")
        compression_k = getattr(self, "compression_k", 0.1)
        compression_bits = getattr(self, "compression_bits", 8)
        if server.current_model:
            state = pickle.loads(server.current_model)
            update_tensors = state
        elif rs.aggregated_gradient:
            update_tensors = pickle.loads(rs.aggregated_gradient)
        else:
            update_tensors = {}

        logger.info(f"[{self.client_address}] Sending {len(update_tensors)} tensors as model update")

        # Apply compression if configured
        compression_metadata = {"method": "none"}
        if compression_method != "none":
            from federated.compression import compress_gradients
            try:
                compressed_update, compression_metadata = compress_gradients(
                    update_tensors, method=compression_method, k=compression_k, bits=compression_bits
                )
                response_data = pickle.dumps(compressed_update)
                orig_size = sum(t.numel() * 4 for t in update_tensors.values() if isinstance(t, torch.Tensor))
                comp_size = len(response_data)
                ratio = orig_size / max(comp_size, 1)
                logger.info(
                    f"[{self.client_address}] Compression applied: "
                    f"method={compression_method}, "
                    f"original={orig_size/1024:.1f}KB, "
                    f"compressed={comp_size/1024:.1f}KB, "
                    f"ratio={ratio:.1f}x"
                )
                update_tensors = None  # Signal that we're sending compressed data
            except Exception as e:
                logger.warning(f"Compression failed ({e}), sending uncompressed")
                compression_metadata = {"method": "none"}
                response_data = pickle.dumps(update_tensors)
        else:
            response_data = pickle.dumps(update_tensors)

        # --- Step 4: send JSON metadata ---
        # Detect large tensors for streaming
        large_names = []
        if server.config.get("use_streaming", True) and update_tensors:
            STREAMING_THRESHOLD = 10 * 1024 * 1024
            for name, tensor in update_tensors.items():
                if isinstance(tensor, torch.Tensor):
                    sz = tensor.numel() * tensor.element_size()
                    if sz > STREAMING_THRESHOLD:
                        large_names.append(name)

        response_meta = {
            "type": "update",
            "n_tensors": len(update_tensors) if update_tensors is not None else -1,
            "round": round_num,
            "compression": compression_metadata,
            "use_streaming": bool(large_names),
            "large_tensors": large_names,
        }
        response_bytes = json.dumps(response_meta).encode("utf-8")
        try:
            self.request.sendall(struct.pack("!I", len(response_bytes)) + response_bytes)
            # --- Step 5: send payload ---
            if compression_method != "none":
                self.request.sendall(struct.pack("!I", len(response_data)) + response_data)
            elif large_names and server.config.get("use_streaming", True):
                # Mixed: send small tensors as pickle, large tensors via streaming
                small = {k: v for k, v in update_tensors.items() if k not in large_names}
                small_data = pickle.dumps(small) if small else b""
                self.request.sendall(struct.pack("!I", len(small_data)) + small_data)
                for name in large_names:
                    self.send_tensor_streaming(name, update_tensors[name])
            else:
                self.request.sendall(struct.pack("!I", len(response_data)) + response_data)
            logger.info(f"[{self.client_address}] Update sent for round {round_num}")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.warning(f"[{self.client_address}] Client disconnected before receiving update: {e}")


class FederatedSocketServer(socketserver.ThreadingTCPServer):
    """TCP server for handling federated learning client connections."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_instance, port: int = 8080, auth_token: Optional[str] = None):
        self.server_instance = server_instance  # FederatedServer instance
        self.auth_token = auth_token
        super().__init__(("0.0.0.0", port), FederatedSocketHandler)
        if self.auth_token:
            logger.info(f"Socket server auth token is set - client authentication enabled")
        else:
            logger.info(f"Socket server auth token is NOT set - clients will connect without authentication")
        logger.info(f"Socket server listening on port {port}")

    def server_close(self):
        logger.info("Socket server shutting down")
        super().server_close()


# ============================================================================
# FastAPI App
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="LISA Federated Learning Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _server: Optional[FederatedServer] = None

    @app.on_event("startup")
    async def startup():
        global _server
        _server = FederatedServer()

    @app.get("/")
    async def root():
        return {"message": "LISA Federated Learning Server", "status": "running"}

    @app.get("/status")
    async def status():
        return _server.get_status()

    @app.post("/register")
    async def register(request: Request):
        body = await request.json()
        client_id = body.get("client_id")
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id required")
        return _server.register_client(client_id)

    @app.post("/submit")
    async def submit(request: Request):
        body = await request.json()
        return _server.receive_gradient(body)

    @app.get("/model/{client_id}")
    async def get_model(client_id: str, since_round: int = 0):
        update = _server.get_model_update(client_id, since_round)
        if not update:
            raise HTTPException(status_code=404, detail="No model update available")
        return JSONResponse({
            "round": update["round"],
            "model_size": update["model_size"],
        })

    @app.get("/round/{round_num}")
    async def get_round(round_num: int):
        status = _server.get_round_status(round_num)
        if not status:
            raise HTTPException(status_code=404, detail="Round not found")
        return status


# ============================================================================
# Demo Mode (no HTTP, runs in-process)
# ============================================================================

class DemoFederatedSimulator:
    """
    Simulate a full federated learning run without HTTP.

    Useful for testing on a single machine.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.server = FederatedServer(config)
        self.clients: Dict[str, Any] = {}

    def add_client(self, client_id: str):
        """Add a simulated client."""
        # Import client class
        from federated.client import FederatedClient
        client = FederatedClient(
            client_id=client_id,
            server_url="http://localhost:8000",  # Won't be used in demo
            config=self.server.config,
        )
        self.clients[client_id] = client
        self.server.register_client(client_id)
        return client

    def run_round(self, round_num: int) -> Dict:
        """Run one federated round with all clients."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}")
        logger.info(f"{'='*60}")

        # Each client computes gradient
        for client_id, client in self.clients.items():
            logger.info(f"  {client_id}: computing gradient...")
            update = client.train_and_submit(round_num)

            # Server receives gradient
            self.server.receive_gradient(update)

        # Aggregation happens automatically when min clients submit
        rs = self.server.round_state.get(round_num)
        if rs:
            while rs.status == "collecting":
                time.sleep(0.1)

        # Get round result
        result = self.server.get_round_status(round_num)
        logger.info(f"  Round {round_num} result: {result}")

        return result or {}

    def run(self, num_rounds: int = 3) -> Dict:
        """Run full federated training simulation."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FEDERATED LEARNING SIMULATION")
        logger.info(f"{'='*60}")
        logger.info(f"Clients: {len(self.clients)}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"Model: {self.config['model_name']}")
        logger.info(f"{'='*60}\n")

        results = []
        for r in range(1, num_rounds + 1):
            result = self.run_round(r)
            results.append(result)

        # Final status
        status = self.server.get_status()

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Rounds: {status['global_round']}")
        logger.info(f"Gradients received: {status['total_gradients_received']}")
        logger.info(f"Avg round time: {status['avg_round_time']:.1f}s")
        logger.info(f"Model size: {status['current_model_size_mb']:.1f} MB")

        return {
            "status": "complete",
            "results": results,
            "final_status": status,
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--mode", choices=["server", "demo", "socket"], default="demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--auth-token", type=str, default=None,
                        help="Shared secret token for client authentication (optional)")
    parser.add_argument("--round-timeout", type=int, default=60,
                        help="Max seconds to wait for clients per round (default 60)")
    parser.add_argument("--compression", choices=["none", "sparsify", "quantize", "both"],
                        default="none",
                        help="Gradient compression method for sending updates to clients")
    parser.add_argument("--compression-k", type=float, default=0.1,
                        help="Sparsification ratio K (fraction to keep, e.g. 0.1 = keep top 10%%)")
    parser.add_argument("--compression-bits", type=int, default=8,
                        help="Quantization bits (8 or 16)")
    parser.add_argument("--dp", action="store_true",
                        help="Enable differential privacy (Gaussian mechanism)")
    parser.add_argument("--noise-multiplier", type=float, default=1.0,
                        help="DP noise multiplier σ (default 1.0, higher = more privacy, more noise)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="DP per-gradient clipping bound C (default 1.0)")
    parser.add_argument("--byzantine", choices=["none", "krum", "trimmed_mean", "norm"],
                        default="none",
                        help="Byzantine-resilient aggregation method")
    parser.add_argument("--byzantine-f", type=int, default=1,
                        help="Max expected malicious clients (for Krum, default 1)")
    parser.add_argument("--byzantine-alpha", type=float, default=0.1,
                        help="Trim fraction for Trimmed Mean (e.g. 0.1 = trim 10%% each tail)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=DEFAULT_CONFIG["checkpoint_dir"],
                        help="Directory for model checkpoints (default: checkpoints)")
    parser.add_argument("--gen-token", action="store_true",
                        help="Generate a random auth token and exit")

    args = parser.parse_args()

    if args.gen_token:
        token = secrets.token_urlsafe(32)
        print(token)
        sys.exit(0)

    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["num_rounds"] = args.rounds
    config["min_clients_per_round"] = args.min_clients
    config["round_wait_secs"] = args.round_timeout
    config["auth_token"] = args.auth_token
    config["compression_method"] = args.compression
    config["compression_k"] = args.compression_k
    config["compression_bits"] = args.compression_bits
    config["byzantine_method"] = args.byzantine
    config["byzantine_f"] = args.byzantine_f
    config["byzantine_alpha"] = args.byzantine_alpha
    config["checkpoint_dir"] = args.checkpoint_dir

    # Differential privacy
    if args.dp:
        dp_cfg = DPConfig(
            enabled=True,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
        config["dp_config"] = dp_cfg
        logger.info(
            f"DP enabled: noise_multiplier={args.noise_multiplier}, "
            f"max_grad_norm={args.max_grad_norm}"
        )

    if args.mode == "server":
        if not FASTAPI_AVAILABLE:
            print("Error: fastapi not installed. Install with: pip install fastapi uvicorn")
            sys.exit(1)

        server = FederatedServer(config)
        app._server = server  # type: ignore

        print(f"Starting HTTP server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.mode == "socket":
        # Socket-based server for fed_client.py
        print(f"Loading model: {args.model} ...")
        server = FederatedServer(config)

        print(f"Starting socket server on {args.host}:{args.port}")
        socket_server = FederatedSocketServer(server, port=args.port, auth_token=args.auth_token)

        print(f"Federated socket server running on port {args.port}")
        print(f"  Model: {args.model}")
        print(f"  Rounds: {args.rounds}")
        print(f"  Min clients/round: {args.min_clients}")
        print(f"  Waiting for clients...")
        print(f"Press Ctrl+C to stop.")

        try:
            socket_server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            socket_server.shutdown()

    else:
        # Demo mode (in-process, no network)
        sim = DemoFederatedSimulator(config)

        # Add simulated clients
        for i in range(1, args.clients + 1):
            client_id = f"client-{i}"
            sim.add_client(client_id)
            logger.info(f"Added client: {client_id}")

        # Run simulation
        results = sim.run(args.rounds)

        print("\n=== RESULTS ===")
        print(json.dumps(results["final_status"], indent=2))


if __name__ == "__main__":
    main()
