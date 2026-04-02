#!/usr/bin/env python3
"""
FederatedPool — Federated server coordination for LISA training

Coordinates multiple heterogeneous devices (Mac MLX + Jetson CUDA)
for LISA federated training:
- Discovers devices on the local network
- Assigns non-overlapping LISA layers per device
- Coordinates rounds of federated training

This runs ON THE SERVER (typically the Jetson or a dedicated server machine).
It does NOT run on the client devices — those run federated_lisa.py.

Usage:
    # As a library
    from federated_pool import FederatedPool
    pool = FederatedPool("http://SERVER_IP:8080", "MY_API_KEY")
    pool.discover_devices()
    pool.assign_layers()
    pool.run_round()
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import struct
import zlib
import pickle
import logging
import hashlib
import threading
import platform
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
except ImportError:
    psutil = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
if hasattr(_log_handler, "setEncoding"):
    _log_handler.setEncoding("utf-8")
_log_file = logging.FileHandler("/tmp/federated_pool.log", encoding="utf-8", mode="a")
_log_file.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[_log_handler, _log_file],
)
logger = logging.getLogger("federated-pool")


# ============================================================================
# Device Descriptors
# ============================================================================

@dataclass
class DeviceInfo:
    """Describes a federated learning participant device."""
    client_id: str
    platform: str          # "macos" or "linux"
    device_type: str       # "mlx", "cuda", "cpu"
    model: str             # HuggingFace model being used
    lisa_layers: List[int] # Layer indices this device trains
    hardware: Dict         # GPU, RAM, chip info
    last_seen: float = field(default_factory=time.time)
    is_connected: bool = True
    rounds_completed: int = 0
    avg_round_time: float = 0.0


# ============================================================================
# LISA Layer Assignment Logic
# ============================================================================

def split_lisa_layers(total_layers: int, devices: List[DeviceInfo],
                      strategy: str = "bottom_top") -> Dict[str, List[int]]:
    """
    Assign non-overlapping LISA layers to each device.

    Strategies:
    - "bottom_top": Mac gets bottom layers (0..N), Jetson gets top layers (N..end)
    - "interleaved": Distribute layers round-robin across devices
    - "balanced": Assign roughly equal chunks to each device
    """
    assignments = {}
    layer_pool = list(range(total_layers))

    if strategy == "bottom_top":
        # Sort devices: prefer cuda for top layers (harder compute)
        cuda_devices = [d for d in devices if d.device_type in ("cuda", "cpu")]
        mlx_devices  = [d for d in devices if d.device_type == "mlx"]
        sorted_devices = cuda_devices + mlx_devices

        chunk_size = (total_layers + len(devices) - 1) // len(devices)
        for i, dev in enumerate(sorted_devices):
            start = i * chunk_size
            end   = min(start + chunk_size, total_layers)
            assignments[dev.client_id] = layer_pool[start:end]

    elif strategy == "interleaved":
        # Round-robin distribution
        for i, dev in enumerate(devices):
            assignments[dev.client_id] = layer_pool[i::len(devices)]

    elif strategy == "balanced":
        chunk_size = (total_layers + len(devices) - 1) // len(devices)
        for i, dev in enumerate(devices):
            start = i * chunk_size
            end   = min(start + chunk_size, total_layers)
            assignments[dev.client_id] = layer_pool[start:end]

    return assignments


# ============================================================================
# Gradient Aggregation
# ============================================================================

def aggregate_gradients_fedavg(
    client_grads: Dict[str, Dict[str, np.ndarray]],
    client_weights: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Aggregate gradients using FedAvg (Federated Averaging).

    Args:
        client_grads: {client_id: {param_name: gradient_ndarray}}
        client_weights: {client_id: weight} (default: equal weight per client)

    Returns:
        Aggregated gradient dict {param_name: aggregated_ndarray}
    """
    if not client_grads:
        return {}

    if client_weights is None:
        client_weights = {cid: 1.0 for cid in client_grads}
    total_weight = sum(client_weights.values())
    if total_weight == 0:
        total_weight = 1.0

    # Get all param names across clients
    all_param_names = set()
    for grads in client_grads.values():
        all_param_names.update(grads.keys())

    aggregated = {}
    for param_name in all_param_names:
        weighted_sum = None
        for client_id, grads in client_grads.items():
            if param_name not in grads:
                continue
            weight = client_weights[client_id] / total_weight
            grad = grads[param_name].astype(np.float64)
            if weighted_sum is None:
                weighted_sum = weight * grad
            else:
                weighted_sum += weight * grad
        aggregated[param_name] = weighted_sum.astype(np.float32) if weighted_sum is not None else None

    return aggregated


# ============================================================================
# Model State Dict Utilities
# ============================================================================

def state_dict_to_numpy(state_dict: Dict) -> Dict[str, np.ndarray]:
    """Convert PyTorch state dict tensors to numpy arrays for transfer."""
    result = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            result[k] = v
    return result


def numpy_to_state_dict(np_dict: Dict, model: nn.Module) -> nn.Module:
    """Load numpy arrays back into a model's state dict."""
    state = model.state_dict()
    for k, v in np_dict.items():
        if k in state:
            state[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
    model.load_state_dict(state, strict=False)
    return model


# ============================================================================
# FederatedPool
# ============================================================================

class FederatedPool:
    """
    Coordinates multiple devices for LISA federated training.

    Runs on the server (typically Jetson or a dedicated machine).
    Clients (Mac MLX, Jetson CUDA) run federated_lisa.py and connect here.

    Responsibilities:
    - Track registered client devices
    - Assign non-overlapping LISA layers
    - Coordinate federated rounds (distribute model → collect gradients → aggregate → update)
    - Monitor device health
    """

    def __init__(self, server_url: str, pool_api_key: str,
                 model_name: str = "microsoft/phi-2",
                 total_layers: int = 0,
                 aggregation: str = "fedavg",
                 round_timeout: int = 300):
        """
        Args:
            server_url: Base URL of the federated HTTP server (e.g. "http://SERVER_IP:8080")
            pool_api_key: API key for authenticating with the server
            model_name: HuggingFace model name (for layer count)
            total_layers: Total number of transformer layers in the model
            aggregation: Aggregation method ("fedavg", "fedprox", "trimmed_mean")
            round_timeout: Seconds to wait for clients per round
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = pool_api_key
        self.model_name = model_name
        self.total_layers = total_layers
        self.aggregation_method = aggregation
        self.round_timeout = round_timeout

        # Client registry
        self.devices: Dict[str, DeviceInfo] = {}
        self.assignments: Dict[str, List[int]] = {}  # client_id → [layer indices]
        self.round_lock = threading.Lock()

        # Round state
        self.current_round: int = 0
        self.round_gradients: Dict[str, Dict[str, np.ndarray]] = {}  # round_id → client_id → grads
        self.round_clients_expected: List[str] = []
        self.global_model: Optional[nn.Module] = None

        # HTTP session
        self.session = requests.Session() if REQUESTS_AVAILABLE else None

        logger.info(f"FederatedPool initialized | server={server_url} | model={model_name}")

    # ── Device Discovery ────────────────────────────────────────────────────

    def discover_devices(self, timeout: float = 5.0,
                         lan_subnet: Optional[str] = None) -> List[DeviceInfo]:
        """
        Scan LAN for active LISA federated clients.

        This uses a simple approach: query the server's /clients endpoint
        which tracks all registered clients. For true LAN discovery,
        use mDNS/Bonjour or a broadcast UDP probe.

        Args:
            timeout: Seconds to wait for discovery probe
            lan_subnet: Subnet to scan (e.g. "192.168.1.") — if None, uses server's subnet

        Returns:
            List of discovered DeviceInfo objects
        """
        discovered = []

        # Method 1: Ask the federated server for registered clients
        try:
            resp = self._request("GET", "/clients")
            if resp.status_code == 200:
                clients = resp.json().get("clients", [])
                for c in clients:
                    info = DeviceInfo(
                        client_id=c["client_id"],
                        platform=c.get("platform", "unknown"),
                        device_type=c.get("device", "unknown"),
                        model=c.get("model", ""),
                        lisa_layers=c.get("lisa_layers", []),
                        hardware=c.get("hardware", {}),
                        last_seen=c.get("last_seen", time.time()),
                    )
                    self.devices[info.client_id] = info
                    discovered.append(info)
                logger.info(f"[Pool] Discovered {len(discovered)} registered clients via server API")
        except Exception as e:
            logger.warning(f"[Pool] Could not query server clients API: {e}")

        # Method 2: UDP broadcast discovery (mDNS-style probe)
        try:
            discovered_udp = self._udp_discovery(timeout=timeout)
            for d in discovered_udp:
                if d.client_id not in self.devices:
                    self.devices[d.client_id] = d
                    discovered.append(d)
            if discovered_udp:
                logger.info(f"[Pool] UDP discovery found {len(discovered_udp)} additional devices")
        except Exception as e:
            logger.warning(f"[Pool] UDP discovery failed: {e}")

        logger.info(f"[Pool] Total devices registered: {len(self.devices)}")
        for d in self.devices.values():
            logger.info(f"  - {d.client_id[:12]} | {d.platform}/{d.device_type} | "
                        f"model={d.model} | layers={d.lisa_layers}")
        return discovered

    def _udp_discovery(self, port: int = 9999, timeout: float = 5.0) -> List[DeviceInfo]:
        """
        Broadcast a LISA discovery probe on LAN and collect responses.
        Clients listening on the same port respond with their DeviceInfo.
        """
        discovered = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)

        probe = json.dumps({"type": "lisa_discovery_probe", "pool": self.server_url}).encode()
        broadcast_addrs = [
            ("<broadcast>", port),
            ("255.255.255.255", port),
        ]

        for addr in broadcast_addrs:
            try:
                sock.sendto(probe, addr)
            except OSError:
                pass

        # Collect responses
        end = time.time() + timeout
        while time.time() < end:
            try:
                sock.settimeout(max(0.1, end - time.time()))
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode())
                if msg.get("type") == "lisa_discovery_response":
                    info = DeviceInfo(
                        client_id=msg.get("client_id", addr[0]),
                        platform=msg.get("platform", "unknown"),
                        device_type=msg.get("device_type", "unknown"),
                        model=msg.get("model", ""),
                        lisa_layers=msg.get("lisa_layers", []),
                        hardware=msg.get("hardware", {}),
                    )
                    discovered.append(info)
                    logger.info(f"[Pool] UDP discovered {addr[0]}: {info.device_type}")
            except socket.timeout:
                break
            except Exception:
                pass

        sock.close()
        return discovered

    # ── Layer Assignment ─────────────────────────────────────────────────────

    def assign_layers(self, strategy: str = "bottom_top") -> Dict[str, List[int]]:
        """
        Assign non-overlapping LISA layers to each registered device.

        Default strategy: CUDA devices get top layers, MLX devices get bottom layers.
        This reflects typical compute distribution where the Jetson (CUDA) handles
        harder top-layer attention while Mac (MLX) handles embedding-adjacent layers.

        Args:
            strategy: "bottom_top" (CUDA=top, MLX=bottom), "interleaved", "balanced"

        Returns:
            Dict mapping client_id → list of assigned layer indices
        """
        if not self.devices:
            logger.warning("[Pool] No devices registered — cannot assign layers")
            return {}

        # Auto-detect total layers from model if not set
        if self.total_layers == 0:
            self.total_layers = self._detect_model_layers()
            logger.info(f"[Pool] Auto-detected {self.total_layers} model layers")

        # Sort devices by compute priority: cuda > cpu > mlx
        def priority(d: DeviceInfo) -> int:
            return {"cuda": 0, "cpu": 1, "mlx": 2}.get(d.device_type, 3)

        sorted_devices = sorted(self.devices.values(), key=priority)
        self.assignments = split_lisa_layers(
            total_layers=self.total_layers,
            devices=sorted_devices,
            strategy=strategy,
        )

        # Notify each device of its assignment
        for client_id, layers in self.assignments.items():
            self._notify_layer_assignment(client_id, layers)

        # Log assignments
        logger.info(f"[Pool] Layer assignments ({strategy}):")
        for cid, layers in self.assignments.items():
            dev = self.devices.get(cid)
            dev_name = f"{dev.platform}/{dev.device_type}" if dev else "?"
            logger.info(f"  {cid[:12]} ({dev_name}) → layers {layers}")

        return self.assignments

    def _detect_model_layers(self) -> int:
        """Detect number of transformer layers from model config."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name)
            # Qwen2: num_hidden_layers; GPT: n_layer; BERT: num_hidden_layers
            return (getattr(config, "num_hidden_layers", 0) or
                    getattr(config, "n_layer", 0) or
                    getattr(config, "num_layers", 0) or
                    28)  # sensible default
        except Exception:
            return 28

    def _notify_layer_assignment(self, client_id: str, layers: List[int]):
        """Send layer assignment notification to a client."""
        try:
            resp = self._request(
                "POST", f"/clients/{client_id}/assignment",
                json={"lisa_layers": layers}
            )
            if resp.status_code in (200, 201):
                logger.info(f"[Pool] Notified {client_id[:12]} of layer assignment: {layers}")
        except Exception as e:
            logger.warning(f"[Pool] Could not notify {client_id[:12]}: {e}")

    # ── Federated Round Coordination ────────────────────────────────────────

    def run_round(self, round_id: Optional[int] = None,
                  min_clients: int = 1) -> Dict[str, Any]:
        """
        Coordinate ONE round of federated training.

        Steps:
        1. Push current global model to participating clients
        2. Wait for gradient submissions from all clients
        3. Aggregate gradients (FedAvg)
        4. Apply aggregated gradients to global model
        5. Return round summary

        Args:
            round_id: Round number (auto-increments if None)
            min_clients: Minimum clients required to proceed

        Returns:
            Round summary dict with metrics
        """
        with self.round_lock:
            if round_id is None:
                round_id = self.current_round + 1
            self.current_round = round_id
            round_str = f"round_{round_id}"

        start_time = time.time()
        participating = [d for d in self.devices.values() if d.is_connected]

        if len(participating) < min_clients:
            logger.warning(f"[Pool] Round {round_id}: only {len(participating)} clients, need {min_clients} — skipping")
            return {"round": round_id, "status": "skipped", "reason": "insufficient_clients"}

        self.round_gradients[round_str] = {}
        self.round_clients_expected = [d.client_id for d in participating]

        logger.info(f"[Pool] === Starting Round {round_id} with {len(participating)} clients ===")

        # Step 1: Distribute global model to all clients
        global_state = self._get_global_model_state()
        for dev in participating:
            self._push_model_to_client(dev.client_id, round_str, global_state)

        # Step 2: Wait for gradient submissions (with timeout)
        received = self._wait_for_gradients(round_str, timeout=self.round_timeout)

        if len(received) < min_clients:
            logger.warning(f"[Pool] Round {round_id}: only {len(received)} clients responded — proceeding anyway")

        # Step 3: Aggregate
        aggregated_grads = aggregate_gradients_fedavg(received)

        # Step 4: Apply gradients to global model
        if aggregated_grads and self.global_model is not None:
            self._apply_gradients_to_global(aggregated_grads)

        elapsed = time.time() - start_time

        summary = {
            "round": round_id,
            "status": "completed",
            "clients_responded": list(received.keys()),
            "clients_expected": self.round_clients_expected,
            "num_clients": len(received),
            "total_layers_assigned": sum(len(v) for v in self.assignments.values()),
            "elapsed_secs": round(elapsed, 2),
            "aggregation": self.aggregation_method,
        }

        logger.info(
            f"[Pool] === Round {round_id} complete | "
            f"clients={len(received)}/{len(participating)} | "
            f"time={elapsed:.1f}s ==="
        )

        return summary

    def run(self, num_rounds: int, min_clients_per_round: int = 1,
            delay_between_rounds: float = 5.0):
        """
        Run multiple federated rounds sequentially.

        Args:
            num_rounds: Total number of rounds to run
            min_clients_per_round: Minimum clients required per round
            delay_between_rounds: Seconds to wait between rounds
        """
        logger.info(f"[Pool] Starting federated training: {num_rounds} rounds")
        for r in range(1, num_rounds + 1):
            logger.info(f"[Pool] [Round {r}/{num_rounds}]")
            summary = self.run_round(round_id=r, min_clients=min_clients_per_round)
            if summary.get("status") == "completed":
                logger.info(
                    f"[Pool] Round {r} summary: "
                    f"clients={summary['num_clients']}, "
                    f"time={summary['elapsed_secs']}s"
                )
            time.sleep(delay_between_rounds)
        logger.info("[Pool] All federated rounds complete")

    # ── Gradient Collection ──────────────────────────────────────────────────

    def _wait_for_gradients(self, round_str: str, timeout: float) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Wait for gradient submissions from clients.
        Returns {client_id: gradient_dict}.
        """
        received = {}
        end_time = time.time() + timeout

        # Poll the server's gradient queue
        while time.time() < end_time:
            remaining = max(0.1, end_time - time.time())
            try:
                resp = self._request(
                    "GET",
                    f"/gradients/{round_str}",
                    timeout=min(5.0, remaining)
                )
                if resp.status_code == 200:
                    data = resp.json()
                    grads_data = data.get("gradients", [])
                    for g in grads_data:
                        cid = g.get("client_id")
                        if cid and cid not in received:
                            # Decompress
                            compressed = g.get("compressed", b"")
                            metadata   = g.get("metadata", {})
                            if compressed:
                                grads = self._decompress_gradients(compressed, metadata)
                            else:
                                grads = g.get("gradients", {})
                            received[cid] = grads
                            logger.info(f"[Pool] Received gradients from {cid[:12]} "
                                        f"({len(grads)} tensors)")

                # Check if all expected clients have responded
                if all(cid in received for cid in self.round_clients_expected):
                    break

            except requests.Timeout:
                continue
            except Exception as e:
                logger.warning(f"[Pool] Gradient poll error: {e}")

            time.sleep(1.0)

        return received

    def _decompress_gradients(self, compressed_bytes: bytes,
                              metadata: Dict) -> Dict[str, np.ndarray]:
        """Decompress client gradient data."""
        import zlib, pickle
        decompressed = pickle.loads(zlib.decompress(compressed_bytes))
        result = {}
        for name, data in decompressed.items():
            if metadata.get("method") == "topk":
                shape = metadata.get("tensors", {}).get(name, {}).get("shape", None)
                arr = np.zeros(shape, dtype=np.float32) if shape else np.array([], dtype=np.float32)
                arr[data["indices"]] = data["values"]
                result[name] = arr
            elif metadata.get("method") == "quantize":
                result[name] = data.astype(np.float32)
            else:
                result[name] = data
        return result

    # ── Model Management ────────────────────────────────────────────────────

    def _get_global_model_state(self) -> Dict:
        """Get current global model state as numpy dict."""
        if self.global_model is None:
            return {}
        return state_dict_to_numpy(self.global_model.state_dict())

    def _push_model_to_client(self, client_id: str, round_str: str, state: Dict):
        """Push global model state to a specific client."""
        try:
            resp = self._request(
                "POST",
                f"/model/{round_str}",
                json={"state_dict": state, "recipient": client_id}
            )
            if resp.status_code in (200, 201):
                logger.info(f"[Pool] Pushed global model to {client_id[:12]} for {round_str}")
            else:
                logger.warning(f"[Pool] Failed to push model to {client_id[:12]}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"[Pool] Could not push model to {client_id[:12]}: {e}")

    def _apply_gradients_to_global(self, gradients: Dict[str, np.ndarray]):
        """Apply aggregated gradients to the global model."""
        if self.global_model is None:
            logger.warning("[Pool] No global model to apply gradients to")
            return

        state = self.global_model.state_dict()
        for name, grad in gradients.items():
            if name in state:
                param = state[name]
                if isinstance(param, torch.Tensor):
                    state[name] = param - 0.01 * torch.from_numpy(grad).to(param.device)
        self.global_model.load_state_dict(state, strict=False)
        logger.info(f"[Pool] Applied {len(gradients)} aggregated gradients to global model")

    def set_global_model(self, model: nn.Module):
        """Set the server's global model (must be called before run_round)."""
        self.global_model = model
        logger.info("[Pool] Global model set")

    # ── Networking Helpers ───────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make authenticated HTTP request to the federated server."""
        url = f"{self.server_url}{path}"
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.api_key
        headers.setdefault("Content-Type", "application/json")
        return self.session.request(method, url, headers=headers, timeout=kwargs.pop("timeout", 60), **kwargs)

    # ── Health & Status ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return current pool status."""
        return {
            "server_url": self.server_url,
            "total_devices": len(self.devices),
            "devices": [
                {
                    "client_id": d.client_id[:12],
                    "platform": d.platform,
                    "device_type": d.device_type,
                    "model": d.model,
                    "assigned_layers": self.assignments.get(d.client_id, []),
                    "is_connected": d.is_connected,
                    "rounds_completed": d.rounds_completed,
                }
                for d in self.devices.values()
            ],
            "current_round": self.current_round,
            "assignments": {cid[:12]: layers for cid, layers in self.assignments.items()},
            "total_model_layers": self.total_layers,
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FederatedPool — LISA Server Coordinator")
    parser.add_argument("--server", required=True, help="Federated server URL (e.g. http://SERVER_IP:8080)")
    parser.add_argument("--api-key", required=True, help="Server API key")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model name for layer detection")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--assign-strategy", default="bottom_top",
                        choices=["bottom_top", "interleaved", "balanced"],
                        help="Layer assignment strategy")
    parser.add_argument("--timeout", type=int, default=300, help="Round timeout in seconds")
    parser.add_argument("--delay-between-rounds", type=float, default=5.0)
    args = parser.parse_args()

    pool = FederatedPool(
        server_url=args.server,
        pool_api_key=args.api_key,
        model_name=args.model,
        round_timeout=args.timeout,
    )

    print("[Pool] Discovering devices...")
    pool.discover_devices()

    if not pool.devices:
        print("[Pool] WARNING: No devices discovered. Starting round anyway (server may have clients).")

    print("[Pool] Assigning layers...")
    pool.assign_layers(strategy=args.assign_strategy)

    print(json.dumps(pool.status(), indent=2))

    print(f"[Pool] Running {args.rounds} rounds...")
    pool.run(
        num_rounds=args.rounds,
        min_clients_per_round=args.min_clients,
        delay_between_rounds=args.delay_between_rounds,
    )

    print("[Pool] Final status:")
    print(json.dumps(pool.status(), indent=2))


if __name__ == "__main__":
    main()
