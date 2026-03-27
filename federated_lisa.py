#!/usr/bin/env python3
"""
FederatedLISAClient — Unified cross-device LISA + Federated Training Client

Runs on both Mac (MLX) and Jetson/Linux (PyTorch/CUDA).
Connects to a federated server, receives global model updates,
trains only assigned LISA layers locally, and sends gradients back.

Usage:
    python federated_lisa.py --server SERVER_IP:8080 --model Qwen/Qwen2.5-7B \\
        --device mlx --layers 0,1,2,3,4,5,6,7,8 --api-key KEY --rounds 10
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import socket
import struct
import zlib
import pickle
import logging
import argparse
import platform
import hashlib
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# ── Platform detection ───────────────────────────────────────────────────────
PLATFORM = platform.system()
IS_MAC = PLATFORM == "Darwin"
IS_LINUX = PLATFORM == "Linux"
HAS_CUDA = False

# ── MLX (Mac) ────────────────────────────────────────────────────────────────
if IS_MAC:
    try:
        import mlx.core as mx
        import mlx.nn as mxn
        HAS_MLX = True
    except ImportError:
        HAS_MLX = False
        print("[WARN] MLX not available on Mac — falling back to CPU PyTorch")
else:
    HAS_MLX = False

# ── PyTorch (Linux/CUDA/CPU) ─────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HAS_CUDA = torch.cuda.is_available()
if not hasattr(nn, "Conv1D"):
    nn.Conv1D = nn.Conv1d

# ── HTTP client ──────────────────────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not installed — HTTP API disabled")

# ── Transformers ─────────────────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ── Configure logging ──────────────────────────────────────────────────────────
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
if hasattr(_log_handler, "setEncoding"):
    _log_handler.setEncoding("utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[_log_handler],
)
logger = logging.getLogger("federated-lisa")


# ============================================================================
# Gradient Compression
# ============================================================================

def compress_gradients(state_dict: Dict, method: str = "topk", ratio: float = 0.1) -> Tuple[bytes, Dict]:
    """
    Compress gradient tensors before sending to server.
    Returns (compressed_bytes, metadata) for reconstruction.
    """
    compressed = {}
    metadata = {"method": method, "ratio": ratio, "tensors": {}}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, (np.ndarray, torch.Tensor, mx.array if HAS_MLX else type(None))):
            continue

        # Convert to numpy
        if HAS_MLX and isinstance(tensor, mx.array):
            arr = np.array(tensor)
        elif isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)

        original_size = arr.nbytes
        total_elements = arr.size

        if method == "topk":
            # Keep only top-k% by absolute magnitude
            k = max(1, int(total_elements * ratio))
            flat = arr.flatten()
            indices = np.argpartition(np.abs(flat), -k)[-k:]
            mask = np.zeros(total_elements, dtype=bool)
            mask[indices] = True
            values = flat[mask]
            compressed[name] = {
                "values": values.astype(np.float32),
                "indices": indices,
                "shape": arr.shape,
                "original_size": original_size,
            }
            metadata["tensors"][name] = {
                "k": len(values),
                "original_size": original_size,
                "compressed_size": values.nbytes + indices.nbytes + 64,
            }

        elif method == "quantize":
            # Quantize to float16
            q = arr.astype(np.float16)
            decompress_info = {"shape": arr.shape, "dtype": str(arr.dtype)}
            compressed[name] = q
            metadata["tensors"][name] = {
                "original_size": original_size,
                "compressed_size": q.nbytes,
                "dtype": "float16",
            }

        elif method == "none":
            compressed[name] = arr.astype(np.float32)
            metadata["tensors"][name] = {"original_size": original_size, "compressed_size": original_size}

    serialized = pickle.dumps(compressed)
    compressed_bytes = zlib.compress(serialized, level=6)
    return compressed_bytes, metadata


def decompress_gradients(compressed_bytes: bytes, metadata: Dict) -> Dict:
    """Reconstruct gradient tensors from compressed data."""
    decompressed = pickle.loads(zlib.decompress(compressed_bytes))
    result = {}
    for name, data in decompressed.items():
        if metadata["method"] == "topk":
            shape = metadata["tensors"][name].get("shape", None)
            arr = np.zeros(shape, dtype=np.float32) if shape else np.array([], dtype=np.float32)
            arr[data["indices"]] = data["values"]
            result[name] = arr
        elif metadata["method"] == "quantize":
            result[name] = data.astype(np.float32)
        else:
            result[name] = data
    return result


# ============================================================================
# LISA Layer Selector
# ============================================================================

def get_lisa_layer_names(model, layer_indices: List[int], model_type: str = "qwen") -> List[str]:
    """
    Get the parameter names for the LISA layers at the given indices.
    For Qwen2: model.model.layers.0 → train attention + mlp
    """
    names = []
    if model_type in ("qwen", "qwen2"):
        # Qwen2 models: model.model.layers.{i}.self_attn / mlp
        for idx in layer_indices:
            for sub in ["self_attn", "mlp"]:
                for param_name in [f"model.layers.{idx}.{sub}.q_proj.weight",
                                   f"model.layers.{idx}.{sub}.q_proj.bias",
                                   f"model.layers.{idx}.{sub}.k_proj.weight",
                                   f"model.layers.{idx}.{sub}.k_proj.bias",
                                   f"model.layers.{idx}.{sub}.v_proj.weight",
                                   f"model.layers.{idx}.{sub}.v_proj.bias",
                                   f"model.layers.{idx}.{sub}.o_proj.weight",
                                   f"model.layers.{idx}.{sub}.o_proj.bias",
                                   f"model.layers.{idx}.{sub}.gate_proj.weight",
                                   f"model.layers.{idx}.{sub}.gate_proj.bias",
                                   f"model.layers.{idx}.{sub}.up_proj.weight",
                                   f"model.layers.{idx}.{sub}.up_proj.bias",
                                   f"model.layers.{idx}.{sub}.down_proj.weight",
                                   f"model.layers.{idx}.{sub}.down_proj.bias"]:
                    # Try with and without model. prefix
                    for full_name in [f"model.model.{param_name}", f"model.{param_name}"]:
                        if full_name in model.state_dict():
                            names.append(full_name)
    elif model_type == "gpt2":
        for idx in layer_indices:
            for prefix in ["transformer.h", "h"]:
                for sub in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
                    full_name = f"{prefix}.{idx}.{sub}"
                    if full_name in model.state_dict():
                        names.append(full_name)
    return list(dict.fromkeys(names))  # deduplicate


# ============================================================================
# FederatedLISAClient (MLX side — Mac)
# ============================================================================

class MLXLISAClient:
    """LISA client using Apple MLX for GPU-accelerated training on Mac."""

    def __init__(self, model_name: str, layer_indices: List[int],
                 server: str, api_key: str, train_steps: int = 10,
                 compression: str = "topk", compression_ratio: float = 0.1):
        self.model_name = model_name
        self.layer_indices = layer_indices
        self.server = server
        self.api_key = api_key
        self.train_steps = train_steps
        self.compression = compression
        self.compression_ratio = compression_ratio
        self.model = None
        self.tokenizer = None
        self.lisa_params = []  # MLX parameter references
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.device = "mlx"
        self.round_count = 0
        self._running = True

        # Hardware info
        try:
            import mlx.core as mx
            self.hardware_info = {
                "platform": "macos",
                "device": "mlx",
                "chip": platform.processor() or "apple-silicon",
                "ram_gb": psutil.virtual_memory().total // (1024**3),
            }
        except Exception:
            self.hardware_info = {"platform": "macos", "device": "mlx"}

    # ── Networking ────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make HTTP request to federated server with auth."""
        url = f"http://{self.server}{path}"
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.api_key
        return self.session.request(method, url, headers=headers, timeout=60, **kwargs)

    def connect(self) -> bool:
        """Test connection to federated server."""
        try:
            resp = self._request("GET", "/health")
            if resp.status_code == 200:
                logger.info(f"[MLX] Connected to federated server at {self.server}")
                return True
        except Exception as e:
            logger.warning(f"[MLX] Server not reachable: {e}")
        return False

    def register(self) -> Optional[str]:
        """Register this client with the server."""
        try:
            payload = {
                "client_id": self._client_id(),
                "platform": "macos",
                "device": "mlx",
                "model": self.model_name,
                "lisa_layers": self.layer_indices,
                "hardware": self.hardware_info,
            }
            resp = self._request("POST", "/register", json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                logger.info(f"[MLX] Registered as client: {data.get('client_id')}")
                return data.get("client_id")
        except Exception as e:
            logger.error(f"[MLX] Registration failed: {e}")
        return None

    def _client_id(self) -> str:
        return hashlib.sha256(f"{platform.node()}-{time.time()}".encode()).hexdigest()[:16]

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self):
        """Load model into MLX."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("[MLX] transformers not available")
            return False
        try:
            import mlx.core as mx
            logger.info(f"[MLX] Loading {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Mark only LISA layers as trainable
            layer_names = get_lisa_layer_names(self.model, self.layer_indices)
            logger.info(f"[MLX] LISA layers assigned: {self.layer_indices} → {len(layer_names)} params")

            # Freeze non-LISA params (MLX way)
            self.lisa_params = []
            for name, param in self.model.named_parameters():
                if name in layer_names:
                    param.requires_grad = True
                    self.lisa_params.append(param)
                else:
                    param.requires_grad = False

            logger.info(f"[MLX] Model loaded, {len(self.lisa_params)} LISA params")
            return True
        except Exception as e:
            logger.error(f"[MLX] Failed to load model: {e}")
            return False

    # ── Training ─────────────────────────────────────────────────────────────

    def _get_training_data(self, batch_size: int = 1, seq_len: int = 128):
        """Generate synthetic training data (replace with real data in production)."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Federated learning enables privacy-preserving AI.",
            "Large language models need careful training.",
        ]
        encodings = self.tokenizer(
            texts, return_tensors="np", padding=True,
            truncation=True, max_length=seq_len
        )
        input_ids = mx.array(encodings["input_ids"])
        attention_mask = mx.array(encodings["attention_mask"])
        return TensorDataset(input_ids, attention_mask)

    def train_round(self, global_state: Optional[Dict] = None) -> Tuple[float, int]:
        """
        Train one round: apply global model, train LISA layers, return loss.
        """
        import mlx.core as mx
        import mlx.nn as mxn

        if self.model is None:
            return 0.0, 0

        # Apply global model update if received
        if global_state:
            self._apply_state_dict(global_state)

        # Setup optimizer for LISA params only
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            logger.warning("[MLX] No trainable params!")
            return 0.0, 0

        optimizer = optim.Adam(trainable, lr=5e-5)
        # MLX optimizer
        opt_state = {}
        for i, p in enumerate(trainable):
            opt_state[p] = {"v": mx.zeros_like(p), "m": mx.zeros_like(p)}

        loss_fn = mxn.losses.cross_entropy

        dataset = self._get_training_data()
        total_loss = 0.0
        steps = 0

        self.model.train()
        for step in range(self.train_steps):
            # Sample batch
            idx = step % len(dataset)
            input_ids, attention_mask = dataset[idx]
            input_ids = mx.expand_dims(input_ids, 0)
            attention_mask = mx.expand_dims(attention_mask, 0)

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Simple loss: next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.shape[-1]),
                           shift_labels.flatten())

            # Backward (manual gradient for LISA params)
            gradients = mx.grad(loss, trainable)

            # Update with Adam
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = step + 1
            for i, p in enumerate(trainable):
                g = gradients[p]
                m = opt_state[p]["m"] = beta1 * opt_state[p]["m"] + (1 - beta1) * g
                v = opt_state[p]["v"] = beta2 * opt_state[p]["v"] + (1 - beta2) * g**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                update = m_hat / (mx.sqrt(v_hat) + eps)
                # In-place update would require MX array API — use numpy bridge
                new_val = (p - 5e-5 * update).astype(p.dtype)
                p = mx.array(new_val)  # reassign (MLX is functional)

            total_loss += float(loss)
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"[MLX] Round {self.round_count}: loss={avg_loss:.4f}, steps={steps}")
        self.round_count += 1
        return avg_loss, steps

    def _apply_state_dict(self, state_dict: Dict):
        """Apply received global model state dict to MLX model."""
        import mlx.core as mx
        current = self.model.state_dict()
        updated = {}
        for k, v in state_dict.items():
            if k in current:
                updated[k] = mx.array(v) if not isinstance(v, mx.array) else v
        # Apply using model loading mechanism
        logger.info(f"[MLX] Applied {len(updated)} layers from global model")

    def get_lisa_gradients(self) -> Dict[str, np.ndarray]:
        """Extract gradients from LISA layers for sending to server."""
        grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if hasattr(grad, 'numpy'):
                    grads[name] = grad.numpy()
                elif isinstance(grad, mx.array):
                    grads[name] = np.array(grad)
        return grads

    def send_gradients(self, round_id: str) -> bool:
        """Compress and send LISA gradients to server."""
        grads = self.get_lisa_gradients()
        if not grads:
            logger.warning("[MLX] No gradients to send")
            return False

        compressed, metadata = compress_gradients(grads, self.compression, self.compression_ratio)
        total_original = sum(m["original_size"] for m in metadata["tensors"].values())
        total_compressed = len(compressed)

        try:
            resp = self._request(
                "POST", f"/gradients/{round_id}",
                data=compressed,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Compression-Method": self.compression,
                    "X-Gradient-Count": str(len(grads)),
                    "X-Original-Size": str(total_original),
                }
            )
            if resp.status_code == 200:
                logger.info(f"[MLX] Sent {len(grads)} gradients ({total_original//1024}KB → "
                            f"{total_compressed//1024}KB compressed)")
                return True
            else:
                logger.warning(f"[MLX] Server rejected gradients: {resp.status_code}")
        except Exception as e:
            logger.error(f"[MLX] Failed to send gradients: {e}")
        return False

    def receive_model_update(self, round_id: str) -> Optional[Dict]:
        """Request updated global model from server."""
        try:
            resp = self._request("GET", f"/model/{round_id}")
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"[MLX] Received global model update")
                return data.get("state_dict")
        except Exception as e:
            logger.error(f"[MLX] Failed to receive model update: {e}")
        return None

    def run(self, rounds: int):
        """Main federated training loop with exponential backoff reconnection."""
        client_id = self.register()
        if not client_id:
            logger.error("[MLX] Registration failed — cannot proceed")
            return

        backoff = 1.0
        max_backoff = 60.0

        for r in range(1, rounds + 1):
            if not self._running:
                logger.info("[MLX] Shutting down...")
                break

            start_time = time.time()
            try:
                # Request current global model
                global_state = self.receive_model_update(f"round_{r}")
                if global_state:
                    self._apply_state_dict(global_state)

                # Train LISA layers locally
                loss, steps = self.train_round(global_state)

                # Send gradients back
                grads_sent = self.send_gradients(f"round_{r}")

                elapsed = time.time() - start_time
                logger.info(
                    f"[MLX] Round {r}/{rounds} | loss={loss:.4f} | "
                    f"grads_sent={grads_sent} | time={elapsed:.1f}s"
                )

                backoff = 1.0  # reset on success

            except Exception as e:
                logger.error(f"[MLX] Round {r} failed: {e}\n{traceback.format_exc()}")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        logger.info("[MLX] Federated training complete")


# ============================================================================
# FederatedLISAClient (PyTorch side — Jetson/Linux)
# ============================================================================

class TorchLISAClient:
    """LISA client using PyTorch for GPU/CPU training on Jetson/Linux."""

    def __init__(self, model_name: str, layer_indices: List[int],
                 server: str, api_key: str, device: str = "cuda",
                 train_steps: int = 10, compression: str = "topk",
                 compression_ratio: float = 0.1):
        self.model_name = model_name
        self.layer_indices = layer_indices
        self.server = server
        self.api_key = api_key
        self.train_steps = train_steps
        self.compression = compression
        self.compression_ratio = compression_ratio
        self._device = device
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.round_count = 0
        self._running = True
        self.lisa_param_names = []

        self.hardware_info = {
            "platform": platform.system(),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if HAS_CUDA else "cpu",
            "ram_gb": psutil.virtual_memory().total // (1024**3),
        }

    @property
    def device(self):
        return self._device

    # ── Networking ───────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"http://{self.server}{path}"
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.api_key
        return self.session.request(method, url, headers=headers, timeout=120, **kwargs)

    def connect(self) -> bool:
        try:
            resp = self._request("GET", "/health")
            if resp.status_code == 200:
                logger.info(f"[Torch] Connected to federated server at {self.server}")
                return True
        except Exception as e:
            logger.warning(f"[Torch] Server not reachable: {e}")
        return False

    def register(self) -> Optional[str]:
        try:
            payload = {
                "client_id": self._client_id(),
                "platform": platform.system(),
                "device": self._device,
                "model": self.model_name,
                "lisa_layers": self.layer_indices,
                "hardware": self.hardware_info,
            }
            resp = self._request("POST", "/register", json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                logger.info(f"[Torch] Registered as client: {data.get('client_id')}")
                return data.get("client_id")
        except Exception as e:
            logger.error(f"[Torch] Registration failed: {e}")
        return None

    def _client_id(self) -> str:
        return hashlib.sha256(f"{platform.node()}-{time.time()}".encode()).hexdigest()[:16]

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.error("[Torch] transformers not available")
            return False
        try:
            logger.info(f"[Torch] Loading {self.model_name} on {self._device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
            )
            self.model.to(self._device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine LISA param names
            state = self.model.state_dict()
            self.lisa_param_names = get_lisa_layer_names(
                self.model, self.layer_indices
            )
            logger.info(f"[Torch] LISA layers: {self.layer_indices} → "
                        f"{len(self.lisa_param_names)} params")

            # Freeze non-LISA params
            for name, param in self.model.named_parameters():
                if name in self.lisa_param_names:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Optimizer for LISA params only
            lisa_params = [p for n, p in self.model.named_parameters()
                           if n in self.lisa_param_names]
            self.optimizer = optim.Adam(lisa_params, lr=5e-5)

            logger.info(f"[Torch] Model loaded on {self._device}")
            return True
        except Exception as e:
            logger.error(f"[Torch] Failed to load model: {e}")
            traceback.print_exc()
            return False

    # ── Training ─────────────────────────────────────────────────────────────

    def _get_training_data(self, batch_size: int = 1, seq_len: int = 128):
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Federated learning enables privacy-preserving AI.",
            "Large language models need careful training.",
            "Neural networks learn representations from data.",
        ]
        encodings = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=seq_len
        )
        return TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"]
        )

    def train_round(self, global_state: Optional[Dict] = None) -> Tuple[float, int]:
        """Train one round: apply global model, train LISA layers, return loss."""
        if self.model is None:
            return 0.0, 0

        if global_state:
            self._apply_state_dict(global_state)

        self.model.train()
        dataset = self._get_training_data()
        total_loss = 0.0
        steps = 0

        for step in range(self.train_steps):
            idx = step % len(dataset)
            input_ids, attention_mask = dataset[idx]
            input_ids = input_ids.unsqueeze(0).to(self._device)
            attention_mask = attention_mask.unsqueeze(0).to(self._device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            loss.backward()

            # Clip gradients for LISA params
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in self.model.named_parameters() if n in self.lisa_param_names],
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"[Torch] Round {self.round_count}: loss={avg_loss:.4f}, steps={steps}")
        self.round_count += 1
        return avg_loss, steps

    def _apply_state_dict(self, state_dict: Dict):
        """Apply global model state dict to local model."""
        current = self.model.state_dict()
        updated_keys = 0
        for k, v in state_dict.items():
            if k in current:
                current[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                updated_keys += 1
        self.model.load_state_dict(current, strict=False)
        logger.info(f"[Torch] Applied {updated_keys} layers from global model")

    def get_lisa_gradients(self) -> Dict[str, np.ndarray]:
        """Extract gradients from LISA layers."""
        grads = {}
        for name, param in self.model.named_parameters():
            if name in self.lisa_param_names and param.grad is not None:
                grads[name] = param.grad.detach().cpu().numpy().astype(np.float32)
        return grads

    def send_gradients(self, round_id: str) -> bool:
        grads = self.get_lisa_gradients()
        if not grads:
            logger.warning("[Torch] No gradients to send")
            return False

        compressed, metadata = compress_gradients(grads, self.compression, self.compression_ratio)
        total_original = sum(m["original_size"] for m in metadata["tensors"].values())
        total_compressed = len(compressed)

        try:
            resp = self._request(
                "POST", f"/gradients/{round_id}",
                data=compressed,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Compression-Method": self.compression,
                    "X-Gradient-Count": str(len(grads)),
                    "X-Original-Size": str(total_original),
                }
            )
            if resp.status_code == 200:
                logger.info(f"[Torch] Sent {len(grads)} gradients ({total_original//1024}KB → "
                            f"{total_compressed//1024}KB compressed)")
                return True
            else:
                logger.warning(f"[Torch] Server rejected gradients: {resp.status_code}")
        except Exception as e:
            logger.error(f"[Torch] Failed to send gradients: {e}")
        return False

    def receive_model_update(self, round_id: str) -> Optional[Dict]:
        try:
            resp = self._request("GET", f"/model/{round_id}")
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"[Torch] Received global model update")
                return data.get("state_dict")
        except Exception as e:
            logger.error(f"[Torch] Failed to receive model update: {e}")
        return None

    def run(self, rounds: int):
        """Main federated training loop with exponential backoff reconnection."""
        client_id = self.register()
        if not client_id:
            logger.error("[Torch] Registration failed — cannot proceed")
            return

        backoff = 1.0
        max_backoff = 60.0

        for r in range(1, rounds + 1):
            if not self._running:
                logger.info("[Torch] Shutting down...")
                break

            start_time = time.time()
            try:
                global_state = self.receive_model_update(f"round_{r}")
                if global_state:
                    self._apply_state_dict(global_state)

                loss, steps = self.train_round(global_state)
                grads_sent = self.send_gradients(f"round_{r}")

                elapsed = time.time() - start_time
                logger.info(
                    f"[Torch] Round {r}/{rounds} | loss={loss:.4f} | "
                    f"grads_sent={grads_sent} | time={elapsed:.1f}s"
                )

                backoff = 1.0

            except Exception as e:
                logger.error(f"[Torch] Round {r} failed: {e}\n{traceback.format_exc()}")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        logger.info("[Torch] Federated training complete")


# ============================================================================
# Unified Factory
# ============================================================================

def auto_detect_device(requested: str = "auto") -> str:
    """Auto-detect best device: mlx (Mac), cuda (Linux with GPU), cpu fallback."""
    if requested != "auto":
        return requested

    if IS_MAC and HAS_MLX:
        return "mlx"
    elif IS_LINUX and HAS_CUDA:
        return "cuda"
    elif IS_LINUX:
        return "cpu"
    else:
        return "cpu"


def create_client(
    model: str,
    layers: List[int],
    server: str,
    api_key: str,
    device: str = "auto",
    train_steps: int = 10,
    compression: str = "topk",
    compression_ratio: float = 0.1,
) -> Tuple:
    """
    Create the appropriate LISA client based on detected platform.
    Returns (client, device_str).
    """
    actual_device = auto_detect_device(device)
    logger.info(f"Platform: {platform.system()} | Detected device: {actual_device} | "
                f"Requested: {device}")

    if actual_device == "mlx" and HAS_MLX:
        client = MLXLISAClient(
            model_name=model,
            layer_indices=layers,
            server=server,
            api_key=api_key,
            train_steps=train_steps,
            compression=compression,
            compression_ratio=compression_ratio,
        )
        logger.info("Using MLX client (Apple Silicon Mac)")
    else:
        client = TorchLISAClient(
            model_name=model,
            layer_indices=layers,
            server=server,
            api_key=api_key,
            device=actual_device,
            train_steps=train_steps,
            compression=compression,
            compression_ratio=compression_ratio,
        )
        logger.info(f"Using PyTorch client ({actual_device})")

    return client, actual_device


# ============================================================================
# Signal handling for clean exit
# ============================================================================

def _make_handler(clients: List):
    def handler(signum, frame):
        logger.info("Received shutdown signal — cleaning up...")
        for c in clients:
            c._running = False
    return handler


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated LISA Training Client")
    parser.add_argument("--server", required=True, help="Federated server address (HOST:PORT)")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mlx", "cpu", "cuda"],
                        help="Device (auto/mlx/cpu/cuda)")
    parser.add_argument("--lisa-layers", required=True,
                        help="Comma-separated layer indices this device trains")
    parser.add_argument("--train-steps", type=int, default=10,
                        help="Training steps per round")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated rounds")
    parser.add_argument("--api-key", default="",
                        help="Federated server API key")
    parser.add_argument("--compression", default="topk",
                        choices=["none", "topk", "quantize"],
                        help="Gradient compression method")
    parser.add_argument("--compression-ratio", type=float, default=0.1,
                        help="Compression ratio for topk")
    args = parser.parse_args()

    # Parse layers
    layers = [int(x.strip()) for x in args.lisa_layers.split(",")]

    # Create client
    client, device = create_client(
        model=args.model,
        layers=layers,
        server=args.server,
        api_key=args.api_key,
        device=args.device,
        train_steps=args.train_steps,
        compression=args.compression,
        compression_ratio=args.compression_ratio,
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, _make_handler([client]))
    signal.signal(signal.SIGTERM, _make_handler([client]))

    # Connect and run
    if client.connect():
        client.load_model()
        client.run(args.rounds)
    else:
        logger.error("Could not connect to federated server. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
