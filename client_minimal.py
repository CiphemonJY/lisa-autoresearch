#!/usr/bin/env python3
"""
Minimal LISA_FTM Federated Client
=================================
Single-file client for low-memory edge devices (RPi Zero 2 W, Jetson Nano).
Reads config from environment or CLI args. Connects to federated server,
trains LoRA adapters on selected layers, sends compressed gradients.

Minimal requirements:
  RPi Zero 2 W:  torch, transformers, datasets, psutil
  Jetson Nano:    torch (CUDA), transformers, datasets, psutil
"""
import argparse
import gc
import json
import logging
import os
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Optional: psutil for RAM monitoring (graceful fallback if unavailable)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# === Device detection ===
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_ram_usage_mb() -> float:
    """Return current process RAM usage in MB, or -1 if psutil unavailable."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return -1.0


def log_ram(log_fn, step: int = 0, prefix: str = ""):
    """Log RAM usage if psutil is available."""
    ram = get_ram_usage_mb()
    cuda_mem = ""
    if torch.cuda.is_available():
        cuda_mem = f" | CUDA {torch.cuda.memory_allocated() / 1e6:.1f}MB"
    if ram > 0:
        log_fn(f"  RAM step {step}{prefix}: {ram:.1f}MB{cuda_mem}")


# === HuggingFace model loading (memory-safe) ===
def load_model_and_tokenizer(model_id: str, device: torch.device,
                               dtype: torch.dtype = torch.float32) -> Tuple:
    """Load model + tokenizer, placing params on device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    log = logging.getLogger("edge")

    log.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading model: {model_id}")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Loaded in {time.time()-t0:.1f}s | {n_params/1e6:.1f}M params")
    return model, tokenizer, config


# === LoRA (minimal, copied from lisa/train_torch.py) ===
class LoRALinear(torch.nn.Module):
    """LoRA for nn.Linear / nn.Conv1D. Replaces y = Wx with y = Wx + BAx."""

    def __init__(self, linear: torch.nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05, target_module_name: str = ""):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout
        self.is_conv1d = isinstance(linear, torch.nn.Conv1d)

        if self.is_conv1d:
            self.in_features = linear.in_channels
            self.out_features = linear.out_channels
        else:
            self.in_features = linear.in_features
            self.out_features = linear.out_features

        self.lora_A = torch.nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
        self.scaling = alpha / rank

        for param in linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.linear(x)
        lora_in = self.lora_dropout(x)
        if self.is_conv1d:
            lora = torch.nn.functional.linear(torch.nn.functional.linear(lora_in, self.lora_A), self.lora_B)
        else:
            lora = torch.nn.functional.linear(torch.nn.functional.linear(lora_in, self.lora_A), self.lora_B)
        return original + lora * self.scaling


def apply_lora_to_model(model: torch.nn.Module, rank: int = 4, alpha: float = 8.0,
                        dropout: float = 0.05) -> int:
    """Apply LoRA to all target linear/conv1d layers. Returns layer count."""
    import torch.nn as nn
    target_modules = ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                      "c_fc", "q_attn", "k_attn", "v_attn"]

    count = 0
    for full_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        name_parts = full_name.split(".")
        if not any(tm in name_parts[-1] for tm in target_modules):
            continue

        lora = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout,
                          target_module_name=full_name)
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            try:
                parent = model.get_submodule(parent_name)
                setattr(parent, attr, lora)
                count += 1
            except KeyError:
                pass
    return count


def freeze_all_except_lora(model: torch.nn.Module) -> int:
    """Freeze all params; unfreeze only lora_* params."""
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def get_trainable_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """Return dict of trainable param name -> param (zero-copy view)."""
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


# === Gradient extraction ===
def extract_gradients(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients for trainable params as CPU tensors (already cloned)."""
    grads = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            grads[name] = param.grad.clone().cpu()
    return grads


# === Compression ===
def sparsify_tensor(tensor: torch.Tensor, k: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only top k% largest magnitudes. Returns (indices, values, shape)."""
    flat = tensor.flatten()
    num_keep = max(1, int(len(flat) * k))
    abs_flat = torch.abs(flat)
    _, top_idx = torch.topk(abs_flat, num_keep)
    values = flat[top_idx]
    indices = top_idx
    shape = torch.tensor(tensor.shape)
    return indices, values, shape


def deserialize_sparse(indices: torch.Tensor, values: torch.Tensor, shape: torch.Tensor,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct tensor from sparse representation."""
    flat = torch.zeros(indices.max().item() + 1 if len(indices) > 0 else 0, dtype=dtype)
    flat[indices] = values
    return flat.view(*shape.tolist())


def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """Quantize float32 tensor to N bits. Returns (quantized, scale, zero_point)."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    if max_val == min_val:
        scale = 1.0
        zero_point = 0.0
    else:
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = -min_val / scale
    quantized = ((tensor / scale) + zero_point).round().to(torch.uint8)
    return quantized, scale, zero_point


def dequantize_tensor(quantized: torch.Tensor, scale: float, zero_point: float,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct float tensor from quantized representation."""
    return (quantized.float() - zero_point) * scale


# === Federated protocol ===
class EdgeClient:
    """Minimal federated client for edge devices."""

    def __init__(self, server_host: str, server_port: int,
                 model_id: str = "EleutherAI/pythia-70m",
                 auth_token: Optional[str] = None,
                 lora_rank: int = 4,
                 lora_alpha: float = 8.0,
                 local_steps: int = 3,
                 timeout: int = 300):
        self.server_host = server_host
        self.server_port = server_port
        self.model_id = model_id
        self.auth_token = auth_token
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.local_steps = local_steps
        self.timeout = timeout

        self.model = None
        self.tokenizer = None
        self.lora_count = 0
        self.round_num = 0
        self.sock = None
        self.device = get_device()
        self.log = logging.getLogger("edge")

    def connect(self) -> bool:
        """Connect to federated server with optional auth."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.server_host, self.server_port))
            self.log.info(f"Connected to {self.server_host}:{self.server_port}")

            if self.auth_token:
                token_bytes = self.auth_token.encode("utf-8")
                self.sock.sendall(struct.pack("!I", len(token_bytes)) + token_bytes)
            else:
                self.sock.sendall(struct.pack("!I", 0))
            return True
        except Exception as e:
            self.log.warning(f"Could not connect: {e}")
            return False

    def send_json(self, data: dict):
        msg = json.dumps(data).encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(msg)) + msg)

    def recv_json(self) -> dict:
        header = self._recv_exact(4)
        if len(header) < 4:
            return {}
        size = struct.unpack("!I", header)[0]
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return data
            data += chunk
        return data

    def send_gradients(self, gradients: dict):
        """Send gradient dict. Uses base64 for Pi Zero (ARM) compatibility."""
        self.send_json({"type": "gradients", "round": self.round_num})
        small_data = pickle.dumps(gradients)
        self.sock.sendall(struct.pack("!I", len(small_data)) + small_data)

    def recv_model_update(self) -> dict:
        """Receive aggregated model update (pickle dict)."""
        grads = {}
        try:
            header = self.sock.recv(4)
            if len(header) < 4:
                return grads
            meta_len = struct.unpack("!I", header)[0]
            meta_bytes = b""
            while len(meta_bytes) < meta_len:
                chunk = self.sock.recv(meta_len - len(meta_bytes))
                if not chunk:
                    break
                meta_bytes += chunk

            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
                self.log.info(f"  Server: {meta}")
            except Exception:
                return grads

            n_header = self.sock.recv(4)
            if len(n_header) < 4:
                return grads
            n_bytes = struct.unpack("!I", n_header)[0]
            raw = b""
            while len(raw) < n_bytes:
                chunk = self.sock.recv(min(65536, n_bytes - len(raw)))
                if not chunk:
                    break
                raw += chunk

            if raw and n_bytes > 0:
                grads = pickle.loads(raw)
        except socket.timeout:
            self.log.warning("Timeout waiting for model update")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self.log.warning(f"Server disconnected: {e}")
        return grads

    def apply_gradients(self, updates: dict):
        """Accumulate gradient update to LoRA params."""
        for name, tensor in updates.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                update = tensor.to(param.device)
                if update.shape == param.shape:
                    param.data.add_(update)
        self.log.info(f"  Applied {len(updates)} gradient updates")

    # --- Model lifecycle ---
    def load_model(self) -> bool:
        """Load model + tokenizer onto detected device."""
        try:
            self.model, self.tokenizer, _ = load_model_and_tokenizer(
                self.model_id, self.device
            )
            return True
        except Exception as e:
            self.log.error(f"Model load failed: {e}")
            return False

    def setup_lora(self) -> int:
        """Apply LoRA and freeze non-trainable params."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self.lora_count = apply_lora_to_model(
            self.model, rank=self.lora_rank, alpha=self.lora_alpha
        )
        freeze_all_except_lora(self.model)
        self.log.info(f"  LoRA applied to {self.lora_count} layers")
        return self.lora_count

    # --- Training ---
    def train_local(self, texts: list, n_steps: int = 3) -> float:
        """Train LoRA layers locally; returns avg loss. No optimizer.step (gradients extracted for server)."""
        if not texts:
            texts = ["The quick brown fox jumps over the lazy dog."] * 50

        if self.lora_count == 0:
            self.setup_lora()

        enc = self.tokenizer(texts, truncation=True, max_length=128,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].clamp(0, self.tokenizer.vocab_size - 1)
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Ensure only LoRA params require gradients
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=0.01,
        )

        self.model.train()
        losses = []
        batch_size = 1 if self.device.type == "cpu" else 4

        for step in range(n_steps):
            idx = torch.randperm(len(input_ids))[:batch_size].tolist()
            ids = input_ids[idx].to(self.device)
            mask = attention_mask[idx].to(self.device)
            labs = labels[idx].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labs.view(-1),
                ignore_index=self.tokenizer.pad_token_id or -100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            losses.append(loss.item())

            if (step + 1) % 5 == 0:
                self.log.info(f"    Step {step+1}/{n_steps} | loss={loss.item():.4f}")

            # Log RAM every 10 steps
            if (step + 1) % 10 == 0:
                log_ram(self.log.info, step + 1)

        return sum(losses) / len(losses)

    # --- Federated round ---
    def run_round(self, data: list = None) -> bool:
        """One federated round: train, extract gradients, send, receive update."""
        self.log.info(f"\n=== Round {self.round_num} ===")

        avg_loss = self.train_local(data, n_steps=self.local_steps)
        self.log.info(f"  Local training done: avg_loss={avg_loss:.4f}")

        # Extract gradients (zero-copy views already cloned above)
        grads = extract_gradients(self.model)
        self.log.info(f"  Extracted {len(grads)} gradient tensors")

        if self.sock is None:
            self._save_gradients_locally(grads)
            self.round_num += 1
            return True

        try:
            self.send_gradients(grads)
            updates = self.recv_model_update()
            if updates:
                self.apply_gradients(updates)
                self.log.info("  Round complete — model updated")
            else:
                self.log.info("  No update from server — saved locally")
                self._save_gradients_locally(grads)
        except Exception as e:
            self.log.warning(f"Round failed: {e}")
            self._save_gradients_locally(grads)

        self.round_num += 1
        return True

    def _save_gradients_locally(self, grads: dict):
        out_dir = Path("output/federated_grads")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(grads, out_dir / f"round_{self.round_num}_grads.pt")
        self.log.info(f"  Saved gradients to {out_dir}/round_{self.round_num}_grads.pt")

    def disconnect(self):
        if self.sock:
            try:
                self.send_json({"type": "disconnect"})
            except Exception:
                pass
            self.sock.close()
            self.sock = None


# === Entry point ===
def main():
    parser = argparse.ArgumentParser(description="Minimal LISA_FTM Edge Client")
    parser.add_argument("--server", default="http://127.0.0.1:8080",
                        help="Server address (host:port or full URL)")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--auth-token", default=os.environ.get("LISA_AUTH_TOKEN", ""))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-steps", type=int, default=3,
                        help="Training steps per federated round")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Parse server host:port
    server_addr = args.server.replace("http://", "").replace("https://", "")
    if ":" in server_addr:
        host, port_str = server_addr.rsplit(":", 1)
        server_port = int(port_str)
    else:
        host = server_addr
        server_port = 8080

    device = torch.device(args.device if args.device != "auto"
                           else "cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("edge")

    log.info("=" * 50)
    log.info("LISA_FTM Minimal Edge Client")
    log.info("=" * 50)
    log.info(f"Model:     {args.model}")
    log.info(f"Device:    {device}")
    log.info(f"Server:    {host}:{server_port}")
    log.info(f"LoRA rank: {args.lora_rank}")
    log.info(f"Steps/rnd: {args.local_steps}")
    log.info(f"Rounds:    {args.rounds}")

    ram0 = get_ram_usage_mb()
    if ram0 > 0:
        log.info(f"RAM start: {ram0:.1f}MB")
    if torch.cuda.is_available():
        log.info(f"CUDA:      {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Initialise client
    client = EdgeClient(
        server_host=host,
        server_port=server_port,
        model_id=args.model,
        auth_token=args.auth_token or None,
        lora_rank=args.lora_rank,
        local_steps=args.local_steps,
        timeout=args.timeout,
    )
    client.device = device

    # Load model
    if not client.load_model():
        log.error("Failed to load model — exiting")
        sys.exit(1)

    # Move model to device
    client.model = client.model.to(device)
    client.setup_lora()

    log_ram(log.info, 0, " (after model load)")

    # Connect to server
    connected = client.connect()

    # Run federated rounds
    for rnd in range(args.rounds):
        client.round_num = rnd
        client.run_round(data=None)
        log_ram(log.info, rnd, " (after round)")
        time.sleep(0.5)

    if connected:
        client.disconnect()

    log.info("\nDone. Minimal edge client complete.")


if __name__ == "__main__":
    main()
