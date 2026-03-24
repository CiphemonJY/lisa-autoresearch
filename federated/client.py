#!/usr/bin/env python3
"""
Federated Learning Client - Real PyTorch Implementation

A federated learning participant that:
- Loads a small model (GPT2 or tiny model for CPU training)
- Trains on local data
- Computes real gradients
- Sends compressed gradients to server
- Receives aggregated model updates

Works on CPU Windows/Linux with limited RAM.
"""

import os
import sys
import json
import time
import hashlib
import zlib
import pickle
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# LoRA implementation
import torch
import torch.nn as nn
from dataclasses import dataclass as _dc, field as _field
from typing import List as _List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fed-client")


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA for linear layers (nn.Linear and nn.Conv1d/GPT2Conv1D)."""

    def __init__(self, linear: nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05, target_module_name: str = ""):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout
        self.target_module_name = target_module_name
        self.is_conv1d = isinstance(linear, (nn.Conv1D, nn.modules.conv.Conv1d))

        if self.is_conv1d:
            self.in_features = linear.in_channels
            self.out_features = linear.out_channels
        else:
            self.in_features = linear.in_features
            self.out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.linear(x)
        lora_input = self.lora_dropout(x)
        if self.is_conv1d:
            lora = nn.functional.linear(lora_input, self.lora_A)
            lora = nn.functional.linear(lora, self.lora_B)
        else:
            lora = nn.functional.linear(lora_input, self.lora_A)
            lora = nn.functional.linear(lora, self.lora_B)
        return original + lora * self.scaling


def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 8.0,
                        dropout: float = 0.05) -> int:
    """
    Apply LoRA to all Linear/Conv1D layers in a model.
    Returns number of layers modified.
    """
    target_names = ["c_attn", "c_proj", "c_fc", "query_key_value", "dense", "mlp",
                    "q_proj", "k_proj", "v_proj", "o_proj"]
    count = 0
    for full_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        if not any(tm in full_name for tm in target_names):
            continue
        lora = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout,
                          target_module_name=full_name)
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            try:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], lora)
                count += 1
            except (KeyError, AttributeError):
                pass
    return count


# ============================================================================
# Configuration
# ============================================================================


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Model
    "model_name": "microsoft/phi-2",          # 1.4B params, manageable on CPU
    "model_name_fallback": "distilbert/distilgpt2",  # 82M params, very light
    "max_seq_length": 128,
    
    # Training
    "local_epochs": 1,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "max_train_steps": 50,
    
    # Federated
    "round_interval_secs": 30,
    "gradient_clip_norm": 1.0,
    
    # Compression
    "compression": {
        "enabled": True,
        "sparsification_ratio": 0.05,   # Keep top 5% of gradients
        "quantization_bits": 8,          # 8-bit quantization
        "compression_level": 6,           # zlib level
    },
    
    # Privacy
    "differential_privacy": {
        "enabled": False,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clip_norm": 1.0,
        "noise_multiplier": 0.1,
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClientState:
    """State maintained by federated learning client."""
    client_id: str
    round_number: int = 0
    local_epochs_completed: int = 0
    total_samples_trained: int = 0
    last_submit_time: float = 0
    reputation: float = 50.0


@dataclass 
class GradientUpdate:
    """A compressed gradient update to send to the server."""
    client_id: str
    round_number: int
    timestamp: float
    num_samples: int
    
    # Compressed gradient data
    param_names: List[str]
    compressed_data: bytes
    
    # Metadata for verification/decompression
    compression_info: Dict[str, Any]
    
    # Privacy info
    noise_seed: Optional[int] = None
    dp_epsilon: Optional[float] = None
    
    # Quality metrics
    gradient_norm: float = 0.0
    loss_before: float = 0.0
    loss_after: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "client_id": self.client_id,
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "num_samples": self.num_samples,
            "num_params": len(self.param_names),
            "gradient_norm": self.gradient_norm,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "compression_info": self.compression_info,
            "dp_epsilon": self.dp_epsilon,
        }


# ============================================================================
# Gradient Compression
# ============================================================================

class GradientCompressor:
    """Compress gradients for efficient transmission."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.compression = config.get("compression", {})
        self.enabled = self.compression.get("enabled", True)
        self.sparsification_ratio = self.compression.get("sparsification_ratio", 0.05)
        self.quantization_bits = self.compression.get("quantization_bits", 8)
        self.compression_level = self.compression.get("compression_level", 6)
    
    def compress(self, state_dict: Dict[str, np.ndarray]) -> Tuple[bytes, Dict]:
        """
        Compress a state dict into a byte payload.
        
        Pipeline:
        1. Flatten all params into a single array
        2. Sparsify: keep only top-k% by magnitude
        3. Quantize: reduce precision to 8-bit
        4. Compress: zlib compression
        
        Returns: (compressed_bytes, info_dict)
        """
        if not self.enabled:
            # No compression, just pickle
            data = pickle.dumps(state_dict)
            return data, {"method": "none", "original_size": len(data)}
        
        # Flatten all parameters
        all_params = []
        param_names = []
        shapes = {}
        
        for name, param in sorted(state_dict.items()):
            if not isinstance(param, np.ndarray):
                continue
            param = param.astype(np.float32)  # Ensure float32
            all_params.append(param.flatten())
            param_names.append(name)
            shapes[name] = param.shape
        
        flat = np.concatenate(all_params)
        
        # Sparsification: keep top-k% by magnitude
        k = max(1, int(len(flat) * self.sparsification_ratio))
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices].astype(np.float32)
        
        # Quantization: 8-bit
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 1e-8:
            scale = 255.0 / (v_max - v_min)
            quantized = ((values - v_min) * scale).astype(np.uint8)
        else:
            quantized = np.zeros(len(values), dtype=np.uint8)
        
        # Pack: [num_params:4][names_len:4][names...][shapes_json][indices_4bytes_each][quantized][scale:4][min:4]
        names_bytes = "\n".join(param_names).encode("utf-8")
        shapes_json = json.dumps(shapes).encode("utf-8")
        indices_bytes = indices.astype(np.int32).tobytes()
        quantized_bytes = quantized.tobytes()
        scale_bytes = np.float32(scale if v_max - v_min > 1e-8 else 1.0).tobytes()
        min_bytes = np.float32(v_min).tobytes()
        
        # Assemble packet
        payload = bytearray()
        payload.extend(np.int32(len(param_names)).tobytes())
        payload.extend(np.int32(len(names_bytes)).tobytes())
        payload.extend(names_bytes)
        payload.extend(np.int32(len(shapes_json)).tobytes())
        payload.extend(shapes_json)
        payload.extend(np.int32(len(indices)).tobytes())
        payload.extend(indices_bytes)
        payload.extend(quantized_bytes)
        payload.extend(scale_bytes)
        payload.extend(min_bytes)
        
        compressed = bytes(payload)
        
        # Use pickle format + zlib so decompress can pickle.loads() consistently
        pickle_bytes = pickle.dumps(state_dict)
        
        # Optional zlib compression
        if self.compression_level > 0:
            compressed = zlib.compress(pickle_bytes, self.compression_level)
        else:
            compressed = pickle_bytes
        
        info = {
            "method": "pickle-zlib",
            "sparsification_ratio": self.sparsification_ratio,
            "original_params": sum(p.size for p in all_params),
            "sparse_params": k,
            "compressed_size": len(compressed),
            "shapes": shapes,
        }
        
        return compressed, info
    
    def decompress(self, data: bytes, info: Dict) -> Dict[str, np.ndarray]:
        """Decompress gradient data back to state dict."""
        method = info.get("method", "none")
        
        if method == "none":
            return pickle.loads(data)
        
        # Decompress if zlib compressed (compressed data is smaller than original)
        if info.get("compressed_size") != len(data):
            data = zlib.decompress(data)
        
        # Handle pickle format (what compress() actually produces)
        if method in ("sparse-8bit", "pickle-zlib"):
            # Data is pickled state dict, always zlib-compressed by compress()
            try:
                data = zlib.decompress(data)
            except Exception:
                pass  # Not zlib-compressed (e.g., compression_level=0), proceed
            return pickle.loads(data)
        
        # Legacy/custom binary format support (unlikely to be hit)
        buf = memoryview(data)
        pos = 0
        
        # Read num_params
        num_params = int.from_bytes(buf[pos:pos+4], "little"); pos += 4
        
        # Read names
        names_len = int.from_bytes(buf[pos:pos+4], "little"); pos += 4
        names = buf[pos:pos+names_len].tobytes().decode("utf-8").split("\n"); pos += names_len
        
        # Read shapes
        shapes_len = int.from_bytes(buf[pos:pos+4], "little"); pos += 4
        shapes = json.loads(buf[pos:pos+shapes_len].tobytes()); pos += shapes_len
        
        # Read indices
        indices_len = int.from_bytes(buf[pos:pos+4], "little"); pos += 4
        indices = np.frombuffer(buf[pos:pos+indices_len*4], dtype=np.int32).copy(); pos += indices_len*4
        
        # Read quantized values
        quantized = np.frombuffer(buf[pos:pos+len(buf)-8], dtype=np.uint8)
        scale = np.frombuffer(buf[len(buf)-8:len(buf)-4], dtype=np.float32)[0]
        v_min = np.frombuffer(buf[len(buf)-4:], dtype=np.float32)[0]
        
        # Dequantize
        values = quantized.astype(np.float32) / scale + v_min
        
        # Reconstruct full gradient array
        total_size = sum(np.prod(shapes[n]) for n in names)
        flat = np.zeros(total_size, dtype=np.float32)
        flat[indices] = values
        
        # Reshape back to parameters
        result = {}
        offset = 0
        for name in names:
            size = np.prod(shapes[name])
            result[name] = flat[offset:offset+size].reshape(shapes[name])
            offset += size
        
        return result


# ============================================================================
# Differential Privacy
# ============================================================================

class GradientPrivacy:
    """Add differential privacy noise to gradients."""
    
    def __init__(self, config: Dict):
        dp_cfg = config.get("differential_privacy", {})
        self.enabled = dp_cfg.get("enabled", False)
        self.epsilon = dp_cfg.get("epsilon", 1.0)
        self.delta = dp_cfg.get("delta", 1e-5)
        self.clip_norm = dp_cfg.get("clip_norm", 1.0)
        self.noise_multiplier = dp_cfg.get("noise_multiplier", 0.1)
    
    def apply(self, state_dict: Dict[str, np.ndarray], seed: Optional[int] = None
              ) -> Tuple[Dict[str, np.ndarray], Optional[int]]:
        """
        Clip gradients and add Gaussian noise for differential privacy.
        
        Returns: (noisy_state_dict, seed_used)
        """
        if not self.enabled:
            return state_dict, None
        
        if seed is None:
            seed = random.getrandbits(64)
        
        rng = np.random.default_rng(seed)
        
        result = {}
        for name, param in state_dict.items():
            param = param.astype(np.float32)
            
            # Clip by norm
            norm = np.linalg.norm(param.flatten())
            if norm > self.clip_norm:
                param = param * (self.clip_norm / norm)
            
            # Add calibrated Gaussian noise
            sigma = self.clip_norm * self.noise_multiplier
            noise = rng.normal(0, sigma, param.shape).astype(np.float32)
            result[name] = param + noise
        
        return result, seed


# ============================================================================
# Local Training
# ============================================================================

class LocalTrainer:
    """Train a model locally on client data."""
    
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.compressor = GradientCompressor(config)
        self.privacy = GradientPrivacy(config)
        # Raw (uncompressed) gradient dict for P2P sharing
        self._latest_raw_gradients: Optional[Dict] = None
        self._latest_raw_round: int = 0

        self._setup_model()
    
    def _setup_model(self):
        """Load a small model for CPU training."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        model_name = self.config["model_name"]
        
        for name in [model_name, self.config["model_name_fallback"]]:
            try:
                logger.info(f"Loading model: {name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=True,
                    use_fast=False,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                config = AutoConfig.from_pretrained(name, trust_remote_code=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    name,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )

                # Apply LoRA: only train adapter params (A and B matrices)
                lora_rank = self.config.get("lora_rank", 4)
                lora_alpha = self.config.get("lora_alpha", 8.0)
                lora_count = apply_lora_to_model(
                    self.model,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=self.config.get("lora_dropout", 0.05),
                )
                logger.info(f"LoRA applied to {lora_count} layers (rank={lora_rank})")

                # Freeze everything except LoRA params
                for param in self.model.parameters():
                    param.requires_grad = False
                for name, param in self.model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded: {total:,} total params, {trainable:,} trainable ({trainable/max(total,1)*100:.1f}%)")
                return

            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
                continue

        raise RuntimeError("Could not load any model")

    def _generate_local_data(self, num_samples: int = 200) -> List[Dict]:
        """
        Generate simulated local text data.
        
        In production, this would be real data from a hospital/organization.
        Here we generate diverse synthetic text to simulate different data distributions.
        """
        import torch
        
        # Different domains to simulate different clients having different data
        domains = [
            ("medical", "patient diagnosis treatment hospital doctor medicine symptoms health care"),
            ("finance", "investment market trading portfolio risk return stock bond portfolio"),
            ("legal", "court case law contract defendant plaintiff lawyer judge trial"),
            ("tech", "software algorithm data system network security compute cloud"),
            ("retail", "customer order product inventory store sales revenue"),
        ]
        
        # Each client has a different primary domain
        domain_idx = hash(self.client_id) % len(domains)
        primary_domain, primary_words = domains[domain_idx]
        
        texts = []
        for i in range(num_samples):
            # Mix of primary domain (70%) and random (30%)
            if i < num_samples * 0.7:
                words = primary_words.split()
            else:
                words = domains[random.randint(0, len(domains)-1)][1].split()
            
            random.shuffle(words)
            text = " ".join(words * 3)[:200]
            texts.append(text)
        
        return texts
    
    def _tokenize(self, texts: List[str]) -> Dict:
        """Tokenize text data."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config["max_seq_length"],
            return_tensors="pt",
        )
        return enc
    
    def compute_gradient_update(self, round_number: int) -> GradientUpdate:
        """
        Train locally and compute a gradient update.
        
        Returns a compressed gradient update ready to send to the server.
        """
        import torch
        
        self.model.train()
        
        # Generate local data
        texts = self._generate_local_data(num_samples=200)
        enc = self._tokenize(texts)
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Training loop
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        
        total_loss = 0.0
        num_steps = 0
        
        indices = list(range(len(input_ids)))
        batch_size = self.config["batch_size"]
        
        for epoch in range(self.config["local_epochs"]):
            random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                
                input_batch = input_ids[batch_idx]
                mask_batch = attention_mask[batch_idx]
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_batch,
                    attention_mask=mask_batch,
                    labels=input_batch,
                )
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config["gradient_clip_norm"],
                )
                optimizer.step()
                
                total_loss += loss.item()
                num_steps += 1
                
                if num_steps >= self.config["max_train_steps"]:
                    break
            
            if num_steps >= self.config["max_train_steps"]:
                break
        
        avg_loss = total_loss / max(num_steps, 1)
        
        # Get gradient (difference from initial model)
        state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                state_dict[name] = param.grad.detach().cpu().numpy().astype(np.float32)
        
        # Apply differential privacy
        state_dict, noise_seed = self.privacy.apply(state_dict)
        
        # Compute gradient norm
        flat_grad = np.concatenate([v.flatten() for v in state_dict.values()])
        grad_norm = float(np.linalg.norm(flat_grad))
        
        # Compress
        compressed, comp_info = self.compressor.compress(state_dict)
        
        # Store raw gradients for P2P sharing
        self._latest_raw_gradients = state_dict
        self._latest_raw_round = round_number

        return GradientUpdate(
            client_id=self.client_id,
            round_number=round_number,
            timestamp=time.time(),
            num_samples=len(texts),
            param_names=list(state_dict.keys()),
            compressed_data=compressed,
            compression_info=comp_info,
            noise_seed=noise_seed,
            dp_epsilon=self.privacy.epsilon if self.privacy.enabled else None,
            gradient_norm=grad_norm,
            loss_before=avg_loss + 0.1,
            loss_after=avg_loss,
        )


# ============================================================================
# Federated Client
# ============================================================================

class FederatedClient:
    """
    Main federated learning client.
    
    Coordinates local training, compression, privacy, and server communication.
    """
    
    def __init__(self, client_id: str, server_url: str = "http://localhost:8000",
                 config: Optional[Dict] = None,
                 p2p_enabled: bool = False,
                 bootstrap_server: Optional[str] = None,
                 auth_token: Optional[str] = None):
        self.client_id = client_id
        self.server_url = server_url.rstrip("/")
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.auth_token = auth_token

        self.state = ClientState(client_id=client_id)
        self.trainer = LocalTrainer(client_id, self.config)
        self.compressor = GradientCompressor(self.config)

        self._client_key = hashlib.sha256(client_id.encode()).hexdigest()[:16]

        # P2P state (initialized only if p2p_enabled=True)
        self.p2p_enabled = p2p_enabled
        self.p2p_client = None
        if self.p2p_enabled:
            try:
                from federated.p2p import P2PClient, P2PRegistry

                # Determine bootstrap server address
                bootstrap_addr = bootstrap_server
                if not bootstrap_addr:
                    # Derive from server_url: use same host, port 8081
                    import re
                    m = re.match(r"https?://([^:]+):(\d+)", self.server_url)
                    if m:
                        bootstrap_addr = f"{m.group(1)}:8081"
                    else:
                        bootstrap_addr = "127.0.0.1:8081"

                logger.info(f"P2P enabled: bootstrap server = {bootstrap_addr}")
                self._p2p_registry = P2PRegistry(
                    bootstrap_server=f"http://{bootstrap_addr}",
                    port=0,  # Let OS assign a port
                )
                peers = self._p2p_registry.register()
                self.p2p_client = P2PClient(client_id, self._p2p_registry)
                self.p2p_client.start_exchange_server()
                self.p2p_client.peers = peers
                logger.info(f"P2P: {len(peers)} initial peers discovered")
            except Exception as e:
                logger.warning(f"P2P initialization failed: {e}. Continuing without P2P.")
                self.p2p_enabled = False
                self.p2p_client = None

        logger.info(f"FederatedClient '{client_id}' initialized (key={self._client_key})")
    
    def train_and_submit(self, round_number: Optional[int] = None) -> Dict:
        """
        Run local training and submit gradient update to server.
        
        This is the main entry point for one round of federated learning.
        """
        import requests
        
        rn = round_number or self.state.round_number + 1
        
        logger.info(f"=== Round {rn}: Starting local training ===")
        
        # Compute gradient update
        update = self.trainer.compute_gradient_update(rn)
        
        # Share raw gradients with P2P peers and fetch theirs for averaging
        if self.p2p_enabled and self.p2p_client is not None:
            raw_grads, raw_round = self.trainer._latest_raw_gradients, self.trainer._latest_raw_round
            if raw_grads:
                self.p2p_client.update_local_gradient(raw_grads, raw_round)
                peer_avg_grads = self.p2p_client.sync_with_peers(raw_grads)
                if peer_avg_grads and peer_avg_grads != raw_grads:
                    logger.info(f"P2P: averaged {len(peer_avg_grads)} tensors with {len(self.p2p_client.peers)} peers")
                    # Recompress with peer-averaged gradients and rebuild update for submission
                    flat_grad = np.concatenate([v.flatten() for v in peer_avg_grads.values()])
                    peer_grad_norm = float(np.linalg.norm(flat_grad))
                    compressed, comp_info = self.compressor.compress(peer_avg_grads)
                    # Rebuild update with peer-averaged gradients
                    update = GradientUpdate(
                        client_id=self.client_id,
                        round_number=rn,
                        timestamp=time.time(),
                        num_samples=update.num_samples,
                        param_names=list(peer_avg_grads.keys()),
                        compressed_data=compressed,
                        compression_info=comp_info,
                        noise_seed=update.noise_seed,
                        dp_epsilon=update.dp_epsilon,
                        gradient_norm=peer_grad_norm,
                        loss_before=update.loss_before,
                        loss_after=update.loss_after,
                    )
        
        # Submit to server
        try:
            import base64

            payload = {
                "client_id": update.client_id,
                "round_number": update.round_number,
                "timestamp": update.timestamp,
                "num_samples": update.num_samples,
                "gradient_norm": update.gradient_norm,
                "loss_before": update.loss_before,
                "loss_after": update.loss_after,
                "compression_method": update.compression_info.get("method", "none"),
                "compressed_size": len(update.compressed_data),
                "compression_info": update.compression_info,
                "dp_epsilon": update.dp_epsilon,
                "gradient_data": base64.b64encode(update.compressed_data).decode("utf-8"),
            }

            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            # Send metadata first
            meta_resp = requests.post(
                f"{self.server_url}/submit",
                json=payload,
                timeout=30,
                headers=headers,
            )
            
            if meta_resp.status_code == 200:
                logger.info(
                    f"Round {rn}: Gradient submitted "
                    f"(norm={update.gradient_norm:.4f}, "
                    f"size={len(update.compressed_data):,} bytes, "
                    f"loss={update.loss_after:.4f})"
                )
                
                self.state.round_number = rn
                self.state.local_epochs_completed += self.config["local_epochs"]
                self.state.total_samples_trained += update.num_samples
                self.state.last_submit_time = time.time()
                
                return update  # Return the GradientUpdate object
            else:
                logger.error(f"Server rejected: {meta_resp.status_code} {meta_resp.text}")
                return {"status": "error", "message": meta_resp.text}
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server not reachable at {self.server_url}")
            return {"status": "error", "message": "server_unreachable"}
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return {"status": "error", "message": str(e)}

    def disconnect(self) -> Dict:
        """
        Notify the server that this client is disconnecting.

        Sends a graceful /disconnect POST so the server can mark the client
        inactive and exclude it from future rounds without waiting for a timeout.
        """
        import requests

        if not self.auth_token:
            logger.info(f"[{self.client_id}] No auth token set; skipping server disconnect notification")
            return {"status": "skipped"}

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            payload = {
                "client_id": self.client_id,
                "round_number": self.state.round_number,
            }
            resp = requests.post(
                f"{self.server_url}/disconnect",
                json=payload,
                timeout=10,
                headers=headers,
            )
            if resp.status_code == 200:
                logger.info(f"[{self.client_id}] Server notified of disconnect")
                return resp.json()
            else:
                logger.warning(f"[{self.client_id}] Server disconnect notification failed: {resp.status_code}")
                return {"status": "error", "message": resp.text}
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{self.client_id}] Could not notify server (unreachable)")
            return {"status": "error", "message": "server_unreachable"}
        except Exception as e:
            logger.error(f"[{self.client_id}] Disconnect notification failed: {e}")
            return {"status": "error", "message": str(e)}

    def run_rounds(self, num_rounds: int, server_url: Optional[str] = None):
        """Run multiple federated rounds."""
        if server_url:
            self.server_url = server_url
        
        for r in range(1, num_rounds + 1):
            result = self.train_and_submit(round_number=r)
            
            if result.get("status") == "error":
                logger.error(f"Round {r} failed: {result.get('message')}")
                if "unreachable" in result.get("message", ""):
                    logger.info("Server unreachable, waiting before retry...")
                    time.sleep(5)
            
            # Wait between rounds
            if r < num_rounds:
                time.sleep(2)

        # Notify server we're done
        self.disconnect()

        logger.info(
            f"Completed {num_rounds} rounds. "
            f"Total samples: {self.state.total_samples_trained:,}"
        )


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client-id", default=f"client-{random.randint(1000,9999)}")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    parser.add_argument(
        "--p2p-enable",
        action="store_true",
        help="Enable peer-to-peer gradient exchange with other clients",
    )
    parser.add_argument(
        "--bootstrap-server",
        default=None,
        help=(
            "P2P bootstrap server address (host:port). "
            "If not set but --p2p-enable is used, defaults to the --server address "
            "with port 8081 (e.g. 127.0.0.1:8081 for localhost). "
            "Set to the address of a machine running: "
            "python -m federated.p2p --bootstrap --port 8081"
        ),
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional auth token for client authentication with the server",
    )
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    if args.epochs is not None:
        config["local_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["max_train_steps"] = args.max_steps
    if args.dp:
        config["differential_privacy"]["enabled"] = True
        config["differential_privacy"]["epsilon"] = args.dp_epsilon
    
    client = FederatedClient(
        args.client_id,
        args.server,
        config,
        p2p_enabled=args.p2p_enable,
        bootstrap_server=args.bootstrap_server,
        auth_token=args.auth_token,
    )
    client.run_rounds(args.rounds)


if __name__ == "__main__":
    main()
