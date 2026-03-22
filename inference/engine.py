#!/usr/bin/env python3
"""
Inference Engine — Clean checkpoint loading and generation.

This module provides a functional inference pipeline for LISA_FTM checkpoints.

Key fixes vs naive approaches:
1. NEVER hardcode hidden_size from a template that doesn't match the checkpoint.
   Extract shape info directly from checkpoint weights.
2. For checkpoints saved after resize_token_embeddings(), extract vocab_size
   from the embedding weight shape (e.g. 50277 vs base model 50304 for Pythia).
3. Use ignore_mismatched_sizes=True so the model can resize embeddings.
4. Prefer full checkpoint directories (model.safetensors + config.json) when
   available; fall back to HuggingFace base + .pt state_dict overlay.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch

# ────────────────────────────────────────────────────────────────────────────
# Backward-compatible re-exports (matches old inference/__init__.py)
# The old LISAInference/KVCache/InferenceConfig were a non-functional simulator.
# We re-create minimal stubs here for API compatibility.
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Stub — kept only for backward API compatibility."""
    model_path: str = ""
    num_layers: int = 96
    hidden_size: int = 12288
    max_new_tokens: int = 512
    temperature: float = 0.7
    vocab_size: int = 32000
    quantization_bits: int = 4
    lisa_ratio: float = 0.05
    use_kv_cache: bool = True

    def get_memory_requirements(self):
        """Stub memory calculation for test compatibility."""
        total_params = self.num_layers * self.hidden_size * self.vocab_size
        return {
            "total_params": total_params,
            "total_memory_gb": total_params * 4 / 1e9,
            "lisah_memory_gb": total_params * self.lisa_ratio * 4 / 1e9,
            "compression_ratio": 4.0,
        }


class KVCache:
    """Stub — kept only for backward API compatibility."""
    def __init__(self, config=None):
        self.config = config or InferenceConfig()
        self.current_len = 0
        self.max_len = 2048
        self._store = {}  # layer_id -> (keys, values)

    def clear(self):
        self.current_len = 0
        self._store.clear()

    def get_memory_usage(self):
        return self.current_len * 512 * 512 * 4  # rough estimate

    def update(self, layer_id, keys, values):
        self._store[layer_id] = (keys, values)
        self.current_len = keys.shape[1] if hasattr(keys, 'shape') else len(keys)

    def get(self, layer_id):
        return self._store.get(layer_id, (None, None))


class LISAInference:
    """Stub — kept only for backward API compatibility.

    The real inference is done via load_checkpoint() + model.generate().
    This stub class exists only so that code importing from the old
    inference.engine module won't break at import time.
    """
    def __init__(self, config=None):
        self.config = config or InferenceConfig()
        num_layers = getattr(self.config, 'num_layers', 96)
        ratio = getattr(self.config, 'lisa_ratio', 0.05)
        self.layer_assignments = {}
        ram_count = max(1, int(num_layers * ratio))
        for i in range(num_layers):
            self.layer_assignments[i] = "ram" if i < ram_count or i >= num_layers - ram_count else "disk"
        self.stats = {"total_inferences": 0}

    def _get_ram_layer_indices(self, count):
        """Return indices of RAM layers (top and bottom count layers)."""
        n = self.config.num_layers
        bottom = list(range(count))
        top = list(range(n - count, n))
        return bottom + top

    def load_model(self, model_path):
        log.warning("LISAInference.load_model is a stub; use load_checkpoint() instead")

    def forward(self, input_ids):
        yield input_ids

    def get_stats(self):
        return self.stats


class InferenceServer:
    """Stub for test compatibility."""
    def __init__(self, config=None):
        self.config = config or InferenceConfig()

    def get_stats(self):
        return {
            "requests_served": 0,
            "inference_stats": {},
            "config": {"num_layers": getattr(self.config, 'num_layers', 96)},
        }


class BatchedInference:
    """Stub for test compatibility."""
    def __init__(self, config=None):
        self.config = config or InferenceConfig()
        self.request_queue = []


# Module-level flags for dependency availability
try:
    import torch as _torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as _numpy
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


__all__ = [
    # Real implementation
    "load_checkpoint",
    "detect_model_type",
    "run_generation",
    "generate",
    "find_latest_checkpoint",
    "inspect_checkpoint",
    # Backward compat stubs
    "LISAInference",
    "InferenceServer",
    "BatchedInference",
    "InferenceConfig",
    "KVCache",
    # Availability flags
    "HAS_TORCH",
    "HAS_NUMPY",
]

log = logging.getLogger("inference-engine")


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint detection
# ────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest .pt checkpoint in output dir. Prefers final_model.pt."""
    checkpoints = list(output_dir.glob("*.pt"))
    if not checkpoints:
        return None
    final = output_dir / "final_model.pt"
    if final.exists():
        return final
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def detect_model_type(checkpoint_dir: Path) -> str:
    """
    Detect model architecture from checkpoint directory.

    Returns: "pythia" (GPT-NeoX), "tinyllama" (Llama), or "unknown"
    """
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if "gpt2" in model_type.lower() or "gpt-neox" in model_type.lower():
            return "pythia"
        elif "llama" in model_type.lower():
            return "tinyllama"
        # Check hidden_size as fallback clue
        n_embd = cfg.get("n_embd", cfg.get("hidden_size", 0))
        if n_embd == 512:
            return "pythia"
        elif n_embd == 2048:
            return "tinyllama"

    # Inspect checkpoint key format
    ckpt = find_latest_checkpoint(checkpoint_dir)
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        first_key = next(iter(state.keys()), "")
        del state
        if "gpt_neox" in first_key:
            return "pythia"
        elif "model.embed_tokens" in first_key or ("lm_head" in first_key and "layers" in first_key):
            return "tinyllama"
    return "unknown"


def inspect_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """Extract shape info from a checkpoint for config building."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    info = {"keys": len(state), "layers": 0}

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "embed" in k or "lm_head" in k:
            info["vocab_size"] = v.shape[0]
            info["hidden_size"] = v.shape[1]
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "gpt_neox" and parts[2] == "layers":
            info["layers"] = max(info["layers"], int(parts[1]) + 1)
        elif len(parts) >= 3 and parts[1] == "layers":
            # Llama/TinyLlama: model.layers.N.xxx
            info["layers"] = max(info["layers"], int(parts[2]) + 1)

    del state
    return info


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def _load_pythia_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Load a Pythia-70m + LoRA checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_path = checkpoint_dir / "config.json"
    safetensors_path = checkpoint_dir / "model.safetensors"
    base_model_id = "EleutherAI/pythia-70m"

    has_full = config_path.exists() and safetensors_path.exists()

    if has_full:
        log.info(f"Full checkpoint layout detected in {checkpoint_dir}")
        config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        base_source = str(checkpoint_dir)
    else:
        log.info(f"No full layout — loading from HuggingFace: {base_model_id}")
        base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

        # Inspect checkpoint for exact shapes
        ckpt = checkpoint_dir / (checkpoint_name or "final_model.pt")
        if not ckpt.exists():
            ckpt = find_latest_checkpoint(checkpoint_dir)
        if ckpt:
            info = inspect_checkpoint(ckpt)
            log.info(f"  Checkpoint info: {info}")
            if "vocab_size" in info:
                log.info(f"  Overriding vocab_size: {base_config.vocab_size} -> {info['vocab_size']}")
                base_config.vocab_size = info["vocab_size"]
            if "hidden_size" in info:
                log.info(f"  Overriding hidden_size: {base_config.hidden_size} -> {info['hidden_size']}")
                base_config.hidden_size = info["hidden_size"]
            if info.get("layers"):
                log.info(f"  Overriding num_hidden_layers: {base_config.num_hidden_layers} -> {info['layers']}")
                base_config.num_hidden_layers = info["layers"]

        config = base_config
        base_source = base_model_id

    log.info(f"  Loading config: hidden={config.hidden_size}, vocab={config.vocab_size}, layers={config.num_hidden_layers}")

    model = AutoModelForCausalLM.from_pretrained(
        base_source, config=config, trust_remote_code=True,
        torch_dtype=torch.float32, ignore_mismatched_sizes=True,
    )

    # Tokenizer
    if (checkpoint_dir / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load .pt checkpoint
    ckpt_path = checkpoint_dir / checkpoint_name if checkpoint_name else find_latest_checkpoint(checkpoint_dir)
    if ckpt_path and ckpt_path.exists():
        log.info(f"Loading state dict: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        log.info(f"  Keys: {len(state)}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"  Missing: {missing[:3]}...")
        if unexpected:
            log.warning(f"  Unexpected (LoRA/normal): {unexpected[:3]}...")
        del state

    return model, tokenizer, config


def _load_tinyllama_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Load a TinyLlama + LoRA checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_path = checkpoint_dir / "config.json"
    safetensors_path = checkpoint_dir / "model.safetensors"
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    has_full = config_path.exists() and safetensors_path.exists()

    if has_full:
        log.info(f"Full checkpoint layout detected in {checkpoint_dir}")
        config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        base_source = str(checkpoint_dir)
    else:
        log.info(f"No full layout — loading from HuggingFace: {base_model_id}")
        base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

        ckpt = checkpoint_dir / (checkpoint_name or "final_model.pt")
        if not ckpt.exists():
            ckpt = find_latest_checkpoint(checkpoint_dir)
        if ckpt:
            info = inspect_checkpoint(ckpt)
            log.info(f"  Checkpoint info: {info}")
            if "vocab_size" in info:
                base_config.vocab_size = info["vocab_size"]
            if "hidden_size" in info:
                base_config.hidden_size = info["hidden_size"]
            if info.get("layers"):
                base_config.num_hidden_layers = info["layers"]

        config = base_config
        base_source = base_model_id

    log.info(f"  Loading config: hidden={config.hidden_size}, vocab={config.vocab_size}, layers={config.num_hidden_layers}")

    model = AutoModelForCausalLM.from_pretrained(
        base_source, config=config, trust_remote_code=True,
        torch_dtype=torch.float32, ignore_mismatched_sizes=True,
    )

    if (checkpoint_dir / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = checkpoint_dir / checkpoint_name if checkpoint_name else find_latest_checkpoint(checkpoint_dir)
    if ckpt_path and ckpt_path.exists():
        log.info(f"Loading state dict: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"  Missing: {missing[:3]}...")
        if unexpected:
            log.warning(f"  Unexpected: {unexpected[:3]}...")
        del state

    return model, tokenizer, config


def load_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Load a model + tokenizer from a checkpoint directory.

    Auto-detects model type (Pythia or TinyLlama) and handles both full
    checkpoint layouts (safetensors + config) and training-output layouts
    (just .pt files).
    """
    if model_type is None:
        model_type = detect_model_type(checkpoint_dir)

    log.info(f"Detected model type: {model_type}")

    if model_type == "pythia":
        return _load_pythia_checkpoint(checkpoint_dir, checkpoint_name)
    elif model_type == "tinyllama":
        return _load_tinyllama_checkpoint(checkpoint_dir, checkpoint_name)
    else:
        # Try Pythia first, then TinyLlama
        try:
            return _load_pythia_checkpoint(checkpoint_dir, checkpoint_name)
        except Exception as e:
            log.warning(f"Pythia load failed ({e}), trying TinyLlama...")
            return _load_tinyllama_checkpoint(checkpoint_dir, checkpoint_name)


# ────────────────────────────────────────────────────────────────────────────
# Generation
# ────────────────────────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate text from a single prompt.

    Returns the decoded string (prompt + generated tokens).
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs = {k: v.clamp(0, tokenizer.vocab_size - 1) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_generation(
    checkpoint_dir: Path,
    prompts: List[str],
    checkpoint_name: Optional[str] = None,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    model_type: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Load a checkpoint and run generation on a list of prompts.

    Returns a list of dicts: [{"prompt": ..., "output": ...}, ...]
    """
    model, tokenizer, config = load_checkpoint(checkpoint_dir, checkpoint_name, model_type)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model ready: {n_params:,} params")

    results = []
    for prompt in prompts:
        text = generate(model, tokenizer, prompt, max_new_tokens, temperature)
        # Sanitize for Windows console
        safe = text.encode("cp1252", errors="replace").decode("cp1252")
        log.info(f"  [{prompt[:40]}] -> [{safe[:60]}]")
        results.append({"prompt": prompt, "output": safe})

    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LISA_FTM Inference Engine")
    parser.add_argument("--dir", type=str, default="output/real_training",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint .pt file name within the directory")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["PyTorch is a", "Machine learning models",
                                "The history of artificial"],
                        help="Generation prompts")
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["pythia", "tinyllama", None],
                        help="Force model type (auto-detected if not set)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root = Path(__file__).parent.parent
    ckpt_dir = root / args.dir

    if not ckpt_dir.exists():
        log.error(f"Checkpoint directory not found: {ckpt_dir}")
        return

    log.info(f"Loading checkpoint from: {ckpt_dir}")
    results = run_generation(
        ckpt_dir,
        args.prompts,
        checkpoint_name=args.checkpoint,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        model_type=args.model_type,
    )

    log.info("\n" + "=" * 60)
    log.info("GENERATION RESULTS")
    log.info("=" * 60)
    for r in results:
        log.info(f"  Prompt: {r['prompt']}")
        log.info(f"  Output: {r['output']}\n")


if __name__ == "__main__":
    main()
