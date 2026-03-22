#!/usr/bin/env python3
"""
LISA-FTM Autoresearch Framework

Autonomous experiment loop for federated fine-tuning research.
The agent edits eval/fed_config.py, then `python eval/autora.py run`
executes the experiment and reports results.

Commands:
  python eval/autora.py run     — Run one experiment with current fed_config.py
  python eval/autora.py best    — Print best experiment so far (lowest perplexity)
  python eval/autora.py suggest — Analyze results and suggest next config
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import logging
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("autora")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load eval/fed_config.py as a module and return its settings dict."""
    config_path = EVAL_DIR / "fed_config.py"
    spec = importlib.util.spec_from_file_location("fed_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = {
        "LISA_BOTTOM_LAYERS": getattr(module, "LISA_BOTTOM_LAYERS", 2),
        "LISA_TOP_LAYERS": getattr(module, "LISA_TOP_LAYERS", 2),
        "LISA_MIDDLE_SAMPLE": getattr(module, "LISA_MIDDLE_SAMPLE", 2),
        "LR": getattr(module, "LR", 3e-4),
        "LOCAL_STEPS": getattr(module, "LOCAL_STEPS", 5),
        "NUM_CLIENTS": getattr(module, "NUM_CLIENTS", 3),
        "ROUNDS": getattr(module, "ROUNDS", 5),
        "MODEL": getattr(module, "MODEL", "EleutherAI/pythia-70m"),
        "COMPRESSION": getattr(module, "COMPRESSION", "both"),
        "COMPRESSION_K": getattr(module, "COMPRESSION_K", 0.1),
        "COMPRESSION_BITS": getattr(module, "COMPRESSION_BITS", 8),
        "NOISE_MULTIPLIER": getattr(module, "NOISE_MULTIPLIER", 0.0),
    }
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Short hash of the config for identification."""
    stable = {k: v for k, v in cfg.items() if k != "timestamp"}
    s = json.dumps(stable, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def next_exp_id() -> int:
    """Find the next experiment number."""
    existing = list(RESULTS_DIR.glob("exp_*.json"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums) + 1


# ---------------------------------------------------------------------------
# LoRA helpers (from fedavg_vs_lisafedavg.py)
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    """LoRA on top of an existing linear: y = Wx + BAx (low-rank)."""

    def __init__(self, linear: torch.nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05, name: str = ""):
        super().__init__()
        self.weight = linear.weight.data.clone()
        self.bias = linear.bias.data.clone() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight.shape
        self.is_conv1d = isinstance(linear, torch.nn.Conv1d)

        self.rank = rank
        self.alpha = alpha
        self.name = name
        self.lora_A = torch.nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
        self.scaling = alpha / rank

        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            original = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_input = self.lora_dropout(x)
        lora = torch.nn.functional.linear(lora_input, self.lora_A)
        lora = torch.nn.functional.linear(lora, self.lora_B)
        return original + lora * self.scaling

    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [self.lora_A, self.lora_B]


class LoraAppliedModel:
    """Apply LoRA to target layers of a model."""

    TARGET_MODULES = [
        "query_key_value", "dense",
        "dense_h_to_4h", "dense_4h_to_h",
        "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc",
    ]

    def __init__(self, model: torch.nn.Module, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.05):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers: Dict[str, LoRALinear] = {}

    def apply_lora(self) -> int:
        import torch.nn as nn
        count = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            name_parts = full_name.split(".")
            if not any(tm in name_parts[-1] for tm in self.TARGET_MODULES):
                continue

            lora = LoRALinear(module, rank=self.rank, alpha=self.alpha,
                              dropout=self.dropout, name=full_name)
            self.lora_layers[full_name] = lora

            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                try:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, attr, lora)
                    count += 1
                except KeyError:
                    pass

        logger.info(f"  LoRA applied to {count} layers (rank={self.rank})")
        return count

    def freeze_all_except_lora(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for lp in self.lora_layers.values():
            for p in lp.trainable_params():
                p.requires_grad = True

    def freeze_only(self, layer_indices: List[int]):
        for param in self.model.parameters():
            param.requires_grad = False
        name_patterns = []
        for idx in layer_indices:
            name_patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])

        for full_name, lora_layer in self.lora_layers.items():
            for pat in name_patterns:
                if pat in full_name:
                    for p in lora_layer.trainable_params():
                        p.requires_grad = True
                    break

    def get_trainable_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def merge_weights(self):
        for lora_layer in self.lora_layers.values():
            with torch.no_grad():
                w = lora_layer.weight
                ba = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                lora_layer.weight = w + ba.view_as(w)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_wikitext(tokenizer, max_seq: int = 128,
                  train_frac: float = 0.85) -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _synthetic_data()

    try:
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        test_ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    except Exception:
        return _synthetic_data()

    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in test_ds["text"] if t.strip() and len(t.strip()) > 20]
    logger.info(f"  wikitext: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def _synthetic_data(n_train: int = 600, n_test: int = 100) -> Tuple[List[str], List[str]]:
    domains = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning neural network training data",
        "federated learning distributed privacy client server",
        "language model transformer attention mechanism",
        "optimization gradient descent learning rate",
    ]
    words = " ".join(domains).split()
    rng = random.Random(42)
    train_texts, test_texts = [], []
    for _ in range(n_train):
        sel = rng.sample(words, min(30, len(words)))
        rng.shuffle(sel)
        train_texts.append(" ".join(sel * 3)[:150])
    for _ in range(n_test):
        sel = rng.sample(words, min(25, len(words)))
        rng.shuffle(sel)
        test_texts.append(" ".join(sel * 3)[:150])
    logger.info(f"  synthetic: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def tokenize_texts(tokenizer, texts: List[str], max_seq: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        max_length=max_seq,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
    }


def partition_data(texts: List[str], n: int, seed: int = 42, non_iid: bool = False) -> List[List[str]]:
    """
    Partition texts across n clients.
    non_iid=True: sequential topic slices (client 0 = samples 0-400,
    client 1 = 400-800, client 2 = 800-1200) to simulate non-IID data skew.
    """
    if non_iid:
        size = 400
        partitions = []
        for i in range(n):
            start = i * size
            end = start + size if i < n - 1 else len(texts)
            partitions.append(texts[start:end])
        return partitions
    else:
        rng = random.Random(seed)
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        size = len(shuffled) // n
        partitions = []
        for i in range(n):
            start = i * size
            end = start + size if i < n - 1 else len(shuffled)
            partitions.append(shuffled[start:end])
        return partitions


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(model: torch.nn.Module, test_enc: Dict[str, torch.Tensor],
                       batch_size: int = 4, max_batches: int = None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = (len(test_enc["input_ids"]) + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    for i in range(n_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, len(test_enc["input_ids"]))
        ids = test_enc["input_ids"][start:end].clone()
        mask = test_enc["attention_mask"][start:end]
        labels = test_enc["labels"][start:end]

        ids = ids.clamp(0, model.config.vocab_size - 1)
        labels = labels.clamp(0, model.config.vocab_size - 1)

        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id or -100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()

    model.train()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# LISA layer selection
# ---------------------------------------------------------------------------

def lisa_select_layers(num_layers: int, bottom: int = 2, top: int = 2,
                       middle: int = 2, seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    bottom_set = list(range(min(bottom, num_layers)))
    top_start = max(0, num_layers - top)
    top_set = list(range(top_start, num_layers))

    middle_start = bottom
    middle_end = max(middle_start, num_layers - top)
    middle_pool = list(range(middle_start, middle_end))
    middle_set = rng.sample(middle_pool, min(middle, len(middle_pool))) if middle_pool else []

    return sorted(set(bottom_set + top_set + middle_set))


# ---------------------------------------------------------------------------
# Gradient compression
# ---------------------------------------------------------------------------

def compress_gradients(delta: Dict[Tuple[str, str], torch.Tensor],
                       cfg: Dict[str, Any]) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Apply compression to gradient deltas based on cfg settings.
    Returns the (optionally compressed) deltas dict.
    Currently implements sparsification: keep only top-K% by magnitude.
    """
    compression = cfg.get("COMPRESSION", "none")
    if compression == "none":
        return delta

    k_frac = cfg.get("COMPRESSION_K", 0.1)
    bit_depth = cfg.get("COMPRESSION_BITS", 8)

    result = {}
    for key, tensor in delta.items():
        if compression in ("sparsify", "both"):
            # Sparsification: keep only top-K% by absolute magnitude
            flat = tensor.flatten().float()
            if len(flat) == 0:
                result[key] = tensor
                continue
            k_count = max(1, int(len(flat) * k_frac))
            threshold = flat.abs().kthvalue(len(flat) - k_count)[0].item()
            mask = flat.abs() >= threshold
            flat = flat * mask.float()
            result[key] = flat.view_as(tensor)
        elif compression in ("quantize", "both"):
            # Simple stochastic quantization to bit_depth bits
            scale = tensor.abs().max().item() if tensor.abs().max().item() > 0 else 1.0
            # Map to [0, 2^bit_depth - 1]
            levels = 2 ** bit_depth - 1
            q = (tensor.float() / scale * levels).round().clamp(0, levels)
            result[key] = (q / levels * scale).view_as(tensor)
        else:
            result[key] = tensor

    return result


# ---------------------------------------------------------------------------
# Gradient snapshots
# ---------------------------------------------------------------------------

def snapshot_lora_state(lora_wrapper: LoraAppliedModel) -> Dict[Tuple[str, str], torch.Tensor]:
    state = {}
    for full_name, lora_layer in lora_wrapper.lora_layers.items():
        state[(full_name, "lora_A")] = lora_layer.lora_A.data.clone().cpu()
        state[(full_name, "lora_B")] = lora_layer.lora_B.data.clone().cpu()
    return state


def compute_grad_deltas(state_before: Dict[Tuple[str, str], torch.Tensor],
                         state_after: Dict[Tuple[str, str], torch.Tensor]
                         ) -> Dict[Tuple[str, str], torch.Tensor]:
    deltas = {}
    for key in state_before:
        deltas[key] = state_after[key] - state_before[key]
    return deltas


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: Dict[str, Any], exp_id: int) -> Dict[str, Any]:
    """
    Run a single federated experiment using cfg settings.
    Returns a results dict with perplexity per round, comm cost, etc.
    """
    MODEL_ID = cfg["MODEL"]
    NUM_CLIENTS = cfg["NUM_CLIENTS"]
    NUM_ROUNDS = cfg["ROUNDS"]
    LR = cfg["LR"]
    LOCAL_STEPS = cfg["LOCAL_STEPS"]
    LISA_BOTTOM = cfg["LISA_BOTTOM_LAYERS"]
    LISA_TOP = cfg["LISA_TOP_LAYERS"]
    LISA_MIDDLE = cfg["LISA_MIDDLE_SAMPLE"]
    LORA_RANK = 4
    LORA_ALPHA = 8.0
    LORA_DROPOUT = 0.05
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 64
    MAX_TEST_SAMPLES = 200
    SEED = 42

    logger.info("=" * 70)
    logger.info(f"AUTORESEARCH EXPERIMENT #{exp_id}")
    logger.info("=" * 70)
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading model and tokenizer...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.float32,
        )
        num_layers = config.num_hidden_layers
        logger.info(f"  Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
        logger.info(f"  Layers: {num_layers}, Hidden: {config.hidden_size}, Vocab: {config.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"status": "error", "message": str(e)}

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Loading wikitext dataset...")
    t0 = time.time()
    train_texts, test_texts = load_wikitext(tokenizer)
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")

    client_partitions = partition_data(train_texts, NUM_CLIENTS)
    logger.info(f"  Partitions: {[len(p) for p in client_partitions]} texts each")

    test_enc = tokenize_texts(tokenizer, test_texts, MAX_SEQ_LEN)
    test_enc = {k: v.clone() for k, v in test_enc.items()}
    # Limit test samples for faster perplexity evaluation
    for k in test_enc:
        test_enc[k] = test_enc[k][:MAX_TEST_SAMPLES]

    # -------------------------------------------------------------------------
    # Apply LoRA
    # -------------------------------------------------------------------------
    logger.info("\n[SETUP] Applying LoRA...")
    lora_wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    lora_count = lora_wrapper.apply_lora()
    logger.info(f"  LoRA applied to {lora_count} layers")

    # Ensure model is fully on CPU before training begins
    model.to('cpu')
    logger.info(f"  Model device: {next(model.parameters()).device}")

    # Save base (pre-LoRA) model state for resets
    base_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # -------------------------------------------------------------------------
    # Federated loop
    # -------------------------------------------------------------------------
    results_by_round = []
    layer_selection_counts: Dict[int, int] = {i: 0 for i in range(num_layers)}
    total_comm_cost = 0

    # Snapshot of server LoRA state at start of each round
    round_base_state = snapshot_lora_state(lora_wrapper)

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
        t0 = time.time()

        round_grad_deltas: List[Dict[Tuple[str, str], torch.Tensor]] = []
        client_weights: List[float] = []
        round_client_results = []

        for cid_idx, texts in enumerate(client_partitions):
            client_id = f"client-{cid_idx+1}"
            n_samples = len(texts)
            client_weights.append(n_samples)

            # Select LISA layers for this client
            seed = round_num * 100 + cid_idx * 7
            selected = lisa_select_layers(num_layers, LISA_BOTTOM, LISA_TOP, LISA_MIDDLE, seed=seed)

            # Track layer selection
            for li in selected:
                layer_selection_counts[li] += 1

            # Freeze only selected layers
            lora_wrapper.freeze_only(selected)

            # Tokenize client data
            enc = tokenize_texts(tokenizer, texts, MAX_SEQ_LEN)

            # Setup optimizer
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

            model.train()
            losses = []
            n_batches = (len(enc["input_ids"]) + BATCH_SIZE - 1) // BATCH_SIZE

            for step in range(LOCAL_STEPS):
                indices = torch.randperm(len(enc["input_ids"])).tolist()
                for i in range(n_batches):
                    idx = indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    ids = enc["input_ids"][idx].clone().clamp(0, tokenizer.vocab_size - 1)
                    mask = enc["attention_mask"][idx]
                    labs = enc["labels"][idx].clone().clamp(0, tokenizer.vocab_size - 1)

                    optimizer.zero_grad()
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
                    loss = outputs.loss
                    if torch.isnan(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    losses.append(loss.item())

            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # Snapshot after training
            state_after = snapshot_lora_state(lora_wrapper)

            # Compute gradient delta (after - before)
            delta = compute_grad_deltas(round_base_state, state_after)

            # Apply compression
            if cfg.get("COMPRESSION", "none") != "none":
                delta = compress_gradients(delta, cfg)

            # Apply differential privacy noise (Gaussian noise on gradient tensors)
            noise_mult = cfg.get("NOISE_MULTIPLIER", 0.0)
            if noise_mult > 0.0:
                for key in delta:
                    grad_std = delta[key].float().abs().mean().item()
                    noise = torch.randn_like(delta[key]) * grad_std * noise_mult
                    delta[key] = delta[key].float() + noise

            round_grad_deltas.append(delta)

            # Compute perplexity (on a sample of test data for speed)
            ppl = compute_perplexity(model, test_enc, max_batches=50)

            # Count grad tensors (per selected layer: 2 tensors — lora_A and lora_B)
            n_grad_tensors = len(selected) * 2

            round_client_results.append({
                "client_id": client_id,
                "train_loss": avg_loss,
                "perplexity": ppl,
                "selected_layers": selected,
                "n_grad_tensors": n_grad_tensors,
            })

            # Reset model to round base state for next client
            for (full_name, lora_layer) in lora_wrapper.lora_layers.items():
                lora_layer.lora_A.data.copy_(round_base_state[(full_name, "lora_A")].clone())
                lora_layer.lora_B.data.copy_(round_base_state[(full_name, "lora_B")].clone())

        # -------------------------------------------------------------------------
        # Aggregate: weighted average of gradient deltas
        # -------------------------------------------------------------------------
        total_samples = sum(client_weights)
        normalized_weights = [w / total_samples for w in client_weights]

        aggregated: Dict[Tuple[str, str], torch.Tensor] = {}
        for delta, weight in zip(round_grad_deltas, normalized_weights):
            for key, d in delta.items():
                if key not in aggregated:
                    aggregated[key] = d.float() * weight
                else:
                    aggregated[key] = aggregated[key] + d.float() * weight

        # Apply aggregated update to server's LoRA layers
        for (layer_name, param_name), grad_delta in aggregated.items():
            if layer_name in lora_wrapper.lora_layers:
                lora_layer = lora_wrapper.lora_layers[layer_name]
                with torch.no_grad():
                    if param_name == "lora_A":
                        lora_layer.lora_A.add_(grad_delta)
                    elif param_name == "lora_B":
                        lora_layer.lora_B.add_(grad_delta)

        # Snapshot new round base state
        round_base_state = snapshot_lora_state(lora_wrapper)

        elapsed = time.time() - t0
        avg_ppl = sum(r["perplexity"] for r in round_client_results) / len(round_client_results)
        round_comm = sum(r["n_grad_tensors"] for r in round_client_results)
        total_comm_cost += round_comm
        selected_all = sorted(set().union(*(set(r["selected_layers"]) for r in round_client_results)))

        logger.info(
            f"  Round {round_num}: perplexity={avg_ppl:.2f} | "
            f"comm={round_comm} tensors | layers={selected_all} | {elapsed:.1f}s"
        )

        results_by_round.append({
            "round": round_num,
            "avg_test_perplexity": avg_ppl,
            "avg_train_loss": sum(r["train_loss"] for r in round_client_results) / len(round_client_results),
            "comm_cost": round_comm,
            "layers_selected_this_round": selected_all,
            "client_results": [
                {"client_id": r["client_id"], "train_loss": r["train_loss"],
                 "perplexity": r["perplexity"], "selected_layers": r["selected_layers"]}
                for r in round_client_results
            ],
        })

        gc.collect()

    # -------------------------------------------------------------------------
    # Build results dict
    # -------------------------------------------------------------------------
    final_ppl = results_by_round[-1]["avg_test_perplexity"] if results_by_round else float("inf")

    results = {
        "exp_id": f"exp_{exp_id:03d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "perplexity_per_round": [r["avg_test_perplexity"] for r in results_by_round],
        "comm_cost_per_round": [r["comm_cost"] for r in results_by_round],
        "final_perplexity": final_ppl,
        "total_comm_cost": total_comm_cost,
        "layer_selection": {
            "bottom": LISA_BOTTOM,
            "top": LISA_TOP,
            "middle": LISA_MIDDLE,
        },
        "layer_counts": {str(k): v for k, v in layer_selection_counts.items()},
        "rounds_detail": results_by_round,
    }

    # Save
    out_path = RESULTS_DIR / f"exp_{exp_id:03d}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")

    # Print summary table
    _print_summary(results, exp_id)

    return results


def _print_summary(results: Dict[str, Any], exp_id: int):
    """Print a formatted summary table."""
    logger.info("\n" + "=" * 70)
    logger.info(f"EXPERIMENT #{exp_id} — SUMMARY")
    logger.info("=" * 70)

    cfg = results["config"]
    logger.info(f"{'Round':>5} | {'Perplexity':>12} | {'Comm Cost':>10} | {'Layers Selected':>30}")
    logger.info("-" * 70)

    for r, ppl, comm in zip(
        range(1, len(results["perplexity_per_round"]) + 1),
        results["perplexity_per_round"],
        results["comm_cost_per_round"],
    ):
        layer_str = str(results["rounds_detail"][r-1]["layers_selected_this_round"])
        logger.info(f"{r:>5} | {ppl:>12.2f} | {comm:>10} | {layer_str:>30}")

    logger.info("-" * 70)
    logger.info(f"{'FINAL':>5} | {results['final_perplexity']:>12.2f} | {results['total_comm_cost']:>10} |")
    logger.info(f"\nConfig: LISA({cfg['LISA_BOTTOM_LAYERS']}+{cfg['LISA_MIDDLE_SAMPLE']}+{cfg['LISA_TOP_LAYERS']}) "
                f"LR={cfg['LR']} LOCAL_STEPS={cfg['LOCAL_STEPS']} "
                f"COMPRESSION={cfg['COMPRESSION']} K={cfg['COMPRESSION_K']} "
                f"bits={cfg['COMPRESSION_BITS']}")
    logger.info(f"Config hash: {results['config_hash']}")


# ---------------------------------------------------------------------------
# best command
# ---------------------------------------------------------------------------

def cmd_best():
    """Print the best experiment so far (lowest final perplexity)."""
    existing = sorted(RESULTS_DIR.glob("exp_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not existing:
        logger.info("No experiments found. Run `python eval/autora.py run` first.")
        return

    best_exp = None
    best_ppl = float("inf")

    for path in existing:
        try:
            with open(path) as f:
                data = json.load(f)
            ppl = data.get("final_perplexity", float("inf"))
            if ppl < best_ppl:
                best_ppl = ppl
                best_exp = data
        except (json.JSONDecodeError, KeyError):
            continue

    if best_exp is None:
        logger.info("No valid experiments found.")
        return

    cfg = best_exp["config"]
    logger.info("\n" + "=" * 70)
    logger.info(f"BEST EXPERIMENT: {best_exp['exp_id']} (perplexity={best_ppl:.2f})")
    logger.info("=" * 70)
    logger.info(f"Config hash: {best_exp['config_hash']}")
    logger.info(f"Timestamp:   {best_exp['timestamp']}")
    logger.info(f"Perplexity:  {best_ppl:.2f}")
    logger.info(f"Comm cost:   {best_exp['total_comm_cost']} tensors total")
    logger.info(f"LISA: bottom={cfg['LISA_BOTTOM_LAYERS']} top={cfg['LISA_TOP_LAYERS']} "
                f"middle={cfg['LISA_MIDDLE_SAMPLE']}")
    logger.info(f"LR={cfg['LR']} LOCAL_STEPS={cfg['LOCAL_STEPS']} CLIENTS={cfg['NUM_CLIENTS']} "
                f"ROUNDS={cfg['ROUNDS']}")
    logger.info(f"Compression: {cfg['COMPRESSION']} K={cfg['COMPRESSION_K']} bits={cfg['COMPRESSION_BITS']}")
    logger.info(f"\nPerplexity curve: {[f'{p:.2f}' for p in best_exp['perplexity_per_round']]}")
    logger.info(f"Comm cost curve:  {best_exp['comm_cost_per_round']}")
    logger.info(f"Results file: {RESULTS_DIR / best_exp['exp_id']}.json")


# ---------------------------------------------------------------------------
# suggest command
# ---------------------------------------------------------------------------

def cmd_suggest():
    """Analyze results and suggest next config direction."""
    existing = sorted(RESULTS_DIR.glob("exp_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if len(existing) < 2:
        logger.info("Not enough data to suggest. Run at least 2 experiments first.")
        return

    experiments = []
    for path in existing:
        try:
            with open(path) as f:
                experiments.append(json.load(f))
        except (json.JSONDecodeError, KeyError):
            continue

    if len(experiments) < 2:
        logger.info("Not enough valid experiments to analyze.")
        return

    logger.info("\n" + "=" * 70)
    logger.info("AUTORESEARCH SUGGESTIONS")
    logger.info("=" * 70)

    # Group by key parameter values
    suggestions: List[str] = []

    # -------------------------------------------------------------------------
    # Analyze LISA_MIDDLE_SAMPLE impact
    # -------------------------------------------------------------------------
    middle_samples: Dict[int, List[Dict]] = {}
    for exp in experiments:
        m = exp["config"].get("LISA_MIDDLE_SAMPLE", 2)
        middle_samples.setdefault(m, []).append(exp)

    if len(middle_samples) >= 2:
        mids = sorted(middle_samples.keys())
        avg_ppls = {m: sum(e["final_perplexity"] for e in grp) / len(grp)
                    for m, grp in middle_samples.items()}
        avg_comm = {m: sum(e["total_comm_cost"] for e in grp) / len(grp)
                     for m, grp in middle_samples.items()}
        logger.info("\n[LISA_MIDDLE_SAMPLE analysis]:")
        for m in mids:
            logger.info(f"  middle={m}: avg_ppl={avg_ppls[m]:.2f} avg_comm={avg_comm[m]:.0f} "
                        f"(n={len(middle_samples[m])})")

        # Suggest direction
        best_m = min(avg_ppls, key=avg_ppls.get)
        if best_m != max(mids):
            reduction = avg_comm[max(mids)] - avg_comm[best_m]
            pct = reduction / avg_comm[max(mids)] * 100 if avg_comm[max(mids)] > 0 else 0
            suggestions.append(
                f"Try LISA_MIDDLE_SAMPLE={min(mids)} — configs with fewer middle layers "
                f"had {pct:.0f}% lower comm cost with ppl change of "
                f"{avg_ppls[max(mids)] - avg_ppls[min(mids)]:+.2f}"
            )

    # -------------------------------------------------------------------------
    # Analyze compression impact
    # -------------------------------------------------------------------------
    compression_types: Dict[str, List[Dict]] = {}
    for exp in experiments:
        c = exp["config"].get("COMPRESSION", "none")
        compression_types.setdefault(c, []).append(exp)

    if len(compression_types) >= 2:
        logger.info("\n[COMPRESSION analysis]:")
        for c, grp in sorted(compression_types.items()):
            avg_p = sum(e["final_perplexity"] for e in grp) / len(grp)
            avg_cc = sum(e["total_comm_cost"] for e in grp) / len(grp)
            logger.info(f"  compression={c}: avg_ppl={avg_p:.2f} avg_comm={avg_cc:.0f} (n={len(grp)})")

    # -------------------------------------------------------------------------
    # Analyze LISA_BOTTOM_LAYERS impact
    # -------------------------------------------------------------------------
    bottom_layers: Dict[int, List[Dict]] = {}
    for exp in experiments:
        b = exp["config"].get("LISA_BOTTOM_LAYERS", 2)
        bottom_layers.setdefault(b, []).append(exp)

    if len(bottom_layers) >= 2:
        logger.info("\n[LISA_BOTTOM_LAYERS analysis]:")
        for b, grp in sorted(bottom_layers.items()):
            avg_p = sum(e["final_perplexity"] for e in grp) / len(grp)
            avg_cc = sum(e["total_comm_cost"] for e in grp) / len(grp)
            logger.info(f"  bottom={b}: avg_ppl={avg_p:.2f} avg_comm={avg_cc:.0f} (n={len(grp)})")

    # -------------------------------------------------------------------------
    # Analyze LR impact
    # -------------------------------------------------------------------------
    lr_buckets: Dict[str, List[Dict]] = {}
    for exp in experiments:
        lr = exp["config"].get("LR", 3e-4)
        bucket = f"{lr:.0e}"
        lr_buckets.setdefault(bucket, []).append(exp)

    if len(lr_buckets) >= 2:
        logger.info("\n[LR analysis]:")
        for bucket, grp in sorted(lr_buckets.items()):
            avg_p = sum(e["final_perplexity"] for e in grp) / len(grp)
            logger.info(f"  LR {bucket}: avg_ppl={avg_p:.2f} (n={len(grp)})")

    # -------------------------------------------------------------------------
    # Analyze LOCAL_STEPS impact
    # -------------------------------------------------------------------------
    local_steps_buckets: Dict[int, List[Dict]] = {}
    for exp in experiments:
        ls = exp["config"].get("LOCAL_STEPS", 5)
        local_steps_buckets.setdefault(ls, []).append(exp)

    if len(local_steps_buckets) >= 2:
        logger.info("\n[LOCAL_STEPS analysis]:")
        for ls, grp in sorted(local_steps_buckets.items()):
            avg_p = sum(e["final_perplexity"] for e in grp) / len(grp)
            avg_cc = sum(e["total_comm_cost"] for e in grp) / len(grp)
            logger.info(f"  steps={ls}: avg_ppl={avg_p:.2f} avg_comm={avg_cc:.0f} (n={len(grp)})")

    # -------------------------------------------------------------------------
    # General suggestions
    # -------------------------------------------------------------------------
    logger.info("\n[SUMMARY]:")
    logger.info(f"  Total experiments analyzed: {len(experiments)}")
    logger.info(f"  Best perplexity so far: {min(e['final_perplexity'] for e in experiments):.2f}")
    logger.info(f"  Lowest comm cost so far: {min(e['total_comm_cost'] for e in experiments)} tensors")

    # Pareto frontier
    pareto = []
    for exp in experiments:
        ppl = exp["final_perplexity"]
        comm = exp["total_comm_cost"]
        dominated = any(
            e["final_perplexity"] < ppl and e["total_comm_cost"] < comm
            for e in experiments if e != exp
        )
        if not dominated:
            pareto.append(exp)

    logger.info(f"  Pareto-optimal configs: {len(pareto)}")
    for p in sorted(pareto, key=lambda x: x["final_perplexity"]):
        cfg = p["config"]
        logger.info(f"    exp={p['exp_id']} ppl={p['final_perplexity']:.2f} comm={p['total_comm_cost']} "
                    f" LISA({cfg['LISA_BOTTOM_LAYERS']}+{cfg['LISA_MIDDLE_SAMPLE']}+{cfg['LISA_TOP_LAYERS']})"
                    f"LR={cfg['LR']} COMPRESSION={cfg['COMPRESSION']}")

    logger.info("\n[TOP SUGGESTIONS]:")
    if suggestions:
        for s in suggestions[:3]:
            logger.info(f"  • {s}")
    else:
        # Generic suggestions if not enough data for specific ones
        # Find the current best by perplexity
        best_by_ppl = min(experiments, key=lambda x: x["final_perplexity"])
        best_by_comm = min(experiments, key=lambda x: x["total_comm_cost"])
        cfg_best = best_by_ppl["config"]

        logger.info(f"  • Best by perplexity: exp={best_by_ppl['exp_id']} ppl={best_by_ppl['final_perplexity']:.2f}")
        logger.info(f"  • Best by comm cost:  exp={best_by_comm['exp_id']} comm={best_by_comm['total_comm_cost']}")

        # Try suggesting based on what hasn't been explored
        if cfg_best["LISA_MIDDLE_SAMPLE"] > 1:
            suggestions.append(
                f"Try reducing LISA_MIDDLE_SAMPLE from {cfg_best['LISA_MIDDLE_SAMPLE']} "
                f"to {cfg_best['LISA_MIDDLE_SAMPLE']-1} to reduce comm cost"
            )
        if cfg_best["LR"] > 1e-4:
            suggestions.append(
                f"Try lowering LR from {cfg_best['LR']} to {cfg_best['LR']/2} "
                "for potentially smoother convergence"
            )
        if cfg_best["LOCAL_STEPS"] < 10:
            suggestions.append(
                f"Try increasing LOCAL_STEPS from {cfg_best['LOCAL_STEPS']} "
                "to 8 for better local training"
            )
        if cfg_best["COMPRESSION"] == "none":
            suggestions.append("Try adding compression (COMPRESSION='sparsify') to reduce comm cost")
        elif cfg_best["COMPRESSION_K"] >= 0.1:
            suggestions.append("Try more aggressive compression (COMPRESSION_K=0.05)")

        for s in suggestions[:3]:
            logger.info(f"  • {s}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LISA-FTM Autoresearch Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run one experiment with current fed_config.py")
    sub.add_parser("best", help="Print best experiment so far (lowest perplexity)")
    sub.add_parser("suggest", help="Analyze results and suggest next config")

    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config()
        exp_id = next_exp_id()
        random.seed(42)
        torch.manual_seed(42)
        results = run_experiment(cfg, exp_id)
        if results.get("status") == "error":
            logger.error(f"Experiment failed: {results.get('message')}")
            sys.exit(1)

    elif args.command == "best":
        cmd_best()

    elif args.command == "suggest":
        cmd_suggest()


if __name__ == "__main__":
    main()
